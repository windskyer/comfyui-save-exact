"""
ComfyUI Save Exact — 保存图片/视频/音频,文件名完全由 filename_prefix / filename 决定,
不追加 00001 之类的自增计数器。默认覆盖同名文件,适合 pipeline.py 这类自动化流程
(输出文件名可预测,无需解析 history 猜文件名)。

节点:
  - SaveImageExact  (images: IMAGE)
  - SaveVideoExact  (video: VIDEO,配合官方 CreateVideo 节点使用)
  - SaveAudioExact  (audio: AUDIO)

命名规则:
  - filename_prefix 作为目录名,filename 作为文件名,最终路径为 filename_prefix/filename
  - filename 可含子目录(如 "scene01/shot03" → filename_prefix/scene01/shot03.ext)
  - filename 为空 → 用 "output" 作为文件名
  - 若名字自带扩展名(.png/.mp4/.flac 等)则尊重之,否则使用 format 参数
  - 批量图片/音频:单张直接存 base.ext;多张存 base_0.ext, base_1.ext ...
  - overwrite=False 时遇到同名文件直接报错,不会静默改名
"""

import os
import json

import numpy as np
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import folder_paths

MEDIA_EXTS = {
    "png", "jpg", "jpeg", "webp",
    "mp4", "webm", "mov", "mkv",
    "flac", "wav", "mp3", "ogg",
}


def _resolve_name(filename_prefix, filename, default_ext):
    """返回 (subfolder, base, ext)。filename_prefix 是目录,filename 是文件名,
    最终路径为 filename_prefix/filename。"""
    prefix = (filename_prefix or "").strip().replace("\\", "/").strip("/")
    name = (filename or "").strip().replace("\\", "/").strip("/") or "output"
    raw = f"{prefix}/{name}" if prefix else name
    subfolder, base = os.path.split(raw)
    root, ext = os.path.splitext(base)
    ext = ext.lstrip(".").lower()
    if ext in MEDIA_EXTS:
        base = root
    else:
        ext = default_ext
    if not base:
        base = "output"
    return subfolder, base, ext


def _prepare_dir(subfolder):
    """在 ComfyUI output 目录下创建子目录,并防止路径穿越。"""
    output_dir = os.path.normpath(folder_paths.get_output_directory())
    full_dir = os.path.normpath(os.path.join(output_dir, subfolder))
    if os.path.commonpath([output_dir, full_dir]) != output_dir:
        raise ValueError(f"非法子目录(试图跳出 output 目录): {subfolder}")
    os.makedirs(full_dir, exist_ok=True)
    return full_dir


def _target_path(full_dir, base, idx, total, ext, overwrite):
    name = f"{base}.{ext}" if total == 1 else f"{base}_{idx}.{ext}"
    path = os.path.join(full_dir, name)
    if not overwrite and os.path.exists(path):
        raise FileExistsError(f"文件已存在且 overwrite=False: {path}")
    return name, path


# --------------------------------------------------------------------------
# 图片
# --------------------------------------------------------------------------
class SaveImageExact:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": ""}),
            },
            "optional": {
                "filename": ("STRING", {"default": ""}),
                "format": (["png", "jpg", "webp"], {"default": "png"}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100}),
                "overwrite": ("BOOLEAN", {"default": True}),
                "embed_workflow": ("BOOLEAN", {"default": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "image/save"
    DESCRIPTION = "保存图片,文件名精确可控,无自增计数器"

    def save(self, images, filename_prefix, filename="", format="png",
             quality=95, overwrite=True, embed_workflow=True,
             prompt=None, extra_pnginfo=None):
        subfolder, base, ext = _resolve_name(filename_prefix, filename, format)
        full_dir = _prepare_dir(subfolder)

        total = images.shape[0]
        results = []
        for i in range(total):
            arr = (images[i].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
            name, path = _target_path(full_dir, base, i, total, ext, overwrite)

            if ext == "png":
                pnginfo = None
                if embed_workflow:
                    pnginfo = PngInfo()
                    if prompt is not None:
                        pnginfo.add_text("prompt", json.dumps(prompt))
                    if extra_pnginfo is not None:
                        for k, v in extra_pnginfo.items():
                            pnginfo.add_text(k, json.dumps(v))
                img.save(path, pnginfo=pnginfo, compress_level=4)
            elif ext in ("jpg", "jpeg"):
                img.convert("RGB").save(path, quality=quality)
            elif ext == "webp":
                img.save(path, quality=quality)
            else:
                img.save(path)

            results.append({"filename": name, "subfolder": subfolder, "type": "output"})

        return {"ui": {"images": results}}


# --------------------------------------------------------------------------
# 视频(接 VIDEO 类型,Wan 2.2 原生工作流用官方 CreateVideo 节点
# 把 IMAGE 批次 + fps [+ AUDIO] 组装成 VIDEO 后接到这里)
# --------------------------------------------------------------------------
class SaveVideoExact:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "filename_prefix": ("STRING", {"default": ""}),
            },
            "optional": {
                "filename": ("STRING", {"default": ""}),
                "format": (["mp4", "auto"], {"default": "mp4"}),
                "codec": (["h264", "auto"], {"default": "h264"}),
                "overwrite": ("BOOLEAN", {"default": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "image/video"
    DESCRIPTION = "保存视频,文件名精确可控,无自增计数器"

    def save(self, video, filename_prefix, filename="", format="mp4",
             codec="h264", overwrite=True, prompt=None, extra_pnginfo=None):
        ext = "mp4" if format == "auto" else format
        subfolder, base, ext = _resolve_name(filename_prefix, filename, ext)
        full_dir = _prepare_dir(subfolder)
        name, path = _target_path(full_dir, base, 0, 1, ext, overwrite)
        if overwrite and os.path.exists(path):
            os.remove(path)  # 某些容器写入器不支持覆盖已存在文件

        # 兼容不同版本的 comfy_api 枚举
        fmt_v, codec_v = format, codec
        try:
            from comfy_api.util import VideoContainer, VideoCodec
            fmt_v = VideoContainer(format) if format != "auto" else VideoContainer.AUTO
            codec_v = VideoCodec(codec) if codec != "auto" else VideoCodec.AUTO
        except Exception:
            pass

        metadata = {}
        if prompt is not None:
            metadata["prompt"] = prompt
        if extra_pnginfo is not None:
            metadata.update(extra_pnginfo)

        try:
            video.save_to(path, format=fmt_v, codec=codec_v, metadata=metadata or None)
        except TypeError:
            # 老版本 save_to 不接受 metadata 参数
            video.save_to(path, format=fmt_v, codec=codec_v)

        return {
            "ui": {
                "images": [{"filename": name, "subfolder": subfolder, "type": "output"}],
                "animated": (True,),
            }
        }


# --------------------------------------------------------------------------
# 音频(AUDIO = {"waveform": [B, C, T], "sample_rate": int})
# --------------------------------------------------------------------------
class SaveAudioExact:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename_prefix": ("STRING", {"default": ""}),
            },
            "optional": {
                "filename": ("STRING", {"default": ""}),
                "format": (["flac", "wav", "mp3"], {"default": "flac"}),
                "overwrite": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "audio"
    DESCRIPTION = "保存音频,文件名精确可控,无自增计数器"

    def save(self, audio, filename_prefix, filename="", format="flac", overwrite=True):
        import torchaudio

        subfolder, base, ext = _resolve_name(filename_prefix, filename, format)
        full_dir = _prepare_dir(subfolder)

        waveform = audio["waveform"]  # [B, C, T]
        sample_rate = int(audio["sample_rate"])
        if waveform.dim() == 2:  # [C, T] → [1, C, T]
            waveform = waveform.unsqueeze(0)

        total = waveform.shape[0]
        results = []
        for i in range(total):
            wav = waveform[i].cpu()
            if wav.dtype != torch.float32:
                wav = wav.float()
            name, path = _target_path(full_dir, base, i, total, ext, overwrite)
            torchaudio.save(path, wav, sample_rate, format=ext)
            results.append({"filename": name, "subfolder": subfolder, "type": "output"})

        return {"ui": {"audio": results}}


NODE_CLASS_MAPPINGS = {
    "SaveImageExact": SaveImageExact,
    "SaveVideoExact": SaveVideoExact,
    "SaveAudioExact": SaveAudioExact,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveImageExact": "Save Image (Exact Filename)",
    "SaveVideoExact": "Save Video (Exact Filename)",
    "SaveAudioExact": "Save Audio (Exact Filename)",
}
