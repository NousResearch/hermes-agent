"""
mimo-multimodal plugin v2 — Image, Audio & Video understanding via Xiaomi MiMo.

Features:
- Image analysis/description (JPEG, PNG, GIF, WebP, BMP)
- Audio transcription/analysis (MP3, WAV, FLAC, M4A, OGG)
- Video description/analysis (MP4, MOV, AVI, WMV)
- Local files (auto base64) and URLs
- Large local files via temp HTTP server (up to 300MB)
- Smart fps auto-tuning based on video duration
- Multi-image support in single request
- Format validation with magic bytes
- Audio extraction from video via ffmpeg
- Context-aware system prompts for better results

Author: xiaopengsvip (https://github.com/xiaopengsvip)
"""

import json
import os
import re
import base64
import socket
import subprocess
import tempfile
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler

BASE_URL = os.environ.get("XIAOMI_BASE_URL", "https://token-plan-sgp.xiaomimimo.com/v1")
MODEL = "mimo-v2.5"
TIMEOUT = 600

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac", ".wma"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".wmv", ".mkv", ".webm", ".flv"}

MIME_MAP = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
    ".mp3": "audio/mpeg", ".wav": "audio/wav", ".flac": "audio/flac",
    ".m4a": "audio/mp4", ".ogg": "audio/ogg", ".aac": "audio/aac",
    ".wma": "audio/x-ms-wma",
    ".mp4": "video/mp4", ".mov": "video/quicktime",
    ".avi": "video/x-msvideo", ".wmv": "video/x-ms-wmv",
    ".mkv": "video/x-matroska", ".webm": "video/webm",
    ".flv": "video/x-flv",
}

# ─── System prompts for better results ───────────────────────────────

IMAGE_SYSTEM_PROMPT = """You are a professional image analysis AI. Follow these guidelines:

1. DESCRIPTION: Describe the image content in detail — objects, people, scenes, colors, composition, lighting, style.

2. CONTEXT: If the image contains text (OCR), signs, labels, or UI elements, extract and describe them.

3. TECHNICAL: Note image quality, format, resolution if relevant.

4. LANGUAGE: Always respond in the same language as the user's prompt. If the prompt is in Chinese, respond in Chinese.

5. ACCURACY: If parts of the image are unclear or ambiguous, note it. Do not hallucinate details you cannot see.

6. STRUCTURE: For complex images, organize your analysis logically (e.g., foreground/background, left/right, main subject/details)."""

AUDIO_SYSTEM_PROMPT = """You are a professional audio analysis AI. Follow these guidelines:

1. TRANSCRIPTION: If asked to transcribe, output the exact spoken content in the original language. Use proper punctuation and paragraph breaks. Identify different speakers if possible (Speaker A, Speaker B...).

2. DESCRIPTION: If asked to describe, analyze: speech content, tone/emotion, background sounds, music genre/instruments, audio quality, language spoken.

3. TIMESTAMP: When possible, indicate approximate timestamps for key moments.

4. LANGUAGE: Always respond in the same language as the user's prompt. If the prompt is in Chinese, respond in Chinese.

5. ACCURACY: If audio is unclear or ambiguous, note it. Do not hallucinate content you cannot hear."""

VIDEO_SYSTEM_PROMPT = """You are a professional video analysis AI. Follow these guidelines:

1. VISUAL: Describe scenes, objects, people (appearance, actions), setting, lighting, camera movement, text/graphics on screen.

2. AUDIO: If video has audio, also describe speech, music, sound effects.

3. STRUCTURE: Break down the video chronologically — opening, key moments, ending.

4. DETAIL: Be specific — colors, quantities, spatial relationships, emotions shown.

5. TIMESTAMP: Use approximate timestamps for scene transitions and key events.

6. LANGUAGE: Always respond in the same language as the user's prompt. If the prompt is in Chinese, respond in Chinese.

7. If the video is too short or unclear, state what you can confidently observe."""

# ─── Prompt presets ──────────────────────────────────────────────────

IMAGE_PRESETS = {
    "describe": "请详细描述这张图片的内容，包括：主体对象、场景环境、颜色构图、光照风格、文字信息（如有）。",
    "ocr": "请识别并提取图片中出现的所有文字内容，包括标题、标识、标签、UI元素等。",
    "explain": "请解释这张图片的含义、用途或上下文。",
    "compare": "请分析这张图片的特点、风格和技术细节。",
    "extract_info": "请从这张图片中提取关键信息：人物、地点、时间、品牌、产品等结构化数据。",
    "caption": "请为这张图片写一段简洁的描述，适合作为图片说明文字。",
}

AUDIO_PRESETS = {
    "transcribe": "请完整转录这段音频的内容，保留原始语言，标注说话人（如果能区分），使用适当的标点和分段。",
    "describe": "请详细描述这段音频的内容，包括：语音内容、说话人语气/情绪、背景声音、音乐风格（如有）、语言种类、音频质量。",
    "summarize": "请听这段音频并给出简洁的摘要，突出关键信息和要点。",
    "translate": "请转录这段音频的内容，然后翻译成中文。",
    "extract_info": "请从这段音频中提取关键信息：人名、地点、时间、数字、事件等结构化数据。",
}

VIDEO_PRESETS = {
    "describe": "请详细描述这个视频的内容，按时间顺序分析每个场景：画面内容、人物动作、环境设置、文字信息、背景音乐/声音。",
    "summarize": "请简洁概括这个视频的主要内容和关键信息。",
    "scenes": "请将这个视频分解为独立场景，每个场景标注大致时间范围，描述画面内容和发生的事情。",
    "ocr": "请识别并提取视频中出现的所有文字内容（字幕、标题、标识等），按时间顺序列出。",
    "action": "请重点分析视频中人物的具体动作和行为，适合动作分析、运动分析场景。",
    "count": "请统计视频中出现的人数、物体数量等可量化信息。",
}

# ─── Helpers ─────────────────────────────────────────────────────────

def _get_api_key():
    return os.environ.get("XIAOMI_API_KEY", "")


def _is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def _detect_mime(path: Path) -> str:
    """Detect MIME type via magic bytes, fallback to extension."""
    try:
        with open(path, "rb") as f:
            header = f.read(16)
        magic_map = {
            b"\xff\xd8\xff": "image/jpeg",
            b"\x89PNG": "image/png",
            b"GIF8": "image/gif",
            b"RIFF": "image/webp",  # WebP also starts with RIFF
            b"BM": "image/bmp",
            b"\x1a\x45\xdf\xa3": "video/x-matroska",  # mkv/webm
            b"\x00\x00\x00": "video/mp4",  # mp4/mov (ftyp box)
            b"ID3": "audio/mpeg",  # mp3 with ID3
            b"\xff\xfb": "audio/mpeg",  # mp3 frame sync
            b"\xff\xf3": "audio/mpeg",
            b"fLaC": "audio/flac",
            b"OggS": "audio/ogg",
            b"\x30\x26\xb2\x75": "video/x-ms-wmv",
        }
        for magic, mime in magic_map.items():
            if header.startswith(magic):
                return mime
    except Exception:
        pass
    return MIME_MAP.get(path.suffix.lower(), "application/octet-stream")


def _get_media_type(path: Path) -> str:
    """Return 'image', 'audio', or 'video' based on extension."""
    ext = path.suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in AUDIO_EXTS:
        return "audio"
    if ext in VIDEO_EXTS:
        return "video"
    # Fallback: check MIME
    mime = _detect_mime(path)
    if mime.startswith("image/"):
        return "image"
    if mime.startswith("audio/"):
        return "audio"
    return "video"


def _file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def _file_to_base64_uri(path: Path) -> str:
    mime = _detect_mime(path)
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass  # Suppress log noise


def _serve_file_temporarily(file_path: Path, timeout=600) -> str:
    """Start a temp HTTP server to serve a local file. Returns URL."""
    port = _find_free_port()
    parent = file_path.parent
    filename = file_path.name

    server = HTTPServer(("0.0.0.0", port), _QuietHandler)

    # Monkey-patch to serve from the file's directory
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(parent), **kw)
        def log_message(self, *a):
            pass

    server.__class__ = type("S", (HTTPServer,), {})
    server.RequestHandlerClass = Handler
    server = HTTPServer(("0.0.0.0", port), Handler)

    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    # Auto-shutdown after timeout
    def auto_shutdown():
        time.sleep(timeout)
        server.shutdown()
    threading.Thread(target=auto_shutdown, daemon=True).start()

    return f"http://127.0.0.1:{port}/{filename}"


def _resolve_source(source: str, media_type: str) -> str:
    """Resolve source to URL or base64 data URI. Handles large local files."""
    if _is_url(source):
        return source

    p = Path(source).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    size = _file_size_mb(p)
    ext = p.suffix.lower()

    # Validate format
    valid_exts = {
        "image": IMAGE_EXTS,
        "audio": AUDIO_EXTS,
        "video": VIDEO_EXTS,
    }.get(media_type, set())
    if ext not in valid_exts:
        raise ValueError(f"Unsupported format '{ext}'. Supported: {', '.join(sorted(valid_exts))}")

    # For files > 50MB, use temp HTTP server instead of base64
    if size > 50:
        if media_type == "audio":
            raise ValueError(f"Audio file too large ({size:.1f}MB, max 50MB). "
                           f"Provide a public URL instead.")
        if media_type == "image":
            raise ValueError(f"Image file too large ({size:.1f}MB, max 50MB). "
                           f"Compress the image or provide a public URL.")
        # Video: use temp server (supports up to 300MB)
        if size > 300:
            raise ValueError(f"Video file too large ({size:.1f}MB, max 300MB). "
                           f"Provide a public URL or compress the video.")
        return _serve_file_temporarily(p)

    return _file_to_base64_uri(p)


def _get_audio_duration(file_path: Path) -> float:
    """Get audio/video duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(file_path)],
            capture_output=True, text=True, timeout=10
        )
        return float(result.stdout.strip())
    except Exception:
        return 0


def _extract_audio_from_video(video_path: Path) -> Path:
    """Extract audio track from video using ffmpeg. Returns temp audio path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le",
             "-ar", "16000", "-ac", "1", tmp.name],
            capture_output=True, timeout=300
        )
        return Path(tmp.name)
    except Exception as e:
        os.unlink(tmp.name)
        raise RuntimeError(f"Failed to extract audio: {e}")


def _auto_fps(duration: float, requested_fps: float = None) -> float:
    """Auto-tune fps based on video duration for optimal token usage."""
    if requested_fps is not None:
        return max(0.1, min(10.0, requested_fps))
    if duration <= 0:
        return 2.0  # Default
    if duration < 10:
        return 5.0   # Short video: high detail
    if duration < 60:
        return 2.0   # Normal video
    if duration < 300:
        return 1.0   # 1-5 min
    if duration < 600:
        return 0.5   # 5-10 min
    return 0.2        # 10+ min: very sparse sampling


def _estimate_tokens(duration: float, fps: float = None, media_type: str = "video") -> dict:
    """Estimate token usage before sending."""
    if media_type == "audio":
        audio_tokens = int(duration * 6.25)
        return {"audio_tokens": audio_tokens, "total_estimate": audio_tokens}

    fps = fps or 2.0
    # Rough estimate: ~576 visual tokens per frame + audio
    frames = duration * fps
    visual_tokens = int(frames * 576)
    audio_tokens = int(duration * 6.25)
    return {
        "frames": int(frames),
        "visual_tokens": visual_tokens,
        "audio_tokens": audio_tokens,
        "total_estimate": visual_tokens + audio_tokens,
    }


def _call_mimo(messages: list, max_tokens: int = 2048) -> str:
    """Call MiMo API and return structured result."""
    api_key = _get_api_key()
    if not api_key:
        return json.dumps({"error": "XIAOMI_API_KEY not set"})

    payload = {
        "model": MODEL,
        "messages": messages,
        "max_completion_tokens": max_tokens,
    }

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            resp = json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        return json.dumps({"error": f"HTTP {e.code}", "detail": body[:500]})
    except Exception as e:
        return json.dumps({"error": str(e)})

    choice = resp.get("choices", [{}])[0]
    msg = choice.get("message", {})
    content = msg.get("content", "")
    reasoning = msg.get("reasoning_content", "")
    usage = resp.get("usage", {})

    # MiMo may put transcription in reasoning_content
    result = content if content else reasoning

    return json.dumps({
        "result": result,
        "content": content,
        "reasoning": reasoning,
        "usage": usage,
        "model": resp.get("model", MODEL),
    }, ensure_ascii=False)


# ─── Tool Schemas ────────────────────────────────────────────────────

IMAGE_SCHEMA = {
    "name": "image_understand",
    "description": (
        "Analyze images using MiMo AI. Supports description, OCR, explanation, "
        "and information extraction. Accepts local files or URLs.\n"
        "Formats: JPEG, PNG, GIF, WebP, BMP.\n"
        "Presets: describe, ocr, explain, compare, extract_info, caption\n"
        "Or write a custom prompt for specific needs."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": "Local file path or public URL to the image file",
            },
            "prompt": {
                "type": "string",
                "description": "What to do with the image. Use a preset name (describe/ocr/explain/compare/extract_info/caption) or write a custom prompt.",
                "default": "describe",
            },
            "max_tokens": {
                "type": "integer",
                "description": "Max output tokens (default 4096)",
                "default": 4096,
            },
        },
        "required": ["source"],
    },
}

AUDIO_SCHEMA = {
    "name": "audio_understand",
    "description": (
        "Analyze audio files using MiMo AI. Supports transcription, description, summarization, "
        "translation, and info extraction. Accepts local files or URLs.\n"
        "Formats: MP3, WAV, FLAC, M4A, OGG, AAC, WMA.\n"
        "Presets: transcribe, describe, summarize, translate, extract_info\n"
        "Or write a custom prompt for specific needs."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": "Local file path or public URL to the audio file",
            },
            "prompt": {
                "type": "string",
                "description": "What to do with the audio. Use a preset name (transcribe/describe/summarize/translate/extract_info) or write a custom prompt.",
                "default": "transcribe",
            },
            "max_tokens": {
                "type": "integer",
                "description": "Max output tokens (default 4096)",
                "default": 4096,
            },
        },
        "required": ["source"],
    },
}

VIDEO_SCHEMA = {
    "name": "video_understand",
    "description": (
        "Analyze video files using MiMo AI. Supports scene description, summarization, "
        "scene breakdown, OCR, action analysis, and object counting.\n"
        "Formats: MP4, MOV, AVI, WMV, MKV, WEBM, FLV.\n"
        "Presets: describe, summarize, scenes, ocr, action, count\n"
        "Auto-tunes fps based on video length to balance detail and token cost."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": "Local file path or public URL to the video file",
            },
            "prompt": {
                "type": "string",
                "description": "What to do with the video. Use a preset name (describe/summarize/scenes/ocr/action/count) or write a custom prompt.",
                "default": "describe",
            },
            "fps": {
                "type": "number",
                "description": "Frames per second (0.1-10). Omit for auto-tuning based on video duration.",
            },
            "media_resolution": {
                "type": "string",
                "description": "Resolution: 'default' (balanced) or 'max' (best for small text/objects)",
                "enum": ["default", "max"],
                "default": "default",
            },
            "extract_audio": {
                "type": "boolean",
                "description": "If true, also extract and transcribe the audio track from the video (requires ffmpeg).",
                "default": False,
            },
            "max_tokens": {
                "type": "integer",
                "description": "Max output tokens (default 4096)",
                "default": 4096,
            },
        },
        "required": ["source"],
    },
}


# ─── Handlers ────────────────────────────────────────────────────────

def image_understand_handler(args: dict, **kwargs) -> str:
    source = args.get("source", "")
    prompt = args.get("prompt", "describe")
    max_tokens = args.get("max_tokens", 4096)

    if not source:
        return json.dumps({"error": "source is required"})

    # Resolve preset
    system_prompt = IMAGE_SYSTEM_PROMPT
    user_prompt = IMAGE_PRESETS.get(prompt, prompt)

    # Resolve source
    try:
        image_data = _resolve_source(source, "image")
    except (FileNotFoundError, ValueError) as e:
        return json.dumps({"error": str(e)})

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data}},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

    return _call_mimo(messages, max_tokens)


def audio_understand_handler(args: dict, **kwargs) -> str:
    source = args.get("source", "")
    prompt = args.get("prompt", "transcribe")
    max_tokens = args.get("max_tokens", 4096)

    if not source:
        return json.dumps({"error": "source is required"})

    # Resolve preset
    system_prompt = AUDIO_SYSTEM_PROMPT
    user_prompt = AUDIO_PRESETS.get(prompt, prompt)

    # Resolve source
    try:
        audio_data = _resolve_source(source, "audio")
    except (FileNotFoundError, ValueError) as e:
        return json.dumps({"error": str(e)})

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "input_audio", "input_audio": {"data": audio_data}},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

    return _call_mimo(messages, max_tokens)


def video_understand_handler(args: dict, **kwargs) -> str:
    source = args.get("source", "")
    prompt = args.get("prompt", "describe")
    fps = args.get("fps", None)  # None = auto
    media_resolution = args.get("media_resolution", "default")
    extract_audio = args.get("extract_audio", False)
    max_tokens = args.get("max_tokens", 4096)

    if not source:
        return json.dumps({"error": "source is required"})

    # Resolve preset
    system_prompt = VIDEO_SYSTEM_PROMPT
    user_prompt = VIDEO_PRESETS.get(prompt, prompt)

    # Resolve source
    try:
        video_data = _resolve_source(source, "video")
    except (FileNotFoundError, ValueError) as e:
        return json.dumps({"error": str(e)})

    # Auto-tune fps if local file
    duration = 0
    if not _is_url(source):
        p = Path(source).expanduser().resolve()
        duration = _get_audio_duration(p)
        if duration > 0 and fps is None:
            fps = _auto_fps(duration)
            # Add token estimate hint
            est = _estimate_tokens(duration, fps, "video")
            user_prompt += f"\n\n[系统提示: 视频时长 {duration:.1f}秒, 采样fps={fps}, 预估消耗 {est['total_estimate']} tokens]"

    fps = fps if fps is not None else 2.0

    content_items = [
        {
            "type": "video_url",
            "video_url": {"url": video_data},
            "fps": fps,
            "media_resolution": media_resolution,
        },
        {"type": "text", "text": user_prompt},
    ]

    # Optionally extract audio track
    if extract_audio and not _is_url(source):
        try:
            p = Path(source).expanduser().resolve()
            audio_path = _extract_audio_from_video(p)
            audio_data = _file_to_base64_uri(audio_path)
            content_items.insert(1, {
                "type": "input_audio",
                "input_audio": {"data": audio_data},
            })
            os.unlink(audio_path)
            user_prompt += "\n\n[系统提示: 已同时提取音频轨道，请结合视频画面和音频进行分析]"
            content_items[-1] = {"type": "text", "text": user_prompt}
        except Exception as e:
            user_prompt += f"\n\n[注意: 音频提取失败({e})，仅分析视频画面]"
            content_items[-1] = {"type": "text", "text": user_prompt}

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content_items},
    ]

    return _call_mimo(messages, max_tokens)


def check_requirements() -> bool:
    return bool(_get_api_key())


def register(ctx):
    ctx.register_tool(
        name="image_understand",
        toolset="vision",
        schema=IMAGE_SCHEMA,
        handler=image_understand_handler,
        check_fn=check_requirements,
        emoji="🖼️",
    )
    ctx.register_tool(
        name="audio_understand",
        toolset="vision",
        schema=AUDIO_SCHEMA,
        handler=audio_understand_handler,
        check_fn=check_requirements,
        emoji="🎵",
    )
    ctx.register_tool(
        name="video_understand",
        toolset="vision",
        schema=VIDEO_SCHEMA,
        handler=video_understand_handler,
        check_fn=check_requirements,
        emoji="🎬",
    )
