"""Native multimodal attachment helpers for inbound gateway messages."""

from __future__ import annotations

import base64
import logging
import mimetypes
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Maximum file size allowed for inline base64 embedding (25 MB).
# Files exceeding this threshold fall back to STT / tool execution to prevent
# memory spikes and provider HTTP 413 (Payload Too Large) errors.
MAX_ATTACHMENT_SIZE_BYTES = 25 * 1024 * 1024

AUDIO_FORMAT_TO_MIME: Dict[str, str] = {
    "mp3": "audio/mpeg",
    "mpeg": "audio/mpeg",
    "mp4": "audio/mp4",
    "m4a": "audio/mp4",
    "ogg": "audio/ogg",
    "oga": "audio/ogg",
    "opus": "audio/ogg",
    "wav": "audio/wav",
    "wave": "audio/wav",
    "flac": "audio/flac",
    "aac": "audio/aac",
    "webm": "audio/webm",
}


def normalize_audio_format(path: Optional[Path] = None, mime: str = "") -> str:
    """Return the canonical format identifier for OpenAI input_audio (e.g. mp3, ogg, wav, flac)."""
    fmt = ""
    # Sniff magic bytes first if a readable file is provided
    if path and path.is_file():
        try:
            with path.open("rb") as f:
                header = f.read(32)
            from tools.audio_container import sniff_container

            container = sniff_container(header)
            if container:
                fmt = container
        except Exception:
            pass

    if not fmt and path and path.suffix:
        fmt = path.suffix.lower().lstrip(".")
    if not fmt and mime:
        fmt = mime.rsplit("/", 1)[-1].lower()
    if fmt in ("mpeg", "mp3"):
        return "mp3"
    if fmt in ("ogg", "oga", "opus"):
        return "ogg"
    if fmt in ("wav", "wave"):
        return "wav"
    return fmt or "mp3"


def normalize_audio_mime(fmt_or_ext: str) -> str:
    """Convert an audio format or extension to a canonical MIME type."""
    clean = (fmt_or_ext or "").strip().lower().lstrip(".")
    if "/" in clean:
        return clean
    return AUDIO_FORMAT_TO_MIME.get(clean, f"audio/{clean or 'mpeg'}")


def transcode_audio_to_supported_format(
    path: Path,
    target_format: str = "mp3",
) -> Optional[Tuple[bytes, str]]:
    """Transcode an audio file to mp3 or wav using ffmpeg if available.

    Returns (transcoded_bytes, new_format) or None if ffmpeg is unavailable or fails.
    """
    import shutil
    import subprocess
    import tempfile

    if not shutil.which("ffmpeg") or not path.is_file():
        return None

    out_suffix = f".{target_format.lstrip('.')}"
    tmp_out_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=out_suffix, delete=False) as tmp_out:
            tmp_out_path = Path(tmp_out.name)
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(path),
            "-vn",
            "-ar",
            "24000",
            "-ac",
            "1",
            str(tmp_out_path),
        ]
        res = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
        if res.returncode == 0 and tmp_out_path.is_file() and tmp_out_path.stat().st_size > 0:
            data = tmp_out_path.read_bytes()
            tmp_out_path.unlink(missing_ok=True)
            return data, target_format
        if tmp_out_path and tmp_out_path.exists():
            tmp_out_path.unlink(missing_ok=True)
    except Exception as exc:
        logger.debug("media_routing: audio transcoding failed: %s", exc)
        if tmp_out_path and tmp_out_path.exists():
            tmp_out_path.unlink(missing_ok=True)
    return None


def supported_input_modalities(provider: str, model: str) -> Set[str]:
    """Return the model's known native input modalities.

    An empty set is deliberately conservative: callers retain their existing
    tool/path fallback when the model is absent from the capability registry.
    """
    try:
        from agent.models_dev import get_model_info

        info = get_model_info(provider, model)
        if info is None and provider == "openrouter" and "/" in model:
            vendor, bare_model = model.split("/", 1)
            info = get_model_info(vendor, bare_model)
        if info is None:
            return set()
        return {str(item).strip().lower() for item in info.input_modalities if item}
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("media_routing: capability lookup failed for %s:%s: %s", provider, model, exc)
        return set()


def _read_as_base64(
    path: Path,
    max_size_bytes: int = MAX_ATTACHMENT_SIZE_BYTES,
) -> Optional[str]:
    try:
        size = path.stat().st_size
        if size > max_size_bytes:
            logger.warning(
                "media_routing: file %s exceeds %d MB limit (%d bytes), skipping native attachment",
                path,
                max_size_bytes // (1024 * 1024),
                size,
            )
            return None
        return base64.b64encode(path.read_bytes()).decode("ascii")
    except Exception as exc:
        logger.warning("media_routing: failed to read %s: %s", path, exc)
        return None


def _mime_type(path: Path, declared_mime: str) -> str:
    if declared_mime and declared_mime != "application/octet-stream":
        return declared_mime
    # Sniff magic bytes for accurate audio/AV container mime resolution
    try:
        if path.is_file():
            with path.open("rb") as f:
                header = f.read(32)
            from tools.audio_container import sniff_container

            container = sniff_container(header)
            if container and container in AUDIO_FORMAT_TO_MIME:
                return AUDIO_FORMAT_TO_MIME[container]
    except Exception:
        pass
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"


def build_native_media_content_parts(
    user_text: str,
    attachments: Iterable[Dict[str, str]],
    max_size_bytes: int = MAX_ATTACHMENT_SIZE_BYTES,
    target_provider: str = "",
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Build OpenRouter/OpenAI-style multimodal content parts.

    Each attachment dict contains ``path``, ``mime_type`` and ``modality``.
    Supported modalities are image, pdf, audio and video. Local content is
    embedded so private gateway cache files never need a public URL.
    """
    media_parts: List[Dict[str, Any]] = []
    hints: List[str] = []
    skipped: List[str] = []

    for attachment in attachments:
        raw_path = str(attachment.get("path") or "")
        path = Path(raw_path)
        if not raw_path or not path.is_file():
            skipped.append(raw_path)
            continue

        encoded = _read_as_base64(path, max_size_bytes=max_size_bytes)
        if encoded is None:
            skipped.append(raw_path)
            continue
        mime = _mime_type(path, str(attachment.get("mime_type") or ""))

        modality = str(attachment.get("modality") or "").lower()
        if not modality:
            if mime.startswith("image/"):
                modality = "image"
            elif mime.startswith("audio/"):
                modality = "audio"
            elif mime.startswith("video/"):
                modality = "video"
            elif mime == "application/pdf" or path.suffix.lower() == ".pdf":
                modality = "pdf"

        if modality == "image":
            media_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{encoded}"},
            })
        elif modality == "pdf":
            media_parts.append({
                "type": "file",
                "file": {
                    "filename": path.name,
                    "file_data": f"data:{mime};base64,{encoded}",
                },
            })
        elif modality == "audio":
            audio_format = normalize_audio_format(path, mime)
            audio_data = encoded
            # Transcode to mp3 if required by OpenAI direct input_audio schema
            if (
                target_provider in ("openai", "azure")
                and audio_format not in ("wav", "mp3")
            ):
                transcoded = transcode_audio_to_supported_format(path, target_format="mp3")
                if transcoded:
                    t_bytes, t_fmt = transcoded
                    audio_data = base64.b64encode(t_bytes).decode("ascii")
                    audio_format = t_fmt
                    mime = "audio/mpeg"

            media_parts.append({
                "type": "input_audio",
                "input_audio": {"data": audio_data, "format": audio_format},
            })
        elif modality == "video":
            media_parts.append({
                "type": "video_url",
                "video_url": {"url": f"data:{mime};base64,{encoded}"},
            })
        else:
            skipped.append(raw_path)
            continue

        file_size_bytes = path.stat().st_size if path.is_file() else 0
        size_str = (
            f"{file_size_bytes / 1024:.1f} KB"
            if file_size_bytes < 1024 * 1024
            else f"{file_size_bytes / (1024 * 1024):.1f} MB"
        )
        hints.append(
            f"[{modality.title()} attachment ({size_str}, {mime}) attached natively to this model request at: "
            f"{raw_path}. Inspect the native attachment directly. Do not claim "
            "that you used a terminal command or extraction tool unless you "
            "actually called one in this turn.]"
        )

    text = (user_text or "").strip()
    if not media_parts:
        return ([{"type": "text", "text": text}] if text else []), skipped

    prompt = text or "Analyze the attached media."
    parts: List[Dict[str, Any]] = [{"type": "text", "text": f"{prompt}\n\n" + "\n".join(hints)}]
    parts.extend(media_parts)
    return parts, skipped


__all__ = [
    "AUDIO_FORMAT_TO_MIME",
    "MAX_ATTACHMENT_SIZE_BYTES",
    "build_native_media_content_parts",
    "normalize_audio_format",
    "normalize_audio_mime",
    "supported_input_modalities",
    "transcode_audio_to_supported_format",
]


