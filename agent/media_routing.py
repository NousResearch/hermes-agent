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
    if path and path.suffix:
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
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"


def build_native_media_content_parts(
    user_text: str,
    attachments: Iterable[Dict[str, str]],
    max_size_bytes: int = MAX_ATTACHMENT_SIZE_BYTES,
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
            media_parts.append({
                "type": "input_audio",
                "input_audio": {"data": encoded, "format": audio_format},
            })
        elif modality == "video":
            media_parts.append({
                "type": "video_url",
                "video_url": {"url": f"data:{mime};base64,{encoded}"},
            })
        else:
            skipped.append(raw_path)
            continue
        hints.append(
            f"[{modality.title()} attached natively to this model request at: "
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
]

