"""Native audio routing and inline payload construction for gateway voice notes."""

from __future__ import annotations

import base64
import logging
import mimetypes
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from tools.audio_container import sniff_container

logger = logging.getLogger(__name__)

# Limit the encoded payload, not only the source file: Base64 expands data by
# roughly one third and the encoded bytes are what enter the provider request.
MAX_INLINE_AUDIO_BASE64_BYTES = 25 * 1024 * 1024

AUDIO_FORMAT_TO_MIME: Dict[str, str] = {
    "mp3": "audio/mpeg",
    "mpeg": "audio/mpeg",
    "mp4": "audio/mp4",
    "m4a": "audio/m4a",
    "ogg": "audio/ogg",
    "oga": "audio/ogg",
    "opus": "audio/ogg",
    "wav": "audio/wav",
    "wave": "audio/wav",
    "flac": "audio/flac",
    "aac": "audio/aac",
    "webm": "audio/webm",
}

_OPENAI_AUDIO_FORMAT_PROVIDERS = {
    "openai",
    "azure",
    "azure-foundry",
    "azure-openai",
}


def _canonical_audio_format(value: str) -> str:
    clean = (value or "").strip().lower().lstrip(".")
    if "/" in clean:
        clean = clean.rsplit("/", 1)[-1]
    if clean in {"mpeg", "mp3"}:
        return "mp3"
    if clean in {"ogg", "oga", "opus"}:
        return "ogg"
    if clean in {"wav", "wave"}:
        return "wav"
    return clean if clean in AUDIO_FORMAT_TO_MIME else ""


def normalize_audio_format(
    path: Optional[Path] = None,
    mime: str = "",
    *,
    header: bytes = b"",
) -> str:
    """Return a canonical provider format, preferring shared magic-byte detection."""
    detected = sniff_container(header) if header else None
    if detected is None and path and path.is_file():
        try:
            with path.open("rb") as source:
                detected = sniff_container(source.read(32))
        except OSError:
            detected = None
    if detected:
        return _canonical_audio_format(detected)
    if path and path.suffix:
        suffix_format = _canonical_audio_format(path.suffix)
        if suffix_format:
            return suffix_format
    return _canonical_audio_format(mime)


def normalize_audio_mime(fmt_or_ext: str) -> str:
    """Convert a supported audio format, extension, or MIME to a canonical MIME."""
    clean = (fmt_or_ext or "").strip().lower().lstrip(".")
    if clean.startswith("audio/"):
        clean = clean.rsplit("/", 1)[-1]
    canonical = _canonical_audio_format(clean)
    return AUDIO_FORMAT_TO_MIME.get(canonical, "audio/mpeg")


def supported_input_modalities(provider: str, model: str) -> Set[str]:
    """Return known native input modalities; unknown models fail closed."""
    try:
        from agent.models_dev import get_model_info

        info = get_model_info(provider, model, allow_network=True)
        if info is None and (provider or "").strip().lower() == "openrouter" and "/" in model:
            vendor, bare_model = model.split("/", 1)
            info = get_model_info(vendor, bare_model, allow_network=True)
        if info is None:
            return set()
        return {
            str(modality).strip().lower()
            for modality in (getattr(info, "input_modalities", ()) or ())
            if modality
        }
    except Exception as exc:  # pragma: no cover - defensive catalog boundary
        logger.debug(
            "audio_routing: capability lookup failed for %s:%s: %s",
            provider,
            model,
            exc,
        )
        return set()


def decide_audio_input_mode(
    provider: str,
    model: str,
    configured_mode: str = "auto",
) -> str:
    """Return ``native`` or ``stt`` for one resolved model turn."""
    provider_norm = (provider or "").strip().lower()
    model_norm = (model or "").strip().lower()
    # api.meta.ai has rejected input_audio for models whose catalog metadata
    # advertised it. Keep this as a hard safety rule, including explicit native.
    if provider_norm in {"meta", "meta-ai"} or "muse-spark" in model_norm or "muse_spark" in model_norm:
        return "stt"

    mode = (configured_mode or "auto").strip().lower()
    if mode == "stt":
        return "stt"
    if mode == "native":
        return "native"
    return "native" if "audio" in supported_input_modalities(provider, model) else "stt"


def _encoded_size(raw_size: int) -> int:
    return 4 * ((raw_size + 2) // 3)


def _read_audio_bytes(path: Path, max_encoded_bytes: int) -> Optional[bytes]:
    try:
        from agent.file_safety import raise_if_read_blocked

        raise_if_read_blocked(str(path))
    except ValueError as exc:
        logger.warning("audio_routing: blocked local voice attachment %s -- %s", path, exc)
        return None
    except Exception:
        pass

    try:
        if not path.is_file():
            return None
        size = path.stat().st_size
        if size <= 0 or _encoded_size(size) > max_encoded_bytes:
            logger.warning(
                "audio_routing: %s exceeds the %d-byte encoded inline limit or is empty",
                path,
                max_encoded_bytes,
            )
            return None
        raw = path.read_bytes()
        if not raw or _encoded_size(len(raw)) > max_encoded_bytes:
            logger.warning(
                "audio_routing: %s changed while reading and no longer fits the encoded inline limit",
                path,
            )
            return None
        return raw
    except OSError as exc:
        logger.warning("audio_routing: failed to read %s: %s", path, exc)
        return None


def transcode_audio_to_supported_format(
    path: Path,
    target_format: str = "mp3",
) -> Optional[Tuple[bytes, str]]:
    """Best-effort ffmpeg conversion for providers limited to MP3/WAV input."""
    ffmpeg = shutil.which("ffmpeg")
    target = _canonical_audio_format(target_format)
    if not ffmpeg or not path.is_file() or target not in {"mp3", "wav"}:
        return None

    output_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=f".{target}", delete=False) as output:
            output_path = Path(output.name)
        result = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(path),
                "-vn",
                "-ar",
                "24000",
                "-ac",
                "1",
                str(output_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=False,
        )
        if result.returncode == 0 and output_path.is_file():
            raw = output_path.read_bytes()
            if raw:
                return raw, target
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("audio_routing: ffmpeg conversion failed for %s: %s", path, exc)
    finally:
        if output_path is not None:
            try:
                output_path.unlink(missing_ok=True)
            except OSError:
                logger.debug("audio_routing: could not remove temporary conversion output %s", output_path)
    return None


def _attachment_hint(path: str, size: int, mime: str) -> str:
    try:
        from tools.credential_files import to_agent_visible_cache_path

        visible_path = to_agent_visible_cache_path(path)
    except Exception:
        visible_path = path
    size_text = f"{size / 1024:.1f} KB" if size < 1024 * 1024 else f"{size / (1024 * 1024):.1f} MB"
    return (
        f"[Voice message ({size_text}, {mime}) attached natively at: {visible_path}. "
        "Inspect the native audio directly.]"
    )


def build_native_audio_content_parts(
    user_text: str,
    attachments: Iterable[Dict[str, str]],
    *,
    target_provider: str = "",
    max_encoded_bytes: int = MAX_INLINE_AUDIO_BASE64_BYTES,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Build OpenAI-style ``input_audio`` parts and report every rejected path."""
    audio_parts: List[Dict[str, Any]] = []
    hints: List[str] = []
    skipped: List[str] = []
    provider = (target_provider or "").strip().lower()

    for item in attachments:
        if isinstance(item, (str, Path)):
            raw_path, declared_mime = str(item), ""
        elif isinstance(item, dict):
            raw_path = str(item.get("path") or "")
            declared_mime = str(item.get("mime_type") or "")
        else:
            continue
        if not raw_path:
            continue

        path = Path(raw_path)
        raw = _read_audio_bytes(path, max_encoded_bytes)
        if raw is None:
            skipped.append(raw_path)
            continue
        audio_format = normalize_audio_format(path, declared_mime, header=raw[:32])
        if not audio_format:
            guessed_mime, _ = mimetypes.guess_type(str(path))
            audio_format = _canonical_audio_format(guessed_mime or "")
        if not audio_format:
            logger.warning("audio_routing: unsupported voice container %s", path)
            skipped.append(raw_path)
            continue

        if provider in _OPENAI_AUDIO_FORMAT_PROVIDERS and audio_format not in {"mp3", "wav"}:
            converted = transcode_audio_to_supported_format(path, "mp3")
            if converted is None:
                logger.warning(
                    "audio_routing: %s requires MP3/WAV and %s could not be converted",
                    provider,
                    path,
                )
                skipped.append(raw_path)
                continue
            raw, audio_format = converted
            if _encoded_size(len(raw)) > max_encoded_bytes:
                logger.warning("audio_routing: converted voice attachment %s exceeds the encoded limit", path)
                skipped.append(raw_path)
                continue

        encoded = base64.b64encode(raw)
        if len(encoded) > max_encoded_bytes:
            skipped.append(raw_path)
            continue
        mime = AUDIO_FORMAT_TO_MIME[audio_format]
        audio_parts.append(
            {
                "type": "input_audio",
                "input_audio": {"data": encoded.decode("ascii"), "format": audio_format},
            }
        )
        hints.append(_attachment_hint(raw_path, len(raw), mime))

    text = (user_text or "").strip()
    if not audio_parts:
        return ([{"type": "text", "text": text}] if text else []), skipped
    prompt = text or "Analyze the attached voice message."
    return [{"type": "text", "text": f"{prompt}\n\n" + "\n".join(hints)}, *audio_parts], skipped


__all__ = [
    "AUDIO_FORMAT_TO_MIME",
    "MAX_INLINE_AUDIO_BASE64_BYTES",
    "build_native_audio_content_parts",
    "decide_audio_input_mode",
    "normalize_audio_format",
    "normalize_audio_mime",
    "supported_input_modalities",
    "transcode_audio_to_supported_format",
]
