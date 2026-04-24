# -*- coding: utf-8 -*-
"""Audio conversion and speech-to-text utilities for QQ Bot voice messages.

All functions are **stateless** — they receive explicit parameters and return
results without depending on any adapter instance state.

Conversion strategy (unified pipeline)::

    1. pilk  — handles QQ's native .silk format
    2. ffmpeg — handles all standard audio formats
    3. raw PCM fallback — last resort, may produce low-quality output

Zero hermes dependencies.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ── Voice detection ───────────────────────────────────────────────────

_VOICE_EXTENSIONS = (
    ".silk",
    ".amr",
    ".mp3",
    ".wav",
    ".ogg",
    ".m4a",
    ".aac",
    ".speex",
    ".flac",
)

_KNOWN_AUDIO_EXTENSIONS = frozenset(_VOICE_EXTENSIONS)


def is_voice_content_type(content_type: str, filename: str) -> bool:
    """Return ``True`` if the attachment is a voice/audio message.

    :param content_type: MIME content-type string.
    :param filename: Attachment filename (used as fallback extension check).
    """
    ct = content_type.strip().lower()
    fn = filename.strip().lower()
    if ct == "voice" or ct.startswith("audio/"):
        return True
    return any(fn.endswith(ext) for ext in _VOICE_EXTENSIONS)


# ── Magic-byte detection ─────────────────────────────────────────────

def guess_audio_ext(data: bytes) -> str:
    """Guess audio file extension from magic bytes.

    :param data: Raw audio bytes (at least the first few bytes).
    :returns: Extension string including the leading dot (e.g. ``'.silk'``).
    """
    if data[:9] == b"#!SILK_V3" or data[:6] == b"#!SILK":
        return ".silk"
    if data[:2] == b"\x02!":
        return ".silk"
    if data[:4] == b"RIFF":
        return ".wav"
    if data[:4] == b"fLaC":
        return ".flac"
    if data[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return ".mp3"
    if data[:4] in (b"\x30\x26\xb2\x75", b"\x4f\x67\x67\x53"):
        return ".ogg"
    if data[:4] in (b"\x00\x00\x00\x20", b"\x00\x00\x00\x1c"):
        return ".amr"
    # Default to .amr — most common QQ voice format.
    return ".amr"


def looks_like_silk(data: bytes) -> bool:
    """Return ``True`` if the bytes appear to be a SILK audio file."""
    return (
        data[:9] == b"#!SILK_V3"
        or data[:6] == b"#!SILK"
        or data[:2] == b"\x02!"
    )


# ── Conversion: pilk ──────────────────────────────────────────────────

async def convert_silk_to_wav(
    src_path: str,
    wav_path: str,
    log_tag: str = "QQBot",
) -> Optional[str]:
    """Convert audio to WAV using the pilk library.

    Tries the file as-is first, then copies to ``.silk`` extension and retries
    (pilk checks the extension to select the decoder).

    :param src_path: Path to the source audio file.
    :param wav_path: Desired output WAV path.
    :param log_tag: Log prefix.
    :returns: *wav_path* on success, ``None`` on failure.
    """
    try:
        import pilk
    except ImportError:
        logger.warning(
            "[%s] pilk not installed — cannot decode SILK audio. Run: pip install pilk",
            log_tag,
        )
        return None

    # Attempt 1: convert as-is.
    try:
        pilk.silk_to_wav(src_path, wav_path, rate=16000)
        if Path(wav_path).exists() and Path(wav_path).stat().st_size > 44:
            logger.debug(
                "[%s] pilk converted %s → wav (%d bytes)",
                log_tag,
                Path(src_path).name,
                Path(wav_path).stat().st_size,
            )
            return wav_path
    except Exception as exc:
        logger.debug("[%s] pilk direct conversion failed: %s", log_tag, exc)

    # Attempt 2: rename to .silk and retry.
    silk_path = src_path.rsplit(".", 1)[0] + ".silk"
    try:
        import shutil

        shutil.copy2(src_path, silk_path)
        pilk.silk_to_wav(silk_path, wav_path, rate=16000)
        if Path(wav_path).exists() and Path(wav_path).stat().st_size > 44:
            logger.debug(
                "[%s] pilk converted %s (as .silk) → wav (%d bytes)",
                log_tag,
                Path(src_path).name,
                Path(wav_path).stat().st_size,
            )
            return wav_path
    except Exception as exc:
        logger.debug("[%s] pilk .silk conversion failed: %s", log_tag, exc)
    finally:
        try:
            os.unlink(silk_path)
        except OSError:
            pass

    return None


# ── Conversion: ffmpeg ────────────────────────────────────────────────

async def convert_ffmpeg_to_wav(
    src_path: str,
    wav_path: str,
    log_tag: str = "QQBot",
) -> Optional[str]:
    """Convert audio to WAV using ffmpeg (16 kHz mono).

    :param src_path: Path to the source audio file.
    :param wav_path: Desired output WAV path.
    :param log_tag: Log prefix.
    :returns: *wav_path* on success, ``None`` on failure.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y",
            "-i", src_path,
            "-ar", "16000",
            "-ac", "1",
            wav_path,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.wait(), timeout=30)
        if proc.returncode != 0:
            stderr = await proc.stderr.read() if proc.stderr else b""
            logger.warning(
                "[%s] ffmpeg failed for %s: %s",
                log_tag,
                Path(src_path).name,
                stderr[:200].decode(errors="replace"),
            )
            return None
    except (asyncio.TimeoutError, FileNotFoundError) as exc:
        logger.warning("[%s] ffmpeg conversion error: %s", log_tag, exc)
        return None

    if not Path(wav_path).exists() or Path(wav_path).stat().st_size <= 44:
        logger.warning(
            "[%s] ffmpeg produced no/empty output for %s",
            log_tag,
            Path(src_path).name,
        )
        return None

    logger.debug(
        "[%s] ffmpeg converted %s → wav (%d bytes)",
        log_tag,
        Path(src_path).name,
        Path(wav_path).stat().st_size,
    )
    return wav_path


# ── Conversion: raw PCM fallback ─────────────────────────────────────

async def convert_raw_to_wav(
    audio_data: bytes,
    wav_path: str,
    log_tag: str = "QQBot",
) -> Optional[str]:
    """Last resort: write audio bytes as raw PCM 16-bit mono 16 kHz WAV.

    :param audio_data: Raw audio bytes to wrap in a WAV container.
    :param wav_path: Desired output WAV path.
    :param log_tag: Log prefix.
    :returns: *wav_path* on success, ``None`` on failure.
    """
    try:
        import wave

        with wave.open(wav_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_data)
        return wav_path
    except Exception as exc:
        logger.debug("[%s] raw PCM fallback failed: %s", log_tag, exc)
        return None


# ── Unified conversion pipeline ───────────────────────────────────────

async def convert_audio_to_wav(
    audio_data: bytes,
    source_hint: str = "",
    log_tag: str = "QQBot",
) -> Optional[str]:
    """Convert raw audio bytes to a temporary WAV file.

    Tries converters in order: pilk → ffmpeg → raw PCM.
    The caller is responsible for deleting the returned file.

    :param audio_data: Raw audio bytes.
    :param source_hint: Filename or URL used for extension guessing.
    :param log_tag: Log prefix.
    :returns: Path to the converted WAV file, or ``None`` on failure.
    """
    ext = _resolve_audio_ext(audio_data, source_hint)

    logger.info(
        "[%s] audio_data size=%d ext=%r first_20_bytes=%r",
        log_tag,
        len(audio_data),
        ext,
        audio_data[:20],
    )

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_src:
        tmp_src.write(audio_data)
        src_path = tmp_src.name

    wav_path = src_path.rsplit(".", 1)[0] + ".wav"

    try:
        result = await convert_silk_to_wav(src_path, wav_path, log_tag)
        if not result:
            result = await convert_ffmpeg_to_wav(src_path, wav_path, log_tag)
        if not result:
            result = await convert_raw_to_wav(audio_data, wav_path, log_tag)
        return result
    finally:
        try:
            os.unlink(src_path)
        except OSError:
            pass


def _resolve_audio_ext(audio_data: bytes, source_hint: str) -> str:
    """Determine the best file extension for *audio_data*.

    Prefers the extension from *source_hint* if it is a known audio format,
    otherwise falls back to magic-byte detection.
    """
    if source_hint:
        suffix = Path(source_hint).suffix.lower()
        if suffix in _KNOWN_AUDIO_EXTENSIONS:
            return suffix
    return guess_audio_ext(audio_data)


# ── STT configuration ─────────────────────────────────────────────────

_STT_PROVIDER_URLS: Dict[str, str] = {
    "zai": "https://open.bigmodel.cn/api/coding/paas/v4",
    "openai": "https://api.openai.com/v1",
    "glm": "https://open.bigmodel.cn/api/coding/paas/v4",
}

_STT_PROVIDER_DEFAULT_MODELS: Dict[str, str] = {
    "zai": "glm-asr",
    "glm": "glm-asr",
}


def resolve_stt_config(extra: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Resolve STT backend configuration from config dict or environment.

    Priority:

    1. ``extra["stt"]`` plugin config (``baseUrl`` + ``apiKey``)
    2. Provider-only config (``extra["stt"]["provider"]`` → derive base URL)
    3. ``QQ_STT_API_KEY`` / ``QQ_STT_BASE_URL`` / ``QQ_STT_MODEL`` env vars

    :param extra: The ``config.extra`` dict from the adapter.
    :returns: Dict with ``base_url``, ``api_key``, ``model`` keys,
        or ``None`` if no STT is configured.
    """
    stt_cfg = extra.get("stt")
    if isinstance(stt_cfg, dict) and stt_cfg.get("enabled") is not False:
        result = _resolve_stt_from_config(stt_cfg)
        if result:
            return result

    return _resolve_stt_from_env()


def _resolve_stt_from_config(stt_cfg: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Extract STT config from the ``stt`` config sub-dict."""
    base_url = str(stt_cfg.get("baseUrl") or stt_cfg.get("base_url", "")).rstrip("/")
    api_key = str(stt_cfg.get("apiKey") or stt_cfg.get("api_key", ""))
    model = str(stt_cfg.get("model", ""))

    if not api_key:
        return None

    if base_url:
        return {
            "base_url": base_url,
            "api_key": api_key,
            "model": model or "whisper-1",
        }

    # Provider-only: derive base URL from provider name.
    provider = str(stt_cfg.get("provider", "zai"))
    derived_url = _STT_PROVIDER_URLS.get(provider, "")
    if not derived_url:
        return None

    default_model = _STT_PROVIDER_DEFAULT_MODELS.get(provider, "whisper-1")
    return {
        "base_url": derived_url,
        "api_key": api_key,
        "model": model or default_model,
    }


def _resolve_stt_from_env() -> Optional[Dict[str, str]]:
    """Extract STT config from environment variables."""
    api_key = os.getenv("QQ_STT_API_KEY", "")
    if not api_key:
        return None
    return {
        "base_url": os.getenv(
            "QQ_STT_BASE_URL",
            "https://open.bigmodel.cn/api/coding/paas/v4",
        ).rstrip("/"),
        "api_key": api_key,
        "model": os.getenv("QQ_STT_MODEL", "glm-asr"),
    }


# ── STT API call ──────────────────────────────────────────────────────

async def call_stt(
    http_client: Any,
    wav_path: str,
    stt_config: Dict[str, str],
    log_tag: str = "QQBot",
) -> Optional[str]:
    """Transcribe a WAV file using an OpenAI-compatible STT API.

    Supports both GLM/Zhipu format (``choices[0].message.content``) and
    standard OpenAI/Whisper format (``text``).

    :param http_client: ``httpx.AsyncClient`` instance.
    :param wav_path: Path to the ``.wav`` file to transcribe.
    :param stt_config: Dict with ``base_url``, ``api_key``, ``model`` keys.
    :param log_tag: Log prefix.
    :returns: Transcript text, or ``None`` on failure.
    """
    base_url = stt_config["base_url"]
    api_key = stt_config["api_key"]
    model = stt_config["model"]

    try:
        with open(wav_path, "rb") as f:
            resp = await http_client.post(
                f"{base_url}/audio/transcriptions",
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": (Path(wav_path).name, f, "audio/wav")},
                data={"model": model},
                timeout=30.0,
            )
        resp.raise_for_status()
        result = resp.json()

        return _extract_stt_text(result)

    except Exception as exc:
        logger.warning(
            "[%s] STT API call failed (model=%s, base=%s): %s",
            log_tag,
            model,
            base_url[:50],
            exc,
        )
        return None


def _extract_stt_text(result: Dict[str, Any]) -> Optional[str]:
    """Extract transcript text from an STT API response.

    Handles both GLM/Zhipu and OpenAI/Whisper response formats.
    """
    # GLM/Zhipu format: {"choices": [{"message": {"content": "..."}}]}
    choices = result.get("choices", [])
    if choices:
        content = choices[0].get("message", {}).get("content", "")
        if content.strip():
            return content.strip()

    # OpenAI/Whisper format: {"text": "..."}
    text = result.get("text", "")
    return text.strip() if text.strip() else None
