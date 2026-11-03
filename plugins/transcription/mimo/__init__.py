"""Xiaomi MiMo ASR backend plugin.

Exposes MiMo ASR as a :class:`TranscriptionProvider` so it can be selected
via ``stt.provider: mimo`` without modifying the built-in STT dispatcher.

Configuration::

    stt:
      provider: mimo
      mimo:
        model: mimo-v2.5-asr
        language: auto          # zh, en, etc.

Requires one of these environment variables:
``MIMO_API_KEY``, ``XIAOMIMIMO_API_KEY``, or ``XIAOMI_API_KEY``.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from agent.transcription_provider import TranscriptionProvider

logger = logging.getLogger(__name__)

DEFAULT_MIMO_STT_MODEL = os.getenv("STT_MIMO_MODEL", "mimo-v2.5-asr")
MIMO_API_BASE_URL = os.getenv("MIMO_API_BASE", "https://api.xiaomimimo.com/v1")

SUPPORTED_MIME_TYPES = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
}

_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
_BACKOFF_SECONDS = (1.0, 2.0, 4.0)
_MAX_RETRIES = 3


def _resolve_api_key() -> Optional[str]:
    """Return the first available MiMo API key from env vars."""
    return (
        os.getenv("MIMO_API_KEY")
        or os.getenv("XIAOMIMIMO_API_KEY")
        or os.getenv("XIAOMI_API_KEY")
    )


def _mime_type_for_path(file_path: str) -> Optional[str]:
    """Map a supported audio extension to its MIME type."""
    suffix = Path(file_path).suffix.lower()
    return SUPPORTED_MIME_TYPES.get(suffix)


def _error_response(error: str, provider: str = "mimo") -> Dict[str, Any]:
    return {
        "success": False,
        "transcript": "",
        "error": error,
        "provider": provider,
    }


class MiMoAsrProvider(TranscriptionProvider):
    """Transcription provider for Xiaomi MiMo ASR.

    MiMo ASR uses an OpenAI-compatible ``/v1/chat/completions`` endpoint
    but accepts audio as a base64 data URL inside the message content.
    """

    @property
    def name(self) -> str:
        return "mimo"

    @property
    def display_name(self) -> str:
        return "MiMo ASR"

    def is_available(self) -> bool:
        return _resolve_api_key() is not None

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": DEFAULT_MIMO_STT_MODEL,
                "display": "MiMo ASR v2.5",
                "languages": ["auto", "zh", "en"],
            }
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MIMO_STT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "MiMo ASR",
            "badge": "paid",
            "tag": "Xiaomi MiMo ASR — /v1/chat/completions with base64 audio",
            "env_vars": [
                {
                    "key": "MIMO_API_KEY",
                    "prompt": "MiMo API key",
                    "url": "https://www.xiaomimimo.com/",
                }
            ],
        }

    def transcribe(
        self,
        file_path: str,
        *,
        model: Optional[str] = None,
        language: Optional[str] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        api_key = _resolve_api_key()
        if not api_key:
            return _error_response(
                "MIMO_API_KEY / XIAOMIMIMO_API_KEY / XIAOMI_API_KEY not set"
            )

        mime_type = _mime_type_for_path(file_path)
        if mime_type is None:
            suffix = Path(file_path).suffix.lower()
            return _error_response(
                f"Unsupported audio format '{suffix}' for MiMo ASR. "
                f"MiMo ASR only supports .wav and .mp3 audio files."
            )

        try:
            data = Path(file_path).read_bytes()
        except OSError as exc:
            return _error_response(f"Cannot read audio file: {exc}")

        audio_b64 = base64.b64encode(data).decode("ascii")
        model_name = model or DEFAULT_MIMO_STT_MODEL
        language_hint = language or "auto"

        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": f"data:{mime_type};base64,{audio_b64}",
                            },
                        }
                    ],
                }
            ],
            "asr_options": {"language": language_hint},
        }

        body = json.dumps(payload).encode("utf-8")
        api_url = f"{MIMO_API_BASE_URL}/chat/completions"
        last_error = ""

        for attempt in range(_MAX_RETRIES + 1):
            req = Request(
                api_url,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "api-key": api_key,
                },
                method="POST",
            )
            try:
                with urlopen(req, timeout=60) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
                transcript_text = result["choices"][0]["message"]["content"].strip()
                logger.info(
                    "Transcribed %s via MiMo ASR (%s, %d chars)",
                    Path(file_path).name,
                    model_name,
                    len(transcript_text),
                )
                return {
                    "success": True,
                    "transcript": transcript_text,
                    "provider": "mimo",
                }
            except HTTPError as exc:
                status = exc.code
                last_error = f"HTTP {status}"
                if status in _RETRYABLE_STATUS and attempt < _MAX_RETRIES:
                    logger.warning(
                        "MiMo ASR transient HTTP %d (attempt %d/%d)",
                        status,
                        attempt + 1,
                        _MAX_RETRIES + 1,
                    )
                    time.sleep(_BACKOFF_SECONDS[attempt])
                    continue
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", errors="replace")[:300]
                except Exception:
                    pass
                return _error_response(f"MiMo ASR error: HTTP {status} {detail}")
            except Exception as exc:
                last_error = str(exc)
                if attempt < _MAX_RETRIES:
                    logger.warning(
                        "MiMo ASR transient error (attempt %d/%d): %s",
                        attempt + 1,
                        _MAX_RETRIES + 1,
                        exc,
                    )
                    time.sleep(_BACKOFF_SECONDS[attempt])
                    continue
                return _error_response(f"MiMo ASR failed after {_MAX_RETRIES + 1} attempts: {last_error}")

        return _error_response(f"MiMo ASR failed: {last_error}")


def register(ctx) -> None:
    """Plugin entry point — wire ``MiMoAsrProvider`` into the registry."""
    ctx.register_transcription_provider(MiMoAsrProvider())
