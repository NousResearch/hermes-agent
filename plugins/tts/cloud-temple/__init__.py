"""Cloud Temple TTS provider.

Uses the OpenAI-compatible TTS endpoint at api.ai.cloud-temple.com.
Model: mlx-community/Kokoro-82M-bf16 (multilingual, ~26 languages).

API:
  POST https://api.ai.cloud-temple.com/v1/audio/speech
  Body: { model, input, voice, lang_code }

The provider is auto-discovered by the TTS registry. Activate with:
  tts.provider: cloud-temple
in config.yaml, or via `hermes tools`.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import requests

from agent.tts_provider import DEFAULT_OUTPUT_FORMAT, TTSProvider

logger = logging.getLogger(__name__)


def _cloud_temple_api_key() -> str:
    try:
        from hermes_cli.config import get_env_value

        return (get_env_value("CLOUD_TEMPLE_API_KEY") or "").strip()
    except Exception:
        return (os.environ.get("CLOUD_TEMPLE_API_KEY") or "").strip()

# Cloud Temple base URL (production)
BASE_URL = "https://api.ai.cloud-temple.com/v1"

DEFAULT_TTS_MODEL = "mlx-community/Kokoro-82M-bf16"

# Kokoro voices — a subset of well-known OpenAI-compatible voice IDs.
_KOKORO_VOICES = [
    {"id": "alloy", "display": "Alloy — neutral", "language": "multi"},
    {"id": "echo", "display": "Echo — male", "language": "multi"},
    {"id": "fable", "display": "Fable — narrative", "language": "multi"},
    {"id": "onyx", "display": "Onyx — deep male", "language": "multi"},
    {"id": "nova", "display": "Nova — female", "language": "multi"},
    {"id": "shimmer", "display": "Shimmer — soft female", "language": "multi"},
]

# User-facing language codes accepted in config. The Cloud Temple Kokoro
# endpoint expects Kokoro's one-letter lang_code values on the wire.
_KOKORO_LANG_CODES = [
    "en", "en-gb", "es", "fr", "hi", "it", "ja", "pt-br", "zh",
]

_LANG_CODE_ALIASES = {
    "a": "a",
    "american": "a",
    "en": "a",
    "en-us": "a",
    "english": "a",
    "us": "a",
    "b": "b",
    "british": "b",
    "en-gb": "b",
    "uk": "b",
    "e": "e",
    "es": "e",
    "es-es": "e",
    "spanish": "e",
    "f": "f",
    "fr": "f",
    "fr-fr": "f",
    "french": "f",
    "h": "h",
    "hi": "h",
    "hindi": "h",
    "i": "i",
    "it": "i",
    "it-it": "i",
    "italian": "i",
    "j": "j",
    "ja": "j",
    "ja-jp": "j",
    "japanese": "j",
    "p": "p",
    "pt": "p",
    "pt-br": "p",
    "portuguese": "p",
    "z": "z",
    "zh": "z",
    "zh-cn": "z",
    "chinese": "z",
}

_FRENCH_MARKERS = (
    " le ", " la ", " les ", " des ", " une ", " vous ", " nous ",
    " pour ", " avec ", " dans ", " que ", " est ", " pas ",
)


def _normalize_lang_code(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower().replace("_", "-")
    if not normalized:
        return None
    return _LANG_CODE_ALIASES.get(normalized)


def _infer_lang_code(text: str) -> str:
    lowered = f" {(text or '').strip().lower()} "
    if any(ch in lowered for ch in "àâçéèêëîïôùûüÿœæ"):
        return "f"
    if any(marker in lowered for marker in _FRENCH_MARKERS):
        return "f"
    if any("\u3040" <= ch <= "\u30ff" for ch in lowered):
        return "j"
    if any("\u4e00" <= ch <= "\u9fff" for ch in lowered):
        return "z"
    if any(ch in lowered for ch in "¿¡ñ"):
        return "e"
    return "a"


def _configured_lang_code(cfg: Dict[str, Any]) -> Optional[str]:
    provider_cfg = cfg.get("cloud-temple") if isinstance(cfg.get("cloud-temple"), dict) else {}
    for source in (provider_cfg, cfg):
        if not isinstance(source, dict):
            continue
        lang_code = _normalize_lang_code(source.get("lang_code"))
        if lang_code:
            return lang_code
        language = _normalize_lang_code(source.get("language"))
        if language:
            return language
    return None


def _load_config() -> Dict[str, Any]:
    """Read ``tts`` section from config.yaml."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("tts") if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception as exc:
        logger.debug("Could not load tts config: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class CloudTempleTTSProvider(TTSProvider):
    """Cloud Temple TTS — Kokoro model via OpenAI-compatible API."""

    @property
    def name(self) -> str:
        return "cloud-temple"

    @property
    def display_name(self) -> str:
        return "Cloud Temple"

    def is_available(self) -> bool:
        return bool(_cloud_temple_api_key())

    def list_voices(self) -> List[Dict[str, Any]]:
        return list(_KOKORO_VOICES)

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": DEFAULT_TTS_MODEL,
                "display": "Kokoro 82M",
                "languages": _KOKORO_LANG_CODES,
                "max_text_length": 5000,
            }
        ]

    def default_voice(self) -> Optional[str]:
        return "alloy"

    def default_model(self) -> Optional[str]:
        return DEFAULT_TTS_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Cloud Temple",
            "badge": "free",
            "tag": "Kokoro multilingual TTS via OpenAI-compatible API",
            "env_vars": [
                {
                    "key": "CLOUD_TEMPLE_API_KEY",
                    "prompt": "Cloud Temple API key",
                    "url": "https://api.ai.cloud-temple.com/",
                }
            ],
        }

    @property
    def voice_compatible(self) -> bool:
        return True

    def synthesize(
        self,
        text: str,
        output_path: str,
        *,
        voice: Optional[str] = None,
        model: Optional[str] = None,
        speed: Optional[float] = None,
        format: str = DEFAULT_OUTPUT_FORMAT,
        **extra: Any,
    ) -> str:
        api_key = _cloud_temple_api_key()
        if not api_key:
            raise RuntimeError(
                "CLOUD_TEMPLE_API_KEY not set. Run `hermes setup` or "
                "`hermes tools` → TTS → Cloud Temple to configure."
            )

        text = (text or "").strip()
        if not text:
            raise ValueError("TTS input text is empty")

        voice = voice or self.default_voice()
        model = model or self.default_model()

        lang_code = _normalize_lang_code(extra.get("lang_code"))
        if not lang_code:
            lang_code = _configured_lang_code(_load_config())
        if not lang_code:
            lang_code = _infer_lang_code(text)

        url = f"{BASE_URL}/audio/speech"
        payload = {
            "model": model,
            "input": text,
            "voice": voice,
            "lang_code": lang_code,
        }
        if speed is not None:
            payload["speed"] = speed

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        if resp.status_code != 200:
            detail = resp.text[:300]
            raise RuntimeError(f"Cloud Temple TTS API returned HTTP {resp.status_code}: {detail}")

        # Write the audio bytes to output_path.
        # The API returns raw audio (mp3 by default for Kokoro).
        with open(output_path, "wb") as fh:
            fh.write(resp.content)

        return output_path


# ---------------------------------------------------------------------------
# Auto-registration
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point."""
    ctx.register_tts_provider(CloudTempleTTSProvider())