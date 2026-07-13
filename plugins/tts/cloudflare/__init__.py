"""Cloudflare Workers AI Aura TTS backend.

Wraps the Cloudflare Workers AI Text-to-Speech endpoint (Deepgram Aura
models) as a :class:`agent.tts_provider.TTSProvider` implementation.

Cloudflare's free tier for Workers AI includes Deepgram Aura voices with
generous limits, making this a valuable addition alongside the built-in
Edge (free but network-dependent) and paid providers (OpenAI, ElevenLabs).

This is a **plugin** provider — it registers via
``ctx.register_tts_provider()`` and is dispatched by
``tools.tts_tool._dispatch_to_plugin_provider()`` only when
``tts.provider: cloudflare`` is set in config and the name is neither a
built-in nor a command-type provider.  See issue #30398 for the plugin
provider design.

Configuration
-------------
Credentials are read from environment variables:

* ``CLOUDFLARE_API_TOKEN`` — Cloudflare API token with Workers AI access.
* ``CLOUDFLARE_ACCOUNT_ID`` — Cloudflare account ID.

Per-provider options (all optional, with sensible defaults) are read
from the ``tts.cloudflare`` section of ``config.yaml``:

.. code-block:: yaml

   tts:
     provider: cloudflare
     cloudflare:
       model: "@cf/deepgram/aura-2-en"   # default
       voice: "asteria"                  # default
       encoding: "mp3"                   # default
       base_url: "https://api.cloudflare.com/client/v4/accounts"

The dispatcher also forwards ``voice`` / ``model`` / ``format`` from
the top-level ``tts.voice`` / ``tts.model`` / ``tts.output_format``
config keys; those take precedence over the ``tts.cloudflare`` section
when set.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from agent.tts_provider import TTSProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "@cf/deepgram/aura-2-en"
DEFAULT_VOICE = "asteria"
DEFAULT_ENCODING = "mp3"
DEFAULT_BASE_URL = "https://api.cloudflare.com/client/v4/accounts"

# Cloudflare Aura practical sync cap; Workers AI docs do not publish a
# hard per-request limit.  Aligned with Gemini / Edge (5000).
MAX_TEXT_LENGTH = 5000

# Cloudflare Workers AI ``encoding`` values.
_VALID_ENCODINGS = frozenset({"mp3", "linear16", "ulaw"})

# Aura English voices available on @cf/deepgram/aura-2-en.
_AURA_VOICES = [
    "asteria",
    "luna",
    "stella",
    "athena",
    "hera",
    "orion",
    "arcas",
    "perseus",
    "boreas",
    "zeus",
]


def _get_env(name: str) -> str:
    """Return a stripped env var value, or empty string."""
    return str(os.environ.get(name, "")).strip()


def _load_cf_config() -> Dict[str, Any]:
    """Read the ``tts.cloudflare`` config section (best-effort).

    Returns an empty dict when the section is absent or config can't be
    loaded — the caller falls back to defaults.
    """
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly() or {}
        tts = cfg.get("tts")
        if not isinstance(tts, dict):
            return {}
        section = tts.get("cloudflare")
        return section if isinstance(section, dict) else {}
    except Exception:  # noqa: BLE001 — config is best-effort
        return {}


class CloudflareTTSProvider(TTSProvider):
    """Cloudflare Workers AI (Deepgram Aura) TTS backend."""

    @property
    def name(self) -> str:
        return "cloudflare"

    @property
    def display_name(self) -> str:
        return "Cloudflare Workers AI"

    def is_available(self) -> bool:
        """Available when both Cloudflare credentials are set."""
        return bool(_get_env("CLOUDFLARE_API_TOKEN")) and bool(
            _get_env("CLOUDFLARE_ACCOUNT_ID")
        )

    def list_voices(self) -> List[Dict[str, Any]]:
        """Return the Aura voice catalog."""
        return [
            {
                "id": voice,
                "display": f"{voice.capitalize()} — Aura English",
                "language": "en-US",
            }
            for voice in _AURA_VOICES
        ]

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": DEFAULT_MODEL,
                "display": "Deepgram Aura 2 English",
                "languages": ["en"],
                "max_text_length": MAX_TEXT_LENGTH,
            }
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def default_voice(self) -> Optional[str]:
        return DEFAULT_VOICE

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Cloudflare Workers AI",
            "badge": "free",
            "tag": "Deepgram Aura voices — free Workers AI tier",
            "env_vars": [
                {
                    "key": "CLOUDFLARE_API_TOKEN",
                    "prompt": "Cloudflare API token",
                    "url": "https://dash.cloudflare.com/profile/api-tokens",
                },
                {
                    "key": "CLOUDFLARE_ACCOUNT_ID",
                    "prompt": "Cloudflare account ID",
                    "url": "https://dash.cloudflare.com/",
                },
            ],
        }

    @property
    def voice_compatible(self) -> bool:
        # Output is MP3 — the dispatcher runs ffmpeg → Opus conversion for
        # voice-bubble delivery when this is True, matching Edge / MiniMax /
        # xAI built-in providers.
        return True

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def synthesize(
        self,
        text: str,
        output_path: str,
        *,
        voice: Optional[str] = None,
        model: Optional[str] = None,
        speed: Optional[float] = None,
        format: str = "mp3",
        **extra: Any,
    ) -> str:
        """Synthesize *text* via the Cloudflare Workers AI TTS endpoint.

        Writes raw audio bytes to *output_path* and returns the path.
        Raises on failure — the dispatcher converts exceptions to the
        standard ``{success: False, error: ...}`` JSON envelope.
        """
        import requests

        api_token = _get_env("CLOUDFLARE_API_TOKEN")
        account_id = _get_env("CLOUDFLARE_ACCOUNT_ID")
        if not api_token or not account_id:
            raise ValueError(
                "CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID must be set. "
                "Create a Cloudflare API token with Workers AI access."
            )

        # Config resolution: dispatcher kwargs (top-level tts.voice /
        # tts.model / tts.output_format) override the tts.cloudflare
        # section, which overrides the built-in defaults.
        cf_config = _load_cf_config()

        resolved_model = str(model or cf_config.get("model") or DEFAULT_MODEL).strip()
        base_url = (
            str(cf_config.get("base_url") or DEFAULT_BASE_URL).strip().rstrip("/")
        )
        resolved_voice = str(
            voice or cf_config.get("voice") or cf_config.get("speaker") or DEFAULT_VOICE
        ).strip()

        # Map the dispatcher ``format`` to a Cloudflare ``encoding``.
        # Cloudflare supports mp3, linear16, ulaw.  When the requested
        # format is wav we use linear16 (PCM) so the output is correct;
        # anything else falls back to mp3.
        encoding = str(cf_config.get("encoding") or "").strip().lower()
        if not encoding:
            fmt = (format or "mp3").lower().strip()
            if fmt == "wav":
                encoding = "linear16"
            elif fmt in _VALID_ENCODINGS:
                encoding = fmt
            else:
                encoding = DEFAULT_ENCODING

        payload: Dict[str, Any] = {
            "text": text,
            "speaker": resolved_voice,
            "encoding": encoding,
        }
        for key in ("container", "sample_rate", "bit_rate"):
            value = cf_config.get(key)
            if value not in (None, ""):
                payload[key] = value

        response = requests.post(
            f"{base_url}/{account_id}/ai/run/{resolved_model}",
            headers={
                "Authorization": f"Bearer {api_token}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )

        status_code = getattr(response, "status_code", 200)
        if not isinstance(status_code, int):
            status_code = 200
        if status_code != 200:
            detail = ""
            try:
                body = response.json()
                errors = body.get("errors") if isinstance(body, dict) else None
                if errors:
                    first = errors[0] if isinstance(errors, list) else errors
                    detail = (
                        first.get("message", "")
                        if isinstance(first, dict)
                        else str(first)
                    )
                if not detail and isinstance(body, dict):
                    error = body.get("error")
                    detail = error.get("message", "") if isinstance(error, dict) else ""
            except Exception:
                detail = getattr(response, "text", "")[:300]
            raise RuntimeError(
                f"Cloudflare TTS API error (HTTP {status_code}): "
                f"{detail or 'unknown error'}"
            )

        response.raise_for_status()
        audio_bytes = response.content

        if not audio_bytes:
            raise RuntimeError("Cloudflare Aura TTS returned empty audio data")

        with open(output_path, "wb") as f:
            f.write(audio_bytes)

        return output_path


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point — wire CloudflareTTSProvider into the registry."""
    ctx.register_tts_provider(CloudflareTTSProvider())
