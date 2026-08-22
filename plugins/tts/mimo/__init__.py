"""Xiaomi MiMo-V2.5 TTS backend (issue #46257).

MiMo-V2.5-TTS exposes speech synthesis through the OpenAI-compatible
``/v1/chat/completions`` endpoint with an ``audio`` output parameter —
NOT the standard ``/v1/audio/speech`` route — so it cannot be served by
the built-in OpenAI TTS provider with a swapped ``base_url``. This
plugin implements the dedicated request/response shape as a
:class:`~agent.tts_provider.TTSProvider`:

- request: ``messages=[{user: style instruction (optional)},
  {assistant: text to synthesize}]`` + ``audio={"format": "wav",
  "voice": ...}``
- response: ``choices[0].message.audio.data`` = base64-encoded WAV

Style control is a MiMo strength: the optional ``user`` message accepts
a natural-language delivery instruction (emotion, pace, role-play), and
inline audio tags such as ``(笑声)`` / ``[开心]`` work inside the
synthesized text itself. See the plugin README for details.

Credentials / endpoint resolution (first hit wins):

1. ``XIAOMI_API_KEY`` (env / secret scope) — official docs name
2. ``MIMO_API_KEY`` — accepted as a fallback alias
3. ``base_url``: ``XIAOMI_BASE_URL`` > ``MIMO_BASE_URL`` >
   ``tts.mimo.base_url`` in config.yaml > the global endpoint

Output formats: MiMo natively returns WAV. ``format="wav"`` writes the
decoded bytes directly; other formats (mp3/ogg/opus/flac, the dispatcher
default is mp3) are converted with ``ffmpeg``, which Hermes voice
delivery paths already depend on.

Limitations (documented in README): preset-voice model only
(``mimo-v2.5-tts``); no voicedesign/voiceclone; no streaming (the
dispatcher falls back to ``synthesize`` + whole-file read); the numeric
``speed`` parameter is ignored — MiMo controls pace via style tags.
"""

from __future__ import annotations

import base64
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.secret_scope import get_secret
from agent.tts_provider import TTSProvider

logger = logging.getLogger(__name__)


DEFAULT_BASE_URL = "https://api.xiaomimimo.com/v1"
DEFAULT_MODEL = "mimo-v2.5-tts"
# Cluster-dependent default voice on the MiMo side ("冰糖" on China
# clusters, "Mia" elsewhere). Passing the explicit meta id keeps the
# provider deterministic across endpoints.
DEFAULT_VOICE = "mimo_default"
DEFAULT_TIMEOUT_SECONDS = 60.0
# Single retry on rate-limit / transient connection errors, matching the
# transient-retry behaviour added for other TTS backends (#75441).
_RETRY_DELAY_SECONDS = 1.0

# Preset voices for mimo-v2.5-tts (official docs, 2026-08). Voice IDs
# are literal — Chinese voice ids are Chinese characters.
_VOICES: List[Dict[str, Any]] = [
    {"id": "mimo_default", "display": "MiMo default (cluster-dependent)",
     "language": "zh-CN/en-US", "gender": ""},
    {"id": "冰糖", "display": "冰糖 — bright female (Chinese)",
     "language": "zh-CN", "gender": "female"},
    {"id": "茉莉", "display": "茉莉 — gentle female (Chinese)",
     "language": "zh-CN", "gender": "female"},
    {"id": "苏打", "display": "苏打 — clear male (Chinese)",
     "language": "zh-CN", "gender": "male"},
    {"id": "白桦", "display": "白桦 — deep male (Chinese)",
     "language": "zh-CN", "gender": "male"},
    {"id": "Mia", "display": "Mia — natural female (English)",
     "language": "en-US", "gender": "female"},
    {"id": "Chloe", "display": "Chloe — warm female (English)",
     "language": "en-US", "gender": "female"},
    {"id": "Milo", "display": "Milo — relaxed male (English)",
     "language": "en-US", "gender": "male"},
    {"id": "Dean", "display": "Dean — steady male (English)",
     "language": "en-US", "gender": "male"},
]


class MiMoTTSError(RuntimeError):
    """Raised for any MiMo TTS failure; the dispatcher converts it into
    the standard ``{success: False, error: ...}`` envelope."""


def _load_mimo_tts_config() -> Dict[str, Any]:
    """Read ``tts.mimo`` from config.yaml (best-effort)."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("tts") if isinstance(cfg, dict) else None
        mimo = section.get("mimo") if isinstance(section, dict) else None
        return mimo if isinstance(mimo, dict) else {}
    except Exception as exc:  # noqa: BLE001 — config failure is non-fatal
        logger.debug("Could not load tts.mimo config: %s", exc)
        return {}


def _resolve_api_key() -> str:
    """Return the MiMo API key (XIAOMI_API_KEY, MIMO_API_KEY fallback)."""
    key = (get_secret("XIAOMI_API_KEY", "") or "").strip()
    if not key:
        key = (get_secret("MIMO_API_KEY", "") or "").strip()
    return key


def _resolve_base_url(cfg: Dict[str, Any]) -> str:
    for env_key in ("XIAOMI_BASE_URL", "MIMO_BASE_URL"):
        value = os.environ.get(env_key, "").strip()
        if value:
            return value.rstrip("/")
    cfg_value = cfg.get("base_url") if isinstance(cfg, dict) else None
    if isinstance(cfg_value, str) and cfg_value.strip():
        return cfg_value.strip().rstrip("/")
    return DEFAULT_BASE_URL


def _resolve_style(cfg: Dict[str, Any]) -> str:
    """Natural-language style instruction for the optional user message."""
    env_style = os.environ.get("MIMO_TTS_STYLE", "").strip()
    if env_style:
        return env_style
    cfg_style = cfg.get("style") if isinstance(cfg, dict) else None
    if isinstance(cfg_style, str) and cfg_style.strip():
        return cfg_style.strip()
    return ""


def _resolve_timeout(cfg: Dict[str, Any]) -> float:
    env_timeout = os.environ.get("MIMO_TTS_TIMEOUT", "").strip()
    raw: Any = env_timeout or (cfg.get("timeout") if isinstance(cfg, dict) else None)
    try:
        value = float(raw)  # type: ignore[arg-type]
        if value > 0:
            return value
    except (TypeError, ValueError):
        pass
    return DEFAULT_TIMEOUT_SECONDS


def _extract_wav_bytes(response: Any) -> bytes:
    """Pull base64 WAV out of a chat-completions response or raise."""
    choices = getattr(response, "choices", None)
    if not choices:
        raise MiMoTTSError(
            "MiMo TTS returned no choices. Check the model id and that "
            "your API key has access to mimo-v2.5-tts."
        )
    message = getattr(choices[0], "message", None)
    audio = getattr(message, "audio", None) if message is not None else None
    data = None
    if isinstance(audio, dict):
        data = audio.get("data")
    else:
        data = getattr(audio, "data", None)
    if not data:
        raise MiMoTTSError(
            "MiMo TTS response contained no audio.data. The request may "
            "have been rejected (rate limit, content filter, or model "
            "unavailable) — check the provider logs."
        )
    try:
        return base64.b64decode(data)
    except Exception as exc:  # noqa: BLE001
        raise MiMoTTSError(f"MiMo TTS returned undecodable audio data: {exc}") from exc


def _convert_with_ffmpeg(wav_path: str, output_path: str) -> None:
    """Convert WAV to the requested container format via ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise MiMoTTSError(
            "MiMo TTS natively outputs WAV and ffmpeg was not found to "
            "convert to the requested format. Install ffmpeg or set "
            "tts.output_format: wav."
        )
    result = subprocess.run(
        [ffmpeg, "-y", "-i", wav_path, output_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()[-500:]
        raise MiMoTTSError(f"ffmpeg conversion for MiMo TTS failed: {stderr}")


class MiMoTTSProvider(TTSProvider):
    """Xiaomi MiMo-V2.5 TTS via chat/completions + audio output param."""

    @property
    def name(self) -> str:
        return "mimo"

    @property
    def display_name(self) -> str:
        return "Xiaomi MiMo"

    @property
    def voice_compatible(self) -> bool:
        # WAV output is fine for voice-bubble delivery: the gateway's
        # delivery pipeline converts to Opus via ffmpeg when needed.
        return True

    def is_available(self) -> bool:
        try:
            return bool(_resolve_api_key())
        except Exception:  # noqa: BLE001 — must never raise
            return False

    def list_voices(self) -> List[Dict[str, Any]]:
        return [dict(entry) for entry in _VOICES]

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": DEFAULT_MODEL,
                "display": "MiMo-V2.5-TTS (preset voices)",
                "languages": ["zh", "en"],
            },
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def default_voice(self) -> Optional[str]:
        return DEFAULT_VOICE

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Xiaomi MiMo",
            "badge": "free (limited-time)",
            "tag": "MiMo-V2.5-TTS — expressive Chinese/English voices",
            "env_vars": [
                {
                    "key": "XIAOMI_API_KEY",
                    "prompt": "Xiaomi MiMo API key",
                    "url": "https://platform.xiaomimimo.com/",
                },
            ],
        }

    def synthesize(
        self,
        text: str,
        output_path: str,
        *,
        voice: Optional[str] = None,
        model: Optional[str] = None,
        speed: Optional[float] = None,  # noqa: ARG002 — MiMo uses style tags
        format: str = "mp3",
        **extra: Any,
    ) -> str:
        text = (text or "").strip()
        if not text:
            raise MiMoTTSError("MiMo TTS received empty text.")

        api_key = _resolve_api_key()
        if not api_key:
            raise MiMoTTSError(
                "XIAOMI_API_KEY (or MIMO_API_KEY) is not set. Get a key "
                "at https://platform.xiaomimimo.com/ and add it to your "
                ".env, or run `hermes tools` -> Text-to-Speech -> Xiaomi MiMo."
            )

        cfg = _load_mimo_tts_config()
        base_url = _resolve_base_url(cfg)
        style = _resolve_style(cfg)
        timeout = _resolve_timeout(cfg)
        model_id = (model or DEFAULT_MODEL).strip() or DEFAULT_MODEL
        voice_id = (voice or DEFAULT_VOICE).strip() or DEFAULT_VOICE

        try:
            import openai
        except ImportError as exc:
            raise MiMoTTSError(
                "openai Python package is not installed (pip install openai)."
            ) from exc

        # Target text goes in the assistant message; the optional user
        # message carries a natural-language style instruction (docs:
        # user-role content is never spoken).
        messages: List[Dict[str, str]] = []
        if style:
            messages.append({"role": "user", "content": style})
        messages.append({"role": "assistant", "content": text})

        client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        try:
            response = self._create_with_retry(
                client,
                model=model_id,
                messages=messages,
                audio={"format": "wav", "voice": voice_id},
            )
        except MiMoTTSError:
            raise
        except Exception as exc:
            raise MiMoTTSError(
                f"MiMo TTS request failed ({base_url}): {exc}"
            ) from exc
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()

        wav_bytes = _extract_wav_bytes(response)

        fmt = (format or "mp3").strip().lower()
        if fmt == "wav":
            Path(output_path).write_bytes(wav_bytes)
            return output_path

        # Non-WAV target: write the native WAV to a temp file, then
        # convert with ffmpeg (Hermes voice delivery already depends on it).
        tmp_wav = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(wav_bytes)
                tmp_wav = tmp.name
            _convert_with_ffmpeg(tmp_wav, output_path)
        finally:
            if tmp_wav and os.path.exists(tmp_wav):
                try:
                    os.remove(tmp_wav)
                except OSError:
                    pass
        return output_path

    def _create_with_retry(self, client: Any, **kwargs: Any) -> Any:
        """chat.completions.create with one retry on transient failures."""
        import time

        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            if not _is_transient(exc):
                raise
            logger.warning(
                "MiMo TTS transient failure (%s); retrying once...", exc
            )
            time.sleep(_RETRY_DELAY_SECONDS)
            return client.chat.completions.create(**kwargs)


def _is_transient(exc: BaseException) -> bool:
    """True for rate limits and connection errors worth one retry."""
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    name = type(exc).__name__
    return name in {"RateLimitError", "APIConnectionError", "APITimeoutError"}


def register(ctx) -> None:
    """Plugin entry point — wire MiMoTTSProvider into the TTS registry."""
    ctx.register_tts_provider(MiMoTTSProvider())
