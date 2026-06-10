"""LiveKit/WebRTC setup helpers for Hermes live voice.

This module deliberately avoids mutating ``~/.hermes/config.yaml``.  The
Telegram voice stack remains unchanged while these helpers prepare the
phone-number-independent pieces of Voice v02:

* preflight checks for LiveKit credentials and SIP readiness;
* JSON payloads for LiveKit SIP trunk and explicit agent dispatch setup;
* optional WebRTC room token generation for browser-call MVP testing;
* redacted realtime-worker readiness checks for LiveKit realtime providers.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
import secrets
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

DEFAULT_AGENT_NAME = "hermes-live-voice"
DEFAULT_ROOM_PREFIX = "hermes-call-"
DEFAULT_ROUTE = "hermes-main"
DEFAULT_HERMES_BRAIN_URL = "http://127.0.0.1:8646/v1/chat/completions"
DEFAULT_HERMES_BRAIN_MODEL = "voice"
DEFAULT_HERMES_BRAIN_TIMEOUT_SECONDS = 8.0
DEFAULT_HERMES_BRAIN_MAX_TOKENS = 450
DEFAULT_PIPELINE_MODE = "realtime"
DEFAULT_MODULAR_STT_PROVIDER = "deepgram"
DEFAULT_MODULAR_TTS_PROVIDER = "cartesia"
DEFAULT_DEEPGRAM_MODEL = "nova-3"
DEFAULT_DEEPGRAM_LANGUAGE = "multi"
DEFAULT_CARTESIA_MODEL = "sonic-2"
DEFAULT_CARTESIA_VOICE = "bf0a246a-8642-498a-9950-80c35e9276b5"
DEFAULT_REALTIME_PROVIDER = "gemini"
DEFAULT_REALTIME_MODEL = "gpt-realtime"
DEFAULT_REALTIME_VOICE = "coral"
DEFAULT_GEMINI_REALTIME_MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"
DEFAULT_GEMINI_REALTIME_VOICE = "Puck"
DEFAULT_XAI_REALTIME_MODEL = "grok-voice-think-fast-1.0"
DEFAULT_XAI_REALTIME_VOICE = "ara"
DEFAULT_REALTIME_VERSION = "v02"
DEFAULT_REALTIME_INSTRUCTIONS = (
    "You are Hermes live voice for Pafi. Keep replies brief, useful, and natural. "
    "Reply in the user's language unless they explicitly ask otherwise. "
    "Do not narrate system status, model internals, or tool execution."
)

_E164_RE = re.compile(r"^\+[1-9]\d{6,14}$")
_SAFE_AGENT_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_SAFE_LIVEKIT_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")
_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]+")
_TRUE_VALUES = {"1", "true", "yes", "on", "enabled"}


@dataclass(frozen=True)
class LiveKitVoiceConfig:
    """Runtime configuration read from environment variables."""

    livekit_url: str = ""
    livekit_api_key: str = ""
    livekit_api_secret: str = ""
    agent_name: str = DEFAULT_AGENT_NAME
    room_prefix: str = DEFAULT_ROOM_PREFIX
    phone_number: str = ""
    sip_provider: str = ""
    hermes_brain_url: str = DEFAULT_HERMES_BRAIN_URL
    hermes_brain_api_key: str = ""
    hermes_brain_model: str = DEFAULT_HERMES_BRAIN_MODEL
    hermes_brain_timeout_seconds: float = DEFAULT_HERMES_BRAIN_TIMEOUT_SECONDS
    hermes_brain_max_tokens: int = DEFAULT_HERMES_BRAIN_MAX_TOKENS
    hermes_brain_allow_remote: bool = False
    hermes_brain_allowed_hosts: tuple[str, ...] = ()
    realtime_enabled: bool = False
    pipeline_mode: str = DEFAULT_PIPELINE_MODE
    realtime_provider: str = DEFAULT_REALTIME_PROVIDER
    realtime_model: str = DEFAULT_REALTIME_MODEL
    realtime_voice: str = DEFAULT_REALTIME_VOICE
    realtime_instructions: str = DEFAULT_REALTIME_INSTRUCTIONS
    stt_provider: str = DEFAULT_MODULAR_STT_PROVIDER
    tts_provider: str = DEFAULT_MODULAR_TTS_PROVIDER
    deepgram_model: str = DEFAULT_DEEPGRAM_MODEL
    deepgram_language: str = DEFAULT_DEEPGRAM_LANGUAGE
    cartesia_model: str = DEFAULT_CARTESIA_MODEL
    cartesia_voice: str = DEFAULT_CARTESIA_VOICE
    openai_api_key: str = ""
    google_api_key: str = ""
    xai_api_key: str = ""
    groq_api_key: str = ""
    deepgram_api_key: str = ""
    cartesia_api_key: str = ""
    elevenlabs_api_key: str = ""

    @property
    def has_credentials(self) -> bool:
        return bool(
            self.livekit_url and self.livekit_api_key and self.livekit_api_secret
        )

    @property
    def has_phone_number(self) -> bool:
        return bool(self.phone_number and _E164_RE.match(self.phone_number))

    @property
    def uses_modular_pipeline(self) -> bool:
        return self.pipeline_mode == "modular"

    @property
    def has_realtime_credentials(self) -> bool:
        if self.realtime_provider == "openai":
            return bool(self.openai_api_key)
        if self.realtime_provider == "gemini":
            return bool(self.google_api_key)
        if self.realtime_provider == "xai":
            return bool(self.xai_api_key)
        return False

    @property
    def has_modular_stt_credentials(self) -> bool:
        if self.stt_provider == "deepgram":
            return bool(self.deepgram_api_key)
        if self.stt_provider == "groq":
            return bool(self.groq_api_key)
        if self.stt_provider == "openai":
            return bool(self.openai_api_key)
        return False

    @property
    def has_modular_tts_credentials(self) -> bool:
        if self.tts_provider == "cartesia":
            return bool(self.cartesia_api_key)
        if self.tts_provider == "elevenlabs":
            return bool(self.elevenlabs_api_key)
        return False

    @property
    def has_modular_credentials(self) -> bool:
        return self.has_modular_stt_credentials and self.has_modular_tts_credentials

    @property
    def has_brain_credentials(self) -> bool:
        return bool(self.hermes_brain_url and self.hermes_brain_api_key)

    def public_dict(self) -> dict[str, str]:
        """Return a redacted view safe for Telegram/status messages."""
        return {
            "livekit_url": self.livekit_url,
            "livekit_api_key": "set" if self.livekit_api_key else "missing",
            "livekit_api_secret": "set" if self.livekit_api_secret else "missing",
            "agent_name": self.agent_name,
            "room_prefix": self.room_prefix,
            "phone_number": self.phone_number if self.phone_number else "missing",
            "sip_provider": self.sip_provider if self.sip_provider else "missing",
            "hermes_brain_url": self.hermes_brain_url,
            "hermes_brain_api_key": "set"
            if self.hermes_brain_api_key
            else "missing",
            "hermes_brain_model": self.hermes_brain_model,
            "hermes_brain_timeout_seconds": str(self.hermes_brain_timeout_seconds),
            "hermes_brain_max_tokens": str(self.hermes_brain_max_tokens),
            "hermes_brain_allow_remote": "true"
            if self.hermes_brain_allow_remote
            else "false",
            "hermes_brain_allowed_hosts": ",".join(self.hermes_brain_allowed_hosts)
            if self.hermes_brain_allowed_hosts
            else "none",
            "realtime_enabled": "true" if self.realtime_enabled else "false",
            "pipeline_mode": self.pipeline_mode,
            "realtime_provider": self.realtime_provider,
            "realtime_model": self.realtime_model,
            "realtime_voice": self.realtime_voice,
            "stt_provider": self.stt_provider,
            "tts_provider": self.tts_provider,
            "deepgram_model": self.deepgram_model,
            "deepgram_language": self.deepgram_language,
            "cartesia_model": self.cartesia_model,
            "cartesia_voice": self.cartesia_voice,
            "openai_api_key": "set" if self.openai_api_key else "missing",
            "google_api_key": "set" if self.google_api_key else "missing",
            "xai_api_key": "set" if self.xai_api_key else "missing",
            "groq_api_key": "set" if self.groq_api_key else "missing",
            "deepgram_api_key": "set" if self.deepgram_api_key else "missing",
            "cartesia_api_key": "set" if self.cartesia_api_key else "missing",
            "elevenlabs_api_key": "set" if self.elevenlabs_api_key else "missing",
        }


def _env_get(env: Mapping[str, str], name: str, default: str = "") -> str:
    return str(env.get(name, default) or "").strip()


def _env_bool(env: Mapping[str, str], name: str, default: bool = False) -> bool:
    value = _env_get(env, name)
    return default if not value else value.lower() in _TRUE_VALUES


def _env_float(
    env: Mapping[str, str],
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    value = _env_get(env, name)
    if not value:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return min(max(parsed, minimum), maximum)


def _env_int(
    env: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    value = _env_get(env, name)
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return min(max(parsed, minimum), maximum)


def _env_csv(env: Mapping[str, str], name: str) -> tuple[str, ...]:
    value = _env_get(env, name)
    if not value:
        return ()
    return tuple(
        item.strip().strip("[]").lower()
        for item in value.split(",")
        if item.strip()
    )


def load_livekit_config(env: Mapping[str, str] | None = None) -> LiveKitVoiceConfig:
    """Load LiveKit voice settings from *env* or ``os.environ``."""
    source = os.environ if env is None else env
    return LiveKitVoiceConfig(
        livekit_url=_env_get(source, "LIVEKIT_URL"),
        livekit_api_key=_env_get(source, "LIVEKIT_API_KEY"),
        livekit_api_secret=_env_get(source, "LIVEKIT_API_SECRET"),
        agent_name=_env_get(source, "HERMES_LIVEKIT_AGENT_NAME", DEFAULT_AGENT_NAME),
        room_prefix=_normalize_room_prefix(
            _env_get(source, "HERMES_LIVEKIT_ROOM_PREFIX", DEFAULT_ROOM_PREFIX)
        ),
        phone_number=_env_get(source, "HERMES_LIVEKIT_PHONE_NUMBER"),
        sip_provider=_env_get(source, "HERMES_LIVEKIT_SIP_PROVIDER"),
        hermes_brain_url=_env_get(
            source, "HERMES_LIVEKIT_HERMES_URL", DEFAULT_HERMES_BRAIN_URL
        ),
        hermes_brain_api_key=_env_get(
            source,
            "HERMES_LIVEKIT_HERMES_API_KEY",
            _env_get(source, "HERMES_VOICE_API_KEY", _env_get(source, "API_SERVER_KEY")),
        ),
        hermes_brain_model=_env_get(
            source, "HERMES_LIVEKIT_HERMES_MODEL", DEFAULT_HERMES_BRAIN_MODEL
        ),
        hermes_brain_timeout_seconds=_env_float(
            source,
            "HERMES_LIVEKIT_HERMES_TIMEOUT_SECONDS",
            DEFAULT_HERMES_BRAIN_TIMEOUT_SECONDS,
            minimum=1.0,
            maximum=15.0,
        ),
        hermes_brain_max_tokens=_env_int(
            source,
            "HERMES_LIVEKIT_HERMES_MAX_TOKENS",
            DEFAULT_HERMES_BRAIN_MAX_TOKENS,
            minimum=64,
            maximum=800,
        ),
        hermes_brain_allow_remote=_env_bool(
            source, "HERMES_LIVEKIT_HERMES_ALLOW_REMOTE", False
        ),
        hermes_brain_allowed_hosts=_env_csv(
            source, "HERMES_LIVEKIT_HERMES_ALLOWED_HOSTS"
        ),
        realtime_enabled=_env_bool(source, "HERMES_LIVEKIT_REALTIME_ENABLED", False),
        pipeline_mode=_env_get(
            source, "HERMES_LIVEKIT_PIPELINE_MODE", DEFAULT_PIPELINE_MODE
        ).lower(),
        realtime_provider=_env_get(
            source, "HERMES_LIVEKIT_REALTIME_PROVIDER", DEFAULT_REALTIME_PROVIDER
        ).lower(),
        realtime_model=_load_realtime_model(source),
        realtime_voice=_load_realtime_voice(source),
        realtime_instructions=_env_get(
            source,
            "HERMES_LIVEKIT_REALTIME_INSTRUCTIONS",
            DEFAULT_REALTIME_INSTRUCTIONS,
        ),
        stt_provider=_env_get(
            source, "HERMES_LIVEKIT_STT_PROVIDER", DEFAULT_MODULAR_STT_PROVIDER
        ).lower(),
        tts_provider=_env_get(
            source, "HERMES_LIVEKIT_TTS_PROVIDER", DEFAULT_MODULAR_TTS_PROVIDER
        ).lower(),
        deepgram_model=_env_get(
            source, "HERMES_LIVEKIT_DEEPGRAM_MODEL", DEFAULT_DEEPGRAM_MODEL
        ),
        deepgram_language=_env_get(
            source, "HERMES_LIVEKIT_DEEPGRAM_LANGUAGE", DEFAULT_DEEPGRAM_LANGUAGE
        ),
        cartesia_model=_env_get(
            source, "HERMES_LIVEKIT_CARTESIA_MODEL", DEFAULT_CARTESIA_MODEL
        ),
        cartesia_voice=_env_get(
            source, "HERMES_LIVEKIT_CARTESIA_VOICE", DEFAULT_CARTESIA_VOICE
        ),
        openai_api_key=_env_get(source, "OPENAI_API_KEY"),
        google_api_key=_env_get(source, "GOOGLE_API_KEY")
        or _env_get(source, "GEMINI_API_KEY"),
        xai_api_key=_env_get(source, "XAI_API_KEY"),
        groq_api_key=_env_get(source, "GROQ_API_KEY"),
        deepgram_api_key=_env_get(source, "DEEPGRAM_API_KEY"),
        cartesia_api_key=_env_get(source, "CARTESIA_API_KEY"),
        elevenlabs_api_key=_env_get(source, "ELEVENLABS_API_KEY"),
    )


def _load_realtime_model(source: Mapping[str, str]) -> str:
    provider = _env_get(
        source, "HERMES_LIVEKIT_REALTIME_PROVIDER", DEFAULT_REALTIME_PROVIDER
    ).lower()
    if provider == "gemini":
        return _env_get(
            source,
            "HERMES_GEMINI_REALTIME_MODEL",
            DEFAULT_GEMINI_REALTIME_MODEL,
        )
    if provider == "xai":
        return _env_get(
            source,
            "HERMES_XAI_REALTIME_MODEL",
            DEFAULT_XAI_REALTIME_MODEL,
        )
    return _env_get(source, "HERMES_OPENAI_REALTIME_MODEL", DEFAULT_REALTIME_MODEL)


def _load_realtime_voice(source: Mapping[str, str]) -> str:
    provider = _env_get(
        source, "HERMES_LIVEKIT_REALTIME_PROVIDER", DEFAULT_REALTIME_PROVIDER
    ).lower()
    if provider == "gemini":
        return _env_get(
            source,
            "HERMES_GEMINI_REALTIME_VOICE",
            DEFAULT_GEMINI_REALTIME_VOICE,
        )
    if provider == "xai":
        return _env_get(
            source,
            "HERMES_XAI_REALTIME_VOICE",
            DEFAULT_XAI_REALTIME_VOICE,
        )
    return _env_get(source, "HERMES_OPENAI_REALTIME_VOICE", DEFAULT_REALTIME_VOICE)


def build_livekit_preflight(
    env: Mapping[str, str] | None = None,
    *,
    require_phone_number: bool = False,
    include_realtime: bool = False,
) -> dict[str, Any]:
    """Build a redacted readiness report for the LiveKit voice path."""
    cfg = load_livekit_config(env)
    issues: list[dict[str, str]] = []

    if not cfg.livekit_url:
        issues.append({
            "severity": "error",
            "code": "missing_livekit_url",
            "message": "Set LIVEKIT_URL before running the WebRTC MVP.",
        })
    elif not _is_valid_livekit_url(cfg.livekit_url):
        issues.append({
            "severity": "error",
            "code": "invalid_livekit_url",
            "message": "LIVEKIT_URL must use wss://, except ws:// is allowed for loopback development only.",
        })
    if not cfg.livekit_api_key:
        issues.append({
            "severity": "error",
            "code": "missing_livekit_api_key",
            "message": "Set LIVEKIT_API_KEY before creating rooms or dispatches.",
        })
    if not cfg.livekit_api_secret:
        issues.append({
            "severity": "error",
            "code": "missing_livekit_api_secret",
            "message": "Set LIVEKIT_API_SECRET before creating rooms or dispatches.",
        })
    if not cfg.agent_name:
        issues.append({
            "severity": "error",
            "code": "missing_agent_name",
            "message": "Set HERMES_LIVEKIT_AGENT_NAME or use the default.",
        })
    else:
        try:
            validate_agent_name(cfg.agent_name)
        except ValueError:
            issues.append({
                "severity": "error",
                "code": "invalid_agent_name",
                "message": "HERMES_LIVEKIT_AGENT_NAME must contain only letters, digits, _ or -.",
            })

    phone_severity = "error" if require_phone_number else "warn"
    if not cfg.phone_number:
        issues.append({
            "severity": phone_severity,
            "code": "missing_phone_number",
            "message": "Set HERMES_LIVEKIT_PHONE_NUMBER after buying the number.",
        })
    elif not _E164_RE.match(cfg.phone_number):
        issues.append({
            "severity": "error",
            "code": "invalid_phone_number",
            "message": "HERMES_LIVEKIT_PHONE_NUMBER must be in +E.164 format.",
        })

    if include_realtime:
        if cfg.pipeline_mode not in {"realtime", "modular"}:
            issues.append({
                "severity": "error",
                "code": "unsupported_pipeline_mode",
                "message": "HERMES_LIVEKIT_PIPELINE_MODE must be realtime or modular.",
            })
        if cfg.realtime_provider not in {"openai", "gemini", "xai"}:
            issues.append({
                "severity": "error",
                "code": "unsupported_realtime_provider",
                "message": "Voice v02 supports HERMES_LIVEKIT_REALTIME_PROVIDER=openai, gemini, or xai.",
            })
        if cfg.realtime_provider == "openai" and not cfg.openai_api_key:
            issues.append({
                "severity": "error",
                "code": "missing_openai_api_key",
                "message": "Set OPENAI_API_KEY before starting the OpenAI Realtime worker.",
            })
        if cfg.realtime_provider == "gemini" and not cfg.google_api_key:
            issues.append({
                "severity": "error",
                "code": "missing_google_api_key",
                "message": "Set GOOGLE_API_KEY or GEMINI_API_KEY before starting the Gemini Live worker.",
            })
        if cfg.realtime_provider == "xai" and not cfg.xai_api_key:
            issues.append({
                "severity": "error",
                "code": "missing_xai_api_key",
                "message": "Set XAI_API_KEY before starting the Grok Voice worker.",
            })
        if cfg.pipeline_mode == "modular":
            if cfg.stt_provider not in {"deepgram", "groq", "openai"}:
                issues.append({
                    "severity": "error",
                    "code": "unsupported_modular_stt_provider",
                    "message": "HERMES_LIVEKIT_STT_PROVIDER must be deepgram, groq, or openai.",
                })
            if cfg.tts_provider not in {"cartesia", "elevenlabs"}:
                issues.append({
                    "severity": "error",
                    "code": "unsupported_modular_tts_provider",
                    "message": "HERMES_LIVEKIT_TTS_PROVIDER must be cartesia or elevenlabs.",
                })
            if cfg.stt_provider == "deepgram" and not cfg.deepgram_api_key:
                issues.append({"severity": "error", "code": "missing_deepgram_api_key", "message": "Set DEEPGRAM_API_KEY for modular Deepgram STT."})
            if cfg.stt_provider == "groq" and not cfg.groq_api_key:
                issues.append({"severity": "error", "code": "missing_groq_api_key", "message": "Set GROQ_API_KEY for modular Groq STT."})
            if cfg.stt_provider == "openai" and not cfg.openai_api_key:
                issues.append({"severity": "error", "code": "missing_openai_api_key", "message": "Set OPENAI_API_KEY for modular OpenAI STT."})
            if cfg.tts_provider == "cartesia" and not cfg.cartesia_api_key:
                issues.append({"severity": "error", "code": "missing_cartesia_api_key", "message": "Set CARTESIA_API_KEY for modular Cartesia TTS."})
            if cfg.tts_provider == "elevenlabs" and not cfg.elevenlabs_api_key:
                issues.append({"severity": "error", "code": "missing_elevenlabs_api_key", "message": "Set ELEVENLABS_API_KEY for modular ElevenLabs TTS."})
        if not cfg.realtime_enabled:
            issues.append({
                "severity": "warn",
                "code": "realtime_worker_disabled",
                "message": "Set HERMES_LIVEKIT_REALTIME_ENABLED=true when ready to run the worker.",
            })

    web_ready = not any(
        issue["severity"] == "error"
        and issue["code"]
        in {
            "missing_livekit_url",
            "invalid_livekit_url",
            "missing_livekit_api_key",
            "missing_livekit_api_secret",
            "missing_agent_name",
        }
        for issue in issues
    )
    realtime_ready = web_ready and (cfg.has_modular_credentials if cfg.uses_modular_pipeline else cfg.has_realtime_credentials)
    sip_ready = web_ready and cfg.has_phone_number
    ok = not any(issue["severity"] == "error" for issue in issues)

    return {
        "ok": ok,
        "ready": {
            "web_mvp": web_ready,
            "realtime_agent": realtime_ready,
            "sip_phone": sip_ready,
        },
        "config": cfg.public_dict(),
        "worker": build_realtime_worker_status(config=cfg),
        "issues": issues,
        "next": _next_steps(
            web_ready=web_ready,
            realtime_ready=realtime_ready,
            realtime_enabled=cfg.realtime_enabled,
            sip_ready=sip_ready,
        ),
    }


def build_realtime_worker_status(
    *, config: LiveKitVoiceConfig | None = None
) -> dict[str, Any]:
    """Return operator-safe worker launch information."""
    cfg = config or load_livekit_config()
    return {
        "agent_name": cfg.agent_name,
        "mode": "manual",
        "enabled": cfg.realtime_enabled,
        "pipeline_mode": cfg.pipeline_mode,
        "provider": cfg.realtime_provider,
        "stt_provider": cfg.stt_provider,
        "tts_provider": cfg.tts_provider,
        "model": cfg.realtime_model,
        "voice": cfg.realtime_voice,
        "deepgram_model": cfg.deepgram_model,
        "deepgram_language": cfg.deepgram_language,
        "cartesia_model": cfg.cartesia_model,
        "cartesia_voice": cfg.cartesia_voice,
        "version": DEFAULT_REALTIME_VERSION,
        "run": ".venv/bin/python -m gateway.livekit_realtime_agent dev",
        "start": ".venv/bin/python -m gateway.livekit_realtime_agent start",
    }


def build_realtime_room_metadata(
    *,
    mode: str = "webrtc",
    extra: Mapping[str, Any] | None = None,
    config: LiveKitVoiceConfig | None = None,
) -> dict[str, Any]:
    """Build dispatch metadata shared by WebRTC and future SIP rooms."""
    cfg = config or load_livekit_config()
    data: dict[str, Any] = {
        "mode": mode,
        "route": DEFAULT_ROUTE,
        "voice_version": DEFAULT_REALTIME_VERSION,
        "pipeline_mode": cfg.pipeline_mode,
        "realtime_provider": cfg.realtime_provider,
        "stt_provider": cfg.stt_provider if cfg.uses_modular_pipeline else "none",
        "tts_provider": cfg.tts_provider if cfg.uses_modular_pipeline else "none",
    }
    if extra:
        data.update(extra)
    return data


def _next_steps(
    *, web_ready: bool, realtime_ready: bool, realtime_enabled: bool, sip_ready: bool
) -> list[str]:
    steps: list[str] = []
    if not web_ready:
        steps.append("Create a LiveKit project and set LIVEKIT_URL/API_KEY/API_SECRET.")
    elif not realtime_ready:
        steps.append("Set realtime provider credentials, then run preflight.")
    elif not realtime_enabled:
        steps.append(
            "Enable the manual worker with HERMES_LIVEKIT_REALTIME_ENABLED=true."
        )
    else:
        steps.append("Start the realtime worker and run a WebRTC room call test.")
    if not sip_ready:
        steps.append(
            "Buy a phone number, then set HERMES_LIVEKIT_PHONE_NUMBER for SIP."
        )
    else:
        steps.append("Create inbound trunk and dispatch rule for the phone number.")
    return steps


def _normalize_room_prefix(value: str) -> str:
    clean = _safe_slug(value or DEFAULT_ROOM_PREFIX)
    return clean if clean.endswith("-") else f"{clean}-"


def _safe_slug(value: str) -> str:
    clean = _SAFE_NAME_RE.sub("-", value.strip()).strip("-_").lower()
    clean = re.sub(r"-{2,}", "-", clean)
    return clean or "hermes"


def _is_loopback_host(host: str | None) -> bool:
    if not host:
        return False
    clean = host.strip("[]").lower()
    if clean == "localhost":
        return True
    try:
        return ipaddress.ip_address(clean).is_loopback
    except ValueError:
        return False


def _is_valid_livekit_url(value: str) -> bool:
    parsed = urlparse(value)
    if parsed.scheme == "wss" and parsed.hostname:
        return True
    return parsed.scheme == "ws" and _is_loopback_host(parsed.hostname)


def validate_agent_name(agent_name: str) -> str:
    """Return a stripped LiveKit agent name or raise for unsafe values."""
    clean_agent_name = agent_name.strip()
    if not _SAFE_AGENT_NAME_RE.match(clean_agent_name):
        raise ValueError("agent_name must contain only letters, digits, _ or -")
    return clean_agent_name


def build_room_name(
    room_prefix: str = DEFAULT_ROOM_PREFIX, seed: str = "", *, suffix: str | None = None
) -> str:
    """Return a LiveKit-safe room name for one live-call session."""
    max_len = 96
    prefix = _normalize_room_prefix(room_prefix)
    stem = _safe_slug(seed) if seed else "session"
    tail = _safe_slug(suffix) if suffix else secrets.token_hex(4)
    tail_segment = f"-{tail}"
    max_prefix_len = max(1, max_len - len(tail_segment) - 1)
    if len(prefix) > max_prefix_len:
        prefix = prefix[: max_prefix_len - 1].rstrip("-_")
        prefix = f"{prefix}-" if prefix else "h-"
    stem_len = max_len - len(prefix) - len(tail_segment)
    clean_stem = stem[:stem_len].strip("-_") if stem_len > 0 else ""
    clean_stem = clean_stem or "s"
    return f"{prefix}{clean_stem}{tail_segment}"[:max_len]


def _json_metadata(metadata: Mapping[str, Any] | None = None) -> str:
    data = {"route": DEFAULT_ROUTE}
    if metadata:
        data.update(metadata)
    return json.dumps(data, separators=(",", ":"), sort_keys=True)


def build_dispatch_rule_payload(
    *,
    agent_name: str = DEFAULT_AGENT_NAME,
    room_prefix: str = DEFAULT_ROOM_PREFIX,
    metadata: Mapping[str, Any] | None = None,
    trunk_ids: Sequence[str] | None = None,
    name: str = "Hermes live voice dispatch",
) -> dict[str, Any]:
    """Build LiveKit SIP dispatch JSON using explicit agent dispatch."""
    clean_agent_name = agent_name.strip()
    if not _SAFE_AGENT_NAME_RE.match(clean_agent_name):
        raise ValueError("agent_name must contain only letters, digits, _ or -")
    payload: dict[str, Any] = {
        "name": name,
        "rule": {
            "dispatchRuleIndividual": {
                "roomPrefix": _normalize_room_prefix(room_prefix)
            }
        },
        "roomConfig": {
            "agents": [
                {"agentName": clean_agent_name, "metadata": _json_metadata(metadata)}
            ]
        },
    }
    clean_trunk_ids = [item.strip() for item in (trunk_ids or []) if item.strip()]
    for trunk_id in clean_trunk_ids:
        if not _SAFE_LIVEKIT_ID_RE.match(trunk_id):
            raise ValueError("trunk_ids must contain only letters, digits, _ or -")
    if clean_trunk_ids:
        payload["trunkIds"] = clean_trunk_ids
    return payload


def build_inbound_trunk_payload(
    phone_number: str,
    *,
    allowed_numbers: Sequence[str] | None = None,
    name: str = "Hermes live voice inbound trunk",
    krisp_enabled: bool = True,
) -> dict[str, Any]:
    """Build LiveKit inbound trunk JSON for a purchased SIP number."""
    phone = phone_number.strip()
    if not _E164_RE.match(phone):
        raise ValueError("phone_number must be in +E.164 format")
    trunk: dict[str, Any] = {
        "name": name,
        "numbers": [phone],
        "krispEnabled": bool(krisp_enabled),
    }
    allowed = [item.strip() for item in (allowed_numbers or []) if item.strip()]
    for number in allowed:
        if not _E164_RE.match(number):
            raise ValueError("allowed_numbers must be in +E.164 format")
    if allowed:
        trunk["allowedNumbers"] = allowed
    return {"trunk": trunk}


def create_web_participant_token(
    *,
    room_name: str,
    participant_identity: str,
    participant_name: str = "Pafi",
    dispatch_agent: bool = True,
    metadata: Mapping[str, Any] | None = None,
    config: LiveKitVoiceConfig | None = None,
) -> str:
    """Create a LiveKit JWT for the browser/WebRTC MVP."""
    try:
        from livekit.api import (
            AccessToken,
            RoomAgentDispatch,
            RoomConfiguration,
            VideoGrants,
        )  # type: ignore
    except Exception as exc:  # pragma: no cover - covered by operator smoke
        raise RuntimeError(
            "Install the livekit optional extra before minting room tokens."
        ) from exc

    cfg = config or load_livekit_config()
    if not cfg.has_credentials:
        raise ValueError(
            "LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET are required"
        )

    token = (
        AccessToken(cfg.livekit_api_key, cfg.livekit_api_secret)
        .with_identity(_safe_slug(participant_identity))
        .with_name(participant_name)
        .with_grants(VideoGrants(room_join=True, room=room_name))
    )
    if dispatch_agent:
        token = token.with_room_config(
            RoomConfiguration(
                agents=[
                    RoomAgentDispatch(
                        agent_name=cfg.agent_name, metadata=_json_metadata(metadata)
                    )
                ]
            )
        )
    return token.to_jwt()


def build_room_token_output(
    *,
    livekit_url: str,
    room: str,
    identity: str,
    token: str,
    show_token: bool = False,
) -> dict[str, Any]:
    """Build room-token CLI output without exposing bearer material by default."""
    data: dict[str, Any] = {
        "livekit_url": livekit_url,
        "room": room,
        "identity": identity,
        "token_sensitive": True,
    }
    if show_token:
        data["token"] = token
    else:
        data["token"] = "redacted"
        data["token_note"] = "Use --show-token only in a private terminal."
    return data


def _print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hermes LiveKit voice helpers")
    sub = parser.add_subparsers(dest="command", required=True)

    preflight = sub.add_parser("preflight")
    preflight.add_argument("--require-phone-number", action="store_true")
    preflight.add_argument("--include-realtime", action="store_true")

    dispatch = sub.add_parser("dispatch-json")
    dispatch.add_argument("--trunk-id", action="append", default=[])
    dispatch.add_argument("--mode", choices=["webrtc", "sip"], default="sip")

    trunk = sub.add_parser("inbound-trunk-json")
    trunk.add_argument("--phone-number", default="")
    trunk.add_argument("--allowed-number", action="append", default=[])

    token = sub.add_parser("room-token")
    token.add_argument("--room", default="")
    token.add_argument("--identity", default="pafi")
    token.add_argument("--name", default="Pafi")
    token.add_argument("--no-agent-dispatch", action="store_true")
    token.add_argument("--show-token", action="store_true")

    sub.add_parser("worker-status")

    args = parser.parse_args(argv)
    cfg = load_livekit_config()

    if args.command == "preflight":
        _print_json(
            build_livekit_preflight(
                require_phone_number=args.require_phone_number,
                include_realtime=args.include_realtime,
            )
        )
        return 0
    if args.command == "dispatch-json":
        _print_json(
            build_dispatch_rule_payload(
                agent_name=cfg.agent_name,
                room_prefix=cfg.room_prefix,
                metadata=build_realtime_room_metadata(mode=args.mode),
                trunk_ids=args.trunk_id,
            )
        )
        return 0
    if args.command == "inbound-trunk-json":
        phone = args.phone_number or cfg.phone_number
        _print_json(
            build_inbound_trunk_payload(phone, allowed_numbers=args.allowed_number)
        )
        return 0
    if args.command == "room-token":
        room = args.room or build_room_name(cfg.room_prefix, args.identity)
        jwt = create_web_participant_token(
            room_name=room,
            participant_identity=args.identity,
            participant_name=args.name,
            dispatch_agent=not args.no_agent_dispatch,
            metadata=build_realtime_room_metadata(mode="webrtc"),
            config=cfg,
        )
        _print_json(
            build_room_token_output(
                livekit_url=cfg.livekit_url,
                room=room,
                identity=args.identity,
                token=jwt,
                show_token=args.show_token,
            )
        )
        return 0
    if args.command == "worker-status":
        _print_json(build_realtime_worker_status(config=cfg))
        return 0
    parser.error("unknown command")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
