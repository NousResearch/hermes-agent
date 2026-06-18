"""Configuration helpers for the voice_call plugin.

Secrets stay in environment variables. Behavioral settings may live in
``config.yaml`` under ``voice_call`` and are read through Hermes' normal config
loader when available.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

_ALLOWED_POLICIES = {
    "no_escalation",
    "transfer_on_request",
    "transfer_if_blocked",
    "take_message",
}


def _load_voice_config() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly()
    except Exception:
        return {}
    section = cfg.get("voice_call", {}) if isinstance(cfg, dict) else {}
    return section if isinstance(section, dict) else {}


def _csv_env(name: str) -> list[str]:
    raw = os.environ.get(name, "")
    return [part.strip() for part in raw.split(",") if part.strip()]


def _config_list(cfg: dict[str, Any], key: str, env_name: str) -> list[str]:
    value = cfg.get(key)
    if isinstance(value, str):
        out = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple)):
        out = [str(part).strip() for part in value if str(part).strip()]
    else:
        out = []
    return out or _csv_env(env_name)


@dataclass(frozen=True)
class CallerProfile:
    on_behalf_of: str = "Jason Lai"
    assistant_name: str = "Hermes"
    callback_number: str = ""
    disclosure: str = "This is Hermes, an AI assistant calling on behalf of Jason Lai."
    transfer_number: str = ""


@dataclass(frozen=True)
class VoiceCallConfig:
    service_url: str = ""
    timeout_seconds: float = 8.0
    allowed_prefixes: list[str] = field(default_factory=list)
    blocked_prefixes: list[str] = field(default_factory=list)
    caller_profile: CallerProfile = field(default_factory=CallerProfile)
    escalation_policies: set[str] = field(default_factory=lambda: set(_ALLOWED_POLICIES))


def load_voice_call_config() -> VoiceCallConfig:
    cfg = _load_voice_config()
    profile = cfg.get("caller_profile", {}) if isinstance(cfg.get("caller_profile"), dict) else {}
    service_url = str(cfg.get("service_url") or os.environ.get("VOICE_CALL_SERVICE_URL", "")).strip().rstrip("/")
    timeout_raw = cfg.get("timeout_seconds", os.environ.get("VOICE_CALL_TIMEOUT_SECONDS", "8"))
    try:
        timeout = max(1.0, min(float(timeout_raw), 30.0))
    except (TypeError, ValueError):
        timeout = 8.0
    return VoiceCallConfig(
        service_url=service_url,
        timeout_seconds=timeout,
        allowed_prefixes=_config_list(cfg, "allowed_prefixes", "VOICE_CALL_ALLOWED_PREFIXES"),
        blocked_prefixes=_config_list(cfg, "blocked_prefixes", "VOICE_CALL_BLOCKED_PREFIXES"),
        caller_profile=CallerProfile(
            on_behalf_of=str(profile.get("on_behalf_of") or os.environ.get("VOICE_CALL_ON_BEHALF_OF", "Jason Lai")),
            assistant_name=str(profile.get("assistant_name") or os.environ.get("VOICE_CALL_ASSISTANT_NAME", "Hermes")),
            callback_number=str(profile.get("callback_number") or os.environ.get("VOICE_CALL_CALLBACK_NUMBER", "")),
            disclosure=str(profile.get("disclosure") or os.environ.get("VOICE_CALL_DISCLOSURE", "This is Hermes, an AI assistant calling on behalf of Jason Lai.")),
            transfer_number=str(profile.get("transfer_number") or os.environ.get("VOICE_CALL_TRANSFER_NUMBER", "")),
        ),
    )


def voice_call_available() -> bool:
    cfg = load_voice_call_config()
    # Tool availability only means Hermes can reach its local service. Twilio
    # credentials are checked by the service before it places real calls.
    return bool(cfg.service_url)


ALLOWED_ESCALATION_POLICIES = _ALLOWED_POLICIES
