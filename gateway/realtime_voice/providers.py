"""Provider registry for Hermes realtime voice sessions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from gateway.realtime_voice.config import RealtimeVoiceConfig
from gateway.realtime_voice.session import RealtimeVoiceSession

RealtimeVoiceSessionFactory = Callable[..., RealtimeVoiceSession]


class RealtimeVoiceProviderError(RuntimeError):
    """Raised when a realtime voice provider cannot be resolved."""


_PROVIDER_ALIASES: dict[str, str] = {
    "xai": "xai",
    "xai-oauth": "xai",
    "grok": "xai",
    "grok-voice": "xai",
}
_PROVIDER_FACTORIES: dict[str, RealtimeVoiceSessionFactory] = {}


def normalize_realtime_voice_provider(provider: str | None) -> str:
    """Normalize a configured provider name to the registry key."""

    raw = str(provider or "xai").strip().lower().replace("_", "-")
    return _PROVIDER_ALIASES.get(raw, raw)


def register_realtime_voice_provider(name: str, factory: RealtimeVoiceSessionFactory, *, aliases: tuple[str, ...] = ()) -> None:
    """Register a realtime voice provider factory."""

    normalized = normalize_realtime_voice_provider(name)
    _PROVIDER_FACTORIES[normalized] = factory
    _PROVIDER_ALIASES[normalized] = normalized
    for alias in aliases:
        clean = str(alias or "").strip().lower().replace("_", "-")
        if clean:
            _PROVIDER_ALIASES[clean] = normalized


def create_realtime_voice_session(
    config: RealtimeVoiceConfig,
    *,
    instructions: str = "",
    on_event: Any = None,
) -> RealtimeVoiceSession:
    """Create the configured realtime voice provider session."""

    provider = normalize_realtime_voice_provider(getattr(config, "provider", "xai"))
    factory = _PROVIDER_FACTORIES.get(provider)
    if factory is None and provider == "xai":
        factory = _load_xai_factory()
    if factory is None:
        supported = sorted(set(_PROVIDER_ALIASES.values()) | set(_PROVIDER_FACTORIES.keys()) | {"xai"})
        raise RealtimeVoiceProviderError(
            f"Realtime voice provider {provider!r} is not supported. Supported providers: {', '.join(supported)}."
        )
    return factory(config, instructions=instructions, on_event=on_event)


def is_realtime_voice_provider_supported(config: RealtimeVoiceConfig) -> bool:
    """Return whether the configured provider can be created by the registry."""

    provider = normalize_realtime_voice_provider(getattr(config, "provider", "xai"))
    return provider == "xai" or provider in _PROVIDER_FACTORIES


def _load_xai_factory() -> RealtimeVoiceSessionFactory:
    from gateway.realtime_voice.xai import XAIRealtimeVoiceSession

    register_realtime_voice_provider(
        "xai",
        XAIRealtimeVoiceSession,
        aliases=("xai-oauth", "grok", "grok-voice"),
    )
    return XAIRealtimeVoiceSession
