"""
Realtime Voice Provider Registry
================================

Central registry for :class:`agent.realtime_voice_provider.RealtimeVoiceProvider`.
Built-in providers and plugin providers use the same contract. A built-in
registration always wins its name; plugins may replace other plugin providers
to keep development reloads predictable.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional, Set

from agent.realtime_voice_provider import (
    REALTIME_VOICE_PROVIDER_API_VERSION,
    RealtimeVoiceProvider,
)

logger = logging.getLogger(__name__)

_providers: Dict[str, RealtimeVoiceProvider] = {}
_built_in_names: Set[str] = set()
_lock = threading.Lock()


def register_provider(
    provider: RealtimeVoiceProvider,
    *,
    built_in: bool = False,
) -> bool:
    """Register a realtime voice provider.

    Returns ``True`` when accepted. Built-ins replace an earlier plugin with
    the same normalized name. Plugin registrations cannot replace a built-in.
    """
    if not isinstance(provider, RealtimeVoiceProvider):
        raise TypeError(
            "register_provider() expects a RealtimeVoiceProvider instance, "
            f"got {type(provider).__name__}"
        )

    name = provider.name
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Realtime voice provider .name must be a non-empty string")
    key = name.strip().lower()

    api_version = getattr(provider, "api_version", None)
    if api_version != REALTIME_VOICE_PROVIDER_API_VERSION:
        logger.warning(
            "Realtime voice provider '%s' targets API v%s; Hermes supports v%s. "
            "Registration ignored.",
            key,
            api_version,
            REALTIME_VOICE_PROVIDER_API_VERSION,
        )
        return False

    with _lock:
        if not built_in and key in _built_in_names:
            logger.warning(
                "Realtime voice provider '%s' shadows a built-in name; "
                "registration ignored. Built-in providers always win.",
                key,
            )
            return False

        existing = _providers.get(key)
        if built_in:
            _built_in_names.add(key)
        _providers[key] = provider

    if existing is not None:
        logger.debug(
            "Realtime voice provider '%s' re-registered (was %r)",
            key,
            type(existing).__name__,
        )
    else:
        logger.debug(
            "Registered realtime voice provider '%s' (%s)",
            key,
            type(provider).__name__,
        )
    return True


def list_providers() -> List[RealtimeVoiceProvider]:
    """Return registered providers sorted by normalized name."""
    with _lock:
        items = list(_providers.items())
    return [provider for _, provider in sorted(items)]


def get_provider(name: str) -> Optional[RealtimeVoiceProvider]:
    """Return a provider by case-insensitive, whitespace-tolerant name."""
    if not isinstance(name, str):
        return None
    with _lock:
        return _providers.get(name.strip().lower())


def is_builtin_provider(name: str) -> bool:
    """Return whether *name* is reserved by an active built-in provider."""
    if not isinstance(name, str):
        return False
    with _lock:
        return name.strip().lower() in _built_in_names


def _reset_for_tests() -> None:
    """Clear all providers and built-in reservations. Test-only."""
    with _lock:
        _providers.clear()
        _built_in_names.clear()
