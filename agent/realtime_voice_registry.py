"""Profile-scoped registry for plugin-provided realtime voice transports."""

from __future__ import annotations

import threading
from typing import Optional

from agent.realtime_voice import RealtimeVoiceProvider
from hermes_constants import hermes_home_key

_providers: dict[str, RealtimeVoiceProvider] = {}
_scoped_providers: dict[str, dict[str, RealtimeVoiceProvider]] = {}
_lock = threading.Lock()


def register_provider(
    provider: RealtimeVoiceProvider, *, scope: Optional[str] = None
) -> None:
    if not isinstance(provider, RealtimeVoiceProvider):
        raise TypeError(
            "register_provider() expects a RealtimeVoiceProvider instance, "
            f"got {type(provider).__name__}"
        )
    name = provider.name
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Realtime voice provider .name must be a non-empty string")
    key = name.strip().lower()
    with _lock:
        target = _providers if scope is None else _scoped_providers.setdefault(scope, {})
        target[key] = provider


def list_providers(*, scope: Optional[str] = None) -> list[RealtimeVoiceProvider]:
    with _lock:
        merged = dict(_providers)
        merged.update(_scoped_providers.get(scope or hermes_home_key(), {}))
    return sorted(merged.values(), key=lambda provider: provider.name)


def get_provider(
    name: str, *, scope: Optional[str] = None
) -> Optional[RealtimeVoiceProvider]:
    if not isinstance(name, str):
        return None
    key = name.strip().lower()
    with _lock:
        return _scoped_providers.get(scope or hermes_home_key(), {}).get(key) or _providers.get(key)


def snapshot_registration(
    name: str, *, scope: Optional[str] = None
) -> Optional[RealtimeVoiceProvider]:
    key = name.strip().lower()
    with _lock:
        target = _providers if scope is None else _scoped_providers.get(scope, {})
        return target.get(key)


def restore_registration(
    name: str,
    current: RealtimeVoiceProvider,
    previous: Optional[RealtimeVoiceProvider],
    *,
    scope: Optional[str] = None,
) -> bool:
    key = name.strip().lower()
    with _lock:
        target = _providers if scope is None else _scoped_providers.setdefault(scope, {})
        if target.get(key) is not current:
            return False
        if previous is None:
            target.pop(key, None)
        else:
            target[key] = previous
        if scope is not None and not target:
            _scoped_providers.pop(scope, None)
    return True


def _reset_for_tests() -> None:
    with _lock:
        _providers.clear()
        _scoped_providers.clear()
