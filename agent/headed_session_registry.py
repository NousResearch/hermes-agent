"""
Headed Cloud Session Registry
=============================

Central map of registered headed-session providers. Populated at load time
via :func:`register_provider`; consumed by session provisioning (dashboard
PTY, WebVNC, CUA) once a provider is selected.

Mirrors :mod:`agent.terminal_env_registry` scope semantics: providers register
into a per-profile scope (multiplexed gateways) or the global base map.
There is no active-provider auto-detect here — callers name the backend.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional

from agent.headed_session import HeadedCloudSessionProvider
from hermes_constants import hermes_home_key

logger = logging.getLogger(__name__)

_providers: Dict[str, HeadedCloudSessionProvider] = {}
_scoped_providers: Dict[str, Dict[str, HeadedCloudSessionProvider]] = {}
_generation = 0
_scoped_generations: Dict[str, int] = {}
_lock = threading.Lock()


def register_provider(
    provider: HeadedCloudSessionProvider, *, scope: Optional[str] = None
) -> None:
    """Register a headed cloud session provider.

    Re-registration (same ``name``) overwrites the previous entry.

    Raises:
        TypeError: not a HeadedCloudSessionProvider instance.
        ValueError: empty name.
    """
    if not isinstance(provider, HeadedCloudSessionProvider):
        raise TypeError(
            "register_provider() expects a HeadedCloudSessionProvider "
            f"instance, got {type(provider).__name__}"
        )
    raw_name = provider.name
    if not isinstance(raw_name, str) or not raw_name.strip():
        raise ValueError("Headed session provider .name must be a non-empty string")
    name = raw_name.strip().lower()
    global _generation
    with _lock:
        target = _providers if scope is None else _scoped_providers.setdefault(scope, {})
        existing = target.get(name)
        target[name] = provider
        if scope is None:
            _generation += 1
        else:
            _scoped_generations[scope] = _scoped_generations.get(scope, 0) + 1
    if existing is not None:
        logger.debug(
            "Headed session provider '%s' re-registered (was %r)",
            name, type(existing).__name__,
        )
    else:
        logger.debug(
            "Registered headed session provider '%s' (%s)",
            name, type(provider).__name__,
        )


def list_providers(*, scope: Optional[str] = None) -> List[HeadedCloudSessionProvider]:
    """Return all registered providers, sorted by name."""
    with _lock:
        merged = dict(_providers)
        merged.update(_scoped_providers.get(scope or hermes_home_key(), {}))
        items = list(merged.values())
    return sorted(items, key=lambda p: p.name)


def get_provider(
    name: str, *, scope: Optional[str] = None
) -> Optional[HeadedCloudSessionProvider]:
    """Return the provider registered under *name*, or None."""
    if not isinstance(name, str):
        return None
    key = name.strip().lower()
    with _lock:
        return (
            _scoped_providers.get(scope or hermes_home_key(), {}).get(key)
            or _providers.get(key)
        )


def snapshot_registration(
    name: str, *, scope: Optional[str] = None
) -> Optional[HeadedCloudSessionProvider]:
    with _lock:
        target = _providers if scope is None else _scoped_providers.get(scope, {})
        return target.get(name.strip().lower())


def restore_registration(
    name: str,
    current: HeadedCloudSessionProvider,
    previous: Optional[HeadedCloudSessionProvider],
    *,
    scope: Optional[str] = None,
) -> bool:
    """Restore a registration only when *current* is still installed."""
    key = name.strip().lower()
    global _generation
    with _lock:
        target = _providers if scope is None else _scoped_providers.setdefault(scope, {})
        if target.get(key) is not current:
            return False
        if previous is None:
            target.pop(key, None)
        else:
            target[key] = previous
        if scope is None:
            _generation += 1
        else:
            _scoped_generations[scope] = _scoped_generations.get(scope, 0) + 1
            if not target:
                _scoped_providers.pop(scope, None)
    return True


def registry_generation(*, scope: Optional[str] = None) -> tuple:
    """Return a cache fingerprint for the global base and one profile."""
    active_scope = scope or hermes_home_key()
    with _lock:
        return _generation, _scoped_generations.get(active_scope, 0)


def _reset_for_tests() -> None:
    """Clear all registrations. Test hook — mirrors sibling registries."""
    global _generation
    with _lock:
        _providers.clear()
        _scoped_providers.clear()
        _scoped_generations.clear()
        _generation = 0
