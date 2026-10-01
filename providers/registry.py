"""Canonical provider registry authority.

This module owns provider registry state, aliases, source precedence, per-HERMES_HOME
layers, and effective provider lookup. Discovery mechanics live in
:mod:`providers.discovery`; callers use the package-level API exported by
:mod:`providers`.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path

from providers.base import ProviderProfile

# Process-wide layer: bundled plugins, pip entry points, legacy providers/<name>.py.
_REGISTRY: dict[str, ProviderProfile] = {}
_ALIASES: dict[str, str] = {}
# Where the CURRENT process-wide registration of each canonical name came from.
_SOURCES: dict[str, str] = {}
_PROVIDER_LIST_CACHE: list[ProviderProfile] | None = None


@dataclass
class _HomeLayer:
    """The $HERMES_HOME provider layer for one profile home."""

    registry: dict[str, ProviderProfile] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)
    stamps: tuple = ()
    stamp_checked_at: float | None = None


_HOME_LAYERS: dict[str, _HomeLayer] = {}
_HOME_LAYERS_LOCK = threading.Lock()
# User-plugin imports register into the currently scanned home. ContextVar keeps
# concurrent profile-home scans isolated without holding a lock across imports.
_REGISTRATION_TARGET: ContextVar[_HomeLayer | None] = ContextVar(
    "_provider_registration_target", default=None
)


def _get_or_create_home_layer(key: str) -> _HomeLayer:
    with _HOME_LAYERS_LOCK:
        layer = _HOME_LAYERS.get(key)
        if layer is None:
            layer = _HOME_LAYERS[key] = _HomeLayer()
    return layer


@contextmanager
def _registration_target(layer: _HomeLayer):
    token = _REGISTRATION_TARGET.set(layer)
    try:
        yield
    finally:
        _REGISTRATION_TARGET.reset(token)


def _snapshot_state() -> tuple[
    dict[str, ProviderProfile],
    dict[str, str],
    dict[str, str],
    list[ProviderProfile] | None,
]:
    """Private snapshot used by discovery tooling that must load a plugin in isolation."""
    cache = None if _PROVIDER_LIST_CACHE is None else list(_PROVIDER_LIST_CACHE)
    return dict(_REGISTRY), dict(_ALIASES), dict(_SOURCES), cache


def _restore_state(
    snapshot: tuple[
        dict[str, ProviderProfile],
        dict[str, str],
        dict[str, str],
        list[ProviderProfile] | None,
    ]
) -> None:
    global _PROVIDER_LIST_CACHE
    registry, aliases, sources, cache = snapshot
    _REGISTRY.clear()
    _REGISTRY.update(registry)
    _ALIASES.clear()
    _ALIASES.update(aliases)
    _SOURCES.clear()
    _SOURCES.update(sources)
    _PROVIDER_LIST_CACHE = None if cache is None else list(cache)


def register_provider(profile: ProviderProfile) -> None:
    """Register a provider profile by canonical name and aliases."""
    global _PROVIDER_LIST_CACHE

    layer = _REGISTRATION_TARGET.get()
    if layer is not None:
        layer.registry[profile.name] = profile
        for alias in profile.aliases:
            layer.aliases[alias] = profile.name
    else:
        from providers.discovery import current_registration_source

        _REGISTRY[profile.name] = profile
        _SOURCES[profile.name] = current_registration_source() or "runtime"
        for alias in profile.aliases:
            _ALIASES[alias] = profile.name
        _PROVIDER_LIST_CACHE = None



def provider_source(name: str) -> str | None:
    """Return the source of the effective profile currently registered as *name*."""
    from providers.discovery import home_layer

    layer = home_layer()
    canonical = layer.aliases.get(name) or _ALIASES.get(name, name)
    if canonical in layer.registry:
        return "user"
    return _SOURCES.get(canonical)


def get_provider_profile(name: str) -> ProviderProfile | None:
    """Look up the effective provider profile by canonical name or alias."""
    from agent.safe_worker_policy import safe_worker_enabled

    if safe_worker_enabled():
        return None

    from providers.discovery import (
        bound_home_layer,
        ensure_process_discovered,
        refresh_home_layer,
    )

    ensure_process_discovered()
    layer, home, key = bound_home_layer()
    checked = refresh_home_layer(layer, home, key)

    def lookup(candidate: str) -> ProviderProfile | None:
        canonical = layer.aliases.get(candidate) or _ALIASES.get(candidate, candidate)
        return layer.registry.get(canonical) or _REGISTRY.get(canonical)

    profile = lookup(name)
    is_custom_route = isinstance(name, str) and name.lower().startswith("custom:")
    if profile is None and not is_custom_route and not checked:
        if refresh_home_layer(layer, home, key, force=True):
            profile = lookup(name)
    if profile is None and is_custom_route:
        profile = lookup("custom")
    return profile


def routed_model_rejects_vision_tool_messages(provider: str, model: str) -> bool:
    """Whether the active route or routed target rejects image tool parts."""
    from providers.identity import is_routing_aggregator

    provider_name = str(provider or "").strip().lower()
    profile = get_provider_profile(provider_name)
    if profile is not None and profile.supports_vision_tool_messages is False:
        return True
    if not is_routing_aggregator(provider_name):
        return False

    target_name, separator, _ = str(model or "").strip().partition("/")
    if not separator or not target_name:
        return False
    target_profile = get_provider_profile(target_name.strip().lower())
    return (
        target_profile is not None
        and target_profile.supports_vision_tool_messages is False
    )


def list_providers() -> list[ProviderProfile]:
    """Return effective canonical profiles for the currently bound profile home."""
    from agent.safe_worker_policy import safe_worker_enabled

    if safe_worker_enabled():
        return []

    from providers.discovery import ensure_process_discovered, home_layer

    ensure_process_discovered()
    layer = home_layer()

    global _PROVIDER_LIST_CACHE
    if _PROVIDER_LIST_CACHE is None:
        seen: set[int] = set()
        cache: list[ProviderProfile] = []
        for profile in _REGISTRY.values():
            if id(profile) not in seen:
                seen.add(id(profile))
                cache.append(profile)
        _PROVIDER_LIST_CACHE = cache

    result = [p for p in _PROVIDER_LIST_CACHE if p.name not in layer.registry]
    result.extend({id(p): p for p in layer.registry.values()}.values())
    return result
