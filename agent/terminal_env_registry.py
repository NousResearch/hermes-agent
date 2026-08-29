"""
Terminal Environment Registry
=============================

Central map of registered pluggable terminal backends. Populated by plugins
at load time via :meth:`PluginContext.register_terminal_environment_provider`;
consumed by :func:`tools.terminal_tool._create_environment` and the
classification helpers spread across the terminal/file/approval/prompt
surfaces.

Unlike the image/video/web/browser registries there is **no active-provider
resolution here**: the active backend is whatever ``TERMINAL_ENV`` /
``terminal.backend`` names, exactly as for built-in backends. The registry's
only job is mapping that name to a provider instance (and answering the
classification questions the core historically answered with frozensets of
built-in names).

Built-in backend names are reserved — :func:`register_provider` rejects a
provider whose ``name`` collides with one, so a plugin can never shadow the
in-tree docker/modal/... implementations.

Mirrors :mod:`agent.browser_registry` scope semantics: providers register
into a per-profile scope (multiplexed gateways) or the global base map.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional

from agent.terminal_env_provider import TerminalEnvironmentProvider
from hermes_constants import hermes_home_key

logger = logging.getLogger(__name__)


#: Names owned by in-tree backends in tools/environments/ — never
#: registrable by plugins. Includes internal-mode aliases (managed_modal).
BUILTIN_BACKEND_NAMES = frozenset({
    "local", "docker", "singularity", "modal", "managed_modal",
    "daytona", "vercel_sandbox", "ssh",
})


_providers: Dict[str, TerminalEnvironmentProvider] = {}
_scoped_providers: Dict[str, Dict[str, TerminalEnvironmentProvider]] = {}
_generation = 0
_scoped_generations: Dict[str, int] = {}
_lock = threading.Lock()


def register_provider(
    provider: TerminalEnvironmentProvider, *, scope: Optional[str] = None
) -> None:
    """Register a terminal environment provider.

    Re-registration (same ``name``) overwrites the previous entry — makes
    hot-reload scenarios (tests, dev loops) behave predictably.

    Raises:
        TypeError: not a TerminalEnvironmentProvider instance.
        ValueError: empty name or collision with a built-in backend name.
    """
    if not isinstance(provider, TerminalEnvironmentProvider):
        raise TypeError(
            f"register_provider() expects a TerminalEnvironmentProvider "
            f"instance, got {type(provider).__name__}"
        )
    raw_name = provider.name
    if not isinstance(raw_name, str) or not raw_name.strip():
        raise ValueError("Terminal environment provider .name must be a non-empty string")
    name = raw_name.strip().lower()
    if name in BUILTIN_BACKEND_NAMES:
        raise ValueError(
            f"Terminal backend name '{name}' is reserved for the built-in "
            f"{name} backend and cannot be registered by a plugin"
        )
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
            "Terminal environment provider '%s' re-registered (was %r)",
            name, type(existing).__name__,
        )
    else:
        logger.debug(
            "Registered terminal environment provider '%s' (%s)",
            name, type(provider).__name__,
        )


def list_providers(*, scope: Optional[str] = None) -> List[TerminalEnvironmentProvider]:
    """Return all registered providers, sorted by name."""
    with _lock:
        merged = dict(_providers)
        merged.update(_scoped_providers.get(scope or hermes_home_key(), {}))
        items = list(merged.values())
    return sorted(items, key=lambda p: p.name)


def get_provider(
    name: str, *, scope: Optional[str] = None
) -> Optional[TerminalEnvironmentProvider]:
    """Return the provider registered under *name*, or None."""
    if not isinstance(name, str):
        return None
    key = name.strip().lower()
    with _lock:
        return (
            _scoped_providers.get(scope or hermes_home_key(), {}).get(key)
            or _providers.get(key)
        )


def plugin_backend_names(*, scope: Optional[str] = None) -> List[str]:
    """Names of all registered plugin backends (sorted)."""
    return [p.name.strip().lower() for p in list_providers(scope=scope)]


def provider_flag(name: str, attr: str, default=False):
    """Read a classification attribute off the provider for *name*.

    Fail-soft: unknown backend or a raising property returns *default* so a
    misbehaving plugin degrades to built-in-equivalent behavior instead of
    taking the terminal tool down.
    """
    provider = get_provider(name)
    if provider is None:
        return default
    try:
        return getattr(provider, attr, default)
    except Exception:
        logger.debug(
            "Terminal environment provider '%s' attribute '%s' raised",
            name, attr, exc_info=True,
        )
        return default


def plugin_strip_env_keys() -> frozenset:
    """Union of every registered provider's ``strip_env_keys``.

    Secrets are stripped for ALL registered backends, not just the active
    one — a token in the process environment is strippable regardless of
    which backend is selected (mirrors how MODAL_*/DAYTONA_API_KEY sit in
    the static tier-1 set unconditionally).
    """
    keys: set = set()
    with _lock:
        all_providers = list(_providers.values())
        for scoped in _scoped_providers.values():
            all_providers.extend(scoped.values())
    for provider in all_providers:
        try:
            keys.update(provider.strip_env_keys)
        except Exception:
            logger.debug(
                "Terminal environment provider strip_env_keys raised",
                exc_info=True,
            )
    return frozenset(keys)


def snapshot_registration(
    name: str, *, scope: Optional[str] = None
) -> Optional[TerminalEnvironmentProvider]:
    with _lock:
        target = _providers if scope is None else _scoped_providers.get(scope, {})
        return target.get(name.strip().lower())


def registry_generation(*, scope: Optional[str] = None) -> tuple:
    """Return a cache fingerprint for the global base and one profile."""
    active_scope = scope or hermes_home_key()
    with _lock:
        return _generation, _scoped_generations.get(active_scope, 0)


def restore_registration(
    name: str,
    current: TerminalEnvironmentProvider,
    previous: Optional[TerminalEnvironmentProvider],
    *,
    scope: Optional[str] = None,
) -> bool:
    """Restore a plugin registration only when *current* is still installed."""
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


def _reset_for_tests() -> None:
    """Clear all registrations. Test hook — mirrors sibling registries."""
    global _generation
    with _lock:
        _providers.clear()
        _scoped_providers.clear()
        _scoped_generations.clear()
        _generation = 0
