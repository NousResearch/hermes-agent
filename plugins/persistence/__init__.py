"""Persistence providers (``persistence.provider`` in config.yaml), resolved by explicit name only.

``sqlite`` is built in and never discovered. Any other name resolves to exactly one of: a bundled package
``plugins/persistence/<name>/`` or an installed distribution's entry point named ``<name>`` in group
``hermes_agent.persistence``. Either one exposes ``register(ctx)`` and calls
``ctx.register_persistence_backend(factory)``, where
``factory(stores: Mapping[str, Path], *, read_only: bool, read_patience_s: Optional[float]) -> PersistenceBackend``.

A name that resolves nowhere, resolves more than once, or fails to load (import, ``register`` or a missing
registration) raises ``ValueError`` naming the provider; nothing falls back to SQLite. User plugin directories
(``$HERMES_HOME/plugins``, ``./.hermes/plugins``) are never consulted: in-process storage code is not user-pluggable,
like config backends (``plugins/config_backends``). ``persistence_assembly`` is the only caller.
"""

from __future__ import annotations

import importlib.metadata
import logging
from pathlib import Path
from typing import Any, Callable, Optional

from plugins import plugin_loader as _loader

__all__ = ["BUILTIN_PROVIDER", "ENTRY_POINTS_GROUP", "backend_factory", "provider_names", "resolve"]

logger = logging.getLogger(__name__)

BUILTIN_PROVIDER = "sqlite"
ENTRY_POINTS_GROUP = "hermes_agent.persistence"
_PERSISTENCE_PLUGINS_DIR = Path(__file__).parent

BackendFactory = Callable[..., Any]


def _bundled_dir(name: str) -> Optional[Path]:
    candidate = _loader.plugin_child_dir(_PERSISTENCE_PLUGINS_DIR, name)
    if candidate is None:
        return None
    return candidate if candidate.is_dir() and (candidate / "__init__.py").exists() else None


def _entry_points(name: Optional[str] = None) -> list[Any]:
    eps = list(importlib.metadata.entry_points(group=ENTRY_POINTS_GROUP))
    return [ep for ep in eps if name is None or ep.name == name]


def provider_names() -> list[str]:
    """Every name ``resolve`` could answer, without importing any provider: the built-in, bundled dirs, entry points."""
    names = {BUILTIN_PROVIDER}
    names.update(child.name for child in _loader.iter_plugin_dirs(_PERSISTENCE_PLUGINS_DIR))
    names.update(ep.name for ep in _entry_points())
    return sorted(names)


def resolve(name: str) -> str:
    """``name`` when exactly one source provides it (the built-in counts for ``sqlite``), without loading it; raises
    ``ValueError`` when none or more than one does."""
    if name == BUILTIN_PROVIDER:
        return name
    sources = ([f"bundled {_bundled_dir(name)}"] if _bundled_dir(name) else []) + [
        f"entry point {ep.value!r}" for ep in _entry_points(name)]
    if not sources:
        raise ValueError(f"unknown persistence provider {name!r} (known: {', '.join(provider_names())})")
    if len(sources) > 1:
        raise ValueError(f"persistence provider {name!r} is provided more than once ({'; '.join(sources)}); "
                         "remove all but one")
    return name


class _Registration(_loader.NoopPluginContext):
    """The ``ctx`` handed to a persistence plugin's ``register``: captures its one backend factory."""

    def __init__(self) -> None:
        self.factory: Optional[BackendFactory] = None

    def register_persistence_backend(self, factory: BackendFactory) -> None:
        if not callable(factory):
            raise TypeError("register_persistence_backend needs a callable backend factory")
        self.factory = factory


def _load_bundled(name: str, plugin_dir: Path) -> Any:
    module = _loader.load_plugin_module(f"{__name__}.{name}", plugin_dir, parents=("plugins", __name__),
                                        logger=logger)
    if module is None:
        raise ImportError(f"importing {plugin_dir} failed (see debug log)")
    return module


def backend_factory(name: str) -> BackendFactory:
    """The backend factory of the plugin provider ``name``; ``ValueError`` naming it when it is the built-in (which
    ``persistence_assembly`` constructs itself, as the storage layer's owner), unknown, provided twice, or fails to
    load."""
    resolve(name)
    if name == BUILTIN_PROVIDER:
        raise ValueError(f"persistence provider {name!r} is built in: persistence_assembly constructs it, not a plugin")
    plugin_dir = _bundled_dir(name)
    try:
        plugin = _load_bundled(name, plugin_dir) if plugin_dir else _entry_points(name)[0].load()
        register = getattr(plugin, "register", None)
        if not callable(register):
            raise TypeError("it has no register(ctx)")
        ctx = _Registration()
        register(ctx)
        if ctx.factory is None:
            raise TypeError("register(ctx) did not call ctx.register_persistence_backend")
    except Exception as exc:  # health: allow BLE001 -- any load failure refuses the configured provider
        raise ValueError(f"persistence provider {name!r} failed to load: {exc}") from exc
    return ctx.factory
