"""Persistence provider discovery: user ``$HERMES_HOME/plugins/<name>/`` dirs that register a ``PersistenceProvider``.

The built-in SQLite backend is core, not discovered here, so it stays the fallback; one provider is active
(``persistence.provider`` in config.yaml, ``sqlite`` = built-in). A provider holds the session store, so it always
loads in-process: under ``plugins.isolation: host`` it is refused rather than moved into the plugin host. A loaded
provider is kept for the life of the process, since every store open asks for it again.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

from plugins import plugin_loader as _loader

logger = logging.getLogger(__name__)

# Synthetic parent package for user-installed providers (keeps them out of the bundled namespace).
_USER_NAMESPACE = "_hermes_user_persistence"

_lock = threading.Lock()
_loaded: dict[str, object] = {}


def _is_persistence_provider_dir(path: Path) -> bool:
    """Cheap text heuristic: ``__init__.py`` mentions the persistence provider contract."""
    init_file = path / "__init__.py"
    if not init_file.exists():
        return False
    try:
        source = init_file.read_text(errors="replace", encoding="utf-8-sig")[:8192]
        return "register_persistence_provider" in source or "PersistenceProvider" in source
    except Exception:
        return False


def find_provider_dir(name: str) -> Optional[Path]:
    """Resolve a provider name to its user plugin directory, or None."""
    user_dir = _loader.user_plugins_dir()
    candidate = _loader.plugin_child_dir(user_dir, name) if user_dir else None
    return candidate if candidate and candidate.is_dir() and _is_persistence_provider_dir(candidate) else None


def load_persistence_provider(name: str):
    """The ``PersistenceProvider`` registered as ``name``; None when no such plugin is installed or it fails to load."""
    with _lock:
        if name in _loaded:
            return _loaded[name]
    provider_dir = find_provider_dir(name)
    if provider_dir is None:
        logger.debug("Persistence provider '%s' not found in user plugins", name)
        return None
    if _loader.refused_as_disabled(name, provider_dir, bundled=False, kind="Persistence provider", logger=logger):
        return None
    from hermes_cli.plugin_isolation import ISOLATION_HOST, isolation_mode
    if isolation_mode() == ISOLATION_HOST:
        logger.warning("Persistence provider '%s' is refused under plugins.isolation: host; "
                       "a session store cannot run in the plugin host", name)
        return None
    provider = _loader.load_named(name, provider_dir, _load_provider_from_dir,
                                  kind="Persistence provider", noun="provider", logger=logger)
    if provider is not None:
        with _lock:
            provider = _loaded.setdefault(name, provider)
    return provider


def loaded_providers() -> list:
    """Every provider this process has loaded, for the process-wide close sweeps."""
    with _lock:
        return list(_loaded.values())


def _load_provider_from_dir(provider_dir: Path):
    """Import a provider module and extract its PersistenceProvider (register(ctx) or subclass)."""
    from persistence.core.provider import PersistenceProvider
    name = provider_dir.name
    mod = _loader.load_plugin_module(
        f"{_USER_NAMESPACE}.{name}", provider_dir, parents=(__package__.rpartition(".")[0], __package__),
        logger=logger, synthetic_namespace=_USER_NAMESPACE)
    return mod and _loader.instance_from_module(
        mod, collector=_ProviderCollector(), collected_attr="provider",
        base_cls=PersistenceProvider, name=name, logger=logger)


class _ProviderCollector(_loader.NoopPluginContext):
    """Fake plugin context that captures register_persistence_provider calls."""

    def __init__(self):
        self.provider = None

    def register_persistence_provider(self, provider):
        self.provider = provider
