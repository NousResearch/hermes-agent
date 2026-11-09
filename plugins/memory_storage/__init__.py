"""Memory storage provider discovery: bundled ``plugins/memory_storage/<name>/`` then user
``$HERMES_HOME/plugins/<name>/`` (bundled wins on collision). The local backend is core
(``hermes_memory_port_local``), not discovered here. Hosted backends are standalone plugins built
on the ``hermes_memory_port`` ABCs and run the shared suite in ``hermes_conformance.memory``; no
production caller uses a provider yet (``hermes_memory_port``).

Providers load in-process only: under ``plugins.isolation: host`` a user provider is refused
(:class:`MemoryStorageRefused`) rather than returned as a host proxy, whose port object would be
opaque and whose calls would not carry the caller's scope. A typed port proxy with a scope
envelope on every call is a follow-up.

A user provider's module is named after its canonical directory, so same-named providers in two
profiles (and their sibling modules) never share a module or its import-time state. The discovery
shape itself is shared with the other named storage kinds (``plugins.plugin_loader``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from plugins import plugin_loader as _loader

if TYPE_CHECKING:
    from hermes_memory_port import MemoryStorageProvider

logger = logging.getLogger(__name__)


class MemoryStorageRefused(RuntimeError):
    """A memory storage provider that exists but may not be loaded here (the reason is the
    message)."""


def _provider_cls():
    from hermes_memory_port import MemoryStorageProvider
    return MemoryStorageProvider


_KIND = _loader.NamedProviderKind(
    kind_dir="memory_storage",
    base_cls=_provider_cls,
    register_method="register_memory_storage",
    markers=("register_memory_storage", "MemoryStorageProvider"),
    kind="Memory storage provider", noun="provider",
    refused=lambda: MemoryStorageRefused, logger=logger)


def find_provider_dir(name: str) -> Optional[Path]:
    """Resolve a provider name to its directory (bundled first, then user-installed)."""
    return _KIND.find_provider_dir(name)


def load_memory_storage(name: str) -> Optional[MemoryStorageProvider]:
    """Load a provider instance by name; ``None`` if not found or it fails to load. Raises
    :class:`MemoryStorageRefused` for a user provider under ``plugins.isolation: host``."""
    return _KIND.load(name)
