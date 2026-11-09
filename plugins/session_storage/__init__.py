"""Session storage provider discovery: bundled ``plugins/session_storage/<name>/`` then user
``$HERMES_HOME/plugins/<name>/`` (bundled wins on collision). The local backend is core
(``hermes_state_port_local``), not discovered here. Hosted backends are standalone plugins built on
the ``hermes_state_port`` ABCs and run the shared suite in ``hermes_conformance.storage``; no
production caller uses a provider yet (``hermes_state_port``).

Providers load in-process only: under ``plugins.isolation: host`` a user provider is refused
(:class:`SessionStorageRefused`) rather than returned as a host proxy, whose nested ports would be
opaque and whose calls would not carry the caller's scope. Typed port proxies with a scope envelope
on every call are a follow-up.

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
    from hermes_state_port import SessionStorageProvider

logger = logging.getLogger(__name__)


class SessionStorageRefused(RuntimeError):
    """A session storage provider that exists but may not be loaded here (the reason is the
    message)."""


def _provider_cls():
    from hermes_state_port import SessionStorageProvider
    return SessionStorageProvider


_KIND = _loader.NamedProviderKind(
    kind_dir="session_storage",
    base_cls=_provider_cls,
    register_method="register_session_storage",
    markers=("register_session_storage", "SessionStorageProvider"),
    kind="Session storage provider", noun="provider",
    refused=lambda: SessionStorageRefused, logger=logger)


def find_provider_dir(name: str) -> Optional[Path]:
    """Resolve a provider name to its directory (bundled first, then user-installed)."""
    return _KIND.find_provider_dir(name)


def load_session_storage(name: str) -> Optional[SessionStorageProvider]:
    """Load a provider instance by name; ``None`` if not found or it fails to load. Raises
    :class:`SessionStorageRefused` for a user provider under ``plugins.isolation: host``."""
    return _KIND.load(name)
