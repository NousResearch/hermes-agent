"""Session-store provider registration and construction.

SQLite remains the default.  Other backends register a factory and are selected explicitly
through ``sessiondb.provider``; an unknown configured name fails closed rather than silently
opening a different store.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from hermes_state import SessionDB


class SessionStore(Protocol):
    """Lifecycle surface the shared registry requires from a session store."""

    db_path: Path
    _shared_registry_owned: bool

    def close(self) -> None: ...


SessionStoreFactory = Callable[[Path], SessionStore]

_PROVIDERS_LOCK = threading.RLock()
_PROVIDERS: dict[str, SessionStoreFactory] = {}


def _sqlite_factory(db_path: Path) -> "SessionDB":
    from hermes_state import SessionDB

    return SessionDB(db_path=db_path)


def _provider_name(name: str) -> str:
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("session-store provider name must not be empty")
    return normalized


def register_session_store_provider(name: str, factory: SessionStoreFactory) -> None:
    """Register one named session-store factory; duplicate names are rejected."""
    normalized = _provider_name(name)
    if not callable(factory):
        raise TypeError("session-store provider factory must be callable")
    with _PROVIDERS_LOCK:
        if normalized in _PROVIDERS:
            raise ValueError(f"session-store provider {normalized!r} is already registered")
        _PROVIDERS[normalized] = factory


def resolve_session_store_provider(
    config: Mapping[str, Any] | None = None,
) -> SessionStoreFactory:
    """Resolve ``sessiondb.provider`` at call time, defaulting to bundled SQLite."""
    if config is None:
        from hermes_cli.config_effective import load_user_config_effective

        config = load_user_config_effective()

    section = config.get("sessiondb", {})
    if section is None:
        section = {}
    if not isinstance(section, Mapping):
        raise ValueError("sessiondb config must be a mapping")

    configured = section.get("provider", "sqlite")
    if not isinstance(configured, str):
        raise ValueError("sessiondb.provider must be a string")
    name = _provider_name(configured or "sqlite")

    with _PROVIDERS_LOCK:
        factory = _PROVIDERS.get(name)
    if factory is None:
        raise ValueError(f"unknown session-store provider {name!r}")
    return factory


def open_session_store(db_path: Path) -> SessionStore:
    """Construct the configured session store for *db_path*."""
    return resolve_session_store_provider()(Path(db_path))


register_session_store_provider("sqlite", _sqlite_factory)
