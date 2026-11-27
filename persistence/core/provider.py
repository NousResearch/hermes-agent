"""A persistence provider plugin: the backend ``persistence.provider: <name>`` assembles in place of the built-in SQLite one.

A user plugin of kind ``exclusive`` hands one to ``register(ctx)`` through ``ctx.register_persistence_provider(provider)``;
``plugins/persistence_providers`` loads it and ``persistence_assembly`` is its only caller. ``stores`` maps each served
``hermes_home_key`` to the ``state.db`` path the SQLite backend would use, which a provider without files reads for its
key only; ``settings`` maps the same keys to that profile's ``persistence`` config section. The assembled backend
presents ``PersistenceBackend`` and declares its own capabilities; one without ``physical_admin`` answers both admin
doors with ``AdminUnsupported``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping, Optional

from .backend import PersistenceBackend
from .context import ProfileScope
from .health import StorageHealth

__all__ = ["PersistenceProvider"]


class PersistenceProvider(ABC):
    """One plugin-supplied persistence backend family, selected by ``name``."""

    name: str

    @abstractmethod
    def assemble(self, stores: Mapping[str, Path], settings: Mapping[str, Mapping[str, Any]], *, read_only: bool = False,
                 read_patience_s: Optional[float] = None, check_drift: bool = False) -> PersistenceBackend:
        """A backend over ``stores``, each configured by its ``settings`` entry; assembling touches no store."""

    @abstractmethod
    def storage_health(self, scope: ProfileScope, settings: Mapping[str, Any]) -> StorageHealth:
        """``scope``'s health latch and readiness probe, answered without creating or migrating its store."""

    def store_location(self, scope: ProfileScope, settings: Mapping[str, Any]) -> Optional[Path]:
        """The file holding ``scope``'s store, or ``None`` for a provider without one."""
        return None

    def export_store(self, scope: ProfileScope, settings: Mapping[str, Any], destination: Path) -> bool:
        """Write ``scope``'s whole store to the file ``destination`` for ``hermes backup``; False when there is no
        store to write. A provider whose store is a file in the profile folder needs none: the backup copies it."""
        return False

    def import_store(self, scope: ProfileScope, settings: Mapping[str, Any], source: Path) -> None:
        """Load a file :meth:`export_store` wrote into ``scope``'s store for ``hermes import``; refuses when the
        store already holds data, so an import never overwrites live state."""
        raise NotImplementedError(f"the {type(self).__name__} provider does not import stores")

    def open_compat_store(self, scope: ProfileScope, settings: Mapping[str, Any], store: str, *,
                          row_factory: Any = None) -> Any:
        """A ``sqlite3.Connection``-compatible connection on the profile's ``store`` (a file store such as
        ``projects`` or ``cron_executions``) for a provider that keeps those stores too; ``None`` leaves the store a
        SQLite file in the profile folder."""
        return None

    def compat_store_exists(self, scope: ProfileScope, settings: Mapping[str, Any], store: str) -> bool:
        """Whether the store :meth:`open_compat_store` serves holds any table yet, where a SQLite store's file would
        exist; only asked of a provider that serves file stores."""
        return False

    def move_compat_store(self, scope: ProfileScope, settings: Mapping[str, Any], store: str, new_store: str) -> bool:
        """Rename a file store as its folder moves (a board archived); False when there was nothing to move."""
        return False

    def drop_compat_store(self, scope: ProfileScope, settings: Mapping[str, Any], store: str) -> bool:
        """Delete a file store for good, as its folder is deleted; False when absent."""
        return False

    def change_token(self, scope: ProfileScope, settings: Mapping[str, Any]) -> Optional[str]:
        """A value that changes whenever ``scope``'s store commits a change (and may change more often), for caches
        that would otherwise watch the store's file; ``None`` when the store does not exist yet."""
        return None

    def close_stores_under(self, directory: Path) -> int:
        """Force-close what this provider holds for profiles under ``directory``; returns how many closed."""
        return 0

    def close_all_stores(self) -> int:
        """Close everything this provider holds open in the process; returns how many closed."""
        return 0
