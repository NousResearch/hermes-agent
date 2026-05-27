"""Kanban storage backend selection.

The selector defaults to the existing SQLite implementation so runtime
behavior remains unchanged while call sites move behind the store boundary.
"""

from __future__ import annotations

import os

from hermes_cli.kanban_store import KanbanStore
from hermes_cli.kanban_store_sqlite import SQLiteKanbanStore

_BACKEND_ENV = "HERMES_KANBAN_BACKEND"
_DEFAULT_BACKEND = "sqlite"


def normalize_backend_name(value: str | None) -> str:
    """Normalize a configured backend name, defaulting to SQLite."""
    name = (value or _DEFAULT_BACKEND).strip().lower()
    return name or _DEFAULT_BACKEND


def create_kanban_store(backend: str | None = None) -> KanbanStore:
    """Create a Kanban store for ``backend``.

    Only SQLite is wired today.  Future backends should be added here and pass
    the shared store contract before any caller switches to them.
    """
    name = normalize_backend_name(backend)
    if name == "sqlite":
        return SQLiteKanbanStore()
    raise ValueError(
        f"Unsupported Kanban store backend {name!r}. "
        "Supported backends: sqlite"
    )


def get_default_kanban_store() -> KanbanStore:
    """Return the store selected by environment/config defaults."""
    return create_kanban_store(os.environ.get(_BACKEND_ENV))


__all__ = [
    "create_kanban_store",
    "get_default_kanban_store",
    "normalize_backend_name",
]
