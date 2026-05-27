"""Kanban storage backend selection.

The selector defaults to the existing SQLite implementation so runtime
behavior remains unchanged unless the operator explicitly selects a backend
through ``HERMES_KANBAN_BACKEND`` or ``kanban.storage.backend`` in config.yaml.
"""

from __future__ import annotations

import os
from typing import Any

from hermes_cli.kanban_store import KanbanStore
from hermes_cli.kanban_store_postgres import PostgresKanbanStore
from hermes_cli.kanban_store_sqlite import SQLiteKanbanStore

_BACKEND_ENV = "HERMES_KANBAN_BACKEND"
_DEFAULT_BACKEND = "sqlite"


def normalize_backend_name(value: str | None) -> str:
    """Normalize a configured backend name, defaulting to SQLite."""
    name = (value or _DEFAULT_BACKEND).strip().lower()
    return name or _DEFAULT_BACKEND


def _load_config_backend() -> str | None:
    """Read the optional Kanban storage backend from config.yaml."""
    try:
        from hermes_cli.config import load_config

        cfg: dict[str, Any] = load_config()
        kanban_cfg = cfg.get("kanban") or {}
        storage_cfg = kanban_cfg.get("storage") or {}
        backend = storage_cfg.get("backend") or kanban_cfg.get("storage_backend")
        return str(backend) if backend else None
    except Exception:
        return None


def create_kanban_store(backend: str | None = None) -> KanbanStore:
    """Create a Kanban store for ``backend``."""
    name = normalize_backend_name(backend)
    if name == "sqlite":
        return SQLiteKanbanStore()
    if name in {"postgres", "postgresql"}:
        return PostgresKanbanStore()
    raise ValueError(
        f"Unsupported Kanban store backend {name!r}. "
        "Supported backends: sqlite, postgres"
    )


def get_default_kanban_store() -> KanbanStore:
    """Return the store selected by env/config defaults."""
    return create_kanban_store(os.environ.get(_BACKEND_ENV) or _load_config_backend())


__all__ = [
    "create_kanban_store",
    "get_default_kanban_store",
    "normalize_backend_name",
]
