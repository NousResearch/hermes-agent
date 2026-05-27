"""Factory tests for selecting the Kanban storage backend."""

from __future__ import annotations

import importlib
import sys

import pytest


def test_default_kanban_store_is_sqlite_when_backend_unset(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_BACKEND", raising=False)

    from hermes_cli.kanban_store_factory import get_default_kanban_store
    from hermes_cli.kanban_store_sqlite import SQLiteKanbanStore

    store = get_default_kanban_store()

    assert isinstance(store, SQLiteKanbanStore)
    assert store.capabilities.backend == "sqlite"


def test_default_kanban_store_accepts_explicit_sqlite_backend(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_BACKEND", "sqlite")

    from hermes_cli.kanban_store_factory import get_default_kanban_store
    from hermes_cli.kanban_store_sqlite import SQLiteKanbanStore

    assert isinstance(get_default_kanban_store(), SQLiteKanbanStore)


def test_default_kanban_store_rejects_unknown_backend(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_BACKEND", "postgres")

    from hermes_cli.kanban_store_factory import get_default_kanban_store

    with pytest.raises(ValueError, match="Unsupported Kanban store backend"):
        get_default_kanban_store()


def test_kanban_cli_module_uses_selected_store_boundary(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_BACKEND", raising=False)
    sys.modules.pop("hermes_cli.kanban", None)

    kanban = importlib.import_module("hermes_cli.kanban")

    assert kanban.kb.capabilities.backend == "sqlite"
    assert kanban.kb.DEFAULT_SPAWN_FAILURE_LIMIT > 0
