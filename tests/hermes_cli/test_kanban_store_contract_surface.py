"""Structural contract tests for Kanban store migration call sites."""

from __future__ import annotations

import inspect

import pytest


REQUIRED_CLI_DISPATCHER_OPERATIONS = {
    "add_notify_sub",
    "archive_task",
    "assign_task",
    "board_exists",
    "board_stats",
    "build_worker_context",
    "child_ids",
    "create_board",
    "delete_archived_task",
    "dispatch_once",
    "edit_completed_task_result",
    "gc_events",
    "gc_worker_logs",
    "get_current_board",
    "has_spawnable_ready",
    "heartbeat_worker",
    "kanban_db_path",
    "known_assignees",
    "latest_summary",
    "link_tasks",
    "list_boards",
    "list_notify_subs",
    "list_profiles_on_disk",
    "list_runs",
    "parent_ids",
    "promote_task",
    "read_board_metadata",
    "read_worker_log",
    "reassign_task",
    "reclaim_task",
    "remove_board",
    "remove_notify_sub",
    "resolve_workspace",
    "run_daemon",
    "schedule_task",
    "set_current_board",
    "set_workspace_path",
    "unlink_tasks",
    "workspaces_root",
    "write_board_metadata",
}

REQUIRED_CLI_DISPATCHER_CONSTANTS = {
    "DEFAULT_BOARD",
    "DEFAULT_CLAIM_TTL_SECONDS",
    "DEFAULT_FAILURE_LIMIT",
    "DEFAULT_SPAWN_FAILURE_LIMIT",
    "VALID_INITIAL_STATUSES",
    "VALID_SORT_ORDERS",
    "VALID_STATUSES",
}


def test_store_protocol_declares_cli_and_dispatcher_surface():
    from hermes_cli.kanban_store import KanbanStore

    protocol_attrs = set(KanbanStore.__dict__)

    missing_ops = sorted(REQUIRED_CLI_DISPATCHER_OPERATIONS - protocol_attrs)
    missing_constants = sorted(REQUIRED_CLI_DISPATCHER_CONSTANTS - protocol_attrs)

    assert missing_ops == []
    assert missing_constants == []


def test_sqlite_store_exposes_required_cli_and_dispatcher_surface():
    from hermes_cli.kanban_store_sqlite import SQLiteKanbanStore

    store = SQLiteKanbanStore()

    missing_ops = [name for name in sorted(REQUIRED_CLI_DISPATCHER_OPERATIONS) if not callable(getattr(store, name, None))]
    missing_constants = [name for name in sorted(REQUIRED_CLI_DISPATCHER_CONSTANTS) if not hasattr(store, name)]

    assert missing_ops == []
    assert missing_constants == []


def test_postgres_backend_has_explicit_runtime_adapter(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_BACKEND", "postgres")

    from hermes_cli.kanban_store_factory import create_kanban_store, get_default_kanban_store
    from hermes_cli.kanban_store_postgres import PostgresKanbanStore

    store = create_kanban_store("postgres")

    assert isinstance(store, PostgresKanbanStore)
    assert store.capabilities.backend == "postgres"
    assert store.capabilities.supports_concurrent_writers is True
    assert store.capabilities.supports_skip_locked is True
    assert store.capabilities.production_ready is True
    assert get_default_kanban_store().capabilities.backend == "postgres"


def test_postgres_store_declares_same_contract_methods_as_protocol():
    from hermes_cli.kanban_store import KanbanStore
    from hermes_cli.kanban_store_postgres import PostgresKanbanStore

    protocol_methods = {
        name
        for name, value in KanbanStore.__dict__.items()
        if inspect.isfunction(value) and not name.startswith("__")
    }
    postgres_methods = {
        name
        for name, value in PostgresKanbanStore.__dict__.items()
        if inspect.isfunction(value) and not name.startswith("__")
    }

    assert sorted(protocol_methods - postgres_methods) == []
