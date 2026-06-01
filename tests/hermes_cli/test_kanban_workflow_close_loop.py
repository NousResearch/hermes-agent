"""Tests for workflow_key column, list_tasks_by_workflow_key, and auto-subscribe.

Covers the close-loop workflow changes made for Kanban orchestrator->worker
notification wiring.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # init_db() creates the default board's DB and runs all migrations
    kb.init_db()
    return home


@pytest.fixture
def kb_conn(kanban_home):
    """A fresh connected kanban DB connection with schema + migrations applied."""
    with kb.connect() as conn:
        yield conn


# ── workflow_key migration & storage ─────────────────────────────────────

def test_workflow_key_column_exists(kb_conn):
    """Migration adds the workflow_key column to the tasks table."""
    cols = {row["name"] for row in kb_conn.execute("PRAGMA table_info(tasks)")}
    assert "workflow_key" in cols, (
        "workflow_key column should exist after init_db/migration"
    )


def test_create_task_stores_workflow_key(kb_conn):
    """create_task accepts and persists workflow_key."""
    tid = kb.create_task(
        kb_conn,
        title="Test workflow task",
        assignee="test-worker",
        workflow_key="wf-close-loop-test",
    )
    row = kb_conn.execute(
        "SELECT workflow_key FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row is not None
    assert row["workflow_key"] == "wf-close-loop-test"


def test_task_object_exposes_workflow_key(kb_conn):
    """Task.from_row carries workflow_key for CLI/API consumers."""
    tid = kb.create_task(
        kb_conn,
        title="Task object workflow",
        assignee="test-worker",
        workflow_key="wf-task-object",
    )

    task = kb.get_task(kb_conn, tid)

    assert task is not None
    assert task.workflow_key == "wf-task-object"


def test_create_task_workflow_key_none_by_default(kb_conn):
    """Tasks created without workflow_key get NULL."""
    tid = kb.create_task(
        kb_conn,
        title="No workflow",
        assignee="test-worker",
    )
    row = kb_conn.execute(
        "SELECT workflow_key FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row is not None
    assert row["workflow_key"] is None


def test_create_task_workflow_key_empty(kb_conn):
    """Empty/whitespace workflow_key is passed through and stored (handler strips it)."""
    tid = kb.create_task(
        kb_conn,
        title="Empty workflow",
        assignee="test-worker",
        workflow_key="   ",
    )
    row = kb_conn.execute(
        "SELECT workflow_key FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row is not None
    # Whitespace string stored as-is by create_task; tool handler strips.
    assert row["workflow_key"] == "   "


# ── list_tasks_by_workflow_key ────────────────────────────────────────────

def test_list_tasks_by_workflow_key_filters_correctly(kb_conn):
    """Only tasks with the given workflow_key are returned."""
    t1 = kb.create_task(kb_conn, title="A", assignee="w", workflow_key="wf-alpha")
    t2 = kb.create_task(kb_conn, title="B", assignee="w", workflow_key="wf-alpha")
    kb.create_task(kb_conn, title="C", assignee="w", workflow_key="wf-beta")
    kb.create_task(kb_conn, title="D", assignee="w")  # no workflow_key

    results = kb.list_tasks_by_workflow_key(kb_conn, "wf-alpha")
    ids = {t.id for t in results}
    assert ids == {t1, t2}


def test_list_tasks_by_workflow_key_empty_key_raises(kb_conn):
    """Empty string raises ValueError."""
    with pytest.raises(ValueError, match="workflow_key is required"):
        kb.list_tasks_by_workflow_key(kb_conn, "")
    with pytest.raises(ValueError, match="workflow_key is required"):
        kb.list_tasks_by_workflow_key(kb_conn, "   ")


def test_list_tasks_by_workflow_key_nonexistent_returns_empty(kb_conn):
    """No tasks with the key -> empty list, not an error."""
    results = kb.list_tasks_by_workflow_key(kb_conn, "no-such-key")
    assert results == []


def test_list_tasks_filter_with_workflow_key(kb_conn):
    """The base list_tasks function filters by workflow_key."""
    t1 = kb.create_task(kb_conn, title="X", assignee="w", workflow_key="wf-x")
    t2 = kb.create_task(kb_conn, title="Y", assignee="w", workflow_key="wf-x")
    kb.create_task(kb_conn, title="Z", assignee="w")

    results = kb.list_tasks(kb_conn, workflow_key="wf-x")
    ids = {t.id for t in results}
    assert ids == {t1, t2}


# ── auto-subscribe from kanban_create tool ────────────────────────────────

def test_auto_subscribe_gateway_source_no_env_is_noop(kb_conn, monkeypatch):
    """When gateway env vars are not set, _auto_subscribe_gateway_source
    is a silent no-op (no subscription created)."""
    monkeypatch.delenv("HERMES_KANBAN_SUB_PLATFORM", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_SUB_CHAT_ID", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_SUB_THREAD_ID", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_SUB_USER_ID", raising=False)

    import hermes_cli.config as hermes_config
    from tools.kanban_tools import _auto_subscribe_gateway_source

    monkeypatch.setattr(hermes_config, "load_config", lambda: {})
    tid = kb.create_task(kb_conn, title="T", assignee="w")
    result = _auto_subscribe_gateway_source(kb, kb_conn, tid)

    subs = kb.list_notify_subs(kb_conn)
    # No env vars -> no subscription
    assert len(subs) == 0
    assert result["reason"] == "no_gateway_source"


def test_auto_subscribe_home_channel_when_enabled(kb_conn, monkeypatch):
    """Local/TUI orchestrator-created tasks can opt into home notifications."""
    monkeypatch.delenv("HERMES_KANBAN_SUB_PLATFORM", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_SUB_CHAT_ID", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_SUB_THREAD_ID", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_SUB_USER_ID", raising=False)
    monkeypatch.setenv("HERMES_PROFILE", "orchestrator")

    import hermes_cli.config as hermes_config
    from tools import kanban_tools

    monkeypatch.setattr(
        hermes_config,
        "load_config",
        lambda: {
            "kanban": {
                "auto_subscribe_home_on_create": True,
                "auto_subscribe_home_platforms": ["telegram"],
            }
        },
    )
    monkeypatch.setattr(
        kanban_tools,
        "_configured_home_channels",
        lambda: [
            {"platform": "telegram", "chat_id": "123456", "thread_id": "789"},
            {"platform": "discord", "chat_id": "ignored", "thread_id": ""},
        ],
    )

    tid = kb.create_task(kb_conn, title="T", assignee="w")
    result = kanban_tools._auto_subscribe_gateway_source(kb, kb_conn, tid)

    assert result["subscribed"] is True
    assert result["source"] == "home_channel"
    assert result["channels"] == ["telegram"]
    subs = kb.list_notify_subs(kb_conn, task_id=tid)
    assert len(subs) == 1
    assert subs[0]["platform"] == "telegram"
    assert subs[0]["chat_id"] == "123456"
    assert subs[0]["thread_id"] == "789"
    assert subs[0]["notifier_profile"] == "orchestrator"


def test_auto_subscribe_gateway_source_creates_sub(kb_conn, monkeypatch):
    """With correct env vars, a subscription row is created."""
    monkeypatch.setenv("HERMES_KANBAN_SUB_PLATFORM", "telegram")
    monkeypatch.setenv("HERMES_KANBAN_SUB_CHAT_ID", "123456")
    monkeypatch.setenv("HERMES_KANBAN_SUB_THREAD_ID", "789")
    monkeypatch.setenv("HERMES_KANBAN_SUB_USER_ID", "user42")
    monkeypatch.setenv("HERMES_PROFILE", "orch-profile")

    from tools.kanban_tools import _auto_subscribe_gateway_source

    tid = kb.create_task(kb_conn, title="T", assignee="w")
    _auto_subscribe_gateway_source(kb, kb_conn, tid)

    subs = kb.list_notify_subs(kb_conn, task_id=tid)
    assert len(subs) == 1
    sub = subs[0]
    assert sub["platform"] == "telegram"
    assert sub["chat_id"] == "123456"
    assert sub["thread_id"] == "789"
    assert sub["user_id"] == "user42"
    # Idempotent — second call does not create a duplicate
    _auto_subscribe_gateway_source(kb, kb_conn, tid)
    subs2 = kb.list_notify_subs(kb_conn, task_id=tid)
    assert len(subs2) == 1


def test_auto_subscribe_gateway_source_accepts_direct_context(kb_conn, monkeypatch):
    """Gateway context can be passed without process-global env mutation."""
    monkeypatch.delenv("HERMES_KANBAN_SUB_PLATFORM", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_SUB_CHAT_ID", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_SUB_THREAD_ID", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_SUB_USER_ID", raising=False)
    monkeypatch.setenv("HERMES_PROFILE", "orch-profile")
    # Stale optional env values must not leak into direct gateway_source
    # context; this protects concurrent gateway sessions from cross-talk.
    monkeypatch.setenv("HERMES_KANBAN_SUB_THREAD_ID", "stale-thread")

    from tools.kanban_tools import _auto_subscribe_gateway_source

    tid = kb.create_task(kb_conn, title="T", assignee="w")
    _auto_subscribe_gateway_source(
        kb,
        kb_conn,
        tid,
        {
            "platform": "whatsapp",
            "chat_id": "chat@example.invalid",
            "thread_id": None,
            "user_id": "user99",
        },
    )

    subs = kb.list_notify_subs(kb_conn, task_id=tid)
    assert len(subs) == 1
    sub = subs[0]
    assert sub["platform"] == "whatsapp"
    assert sub["chat_id"] == "chat@example.invalid"
    assert sub["thread_id"] == ""
    assert sub["user_id"] == "user99"


def test_auto_subscribe_idempotent_across_boards(kb_conn, monkeypatch):
    """Subscription uses INSERT OR IGNORE — second call is harmless."""
    monkeypatch.setenv("HERMES_KANBAN_SUB_PLATFORM", "discord")
    monkeypatch.setenv("HERMES_KANBAN_SUB_CHAT_ID", "abc")

    from tools.kanban_tools import _auto_subscribe_gateway_source

    tid = kb.create_task(kb_conn, title="T", assignee="w")
    _auto_subscribe_gateway_source(kb, kb_conn, tid)
    _auto_subscribe_gateway_source(kb, kb_conn, tid)
    _auto_subscribe_gateway_source(kb, kb_conn, tid)

    subs = kb.list_notify_subs(kb_conn, task_id=tid)
    assert len(subs) == 1
