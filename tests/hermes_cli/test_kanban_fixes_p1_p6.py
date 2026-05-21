"""Targeted regressions for kanban auto-subscribe and rerun fixes."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_conn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> sqlite3.Connection:
    """Fresh in-memory kanban DB with an isolated Hermes home."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(kb.SCHEMA_SQL)
    kb._migrate_add_optional_columns(conn)
    try:
        yield conn
    finally:
        conn.close()


def test_create_task_with_subscribe_creates_notification_subscription(
    kanban_conn: sqlite3.Connection,
) -> None:
    tid = kb.create_task(
        kanban_conn,
        title="subscribed task",
        assignee="alice",
        subscribe={
            "platform": "telegram",
            "chat_id": "chat-1",
            "thread_id": "thread-7",
            "user_id": "user-9",
        },
    )

    subs = kb.list_notify_subs(kanban_conn, task_id=tid)
    assert len(subs) == 1
    assert subs[0]["platform"] == "telegram"
    assert subs[0]["chat_id"] == "chat-1"
    assert subs[0]["thread_id"] == "thread-7"
    assert subs[0]["user_id"] == "user-9"


def test_create_task_without_subscribe_does_not_create_notification_subscription(
    kanban_conn: sqlite3.Connection,
) -> None:
    tid = kb.create_task(kanban_conn, title="plain task", assignee="alice")

    assert kb.list_notify_subs(kanban_conn, task_id=tid) == []


def test_create_task_auto_subscribes_from_notification_env(
    kanban_conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HERMES_NOTIFY_PLATFORM", "discord")
    monkeypatch.setenv("HERMES_NOTIFY_CHAT_ID", "channel-42")
    monkeypatch.setenv("HERMES_NOTIFY_THREAD_ID", "topic-11")

    tid = kb.create_task(kanban_conn, title="env-subscribed", assignee="alice")

    subs = kb.list_notify_subs(kanban_conn, task_id=tid)
    assert len(subs) == 1
    assert subs[0]["platform"] == "discord"
    assert subs[0]["chat_id"] == "channel-42"
    assert subs[0]["thread_id"] == "topic-11"


def test_rerun_resets_completed_task_to_ready(kanban_conn: sqlite3.Connection) -> None:
    tid = kb.create_task(kanban_conn, title="done task", assignee="alice")
    assert kb.claim_task(kanban_conn, tid) is not None
    assert kb.complete_task(kanban_conn, tid, result="first pass")

    assert kb.rerun_task(kanban_conn, tid, reason="second pass")

    task = kb.get_task(kanban_conn, tid)
    assert task is not None
    assert task.status == "ready"
    assert task.claim_lock is None
    assert task.current_run_id is None
    assert task.completed_at is None
    assert task.consecutive_failures == 0
    events = kb.list_events(kanban_conn, tid)
    assert events[-1].kind == "rerun"
    assert events[-1].payload["reason"] == "second pass"


def test_rerun_resets_blocked_task(kanban_conn: sqlite3.Connection) -> None:
    tid = kb.create_task(kanban_conn, title="blocked task", assignee="alice")
    assert kb.claim_task(kanban_conn, tid) is not None
    assert kb.block_task(kanban_conn, tid, reason="waiting on review")
    kanban_conn.execute(
        "UPDATE tasks SET consecutive_failures = 3, last_failure_error = 'boom' WHERE id = ?",
        (tid,),
    )

    assert kb.rerun_task(kanban_conn, tid, reason="retry after review")

    task = kb.get_task(kanban_conn, tid)
    assert task is not None
    assert task.status == "ready"
    assert task.consecutive_failures == 0
    assert task.last_failure_error is None


def test_rerun_respects_parent_gates(kanban_conn: sqlite3.Connection) -> None:
    parent = kb.create_task(kanban_conn, title="parent", assignee="lead")
    child = kb.create_task(
        kanban_conn,
        title="child",
        assignee="worker",
        parents=[parent],
    )
    assert kb.claim_task(kanban_conn, parent) is not None
    assert kb.complete_task(kanban_conn, parent, result="ready child")
    assert kb.claim_task(kanban_conn, child) is not None
    assert kb.complete_task(kanban_conn, child, result="child complete")

    kanban_conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (parent,))

    assert kb.rerun_task(kanban_conn, child, reason="parent reopened")

    task = kb.get_task(kanban_conn, child)
    assert task is not None
    assert task.status == "todo"


def test_rerun_can_reassign_task(kanban_conn: sqlite3.Connection) -> None:
    tid = kb.create_task(kanban_conn, title="reassign me", assignee="alice")
    assert kb.claim_task(kanban_conn, tid) is not None
    assert kb.complete_task(kanban_conn, tid, result="done")

    assert kb.rerun_task(kanban_conn, tid, new_assignee="bob")

    task = kb.get_task(kanban_conn, tid)
    assert task is not None
    assert task.assignee == "bob"
    assert task.status == "ready"


@pytest.mark.parametrize(
    ("initial_setup", "expected_status"),
    [
        ("ready", "ready"),
        ("running", "running"),
        ("todo", "todo"),
    ],
)
def test_rerun_rejects_non_terminal_tasks(
    kanban_conn: sqlite3.Connection,
    initial_setup: str,
    expected_status: str,
) -> None:
    if initial_setup == "todo":
        parent = kb.create_task(kanban_conn, title="parent", assignee="lead")
        tid = kb.create_task(
            kanban_conn,
            title="todo child",
            assignee="worker",
            parents=[parent],
        )
    else:
        tid = kb.create_task(kanban_conn, title=f"{initial_setup} task", assignee="alice")
        if initial_setup == "running":
            assert kb.claim_task(kanban_conn, tid) is not None

    assert not kb.rerun_task(kanban_conn, tid, reason="should fail")
    assert kb.get_task(kanban_conn, tid).status == expected_status


def test_rerun_preserves_notify_subscriptions_and_advances_cursor(
    kanban_conn: sqlite3.Connection,
) -> None:
    tid = kb.create_task(
        kanban_conn,
        title="notify me",
        assignee="alice",
        subscribe={"platform": "telegram", "chat_id": "chat-9"},
    )
    assert kb.claim_task(kanban_conn, tid) is not None
    assert kb.complete_task(kanban_conn, tid, result="done")
    before = kb.list_notify_subs(kanban_conn, task_id=tid)
    assert len(before) == 1
    assert before[0]["last_event_id"] == 0

    assert kb.rerun_task(kanban_conn, tid, reason="fresh attempt")

    subs = kb.list_notify_subs(kanban_conn, task_id=tid)
    assert len(subs) == 1
    assert subs[0]["platform"] == "telegram"
    assert subs[0]["chat_id"] == "chat-9"
    max_event_id = max(event.id for event in kb.list_events(kanban_conn, tid))
    assert subs[0]["last_event_id"] == max_event_id
    cursor, unseen = kb.unseen_events_for_sub(
        kanban_conn,
        task_id=tid,
        platform="telegram",
        chat_id="chat-9",
    )
    assert cursor == max_event_id
    assert unseen == []


def test_build_worker_context_includes_parent_workspace_path(
    kanban_conn: sqlite3.Connection,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "parent-workspace"
    workspace.mkdir()

    parent = kb.create_task(
        kanban_conn,
        title="parent",
        assignee="researcher",
        workspace_kind="dir",
        workspace_path=str(workspace),
    )
    assert kb.claim_task(kanban_conn, parent) is not None
    assert kb.complete_task(kanban_conn, parent, result="parent result")
    child = kb.create_task(
        kanban_conn,
        title="child",
        assignee="writer",
        parents=[parent],
    )

    ctx = kb.build_worker_context(kanban_conn, child)
    assert f"_Workspace_: `{workspace}`" in ctx


def test_build_worker_context_omits_parent_workspace_when_absent(
    kanban_conn: sqlite3.Connection,
) -> None:
    parent = kb.create_task(kanban_conn, title="parent", assignee="researcher")
    assert kb.claim_task(kanban_conn, parent) is not None
    assert kb.complete_task(kanban_conn, parent, result="parent result")
    child = kb.create_task(
        kanban_conn,
        title="child",
        assignee="writer",
        parents=[parent],
    )

    ctx = kb.build_worker_context(kanban_conn, child)
    assert "_Workspace_:" not in ctx


def test_build_worker_context_omits_removed_parent_workspace(
    kanban_conn: sqlite3.Connection,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "removed-parent-workspace"
    workspace.mkdir()

    parent = kb.create_task(
        kanban_conn,
        title="parent",
        assignee="researcher",
        workspace_kind="dir",
        workspace_path=str(workspace),
    )
    assert kb.claim_task(kanban_conn, parent) is not None
    assert kb.complete_task(kanban_conn, parent, result="parent result")
    workspace.rmdir()
    child = kb.create_task(
        kanban_conn,
        title="child",
        assignee="writer",
        parents=[parent],
    )

    ctx = kb.build_worker_context(kanban_conn, child)
    assert "_Workspace_:" not in ctx


def test_cli_create_auto_subscribes_from_notification_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_NOTIFY_PLATFORM", "slack")
    monkeypatch.setenv("HERMES_NOTIFY_CHAT_ID", "chan-77")
    monkeypatch.setenv("HERMES_NOTIFY_THREAD_ID", "thread-3")
    monkeypatch.setenv("HERMES_NOTIFY_USER_ID", "user-2")
    kb.init_db()

    out = kc.run_slash("create 'cli env subscribe' --assignee alice")
    tid = re.search(r"(t_[a-f0-9]+)", out).group(1)

    with kb.connect() as conn:
        subs = kb.list_notify_subs(conn, task_id=tid)
    assert len(subs) == 1
    assert subs[0]["platform"] == "slack"
    assert subs[0]["chat_id"] == "chan-77"
    assert subs[0]["thread_id"] == "thread-3"
    assert subs[0]["user_id"] == "user-2"
