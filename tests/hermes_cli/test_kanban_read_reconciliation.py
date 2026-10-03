"""Read-path regression coverage for Kanban readiness reconciliation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Use a real isolated board for each read-path scenario."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _create_initial_blocked_task(title: str = "approval gate") -> str:
    """Create a task with initial_status='blocked' via public API."""
    with kbc.connect() as conn:
        return kb.create_task(conn, title=title, initial_status="blocked")


def _create_legacy_blocked_task(title: str = "legacy gate") -> str:
    """Construct a legacy / non-sticky blocked row without a ``blocked`` event.

    Simulates pre-#109334 rows or direct DB triage where a card has status
    'blocked' but no sticky 'blocked' event in ``task_events``. Because it
    lacks a sticky block event, default ``recompute_ready()`` would auto-promote
    it to 'ready'. Testing read surfaces against this guarantees that the
    protection comes from ``include_blocked=False`` rather than from sticky
    event checking (which passes automatically once #109334 lands).
    """
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title=title)
        conn.execute("UPDATE tasks SET status = 'blocked' WHERE id = ?", (task_id,))
        conn.commit()
        return task_id


def _task_state(task_id: str) -> tuple[str, list[str]]:
    with kbc.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        return task.status, [event.kind for event in kb.list_events(conn, task_id)]


def test_cli_list_does_not_recover_legacy_blocked_tasks(kanban_home: Path) -> None:
    """A CLI list must not recover legacy/non-sticky blocked rows.

    This directly tests the ``include_blocked=False`` read-surface filter
    against rows that do not have a sticky block event in task_events.
    """
    task_id = _create_legacy_blocked_task("legacy cli gate")

    payload = json.loads(kc.run_slash("list --json --status blocked"))

    assert [task["id"] for task in payload] == [task_id]
    assert payload[0]["status"] == "blocked"
    status, events = _task_state(task_id)
    assert status == "blocked"
    assert "promoted" not in events


def test_tool_list_does_not_recover_legacy_blocked_tasks(kanban_home: Path) -> None:
    """The orchestrator list tool must not recover legacy/non-sticky blocked rows."""
    task_id = _create_legacy_blocked_task("legacy tool gate")

    from tools import kanban_tools as kt

    payload = json.loads(kt._handle_list({"status": "blocked", "limit": 10}))

    assert payload["promoted"] == 0
    assert [task["id"] for task in payload["tasks"]] == [task_id]
    status, events = _task_state(task_id)
    assert status == "blocked"
    assert "promoted" not in events


def test_cli_list_does_not_recover_initial_blocked_tasks(kanban_home: Path) -> None:
    """A CLI list must not release a task created with initial_status='blocked'."""
    task_id = _create_initial_blocked_task()

    payload = json.loads(kc.run_slash("list --json --status blocked"))

    assert [task["id"] for task in payload] == [task_id]
    assert payload[0]["status"] == "blocked"
    status, events = _task_state(task_id)
    assert status == "blocked"
    assert "promoted" not in events


def test_cli_list_keeps_todo_reconciliation(kanban_home: Path) -> None:
    """Read-time todo reconciliation remains available for compatibility."""
    with kbc.connect() as conn:
        parent_id = kb.create_task(conn, title="parent")
        child_id = kb.create_task(conn, title="dependent", parents=[parent_id])
        conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (parent_id,))
        conn.commit()

    payload = json.loads(kc.run_slash("list --json"))
    listed = next(task for task in payload if task["id"] == child_id)

    assert listed["status"] == "ready"
    with kbc.connect() as conn:
        assert kb.get_task(conn, child_id).status == "ready"


def test_read_skips_circuit_breaker_recovery_contract(kanban_home: Path) -> None:
    """Document intentional trade-off: reads do not recover transient circuit-breaker blocks.

    Even if consecutive_failures < failure_limit (which default recompute_ready
    auto-recovers), list reads treat all blocked states as stable and observational.
    In deployments using list without a background dispatcher, circuit-breaker
    cards do not auto-recover on read; recovery is deferred to dispatch/lifecycle.
    """
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="circuit-breaker candidate")
        conn.execute(
            "UPDATE tasks SET status = 'blocked', consecutive_failures = 1, "
            "last_failure_error = 'transient timeout' WHERE id = ?",
            (task_id,),
        )
        conn.commit()

    # CLI read preserves blocked state
    payload = json.loads(kc.run_slash("list --json --status blocked"))
    assert [task["id"] for task in payload] == [task_id]
    status, events = _task_state(task_id)
    assert status == "blocked"
    assert "promoted" not in events

    # Tool read preserves blocked state and reports promoted=0
    from tools import kanban_tools as kt

    tool_payload = json.loads(kt._handle_list({"status": "blocked", "limit": 10}))
    assert tool_payload["promoted"] == 0
    status, events = _task_state(task_id)
    assert status == "blocked"
    assert "promoted" not in events

    # In contrast, default recompute_ready (run by dispatcher) recovers it
    with kbc.connect() as conn:
        promoted = kb.recompute_ready(conn)
        assert promoted == 1
        task = kb.get_task(conn, task_id)
        assert task.status == "ready"
        assert task.consecutive_failures == 1


def test_dispatch_recompute_keeps_nonsticky_block_recovery(kanban_home: Path) -> None:
    """The dispatcher-facing default still recovers a transient block."""
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="transient block")
        conn.execute("UPDATE tasks SET status = 'blocked' WHERE id = ?", (task_id,))
        conn.commit()

        assert kb.recompute_ready(conn) == 1
        assert kb.get_task(conn, task_id).status == "ready"
