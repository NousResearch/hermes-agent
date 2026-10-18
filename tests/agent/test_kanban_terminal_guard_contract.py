#!/usr/bin/env python3
"""TDD: the kanban exit reconciliation must run when a worker ends without a terminal call.

These tests define the contract for `agent/kanban_terminal_guard.py`:

  1. It is a no-op when HERMES_KANBAN_TASK is unset (never touch non-worker runs).
  2. It does nothing when the worker already called kanban_complete/kanban_block.
  3. It records a `protocol_violation` event when the worker is still `running`
     and its own exit path never reached the in-loop nudge (the 27 early returns).
  4. It never raises — a guard failure must not swallow the worker's exit code.
  5. It reports which action it took so the caller can log it.
"""
import os

import pytest


@pytest.fixture
def clean_env(monkeypatch):
    for var in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD",
                "KANBAN_EXIT_RECONCILE"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_noop_without_task_env(clean_env):
    from agent.kanban_terminal_guard import reconcile_worker_exit
    clean_env.delenv("HERMES_KANBAN_TASK", raising=False)
    assert reconcile_worker_exit(completed=False, result=None) == "skipped"


def test_noop_when_disabled(clean_env):
    from agent.kanban_terminal_guard import reconcile_worker_exit
    clean_env.setenv("HERMES_KANBAN_TASK", "t_test")
    clean_env.setenv("KANBAN_EXIT_RECONCILE", "0")
    assert reconcile_worker_exit(completed=False, result=None) == "skipped"


def test_noop_when_session_already_terminal(clean_env):
    """If the conversation history already shows a terminal kanban tool call, do nothing."""
    from agent.kanban_terminal_guard import reconcile_worker_exit
    clean_env.setenv("HERMES_KANBAN_TASK", "t_test")
    result = {
        "messages": [
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "kanban_complete", "arguments": "{}"}}]},
            {"role": "tool", "name": "kanban_complete", "content": "ok"},
        ],
    }
    assert reconcile_worker_exit(completed=True, result=result) == "skipped"


def test_records_violation_when_still_running(clean_env, tmp_path):
    """The core case: worker still running, no terminal call -> record the event."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from agent.kanban_terminal_guard import reconcile_worker_exit

    db = tmp_path / "kanban.db"
    clean_env.setenv("HERMES_KANBAN_DB", str(db))
    kbc.init_db(db)
    with kbc.connect_closing(db) as conn:
        task_id = kb.create_task(conn, title="probe", assignee="worker-test")
        conn.execute("UPDATE tasks SET status='running' WHERE id=?", (task_id,))
        conn.commit()

    clean_env.setenv("HERMES_KANBAN_TASK", task_id)
    result = {"messages": [{"role": "assistant", "content": "I will write the report now."}]}
    assert reconcile_worker_exit(completed=False, result=result) == "recorded"


def test_never_raises(clean_env):
    """A broken DB must not break the worker's exit."""
    from agent.kanban_terminal_guard import reconcile_worker_exit
    clean_env.setenv("HERMES_KANBAN_TASK", "t_test")
    clean_env.setenv("HERMES_KANBAN_DB", "/nonexistent/dir/kanban.db")
    # must return a string, not raise
    assert isinstance(reconcile_worker_exit(completed=False, result={}), str)


def test_returns_action_string(clean_env):
    from agent.kanban_terminal_guard import reconcile_worker_exit
    clean_env.setenv("HERMES_KANBAN_TASK", "t_test")
    out = reconcile_worker_exit(completed=False, result={"messages": []})
    assert out in {"skipped", "recorded", "not_running", "error"}
