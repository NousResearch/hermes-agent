#!/usr/bin/env python3
"""Contract for `agent/kanban_terminal_guard.py`.

The board is the authority: a worker that is still `running` at process exit
gets a `protocol_violation` even if it *attempted* a terminal kanban call that
did not land. A worker whose task left `running` is left alone. The guard never
raises, and it reports `error` when its write did not commit instead of
claiming `recorded`.
"""
import contextlib
import json

import pytest


@pytest.fixture
def clean_env(monkeypatch):
    for var in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD",
                "KANBAN_EXIT_RECONCILE"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def _make_task(tmp_path, monkeypatch, status):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    db = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db))
    kbc.init_db(db)
    with kbc.connect_closing(db) as conn:
        task_id = kb.create_task(conn, title="probe", assignee="worker-test")
        conn.execute("UPDATE tasks SET status=? WHERE id=?", (status, task_id))
        conn.commit()
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    return db, task_id


def _events(db, task_id):
    import sqlite3

    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(
            "SELECT kind, payload FROM task_events WHERE task_id=? ORDER BY id",
            (task_id,),
        ).fetchall()
    finally:
        conn.close()
    return [{"kind": kind, "payload": json.loads(payload) if payload else None}
            for kind, payload in rows]


def test_noop_without_task_env(clean_env):
    from agent.kanban_terminal_guard import reconcile_worker_exit
    clean_env.delenv("HERMES_KANBAN_TASK", raising=False)
    assert reconcile_worker_exit(completed=False, result=None) == "skipped"


def test_noop_when_disabled(clean_env):
    from agent.kanban_terminal_guard import reconcile_worker_exit
    clean_env.setenv("HERMES_KANBAN_TASK", "t_test")
    clean_env.setenv("KANBAN_EXIT_RECONCILE", "0")
    assert reconcile_worker_exit(completed=False, result=None) == "skipped"


def test_noop_when_task_left_running(clean_env, tmp_path):
    """A landed terminal call moves the task; the funnel must not double-report."""
    from agent.kanban_terminal_guard import reconcile_worker_exit

    _make_task(tmp_path, clean_env, "completed")
    result = {
        "messages": [
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "kanban_complete", "arguments": "{}"}}]},
            {"role": "tool", "name": "kanban_complete", "content": "ok"},
        ],
    }
    assert reconcile_worker_exit(completed=True, result=result) == "not_running"


def test_records_violation_when_still_running(clean_env, tmp_path):
    """The core case: worker still running, no terminal call -> record the event."""
    from agent.kanban_terminal_guard import reconcile_worker_exit

    db, task_id = _make_task(tmp_path, clean_env, "running")
    result = {"messages": [{"role": "assistant", "content": "I will write the report now."}]}
    assert reconcile_worker_exit(completed=False, result=result) == "recorded"

    events = [e for e in _events(db, task_id) if e["kind"] == "protocol_violation"]
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["exit_reconcile"] is True
    assert payload["session_called_terminal"] is False
    assert payload["completed_flag"] is False
    assert "without a terminal kanban call" in payload["reason"]


def test_attempted_terminal_call_that_did_not_land_is_recorded(clean_env, tmp_path):
    """An attempted (but unlanded) terminal call must not suppress reconciliation."""
    from agent.kanban_terminal_guard import reconcile_worker_exit

    db, task_id = _make_task(tmp_path, clean_env, "running")
    result = {
        "messages": [
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "kanban_complete", "arguments": "{}"}}]},
        ],
    }
    assert reconcile_worker_exit(completed=False, result=result) == "recorded"

    events = [e for e in _events(db, task_id) if e["kind"] == "protocol_violation"]
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["session_called_terminal"] is True
    assert "did not land" in payload["reason"]


def test_fallback_connection_when_connect_closing_unavailable(clean_env, tmp_path, monkeypatch):
    """Older runtime layout: `kanban_db.connect`, no `kanban_db_connect`."""
    import sqlite3

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from agent.kanban_terminal_guard import reconcile_worker_exit

    db, task_id = _make_task(tmp_path, clean_env, "running")
    monkeypatch.delattr(kbc, "connect_closing", raising=False)

    closed = []

    class _Proxy:
        def __init__(self, real):
            self._real = real

        def __getattr__(self, name):
            return getattr(self._real, name)

        def close(self):
            closed.append(True)
            self._real.close()

    monkeypatch.setattr(kb, "connect", lambda: _Proxy(sqlite3.connect(db)), raising=False)
    assert reconcile_worker_exit(completed=False, result={"messages": []}) == "recorded"
    assert closed == [True]
    assert any(e["kind"] == "protocol_violation" for e in _events(db, task_id))


def test_commit_failure_is_error_not_recorded(clean_env, tmp_path, monkeypatch):
    """A failed commit must not be reported as a recorded violation."""
    import agent.kanban_terminal_guard as guard

    _make_task(tmp_path, clean_env, "running")

    class _FakeConn:
        rowcount = 1

        def execute(self, *args, **kwargs):
            return self

        def fetchone(self):
            return {"status": "running"}

        def commit(self):
            raise RuntimeError("disk full")

    @contextlib.contextmanager
    def _fake_board():
        yield _FakeConn()

    monkeypatch.setattr(guard, "_board_connection", _fake_board)
    assert guard.reconcile_worker_exit(completed=False, result={"messages": []}) == "error"


def test_never_raises(clean_env):
    """A broken DB must not break the worker's exit."""
    from agent.kanban_terminal_guard import reconcile_worker_exit
    clean_env.setenv("HERMES_KANBAN_TASK", "t_test")
    clean_env.setenv("HERMES_KANBAN_DB", "/nonexistent/dir/kanban.db")
    assert isinstance(reconcile_worker_exit(completed=False, result={}), str)


def test_returns_action_string(clean_env, tmp_path):
    from agent.kanban_terminal_guard import reconcile_worker_exit
    _make_task(tmp_path, clean_env, "running")
    out = reconcile_worker_exit(completed=False, result={"messages": []})
    assert out in {"skipped", "recorded", "not_running", "error"}


def test_status_change_before_append_does_not_record(clean_env, tmp_path, monkeypatch):
    """A reclaim that lands before the write must win: no violation is persisted."""
    import sqlite3

    import agent.kanban_terminal_guard as guard

    db, task_id = _make_task(tmp_path, clean_env, "running")

    class _RacingConn:
        def __init__(self, real):
            self._real = real

        def __getattr__(self, name):
            return getattr(self._real, name)

        def execute(self, sql, *args):
            if "INSERT INTO task_events" in sql and "status = 'running'" in sql:
                # Simulate the dispatcher reclaim committing in the check/write window.
                other = sqlite3.connect(db)
                try:
                    other.execute(
                        "UPDATE tasks SET status='completed' WHERE id=?", (task_id,)
                    )
                    other.commit()
                finally:
                    other.close()
            return self._real.execute(sql, *args)

    @contextlib.contextmanager
    def _racing_board():
        conn = sqlite3.connect(db)
        try:
            yield _RacingConn(conn)
        finally:
            conn.close()

    monkeypatch.setattr(guard, "_board_connection", _racing_board)
    assert guard.reconcile_worker_exit(completed=False, result={"messages": []}) == "not_running"
    assert not [e for e in _events(db, task_id) if e["kind"] == "protocol_violation"]


def test_attempted_terminal_call_with_error_keeps_both_facts(clean_env, tmp_path):
    """An error reason must not erase that the terminal call did not land."""
    from agent.kanban_terminal_guard import reconcile_worker_exit

    db, task_id = _make_task(tmp_path, clean_env, "running")
    result = {
        "error": "truncated tool-call arguments",
        "messages": [
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "kanban_block", "arguments": "{}"}}]},
        ],
    }
    assert reconcile_worker_exit(completed=False, result=result) == "recorded"
    events = [e for e in _events(db, task_id) if e["kind"] == "protocol_violation"]
    reason = events[0]["payload"]["reason"]
    assert "truncated tool-call arguments" in reason
    assert "did not land" in reason
