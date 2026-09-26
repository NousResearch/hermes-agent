"""Budget displays are read-only snapshots of the existing attempt deadline."""

import os
from contextlib import closing
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def attempt(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    with closing(kbc.connect()) as conn:
        tid = kb.create_task(conn, title="Budget example", assignee="worker",
                             body="Preserve the ordinary task body.",
                             max_runtime_seconds=120)
        assert kb.claim_task(conn, tid) is not None
        run_id = kb.get_task(conn, tid).current_run_id
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET started_at=1000 WHERE id=?", (tid,))
            conn.execute("UPDATE task_runs SET started_at=2000 WHERE id=?", (run_id,))
        yield conn, tid, run_id


@pytest.mark.parametrize(
    "changes,state,remaining,basis",
    [
        ({}, "active", 70, "current_run"),
        ({"now": 2120}, "exhausted", 0, "current_run"),
        ({"now": 2150}, "exhausted", 0, "current_run"),
        ({"cap": 0}, "exhausted", 0, "current_run"),
        ({"cap": -1}, "exhausted", 0, "current_run"),
        ({"cap": None}, "unbounded", None, None),
        ({"status": "ready"}, "not_running", None, None),
        ({"status": "done"}, "not_running", None, None),
        ({"run": None, "task_start": 2000}, "active", 70, "task_fallback"),
        ({"run": 999999, "task_start": 2000}, "active", 70, "task_fallback"),
        ({"run": None, "task_start": None}, "unknown", None, None),
        ({"run_start": "invalid"}, "unknown", None, None),
        ({"cap": "invalid"}, "unknown", None, None),
        ({"now": 1999}, "unknown", None, "current_run"),
        ({"ended": 2020}, "unknown", None, None),
        ({"foreign_run": True}, "unknown", None, None),
    ],
)
def test_snapshot_matches_attempt_without_mutating(attempt, changes, state, remaining, basis):
    from hermes_cli.kanban_runtime_budget import runtime_budget_snapshot

    conn, tid, run_id = attempt
    other_task = (kb.create_task(conn, title="Other fixture", assignee="worker")
                  if changes.get("foreign_run") else None)
    task_columns = {"cap": "max_runtime_seconds", "status": "status",
                    "run": "current_run_id", "task_start": "started_at"}
    with kb.write_txn(conn):
        for key, column in task_columns.items():
            if key in changes:
                conn.execute(f"UPDATE tasks SET {column}=? WHERE id=?", (changes[key], tid))
        if "run_start" in changes:
            conn.execute("UPDATE task_runs SET started_at=? WHERE id=?", (changes["run_start"], run_id))
        if "ended" in changes:
            conn.execute("UPDATE task_runs SET ended_at=? WHERE id=?", (changes["ended"], run_id))
        if changes.get("foreign_run"):
            # Deliberately corrupt only the disposable fixture's run binding.
            conn.execute("UPDATE task_runs SET task_id=? WHERE id=?", (other_task, run_id))
    before = list(conn.iterdump())
    snapshot = runtime_budget_snapshot(conn, tid, observed_at=changes.get("now", 2050))
    assert snapshot.state == state
    assert snapshot.remaining_seconds == remaining
    assert snapshot.start_basis == basis
    assert snapshot.task_id == tid
    if state in {"active", "exhausted"}:
        assert snapshot.deadline_at == snapshot.started_at + snapshot.limit_seconds
        assert snapshot.elapsed_seconds == snapshot.observed_at - snapshot.started_at
        assert remaining == max(0, snapshot.deadline_at - snapshot.observed_at)
    assert list(conn.iterdump()) == before
    with pytest.raises(ValueError, match="unknown task"):
        runtime_budget_snapshot(conn, "missing-task", observed_at=2050)


def test_context_snapshot_and_heartbeat_preserve_hard_cutoff(attempt, monkeypatch):
    conn, tid, run_id = attempt
    calls = []

    def clock():
        calls.append(2050)
        return 2050

    with monkeypatch.context() as frozen:
        frozen.setattr(kb.time, "time", clock)
        context = kb.build_worker_context(conn, tid)
        assert len(calls) == 1
    assert "Runtime budget snapshot:" in context
    assert "70s remaining" in context
    assert f"run={run_id}" in context
    assert "1970-01-01T00:35:20Z" in context  # 2000 + 120, not original task start.
    assert "Max runtime: 120s" in context
    assert "Preserve the ordinary task body." in context

    from hermes_cli.kanban_runtime_budget import runtime_budget_snapshot

    before = runtime_budget_snapshot(conn, tid, observed_at=2050)
    monkeypatch.setattr(kb.time, "time", lambda: 2050)
    assert kb.heartbeat_claim(conn, tid, ttl_seconds=600)
    assert runtime_budget_snapshot(conn, tid, observed_at=2050) == before
    kbd._set_worker_pid(conn, tid, os.getpid())
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)
    signals = []
    assert kbd.enforce_max_runtime(conn, signal_fn=lambda *args: signals.append(args)) == []
    assert signals == []
    monkeypatch.setattr(kb.time, "time", lambda: before.deadline_at)
    assert kbd.enforce_max_runtime(conn, signal_fn=lambda *args: signals.append(args)) == [tid]
    assert signals
    event = next(e for e in kb.list_events(conn, tid) if e.kind == "timed_out")
    assert event.payload["elapsed_seconds"] == event.payload["limit_seconds"] == 120
