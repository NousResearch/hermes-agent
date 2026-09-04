"""Regression tests for task_run rows left running after task handoff.

A task transition can already be terminal/blocked/reclaimed while its historical
run row remains ``status='running'``.  Card-level recovery cannot see that debt
because the task itself is no longer running.  The supported reconciliation
must close only detached runs, preserve the task state, and append an audit
event in the same transaction.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def conn(tmp_path: Path):
    db = kb.connect(tmp_path / "kanban.db")
    try:
        yield db
    finally:
        db.close()


def _claim(conn, title: str = "work"):
    task_id = kb.create_task(conn, title=title, assignee="worker")
    claimed = kb.claim_task(conn, task_id, claimer=f"{kb._claimer_id().split(':', 1)[0]}:test")
    assert claimed is not None and claimed.current_run_id is not None
    return task_id, claimed.current_run_id


def test_reconciles_running_run_detached_from_blocked_task_with_audit(conn) -> None:
    task_id, run_id = _claim(conn)
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status='blocked', current_run_id=NULL, "
            "claim_lock=NULL, claim_expires=NULL, worker_pid=NULL WHERE id=?",
            (task_id,),
        )

    assert kb.reconcile_stale_task_runs(conn) == [run_id]

    task = kb.get_task(conn, task_id)
    assert task is not None and task.status == "blocked"
    run = conn.execute("SELECT * FROM task_runs WHERE id=?", (run_id,)).fetchone()
    assert run["status"] == "reconciled"
    assert run["outcome"] == "reconciled"
    assert run["ended_at"] is not None
    events = [event for event in kb.list_events(conn, task_id) if event.kind == "run_reconciled"]
    assert len(events) == 1
    assert events[0].run_id == run_id
    assert events[0].payload["task_status"] == "blocked"
    assert events[0].payload["reason"] == "task_not_running"


def test_reconciles_superseded_running_run_without_touching_current_run(conn) -> None:
    task_id, old_run_id = _claim(conn)
    with kb.write_txn(conn):
        current = conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, started_at) "
            "VALUES (?, 'worker', 'running', 1)",
            (task_id,),
        ).lastrowid
        conn.execute("UPDATE tasks SET current_run_id=? WHERE id=?", (current, task_id))

    assert kb.reconcile_stale_task_runs(conn) == [old_run_id]
    assert kb.get_task(conn, task_id).current_run_id == current
    assert conn.execute("SELECT status FROM task_runs WHERE id=?", (current,)).fetchone()[0] == "running"


def test_live_pid_defers_detached_run_reconciliation(conn) -> None:
    task_id, run_id = _claim(conn)
    sleeper = subprocess.Popen(["sleep", "30"])
    try:
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status='blocked', current_run_id=NULL, "
                "claim_lock=NULL, claim_expires=NULL, worker_pid=NULL WHERE id=?",
                (task_id,),
            )
            conn.execute("UPDATE task_runs SET worker_pid=? WHERE id=?", (sleeper.pid, run_id))
        assert kb.reconcile_stale_task_runs(conn) == []
        assert conn.execute("SELECT status FROM task_runs WHERE id=?", (run_id,)).fetchone()[0] == "running"
    finally:
        sleeper.terminate()
        sleeper.wait()


def test_dispatch_tick_reconciles_detached_runs_when_enabled(conn) -> None:
    task_id, run_id = _claim(conn)
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status='blocked', current_run_id=NULL, "
            "claim_lock=NULL, claim_expires=NULL, worker_pid=NULL WHERE id=?",
            (task_id,),
        )

    result = kb.dispatch_once(conn, dry_run=True, spawn_fn=lambda *_a, **_k: (True, ""))
    assert result.reconciled_runs == [run_id]
