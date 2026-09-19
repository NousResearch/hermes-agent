"""Dispatcher reconciliation keeps task lifecycle state and attempt state coherent."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    connection = kbc.connect()
    try:
        yield connection
    finally:
        connection.close()


def _diverge_nonrunning_task(conn, status: str) -> tuple[str, int]:
    task_id = kb.create_task(conn, title="diverged", assignee="coder")
    claimed = kb.claim_task(conn, task_id)
    assert claimed is not None and claimed.current_run_id is not None
    run_id = claimed.current_run_id
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status = ?, claim_lock = 'host:dead', claim_expires = ?, "
            "worker_pid = ?, worker_started_at = 'old-boot|1' WHERE id = ?",
            (status, int(time.time()) + 3600, 424242, task_id),
        )
    return task_id, run_id


@pytest.mark.parametrize("status", ["ready", "done"])
def test_reconciliation_closes_open_run_without_rewriting_requested_lifecycle(conn, status):
    task_id, run_id = _diverge_nonrunning_task(conn, status)

    repaired = kbd.reconcile_task_run_invariants(conn)

    task = conn.execute(
        "SELECT status, current_run_id, claim_lock, claim_expires, worker_pid, worker_started_at "
        "FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    run = conn.execute(
        "SELECT status, outcome, ended_at FROM task_runs WHERE id = ?", (run_id,)
    ).fetchone()
    events = conn.execute(
        "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (task_id,)
    ).fetchall()

    assert repaired == [task_id]
    assert dict(task) == {
        "status": status,
        "current_run_id": None,
        "claim_lock": None,
        "claim_expires": None,
        "worker_pid": None,
        "worker_started_at": None,
    }
    assert dict(run) == {
        "status": "reconciled_state_divergence",
        "outcome": "reconciled_state_divergence",
        "ended_at": pytest.approx(time.time(), abs=5),
    }
    assert [event["kind"] for event in events].count("reconciled_state_divergence") == 1


def test_reconciliation_leaves_consistent_running_attempt_unchanged(conn):
    task_id = kb.create_task(conn, title="healthy", assignee="coder")
    claimed = kb.claim_task(conn, task_id)
    assert claimed is not None and claimed.current_run_id is not None
    run_id = claimed.current_run_id
    before_event_count = conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id = ?", (task_id,)
    ).fetchone()[0]

    repaired = kbd.reconcile_task_run_invariants(conn)

    task = conn.execute(
        "SELECT status, current_run_id, claim_lock FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    run = conn.execute(
        "SELECT status, outcome, ended_at FROM task_runs WHERE id = ?", (run_id,)
    ).fetchone()
    after_event_count = conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id = ?", (task_id,)
    ).fetchone()[0]

    assert repaired == []
    assert dict(task)["status"] == "running"
    assert task["current_run_id"] == run_id
    assert task["claim_lock"] is not None
    assert dict(run) == {"status": "running", "outcome": None, "ended_at": None}
    assert after_event_count == before_event_count


def test_reconciliation_requeues_running_task_without_an_open_attempt(conn):
    task_id = kb.create_task(conn, title="missing attempt", assignee="coder")
    claimed = kb.claim_task(conn, task_id)
    assert claimed is not None
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE task_runs SET status = 'reclaimed', outcome = 'reclaimed', ended_at = ? "
            "WHERE task_id = ?",
            (int(time.time()), task_id),
        )
        conn.execute("UPDATE tasks SET current_run_id = NULL WHERE id = ?", (task_id,))

    repaired = kbd.reconcile_task_run_invariants(conn)

    task = conn.execute(
        "SELECT status, current_run_id, claim_lock, worker_pid FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    events = conn.execute(
        "SELECT kind FROM task_events WHERE task_id = ?", (task_id,)
    ).fetchall()

    assert repaired == [task_id]
    assert dict(task) == {
        "status": "ready",
        "current_run_id": None,
        "claim_lock": None,
        "worker_pid": None,
    }
    assert [event["kind"] for event in events].count("reconciled_state_divergence") == 1


def test_completion_finalization_closes_every_open_attempt(conn):
    task_id = kb.create_task(conn, title="finalize", assignee="coder")
    claimed = kb.claim_task(conn, task_id)
    assert claimed is not None and claimed.current_run_id is not None
    current_run_id = claimed.current_run_id
    with kb.write_txn(conn):
        extra_run_id = conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, started_at) VALUES (?, ?, 'running', ?)",
            (task_id, "coder", int(time.time())),
        ).lastrowid

    assert kb.complete_task(conn, task_id, expected_run_id=current_run_id, summary="finished")

    task = conn.execute(
        "SELECT status, current_run_id FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    runs = conn.execute(
        "SELECT id, outcome, ended_at FROM task_runs WHERE task_id = ? ORDER BY id", (task_id,)
    ).fetchall()
    events = conn.execute(
        "SELECT kind FROM task_events WHERE task_id = ?", (task_id,)
    ).fetchall()

    assert dict(task) == {"status": "done", "current_run_id": None}
    assert [dict(run) for run in runs] == [
        {"id": current_run_id, "outcome": "completed", "ended_at": pytest.approx(time.time(), abs=5)},
        {"id": extra_run_id, "outcome": "reconciled_state_divergence", "ended_at": pytest.approx(time.time(), abs=5)},
    ]
    assert [event["kind"] for event in events].count("reconciled_state_divergence") == 1


def test_dispatch_reconciles_dead_divergence_before_capacity_and_spawns_ready_work(conn, monkeypatch):
    import hermes_cli.profiles as profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)
    diverged_task, run_id = _diverge_nonrunning_task(conn, "done")
    eligible_task = kb.create_task(conn, title="eligible", assignee="coder")

    result = kbd.dispatch_once(
        conn,
        dry_run=True,
        max_spawn=1,
        max_in_progress=1,
    )

    run = conn.execute(
        "SELECT outcome, ended_at FROM task_runs WHERE id = ?", (run_id,)
    ).fetchone()
    divergence_events = conn.execute(
        "SELECT kind FROM task_events WHERE task_id = ?", (diverged_task,)
    ).fetchall()

    assert result.reconciled_state_divergence == [diverged_task]
    assert result.spawned == [(eligible_task, "coder", "")]
    assert dict(run)["outcome"] == "reconciled_state_divergence"
    assert run["ended_at"] is not None
    assert [event["kind"] for event in divergence_events].count("reconciled_state_divergence") == 1
