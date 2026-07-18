"""Concurrency and restart recovery tests for Kanban run finalization."""

from __future__ import annotations

import concurrent.futures
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _run_row(conn: sqlite3.Connection, run_id: int) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM task_runs WHERE id = ?", (run_id,)).fetchone()
    assert row is not None
    return row


def _claimed_task() -> tuple[str, int]:
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="finalization race", assignee="coder")
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None
        assert claimed.current_run_id is not None
        return task_id, claimed.current_run_id


def test_concurrent_duplicate_completion_finalizes_run_once(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="complete once", assignee="coder")
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None
        run_id = claimed.current_run_id
        assert run_id is not None

    def complete(index: int) -> bool:
        with kb.connect() as conn:
            return kb.complete_task(
                conn,
                task_id,
                summary=f"completion {index}",
                expected_run_id=run_id,
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(complete, range(8)))

    assert results.count(True) == 1
    assert results.count(False) == 7
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        run = _run_row(conn, run_id)
        terminal_events = conn.execute(
            "SELECT kind FROM task_events "
            "WHERE task_id = ? AND kind IN ('completed', 'blocked')",
            (task_id,),
        ).fetchall()

    assert task is not None
    assert task.status == "done"
    assert task.current_run_id is None
    assert run["status"] == "done"
    assert run["outcome"] == "completed"
    assert run["ended_at"] is not None
    assert [row["kind"] for row in terminal_events] == ["completed"]


def test_complete_block_race_keeps_task_and_run_terminal_state_consistent(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="one terminal outcome", assignee="coder")
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None
        run_id = claimed.current_run_id
        assert run_id is not None

    def complete() -> tuple[str, bool]:
        with kb.connect() as conn:
            return "completed", kb.complete_task(
                conn,
                task_id,
                summary="completed",
                expected_run_id=run_id,
            )

    def block() -> tuple[str, bool]:
        with kb.connect() as conn:
            return "blocked", kb.block_task(
                conn,
                task_id,
                reason="blocked",
                expected_run_id=run_id,
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(complete), pool.submit(block)]
        results = [future.result() for future in futures]

    winner = next(name for name, succeeded in results if succeeded)
    assert sum(succeeded for _, succeeded in results) == 1

    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        run = _run_row(conn, run_id)
        terminal_events = conn.execute(
            "SELECT kind FROM task_events "
            "WHERE task_id = ? AND kind IN ('completed', 'blocked')",
            (task_id,),
        ).fetchall()

    assert task is not None
    assert task.current_run_id is None
    assert run["ended_at"] is not None
    if winner == "completed":
        assert task.status == "done"
        assert (run["status"], run["outcome"]) == ("done", "completed")
    else:
        assert task.status == "blocked"
        assert (run["status"], run["outcome"]) == ("blocked", "blocked")
    assert [row["kind"] for row in terminal_events] == [winner]


def test_complete_reclaim_race_finalizes_run_once(kanban_home):
    task_id, run_id = _claimed_task()

    def complete() -> bool:
        with kb.connect() as conn:
            return kb.complete_task(
                conn, task_id, result="finished", expected_run_id=run_id
            )

    def reclaim() -> bool:
        with kb.connect() as conn:
            return kb.reclaim_task(
                conn, task_id, reason="race", signal_fn=lambda *_: None
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(complete), pool.submit(reclaim)]
        results = [future.result() for future in futures]

    assert sorted(results) == [False, True]
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        run = conn.execute("SELECT * FROM task_runs WHERE id = ?", (run_id,)).fetchone()
        terminal_events = conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? "
            "AND kind IN ('completed', 'reclaimed')",
            (task_id,),
        ).fetchall()

    assert task is not None
    assert task.current_run_id is None
    assert run["ended_at"] is not None
    assert (task.status, run["outcome"]) in {
        ("done", "completed"),
        ("ready", "reclaimed"),
    }
    assert len(terminal_events) == 1


def test_complete_worker_exit_race_finalizes_run_once(kanban_home, monkeypatch):
    task_id, run_id = _claimed_task()
    pid = 991337
    with kb.connect() as conn:
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET worker_pid = ? WHERE id = ?", (pid, task_id)
            )
            conn.execute(
                "UPDATE task_runs SET worker_pid = ?, started_at = started_at - 120 "
                "WHERE id = ?",
                (pid, run_id),
            )
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    kb._record_worker_exit(pid, 1 << 8)

    def complete() -> bool:
        with kb.connect() as conn:
            return kb.complete_task(
                conn, task_id, result="finished", expected_run_id=run_id
            )

    def reap() -> list[str]:
        with kb.connect() as conn:
            return kb.detect_crashed_workers(conn)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(complete), pool.submit(reap)]
        complete_result, crashed = [future.result() for future in futures]

    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        run = conn.execute("SELECT * FROM task_runs WHERE id = ?", (run_id,)).fetchone()
        terminal_events = conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? "
            "AND kind IN ('completed', 'crashed')",
            (task_id,),
        ).fetchall()

    assert task is not None
    assert task.current_run_id is None
    assert run["ended_at"] is not None
    if complete_result:
        assert crashed == []
        assert (task.status, run["outcome"]) == ("done", "completed")
    else:
        assert crashed == [task_id]
        assert (task.status, run["outcome"]) == ("ready", "crashed")
    assert len(terminal_events) == 1


def test_dispatch_reconciles_orphan_run_before_considering_new_claims(kanban_home):
    """A restart repairs leaked run rows even when no task can be claimed."""
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="leaked terminal run", assignee=None)
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None
        run_id = claimed.current_run_id
        assert run_id is not None
        # Simulate a process dying after making the task terminal but before it
        # finalized the run row or cleared current_run_id.
        conn.execute(
            "UPDATE tasks SET status = 'done', completed_at = unixepoch(), "
            "claim_lock = NULL, claim_expires = NULL, worker_pid = NULL "
            "WHERE id = ?",
            (task_id,),
        )

        kb.dispatch_once(conn, spawn_fn=lambda *_args: None)
        assert kb.reconcile_orphaned_runs(conn) == 0

        task = kb.get_task(conn, task_id)
        run = _run_row(conn, run_id)
        events = conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id",
            (task_id,),
        ).fetchall()

    assert task is not None
    assert task.status == "done"
    assert task.current_run_id is None
    assert run["status"] == "reclaimed"
    assert run["outcome"] == "reclaimed"
    assert run["ended_at"] is not None
    assert [row["kind"] for row in events].count("run_reconciled") == 1
