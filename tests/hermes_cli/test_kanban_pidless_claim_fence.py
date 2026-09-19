"""Pid-less claim fence (#113004): a fresh claim that never spawned a worker
is still a live claim.

The terminal fence on ``complete_task`` / ``request_review`` used to be gated
on worker-pid liveness, so a control-plane / CLI / library lane that claimed
via ``claim_task`` (no worker spawned) had its claim and open run closed by a
foreign lane with no ``expected_run_id``. These tests pin the fix.
"""
import os
import time
from pathlib import Path

import pytest

import hermes_cli.kanban_db as kb
from hermes_cli import kanban_db_connect as kbc

@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    with kbc.connect() as c:
        yield c


def _pidless_claim(conn):
    """Claim without spawning a worker: claim_lock set, worker_pid NULL."""
    tid = kb.create_task(conn, title="pidless claim", assignee="lane-K")
    kb.claim_task(conn, tid, claimer="lane-K")
    row = conn.execute(
        "SELECT current_run_id, worker_pid FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row["worker_pid"] is None
    assert row["current_run_id"] is not None
    return tid, row["current_run_id"]


def _foreign_complete(conn, tid):
    with pytest.raises(kb.LiveClaimError) as exc:
        kb.complete_task(conn, tid, result="lane D says done")
    return str(exc.value)


def test_foreign_complete_of_pidless_claim_refused(conn):
    tid, run_id = _pidless_claim(conn)
    msg = _foreign_complete(conn, tid)
    assert "live claim" in msg
    row = conn.execute(
        "SELECT status, claim_lock, current_run_id FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row["status"] == "running"
    assert row["claim_lock"] == "lane-K"
    assert row["current_run_id"] == run_id


def test_foreign_request_review_of_pidless_claim_refused(conn):
    tid, _ = _pidless_claim(conn)
    ok, reason = kb.request_review(conn, tid, with_reason=True)
    assert ok is False
    assert "live claim" in reason
    row = conn.execute(
        "SELECT status, claim_lock, current_run_id FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row["status"] == "running"
    assert row["claim_lock"] == "lane-K"


def test_owner_complete_of_pidless_claim_allowed(conn):
    """The claiming lane completes its own claim with its run id — no force needed."""
    tid, run_id = _pidless_claim(conn)
    assert kb.complete_task(conn, tid, result="owner done", expected_run_id=run_id) is True
    run = conn.execute("SELECT ended_at FROM task_runs WHERE id = ?", (run_id,)).fetchone()
    assert run["ended_at"] is not None


def test_force_complete_of_pidless_claim_allowed(conn):
    """force=True stays the explicit operator override."""
    tid, run_id = _pidless_claim(conn)
    assert kb.complete_task(conn, tid, result="operator override", force=True) is True
    run = conn.execute("SELECT ended_at FROM task_runs WHERE id = ?", (run_id,)).fetchone()
    assert run["ended_at"] is not None


def test_expired_pidless_claim_not_fenced(conn):
    """An expired claim belongs to the reaper (reclaim_stale_tasks), not the fence."""
    tid, _ = _pidless_claim(conn)
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET claim_expires = ? WHERE id = ?",
            (int(time.time()) - 1, tid),
        )
    assert kb.complete_task(conn, tid, result="claim long expired") is True
