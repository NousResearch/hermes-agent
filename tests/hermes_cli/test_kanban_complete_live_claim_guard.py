"""Invariant: ``complete_task`` never closes a live worker's run for a caller that
neither owns the run nor asked for an operator override (issue #111764).

A claim-less completion (a human at the CLI, an orchestrator session — anything
without ``HERMES_KANBAN_*`` env) used to be authorised by task status alone, so it
marked a ``running`` card done and ``_end_run`` closed the dispatcher worker's run
row while that worker kept executing. The guard mirrors ``request_review``'s: a
``running`` task under a live claim needs ``expected_run_id`` or ``force=True``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd
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


def _claimed_running_task(conn, *, live_worker: bool = True) -> tuple[str, int]:
    tid = kb.create_task(conn, title="live", assignee="coder")
    assert kb.claim_task(conn, tid, claimer=kb._claimer_id()) is not None
    if live_worker:
        # This process stands in for the spawned worker: alive, fingerprinted.
        kbd._set_worker_pid(conn, tid, os.getpid())
    return tid, kb._current_run_id(conn, tid)


def test_claimless_complete_refuses_live_run_until_forced(conn):
    tid, run_id = _claimed_running_task(conn)

    with pytest.raises(kb.LiveClaimError):
        kb.complete_task(conn, tid, result="someone else says done")

    # Nothing moved: the worker's run is still open and it can still finish its own card.
    run = conn.execute("SELECT ended_at FROM task_runs WHERE id = ?", (run_id,)).fetchone()
    assert run["ended_at"] is None
    assert conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()["status"] == "running"
    assert kb.complete_task(conn, tid, result="worker done", expected_run_id=run_id) is True

    # Explicit operator override still closes a live run.
    tid2, run2 = _claimed_running_task(conn)
    assert kb.complete_task(conn, tid2, result="operator override", force=True) is True
    run = conn.execute("SELECT ended_at, outcome FROM task_runs WHERE id = ?", (run2,)).fetchone()
    assert run["ended_at"] is not None and run["outcome"] == "completed"


def test_claimless_complete_of_claim_without_live_worker_unchanged(conn):
    """A claim whose worker never spawned (or is gone) protects no live run: the
    library / CLI flow that claims and then completes keeps working."""
    tid, run_id = _claimed_running_task(conn, live_worker=False)
    assert kb.complete_task(conn, tid, result="done") is True
    run = conn.execute("SELECT ended_at FROM task_runs WHERE id = ?", (run_id,)).fetchone()
    assert run["ended_at"] is not None


def test_claimless_complete_of_unclaimed_card_unchanged(conn):
    """The legitimate manual flow — completing a card nobody is working on — needs no proof."""
    tid = kb.create_task(conn, title="admin", assignee="coder")
    assert kb.complete_task(conn, tid, result="done") is True
    assert conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()["status"] == "done"


def test_request_review_shares_the_live_worker_fence(conn):
    """``request_review`` keys on the same liveness as ``complete_task``: a claim
    without a live worker process is not a live claim (the human/library flow
    ``claim`` -> ``request_review`` works), a live worker's claim still is."""
    tid, _ = _claimed_running_task(conn, live_worker=False)
    assert kb.request_review(conn, tid, summary="handoff") is True
    assert kb.get_task(conn, tid).status == "review"

    tid2, run2 = _claimed_running_task(conn)
    ok, reason = kb.request_review(conn, tid2, summary="steal", with_reason=True)
    assert ok is False and "live claim" in reason
    assert kb.request_review(conn, tid2, summary="own", expected_run_id=run2) is True


def _redispatch(conn, tid: str) -> int:
    """Simulate the dispatcher re-dispatch shape (t_643c3a0e): end the current
    run as ended/superseded, release the claim, claim a successor run."""
    with kb.write_txn(conn):
        kb._end_run(conn, tid, outcome="blocked", status="blocked")
        conn.execute(
            "UPDATE tasks SET status = 'ready', claim_lock = NULL, claim_expires = NULL, "
            "worker_pid = NULL, worker_started_at = NULL WHERE id = ?",
            (tid,),
        )
    claimed = kb.claim_task(conn, tid, claimer=kb._claimer_id())
    assert claimed is not None
    return kb._current_run_id(conn, tid)


def test_stale_env_run_id_mismatch_recovers_with_warning(conn, monkeypatch):
    """Re-dispatch left the agent env pointing at the ended predecessor run:
    the worker (this process, owner of the CURRENT claim) still completes, and
    the recovery is auditable (warning event) instead of a bare failure."""
    tid, stale_run = _claimed_running_task(conn)
    current_run = _redispatch(conn, tid)
    assert current_run != stale_run
    # The current claim's worker is this process (stale-env caller).
    kbd._set_worker_pid(conn, tid, os.getpid())
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(stale_run))

    assert kb.complete_task(conn, tid, result="stale-env worker done", expected_run_id=stale_run) is True
    assert conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()["status"] == "done"
    events = conn.execute(
        "SELECT run_id, payload FROM task_events WHERE task_id = ? AND kind = 'completion_run_mismatch_recovered'",
        (tid,),
    ).fetchall()
    assert len(events) == 1
    payload = __import__("json").loads(events[0]["payload"])
    assert payload["stale_run_id"] == stale_run
    assert payload["current_run_id"] == current_run
    # The recovery event is grouped under the CURRENT run; the stale run row stays ended.
    assert events[0]["run_id"] == current_run
    stale = conn.execute("SELECT ended_at, status FROM task_runs WHERE id = ?", (stale_run,)).fetchone()
    assert stale["ended_at"] is not None and stale["status"] != "running"


def test_stale_run_id_mismatch_with_foreign_live_owner_still_fails(conn, monkeypatch):
    """A different live worker owns the CURRENT claim: never complete past it
    (two live competing agents must not both complete)."""
    tid, stale_run = _claimed_running_task(conn)
    current_run = _redispatch(conn, tid)
    competing = subprocess.Popen(["sleep", "30"])
    try:
        kbd._set_worker_pid(conn, tid, competing.pid)
        monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(stale_run))

        assert kb.complete_task(conn, tid, result="sneaky", expected_run_id=stale_run) is False
        assert conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()["status"] == "running"
        assert conn.execute("SELECT current_run_id FROM tasks WHERE id = ?", (tid,)).fetchone()["current_run_id"] == current_run
        assert conn.execute(
            "SELECT 1 FROM task_events WHERE task_id = ? AND kind = 'completion_run_mismatch_recovered'", (tid,)
        ).fetchone() is None
    finally:
        competing.terminate()
        competing.wait()


def test_stale_run_id_mismatch_with_dead_owner_still_fails(conn, monkeypatch):
    """Current claim owned by a DEAD worker: not recoverable (a stale claim is
    reclaimed by the dispatcher, not completed by a stale-env stranger)."""
    tid, stale_run = _claimed_running_task(conn)
    current_run = _redispatch(conn, tid)
    dead = subprocess.Popen(["sleep", "30"])
    dead.terminate()
    dead.wait()  # reaped: pid is not alive
    kbd._set_worker_pid(conn, tid, dead.pid)
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(stale_run))

    assert kb.complete_task(conn, tid, result="sneaky", expected_run_id=stale_run) is False
    assert conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()["status"] == "running"


def test_stale_run_id_mismatch_cross_task_env_still_fails(conn, monkeypatch):
    """Env scoped to a DIFFERENT task: the mismatch is never recoverable."""
    tid, stale_run = _claimed_running_task(conn)
    _redispatch(conn, tid)
    kbd._set_worker_pid(conn, tid, os.getpid())
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_someother")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(stale_run))

    assert kb.complete_task(conn, tid, result="sneaky", expected_run_id=stale_run) is False
    assert conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()["status"] == "running"


def test_stale_run_id_mismatch_with_unowned_current_claim_recovers(conn, monkeypatch):
    """Current claim never spawned a worker (unowned): a stale-env caller scoped
    to the task may still complete, with the warning event."""
    tid, stale_run = _claimed_running_task(conn, live_worker=False)
    current_run = _redispatch(conn, tid)  # leaves worker_pid NULL on the successor claim
    assert current_run != stale_run
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(stale_run))

    assert kb.complete_task(conn, tid, result="stale-env done", expected_run_id=stale_run) is True
    assert conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()["status"] == "done"
    assert conn.execute(
        "SELECT 1 FROM task_events WHERE task_id = ? AND kind = 'completion_run_mismatch_recovered'", (tid,)
    ).fetchone() is not None


def test_stale_run_id_still_active_run_not_recoverable(conn, monkeypatch):
    """The stale env run row is still open (NOT ended/superseded): not a
    re-dispatch artifact, keep failing exactly as before."""
    tid, stale_run = _claimed_running_task(conn)
    current_run = _redispatch(conn, tid)
    # Force the stale run row back open: env says this run is still live.
    with kb.write_txn(conn):
        conn.execute("UPDATE task_runs SET ended_at = NULL, status = 'running' WHERE id = ?", (stale_run,))
    kbd._set_worker_pid(conn, tid, os.getpid())
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(stale_run))

    assert kb.complete_task(conn, tid, result="sneaky", expected_run_id=stale_run) is False
    assert conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()["status"] == "running"
