"""Invariant: ``complete_task`` never closes a live worker's run for a caller that
neither owns the run nor asked for an operator override (issue #111764).

A claim-less completion (a human at the CLI, an orchestrator session — anything
without ``HERMES_KANBAN_*`` env) used to be authorised by task status alone, so it
marked a ``running`` card done and ``_end_run`` closed the dispatcher worker's run
row while that worker kept executing. The guard mirrors ``request_review``'s: a
``running`` task under a live claim needs ``expected_run_id`` or ``force=True``.

Issue #120159 extends the same guard to claims that never spawned a worker — a
CLI/library ``hermes kanban claim`` leaves ``worker_pid`` NULL, so process
liveness can't be the authority and the claim's own TTL is instead: an unexpired
worker-less claim is a live claim, with a claimer CAS (``claimer == claim_lock``,
the check ``heartbeat_claim`` already performs) letting the claim's owner close
its own claim without an override.
"""

from __future__ import annotations

import os
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


def _claimed_running_task(
    conn, *, live_worker: bool = True, claimer: str | None = None,
) -> tuple[str, int]:
    tid = kb.create_task(conn, title="live", assignee="coder")
    assert kb.claim_task(conn, tid, claimer=claimer or kb._claimer_id()) is not None
    if live_worker:
        # This process stands in for the spawned worker: alive, fingerprinted.
        kbd._set_worker_pid(conn, tid, os.getpid())
    return tid, kb._current_run_id(conn, tid)


# A claim lock issued by *another* session (``hermes kanban claim`` from a
# different shell / host) — never this process's ``_claimer_id()``.
_FOREIGN = "otherhost:4321"


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


# ---------------------------------------------------------------------------
# #120159: claims that never spawned a worker (CLI / library claims)
# ---------------------------------------------------------------------------


def test_workerless_claim_refuses_foreign_session_within_ttl(conn):
    """A ``hermes kanban claim`` from another session has no worker process to
    prove liveness with, so its TTL is the liveness authority: a second
    session's claim-less ``complete`` must not close it underneath."""
    tid, run_id = _claimed_running_task(conn, live_worker=False, claimer=_FOREIGN)

    with pytest.raises(kb.LiveClaimError):
        kb.complete_task(conn, tid, result="second session says done")

    # Nothing moved: the claimer's run is open, the claim is intact.
    run = conn.execute("SELECT ended_at FROM task_runs WHERE id = ?", (run_id,)).fetchone()
    assert run["ended_at"] is None
    row = conn.execute("SELECT status, claim_lock FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert row["status"] == "running" and row["claim_lock"] == _FOREIGN

    # The run's owner still finishes it with proof of ownership...
    assert kb.complete_task(conn, tid, result="owner done", expected_run_id=run_id) is True

    # ...and an operator override still closes a worker-less claim too.
    tid2, _ = _claimed_running_task(conn, live_worker=False, claimer=_FOREIGN)
    assert kb.complete_task(conn, tid2, result="operator override", force=True) is True


def test_claimer_cas_closes_its_own_workerless_claim(conn):
    """The claim's owner identifies itself with the claim lock itself — the same
    CAS ``heartbeat_claim`` runs — so claim -> work -> complete keeps working."""
    tid, run_id = _claimed_running_task(conn, live_worker=False, claimer=_FOREIGN)

    assert kb.complete_task(conn, tid, result="done", claimer=_FOREIGN) is True
    run = conn.execute("SELECT ended_at, outcome FROM task_runs WHERE id = ?", (run_id,)).fetchone()
    assert run["ended_at"] is not None and run["outcome"] == "completed"
    assert conn.execute(
        "SELECT status FROM tasks WHERE id = ?", (tid,)
    ).fetchone()["status"] == "done"


def test_expired_workerless_claim_is_claim_less_completable(conn):
    """TTL is the deadline: once ``claim_expires`` passes the claim stops being
    live (``release_stale_claims`` reclaims it), so the plain manual completion
    an operator has always had keeps working."""
    import time

    tid, _ = _claimed_running_task(conn, live_worker=False, claimer=_FOREIGN)
    conn.execute(
        "UPDATE tasks SET claim_expires = ? WHERE id = ?", (int(time.time()) - 1, tid)
    )

    assert kb.complete_task(conn, tid, result="ttl passed") is True
    assert conn.execute(
        "SELECT status FROM tasks WHERE id = ?", (tid,)
    ).fetchone()["status"] == "done"


def test_request_review_shares_the_workerless_claim_fence(conn):
    """``request_review`` keys on the same liveness as ``complete_task``: a
    worker-less claim is live until its TTL, the claimer CAS and force bypass it."""
    tid, _ = _claimed_running_task(conn, live_worker=False, claimer=_FOREIGN)

    ok, reason = kb.request_review(conn, tid, summary="steal", with_reason=True)
    assert ok is False and "live claim" in reason
    assert kb.get_task(conn, tid).status == "running"

    assert kb.request_review(conn, tid, summary="own", claimer=_FOREIGN) is True
    assert kb.get_task(conn, tid).status == "review"


def test_cli_complete_refuses_foreign_workerless_claim(conn):
    """End-to-end: ``hermes kanban complete`` on a card another session claimed
    reports the refusal and points at ``--force`` / ``reclaim`` instead of
    silently closing the claim."""
    from hermes_cli import kanban as kc

    tid, _ = _claimed_running_task(conn, live_worker=False, claimer=_FOREIGN)

    out = kc.run_slash(f"complete {tid} --result 'second session'")
    assert "cannot complete" in out and "--force" in out, out
    assert conn.execute(
        "SELECT status FROM tasks WHERE id = ?", (tid,)
    ).fetchone()["status"] == "running"
