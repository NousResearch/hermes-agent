"""Invariant: terminal transitions cannot take over active Kanban claims.

A claim-less completion (a human at the CLI, an orchestrator session — anything
without ``HERMES_KANBAN_*`` env) used to be authorised by task status alone, so it
marked a ``running`` card done and ``_end_run`` closed the dispatcher worker's run
row while that worker kept executing. The guard mirrors ``request_review``'s: a
``running`` task under a live claim needs ``expected_run_id`` or ``force=True``.

Issue #120159 extends that fence to unexpired CLI/library claims without a spawned
worker: their recorded claimer may complete or request review, while unrelated
callers must use the active run id or an explicit override.
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


def test_ttl_claim_requires_matching_claimer_or_current_run(conn):
    """An unexpired non-worker claim cannot be taken over without its lock or run."""
    tid = kb.create_task(conn, title="cli claim", assignee="operator")
    claimed = kb.claim_task(conn, tid, ttl_seconds=300, claimer="cli-operator")
    assert claimed is not None
    run_id = claimed.current_run_id
    task = kb.get_task(conn, tid)
    assert task.status == "running"
    assert task.worker_pid is None
    assert task.claim_lock == "cli-operator"

    with pytest.raises(kb.LiveClaimError):
        kb.complete_task(conn, tid, result="takeover")
    with pytest.raises(kb.LiveClaimError):
        kb.complete_task(conn, tid, result="takeover", claimer="other-operator")

    run = conn.execute("SELECT ended_at FROM task_runs WHERE id = ?", (run_id,)).fetchone()
    assert run["ended_at"] is None
    assert kb.heartbeat_claim(conn, tid, ttl_seconds=300, claimer="cli-operator")
    assert kb.complete_task(conn, tid, result="done", claimer="cli-operator") is True
    run = conn.execute("SELECT ended_at FROM task_runs WHERE id = ?", (run_id,)).fetchone()
    assert run["ended_at"] is not None


def test_ttl_claim_preserves_expected_run_id_cas(conn):
    """The existing run-id ownership CAS remains an alternative to the claimer."""
    tid = kb.create_task(conn, title="run CAS", assignee="operator")
    claimed = kb.claim_task(conn, tid, ttl_seconds=300, claimer="cli-operator")
    assert claimed is not None and claimed.current_run_id is not None

    assert not kb.complete_task(
        conn, tid, result="stale", expected_run_id=claimed.current_run_id + 1,
    )
    assert kb.get_task(conn, tid).status == "running"
    assert kb.complete_task(conn, tid, result="done", expected_run_id=claimed.current_run_id)


def test_expired_ttl_claim_is_manually_completable(conn):
    """TTL-only protection ends at expiry; no worker process needs recovery."""
    tid = kb.create_task(conn, title="expired", assignee="operator")
    assert kb.claim_task(conn, tid, ttl_seconds=300, claimer="cli-operator") is not None
    conn.execute("UPDATE tasks SET claim_expires = 0 WHERE id = ?", (tid,))
    assert kb.complete_task(conn, tid, result="manual completion") is True


def test_claimless_complete_of_unclaimed_card_unchanged(conn):
    """The legitimate manual flow — completing a card nobody is working on — needs no proof."""
    tid = kb.create_task(conn, title="admin", assignee="coder")
    assert kb.complete_task(conn, tid, result="done") is True
    assert conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()["status"] == "done"


def test_request_review_shares_the_live_worker_fence(conn):
    """``request_review`` uses the same worker and TTL claim fence as completion."""
    tid = kb.create_task(conn, title="cli review", assignee="operator")
    assert kb.claim_task(conn, tid, ttl_seconds=300, claimer="cli-operator") is not None
    ok, reason = kb.request_review(conn, tid, summary="takeover", with_reason=True)
    assert ok is False and "live claim" in reason
    assert kb.request_review(conn, tid, summary="handoff", claimer="cli-operator") is True
    assert kb.get_task(conn, tid).status == "review"

    tid2, run2 = _claimed_running_task(conn)
    ok, reason = kb.request_review(conn, tid2, summary="steal", with_reason=True)
    assert ok is False and "live claim" in reason
    assert kb.request_review(conn, tid2, summary="own", expected_run_id=run2) is True
