"""FIFO waiter fairness for the session turn lease (card t_d4f85953, Fix B / B-2).

The lease gives strict mutual exclusion but no queue: the waiter polls, and a fresh turn that
arrives inside the release window takes the conversation ahead of a turn that has been waiting
for minutes. Those turns are not competing for CPU, they are competing for the session, so the
fresh claim loses its whole generation of transcript work and reports
``session_turn_lease_timeout`` to its caller.

The wait is therefore published in ``session_turn_waiters`` for as long as the waiter is
actually waiting. A fresh claim is denied while a live waiter is ahead of it in that table, so
the handoff goes to the oldest waiter instead of to whoever happens to poll first.
"""

from __future__ import annotations

import os
import threading
import time
from types import SimpleNamespace

import hermes_state
from hermes_state import SessionDB


def _holder(tag: str) -> str:
    return f"pid={os.getpid()}:turn={tag}:platform=test"


def _wait_for_waiter_count(db: SessionDB, session_id: str, count: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if db.session_turn_waiter_count(session_id) >= count:
            return True
        time.sleep(0.01)
    return False


def test_fresh_claim_is_denied_while_a_waiter_is_ahead(tmp_path):
    """The release window must not hand the conversation to a fresh claimant."""
    db = SessionDB(tmp_path / "state.db")
    db.create_session("shared", source="test")
    holder = _holder("holder")
    assert db.try_acquire_session_turn_lease("shared", holder, ttl_seconds=30)

    results: dict = {}

    def wait():
        results["acquired"] = db.acquire_session_turn_lease(
            "shared",
            _holder("waiter"),
            ttl_seconds=30,
            wait_seconds=5,
            poll_interval_seconds=0.02,
        )

    thread = threading.Thread(target=wait, daemon=True)
    thread.start()
    try:
        assert _wait_for_waiter_count(db, "shared", 1), "the waiting turn never published its wait"

        # The release window: a fresh turn arriving here must not jump the queue.
        assert (
            db.try_acquire_session_turn_lease("shared", _holder("fresh"), ttl_seconds=30) is False
        )

        db.release_session_turn_lease("shared", holder)
        thread.join(timeout=5)
        assert results.get("acquired") is True, "the waiter lost the handoff to a fresh claim"
        assert db.session_turn_waiter_count("shared") == 0

        # The waiter owns it now; the fresh claimant is still fenced out.
        assert (
            db.try_acquire_session_turn_lease("shared", _holder("fresh"), ttl_seconds=30) is False
        )
        db.release_session_turn_lease("shared", _holder("waiter"))
    finally:
        thread.join(timeout=1)


def test_oldest_waiter_wins_the_handoff(tmp_path):
    """Two waiters queue in arrival order, not in poll-luck order."""
    db = SessionDB(tmp_path / "state.db")
    db.create_session("shared", source="test")
    holder = _holder("holder")
    assert db.try_acquire_session_turn_lease("shared", holder, ttl_seconds=30)

    results: dict = {}

    def wait(tag: str):
        results[tag] = db.acquire_session_turn_lease(
            "shared",
            _holder(tag),
            ttl_seconds=30,
            wait_seconds=5,
            poll_interval_seconds=0.02,
        )

    first = threading.Thread(target=wait, args=("waiter-1",), daemon=True)
    second = threading.Thread(target=wait, args=("waiter-2",), daemon=True)
    first.start()
    try:
        assert _wait_for_waiter_count(db, "shared", 1)
        second.start()
        assert _wait_for_waiter_count(db, "shared", 2)

        db.release_session_turn_lease("shared", holder)
        first.join(timeout=5)
        time.sleep(0.3)
        assert results.get("waiter-1") is True, "the older waiter did not win the handoff"
        assert results.get("waiter-2") is None, "the newer waiter jumped the queue"

        db.release_session_turn_lease("shared", _holder("waiter-1"))
        second.join(timeout=5)
        assert results.get("waiter-2") is True
        db.release_session_turn_lease("shared", _holder("waiter-2"))
    finally:
        first.join(timeout=1)
        second.join(timeout=1)


def test_expired_wait_leaves_no_stale_waiter_row(tmp_path):
    """A waiter that gave up must not fence the session behind it."""
    db = SessionDB(tmp_path / "state.db")
    db.create_session("shared", source="test")
    holder = _holder("holder")
    assert db.try_acquire_session_turn_lease("shared", holder, ttl_seconds=30)

    assert (
        db.acquire_session_turn_lease(
            "shared",
            _holder("waiter"),
            wait_seconds=0.2,
            poll_interval_seconds=0.02,
        )
        is False
    )
    assert db.session_turn_waiter_count("shared") == 0

    db.release_session_turn_lease("shared", holder)
    assert db.try_acquire_session_turn_lease("shared", _holder("fresh"), ttl_seconds=5) is True


def test_waiter_row_of_a_dead_process_is_pruned(tmp_path, monkeypatch):
    """A waiter killed mid-wait must not wedge the conversation forever."""
    db = SessionDB(tmp_path / "state.db")
    db.create_session("shared", source="test")
    db.register_session_turn_waiter("shared", "pid=424242:turn=dead:platform=test")

    monkeypatch.setattr(hermes_state, "psutil", SimpleNamespace(pid_exists=lambda pid: False))

    assert db.try_acquire_session_turn_lease("shared", _holder("fresh"), ttl_seconds=5) is True


def test_ancient_waiter_row_is_pruned(tmp_path):
    """A waiter row older than the longest legitimate wait cannot fence the session."""
    db = SessionDB(tmp_path / "state.db")
    db.create_session("shared", source="test")
    db.register_session_turn_waiter(
        "shared", _holder("ghost"), enqueued_at=time.time() - 100_000.0
    )

    assert db.try_acquire_session_turn_lease("shared", _holder("fresh"), ttl_seconds=5) is True
    assert db.session_turn_waiter_count("shared") == 0


def test_waiter_fairness_follows_the_conversation_root(tmp_path):
    """Fairness is keyed on the conversation, not on the rotated session id."""
    db = SessionDB(tmp_path / "state.db")
    db.create_session("root", source="test")
    db.end_session("root", "compression")
    db.create_session("child", source="test", parent_session_id="root")

    holder = _holder("holder")
    assert db.try_acquire_session_turn_lease("root", holder, ttl_seconds=30)
    db.register_session_turn_waiter("child", _holder("waiter"))

    # A fresh claim on the ROOT is behind the waiter that queued on the child tip.
    assert db.try_acquire_session_turn_lease("root", _holder("fresh"), ttl_seconds=5) is False
    assert db.session_turn_waiter_count("child") == 1

    db.release_session_turn_lease("root", holder)
    db.clear_session_turn_waiter("child", _holder("waiter"))
    assert db.try_acquire_session_turn_lease("root", _holder("fresh"), ttl_seconds=5) is True


def test_holder_and_waiter_readers_track_live_state(tmp_path):
    """The admission receipt (B-3) reads the holder and the queue depth from here."""
    db = SessionDB(tmp_path / "state.db")
    db.create_session("shared", source="test")
    assert db.current_session_turn_lease_holder("shared") is None
    assert db.session_turn_waiter_count("shared") == 0
    assert db.oldest_session_turn_waiter_age("shared") is None

    holder = _holder("holder")
    assert db.try_acquire_session_turn_lease("shared", holder, ttl_seconds=30)
    assert db.current_session_turn_lease_holder("shared") == holder

    db.register_session_turn_waiter("shared", _holder("waiter-1"))
    assert db.session_turn_waiter_count("shared") == 1
    age = db.oldest_session_turn_waiter_age("shared")
    assert age is not None and 0.0 <= age < 5.0

    db.clear_session_turn_waiter("shared", _holder("waiter-1"))
    assert db.session_turn_waiter_count("shared") == 0
    assert db.oldest_session_turn_waiter_age("shared") is None


def test_run_without_wait_never_publishes_a_waiter_row(tmp_path):
    """The single-shot claim path stays as cheap as it was; no row, no prune, no delay."""
    db = SessionDB(tmp_path / "state.db")
    db.create_session("shared", source="test")
    assert db.try_acquire_session_turn_lease("shared", _holder("first"), ttl_seconds=5) is True
    assert db.try_acquire_session_turn_lease("shared", _holder("second"), ttl_seconds=5) is False
    assert db.session_turn_waiter_count("shared") == 0
