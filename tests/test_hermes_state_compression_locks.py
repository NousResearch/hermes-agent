"""Tests for ``SessionDB`` compression-lock primitives.

These cover the atomic per-session lock that prevents two compression
paths from racing on the same ``session_id`` and producing orphan child
sessions (Damien's "parent → two orphan children" repro shape, see
``tests/agent/test_compression_concurrent_fork.py`` for the
behavioural regression test).

Focus here: the lock primitives themselves (acquire, release, TTL,
diagnostic accessor) — not the wiring into compression.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path: Path) -> SessionDB:
    return SessionDB(tmp_path / "state.db")


# ----------------------------------------------------------------------
# Single-holder semantics
# ----------------------------------------------------------------------


def test_acquire_succeeds_when_unlocked(db: SessionDB) -> None:
    assert db.try_acquire_compression_lock("sess1", "holder1") is True
    assert db.get_compression_lock_holder("sess1") == "holder1"


def test_acquire_blocks_second_holder(db: SessionDB) -> None:
    assert db.try_acquire_compression_lock("sess1", "holder1") is True
    assert db.try_acquire_compression_lock("sess1", "holder2") is False
    # First holder still owns it
    assert db.get_compression_lock_holder("sess1") == "holder1"


def test_release_allows_reacquire(db: SessionDB) -> None:
    db.try_acquire_compression_lock("sess1", "holder1")
    db.release_compression_lock("sess1", "holder1")
    assert db.get_compression_lock_holder("sess1") is None
    assert db.try_acquire_compression_lock("sess1", "holder2") is True


def test_release_with_wrong_holder_is_noop(db: SessionDB) -> None:
    db.try_acquire_compression_lock("sess1", "holder1")
    # Late-returning compressor must not release a lock it doesn't own
    db.release_compression_lock("sess1", "holder_other")
    assert db.get_compression_lock_holder("sess1") == "holder1"


def test_release_when_unlocked_is_noop(db: SessionDB) -> None:
    # No exception, no state change
    db.release_compression_lock("never_locked", "holder1")
    assert db.get_compression_lock_holder("never_locked") is None


# ----------------------------------------------------------------------
# Per-session isolation
# ----------------------------------------------------------------------


def test_locks_are_per_session(db: SessionDB) -> None:
    assert db.try_acquire_compression_lock("sess1", "holder1") is True
    # Different session: independent lock
    assert db.try_acquire_compression_lock("sess2", "holder2") is True
    assert db.get_compression_lock_holder("sess1") == "holder1"
    assert db.get_compression_lock_holder("sess2") == "holder2"


# ----------------------------------------------------------------------
# TTL / expiry recovery
# ----------------------------------------------------------------------


def test_expired_lock_is_reclaimable(db: SessionDB) -> None:
    """A crashed compressor must not permanently block the session."""
    # Acquire with a very short TTL
    db.try_acquire_compression_lock("sess1", "crashed_holder", ttl_seconds=0.5)
    time.sleep(1.0)
    # Holder check honours expiry
    assert db.get_compression_lock_holder("sess1") is None
    # New holder can claim it
    assert db.try_acquire_compression_lock("sess1", "fresh_holder") is True
    assert db.get_compression_lock_holder("sess1") == "fresh_holder"


def test_non_expired_lock_is_held(db: SessionDB) -> None:
    db.try_acquire_compression_lock("sess1", "holder1", ttl_seconds=60)
    # Immediately after, still held
    assert db.try_acquire_compression_lock("sess1", "holder2") is False


# ----------------------------------------------------------------------
# Empty / invalid input
# ----------------------------------------------------------------------


def test_acquire_empty_session_id_returns_false(db: SessionDB) -> None:
    assert db.try_acquire_compression_lock("", "holder1") is False


def test_release_empty_session_id_is_noop(db: SessionDB) -> None:
    # No exception
    db.release_compression_lock("", "holder1")


def test_holder_empty_session_id_returns_none(db: SessionDB) -> None:
    assert db.get_compression_lock_holder("") is None


# ----------------------------------------------------------------------
# Concurrency: real threads racing on the same session_id
# ----------------------------------------------------------------------


def test_concurrent_acquire_only_one_winner(db: SessionDB) -> None:
    """Damien's race shape: N threads call acquire on the same session_id;
    exactly one must win, the rest must be cleanly rejected."""
    results: list[bool] = []
    barrier = threading.Barrier(8)
    lock = threading.Lock()

    def try_acquire(idx: int) -> None:
        holder = f"thread_{idx}"
        barrier.wait()  # synchronize start
        got = db.try_acquire_compression_lock("contended_session", holder)
        with lock:
            results.append(got)

    threads = [threading.Thread(target=try_acquire, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Exactly one thread acquired
    assert sum(1 for r in results if r is True) == 1
    assert sum(1 for r in results if r is False) == 7
    # The single winner still owns it
    assert db.get_compression_lock_holder("contended_session") is not None
