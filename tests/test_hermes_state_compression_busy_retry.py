"""A live compression lock must delay a concurrent append, not destroy the turn.

``append_message`` refused immediately when another writer held the session's
compression lock. The conversation loop turns that into
``session_persistence_failed`` and tells the operator to check disk space and
permissions, when in fact the store is healthy and busy for a few seconds.

The write lock already gets a patience budget for exactly this reason (#74478).
A compression hold is even more bounded, since the lock row carries its own
``expires_at``, so the same budget applies here.

The sibling condition, a compressor finding its own lease gone, is permanent
and must still fail fast rather than spin out the whole budget.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from hermes_state import CompressionSessionBusyError, SessionDB


@pytest.fixture
def db(tmp_path: Path) -> SessionDB:
    d = SessionDB(tmp_path / "state.db")
    d.create_session("sess1", source="test")
    return d


def test_append_waits_out_a_live_compression_lock(db: SessionDB) -> None:
    """The classic race: a steer lands while compression owns the session."""
    assert db.try_acquire_compression_lock("sess1", "compressor") is True

    released = threading.Event()

    def _release_soon():
        time.sleep(0.3)
        db.release_compression_lock("sess1", "compressor")
        released.set()

    t = threading.Thread(target=_release_soon, daemon=True)
    t.start()
    try:
        started = time.monotonic()
        # No compression_lock_holder: this is an ordinary turn writer.
        db.append_message("sess1", role="user", content="steered mid-compression")
        elapsed = time.monotonic() - started
    finally:
        t.join(timeout=5)

    assert released.is_set(), "test bug: lock was never released"
    assert elapsed >= 0.25, "append returned before the lock could clear"
    rows = db.get_messages("sess1")
    assert any(r["content"] == "steered mid-compression" for r in rows), (
        "the message the user sent was lost"
    )


def test_append_still_gives_up_when_the_lock_never_clears(
    db: SessionDB, monkeypatch
) -> None:
    """The wait is bounded: a lock that never clears is still refused.

    The lease is a correctness boundary, so a genuinely long-running or wedged
    compression must not end with a stale turn landing in the parent.
    """
    monkeypatch.setattr(SessionDB, "_COMPRESSION_BUSY_WAIT_S", 0.5)
    assert db.try_acquire_compression_lock("sess1", "compressor") is True

    started = time.monotonic()
    with pytest.raises(CompressionSessionBusyError):
        db.append_message("sess1", role="user", content="never lands")
    elapsed = time.monotonic() - started

    assert elapsed >= 0.4, "gave up before spending the patience budget"
    assert elapsed < 10, "did not give up within a bounded time"


def test_the_lock_owner_is_never_delayed_by_its_own_lock(db: SessionDB) -> None:
    assert db.try_acquire_compression_lock("sess1", "compressor") is True

    started = time.monotonic()
    db.append_message(
        "sess1",
        role="assistant",
        content="written by the compressor",
        compression_lock_holder="compressor",
    )
    assert time.monotonic() - started < 0.2


def test_transient_error_is_a_subclass_of_the_original(db: SessionDB) -> None:
    """Existing `except CompressionSessionBusyError` handlers must still catch."""
    from hermes_state import SessionCompressionInProgressError

    assert issubclass(SessionCompressionInProgressError, CompressionSessionBusyError)


def test_no_lock_means_no_delay(db: SessionDB) -> None:
    started = time.monotonic()
    db.append_message("sess1", role="user", content="uncontended")
    assert time.monotonic() - started < 0.2


def test_a_lost_compression_lease_still_fails_fast(db: SessionDB) -> None:
    """The other CompressionSessionBusyError case must NOT be retried.

    ``publish_compression_child`` raises the same base class when the
    compressor discovers its own lease is gone. That is permanent, so
    retrying would burn the whole patience budget before failing anyway.
    Only the transient subclass raised by ``append_message`` is retried.
    """
    started = time.monotonic()
    with pytest.raises(CompressionSessionBusyError):
        db.publish_compression_child(
            parent_session_id="sess1",
            child_session_id="child1",
            source="test",
            messages=[{"role": "user", "content": "compacted"}],
            compression_lock_holder="not-the-holder",
            require_compression_lease=True,
        )
    assert time.monotonic() - started < 0.5, (
        "a lost lease is permanent and must not spend the retry budget"
    )
