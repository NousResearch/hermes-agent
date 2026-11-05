"""Appends flow freely during compression; the commit preserves them (#75316).

HISTORY: ``append_message`` used to refuse while another writer held the
session's compression lock, with a short busy-wait (#75264 → #75083). That
fenced ordinary transcript writes behind a lease whose real job is stopping
two COMPRESSIONS colliding — turns died as ``session_persistence_failed``
whenever a slow provider summary overlapped an incoming message (#74568,
#77386), and a stale lock from a dead PID blocked writes for the full TTL.

CURRENT CONTRACT (watermark commit): appends never check compression_locks.
``archive_and_compact()`` takes a watermark captured at compression start and
re-sequences every row that arrived after it (the concurrent tail) back into
the live transcript, atomically, instead of archiving it with the snapshot.
The commit itself is holder-fenced: a compression whose lease was lost cannot
publish.
"""

from __future__ import annotations

import time

import pytest

from hermes_state_errors import CompressionSessionBusyError
from persistence.sqlite.conversation._store.compression import publish_compression_child
from persistence.sqlite.conversation._store.messages import append_message, get_messages
from persistence.sqlite.conversation._store.sessions import create_session


@pytest.fixture
def db(backend, scope_a):
    store = backend.open(scope_a)
    create_session(store.holder(), "sess1", source="test")
    return store


def test_append_is_never_blocked_by_a_foreign_compression_lock(db) -> None:
    """The classic race: a steer lands while compression owns the session.

    Old behavior: busy-wait then land (or die on timeout). New behavior: the
    append lands IMMEDIATELY — the watermark commit is what protects it.
    """
    assert db.coordination.try_acquire_compression_lock(session_id="sess1", holder="compressor") is True

    started = time.monotonic()
    append_message(db.holder(), "sess1", role="user", content="steered mid-compression")
    elapsed = time.monotonic() - started

    assert elapsed < 0.5, "append must not wait on a compression lease"
    rows = get_messages(db.holder(), "sess1")
    assert any(r["content"] == "steered mid-compression" for r in rows)


def test_append_is_never_blocked_by_a_stale_dead_pid_lock(db) -> None:
    """A crashed compressor's unexpired lock must not fence writes (#74568)."""
    assert db.coordination.try_acquire_compression_lock(
        session_id="sess1", holder="pid-9999999-long-gone", ttl_seconds=3600
    ) is True
    started = time.monotonic()
    append_message(db.holder(), "sess1", role="user", content="lands despite stale lock")
    assert time.monotonic() - started < 0.5
    rows = get_messages(db.holder(), "sess1")
    assert any(r["content"] == "lands despite stale lock" for r in rows)


def test_the_lock_owner_append_still_works(db) -> None:
    assert db.coordination.try_acquire_compression_lock(session_id="sess1", holder="compressor") is True
    append_message(
        db.holder(),
        "sess1",
        role="assistant",
        content="written by the compressor",
        compression_lock_holder="compressor",
    )
    rows = get_messages(db.holder(), "sess1")
    assert any(r["content"] == "written by the compressor" for r in rows)


def test_a_lost_compression_lease_still_fails_fast(db) -> None:
    """``publish_compression_child`` with a lost lease is permanent — no retry."""
    started = time.monotonic()
    with pytest.raises(CompressionSessionBusyError):
        publish_compression_child(
            db.holder(),
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
