"""The SQLite coordination repository's compression-lock refresh: only the holder extends its lease, an expired lease stays reclaimable, and a holder whose lock was reclaimed cannot take it back."""

import time

import pytest

from persistence.sqlite.conversation._store.sessions import create_session


@pytest.fixture
def store(backend, scope_a):
    return backend.open(scope_a)


@pytest.fixture
def db(store):
    """The store's StoreHolder, shared with its repositories."""
    return store.holder()


def test_refresh_compression_lock_requires_holder_and_preserves_reclaimability(db, store, monkeypatch):
    create_session(db, "s1", "cli")
    locks = store.coordination

    monkeypatch.setattr(time, "time", lambda: 1000.0)
    assert locks.try_acquire_compression_lock(session_id="s1", holder="holder-a", ttl_seconds=10.0) is True

    original_expires = db._conn.execute(
        "SELECT expires_at FROM compression_locks WHERE session_id = ?",
        ("s1",),
    ).fetchone()[0]

    monkeypatch.setattr(time, "time", lambda: 1005.0)
    assert locks.refresh_compression_lock(session_id="s1", holder="holder-a", ttl_seconds=10.0) is True
    refreshed_expires = db._conn.execute(
        "SELECT expires_at FROM compression_locks WHERE session_id = ?",
        ("s1",),
    ).fetchone()[0]
    assert refreshed_expires > original_expires

    assert locks.refresh_compression_lock(session_id="s1", holder="holder-b", ttl_seconds=10.0) is False

    monkeypatch.setattr(time, "time", lambda: 1016.0)
    assert locks.try_acquire_compression_lock(session_id="s1", holder="holder-b", ttl_seconds=10.0) is True


def test_refresh_cannot_resurrect_a_lock_already_reclaimed(db, store, monkeypatch):
    """Once a competitor owns the row, the old holder's refresh must fail.

    The guard is the ``holder`` match, not the clock: a reclaim replaces
    ``holder``, so the previous owner's UPDATE matches nothing.
    """
    create_session(db, "s1", "cli")
    locks = store.coordination

    monkeypatch.setattr(time, "time", lambda: 1000.0)
    assert locks.try_acquire_compression_lock(session_id="s1", holder="holder-a", ttl_seconds=10.0) is True

    # holder-a's lease lapses and holder-b legitimately reclaims it.
    monkeypatch.setattr(time, "time", lambda: 1020.0)
    assert locks.try_acquire_compression_lock(session_id="s1", holder="holder-b", ttl_seconds=10.0) is True

    # holder-a coming back late must NOT steal it back.
    assert locks.refresh_compression_lock(session_id="s1", holder="holder-a", ttl_seconds=10.0) is False
    current = db._conn.execute(
        "SELECT holder FROM compression_locks WHERE session_id = ?",
        ("s1",),
    ).fetchone()[0]
    assert current == "holder-b"
