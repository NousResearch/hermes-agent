"""Regression test for #44037 — holographic provider leaked its SQLite
connection to GC on shutdown instead of closing it.

The corruption-mechanism framing in #44037 (TLS bytes written into the DB via
an fd-recycle race) was not reproducible from the code: dropping a sqlite
connection flushes valid pages through SQLite's own VFS, never TLS framing, and
the provider is at most a *releaser* of DB fds, not the TLS-flushing owner.

But the underlying resource-hygiene bug is real and is what this test pins:
``HolographicMemoryProvider.shutdown()`` must call ``MemoryStore.close()`` so
the ``check_same_thread=False`` connection's fd is released deterministically
on shutdown, rather than at a non-deterministic GC time on an arbitrary thread.
"""

import sqlite3

import pytest

from plugins.memory.holographic import HolographicMemoryProvider


def _make_provider(tmp_path):
    db_path = str(tmp_path / "memory_store.db")
    provider = HolographicMemoryProvider(config={"db_path": db_path, "hrr_dim": 64})
    provider.initialize(session_id="test-session")
    return provider


def test_shutdown_closes_store_connection(tmp_path):
    provider = _make_provider(tmp_path)
    store = provider._store
    assert store is not None
    conn = store._conn

    # Connection is live before shutdown.
    conn.execute("SELECT 1").fetchone()

    provider.shutdown()

    # References are dropped...
    assert provider._store is None
    assert provider._retriever is None

    # ...AND the underlying connection was actually closed (not left to GC).
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")


def test_release_all_under_closes_connections_inside_directory_only(tmp_path):
    """Regression test for #88347 — profile delete must break refcounted handles.

    ``close()`` alone can never free a database held by a live agent (refs >= 1),
    and on Windows an open SQLite handle makes rmtree of the profile directory
    fail with WinError 32. ``release_all_under`` force-closes exactly the shared
    connections under the doomed directory and leaves everything else alone.
    """
    from plugins.memory.holographic.store import MemoryStore

    profile_dir = tmp_path / "profiles" / "default-2"
    profile_dir.mkdir(parents=True)
    inside = MemoryStore(db_path=profile_dir / "memory_store.db", hrr_dim=64)
    outside = MemoryStore(db_path=tmp_path / "other" / "memory_store.db", hrr_dim=64)
    inside_conn = inside._conn
    outside_conn = outside._conn
    # Both live before the release; the inside one is held "forever" (refs=1,
    # like a live agent's provider — nobody calls close()).
    inside_conn.execute("SELECT 1").fetchone()
    outside_conn.execute("SELECT 1").fetchone()

    try:
        released = MemoryStore.release_all_under(profile_dir)
        assert released == 1

        # The doomed connection is really closed despite refs > 0 ...
        with pytest.raises(sqlite3.ProgrammingError):
            inside_conn.execute("SELECT 1")
        # ... and the registry entry is gone, so a second release finds
        # nothing (the CLI-process no-op case).
        assert str((profile_dir / "memory_store.db").resolve()) not in MemoryStore._shared
        assert MemoryStore.release_all_under(profile_dir) == 0
        # A new store on the same path reopens fresh instead of reusing
        # the dead connection.
        reopened = MemoryStore(db_path=profile_dir / "memory_store.db", hrr_dim=64)
        reopened._conn.execute("SELECT 1").fetchone()

        # The sibling outside the directory is untouched.
        outside_conn.execute("SELECT 1").fetchone()
    finally:
        inside.close()
        reopened.close()
        outside.close()




def test_stale_holder_close_does_not_evict_fresh_registry_entry(tmp_path):
    """Follow-up to #88347 — a stale holder's late ``close()`` must be inert.

    After ``release_all_under`` force-closes a profile's connection, a store
    re-created on the same path registers a FRESH shared entry under the same
    key. If the stale holder (whose entry was force-closed) then calls
    ``close()``, it must not pop the fresh entry out of the registry — that
    would let a third store open a SECOND connection to the same database and
    silently reintroduce the multi-writer contention the registry prevents.
    """
    from plugins.memory.holographic.store import MemoryStore

    profile_dir = tmp_path / "profiles" / "default-2"
    profile_dir.mkdir(parents=True)
    db_path = profile_dir / "memory_store.db"

    stale = MemoryStore(db_path=db_path, hrr_dim=64)
    assert MemoryStore.release_all_under(profile_dir) == 1

    fresh = MemoryStore(db_path=db_path, hrr_dim=64)
    key = fresh._key
    fresh_entry = MemoryStore._shared[key]

    # The stale holder's late close must leave the fresh entry registered...
    stale.close()
    assert MemoryStore._shared.get(key) is fresh_entry
    # ...and a third store must attach to the SAME shared connection.
    third = MemoryStore(db_path=db_path, hrr_dim=64)
    try:
        assert third._conn is fresh._conn
    finally:
        third.close()
        fresh.close()
    # Normal last-holder close still evicts its own entry.
    assert key not in MemoryStore._shared
