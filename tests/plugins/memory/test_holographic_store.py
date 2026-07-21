"""Tests for the holographic MemoryStore shared-connection registry.

MemoryStore instances pointing at the same database file must share one
process-wide SQLite connection and one re-entrant lock. Multiple providers
coexist in a single process (the main agent plus every delegate_task
subagent); when each instance owned a private connection they raced as
independent WAL writers and intermittently failed with "database is locked".

Covers: connection sharing/refcounting, close() semantics, cross-instance
visibility, concurrent multi-instance writers, and write-lock release after
a failed write.
"""

import sqlite3
import threading

import pytest

from plugins.memory.holographic.store import MemoryStore


@pytest.fixture(autouse=True)
def _clean_shared_registry():
    """Each test starts and ends with an empty shared-connection registry."""
    # Drop any leakage from earlier tests in the same process.
    for entry in list(MemoryStore._shared.values()):
        try:
            entry["conn"].close()
        except sqlite3.Error:
            pass
    MemoryStore._shared.clear()
    yield
    leaked = list(MemoryStore._shared)
    for entry in list(MemoryStore._shared.values()):
        try:
            entry["conn"].close()
        except sqlite3.Error:
            pass
    MemoryStore._shared.clear()
    assert not leaked, f"test leaked shared connections: {leaked}"


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "memory_store.db"


class TestSharedConnection:
    def test_same_path_shares_one_connection(self, db_path):
        a = MemoryStore(db_path)
        b = MemoryStore(db_path)
        try:
            assert a._conn is b._conn
            assert a._lock is b._lock
            assert len(MemoryStore._shared) == 1
            assert MemoryStore._shared[str(a.db_path)]["refs"] == 2
        finally:
            a.close()
            b.close()

    def test_different_paths_get_distinct_connections(self, tmp_path):
        a = MemoryStore(tmp_path / "one.db")
        b = MemoryStore(tmp_path / "two.db")
        try:
            assert a._conn is not b._conn
            assert len(MemoryStore._shared) == 2
        finally:
            a.close()
            b.close()

    def test_symlinked_path_shares_connection(self, tmp_path):
        """A symlink to the same DB file must hit the same registry entry —
        otherwise two connections to one file silently reintroduce the
        multi-writer contention the registry exists to prevent."""
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        link_dir = tmp_path / "link"
        link_dir.symlink_to(real_dir)

        a = MemoryStore(real_dir / "memory_store.db")
        b = MemoryStore(link_dir / "memory_store.db")
        try:
            assert a._conn is b._conn
            assert len(MemoryStore._shared) == 1
        finally:
            a.close()
            b.close()

    def test_writes_visible_across_instances(self, db_path):
        a = MemoryStore(db_path)
        b = MemoryStore(db_path)
        try:
            fact_id = a.add_fact("Hermes likes shared connections", category="test")
            facts = b.list_facts(category="test")
            assert [f["fact_id"] for f in facts] == [fact_id]
        finally:
            a.close()
            b.close()

    def test_schema_initialised_once_per_connection(self, db_path):
        a = MemoryStore(db_path)
        b = MemoryStore(db_path)  # must not re-run schema init / WAL probe
        try:
            assert MemoryStore._shared[str(a.db_path)]["ready"] is True
            b.add_fact("schema still works")
        finally:
            a.close()
            b.close()


class TestCloseSemantics:
    def test_closing_one_instance_keeps_sibling_alive(self, db_path):
        a = MemoryStore(db_path)
        b = MemoryStore(db_path)
        a.close()
        try:
            # The shared connection must survive the sibling's close().
            fact_id = b.add_fact("survivor write")
            assert fact_id > 0
        finally:
            b.close()

    def test_last_close_releases_connection(self, db_path):
        a = MemoryStore(db_path)
        b = MemoryStore(db_path)
        conn = a._conn
        a.close()
        b.close()
        assert MemoryStore._shared == {}
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")

    def test_close_is_idempotent(self, db_path):
        a = MemoryStore(db_path)
        b = MemoryStore(db_path)
        a.close()
        a.close()  # double close must not steal b's reference
        try:
            b.add_fact("still alive after double close")
            assert MemoryStore._shared[str(b.db_path)]["refs"] == 1
        finally:
            b.close()

    def test_context_manager_releases_reference(self, db_path):
        with MemoryStore(db_path) as store:
            store.add_fact("context managed")
        assert MemoryStore._shared == {}

    def test_reopen_after_full_close(self, db_path):
        with MemoryStore(db_path) as store:
            store.add_fact("first lifetime")
        with MemoryStore(db_path) as store:
            facts = store.list_facts()
        assert [f["content"] for f in facts] == ["first lifetime"]


class TestConcurrency:
    def test_concurrent_multi_instance_writers(self, db_path):
        """Many instances writing from many threads must never hit
        'database is locked' — the failure mode of per-instance connections."""
        n_threads, n_facts = 8, 15
        errors: list[BaseException] = []

        def writer(idx: int) -> None:
            store = MemoryStore(db_path)
            try:
                for i in range(n_facts):
                    store.add_fact(f"fact thread={idx} seq={i}", category="load")
            except BaseException as exc:  # noqa: BLE001 - recorded for assert
                errors.append(exc)
            finally:
                store.close()

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"concurrent writers failed: {errors[:3]}"
        with MemoryStore(db_path) as store:
            facts = store.list_facts(category="load", limit=500)
        assert len(facts) == n_threads * n_facts
        assert MemoryStore._shared == {}

    def test_failed_write_does_not_pin_write_lock(self, db_path, monkeypatch):
        """A write that raises mid-method must not leave an open transaction
        holding the SQLite write lock (autocommit isolation_level=None)."""
        broken = MemoryStore(db_path)
        sibling = MemoryStore(db_path)
        try:
            monkeypatch.setattr(
                MemoryStore,
                "_rebuild_bank",
                lambda self, category: (_ for _ in ()).throw(RuntimeError("boom")),
            )
            with pytest.raises(RuntimeError, match="boom"):
                broken.add_fact("write that fails after the INSERT")
            monkeypatch.undo()

            # No dangling transaction: the connection reports autocommit state
            # and the sibling can write immediately.
            assert broken._conn.in_transaction is False
            sibling.add_fact("sibling write right after the failure")
        finally:
            broken.close()
            sibling.close()


class TestProviderShutdown:
    """The provider's shutdown() must release its shared connection, not just
    drop the reference. Leaving finalization to GC keeps the connection (and
    its write lock) alive on a long-running gateway, which is exactly the
    "database is locked" contention the shared-connection registry removes."""

    def test_shutdown_releases_shared_connection(self, db_path):
        from plugins.memory.holographic import HolographicMemoryProvider

        provider = HolographicMemoryProvider(config={"db_path": str(db_path)})
        provider.initialize("session-shutdown")
        assert MemoryStore._shared[str(db_path)]["refs"] == 1

        provider.shutdown()

        assert provider._store is None
        assert MemoryStore._shared == {}

    def test_shutdown_keeps_sibling_provider_alive(self, db_path):
        from plugins.memory.holographic import HolographicMemoryProvider

        a = HolographicMemoryProvider(config={"db_path": str(db_path)})
        b = HolographicMemoryProvider(config={"db_path": str(db_path)})
        a.initialize("session-a")
        b.initialize("session-b")
        assert MemoryStore._shared[str(db_path)]["refs"] == 2

        a.shutdown()
        # Sibling still holds a live, writable connection.
        assert MemoryStore._shared[str(db_path)]["refs"] == 1
        assert b._store is not None
        b._store.add_fact("write after sibling shutdown")
        b.shutdown()
        assert MemoryStore._shared == {}


class TestProviderRetrieverDimAgreement:
    """The provider must not construct FactRetriever with the raw configured
    hrr_dim — MemoryStore may have adopted a different persisted dim, and the
    retriever must agree with the store or every query is rejected as a
    dimension mismatch (issue #68682 follow-up)."""

    def test_provider_retriever_uses_store_adopted_dim(self, db_path):
        from plugins.memory.holographic import HolographicMemoryProvider

        # First session: persist hrr_dim=256.
        seed = HolographicMemoryProvider(config={"db_path": str(db_path), "hrr_dim": 256})
        seed.initialize("seed-session")
        seed._store.add_fact("Peppi works on the backend team.", category="project")
        seed.shutdown()

        # Second session: config now says 1024, but the DB already has 256.
        provider = HolographicMemoryProvider(config={"db_path": str(db_path), "hrr_dim": 1024})
        try:
            provider.initialize("later-session")
            assert provider._store.hrr_dim == 256
            assert provider._retriever.hrr_dim == 256
            # Must not raise, and must actually find the seeded fact.
            results = provider._retriever.search("backend", min_trust=0.0)
            assert isinstance(results, list)
        finally:
            provider.shutdown()


class TestHrrDimPersistence:
    """A stored hrr_dim must survive process restarts. Before this fix,
    hrr_dim came only from the constructor/config, so a config change
    between sessions silently mixed vector dimensions across facts and
    crashed similarity() (issue #68682)."""

    def test_fresh_store_persists_constructor_dim(self, db_path):
        store = MemoryStore(db_path, hrr_dim=512)
        try:
            row = store._conn.execute(
                "SELECT value FROM _meta WHERE key = 'hrr_dim'"
            ).fetchone()
            assert row is not None
            assert int(row["value"]) == 512
        finally:
            store.close()

    def test_reopen_adopts_persisted_dim_over_new_config(self, db_path):
        store = MemoryStore(db_path, hrr_dim=256)
        store.close()

        # Simulate a later session with a *different* configured hrr_dim.
        store2 = MemoryStore(db_path, hrr_dim=1024)
        try:
            assert store2.hrr_dim == 256
        finally:
            store2.close()

    def test_rebuild_all_vectors_updates_persisted_dim(self, db_path):
        store = MemoryStore(db_path, hrr_dim=256)
        try:
            store.add_fact("a fact to migrate")
            store.rebuild_all_vectors(dim=1024)
            row = store._conn.execute(
                "SELECT value FROM _meta WHERE key = 'hrr_dim'"
            ).fetchone()
            assert int(row["value"]) == 1024
        finally:
            store.close()

        store2 = MemoryStore(db_path, hrr_dim=999)
        try:
            assert store2.hrr_dim == 1024
        finally:
            store2.close()

    def test_rebuild_bank_skips_corrupt_vector_without_crashing(self, db_path):
        """A legacy mixed-dim DB must not crash add_fact/_rebuild_bank — a
        stray vector at the wrong dimension must be skipped, counted, and
        excluded from the bank's fact_count, not fed into hrr.bundle()."""
        import numpy as np

        store = MemoryStore(db_path, hrr_dim=1024)
        try:
            store.add_fact("A fact with a valid 1024-dim vector.", category="mix")
            # Simulate a leftover fact vector from a prior 256-dim session by
            # writing a mismatched-length blob directly.
            bad_vec = np.zeros(256, dtype=np.float64)
            store._conn.execute(
                "INSERT INTO facts (content, category, hrr_vector) VALUES (?, ?, ?)",
                ("A corrupt fact from an old session.", "mix", bad_vec.tobytes()),
            )
            store._conn.commit()

            # Must not raise (bundle() broadcasting different-length vectors).
            store._rebuild_bank("mix")

            row = store._conn.execute(
                "SELECT fact_count FROM memory_banks WHERE bank_name = 'cat:mix'"
            ).fetchone()
            assert row["fact_count"] == 1  # only the accepted 1024-dim vector
        finally:
            store.close()

    def test_second_instance_on_shared_connection_adopts_dim(self, db_path):
        """A second MemoryStore attaching to an already-initialised shared
        connection must adopt the same persisted dim as the first instance,
        not keep its own constructor value. _init_db (and therefore dim
        adoption) only runs once per shared connection — the fix must sync
        the dim onto every later attacher, not just the first."""
        a = MemoryStore(db_path, hrr_dim=256)
        try:
            b = MemoryStore(db_path, hrr_dim=1024)
            try:
                assert b.hrr_dim == 256
                assert a.hrr_dim == b.hrr_dim
            finally:
                b.close()
        finally:
            a.close()
