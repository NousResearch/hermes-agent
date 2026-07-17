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

from plugins.memory.holographic.retrieval import FactRetriever
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


@pytest.fixture
def store(db_path):
    """A MemoryStore that closes on teardown so the shared registry stays clean
    (the autouse fixture asserts no connection leaks)."""
    s = MemoryStore(db_path)
    try:
        yield s
    finally:
        s.close()


class TestSupersede:
    """supersede() inserts a corrected fact, retires the old one, and records
    lineage as one all-or-nothing transaction, without destroying old wording."""

    def test_returns_new_live_fact_and_retires_old(self, store):
        old_id = store.add_fact("Deploy runs on Mondays", category="project")
        new_id = store.supersede(old_id, "Deploy runs on Fridays")
        assert new_id != old_id
        new_row = store._conn.execute(
            "SELECT superseded_at FROM facts WHERE fact_id = ?", (new_id,)
        ).fetchone()
        old_row = store._conn.execute(
            "SELECT superseded_at FROM facts WHERE fact_id = ?", (old_id,)
        ).fetchone()
        assert new_row["superseded_at"] is None      # new version is live
        assert old_row["superseded_at"] is not None  # old version retired

    def test_records_lineage_edge(self, store):
        old_id = store.add_fact("Deploy runs on Mondays")
        new_id = store.supersede(old_id, "Deploy runs on Fridays")
        row = store._conn.execute(
            "SELECT 1 FROM fact_supersedes WHERE new_id = ? AND old_id = ?",
            (new_id, old_id),
        ).fetchone()
        assert row is not None

    def test_old_content_preserved(self, store):
        old_id = store.add_fact("Deploy runs on Mondays")
        store.supersede(old_id, "Deploy runs on Fridays")
        row = store._conn.execute(
            "SELECT content FROM facts WHERE fact_id = ?", (old_id,)
        ).fetchone()
        assert row["content"] == "Deploy runs on Mondays"

    def test_inherits_old_category_by_default(self, store):
        old_id = store.add_fact("Deploy runs on Mondays", category="project")
        new_id = store.supersede(old_id, "Deploy runs on Fridays")
        cat = store._conn.execute(
            "SELECT category FROM facts WHERE fact_id = ?", (new_id,)
        ).fetchone()["category"]
        assert cat == "project"

    def test_category_override(self, store):
        old_id = store.add_fact("Deploy runs on Mondays", category="project")
        new_id = store.supersede(old_id, "Deploy runs on Fridays", category="tool")
        cat = store._conn.execute(
            "SELECT category FROM facts WHERE fact_id = ?", (new_id,)
        ).fetchone()["category"]
        assert cat == "tool"

    def test_missing_old_raises_keyerror(self, store):
        with pytest.raises(KeyError):
            store.supersede(9999, "Deploy runs on Fridays")

    def test_empty_content_raises(self, store):
        old_id = store.add_fact("Deploy runs on Mondays")
        with pytest.raises(ValueError):
            store.supersede(old_id, "   ")

    def test_identical_content_raises(self, store):
        old_id = store.add_fact("Deploy runs on Mondays")
        with pytest.raises(ValueError):
            store.supersede(old_id, "  Deploy runs on Mondays  ")

    def test_rejects_duplicate_target_content(self, store):
        # facts.content is UNIQUE, so add_fact() would dedupe onto an existing
        # row and return its id. supersede must refuse rather than retire old_id
        # against, and record lineage to, a pre-existing (possibly retired) fact.
        old_id = store.add_fact("Deploy runs on Mondays", category="project")
        other_id = store.add_fact("Deploy runs on Fridays", category="general")
        with pytest.raises(ValueError):
            store.supersede(old_id, "Deploy runs on Fridays")
        # old_id stays live and no lineage edge was written to other_id.
        old_row = store._conn.execute(
            "SELECT superseded_at FROM facts WHERE fact_id = ?", (old_id,)
        ).fetchone()
        assert old_row["superseded_at"] is None
        n_edges = store._conn.execute(
            "SELECT COUNT(*) AS c FROM fact_supersedes"
        ).fetchone()["c"]
        assert n_edges == 0
        assert other_id != old_id  # sanity: the collision was a distinct fact

    def test_rolls_back_atomically_on_failure(self, store):
        # Inject a failure at the lineage INSERT (drop its table) and prove the
        # whole transition rolls back: no new version persists and old_id is not
        # retired. On a non-transactional write path the new fact and the
        # retirement would survive as a partial supersede.
        old_id = store.add_fact("Deploy runs on Mondays")
        store._conn.execute("DROP TABLE fact_supersedes")
        with pytest.raises(sqlite3.OperationalError):
            store.supersede(old_id, "Deploy runs on Fridays")
        old_row = store._conn.execute(
            "SELECT superseded_at FROM facts WHERE fact_id = ?", (old_id,)
        ).fetchone()
        assert old_row["superseded_at"] is None  # not retired
        n_new = store._conn.execute(
            "SELECT COUNT(*) AS c FROM facts WHERE content = ?",
            ("Deploy runs on Fridays",),
        ).fetchone()["c"]
        assert n_new == 0  # new version rolled back


class TestSupersededRecallFilter:
    """A superseded fact must vanish from default recall paths while the live
    version still surfaces."""

    def _two_versions(self, store):
        old_id = store.add_fact("Project Hermes runs on the old server alpha")
        new_id = store.supersede(old_id, "Project Hermes runs on the new server beta")
        return old_id, new_id

    def test_search_facts_excludes_superseded(self, store):
        old_id, new_id = self._two_versions(store)
        ids = [f["fact_id"] for f in store.search_facts("Project Hermes server")]
        assert old_id not in ids
        assert new_id in ids

    def test_list_facts_excludes_superseded(self, store):
        old_id, new_id = self._two_versions(store)
        ids = [f["fact_id"] for f in store.list_facts()]
        assert old_id not in ids
        assert new_id in ids

    def test_retriever_search_excludes_superseded(self, store):
        old_id, new_id = self._two_versions(store)
        ids = [f["fact_id"] for f in FactRetriever(store).search("Project Hermes server")]
        assert old_id not in ids
        assert new_id in ids


class TestRemoveFactLineageCleanup:
    """remove_fact() drops the fact's fact_supersedes edges so no dangling
    lineage rows remain."""

    def test_removing_new_id_clears_lineage(self, store):
        old_id = store.add_fact("Deploy runs on Mondays")
        new_id = store.supersede(old_id, "Deploy runs on Fridays")
        store.remove_fact(new_id)
        n = store._conn.execute(
            "SELECT COUNT(*) AS c FROM fact_supersedes WHERE new_id = ? OR old_id = ?",
            (new_id, new_id),
        ).fetchone()["c"]
        assert n == 0

    def test_removing_old_id_clears_lineage(self, store):
        old_id = store.add_fact("Deploy runs on Mondays")
        new_id = store.supersede(old_id, "Deploy runs on Fridays")
        store.remove_fact(old_id)
        n = store._conn.execute(
            "SELECT COUNT(*) AS c FROM fact_supersedes WHERE new_id = ? OR old_id = ?",
            (old_id, old_id),
        ).fetchone()["c"]
        assert n == 0


class TestTrace:
    """trace() walks the supersede chain backward, including retired versions."""

    def test_walks_supersede_chain_newest_first(self, store):
        v1 = store.add_fact("Deploy runs on Mondays")
        v2 = store.supersede(v1, "Deploy runs on Wednesdays")
        v3 = store.supersede(v2, "Deploy runs on Fridays")
        chain = FactRetriever(store).trace(v3)
        assert [r["fact_id"] for r in chain] == [v3, v2, v1]
        assert [r["depth"] for r in chain] == [0, 1, 2]

    def test_unknown_fact_returns_empty(self, store):
        assert FactRetriever(store).trace(9999) == []
