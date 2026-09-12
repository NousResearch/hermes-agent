"""Exercise PostgreSQL maintenance through real SessionDB construction and writes.

Only the SQL connection boundary is replaced: a separate temporary SQLite
store supplies real rows/transactions, while the boundary models PostgreSQL
advisory-lock lifetime and refuses PRAGMAs. Engine/protocol parity belongs in
the PostgreSQL integration suite.
"""

from __future__ import annotations

import sqlite3
import threading
import time

import pytest

import hermes_state
import hermes_state_postgres


class _Server:
    def __init__(self, path):
        self.path = path
        self.connections = []
        self.statements = []
        self.lock_owner = None
        self.guard = threading.Lock()
        self.pause_prune = False
        self.prune_started = threading.Event()
        self.resume_prune = threading.Event()
        self.fail_once = None
        self.vacuums = 0

    def connect(self, dsn):
        conn = _Connection(self, dsn)
        self.connections.append(conn)
        return conn


class _Connection:
    def __init__(self, server, dsn):
        self.server = server
        self._dsn = dsn
        self.closed = False
        self.conn = sqlite3.connect(server.path, isolation_level=None, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")

    def execute(self, sql, params=()):
        upper = " ".join(sql.upper().split())
        self.server.statements.append(upper)
        if upper.startswith("PRAGMA"):
            raise AssertionError("PostgreSQL received a SQLite PRAGMA")
        if "PG_TRY_ADVISORY_XACT_LOCK" in upper:
            if self.server.fail_once == "claim":
                self.server.fail_once = None
                raise RuntimeError("maintenance claim failed")
            with self.server.guard:
                acquired = self.server.lock_owner in (None, self)
                if acquired:
                    self.server.lock_owner = self
            return self.conn.execute("SELECT ?", (acquired,))
        if upper.startswith("SELECT S.ID FROM SESSIONS S WHERE"):
            if self.server.fail_once == "prune":
                self.server.fail_once = None
                raise RuntimeError("maintenance prune failed")
            if self.server.pause_prune:
                self.server.pause_prune = False
                self.server.prune_started.set()
                if not self.server.resume_prune.wait(5):
                    raise AssertionError("test never released the prune operation")
        if upper == "VACUUM":
            assert not self.conn.in_transaction, "VACUUM cannot run inside a transaction"
            self.server.vacuums += 1
        # PostgreSQL row locks are tested by the engine suite. SQLite's write
        # transaction already serializes the actual row operations in this fixture.
        sql = sql.replace(" FOR UPDATE", "")
        return self.conn.execute(sql, params)

    def _release_lock(self):
        with self.server.guard:
            if self.server.lock_owner is self:
                self.server.lock_owner = None

    def commit(self):
        self.conn.commit()
        self._release_lock()

    def rollback(self):
        self.conn.rollback()
        self._release_lock()

    def close(self):
        self.conn.close()
        self.closed = True
        self._release_lock()


@pytest.fixture
def pg_store(tmp_path, monkeypatch):
    home = tmp_path / "pg_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH)
    remote_path = tmp_path / "remote.sqlite"
    # Real schema, metadata, and all ordinary SessionDB call paths remain live.
    with hermes_state.SessionDB(db_path=remote_path):
        pass
    server = _Server(remote_path)
    monkeypatch.setattr(hermes_state_postgres, "connect_postgres", server.connect)
    monkeypatch.setattr(
        hermes_state_postgres, "maybe_open_postgres",
        lambda read_only, schema_version, dsn_override=None: server.connect(dsn_override),
    )
    stores = []

    def open_store(profile_home=home):
        profile_home.mkdir(parents=True, exist_ok=True)
        db = hermes_state.SessionDB(
            db_path=profile_home / "state.db", postgres_dsn="postgresql://db.invalid/hermes",
        )
        stores.append(db)
        return db

    db = open_store()
    try:
        yield db, server, open_store
    finally:
        server.resume_prune.set()
        for store in stores:
            store.close()
        for conn in server.connections:
            if not conn.closed:
                conn.close()


def _old_session(db, sid, *, ended):
    db.create_session(sid, source="cli")
    old = time.time() - 120 * 86400
    db._write_sql("UPDATE sessions SET started_at = ? WHERE id = ?", (old, sid))
    if ended:
        db.end_session(sid, "done")


def _start(call):
    results = []

    def run():
        try:
            results.append(call())
        except BaseException as exc:
            results.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, results


def test_vacuum_completes_without_local_database_file(pg_store):
    db, server, _ = pg_store
    db.create_session("retained", source="cli")
    assert not db.db_path.exists()
    thread, results = _start(db.vacuum)
    thread.join(3)
    if thread.is_alive():
        # The old implementation reentered this non-reentrant lock during its
        # SQLite identity probe. Unblock the failing call so teardown can close
        # its real connection rather than leave a hung test worker behind.
        db._lock.release()
        thread.join(3)
        pytest.fail("PostgreSQL vacuum deadlocked while probing a local state.db")
    assert results == [0]
    assert server.vacuums == 1
    assert db.get_session("retained") is not None
    assert not any(sql.startswith("PRAGMA") for sql in server.statements)
    assert not db.db_path.exists()


def test_sqlite_page_metrics_are_unavailable_without_queries(pg_store):
    db, server, _ = pg_store
    before = len(server.statements)
    assert db.logical_size_bytes() is None
    assert db._freelist_ratio() is None
    assert len(server.statements) == before


def test_auto_maintenance_prunes_then_spares_freshly_closed_orphans(pg_store):
    db, server, _ = pg_store
    _old_session(db, "expired", ended=True)
    _old_session(db, "orphan", ended=False)
    db.create_session("active", source="cli")

    result = db.maybe_auto_prune_and_vacuum()

    assert result == {"skipped": False, "pruned": 1, "closed": 1,
                      "vacuumed": True, "freelist_ratio": None}
    assert db.get_session("expired") is None
    assert db.get_session("orphan")["end_reason"] == "startup_orphan_reap"
    assert db.get_session("active")["ended_at"] is None
    assert db.get_meta("last_vacuum") is not None
    assert db.maybe_auto_prune_and_vacuum()["skipped"] is True
    assert server.lock_owner is None
    assert all(conn.closed for conn in server.connections if conn is not db._conn)
    assert not list(db.db_path.parent.glob("state.db*"))


@pytest.mark.parametrize("same_handle", [False, True])
def test_maintenance_contender_skips_before_pruning(pg_store, tmp_path, same_handle):
    db, server, open_store = pg_store
    peer = db if same_handle else open_store(tmp_path / "another_host_home")
    _old_session(db, "expired", ended=True)
    server.pause_prune = True
    thread, results = _start(lambda: db.maybe_auto_prune_and_vacuum(vacuum=False))
    try:
        assert server.prune_started.wait(3)
        contender = peer.maybe_auto_prune_and_vacuum(vacuum=False)
        assert contender["skipped"] is True
        assert contender["pruned"] == 0
    finally:
        server.resume_prune.set()
        thread.join(3)
    assert not thread.is_alive()
    assert len(results) == 1 and isinstance(results[0], dict), results
    assert results[0]["pruned"] == 1
    assert server.lock_owner is None


@pytest.mark.parametrize("stage", ["claim", "prune"])
def test_failure_releases_maintenance_lock_and_allows_retry(pg_store, stage):
    db, server, _ = pg_store
    _old_session(db, "expired", ended=True)
    server.fail_once = stage
    failed = db.maybe_auto_prune_and_vacuum(vacuum=False)
    assert failed["error"] == f"maintenance {stage} failed"
    assert server.lock_owner is None
    assert db.get_session("expired") is not None
    assert db.get_meta("last_auto_prune") is None
    retried = db.maybe_auto_prune_and_vacuum(vacuum=False)
    assert retried["pruned"] == 1
    assert "error" not in retried
    assert all(conn.closed for conn in server.connections if conn is not db._conn)
