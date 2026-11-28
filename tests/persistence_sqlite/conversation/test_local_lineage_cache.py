"""The SQLite listings reuse a local lineage projection while the store is unchanged, and never reuse one across a
committed write, an open transaction or a write that landed during the scan."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from persistence.conversation.sessions import NewSession, SessionCountQuery, SessionQuery, SessionWindow
from persistence.sqlite.conversation._store.local import POLICY_PREFIX, _cached_local_lineage_index
from tests.persistence_sqlite.conftest import store_for

RECEIPT = json.dumps({"session_id": "owner", "lineage": ["owner", "head"]})


@pytest.fixture
def store(backend, scope_a):
    store = backend.open(scope_a)
    store.sessions.create(NewSession("owner", "cli"))
    store.sessions.create(NewSession("head", "cli"))
    _write("UPDATE sessions SET end_reason='session_reset' WHERE id='owner'", scope=scope_a)
    _write("INSERT INTO state_meta(key,value) VALUES(?,?)", (POLICY_PREFIX + "owner", RECEIPT), scope=scope_a)
    return store


def _write(sql, params=(), *, scope):
    with sqlite3.connect(store_for(scope)) as connection:
        connection.execute(sql, params)


def _listed(store):
    return [row.session.session_id for row in store.sessions.list_sessions(SessionQuery(), SessionWindow(limit=10))]


def _receipt_scans(statements):
    return [sql for sql in statements if "FROM state_meta" in sql and "GLOB" in sql and POLICY_PREFIX in sql]


def test_unchanged_session_lists_and_counts_scan_receipts_once(store):
    statements = []
    with store.holder()._read_ctx() as conn:
        conn.set_trace_callback(statements.append)
    try:
        for _ in range(5):
            assert _listed(store) == ["head"]
            assert store.sessions.count_sessions(SessionCountQuery(exclude_children=True)) == 1
    finally:
        with store.holder()._read_ctx() as conn:
            conn.set_trace_callback(None)
    assert len(_receipt_scans(statements)) == 1


@pytest.mark.parametrize("mutation", ["extend", "retire", "end_reason", "delete_head"])
def test_committed_writer_changes_refresh_the_pooled_reader(store, scope_a, mutation):
    assert _listed(store) == ["head"]
    if mutation == "extend":
        store.sessions.create(NewSession("tail", "cli"))
        _write("UPDATE sessions SET end_reason='session_reset' WHERE id='head'", scope=scope_a)
        _write("UPDATE state_meta SET value=? WHERE key=?",
               (json.dumps({"session_id": "owner", "lineage": ["owner", "head", "tail"]}), POLICY_PREFIX + "owner"),
               scope=scope_a)
        expected = {"tail"}
    elif mutation == "retire":
        _write("DELETE FROM state_meta WHERE key=?", (POLICY_PREFIX + "owner",), scope=scope_a)
        expected = {"owner", "head"}
    elif mutation == "end_reason":
        _write("UPDATE sessions SET end_reason='compression' WHERE id='owner'", scope=scope_a)
        expected = {"owner", "head"}
    else:
        _write("DELETE FROM sessions WHERE id='head'", scope=scope_a)
        expected = set()
    assert set(_listed(store)) == expected
    assert store.sessions.count_sessions(SessionCountQuery(exclude_children=True)) == len(expected)


@pytest.mark.parametrize("finish", ["commit", "rollback"])
def test_same_connection_transactions_do_not_publish_intermediate_results(store, finish):
    holder = store.holder()
    with holder._lock:
        conn = holder._conn
        before = _cached_local_lineage_index(holder, conn)
        conn.execute("BEGIN")
        conn.execute("DELETE FROM state_meta WHERE key=?", (POLICY_PREFIX + "owner",))
        assert _cached_local_lineage_index(holder, conn) == (set(), {})
        getattr(conn, finish)()
        expected = (set(), {}) if finish == "commit" else before
        assert _cached_local_lineage_index(holder, conn) == expected
        # A same-connection autocommit write invalidates too (data_version alone cannot see it).
        conn.execute("INSERT OR REPLACE INTO state_meta(key,value) VALUES(?,?)", (POLICY_PREFIX + "owner", RECEIPT))
        assert _cached_local_lineage_index(holder, conn) == before


def test_read_snapshot_does_not_outlive_its_transaction(store, scope_a):
    holder = store.holder()
    with holder._read_ctx() as reader:
        before = _cached_local_lineage_index(holder, reader)
        reader.execute("BEGIN")
        assert _cached_local_lineage_index(holder, reader) == before
        _write("DELETE FROM state_meta WHERE key=?", (POLICY_PREFIX + "owner",), scope=scope_a)
        assert _cached_local_lineage_index(holder, reader) == before
        reader.rollback()
        assert _cached_local_lineage_index(holder, reader) == (set(), {})


def test_commit_during_projection_is_not_cached_as_a_stable_result(store, scope_a):
    holder = store.holder()
    statements = []
    changed = False
    with holder._read_ctx() as reader:
        def trace(sql):
            nonlocal changed
            statements.append(sql)
            if not changed and "SELECT id,end_reason" in sql:
                changed = True
                _write("DELETE FROM state_meta WHERE key=?", (POLICY_PREFIX + "owner",), scope=scope_a)
        reader.set_trace_callback(trace)
        try:
            _cached_local_lineage_index(holder, reader)
            assert changed
            assert _cached_local_lineage_index(holder, reader) == (set(), {})
            assert _cached_local_lineage_index(holder, reader) == (set(), {})
        finally:
            reader.set_trace_callback(None)
    assert len(_receipt_scans(statements)) == 2


def test_callers_cannot_mutate_the_cached_projection(store):
    holder = store.holder()
    with holder._read_ctx() as reader:
        superseded, heads = _cached_local_lineage_index(holder, reader)
        superseded.clear()
        heads["head"][1].append("foreign-session")
        heads.clear()
        assert _cached_local_lineage_index(holder, reader) == ({"owner"}, {"head": ("owner", ["owner", "head"])})


def test_connection_churn_is_bounded_and_other_stores_stay_isolated(store, backend, scope_b):
    holder = store.holder()
    conns = [sqlite3.connect(f"file:{holder.db_path}?mode=ro", uri=True) for _ in range(12)]
    try:
        for conn in conns:
            assert _cached_local_lineage_index(holder, conn)[0] == {"owner"}
        cache, _lock = holder._local_lineage_cache
        assert len(cache) <= 8
        other = backend.open(scope_b).holder()
        with other._read_ctx() as conn:
            assert _cached_local_lineage_index(other, conn) == (set(), {})
    finally:
        for conn in conns:
            conn.close()


def test_concurrent_lists_keep_the_same_projection(store):
    barrier = Barrier(6)

    def read(_):
        barrier.wait(timeout=10)
        return [set(_listed(store)) for _ in range(5)]

    with ThreadPoolExecutor(max_workers=6) as pool:
        for result in pool.map(read, range(6)):
            assert result == [{"head"}] * 5
