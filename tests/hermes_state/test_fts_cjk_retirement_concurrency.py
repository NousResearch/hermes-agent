"""Retirement is atomic across connections and health checks leave writers free."""

import sqlite3

import pytest

import hermes_state_repair as repair
from hermes_state import SessionDB
from hermes_state_fts import drop_retired_cjk_index
from tests.hermes_state import test_fts_cjk_index_retirement as retirement
from tests.hermes_state.test_fts_cjk_index_retirement import (
    _cjk_residue,
    _store_snapshot,
)

cjk_db = retirement.cjk_db


def test_retirement_excludes_an_interleaved_retirement(cjk_db):
    competing_results = []

    class InterleavedCursor(sqlite3.Cursor):
        def execute(self, sql, parameters=()):
            if sql.startswith("ALTER TABLE messages_fts_cjk_data RENAME"):
                with sqlite3.connect(cjk_db, isolation_level=None, timeout=0) as second:
                    try:
                        competing_results.append(drop_retired_cjk_index(second.cursor()))
                    except sqlite3.OperationalError as exc:
                        competing_results.append(str(exc))
            return super().execute(sql, parameters)

    with sqlite3.connect(cjk_db, isolation_level=None) as first:
        assert drop_retired_cjk_index(first.cursor(factory=InterleavedCursor))
        assert not first.in_transaction
    assert competing_results == ["database is locked"]
    SessionDB(db_path=cjk_db).close()
    assert _cjk_residue(cjk_db) == []


def test_repair_integrity_scan_allows_a_concurrent_writer(cjk_db, monkeypatch):
    before = _store_snapshot(cjk_db)
    connect = repair._connect_repair_durable
    writes = []

    def trace(sql):
        if sql.upper() == "PRAGMA INTEGRITY_CHECK":
            with sqlite3.connect(cjk_db, timeout=0) as writer:
                try:
                    writer.execute("UPDATE sessions SET source = source WHERE id = 's1'")
                    writer.commit()
                    writes.append("ok")
                except sqlite3.OperationalError as exc:
                    writes.append(str(exc))

    def traced_connect(*args, **kwargs):
        conn = connect(*args, **kwargs)
        conn.set_trace_callback(trace)
        return conn

    monkeypatch.setattr(repair, "_connect_repair_durable", traced_connect)
    assert repair._db_opens_cleanly(cjk_db) is None
    assert writes == ["ok"]
    assert _store_snapshot(cjk_db) == before


@pytest.mark.parametrize("outer_transaction", [False, True])
def test_failed_retirement_preserves_the_store(cjk_db, outer_transaction):
    before = _store_snapshot(cjk_db)

    class FailingCursor(sqlite3.Cursor):
        def execute(self, sql, parameters=()):
            if sql.startswith("ALTER TABLE messages_fts_cjk_data RENAME"):
                raise sqlite3.OperationalError("injected rename failure")
            return super().execute(sql, parameters)

    with sqlite3.connect(cjk_db, isolation_level=None) as conn:
        if outer_transaction:
            conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(sqlite3.OperationalError, match="injected rename failure"):
            drop_retired_cjk_index(conn.cursor(factory=FailingCursor))
        assert conn.in_transaction is outer_transaction
        if outer_transaction:
            conn.rollback()
    assert _store_snapshot(cjk_db) == before


def test_successful_retirement_leaves_the_callers_transaction_open(cjk_db):
    before = _store_snapshot(cjk_db)
    with sqlite3.connect(cjk_db, isolation_level=None) as conn:
        conn.execute("BEGIN IMMEDIATE")
        assert drop_retired_cjk_index(conn.cursor())
        assert conn.in_transaction
        conn.rollback()
    assert _store_snapshot(cjk_db) == before


def test_snapshot_failure_is_reported_without_mutating_the_store(cjk_db, monkeypatch):
    before = _store_snapshot(cjk_db)

    def fail_snapshot(*args, **kwargs):
        raise OSError("snapshot storage unavailable")

    monkeypatch.setattr(repair, "_copy_database_snapshot", fail_snapshot)
    assert repair._db_opens_cleanly(cjk_db) == "snapshot storage unavailable"
    assert _store_snapshot(cjk_db) == before


def test_retirement_rechecks_after_acquiring_the_write_lock(cjk_db):
    class InterleavedCursor(sqlite3.Cursor):
        def execute(self, sql, parameters=()):
            if sql == "BEGIN IMMEDIATE":
                with sqlite3.connect(cjk_db, isolation_level=None) as second:
                    assert drop_retired_cjk_index(second.cursor())
            return super().execute(sql, parameters)

    with sqlite3.connect(cjk_db, isolation_level=None) as first:
        assert not drop_retired_cjk_index(first.cursor(factory=InterleavedCursor))
        assert not first.in_transaction
    assert _cjk_residue(cjk_db) == []


def test_a_full_temporary_volume_is_not_reported_as_damage(cjk_db, monkeypatch):
    """The probe judges a copy of the store; a copy that does not fit says nothing about the store itself."""
    before = _store_snapshot(cjk_db)

    def full_volume(*args, **kwargs):
        raise sqlite3.OperationalError("database or disk is full")

    monkeypatch.setattr(repair, "_copy_database_snapshot", full_volume)
    assert repair._db_opens_cleanly(cjk_db) is None
    assert _store_snapshot(cjk_db) == before

