"""``_rebuild_table`` is one write transaction (port of qwibitai/nanoclaw#3766's idea:
hold the write lock across a migration so two openers cannot interleave).

The writer connection is autocommit, so RENAME / CREATE / copy / DROP used to commit one
by one. A sibling process opening state.db between RENAME and CREATE runs SCHEMA_SQL's
``CREATE TABLE IF NOT EXISTS`` first; the rebuilder's CREATE then fails with "table
already exists", the rows are stranded in the ``*_legacy`` copy and the live table is
empty — silent data loss on the gateway + CLI concurrent-open path.
"""

import sqlite3

import pytest

from hermes_state_schema import SessionSchemaMixin

DDL = "CREATE TABLE t (a INTEGER, b TEXT, c INTEGER NOT NULL DEFAULT 0)"
COPY = "INSERT INTO t (a, b) SELECT a, b FROM t_legacy"


def _db(tmp_path):
    conn = sqlite3.connect(tmp_path / "t.db", isolation_level=None, timeout=0.05)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE t (a INTEGER PRIMARY KEY, b TEXT)")
    conn.executemany("INSERT INTO t VALUES (?, ?)", [(1, "x"), (2, "y")])
    return conn


def test_sibling_opener_between_rename_and_create_cannot_strand_rows(tmp_path):
    conn = _db(tmp_path)
    sibling = sqlite3.connect(tmp_path / "t.db", isolation_level=None, timeout=0.05)
    seen = {}

    class InterleavingCursor(sqlite3.Cursor):
        def execute(self, sql, *args):
            result = super().execute(sql, *args)
            if sql.startswith("ALTER TABLE t RENAME"):
                # A sibling's schema bootstrap fires in the RENAME→CREATE gap. Its snapshot still
                # shows the uncommitted-away ``t``, so IF NOT EXISTS is a no-op, and any real write
                # must be refused (busy) while the rebuild holds the write lock — never applied.
                try:
                    sibling.execute("CREATE TABLE IF NOT EXISTS t (a INTEGER, b TEXT, c INTEGER)")
                    sibling.execute("INSERT INTO t (a, b) VALUES (3, 'z')")
                    seen["sibling"] = "applied"
                except sqlite3.OperationalError as exc:
                    seen["sibling"] = "busy" if ("locked" in str(exc) or "busy" in str(exc)) else str(exc)
            return result

    SessionSchemaMixin._rebuild_table(conn.cursor(InterleavingCursor), "t", "t_legacy", DDL, COPY)
    assert seen["sibling"] == "busy"
    assert not conn.in_transaction
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert names == {"t"}
    assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 2
    assert conn.execute("SELECT c FROM t WHERE a = 1").fetchone()[0] == 0  # new shape landed
    sibling.close()
    conn.close()


def test_rebuild_failure_rolls_back_rename(tmp_path):
    conn = _db(tmp_path)
    with pytest.raises(sqlite3.OperationalError):
        SessionSchemaMixin._rebuild_table(conn.cursor(), "t", "t_legacy", DDL, "INSERT INTO nope SELECT * FROM t_legacy")
    assert not conn.in_transaction
    assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 2
    assert conn.execute("SELECT name FROM sqlite_master WHERE name = 't_legacy'").fetchone() is None
    conn.close()
