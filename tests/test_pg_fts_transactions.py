"""Canonical message writes survive failures in their optional search index.

SQLite executes real row writes and savepoints here; the driver boundary
models PostgreSQL's failed-transaction status and COMMIT-as-ROLLBACK behavior.
Live PostgreSQL counterparts live in test_pg_parity_smoke.py.
"""

import re
import sqlite3
from types import SimpleNamespace

import pytest

import hermes_state_postgres as pg


class _Cursor:
    def __init__(self, driver):
        self.driver = driver
        self.result = None

    def execute(self, sql, params=()):
        self.result = self.driver.execute(sql, params)
        return self

    def fetchone(self):
        return self.result.fetchone()

    @property
    def description(self):
        return self.result.description

    @property
    def rowcount(self):
        return self.result.rowcount


class _AbortingDriver:
    """Fault boundary with real transactional storage, not a SQL oracle."""

    def __init__(self, path, *, failure=None):
        psycopg = pytest.importorskip("psycopg")
        self.status = psycopg.pq.TransactionStatus
        self.error_type = psycopg.errors.ProgramLimitExceeded
        self.aborted_error = psycopg.errors.InFailedSqlTransaction
        self.db = sqlite3.connect(path, isolation_level=None)
        self.db.execute(
            "CREATE TABLE messages (id INTEGER PRIMARY KEY, content TEXT, "
            "tool_name TEXT, tool_calls TEXT, fts_content TEXT)"
        )
        self.info = SimpleNamespace(transaction_status=self.status.IDLE)
        self.failure = failure
        self.commit_calls = 0
        self.closed = False

    def cursor(self):
        return _Cursor(self)

    def execute(self, sql, params=()):
        recovering = sql.startswith("ROLLBACK")
        if self.info.transaction_status == self.status.INERROR and not recovering:
            raise self.aborted_error("current transaction is aborted")
        probing = "pg_catalog.pg_attribute" in sql
        indexing = sql.startswith("UPDATE messages SET fts_content = to_tsvector")
        if (probing and self.failure == "probe") or (indexing and self.failure == "index"):
            self.info.transaction_status = self.status.INERROR if self.db.in_transaction else self.status.IDLE
            raise self.error_type("injected derived-index failure")
        if probing:
            sql = "SELECT 1 FROM pragma_table_info('messages') WHERE name = 'fts_content'"
        elif indexing:
            sql = (
                "UPDATE messages SET fts_content = COALESCE(content, '') || ' ' || "
                "COALESCE(tool_name, '') || ' ' || COALESCE(tool_calls, '') WHERE id = ?"
            )
        elif sql == "BEGIN ISOLATION LEVEL SERIALIZABLE":
            sql = "BEGIN IMMEDIATE"
        sql = re.sub(r"%\((\w+)\)s", r":\1", sql).replace("%s", "?")
        try:
            result = self.db.execute(sql, params)
        except sqlite3.Error:
            if self.db.in_transaction:
                self.info.transaction_status = self.status.INERROR
            raise
        self.info.transaction_status = self.status.INTRANS if self.db.in_transaction else self.status.IDLE
        return result

    def commit(self):
        self.commit_calls += 1
        if self.info.transaction_status == self.status.INERROR:
            self.db.rollback()
        else:
            self.db.commit()
        self.info.transaction_status = self.status.IDLE

    def rollback(self):
        self.db.rollback()
        self.info.transaction_status = self.status.IDLE

    def close(self):
        self.db.close()
        self.closed = True


@pytest.fixture
def adapter(tmp_path):
    raw = _AbortingDriver(tmp_path / "messages.db")
    conn = pg._PostgresConnection(raw)
    try:
        yield conn, raw
    finally:
        conn.close()


@pytest.mark.parametrize("failure", ["probe", "index"])
@pytest.mark.parametrize("transaction", [False, True])
def test_optional_index_failure_preserves_insert(adapter, failure, transaction):
    conn, raw = adapter
    raw.failure = failure
    if transaction:
        conn.execute("BEGIN IMMEDIATE")
    cursor = conn.execute(
        "INSERT INTO messages (content, tool_name, tool_calls) VALUES (?, ?, ?)",
        ("canonical message", "terminal", None),
    )
    conn.commit()
    assert raw.db.execute(
        "SELECT content, fts_content FROM messages WHERE id = ?", (cursor.lastrowid,)
    ).fetchone() == ("canonical message", None)
    assert raw.info.transaction_status == raw.status.IDLE


@pytest.mark.parametrize("named", [False, True])
def test_index_hook_uses_insert_fields(adapter, named):
    conn, raw = adapter
    values = {"tool_name": "terminal", "content": "canonical message", "tool_calls": "tool metadata"}
    binds = ":tool_name, :content, :tool_calls" if named else "?, ?, ?"
    params = values if named else tuple(values.values())
    conn.execute("BEGIN IMMEDIATE")
    cursor = conn.execute(
        f"INSERT INTO messages (tool_name, content, tool_calls) VALUES ({binds})", params
    )
    conn.commit()
    assert raw.db.execute(
        "SELECT fts_content FROM messages WHERE id = ?", (cursor.lastrowid,)
    ).fetchone()[0] == "canonical message terminal tool metadata"


def test_aborted_transaction_cannot_report_successful_commit(adapter):
    conn, raw = adapter
    conn.execute("BEGIN IMMEDIATE")
    conn.execute("INSERT INTO messages (content) VALUES (?)", ("must roll back",))
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("SELECT * FROM missing_table")
    with pytest.raises(RuntimeError, match="transaction is aborted"):
        conn.commit()
    assert raw.commit_calls == 0
    conn.rollback()
    assert raw.db.execute("SELECT count(*) FROM messages").fetchone()[0] == 0


def test_backfill_completeness_observes_other_writers(adapter, tmp_path):
    conn, raw = adapter
    conn.execute("INSERT INTO messages (content) VALUES (?)", ("indexed message",))
    assert pg._fts_backfill_complete(conn)
    with sqlite3.connect(tmp_path / "messages.db") as other:
        other.execute("INSERT INTO messages (content) VALUES (?)", ("unindexed message",))
    assert not pg._fts_backfill_complete(conn)


def test_index_hook_handles_clone_and_literal_inserts(adapter):
    conn, raw = adapter
    conn.execute("BEGIN IMMEDIATE")
    original = conn.execute(
        "INSERT INTO messages (content, tool_name, tool_calls) VALUES ('literal message', ?, ?)",
        ("terminal", "tool metadata"),
    ).lastrowid
    conn.execute(
        "INSERT INTO messages (content, tool_name, tool_calls, fts_content) "
        "SELECT content, tool_name, tool_calls, fts_content FROM messages WHERE id = ?",
        (original,),
    )
    conn.commit()
    rows = raw.db.execute("SELECT content, fts_content FROM messages ORDER BY id").fetchall()
    assert len(rows) == 2
    assert all(row == ("literal message", "literal message terminal tool metadata") for row in rows)


def test_named_bind_names_need_not_match_column_names(adapter):
    conn, raw = adapter
    conn.execute(
        "INSERT INTO messages (content, tool_name) VALUES (:body, :name)",
        {"body": "named message", "name": "terminal"},
    )
    assert raw.db.execute("SELECT fts_content FROM messages").fetchone()[0] == "named message terminal "
