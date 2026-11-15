"""The cron executions ledger must not take the schema write lock on every connection.

Under normal multi-job concurrency a per-connection DDL sequence spent the whole busy budget
of ``cron/executions.db`` on statements that changed nothing, and 7.1% of a day's executions
failed with ``database is locked`` before the job's payload ever ran (#155907). The tests
below pin the two halves of the fix: no ``ALTER`` on a table that already has the column, and
no DDL at all on a connection whose file is already at ``SCHEMA_VERSION``.
"""

import sqlite3

import pytest

from cron import executions
from hermes_cli import sqlite_util as su

_DDL = ("create table", "alter table", "create index", "create unique index", "drop table")


class _SqlTrace:
    """Every statement a connection runs, so a test can count the schema writes in it."""

    def __init__(self, conn):
        self.statements = []
        conn.set_trace_callback(self.statements.append)

    def ddl(self):
        return [s for s in self.statements if s.strip().lower().startswith(_DDL)]


@pytest.fixture()
def ledger_db(tmp_path, monkeypatch):
    path = tmp_path / "cron" / "executions.db"
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", path)
    from cron import incidents
    monkeypatch.setattr(incidents, "EXECUTIONS_FILE", path)
    return path


def test_add_column_does_not_alter_when_the_column_is_there():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (id TEXT PRIMARY KEY)")
    assert su.add_column_if_missing(conn, "t", "extra", "extra TEXT") is True
    trace = _SqlTrace(conn)
    # The second call must decide from the schema, not from a failed ALTER: on a real file the
    # ALTER would take the write lock before SQLite reports the duplicate.
    assert su.add_column_if_missing(conn, "t", "extra", "extra TEXT") is False
    assert trace.ddl() == [], f"a no-op migration still issued DDL: {trace.ddl()}"
    conn.close()


def test_first_connection_migrates_and_stamps_the_version(ledger_db):
    conn = executions._connect()
    try:
        assert su.get_user_version(conn) == executions.SCHEMA_VERSION
        columns = {r[1] for r in conn.execute("PRAGMA table_info(executions)")}
        assert {"delivery_outcome", "scheduled_instant", "progress_at"} <= columns
        # Both tables live in this file, so one gate has to migrate both.
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='cron_incidents'"
        ).fetchone() is not None
    finally:
        conn.close()


def test_later_connections_run_no_ddl_at_all(ledger_db, monkeypatch):
    """The DDL used to run *inside* open_db, so the trace has to cover the connect itself."""
    executions._connect().close()
    statements = []
    real_connect = sqlite3.connect

    def traced_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        conn.set_trace_callback(statements.append)
        return conn

    monkeypatch.setattr(sqlite3, "connect", traced_connect)
    conn = executions._connect()
    try:
        conn.execute(
            "INSERT INTO executions (id, job_id, source, process_id, pid, status, claimed_at) "
            "VALUES ('e1', 'j1', 'cron', 'p', 1, 'claimed', '2026-11-15T00:00:00Z')"
        )
        conn.commit()
    finally:
        conn.close()
    ddl = [s for s in statements if s.strip().lower().startswith(_DDL)]
    assert ddl == [], (
        "an up-to-date ledger still ran schema DDL while opening a connection: "
        f"{ddl}"
    )


def test_database_restored_from_an_older_backup_is_migrated(ledger_db):
    """A backup carries no user_version, so the gate re-runs the migration rather than trusting
    an in-process flag."""
    ledger_db.parent.mkdir(parents=True, exist_ok=True)
    legacy = sqlite3.connect(ledger_db)
    legacy.execute(
        "CREATE TABLE executions ("
        " id TEXT PRIMARY KEY, job_id TEXT NOT NULL, source TEXT NOT NULL,"
        " process_id TEXT NOT NULL, pid INTEGER NOT NULL, process_started_at INTEGER,"
        " status TEXT NOT NULL, handoff_pending INTEGER NOT NULL DEFAULT 0,"
        " claimed_at TEXT NOT NULL, started_at TEXT, finished_at TEXT, error TEXT)"
    )
    legacy.execute("INSERT INTO executions (id, job_id, source, process_id, pid, status, claimed_at)"
                   " VALUES ('old', 'j', 'cron', 'p', 1, 'completed', '2026-01-01T00:00:00Z')")
    legacy.commit()
    assert su.get_user_version(legacy) == 0
    legacy.close()

    conn = executions._connect()
    try:
        assert su.get_user_version(conn) == executions.SCHEMA_VERSION
        columns = {r[1] for r in conn.execute("PRAGMA table_info(executions)")}
        assert {"delivery_outcome", "scheduled_instant", "progress_at"} <= columns
        # The widened CHECK accepts the states added after this backup was taken.
        conn.execute("UPDATE executions SET status='missed' WHERE id='old'")
        conn.commit()
        assert conn.execute("SELECT status FROM executions WHERE id='old'").fetchone()[0] == "missed"
    finally:
        conn.close()


def test_incident_connection_uses_the_same_gate(ledger_db):
    """cron.incidents opens the same file; a second per-module stamp would starve one side."""
    from cron import incidents
    executions._connect().close()
    conn = incidents._connect()
    try:
        trace = _SqlTrace(conn)
        conn.execute("SELECT count(*) FROM cron_incidents").fetchone()
        assert trace.ddl() == []
        assert su.get_user_version(conn) == executions.SCHEMA_VERSION
    finally:
        conn.close()
