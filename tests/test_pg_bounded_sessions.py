"""PostgreSQL bounded browsing contracts, plus opt-in live engine checks.

The server-free boundary executes the actual browse query on a temporary
SQLite store and models only PostgreSQL's transaction/timeout commands. It
has no SQLite progress-handler API. The live cases use connection-local
TEMP tables only and never initialize or mutate a persistent PostgreSQL schema.
"""

from __future__ import annotations

import os
import sqlite3

import pytest

import hermes_state_postgres
from hermes_state import SessionDB


class _QueryCanceled(Exception):
    sqlstate = "57014"


class _ReadSqlBoundary:
    def __init__(self, connection):
        self.connection = connection
        self._dsn = "postgresql://test.invalid/bounded-browse"
        self.timeout_ms = None
        self.timeouts = []
        self.read_only = False
        self.failure = None
        self.fail_setup = False
        self.rollbacks = 0

    def execute(self, sql, params=()):
        normalized = " ".join(sql.split())
        if normalized == "BEGIN READ ONLY":
            self.read_only = True
            return self.connection.execute("BEGIN")
        if "set_config('statement_timeout'" in normalized:
            assert self.read_only and self.connection.in_transaction
            if self.fail_setup:
                raise self.failure
            self.timeout_ms = int(str(params[0]).removesuffix("ms"))
            self.timeouts.append(self.timeout_ms)
            return self.connection.execute("SELECT ?", params)
        if normalized.startswith("WITH RECURSIVE"):
            assert self.read_only and self.connection.in_transaction
            assert self.timeout_ms is not None and self.timeout_ms > 0
            if self.failure is not None:
                raise self.failure
        return self.connection.execute(sql, params)

    def rollback(self):
        self.connection.rollback()
        self.timeout_ms = None
        self.read_only = False
        self.rollbacks += 1

    def close(self):
        self.connection.close()


def _open_pg_store(home, monkeypatch, connection):
    monkeypatch.setattr(
        hermes_state_postgres, "maybe_open_postgres",
        lambda read_only, schema_version, dsn_override=None: connection,
    )
    return SessionDB(
        db_path=home / "state.db", postgres_dsn="postgresql://test.invalid/bounded-browse",
    )


@pytest.fixture
def stores(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    source = SessionDB(db_path=tmp_path / "source.db")
    raw = sqlite3.connect(source.db_path, isolation_level=None)
    raw.row_factory = sqlite3.Row
    raw.execute("PRAGMA query_only = ON")
    boundary = _ReadSqlBoundary(raw)
    postgres = _open_pg_store(home, monkeypatch, boundary)
    try:
        yield source, postgres, boundary
    finally:
        postgres.close()
        source.close()


def _session(db, session_id, activity, *, source="cli", parent=None, config=None, title=None):
    db.create_session(session_id, source=source, parent_session_id=parent, model_config=config)
    db.append_message(session_id, "user", f"preview of {session_id}", timestamp=activity)
    db._write_sql(
        "UPDATE sessions SET started_at = ?, last_activity_at = ?, title = ? WHERE id = ?",
        (activity - 1, activity, title, session_id),
    )


def _seed_visibility(db):
    _session(db, "root", 100, title="Logical conversation")
    db.end_session("root", "compression")
    _session(db, "tip", 300, parent="root")
    _session(db, "branch", 200, parent="root", config={"_branched_from": "root"})
    _session(db, "reset-parent", 90)
    db.end_session("reset-parent", "session_reset")
    _session(db, "reset-child", 250, parent="reset-parent", config={"_reset_from": "reset-parent"})
    _session(db, "plain", 150)
    _session(db, "delegated", 400, config={"_delegate_from": "root"})
    _session(db, "cron", 500, source="cron")
    _session(db, "hidden", 600)
    db.set_session_hidden("hidden", True)
    _session(db, "archived", 700)
    db.set_session_archived("archived", True)


def test_browse_preserves_visibility_compression_and_projection(stores):
    source, postgres, boundary = stores
    _seed_visibility(source)
    options = {"limit": 4, "exclude_sources": ["cron"]}
    expected = source.list_recent_sessions_bounded(**options)
    actual = postgres.list_recent_sessions_bounded(**options)
    assert actual == expected
    assert [row["id"] for row in actual] == ["tip", "reset-child", "branch", "plain"]
    assert actual[0]["_lineage_root_id"] == "root"
    assert actual[0]["title"] == "Logical conversation"
    assert actual[0]["preview"] == "preview of tip"
    assert actual[2]["_lineage_root_id"] is None
    assert not boundary.connection.in_transaction
    assert boundary.timeout_ms is None


@pytest.mark.parametrize("shape", ["deep", "fanout", "cycle"])
def test_browse_omits_unresolved_or_cyclic_lineages_at_total_row_cap(stores, shape):
    source, postgres, _ = stores
    if shape == "cycle":
        _session(source, "cycle-a", 100)
        _session(source, "cycle-b", 101, parent="cycle-a")
        source.end_session("cycle-a", "compression")
        source.end_session("cycle-b", "compression")
        source._write_sql("UPDATE sessions SET parent_session_id = ? WHERE id = ?", ("cycle-b", "cycle-a"))
    else:
        parent = None
        for index in range(40):
            session_id = f"part-{index}"
            _session(source, session_id, 100 + index, parent=parent)
            if parent is not None:
                source.end_session(parent, "compression")
            if shape == "deep" or parent is None:
                parent = session_id
    _session(source, "visible-peer", 300)
    options = {"limit": 5, "candidate_limit": 8, "lineage_limit": 8}
    actual = postgres.list_recent_sessions_bounded(**options)
    assert actual == source.list_recent_sessions_bounded(**options)
    assert [row["id"] for row in actual] == ["visible-peer"]


def test_browse_caps_forward_walk_before_an_unreachable_tip(stores):
    source, postgres, _ = stores
    _session(source, "root", 300)
    source.end_session("root", "compression")
    _session(source, "tip", 100, parent="root")
    options = {"limit": 1, "candidate_limit": 1, "lineage_limit": 1}
    assert source.list_recent_sessions_bounded(**options) == []
    assert postgres.list_recent_sessions_bounded(**options) == []


def test_browse_timeout_rolls_back_and_leaves_later_reads_usable(stores):
    source, postgres, boundary = stores
    _session(source, "visible", 100)
    boundary.failure = _QueryCanceled("canceling statement due to statement timeout")
    with pytest.raises(TimeoutError, match="recent-session browse exceeded") as caught:
        postgres.list_recent_sessions_bounded(timeout_seconds=0.01)
    assert caught.value.__cause__ is boundary.failure
    assert not boundary.connection.in_transaction
    assert boundary.timeout_ms is None and not boundary.read_only
    boundary.failure = None
    assert postgres.get_session("visible")["id"] == "visible"
    assert postgres.list_recent_sessions_bounded()[0]["id"] == "visible"


@pytest.mark.parametrize("fail_setup", [False, True])
def test_browse_other_failures_propagate_and_reset_transaction(stores, fail_setup):
    _, postgres, boundary = stores
    boundary.failure = RuntimeError("the query could not run")
    boundary.fail_setup = fail_setup
    with pytest.raises(RuntimeError) as caught:
        postgres.list_recent_sessions_bounded()
    assert caught.value is boundary.failure
    assert not boundary.connection.in_transaction
    assert boundary.timeout_ms is None and not boundary.read_only


def test_zero_deadline_never_disables_the_server_timeout(stores):
    _, postgres, boundary = stores
    assert postgres.list_recent_sessions_bounded(timeout_seconds=0) == []
    assert boundary.timeouts and all(value > 0 for value in boundary.timeouts)
    assert boundary.timeout_ms is None


@pytest.fixture
def live_connection():
    psycopg = pytest.importorskip("psycopg")
    dsn = os.environ.get("HERMES_PG_TEST_DSN")
    if not dsn:
        pytest.skip("set HERMES_PG_TEST_DSN to an isolated PostgreSQL database for live browse checks")
    raw = psycopg.connect(dsn, autocommit=True)
    try:
        yield raw
    finally:
        raw.close()


def test_live_browse_matches_sqlite_using_only_temporary_tables(stores, live_connection, tmp_path, monkeypatch):
    from hermes_state_pg_schema import _pg_column_type
    from hermes_state_postgres import _PostgresConnection

    source, _, _ = stores
    _seed_visibility(source)
    adapter = _PostgresConnection(live_connection)
    for table in ("sessions", "messages"):
        columns = source._read_all(f"PRAGMA table_info({table})")
        declarations = ", ".join(f'"{row[1]}" {_pg_column_type(row[2])}' for row in columns)
        live_connection.execute(f'CREATE TEMP TABLE "{table}" ({declarations})')
        names = ", ".join(f'"{row[1]}"' for row in columns)
        placeholders = ", ".join("%s" for _ in columns)
        with live_connection.cursor() as cursor:
            cursor.executemany(
                f'INSERT INTO "{table}" ({names}) VALUES ({placeholders})',
                [tuple(row) for row in source._read_all(f'SELECT * FROM "{table}"')],
            )
    postgres = _open_pg_store(tmp_path / "live-home", monkeypatch, adapter)
    try:
        options = {"limit": 4, "exclude_sources": ["cron"]}
        assert postgres.list_recent_sessions_bounded(**options) == source.list_recent_sessions_bounded(**options)
    finally:
        postgres.close()


def test_live_deadline_cancels_and_restores_connection_settings(live_connection):
    from hermes_state_pg_sessions import read_bounded_sessions_postgres
    from hermes_state_postgres import _PostgresConnection

    adapter = _PostgresConnection(live_connection)
    previous = live_connection.execute("SHOW statement_timeout").fetchone()[0]
    with pytest.raises(TimeoutError, match="recent-session browse exceeded"):
        read_bounded_sessions_postgres(adapter, "SELECT pg_sleep(?)", (1,), timeout_seconds=0.01)
    assert live_connection.execute("SHOW statement_timeout").fetchone()[0] == previous
    assert live_connection.execute("SELECT 1").fetchone()[0] == 1
