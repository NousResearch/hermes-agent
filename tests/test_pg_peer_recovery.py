"""Peer recovery through real PostgreSQL binding and temporary state stores.

The server-free raw-driver boundary executes the actual bound SQL on SQLite
and rejects an unknown-typed parameter used only in a bare IS NULL predicate,
which PostgreSQL cannot type. Optional native cases use TEMP tables only.
"""

from __future__ import annotations

import os
import re
import sqlite3

import pytest

import hermes_state_postgres
from hermes_state import SessionDB
from hermes_state_postgres import _PostgresConnection


class _WireCursor:
    def __init__(self, cursor):
        self.cursor = cursor

    def execute(self, sql, params):
        from psycopg._queries import PostgresQuery
        from psycopg.adapt import Transformer

        query = PostgresQuery(Transformer())
        query.convert(sql, params)
        for match in re.finditer(rb"\$(\d+)\s+IS\s+(?:NOT\s+)?NULL\b", query.query, re.IGNORECASE):
            index = int(match[1])
            uses = re.findall(rb"\$" + match[1] + rb"\b", query.query)
            assert query.types[index - 1] or len(uses) > 1, (
                f"PostgreSQL cannot determine the type of standalone NULL-test parameter ${index}"
            )
        self.cursor.execute(query.query.decode(), params)
        return self

    def fetchone(self):
        return self.cursor.fetchone()

    @property
    def description(self):
        return self.cursor.description


class _WireConnection:
    def __init__(self, connection):
        self.connection = connection

    def cursor(self):
        return _WireCursor(self.connection.cursor())

    def close(self):
        self.connection.close()


def _open_store(path, monkeypatch, adapter):
    monkeypatch.setattr(
        hermes_state_postgres, "maybe_open_postgres",
        lambda read_only, schema_version, dsn_override=None: adapter,
    )
    return SessionDB(db_path=path, postgres_dsn="postgresql://test.invalid/peer-recovery")


def _seed_peers(db):
    for session_id, profile, activity, chat_id in (
        ("own", "alpha", 100, "42"),
        ("legacy", None, 50, "42"),
        ("sibling", "beta", 200, "42"),
        ("foreign-chat", "alpha", 300, "different-chat"),
    ):
        db.create_session(
            session_id, "telegram", user_id="42", chat_id=chat_id, chat_type="dm",
            session_key=f"old-key:{session_id}", profile_name=profile,
        )
        db.append_message(session_id, "user", f"conversation with {session_id}", timestamp=activity)
        db._write_sql("UPDATE sessions SET last_activity_at = ? WHERE id = ?", (activity, session_id))


def _recover(db):
    return db.find_latest_gateway_session_for_peer(
        source="telegram", session_key="missing-current-key", user_id="42", chat_id="42", chat_type="dm",
    )


@pytest.fixture(params=["alpha", None], ids=["profile-owner", "no-profile-owner"])
def stores(request, tmp_path, monkeypatch):
    pytest.importorskip("psycopg")
    root = tmp_path / "hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_PG_ADAPTER_STRICT", "1")
    owner = request.param
    path = root / "profiles" / owner / "state.db" if owner else tmp_path / "detached" / "state.db"
    source = SessionDB(db_path=tmp_path / "source.db")
    _seed_peers(source)
    raw = sqlite3.connect(source.db_path, isolation_level=None)
    raw.execute("PRAGMA query_only = ON")
    postgres = _open_store(path, monkeypatch, _PostgresConnection(_WireConnection(raw)))
    try:
        yield source, postgres, owner, path
    finally:
        postgres.close()
        source.close()


def test_peer_fallback_keeps_the_profile_fence_after_postgres_binding(stores):
    _, postgres, owner, _ = stores
    recovered = _recover(postgres)
    assert recovered["id"] == ("own" if owner else "sibling")
    assert recovered["profile_name"] == (owner or "beta")


def test_peer_fallback_allows_a_legacy_null_profile(stores):
    source, postgres, _, _ = stores
    source._write_sql("UPDATE sessions SET last_activity_at = ? WHERE id = ?", (400, "legacy"))
    recovered = _recover(postgres)
    assert recovered["id"] == "legacy"
    assert recovered["profile_name"] is None


def test_live_peer_fallback_uses_only_temporary_tables(stores, monkeypatch):
    from hermes_state_pg_schema import _pg_column_type

    psycopg = pytest.importorskip("psycopg")
    dsn = os.environ.get("HERMES_PG_TEST_DSN")
    if not dsn:
        pytest.skip("set HERMES_PG_TEST_DSN to an isolated PostgreSQL database for live peer recovery")
    source, _, owner, path = stores
    raw = psycopg.connect(dsn, autocommit=True)
    postgres = None
    try:
        for table in ("sessions", "messages", "system_prompts"):
            columns = source._read_all(f"PRAGMA table_info({table})")
            declarations = ", ".join(f'"{row[1]}" {_pg_column_type(row[2])}' for row in columns)
            raw.execute(f'CREATE TEMP TABLE "{table}" ({declarations})')
            names = ", ".join(f'"{row[1]}"' for row in columns)
            placeholders = ", ".join("%s" for _ in columns)
            with raw.cursor() as cursor:
                cursor.executemany(
                    f'INSERT INTO "{table}" ({names}) VALUES ({placeholders})',
                    [tuple(row) for row in source._read_all(f'SELECT * FROM "{table}"')],
                )
        postgres = _open_store(path, monkeypatch, _PostgresConnection(raw))
        recovered = _recover(postgres)
        assert recovered["id"] == ("own" if owner else "sibling")
    finally:
        if postgres is not None:
            postgres.close()
        raw.close()
