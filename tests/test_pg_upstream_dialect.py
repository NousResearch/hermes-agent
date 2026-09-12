"""Behavioral coverage for upstream state SQL on the PostgreSQL adapter.

Binding tests need psycopg but no server. Live parity cases accept an isolated
HERMES_PG_TEST_DSN. The lease race uses a unique table removed in finally;
other parity cases use connection-local temporary tables.
"""

import os
import sqlite3

import pytest

from hermes_state_pg_sql import _translate_sql


@pytest.fixture(autouse=True)
def strict_adapter(monkeypatch):
    monkeypatch.setenv("HERMES_PG_ADAPTER_STRICT", "1")


@pytest.mark.parametrize("sql, params", [
    ("SELECT 'why? 100% correct CHAR(9)', ?", ("literal? %",)),
    ("SELECT 'don''t change json_object() or INSTR(x,y)', ?, 11 % 3", (7,)),
    ("SELECT ? -- ignored ? and 100%\n, '50%%'", (None,)),
])
def test_parameter_binding_preserves_literal_values(sql, params):
    pytest.importorskip("psycopg")
    from psycopg._queries import PostgresQuery
    from psycopg.adapt import Transformer

    # Exercise psycopg's real formatting step. Both SQL expressions here are
    # portable; SQLite accepts the resulting $1 binds and proves literals have
    # the same values before and after that step.
    query = PostgresQuery(Transformer())
    query.convert(_translate_sql(sql), params)
    with sqlite3.connect(":memory:") as db:
        assert db.execute(query.query.decode(), params).fetchall() == db.execute(sql, params).fetchall()


def test_native_postgres_and_sqlite_binds_can_share_a_statement():
    pytest.importorskip("psycopg")
    from psycopg._queries import PostgresQuery
    from psycopg.adapt import Transformer

    query = PostgresQuery(Transformer())
    query.convert(_translate_sql("SELECT %s, ?, 'literal %s?'"), ("fts-query", 20))
    with sqlite3.connect(":memory:") as db:
        assert db.execute(query.query.decode(), ("fts-query", 20)).fetchone() == (
            "fts-query", 20, "literal %s?",
        )


def test_named_session_import_binds_preserve_stored_values():
    pytest.importorskip("psycopg")
    from psycopg._queries import PostgresQuery
    from psycopg.adapt import Transformer

    from hermes_state_common import SCHEMA_SQL
    from hermes_state_portability import (
        _IMPORT_FLOAT_COLS, _IMPORT_INT_COLS, _IMPORT_PASSTHROUGH_COLS, _IMPORT_SESSION_INSERT_SQL,
    )

    params = {
        **{name: None for name in (*_IMPORT_PASSTHROUGH_COLS, *_IMPORT_FLOAT_COLS)},
        **{name: 0 for name in _IMPORT_INT_COLS},
        "id": "named-import", "source": "import", "system_prompt_hash": None,
        "started_at": 123.25, "archived": 0,
        "title": "Literal ':value' and 100% %(native)s?",
        "model_config": '{"route": ":value::jsonb"}',
    }
    query = PostgresQuery(Transformer())
    query.convert(_translate_sql(_IMPORT_SESSION_INSERT_SQL), params)
    with sqlite3.connect(":memory:") as source, sqlite3.connect(":memory:") as target:
        source.executescript(SCHEMA_SQL)
        target.executescript(SCHEMA_SQL)
        source.execute(_IMPORT_SESSION_INSERT_SQL, params)
        target.execute(query.query.decode(), tuple(params[name] for name in query._order or ()))
        assert target.execute("SELECT * FROM sessions").fetchone() == source.execute("SELECT * FROM sessions").fetchone()


def test_named_binds_preserve_native_placeholders_casts_and_quoted_sql():
    pytest.importorskip("psycopg")
    from psycopg._queries import PostgresQuery
    from psycopg.adapt import Transformer

    params = {"value": "first :name?", "native": "second 100%"}
    query = PostgresQuery(Transformer())
    query.convert(_translate_sql(
        'SELECT :value AS "quoted:name", %(native)s, :value, '
        "'literal :ignored %(missing)s ? 100%', 11 % 3 "
        "/* :ignored %(missing)s */ -- :ignored %(missing)s ?\n"
    ), params)
    with sqlite3.connect(":memory:") as db:
        assert db.execute(query.query.decode(), tuple(params[name] for name in query._order or ())).fetchone() == (
            params["value"], params["native"], params["value"], "literal :ignored %(missing)s ? 100%", 2,
        )
    query.convert(_translate_sql("SELECT :value::text, %(native)s::text"), params)
    assert b"$1::text" in query.query and b"$2::text" in query.query


@pytest.fixture
def dialect_pair():
    psycopg = pytest.importorskip("psycopg")
    dsn = os.environ.get("HERMES_PG_TEST_DSN")
    if not dsn:
        pytest.skip("set HERMES_PG_TEST_DSN to an isolated PostgreSQL database for live dialect parity")
    source = sqlite3.connect(":memory:")
    target = psycopg.connect(dsn, autocommit=True)
    try:
        for ddl in (
            "CREATE TEMP TABLE sessions (id TEXT PRIMARY KEY, model_config TEXT, "
            "session_key TEXT, parent_session_id TEXT, cwd TEXT, git_repo_root TEXT, "
            "git_branch TEXT, profile_name TEXT, started_at DOUBLE PRECISION, "
            "ended_at DOUBLE PRECISION, end_reason TEXT)",
            "CREATE TEMP TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, "
            "role TEXT, content TEXT, display_kind TEXT, timestamp DOUBLE PRECISION)",
        ):
            source.execute(ddl)
            target.execute(ddl)
        yield source, target
    finally:
        source.close()
        target.close()


def _both(pair, sql, params=()):
    source, target = pair
    left = source.execute(sql, params)
    right = target.execute(_translate_sql(sql), params)
    return left, right


@pytest.mark.parametrize("length, offset", [(-1, 0), (-1, 2), (2, 1), (0, 0)])
def test_unlimited_message_pages_match_sqlite(dialect_pair, length, offset):
    for i in range(5):
        _both(dialect_pair, "INSERT INTO messages(id, content) VALUES (?, ?)", (i, str(i)))
    left, right = _both(
        dialect_pair, "SELECT id FROM messages ORDER BY id LIMIT ? OFFSET ?", (length, offset),
    )
    assert right.fetchall() == left.fetchall()
    left, right = _both(dialect_pair, "SELECT id FROM messages ORDER BY id LIMIT -1 OFFSET 2")
    assert right.fetchall() == left.fetchall()


@pytest.mark.parametrize("content", [None, "original message", "absent"])
def test_null_safe_content_guards_match_sqlite(dialect_pair, content):
    for i, text in enumerate((None, "original message")):
        _both(dialect_pair, "INSERT INTO messages(id, content) VALUES (?, ?)", (i, text))
    for operator in ("IS", "IS NOT"):
        left, right = _both(
            dialect_pair, f"SELECT id FROM messages WHERE content {operator} ? ORDER BY id", (content,),
        )
        assert right.fetchall() == left.fetchall()


@pytest.mark.parametrize("model_config", [None, "invalid-json", "{}", '{"_delegate_from":"parent"}'])
def test_all_nested_json_marker_reads_match_sqlite(dialect_pair, model_config):
    from hermes_state_common import _sql_json_extract

    _both(dialect_pair, "INSERT INTO sessions(id, model_config) VALUES (?, ?)", ("session", model_config))
    # Rich session/lineage queries repeat marker predicates. Translation must
    # visit every call, even once the query grows beyond ten JSON operations.
    markers = ", ".join(_sql_json_extract("model_config", "$._delegate_from") for _ in range(20))
    left, right = _both(dialect_pair, f"SELECT {markers} FROM sessions WHERE id = ?", ("session",))
    assert right.fetchone() == left.fetchone()


@pytest.mark.parametrize("case", ["ordinary", "skill", "merged", "hidden", "summary"])
def test_upstream_session_preview_matches_sqlite(dialect_pair, case):
    from agent.context_compressor import SUMMARY_PREFIX, _MERGED_SUMMARY_DELIMITER
    from agent.skill_commands import SKILL_SCAFFOLD_SQL_LIKE
    from hermes_state_sessions import _PREVIEW_COL_SQL

    content = {
        "ordinary": "A normal\nmessage without any compaction markers.",
        "skill": SKILL_SCAFFOLD_SQL_LIKE.removesuffix("%") + "skill body " * 100 + "the actual task at the tail",
        "merged": "Preserved user instruction\n" + _MERGED_SUMMARY_DELIMITER + "\n" + SUMMARY_PREFIX + "summary",
        "hidden": "Model-facing scaffolding",
        "summary": SUMMARY_PREFIX + "only a summary",
    }[case]
    _both(dialect_pair, "INSERT INTO sessions(id) VALUES (?)", ("session",))
    _both(dialect_pair,
          "INSERT INTO messages(id, session_id, role, content, display_kind, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
          (1, "session", "user", content, "hidden" if case == "hidden" else None, 1.0))
    left, right = _both(dialect_pair, f"SELECT {_PREVIEW_COL_SQL} FROM sessions s WHERE s.id = ?", ("session",))
    assert right.fetchone() == left.fetchone()


@pytest.mark.parametrize("child_key", [None, "agent:alpha:chat:child", "agent:beta:chat:child"])
def test_upstream_parent_profile_inheritance_matches_sqlite(dialect_pair, child_key):
    from hermes_state_sessions import _INHERIT_PARENT_META_SQL

    _both(dialect_pair,
          "INSERT INTO sessions(id, session_key, profile_name, cwd) VALUES (?, ?, ?, ?)",
          ("parent", "agent:alpha:chat:parent", "alpha", "/work"))
    _both(dialect_pair,
          "INSERT INTO sessions(id, parent_session_id, session_key) VALUES (?, ?, ?)",
          ("child", "parent", child_key))
    _both(dialect_pair, _INHERIT_PARENT_META_SQL, ("child",))
    left, right = _both(dialect_pair, "SELECT profile_name, cwd FROM sessions WHERE id = ?", ("child",))
    assert right.fetchone() == left.fetchone()


def test_serializable_claim_preserves_a_concurrently_refreshed_lease():
    psycopg = pytest.importorskip("psycopg")
    from uuid import uuid4

    from psycopg import sql
    from psycopg.errors import SerializationFailure

    from hermes_state_compression import _claim_lease_row
    from hermes_state_postgres import _PostgresConnection

    dsn = os.environ.get("HERMES_PG_TEST_DSN")
    if not dsn:
        pytest.skip("set HERMES_PG_TEST_DSN to an isolated PostgreSQL database for live lease parity")
    table = "hermes_lease_parity_" + uuid4().hex
    table_id = sql.Identifier(table)
    # A TEMP table cannot be accessed by a second connection. Create only
    # this uniquely named table and remove it even when the assertion fails.
    with psycopg.connect(dsn, autocommit=True) as refresh, psycopg.connect(dsn, autocommit=True) as raw_claim:
        claimant = _PostgresConnection(raw_claim)
        created = False
        try:
            refresh.execute(sql.SQL(
                "CREATE TABLE {} (conversation_id TEXT PRIMARY KEY, holder TEXT NOT NULL, "
                "acquired_at DOUBLE PRECISION NOT NULL, expires_at DOUBLE PRECISION NOT NULL)"
            ).format(table_id))
            created = True
            refresh.execute(sql.SQL("INSERT INTO {} VALUES (%s, %s, %s, %s)").format(table_id),
                            ("thread", "owner", 0.0, 1.0))

            def refresh_after_stale_read(holder, expires_at):
                assert holder == "owner" and expires_at <= 10.0
                refresh.execute(sql.SQL(
                    "UPDATE {} SET expires_at = %s WHERE conversation_id = %s AND holder = %s"
                ).format(table_id), (100.0, "thread", "owner"))
                return True

            claimant.execute("BEGIN IMMEDIATE")
            with pytest.raises(SerializationFailure):
                _claim_lease_row(claimant, table, "conversation_id", "thread", "replacement", 10.0, 20.0,
                                 refresh_after_stale_read)
            claimant.rollback()
            assert refresh.execute(sql.SQL("SELECT holder, expires_at FROM {}").format(table_id)).fetchone() == (
                "owner", 100.0,
            )
            # Retrying the whole callback sees the renewed expiry and refuses
            # the claimant, rather than deleting the current owner's lease.
            claimant.execute("BEGIN IMMEDIATE")
            assert _claim_lease_row(
                claimant, table, "conversation_id", "thread", "replacement", 10.0, 20.0,
                lambda holder, expires_at: expires_at <= 10.0,
            ) == (False, None)
            claimant.commit()
        finally:
            claimant.rollback()
            if created:
                refresh.execute(sql.SQL("DROP TABLE {}").format(table_id))
