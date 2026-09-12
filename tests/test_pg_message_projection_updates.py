"""Canonical mutations and PostgreSQL message projections stay in agreement.

Adapter cases execute the actual trigger SQL bodies on transactional SQLite.
Native cases use a temporary schema in an explicit HERMES_PG_TEST_DSN database.
"""

import os
import uuid

import pytest

import hermes_state_postgres as pg
from hermes_state import SessionDB
from hermes_state_pg_schema import SCHEMA_SQL_POSTGRES, _schema_statements
from hermes_state_pg_triggers import POSTGRES_MESSAGE_TRIGGER_SQL, install_postgres_message_triggers
from tests.test_pg_clone_rows import message_store  # noqa: F401


@pytest.fixture(params=["adapter", "postgres"])
def projection_store(request, tmp_path, monkeypatch):
    if request.param == "adapter":
        yield request.getfixturevalue("message_store")[0]
        return
    psycopg = pytest.importorskip("psycopg")
    dsn = os.environ.get("HERMES_PG_TEST_DSN")
    if not dsn:
        pytest.skip("set HERMES_PG_TEST_DSN to an isolated PostgreSQL 16+ database")
    from psycopg import sql

    raw = psycopg.connect(dsn, autocommit=True)
    namespace = "hermes_projection_" + uuid.uuid4().hex
    created = False
    db = None
    try:
        raw.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(namespace)))
        created = True
        raw.execute(sql.SQL("SET search_path TO {}, pg_catalog").format(sql.Identifier(namespace)))
        # Use the real PostgreSQL tables/indexes without provisioning optional
        # database-wide extensions for a connection-local behavior test.
        for statement in _schema_statements(SCHEMA_SQL_POSTGRES):
            raw.execute(statement)
        conn = pg._PostgresConnection(raw)
        install_postgres_message_triggers(conn)
        conn.execute("INSERT INTO sessions (id, source, started_at) VALUES (?, 'test', 1)", ("parent",))
        conn.execute("INSERT INTO sessions (id, source, started_at) VALUES (?, 'test', 1)", ("child",))
        monkeypatch.setattr(pg, "maybe_open_postgres", lambda *args, **kwargs: conn)
        db = SessionDB(tmp_path / "nominal.db", postgres_dsn=dsn)
        yield db
    finally:
        try:
            raw.rollback()
            if created:
                raw.execute("SET search_path TO pg_catalog")
                raw.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(namespace)))
        finally:
            if db is not None:
                db.close()
            else:
                raw.close()


def _rows(db, session_id="parent"):
    return [dict(row) for row in db._read_all(
        "SELECT * FROM messages WHERE session_id = ? ORDER BY id", (session_id,),
    )]


def _assert_display_matches_canonical(db, session_id="parent"):
    visible = [row for row in _rows(db, session_id) if row["active"] or row["compacted"]]
    expected = [(row["id"], row["content"]) for row in db._dedupe_display_generations(visible)]
    displayed = db.get_messages(session_id, include_compacted=True, limit=50)
    assert [(row["id"], row["content"]) for row in displayed] == expected
    return displayed


def _duplicate_group(db):
    first = db.append_message("parent", role="assistant", content="same message", timestamp=10)
    between = db.append_message("parent", role="user", content="between copies", timestamp=20)
    last = db.append_message("parent", role="assistant", content="same message", timestamp=10)
    return first, between, last


def test_assistant_repair_invalidates_or_refreshes_derived_search_text(projection_store):
    db = projection_store
    row_id = db.append_message("parent", role="assistant", content="", timestamp=10)
    db.get_messages("parent", include_compacted=True, limit=50)
    assert db.append_messages_batch(
        "parent", [{"_row_id": row_id, "role": "assistant", "content": "recovered final response"}],
    ) == 0
    stored = _rows(db)[0]
    assert stored["content"] == "recovered final response"
    assert stored["fts_content"] is None
    assert not pg._fts_backfill_complete(db._conn)
    assert [row["id"] for row in db.search_messages(
        "recovered final response", fields=("id",),
    )] == [row_id]


def test_repaired_compaction_copy_keeps_indexed_display_in_sync(projection_store):
    db = projection_store
    db.append_message("parent", role="assistant", content="", timestamp=10)
    # Legacy compactions retain a display-visible original and active copy.
    db.archive_and_compact("parent", [
        {"role": "user", "content": "summary", "timestamp": 20},
        {"role": "assistant", "content": "", "timestamp": 10},
    ])
    db.get_messages("parent", include_compacted=True, limit=50)
    copied_id = db.get_messages("parent")[-1]["id"]
    db.append_messages_batch(
        "parent", [{"_row_id": copied_id, "role": "assistant", "content": "recovered response"}],
    )
    _assert_display_matches_canonical(db)


def test_indexed_append_needs_no_display_backfill_and_keeps_null_import_identity(projection_store):
    db = projection_store
    first, between, last = _duplicate_group(db)
    writes_before_read = db._write_count
    displayed = _assert_display_matches_canonical(db)
    assert [row["id"] for row in displayed] == [last, between]
    assert db._write_count == writes_before_read
    assert [row["display_order"] for row in _rows(db)] == [first, between, first]

    # Raw imports cannot reuse an identity derived from differently encoded
    # source content. The INSERT trigger must leave NULL for canonical backfill.
    imported = db._execute_write(lambda conn: conn.execute(
        "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        ("parent", "user", "imported content", 30),
    ).lastrowid)
    before_read = _rows(db)[-1]
    assert before_read["display_identity"] is None
    assert before_read["display_order"] == imported
    _assert_display_matches_canonical(db)
    assert _rows(db)[-1]["display_identity"] is not None


@pytest.mark.parametrize("mutation", ["delete", "hide"])
def test_removing_earliest_copy_reorders_surviving_peers_only(projection_store, mutation):
    db = projection_store
    first, between, last = _duplicate_group(db)
    db.append_message("child", role="assistant", content="same message", timestamp=10)
    child_before = _rows(db, "child")
    _assert_display_matches_canonical(db)
    statement = "DELETE FROM messages WHERE id = ?" if mutation == "delete" else (
        "UPDATE messages SET active = 0, compacted = 0 WHERE id = ?"
    )
    db._execute_write(lambda conn: conn.execute(statement, (first,)))
    assert _rows(db, "child") == child_before
    assert [row["id"] for row in _assert_display_matches_canonical(db)] == [between, last]
    if mutation == "hide":
        db._execute_write(lambda conn: conn.execute("UPDATE messages SET active = 1 WHERE id = ?", (first,)))
        assert [row["id"] for row in _assert_display_matches_canonical(db)] == [last, between]


@pytest.mark.parametrize("column, replacement", [
    ("tool_name", "replacement_tool"),
    ("tool_calls", '[{"id": "new-call"}]'),
    ("display_metadata", '{"source": "updated"}'),
])
def test_raw_update_invalidates_only_affected_search_text(projection_store, column, replacement):
    db = projection_store
    first, _between, last = _duplicate_group(db)
    before = {row["id"]: row for row in _rows(db)}
    db._execute_write(lambda conn: conn.execute(
        f"UPDATE messages SET {column} = ? WHERE id = ?", (replacement, last),
    ))
    after = {row["id"]: row for row in _rows(db)}
    assert after[first]["fts_content"] == before[first]["fts_content"]
    expected_fts = before[last]["fts_content"] if column == "display_metadata" else None
    assert after[last]["fts_content"] == expected_fts
    _assert_display_matches_canonical(db)
    # A derived-only write must keep its values; it cannot invalidate itself.
    stable = _rows(db)
    db._execute_write(lambda conn: conn.execute(
        "UPDATE messages SET display_order = display_order, display_identity = display_identity, "
        "fts_content = fts_content WHERE session_id = ?", ("parent",),
    ))
    assert _rows(db) == stable


def test_postgres_parses_trigger_ddl_and_procedural_bodies():
    pglast = pytest.importorskip("pglast")
    for statement in POSTGRES_MESSAGE_TRIGGER_SQL:
        pglast.parse_sql(statement)
        if "CREATE OR REPLACE FUNCTION" in statement:
            pglast.parse_plpgsql(statement)
