"""Session DB helpers for dashboard memory optimization."""

import time

import pytest

from hermes_state import SessionDB, SCHEMA_VERSION


@pytest.fixture
def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        yield database
    finally:
        database.close()


def test_schema_includes_last_active_column(db):
    row = db._conn.execute("PRAGMA table_info(sessions)").fetchall()
    columns = {r[1] for r in row}
    assert "last_active" in columns
    assert SCHEMA_VERSION >= 17


def test_append_message_bumps_last_active(db):
    db.create_session("s1", source="cli")
    before = time.time()
    db.append_message("s1", "user", "hello")
    session = db.get_session("s1")
    assert session["last_active"] is not None
    assert float(session["last_active"]) >= before - 1


def test_session_count_cache_avoids_repeated_queries(db):
    db.create_session("a", source="cli")
    db.create_session("b", source="desktop")

    first = db.session_count(use_cache=True, cache_ttl=60.0)
    second = db.session_count(use_cache=True, cache_ttl=60.0)
    assert first == second == 2

    db.create_session("c", source="cli")
    third = db.session_count(use_cache=True, cache_ttl=60.0)
    assert third == 3


def test_session_counts_by_source_aggregate(db):
    db.create_session("a", source="cli")
    db.create_session("b", source="desktop")
    db.create_session("c", source="cli")

    counts = db.session_counts_by_source(include_archived=True)
    assert counts["cli"] == 2
    assert counts["desktop"] == 1


def test_list_sessions_rich_recent_uses_last_active(db):
    base = time.time() - 1000
    db.create_session("old", source="cli")
    db.create_session("new", source="cli")
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, last_active = ? WHERE id = 'old'",
        (base, base),
    )
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, last_active = ? WHERE id = 'new'",
        (base + 10, base + 500),
    )
    db._conn.commit()

    rows = db.list_sessions_rich(limit=10, order_by_last_active=True)
    assert [r["id"] for r in rows][:2] == ["new", "old"]
