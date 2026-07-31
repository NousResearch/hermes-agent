"""FTS UPDATE OF narrowing + migration (#73639 retargeted onto split SessionDB)."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from hermes_state import SessionDB
from hermes_state_schema import SessionSchemaMixin


def _trigger_sql(conn: sqlite3.Connection, name: str) -> str | None:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='trigger' AND name=?",
        (name,),
    ).fetchone()
    return row[0] if row else None


def test_fresh_db_installs_update_of_triggers(tmp_path: Path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        sql = _trigger_sql(db._conn, "messages_fts_update")
        assert sql is not None
        compact = " ".join(sql.split()).upper()
        assert "AFTER UPDATE OF " in compact
        assert "CONTENT" in compact
        assert "TOOL_NAME" in compact
        assert "TOOL_CALLS" in compact

        tri = _trigger_sql(db._conn, "messages_fts_trigram_update")
        if tri:  # trigram may be unavailable on some builds
            tcompact = " ".join(tri.split()).upper()
            assert "AFTER UPDATE OF " in tcompact
    finally:
        db.close()


def test_migrate_replaces_broad_update_trigger(tmp_path: Path):
    path = tmp_path / "state.db"
    db = SessionDB(db_path=path)
    try:
        # Force a broad trigger the way older installs had it.
        db._conn.execute("DROP TRIGGER IF EXISTS messages_fts_update")
        db._conn.execute(
            """
            CREATE TRIGGER messages_fts_update AFTER UPDATE ON messages
            BEGIN
                SELECT 1;
            END
            """
        )
        db._conn.commit()
        before = _trigger_sql(db._conn, "messages_fts_update")
        assert "AFTER UPDATE OF" not in " ".join(before.split()).upper()

        dropped = db._migrate_broad_fts_update_triggers(db._conn)
        db._conn.commit()
        assert dropped >= 1

        after = _trigger_sql(db._conn, "messages_fts_update")
        assert after is not None
        compact = " ".join(after.split()).upper()
        assert "AFTER UPDATE OF " in compact
    finally:
        db.close()


def test_needs_narrowing_helper():
    assert SessionSchemaMixin._fts_update_trigger_needs_narrowing(
        "CREATE TRIGGER t AFTER UPDATE ON messages BEGIN SELECT 1; END"
    )
    assert not SessionSchemaMixin._fts_update_trigger_needs_narrowing(
        "CREATE TRIGGER t AFTER UPDATE OF content ON messages BEGIN SELECT 1; END"
    )
    assert not SessionSchemaMixin._fts_update_trigger_needs_narrowing(None)


def test_status_only_update_does_not_require_content_change(tmp_path: Path):
    """Smoke: DB opens and accepts message updates under narrowed triggers."""
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        sid = "s1"
        db.create_session(sid, source="test")
        mid = db.append_message(sid, role="user", content="hello searchable")
        db._conn.execute(
            "UPDATE messages SET content = content WHERE id = ?",
            (mid,),
        )
        db._conn.commit()
        assert _trigger_sql(db._conn, "messages_fts_update")
    finally:
        db.close()
