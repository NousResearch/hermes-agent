"""FTS UPDATE OF narrowing + migration (#73639 retargeted onto split SessionDB)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hermes_state import SessionDB


def _trigger_sql(conn: sqlite3.Connection, name: str) -> str | None:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='trigger' AND name=?",
        (name,),
    ).fetchone()
    return row[0] if row else None


def _assert_nonindexed_updates_bypass_missing_fts_target(
    db: SessionDB, message_id: int
) -> None:
    """Prove the UPDATE OF gate, not the trigger's content-change WHEN."""
    db._conn.execute("DROP TABLE messages_fts")
    assert _trigger_sql(db._conn, "messages_fts_update") is not None

    db._conn.execute(
        "UPDATE messages SET active = 0, compacted = 1, observed = 1 "
        "WHERE id = ?",
        (message_id,),
    )
    with pytest.raises(sqlite3.OperationalError, match=r"no such table.*messages_fts"):
        db._conn.execute(
            "UPDATE messages SET content = 'changed' WHERE id = ?",
            (message_id,),
        )


def _install_legacy_inline_base_fts(db: SessionDB) -> None:
    """Replace v23 FTS with the broad inline shape shipped by v11..v22."""
    db._drop_fts_triggers(db._conn)
    db._conn.executescript(
        """
        DROP TABLE IF EXISTS messages_fts;
        DROP TABLE IF EXISTS messages_fts_trigram;
        DROP VIEW IF EXISTS messages_fts_trigram_src;

        CREATE VIRTUAL TABLE messages_fts USING fts5(content);
        CREATE TRIGGER messages_fts_insert AFTER INSERT ON messages BEGIN
            INSERT INTO messages_fts(rowid, content) VALUES (
                new.id,
                COALESCE(new.content, '') || ' ' ||
                COALESCE(new.tool_name, '') || ' ' ||
                COALESCE(new.tool_calls, '')
            );
        END;
        CREATE TRIGGER messages_fts_delete AFTER DELETE ON messages BEGIN
            DELETE FROM messages_fts WHERE rowid = old.id;
        END;
        CREATE TRIGGER messages_fts_update AFTER UPDATE ON messages BEGIN
            DELETE FROM messages_fts WHERE rowid = old.id;
            INSERT INTO messages_fts(rowid, content) VALUES (
                new.id,
                COALESCE(new.content, '') || ' ' ||
                COALESCE(new.tool_name, '') || ' ' ||
                COALESCE(new.tool_calls, '')
            );
        END;
        """
    )


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




def test_v23_status_only_update_bypasses_fts_trigger_body(tmp_path: Path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        sid = "s1"
        db.create_session(sid, source="test")
        mid = db.append_message(sid, role="user", content="hello searchable")
        _assert_nonindexed_updates_bypass_missing_fts_target(db, mid)
    finally:
        db.close()


def test_legacy_status_only_update_bypasses_migrated_fts_trigger(tmp_path: Path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        sid = "s1"
        db.create_session(sid, source="test")
        mid = db.append_message(sid, role="user", content="legacy searchable")
        _install_legacy_inline_base_fts(db)

        assert db._db_has_legacy_inline_fts(db._conn)
        assert db._migrate_broad_fts_update_triggers(db._conn) >= 1
        assert "AFTER UPDATE OF" in " ".join(
            _trigger_sql(db._conn, "messages_fts_update").split()
        ).upper()

        _assert_nonindexed_updates_bypass_missing_fts_target(db, mid)
    finally:
        db.close()
