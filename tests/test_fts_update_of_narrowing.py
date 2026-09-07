"""FTS UPDATE OF narrowing + migration (#73639 retargeted onto split SessionDB)."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from hermes_state import SessionDB
from hermes_state_common import FTS_CJK_STALE_KEY
from hermes_state_schema import SessionSchemaMixin


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


def test_needs_narrowing_helper():
    assert SessionSchemaMixin._fts_update_trigger_needs_narrowing(
        "CREATE TRIGGER t AFTER UPDATE ON messages BEGIN SELECT 1; END"
    )
    assert not SessionSchemaMixin._fts_update_trigger_needs_narrowing(
        "CREATE TRIGGER t AFTER UPDATE OF content ON messages BEGIN SELECT 1; END"
    )
    assert not SessionSchemaMixin._fts_update_trigger_needs_narrowing(None)


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


def test_cjk_ensure_failure_marks_unavailable_and_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "state.db"
    db = SessionDB(db_path=path)
    try:
        db._conn.execute("DROP TRIGGER IF EXISTS messages_fts_cjk_update")
        db._conn.execute(
            "CREATE TRIGGER messages_fts_cjk_update "
            "AFTER UPDATE ON messages BEGIN SELECT 1; END"
        )
        db._fts_cjk_available = True

        def _fail_cjk_ensure(_cursor):
            raise sqlite3.DatabaseError("injected CJK ensure failure")

        monkeypatch.setattr(db, "_ensure_fts_cjk_schema", _fail_cjk_ensure)

        with pytest.raises(sqlite3.DatabaseError, match="injected CJK ensure failure"):
            db._migrate_broad_fts_update_triggers(db._conn)
        assert db._fts_cjk_available is False
        assert _trigger_sql(db._conn, "messages_fts_cjk_update") is None

        # The DROP is autocommitted.  Fail-closed therefore needs a durable
        # breadcrumb that other processes can observe, not just an instance
        # flag on the SessionDB whose initialization is aborting.
        with sqlite3.connect(path) as observer:
            stale = observer.execute(
                "SELECT value FROM state_meta WHERE key = ?",
                (FTS_CJK_STALE_KEY,),
            ).fetchone()
            assert stale == ("1",)
            assert _trigger_sql(observer, "messages_fts_cjk_update") is None
    finally:
        db.close()


def test_cjk_soft_fail_ensure_without_raise_marks_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Production ensure swallows OperationalError — still must quarantine."""
    path = tmp_path / "state.db"
    db = SessionDB(db_path=path)
    try:
        db._conn.execute("DROP TRIGGER IF EXISTS messages_fts_cjk_update")
        db._conn.execute(
            "CREATE TRIGGER messages_fts_cjk_update "
            "AFTER UPDATE ON messages BEGIN SELECT 1; END"
        )
        db._fts_cjk_available = True

        def _soft_fail_cjk_ensure(_cursor):
            # Mirrors real _ensure_fts_cjk_schema OperationalError path:
            # clear availability, do not raise, do not recreate triggers.
            db._fts_cjk_available = False

        monkeypatch.setattr(db, "_ensure_fts_cjk_schema", _soft_fail_cjk_ensure)

        dropped = db._migrate_broad_fts_update_triggers(db._conn)
        assert dropped >= 1
        assert db._fts_cjk_available is False
        assert _trigger_sql(db._conn, "messages_fts_cjk_update") is None

        with sqlite3.connect(path) as observer:
            stale = observer.execute(
                "SELECT value FROM state_meta WHERE key = ?",
                (FTS_CJK_STALE_KEY,),
            ).fetchone()
            assert stale == ("1",)
            assert _trigger_sql(observer, "messages_fts_cjk_update") is None
    finally:
        db.close()


def test_cjk_broad_trigger_is_restored_as_update_of(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db._conn.execute("DROP TRIGGER IF EXISTS messages_fts_cjk_update")
        db._conn.execute(
            "CREATE TRIGGER messages_fts_cjk_update "
            "AFTER UPDATE ON messages BEGIN SELECT 1; END"
        )

        def _restore_cjk_update_trigger(cursor):
            cursor.execute(
                "CREATE TRIGGER messages_fts_cjk_update "
                "AFTER UPDATE OF content, tool_name, tool_calls ON messages "
                "BEGIN SELECT 1; END"
            )

        monkeypatch.setattr(db, "_ensure_fts_cjk_schema", _restore_cjk_update_trigger)

        assert db._migrate_broad_fts_update_triggers(db._conn) == 1
        cjk_sql = _trigger_sql(db._conn, "messages_fts_cjk_update")
        assert cjk_sql is not None
        assert "AFTER UPDATE OF" in " ".join(cjk_sql.split()).upper()
    finally:
        db.close()


def test_cjk_ensure_survives_presence_check_failure_when_tokenizer_unloaded(
    tmp_path: Path,
):
    """``_ensure_fts_cjk_schema`` is documented never to raise (#98834).

    The tokenizer-not-loaded branch runs its sqlite_master presence check
    and self-heal statements before any try/except guards them; a transient
    sqlite3.OperationalError there (e.g. "database is locked") must degrade
    to "no cjk index", not propagate.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db._fts_cjk_loaded = False
        db._fts_cjk_available = True

        class _LockedOnMasterCursor:
            def execute(self, sql, *args, **kwargs):
                if "sqlite_master" in sql:
                    raise sqlite3.OperationalError("database is locked")
                raise AssertionError(f"unexpected query: {sql}")

        db._ensure_fts_cjk_schema(_LockedOnMasterCursor())
        assert db._fts_cjk_available is False
    finally:
        db.close()


def test_legacy_migration_does_not_drop_cjk_trigger(tmp_path: Path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        _install_legacy_inline_base_fts(db)
        db._conn.execute("DROP TRIGGER IF EXISTS messages_fts_cjk_update")
        db._conn.execute(
            "CREATE TRIGGER messages_fts_cjk_update "
            "AFTER UPDATE ON messages BEGIN SELECT 1; END"
        )

        assert db._migrate_broad_fts_update_triggers(db._conn) >= 1
        cjk_sql = _trigger_sql(db._conn, "messages_fts_cjk_update")
        assert cjk_sql is not None
        assert "AFTER UPDATE OF" not in " ".join(cjk_sql.split()).upper()
    finally:
        db.close()
