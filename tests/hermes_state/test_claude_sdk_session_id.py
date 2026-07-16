"""The sessions.claude_sdk_session_id column (W3 continuity, #25267).

Declarative migration: the column lives in SCHEMA_SQL and
_reconcile_columns adds it to older DBs on startup — so a fresh DB and an
upgraded DB both expose it, nullable.
"""

from hermes_state import SessionDB


def test_column_exists_null_by_default_and_round_trips(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("sess-cc-1", source="telegram")
        row = db.get_session("sess-cc-1")
        assert "claude_sdk_session_id" in row
        assert row["claude_sdk_session_id"] is None

        db.update_claude_sdk_session_id("sess-cc-1", "sdk-uuid-42")
        assert db.get_session("sess-cc-1")["claude_sdk_session_id"] == "sdk-uuid-42"

        # Clearing (error retire) round-trips to NULL.
        db.update_claude_sdk_session_id("sess-cc-1", None)
        assert db.get_session("sess-cc-1")["claude_sdk_session_id"] is None
    finally:
        db.close()


def test_new_session_row_never_inherits_an_id(tmp_path):
    # /new and expiry rotate to a NEW Hermes session row — fresh-by-keying:
    # the new row must carry no resume id.
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("sess-old", source="telegram")
        db.update_claude_sdk_session_id("sess-old", "sdk-uuid-1")
        db.create_session("sess-new", source="telegram")
        assert db.get_session("sess-new")["claude_sdk_session_id"] is None
    finally:
        db.close()


def test_fts_probe_error_classifier():
    # Validator C2: only a MISSING fts object may disable read-only search;
    # a transient lock must never latch a silent false-empty.
    import sqlite3

    from hermes_state import _fts_object_missing

    assert _fts_object_missing(sqlite3.OperationalError("no such table: messages_fts"))
    assert _fts_object_missing(sqlite3.OperationalError("no such module: fts5"))
    assert not _fts_object_missing(sqlite3.OperationalError("database is locked"))
    assert not _fts_object_missing(sqlite3.OperationalError("disk I/O error"))
