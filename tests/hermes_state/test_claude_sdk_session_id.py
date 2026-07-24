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


def _read_only_db_with_trigram_probe_error(tmp_path, monkeypatch, message):
    """Open a read-only SessionDB whose TRIGRAM probe raises `message`.

    The seed DB is created first with a normal write handle (schema load),
    THEN sqlite3.connect is wrapped so only the trigram probe statement
    errors — the messages_fts probe and everything else run for real.
    """
    import sqlite3

    import hermes_state

    db_path = tmp_path / "state.db"
    SessionDB(db_path=db_path).close()

    real_connect = sqlite3.connect

    class _ProbeErrorConn:
        def __init__(self, real):
            self.__dict__["_real"] = real

        def execute(self, sql, *args, **kwargs):
            if "messages_fts_trigram" in sql:
                raise sqlite3.OperationalError(message)
            return self.__dict__["_real"].execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.__dict__["_real"], name)

        def __setattr__(self, name, value):
            setattr(self.__dict__["_real"], name, value)

    monkeypatch.setattr(
        hermes_state.sqlite3,
        "connect",
        lambda *args, **kwargs: _ProbeErrorConn(real_connect(*args, **kwargs)),
    )
    return SessionDB(db_path=db_path, read_only=True)


def test_trigram_probe_transient_error_keeps_trigram_available(tmp_path, monkeypatch):
    # Same transient-vs-absent rule as the messages_fts probe directly above
    # it: a lock during a checkpoint must not latch the LIKE fallback (which
    # ORs tokens and drops NOT/rank) for the handle's lifetime. A wrongly-kept
    # True costs nothing — search_messages catches the per-query error and
    # falls through to LIKE.
    db = _read_only_db_with_trigram_probe_error(
        tmp_path, monkeypatch, "database is locked"
    )
    try:
        assert db._trigram_available is True
        assert db._fts_enabled is True
    finally:
        db.close()


def test_trigram_probe_missing_table_disables_trigram(tmp_path, monkeypatch):
    db = _read_only_db_with_trigram_probe_error(
        tmp_path, monkeypatch, "no such table: messages_fts_trigram"
    )
    try:
        assert db._trigram_available is False
    finally:
        db.close()


def test_trigram_probe_missing_tokenizer_disables_trigram(tmp_path, monkeypatch):
    # A build with FTS5 but without the trigram tokenizer (SQLite < 3.34)
    # raises "no such tokenizer: trigram" — persistent absence, same latch as
    # a missing table. _fts_object_missing alone does NOT classify this one;
    # the probe must also consult _is_trigram_unavailable_error.
    db = _read_only_db_with_trigram_probe_error(
        tmp_path, monkeypatch, "no such tokenizer: trigram"
    )
    try:
        assert db._trigram_available is False
    finally:
        db.close()
