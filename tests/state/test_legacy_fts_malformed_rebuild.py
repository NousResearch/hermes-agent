"""Regression test for legacy inline FTS rebuild on malformed index error (#86027).

SQLite 3.53+ reports 'malformed inverted index for FTS5 table main.messages_fts_trigram'
or 'database disk image is malformed' on DELETE FROM an existing legacy inline
FTS5 table created under an older SQLite version (e.g. 3.46.1).

SessionDB._rebuild_legacy_fts_indexes must catch this error, drop the corrupt virtual
table and triggers, recreate the schema, and repopulate cleanly from canonical messages.
"""

import sqlite3
import pytest

from hermes_state import (
    LEGACY_FTS_SQL,
    LEGACY_FTS_TRIGRAM_SQL,
    SCHEMA_SQL,
    SessionDB,
)


def _create_legacy_db(db_path):
    """Create a database with legacy inline FTS5 schema."""
    conn = sqlite3.connect(str(db_path))
    # Base tables from schema
    conn.executescript(SCHEMA_SQL)
    # Legacy inline FTS
    conn.executescript(LEGACY_FTS_SQL)
    conn.executescript(LEGACY_FTS_TRIGRAM_SQL)
    conn.execute(
        "INSERT INTO sessions(id, title, started_at, source) VALUES('sess-1', 'Test', 1000.0, 'cli')"
    )
    for i in range(1, 11):
        conn.execute(
            "INSERT INTO messages(id, session_id, role, content, tool_name, tool_calls, timestamp) "
            "VALUES(?, 'sess-1', 'user', ?, NULL, NULL, ?)",
            (i, f"Message content {i} with searchable keyword", 1000.0 + i),
        )
    conn.commit()
    conn.close()


def _corrupt_trigram_fts(db_path):
    """Corrupt the trigram FTS shadow table to simulate malformed inverted index."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "UPDATE messages_fts_trigram_data "
        "SET block = X'DEADBEEFDEADBEEFDEADBEEFDEADBEEF'"
    )
    conn.commit()
    conn.close()


def _corrupt_base_fts(db_path):
    """Corrupt the base FTS shadow table to simulate malformed inverted index."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "UPDATE messages_fts_data "
        "SET block = X'DEADBEEFDEADBEEFDEADBEEFDEADBEEF'"
    )
    conn.commit()
    conn.close()


class TestLegacyFtsMalformedRebuild:
    def test_session_open_heals_corrupt_trigram_index(self, tmp_path):
        """A trigger-complete legacy DB whose trigram index is corrupt must
        self-heal on a normal SessionDB open, preserving the inline FTS shape
        (not silently demoting to the v23 external-content schema)."""
        db_path = tmp_path / "state.db"
        _create_legacy_db(db_path)
        _corrupt_trigram_fts(db_path)

        # Verify the corruption manifests on a raw connection before opening.
        conn = sqlite3.connect(str(db_path))
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("DELETE FROM messages_fts_trigram")
        conn.close()

        # A production-shaped open (all six triggers present) must notice and
        # rebuild the corrupt index, not skip the rebuild gate.
        db = SessionDB(db_path=db_path)
        try:
            assert db._db_has_legacy_inline_fts(db._conn)
            quick_check = [r[0] for r in db._conn.execute("PRAGMA quick_check").fetchall()]
            assert quick_check == ["ok"]
            matches = db._conn.execute(
                "SELECT COUNT(*) FROM messages_fts_trigram "
                "WHERE messages_fts_trigram MATCH 'keyword'"
            ).fetchone()[0]
            assert matches == 10
        finally:
            db.close()

    def test_session_open_heals_corrupt_base_fts_index(self, tmp_path):
        """Same self-heal for the base messages_fts inline index."""
        db_path = tmp_path / "state.db"
        _create_legacy_db(db_path)
        _corrupt_base_fts(db_path)

        conn = sqlite3.connect(str(db_path))
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("DELETE FROM messages_fts")
        conn.close()

        db = SessionDB(db_path=db_path)
        try:
            assert db._db_has_legacy_inline_fts(db._conn)
            quick_check = [r[0] for r in db._conn.execute("PRAGMA quick_check").fetchall()]
            assert quick_check == ["ok"]
            matches = db._conn.execute(
                "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'keyword'"
            ).fetchone()[0]
            assert matches == 10
        finally:
            db.close()

    def test_integrity_probe_runs_once_per_sqlite_engine(self, tmp_path):
        """A healthy legacy DB must not rerun FTS integrity-check on every open."""
        db_path = tmp_path / "state.db"
        _create_legacy_db(db_path)
        calls = {"n": 0}
        original = SessionDB._legacy_fts_index_corrupt

        def _counting(self, cursor, *, include_trigram):
            calls["n"] += 1
            return original(self, cursor, include_trigram=include_trigram)

        SessionDB._legacy_fts_index_corrupt = _counting
        try:
            first = SessionDB(db_path=db_path)
            first.close()
            assert calls["n"] == 1
            marker = sqlite3.connect(str(db_path)).execute(
                "SELECT value FROM state_meta WHERE key = 'fts_integrity_engine'"
            ).fetchone()
            assert marker and str(marker[0]).startswith("fts5:")

            second = SessionDB(db_path=db_path)
            second.close()
            assert calls["n"] == 1
        finally:
            SessionDB._legacy_fts_index_corrupt = original

    def test_non_malformed_error_is_not_classified_as_corruption(self):
        """Only the malformed-index class justifies a destructive rebuild; a
        transient lock/busy/IO DatabaseError must fall through to the caller's
        retry path untouched."""
        locked = sqlite3.OperationalError("database is locked")
        busy = sqlite3.OperationalError("database is busy")
        io_error = sqlite3.OperationalError("disk I/O error")
        malformed = sqlite3.DatabaseError(
            "malformed inverted index for FTS5 table main.messages_fts_trigram"
        )
        disk_image = sqlite3.DatabaseError("database disk image is malformed")

        for exc in (locked, busy, io_error):
            assert SessionDB._is_malformed_fts_index_error(exc) is False
        for exc in (malformed, disk_image):
            assert SessionDB._is_malformed_fts_index_error(exc) is True
