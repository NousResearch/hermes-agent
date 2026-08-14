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
    def test_rebuild_legacy_fts_indexes_recovers_from_corrupt_trigram(self, tmp_path):
        db_path = tmp_path / "state.db"
        _create_legacy_db(db_path)
        _corrupt_trigram_fts(db_path)

        # Verify that DELETE FROM messages_fts_trigram fails on the corrupt table
        conn = sqlite3.connect(str(db_path))
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("DELETE FROM messages_fts_trigram")
        conn.close()

        # Rebuild using SessionDB._rebuild_legacy_fts_indexes
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        SessionDB._rebuild_legacy_fts_indexes(cursor, include_trigram=True)
        conn.commit()

        # Verify index is healthy and populated
        quick_check = [r[0] for r in conn.execute("PRAGMA quick_check").fetchall()]
        assert quick_check == ["ok"]

        # Verify search queries work
        matches = conn.execute(
            "SELECT COUNT(*) FROM messages_fts_trigram WHERE messages_fts_trigram MATCH 'keyword'"
        ).fetchone()[0]
        assert matches == 10

        base_matches = conn.execute(
            "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'keyword'"
        ).fetchone()[0]
        assert base_matches == 10
        conn.close()

    def test_rebuild_legacy_fts_indexes_recovers_from_corrupt_base_fts(self, tmp_path):
        db_path = tmp_path / "state.db"
        _create_legacy_db(db_path)
        _corrupt_base_fts(db_path)

        # Verify that DELETE FROM messages_fts fails on the corrupt table
        conn = sqlite3.connect(str(db_path))
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("DELETE FROM messages_fts")
        conn.close()

        # Rebuild using SessionDB._rebuild_legacy_fts_indexes
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        SessionDB._rebuild_legacy_fts_indexes(cursor, include_trigram=True)
        conn.commit()

        # Verify index is healthy and populated
        quick_check = [r[0] for r in conn.execute("PRAGMA quick_check").fetchall()]
        assert quick_check == ["ok"]

        base_matches = conn.execute(
            "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'keyword'"
        ).fetchone()[0]
        assert base_matches == 10
        conn.close()
