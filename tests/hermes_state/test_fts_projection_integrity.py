"""Regression for the ``messages_fts`` external-content drift (issue #114169).

The base word index used to read its external content from the raw ``messages``
table while the triggers indexed only a bounded prefix of every long tool row
and the boundary itself moved with a ``state_meta`` high-water marker. FTS5's
strict ``rank=1`` integrity check re-reads the content source and compares it
with the stored token stream, so that construction could not stay consistent: a
long tool row was enough to make the check report ``fts5: checksum mismatch``,
and the delete/update commands sent full content for a truncated row, leaving
index tokens behind.

These are behaviour contracts on the projection, not on its shape: the strict
probe must survive arbitrary churn, and an index left reading raw ``messages``
must realign once, on open.
"""

import sqlite3

import pytest

from hermes_state import SessionDB
from hermes_state_common import FTS_STORAGE_VERSION, FTS_TOOL_CONTENT_PREFIX_CHARS

LONG_TOOL_ROW = "prefix " + ("padding " * ((FTS_TOOL_CONTENT_PREFIX_CHARS // 8) + 64)) + " tailtoken"


def _strict_integrity_probe(db) -> None:
    """FTS5's strict check: re-reads the content source and compares it with the
    stored token stream. Raises ``sqlite3.DatabaseError`` when they disagree."""
    db._conn.execute(
        "INSERT INTO messages_fts(messages_fts, rank) VALUES('integrity-check', 1)"
    )


@pytest.fixture
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    if not session_db._fts_enabled:
        session_db.close()
        pytest.skip("SQLite FTS5 unavailable")
    session_db.create_session("session", source="cli")
    try:
        yield session_db
    finally:
        session_db.close()


def test_strict_integrity_probe_survives_tool_row_churn(db):
    """Every write path must leave the index readable by FTS5's own checker.

    Pre-fix, the long-row assertion raises ``DatabaseError: fts5: checksum
    mismatch for table "messages_fts"``: the row was indexed as a prefix while
    ``content='messages'`` kept the full body, so the checker disagreed with the
    stored tokens. The UPDATE and DELETE legs are the triggers whose content
    command used to be re-evaluated against a moving mark.
    """
    _strict_integrity_probe(db)

    long_id = db.append_message(
        "session", role="tool", content=LONG_TOOL_ROW, tool_name="terminal"
    )
    _strict_integrity_probe(db)

    db._execute_write(
        lambda conn: conn.execute(
            "UPDATE messages SET content = ? WHERE id = ?", ("short body", long_id)
        )
    )
    _strict_integrity_probe(db)

    db.append_message("session", role="user", content="short user row")
    _strict_integrity_probe(db)

    db._execute_write(
        lambda conn: conn.execute("DELETE FROM messages WHERE id = ?", (long_id,))
    )
    _strict_integrity_probe(db)


def test_index_reading_raw_messages_realigns_once_on_open(tmp_path):
    """A store whose index still reads raw ``messages`` (the shipped shape for
    versions before the aligned projection) realigns on the next open, records
    the new storage version, and is consistent from then on."""
    path = tmp_path / "state.db"
    first = SessionDB(db_path=path)
    if not first._fts_enabled:
        first.close()
        pytest.skip("SQLite FTS5 unavailable")
    first.create_session("session", source="cli")
    row_id = first.append_message(
        "session", role="tool", content=LONG_TOOL_ROW, tool_name="terminal"
    )

    # Rewind to the pre-fix on-disk shape: the base index reads raw `messages`,
    # and its tokens were written from the bounded projection the triggers used
    # for tool rows. That mixture IS the drift: the checker re-reads the full
    # body and disagrees with the stored prefix.
    first._conn.execute("DROP TABLE messages_fts")
    first._conn.execute(
        "CREATE VIRTUAL TABLE messages_fts USING fts5("
        "content, tool_name, tool_calls, content='messages', content_rowid='id')"
    )
    first._conn.execute(
        "INSERT INTO messages_fts(rowid, content, tool_name, tool_calls) VALUES(?, ?, ?, ?)",
        (row_id, LONG_TOOL_ROW[:FTS_TOOL_CONTENT_PREFIX_CHARS], "terminal", None),
    )
    first._conn.execute(
        "INSERT INTO state_meta(key, value) VALUES('fts_storage_version', '2') "
        "ON CONFLICT(key) DO UPDATE SET value = '2'"
    )
    first._conn.execute(
        "INSERT OR REPLACE INTO state_meta(key, value) VALUES('fts_tool_full_content_high_water', ?)",
        (str(row_id),),
    )
    with pytest.raises(sqlite3.DatabaseError):
        _strict_integrity_probe(first)
    first.close()

    migrated = SessionDB(db_path=path)
    try:
        assert migrated.get_meta("fts_storage_version") == str(FTS_STORAGE_VERSION)
        assert migrated.get_meta("fts_tool_full_content_high_water") is None
        _strict_integrity_probe(migrated)
        index_sql = migrated._conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'messages_fts'"
        ).fetchone()[0]
        assert "messages_fts_src" in index_sql
        # The realigned index answers the prefix; the tail comes from the
        # explicit tool path, which reads stored content rather than the index.
        assert migrated.search_messages("prefix", role_filter=["tool"])
        assert [
            row["id"]
            for row in migrated.search_messages("tailtoken", role_filter=["tool"])
        ] == [row_id]
    finally:
        migrated.close()
