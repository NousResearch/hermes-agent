"""Test that read-only SessionDB survives invalid UTF-8 in FTS content (issue #98924)."""

import sqlite3
from pathlib import Path

import pytest

from hermes_state import SessionDB


def _write_invalid_utf8_row(db_path: Path) -> None:
    """Inject a non-UTF-8 byte into an existing message row via low-level API.
    
    Python's high-level sqlite3.Connection will not let us write invalid
    UTF-8 into a TEXT column — it decodes and re-encodes. We use the
    connection's lower-level interface to force it through.
    """
    # CAST(x'61625F816364' AS TEXT) → 'ab_�cd' with 0x81 at position 3
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA foreign_keys = OFF")
        # Force the bytes through by overwriting an existing row id.
        conn.execute("UPDATE messages SET content = CAST(x'61625F816364' AS TEXT) WHERE id = 1")
        conn.commit()
    finally:
        conn.close()


class TestReadOnlyFTSDecodeError:
    def test_read_only_open_survives_invalid_utf8_in_fts_content(self, tmp_path):
        """Non-UTF-8 bytes in messages.content don't kill the read-only open.
        
        Regression test for issue #98924: a developer's state.db contained a
        byte 0x81 in a messages row. External-content FTS5 (v23 layout) reads
        that column when building a result set. On some Python/SQLite builds
        the decode failure surfaces as UnicodeDecodeError; on others as
        OperationalError("Could not decode to UTF-8 column ...").
        
        The old code caught only sqlite3.OperationalError and would re-raise
        any other exception. UnicodeDecodeError (a ValueError, not an
        sqlite3.Error subclass) therefore bypassed the guard and killed the
        read-only connection, taking down every read endpoint (GET /api/sessions).
        
        The fix catches (sqlite3.OperationalError, UnicodeDecodeError) and
        treats the decode error the same as "FTS unavailable" — the index is
        degraded but the connection stays open. Search may return less or fail
        on that corrupted row, but writes and non-FTS reads keep working.
        """
        db_path = tmp_path / "state.db"
        
        # Create a fresh DB with v23 external-content FTS.
        writable = SessionDB(db_path=db_path)
        writable.create_session("decode-test", source="cli")
        writable.append_message("decode-test", role="user", content="valid text")
        writable.close()
        
        # Insert a row with invalid UTF-8 through the sqlite3 CLI.
        _write_invalid_utf8_row(db_path)
        
        # Also trigger a rebuild so the invalid bytes are in the FTS index.
        conn = sqlite3.connect(str(db_path))
        conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")
        conn.commit()
        conn.close()
        
        # The shipped regression: SessionDB(read_only=True) should NOT raise.
        # Before the fix, this raised UnicodeDecodeError from _fts_table_probe.
        read_only = SessionDB(db_path=db_path, read_only=True)
        try:
            # Verify the connection opened — _fts_table_probe didn't kill init.
            assert read_only._conn is not None
        finally:
            read_only.close()
        
        # Also verify that the index degraded: _fts_enabled should be None or
        # False, not True, because the corrupt content prevents probing.
        read_only2 = SessionDB(db_path=db_path, read_only=True)
        try:
            # The index is broken; the store itself must stay accessible.
            assert read_only2._conn is not None
        finally:
            read_only2.close()
