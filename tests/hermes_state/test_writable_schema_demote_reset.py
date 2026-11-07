"""Failed FTS demote must reset writable_schema without masking the DELETE error."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from unittest import mock

import pytest

from hermes_state import SessionDB
from hermes_state_common import SCHEMA_SQL


def _build_v22_db(db_path: Path) -> None:
    """Minimal v22-shaped DB: inline FTS virtual tables so demote enters schema surgery."""
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA_SQL)
    conn.executescript("""
        DROP TABLE IF EXISTS messages_fts;
        DROP TABLE IF EXISTS messages_fts_trigram;
        DROP VIEW IF EXISTS messages_fts_trigram_src;

        CREATE VIRTUAL TABLE messages_fts USING fts5(content);
        CREATE TRIGGER messages_fts_insert AFTER INSERT ON messages BEGIN
            INSERT INTO messages_fts(rowid, content) VALUES (
                new.id,
                COALESCE(new.content, '')
            );
        END;

        CREATE VIRTUAL TABLE messages_fts_trigram USING fts5(content, tokenize='trigram');
        CREATE TRIGGER messages_fts_trigram_insert AFTER INSERT ON messages BEGIN
            INSERT INTO messages_fts_trigram(rowid, content) VALUES (
                new.id,
                COALESCE(new.content, '')
            );
        END;
    """)
    conn.execute("DELETE FROM schema_version")
    conn.execute("INSERT INTO schema_version (version) VALUES (22)")
    conn.execute(
        "INSERT INTO sessions (id, source, started_at) VALUES ('s1', 'cli', ?)",
        (time.time(),),
    )
    conn.commit()
    conn.close()


def test_demote_resets_writable_schema_on_delete_error(tmp_path):
    """If the sqlite_master DELETE inside writable_schema=ON raises,
    the finally block must reset the switch on the long-lived write
    connection before the exception propagates."""
    db_path = tmp_path / "v22.db"
    _build_v22_db(db_path)

    db = SessionDB(db_path=db_path)
    try:
        conn = db._conn
        assert conn is not None
        orig_execute = conn.execute

        def boom(sql, parameters=()):
            if sql.strip().startswith("DELETE FROM sqlite_master"):
                raise sqlite3.OperationalError("simulated delete failure")
            return orig_execute(sql, parameters)

        with mock.patch.object(conn, "execute", side_effect=boom):
            with pytest.raises(sqlite3.OperationalError, match="simulated delete failure"):
                db._demote_legacy_fts_to_trash()

        value = conn.execute("PRAGMA writable_schema").fetchone()[0]
        assert value in (0, None, False), f"writable_schema should be off after failed demote, got {value!r}"
        assert conn.execute("SELECT 1").fetchone()[0] == 1
    finally:
        db.close()
