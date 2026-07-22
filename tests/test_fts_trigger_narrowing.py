"""#68858/#68891: FTS UPDATE triggers must only fire on content column changes.

Status-only updates (active, compacted, observed) must NOT trigger
FTS delete/reinsert, which saturates disk I/O on large state.db during
in-place compaction.

Migration tests (#68891 review): broad→narrow trigger migration must be
conditional, atomic, idempotent, and must NOT rebuild FTS indexes.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "state.db"


def _create_db(db_path: Path) -> sqlite3.Connection:
    """Create a fresh state.db with the FTS schema from hermes_state."""
    from hermes_state import FTS_SQL, FTS_TRIGRAM_SQL

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE messages ("
        "id INTEGER PRIMARY KEY, content TEXT, tool_name TEXT, "
        "tool_calls TEXT, active INTEGER, compacted INTEGER, "
        "observed INTEGER, api_content TEXT)"
    )
    conn.executescript(FTS_SQL)
    conn.executescript(FTS_TRIGRAM_SQL)
    conn.commit()
    return conn


def _create_db_with_broad_triggers(db_path: Path) -> sqlite3.Connection:
    """Create a state.db with old-style broad UPDATE triggers (pre-migration)."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE messages ("
        "id INTEGER PRIMARY KEY, content TEXT, tool_name TEXT, "
        "tool_calls TEXT, active INTEGER, compacted INTEGER, "
        "observed INTEGER, api_content TEXT)"
    )
    # Create FTS tables without triggers
    conn.execute("CREATE VIRTUAL TABLE messages_fts USING fts5(content)")
    conn.execute("CREATE VIRTUAL TABLE messages_fts_trigram USING fts5(content, tokenize='trigram')")
    # Old-style broad triggers (the ones we're migrating from)
    conn.executescript("""
        CREATE TRIGGER messages_fts_insert AFTER INSERT ON messages BEGIN
            INSERT INTO messages_fts(rowid, content) VALUES (
                new.id,
                COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
            );
        END;
        CREATE TRIGGER messages_fts_delete AFTER DELETE ON messages BEGIN
            DELETE FROM messages_fts WHERE rowid = old.id;
        END;
        CREATE TRIGGER messages_fts_update AFTER UPDATE ON messages BEGIN
            DELETE FROM messages_fts WHERE rowid = old.id;
            INSERT INTO messages_fts(rowid, content) VALUES (
                new.id,
                COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
            );
        END;
        CREATE TRIGGER messages_fts_trigram_insert AFTER INSERT ON messages BEGIN
            INSERT INTO messages_fts_trigram(rowid, content) VALUES (
                new.id,
                COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
            );
        END;
        CREATE TRIGGER messages_fts_trigram_delete AFTER DELETE ON messages BEGIN
            DELETE FROM messages_fts_trigram WHERE rowid = old.id;
        END;
        CREATE TRIGGER messages_fts_trigram_update AFTER UPDATE ON messages BEGIN
            DELETE FROM messages_fts_trigram WHERE rowid = old.id;
            INSERT INTO messages_fts_trigram(rowid, content) VALUES (
                new.id,
                COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
            );
        END;
    """)
    conn.commit()
    return conn


def _get_trigger_sql(conn, name: str) -> str:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='trigger' AND name=?", (name,)
    ).fetchone()
    return row[0] if row else ""


def test_fts_update_trigger_fires_on_content_change(db_path):
    """Updating content must reindex FTS."""
    conn = _create_db(db_path)
    conn.execute(
        "INSERT INTO messages (id, content, tool_name, tool_calls, active) "
        "VALUES (1, 'hello world', '', '', 1)"
    )
    conn.commit()

    rows = conn.execute("SELECT content FROM messages_fts WHERE content MATCH 'hello'").fetchall()
    assert len(rows) == 1

    conn.execute("UPDATE messages SET content = 'goodbye world' WHERE id = 1")
    conn.commit()

    rows = conn.execute("SELECT content FROM messages_fts WHERE content MATCH 'goodbye'").fetchall()
    assert len(rows) == 1
    rows = conn.execute("SELECT content FROM messages_fts WHERE content MATCH 'hello'").fetchall()
    assert len(rows) == 0
    conn.close()


def test_fts_update_trigger_does_not_fire_on_status_only_change(db_path):
    """Updating only active/compacted must NOT trigger FTS reindex (#68858)."""
    conn = _create_db(db_path)
    conn.execute(
        "INSERT INTO messages (id, content, tool_name, tool_calls, active) "
        "VALUES (1, 'keep this text', '', '', 1)"
    )
    conn.commit()

    conn.execute("UPDATE messages SET active = 0, compacted = 1 WHERE id = 1")
    conn.commit()

    rows = conn.execute("SELECT content FROM messages_fts WHERE content MATCH 'keep'").fetchall()
    assert len(rows) == 1
    conn.close()


def test_trigram_update_trigger_fires_on_content_change(db_path):
    """Updating content must reindex trigram FTS."""
    conn = _create_db(db_path)
    conn.execute(
        "INSERT INTO messages (id, content, tool_name, tool_calls, active) "
        "VALUES (1, '中文测试', '', '', 1)"
    )
    conn.commit()

    rows = conn.execute(
        "SELECT content FROM messages_fts_trigram WHERE content MATCH '中文测'"
    ).fetchall()
    assert len(rows) == 1

    conn.execute("UPDATE messages SET content = '日本語テスト' WHERE id = 1")
    conn.commit()

    rows = conn.execute(
        "SELECT content FROM messages_fts_trigram WHERE content MATCH '日本語'"
    ).fetchall()
    assert len(rows) == 1
    conn.close()


def test_trigram_update_trigger_does_not_fire_on_status_only_change(db_path):
    """Updating only active/compacted must NOT trigger trigram FTS reindex."""
    conn = _create_db(db_path)
    conn.execute(
        "INSERT INTO messages (id, content, tool_name, tool_calls, active) "
        "VALUES (1, '中文保持测试', '', '', 1)"
    )
    conn.commit()

    conn.execute("UPDATE messages SET active = 0, compacted = 1 WHERE id = 1")
    conn.commit()

    rows = conn.execute(
        "SELECT content FROM messages_fts_trigram WHERE content MATCH '中文保持'"
    ).fetchall()
    assert len(rows) == 1
    conn.close()


# ── #68891 migration tests ────────────────────────────────────────────


def test_migration_detects_broad_trigger(db_path):
    """Broad UPDATE triggers must be detected as needing migration."""
    conn = _create_db_with_broad_triggers(db_path)
    sql = _get_trigger_sql(conn, "messages_fts_update")
    assert "AFTER UPDATE ON messages BEGIN" in sql
    assert "AFTER UPDATE OF content" not in sql
    conn.close()


def test_migration_does_not_fire_on_already_narrow(db_path):
    """Already-narrowed triggers must NOT be flagged for migration (idempotent)."""
    conn = _create_db(db_path)  # _create_db uses FTS_SQL which has narrowed triggers
    sql = _get_trigger_sql(conn, "messages_fts_update")
    assert "AFTER UPDATE OF content, tool_name, tool_calls ON messages" in sql
    conn.close()


def test_broad_trigger_still_works_before_migration(db_path):
    """Broad triggers still index content correctly (they over-index, not miss)."""
    conn = _create_db_with_broad_triggers(db_path)
    conn.execute(
        "INSERT INTO messages (id, content, tool_name, tool_calls, active) "
        "VALUES (1, 'test content', '', '', 1)"
    )
    conn.commit()
    rows = conn.execute("SELECT content FROM messages_fts WHERE content MATCH 'test'").fetchall()
    assert len(rows) == 1
    conn.close()


def test_narrowed_trigger_preserves_existing_fts_content(db_path):
    """After migration, existing FTS content must remain valid — no rebuild needed."""
    conn = _create_db_with_broad_triggers(db_path)
    # Insert data (broad triggers index it)
    conn.execute(
        "INSERT INTO messages (id, content, tool_name, tool_calls, active) "
        "VALUES (1, 'preserved text', '', '', 1)"
    )
    conn.commit()
    # Verify FTS has the data
    rows = conn.execute("SELECT content FROM messages_fts WHERE content MATCH 'preserved'").fetchall()
    assert len(rows) == 1

    # Now drop broad UPDATE triggers and recreate with narrowed definitions
    conn.executescript("""
        DROP TRIGGER IF EXISTS messages_fts_update;
        DROP TRIGGER IF EXISTS messages_fts_trigram_update;
        CREATE TRIGGER messages_fts_update AFTER UPDATE OF content, tool_name, tool_calls ON messages BEGIN
            DELETE FROM messages_fts WHERE rowid = old.id;
            INSERT INTO messages_fts(rowid, content) VALUES (
                new.id,
                COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
            );
        END;
        CREATE TRIGGER messages_fts_trigram_update AFTER UPDATE OF content, tool_name, tool_calls ON messages BEGIN
            DELETE FROM messages_fts_trigram WHERE rowid = old.id;
            INSERT INTO messages_fts_trigram(rowid, content) VALUES (
                new.id,
                COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
            );
        END;
    """)
    conn.commit()

    # FTS content must still be there — no rebuild needed
    rows = conn.execute("SELECT content FROM messages_fts WHERE content MATCH 'preserved'").fetchall()
    assert len(rows) == 1, "FTS content must survive migration without rebuild"
    conn.close()


def test_migration_source_is_idempotent(db_path):
    """The migration code in hermes_state.py must inspect, not unconditionally drop."""
    import inspect
    from hermes_state import SessionDB

    src = inspect.getsource(SessionDB._init_schema)
    # Must inspect sqlite_master for trigger definition
    assert "sqlite_master" in src, "Migration must inspect sqlite_master.sql"
    assert "BEGIN IMMEDIATE" in src, "Migration must use BEGIN IMMEDIATE for atomicity"
    # Must NOT unconditionally force rebuild after narrowing
    assert "triggers_need_repair = True  # force rebuild" not in src, \
        "Migration must not force FTS rebuild when narrowing triggers"