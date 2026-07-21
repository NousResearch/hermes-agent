"""#68858: FTS UPDATE triggers must only fire on content column changes.

Status-only updates (active, compacted, observed) must NOT trigger
FTS delete/reinsert, which saturates disk I/O on large state.db during
in-place compaction.
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


def test_fts_update_trigger_fires_on_content_change(db_path):
    """Updating content must reindex FTS."""
    conn = _create_db(db_path)
    conn.execute(
        "INSERT INTO messages (id, content, tool_name, tool_calls, active) "
        "VALUES (1, 'hello world', '', '', 1)"
    )
    conn.commit()

    # Verify FTS has the initial content
    rows = conn.execute("SELECT content FROM messages_fts WHERE content MATCH 'hello'").fetchall()
    assert len(rows) == 1

    # Update content → FTS must reindex
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

    # Status-only update: compaction marks active=0, compacted=1
    conn.execute("UPDATE messages SET active = 0, compacted = 1 WHERE id = 1")
    conn.commit()

    # FTS content must be unchanged — the row was never deleted/reinserted
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
