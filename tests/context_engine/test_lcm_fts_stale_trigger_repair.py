"""Regression tests for messages_fts stale-trigger detection + repair.

Background: an older LCM schema indexed a ``content`` column on ``messages``;
the current schema renamed it to ``search_content`` and recreated the FTS
virtual table accordingly. On databases created before the rename, the three
``msg_fts_*`` triggers were left pointing at the dropped ``content`` column.
They still exist *by name*, so the name-only ``_fts_missing_triggers`` check
reported "ok" while every message INSERT aborted at runtime with
``table messages_fts has no column named content`` — LCM could not persist a
single message, yet doctor/repair considered the store healthy.

These tests lock in the body-aware drift detection (``_fts_stale_triggers``)
and the drop-then-recreate repair path so the regression cannot silently
return.
"""

from __future__ import annotations

import sqlite3

from plugins.context_engine.lcm.db_bootstrap import (
    _fts_missing_triggers,
    _fts_stale_triggers,
    external_content_fts_needs_repair,
    repair_external_content_fts,
)
from plugins.context_engine.lcm.store import build_message_fts_spec

# Mirrors the messages-table columns the live schema carries (both the legacy
# ``content`` and current ``search_content`` exist on a drifted DB).
_MESSAGES_DDL = """
CREATE TABLE messages (
    store_id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    source TEXT DEFAULT '',
    role TEXT NOT NULL,
    content TEXT,
    tool_call_id TEXT,
    tool_calls TEXT,
    tool_name TEXT,
    timestamp REAL NOT NULL,
    token_estimate INTEGER DEFAULT 0,
    pinned INTEGER DEFAULT 0,
    search_content TEXT
);
CREATE VIRTUAL TABLE messages_fts USING fts5(
    search_content, content="messages", content_rowid="store_id"
);
"""

# The pre-rename triggers: correct names, but bodies reference the dropped
# ``content`` column. This is the exact on-disk shape that broke Apollo's store.
_STALE_TRIGGERS = """
CREATE TRIGGER msg_fts_insert AFTER INSERT ON messages BEGIN
  INSERT INTO messages_fts(rowid, content) VALUES (new.store_id, new.content);
END;
CREATE TRIGGER msg_fts_delete AFTER DELETE ON messages BEGIN
  INSERT INTO messages_fts(messages_fts, rowid, content)
    VALUES('delete', old.store_id, old.content);
END;
CREATE TRIGGER msg_fts_update AFTER UPDATE OF content ON messages BEGIN
  INSERT INTO messages_fts(messages_fts, rowid, content)
    VALUES('delete', old.store_id, old.content);
  INSERT INTO messages_fts(rowid, content) VALUES (new.store_id, new.content);
END;
"""


def _make_db(conn: sqlite3.Connection, *, stale: bool) -> None:
    conn.executescript(_MESSAGES_DDL)
    if stale:
        conn.executescript(_STALE_TRIGGERS)
    else:
        for trigger_sql in build_message_fts_spec().trigger_sqls:
            conn.execute(trigger_sql)
    conn.commit()


def _insert(conn: sqlite3.Connection, body: str) -> None:
    conn.execute(
        "INSERT INTO messages (session_id, role, content, timestamp, search_content) "
        "VALUES ('s', 'user', ?, 1.0, ?)",
        (body, body),
    )


def test_stale_trigger_is_invisible_to_name_only_check():
    """The pre-existing name-only check must NOT flag the drift (documents the gap)."""
    conn = sqlite3.connect(":memory:")
    _make_db(conn, stale=True)
    spec = build_message_fts_spec()
    # Triggers exist by name, so the name-only check sees nothing wrong...
    assert _fts_missing_triggers(conn, spec) is False
    # ...but the body-aware check catches the drift.
    assert _fts_stale_triggers(conn, spec) is True
    assert external_content_fts_needs_repair(conn, spec) is True
    conn.close()


def test_stale_trigger_aborts_insert_before_repair():
    """Reproduce the real symptom: an INSERT aborts on the missing column."""
    conn = sqlite3.connect(":memory:")
    _make_db(conn, stale=True)
    raised = False
    try:
        _insert(conn, "hello")
    except sqlite3.OperationalError as exc:
        raised = True
        assert "messages_fts has no column named content" in str(exc)
        conn.rollback()
    assert raised, "expected the stale trigger to abort the insert"
    conn.close()


def test_repair_recreates_stale_triggers_and_insert_persists():
    """The full fix: repair drops+recreates the drifted triggers; writes persist+index."""
    conn = sqlite3.connect(":memory:")
    _make_db(conn, stale=True)
    spec = build_message_fts_spec()

    result = repair_external_content_fts(conn, spec, throttle=False)
    assert result["triggers_recreated"] is True
    assert external_content_fts_needs_repair(conn, spec) is False

    # Insert now persists AND is reachable through the FTS index.
    _insert(conn, "HELLO_TOKEN")
    conn.commit()
    assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1
    hits = conn.execute(
        "SELECT m.store_id FROM messages_fts f JOIN messages m ON f.rowid = m.store_id "
        "WHERE messages_fts MATCH 'HELLO_TOKEN'"
    ).fetchall()
    assert hits, "repaired trigger should index the inserted row"
    conn.close()


def test_healthy_db_is_not_flagged_for_repair():
    """Discriminating guard: a correct schema must NOT be reported as needing repair."""
    conn = sqlite3.connect(":memory:")
    _make_db(conn, stale=False)
    spec = build_message_fts_spec()
    assert _fts_stale_triggers(conn, spec) is False
    assert _fts_missing_triggers(conn, spec) is False
    assert external_content_fts_needs_repair(conn, spec) is False
    # And it works end to end without any repair.
    _insert(conn, "OK_TOKEN")
    conn.commit()
    hits = conn.execute(
        "SELECT m.store_id FROM messages_fts f JOIN messages m ON f.rowid = m.store_id "
        "WHERE messages_fts MATCH 'OK_TOKEN'"
    ).fetchall()
    assert hits
    conn.close()


def test_missing_trigger_still_repaired():
    """Sibling case: a fully-missing trigger is still detected and recreated."""
    conn = sqlite3.connect(":memory:")
    _make_db(conn, stale=False)
    spec = build_message_fts_spec()
    conn.execute("DROP TRIGGER msg_fts_insert")
    conn.commit()
    assert _fts_missing_triggers(conn, spec) is True
    assert external_content_fts_needs_repair(conn, spec) is True

    result = repair_external_content_fts(conn, spec, throttle=False)
    assert result["triggers_recreated"] is True
    assert external_content_fts_needs_repair(conn, spec) is False
    conn.close()
