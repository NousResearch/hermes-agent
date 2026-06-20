"""End-to-end legacy-DB migration tests for the LCM message store.

These tests exercise the REAL ``MessageStore.__init__`` startup path against
hand-built fixtures that reproduce the two on-disk drift shapes seen on the live
fleet, then prove a write actually persists AND is FTS-searchable afterwards.

Why this layer (and why the original bug slipped through): every prior LCM test
created a *fresh* store on the *current* schema, so the FTS triggers were always
correct and writes always worked. The bug only existed on databases created
before the ``content -> search_content`` column rename and then carried forward.
A fresh-store test can never reproduce that — only opening a pre-existing
legacy DB through the genuine init path does. These tests build that legacy
state explicitly so a future schema change that breaks old-DB upgrade fails
loudly here.

Two drift shapes, both from real fleet DBs (2026-06-19):

* **Half-migrated (Apollo):** ``messages`` already has ``search_content`` and the
  FTS table indexes ``search_content``, but the ``msg_fts_*`` triggers were left
  referencing the dropped ``content`` column -> every INSERT aborts.
* **Never-migrated (daedalus):** ``messages`` has only ``content``, the FTS table
  indexes ``content``, triggers reference ``content``. Internally consistent on
  the OLD schema; the store must migrate it to ``search_content`` on open without
  crashing, then writes must work.
"""

from __future__ import annotations

import sqlite3

import pytest

from plugins.context_engine.lcm.store import MessageStore


def _insert_and_search(db_path: str, token: str) -> bool:
    """Insert a row through the live schema and confirm it is FTS-searchable."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp, search_content) "
            "VALUES ('s', 'user', ?, 1.0, ?)",
            (token, token),
        )
        conn.commit()
        hits = conn.execute(
            "SELECT m.store_id FROM messages_fts f "
            "JOIN messages m ON f.rowid = m.store_id "
            "WHERE messages_fts MATCH ?",
            (token,),
        ).fetchall()
        return bool(hits)
    finally:
        conn.close()


def _insert_trigger_column(db_path: str) -> str:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='trigger' AND name='msg_fts_insert'"
        ).fetchone()
        if not row:
            return "MISSING"
        return "search_content" if "search_content" in row[0] else "content"
    finally:
        conn.close()


def _build_half_migrated_db(db_path: str) -> None:
    """Apollo shape: search_content column present, but triggers still on content."""
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE messages (
            store_id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, source TEXT DEFAULT '',
            role TEXT NOT NULL, content TEXT, tool_call_id TEXT, tool_calls TEXT,
            tool_name TEXT, timestamp REAL NOT NULL, token_estimate INTEGER DEFAULT 0,
            pinned INTEGER DEFAULT 0, search_content TEXT
        );
        CREATE VIRTUAL TABLE messages_fts USING fts5(
            search_content, content="messages", content_rowid="store_id"
        );
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
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT);
        """
    )
    conn.commit()
    conn.close()


def _build_never_migrated_db(db_path: str) -> None:
    """daedalus shape: only a content column; FTS + triggers all on content."""
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE messages (
            store_id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, source TEXT DEFAULT '',
            role TEXT NOT NULL, content TEXT, tool_call_id TEXT, tool_calls TEXT,
            tool_name TEXT, timestamp REAL NOT NULL, token_estimate INTEGER DEFAULT 0,
            pinned INTEGER DEFAULT 0
        );
        CREATE VIRTUAL TABLE messages_fts USING fts5(
            content, content="messages", content_rowid="store_id"
        );
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
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT);
        """
    )
    conn.commit()
    conn.close()


def test_half_migrated_db_opens_and_writes(tmp_path):
    """A half-migrated (Apollo-shape) DB must self-repair on open and accept writes."""
    db = str(tmp_path / "half.db")
    _build_half_migrated_db(db)
    assert _insert_trigger_column(db) == "content"  # drifted before open

    # The genuine startup path. Must not raise.
    MessageStore(db)

    assert _insert_trigger_column(db) == "search_content"  # repaired
    assert _insert_and_search(db, "HALF_TOKEN") is True


def test_never_migrated_db_opens_and_writes(tmp_path):
    """A never-migrated (daedalus-shape) DB must migrate on open and accept writes."""
    db = str(tmp_path / "never.db")
    _build_never_migrated_db(db)

    # Pre: messages has no search_content column at all.
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    pre_cols = [r[1] for r in conn.execute("PRAGMA table_info(messages)")]
    conn.close()
    assert "search_content" not in pre_cols

    # The genuine startup path. Must not raise (the original isolated-repair
    # crash 'no such column: T.search_content' happened when the column-add
    # migration was skipped; the real init runs it first).
    MessageStore(db)

    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    post_cols = [r[1] for r in conn.execute("PRAGMA table_info(messages)")]
    conn.close()
    assert "search_content" in post_cols
    assert _insert_trigger_column(db) == "search_content"
    assert _insert_and_search(db, "NEVER_TOKEN") is True


def test_fresh_db_opens_and_writes(tmp_path):
    """Baseline: a brand-new store (no legacy fixture) still works end to end."""
    db = str(tmp_path / "fresh.db")
    MessageStore(db)
    assert _insert_trigger_column(db) == "search_content"
    assert _insert_and_search(db, "FRESH_TOKEN") is True


def test_reopen_is_idempotent(tmp_path):
    """Opening an already-repaired DB a second time is a no-op and still writes."""
    db = str(tmp_path / "reopen.db")
    _build_half_migrated_db(db)
    MessageStore(db)
    # Second open must not re-break anything.
    MessageStore(db)
    assert _insert_trigger_column(db) == "search_content"
    assert _insert_and_search(db, "REOPEN_TOKEN") is True
