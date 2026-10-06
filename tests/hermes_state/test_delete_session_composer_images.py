"""Tests for the integration between SessionDB.delete_session / delete_sessions
and the desktop composer-images cleanup.

Behavior contracts asserted (never snapshots):

- After ``delete_session``, a composer-image referenced *only* by the deleted
  session is unlinked; one still referenced by a surviving branch or sibling
  session stays on disk.
- Delegate-child cascades carry their refs into the deleted set, so an image
  attached only inside a delegate subagent is removed with the parent.
- ``delete_sessions`` batch behaves the same per row as the single-row call.
- File-system errors during composer cleanup must never abort the DB delete
  (deletion already committed; file removal is best-effort outside the TX).
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest

import hermes_state
from hermes_state import SessionDB


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    parent_session_id TEXT,
    model_config TEXT,
    title TEXT,
    ended_at REAL,
    archived INTEGER NOT NULL DEFAULT 0,
    hidden INTEGER NOT NULL DEFAULT 0,
    pinned INTEGER NOT NULL DEFAULT 0,
    last_read_at REAL,
    tool_names TEXT,
    system_prompt_hash TEXT
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL,
    content TEXT,
    tool_call_id TEXT,
    tool_calls TEXT,
    tool_name TEXT,
    effect_disposition TEXT,
    timestamp REAL NOT NULL,
    token_count INTEGER,
    finish_reason TEXT,
    reasoning TEXT,
    reasoning_content TEXT,
    reasoning_details TEXT,
    codex_reasoning_items TEXT,
    codex_message_items TEXT,
    platform_message_id TEXT,
    observed INTEGER DEFAULT 0,
    _compressed_summary INTEGER NOT NULL DEFAULT 0,
    active INTEGER NOT NULL DEFAULT 1,
    compacted INTEGER NOT NULL DEFAULT 0,
    api_content TEXT,
    display_kind TEXT,
    display_metadata TEXT,
    display_identity BLOB,
    display_order INTEGER,
    message_uid TEXT,
    absorbed_message_uids TEXT,
    tool_call_uids TEXT,
    tool_call_uid TEXT
);
CREATE TABLE IF NOT EXISTS system_prompts (hash TEXT PRIMARY KEY, content TEXT);
"""


def _init_db(db_path: Path, userdata_dir: Path) -> SessionDB:
    """Create a SessionDB at *db_path* seeded with the minimum schema used by
    delete_session (sessions, messages, system_prompts)."""
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA_SQL)
    conn.commit()
    conn.close()
    # Swap the shared DEFAULT_DB_PATH so SessionDB() uses our test file.
    import hermes_state_sessions  # noqa: F401  import side-effect free but loads the module
    db = SessionDB(db_path=db_path, read_only=False, journal_mode="OFF")
    return db


def _seed_message(db: SessionDB, sid: str, text: str) -> None:
    """Insert a single user message into *sid* through the live writer so the
    delete_session code path reads consistent rows."""

    def _w(conn):
        conn.execute("INSERT OR IGNORE INTO sessions (id) VALUES (?)", (sid,))
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?,?,?,?)",
            (sid, "user", text, time.time()),
        )

    db._execute_write(_w)


def _composer_path(userdata: Path, name: str) -> Path:
    p = userdata / "Hermes" / "composer-images" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x89PNG\r\n\x1a\n")
    return p


@pytest.fixture()
def setup(tmp_path, monkeypatch):
    """Prepare a temp composer dir (via env override) + empty SessionDB."""
    userdata = tmp_path / "userdata"
    home = tmp_path / "home"
    home.mkdir()
    userdata.mkdir()
    monkeypatch.setenv("HERMES_DESKTOP_USER_DATA_DIR", str(userdata / "Hermes"))
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Force reimport of desktop_composer_images so env vars are honored.
    from hermes_cli import desktop_composer_images
    import importlib
    importlib.reload(desktop_composer_images)
    db_path = home / "state.db"
    db = _init_db(db_path, userdata / "Hermes")
    return userdata, home, db


class TestDeleteSessionCleansComposerImages:
    def test_deletes_image_only_in_deleted_session(self, setup):
        userdata, _home, db = setup
        only = _composer_path(userdata, "only.png")
        _seed_message(db, "S", f"caption\n@image:{only}")
        assert db.delete_session("S") is True
        assert not only.exists()

    def test_keeps_shared_image_referenced_by_another_session(self, setup):
        userdata, _home, db = setup
        shared = _composer_path(userdata, "shared.png")
        _seed_message(db, "A", f"@image:{shared}")
        _seed_message(db, "B", f"also\n@image:{shared}")
        db.delete_session("A")
        # B still references it → stays.
        assert shared.exists()
        db.delete_session("B")
        # Now nobody references it → gone (via B's cleanup call).
        assert not shared.exists()

    def test_delegate_child_attachment_removed_with_parent(self, setup):
        import json
        userdata, _home, db = setup
        child_img = _composer_path(userdata, "inside_subagent.png")

        def _w(conn):
            conn.execute("INSERT INTO sessions (id) VALUES ('P')")
            conn.execute(
                "INSERT INTO sessions (id, parent_session_id, model_config) VALUES (?,?,?)",
                ("C", "P", json.dumps({"_delegate_from": "P"})),
            )
            conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp) VALUES ('C','user',?,?)",
                (f"@image:{child_img}", time.time()),
            )

        db._execute_write(_w)
        assert db.delete_session("P") is True
        assert not child_img.exists()

    def test_branch_child_keeps_ref_after_parent_orphaned(self, setup):
        userdata, _home, db = setup
        keep = _composer_path(userdata, "branch.png")

        def _w(conn):
            conn.execute("INSERT INTO sessions (id) VALUES ('P')")
            conn.execute(
                "INSERT INTO sessions (id, parent_session_id) VALUES (?,?)",
                ("B", "P"),
            )
            conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp) VALUES ('B','user',?,?)",
                (f"@image:{keep}", time.time()),
            )

        db._execute_write(_w)
        # Parent P deleted; branch B is orphaned (not deleted) and must keep its ref.
        assert db.delete_session("P") is True
        assert keep.exists()

    def test_delete_sessions_batch_matches_semantics(self, setup):
        userdata, _home, db = setup
        a = _composer_path(userdata, "a.png")
        b = _composer_path(userdata, "b.png")
        shared = _composer_path(userdata, "shared.png")
        keep = _composer_path(userdata, "keep.png")
        _seed_message(db, "S1", f"@image:{a}\n@image:{shared}")
        _seed_message(db, "S2", f"@image:{b}\n@image:{shared}")
        _seed_message(db, "S3", f"@image:{keep}")
        count = db.delete_sessions(["S1", "S2"])
        assert count == 2
        assert not a.exists() and not b.exists()
        # shared is referenced by nobody now → deleted.
        assert not shared.exists()
        # keep belongs to the surviving S3 → still present.
        assert keep.exists()

    def test_file_errors_during_cleanup_never_abort_db_delete(self, setup, caplog):
        """Composer cleanup runs outside the write TX; a failure there (permissions,
        file already gone, etc.) must not make the delete_session raise. The
        cleanup helper wraps in ``suppress(Exception)`` — this regression test
        keeps it that way by swapping _safe_unlink_many to a raising stub."""
        userdata, _home, db = setup
        img = _composer_path(userdata, "stubborn.png")
        _seed_message(db, "S", f"@image:{img}")
        from hermes_cli import desktop_composer_images as dci
        called: list[bool] = []

        def explode(paths):
            called.append(True)
            raise OSError("simulated disk full")

        import hermes_state_sessions as s
        # Monkey-patch the imported cleanup function via the module it's called in.
        import hermes_cli.desktop_composer_images as mod
        original = mod._safe_unlink_many
        mod._safe_unlink_many = explode
        try:
            # Must not raise even though the cleanup path raises.
            assert db.delete_session("S") is True
        finally:
            mod._safe_unlink_many = original
        # DB delete still committed → session row is gone.
        assert db._read_one("SELECT COUNT(*) FROM sessions WHERE id = 'S'")[0] == 0
