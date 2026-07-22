"""Tests for issue #42467: profile DBs with lagged schemas must not crash
list_sessions_rich / session_count when opened read-only.

The root cause: ``get_profiles_sessions`` opens each profile's ``state.db``
with ``read_only=True``, which skips schema reconciliation.  If a profile's
DB was created by an older version (no ``archived`` column yet), the
``list_sessions_rich`` query references ``s.archived`` and raises
``OperationalError: no such column: s.archived``.  The web-server endpoint
catches this silently, making the affected profile show zero sessions.

The fix: ``list_sessions_rich`` and ``session_count`` now probe for the
``archived`` column before referencing it, and skip the filter when it is
absent (safe because older DBs never contain archived sessions).
"""

import sqlite3
from pathlib import Path

import pytest

from hermes_state import SessionDB


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_LEGACY_SCHEMA = """\
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    source TEXT,
    model TEXT,
    started_at REAL NOT NULL,
    ended_at REAL,
    message_count INTEGER DEFAULT 0,
    end_reason TEXT,
    parent_session_id TEXT,
    system_prompt TEXT,
    model_config TEXT,
    cwd TEXT,
    title TEXT,
    FOREIGN KEY (parent_session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL,
    content TEXT,
    timestamp REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS state_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


def _create_legacy_db(path: Path) -> None:
    """Create a DB with an older schema that lacks the ``archived`` column."""
    conn = sqlite3.connect(str(path))
    conn.executescript(_LEGACY_SCHEMA)
    # Insert a session so list_sessions_rich has something to return.
    conn.execute(
        "INSERT INTO sessions (id, source, started_at, message_count) "
        "VALUES (?, ?, ?, ?)",
        ("sess_legacy_1", "cli", 1700000000.0, 3),
    )
    conn.execute(
        "INSERT INTO messages (session_id, role, content, timestamp) "
        "VALUES (?, ?, ?, ?)",
        ("sess_legacy_1", "user", "Hello from legacy DB", 1700000000.0),
    )
    conn.commit()
    conn.close()


def _create_current_db(path: Path) -> None:
    """Create a DB with the current schema (has ``archived`` column)."""
    db = SessionDB(db_path=path)
    db.create_session(session_id="sess_current_1", source="cli")
    db.append_message(
        session_id="sess_current_1",
        role="user",
        content="Hello from current DB",
    )
    db.close()


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestIssue42467_LegacySchemaReadonly:
    """list_sessions_rich / session_count must not crash when the DB lacks
    the ``archived`` column (read-only open, no reconciliation)."""

    def test_list_sessions_rich_tolerates_missing_archived_column(self, tmp_path):
        db_path = tmp_path / "legacy_state.db"
        _create_legacy_db(db_path)

        db = SessionDB(db_path=db_path, read_only=True)
        try:
            rows = db.list_sessions_rich(limit=50)
            assert len(rows) == 1
            assert rows[0]["id"] == "sess_legacy_1"
            # The ``archived`` key should either be absent or falsy (column
            # didn't exist so SQLite won't return it in the row).
            assert not rows[0].get("archived", False)
        finally:
            db.close()

    def test_list_sessions_rich_archived_only_graceful(self, tmp_path):
        """Requesting archived-only on a legacy DB returns empty, no crash."""
        db_path = tmp_path / "legacy_state.db"
        _create_legacy_db(db_path)

        db = SessionDB(db_path=db_path, read_only=True)
        try:
            rows = db.list_sessions_rich(limit=50, archived_only=True)
            # Column absent → filter skipped → all sessions returned.
            # This is correct: older DBs never have archived sessions.
            assert len(rows) == 1
        finally:
            db.close()

    def test_list_sessions_rich_include_archived_graceful(self, tmp_path):
        """include_archived=True on a legacy DB returns all rows, no crash."""
        db_path = tmp_path / "legacy_state.db"
        _create_legacy_db(db_path)

        db = SessionDB(db_path=db_path, read_only=True)
        try:
            rows = db.list_sessions_rich(limit=50, include_archived=True)
            assert len(rows) == 1
        finally:
            db.close()

    def test_session_count_tolerates_missing_archived_column(self, tmp_path):
        db_path = tmp_path / "legacy_state.db"
        _create_legacy_db(db_path)

        db = SessionDB(db_path=db_path, read_only=True)
        try:
            count = db.session_count()
            assert count == 1
        finally:
            db.close()

    def test_session_count_archived_only_graceful(self, tmp_path):
        db_path = tmp_path / "legacy_state.db"
        _create_legacy_db(db_path)

        db = SessionDB(db_path=db_path, read_only=True)
        try:
            count = db.session_count(archived_only=True)
            # Column absent → filter skipped → all rows counted.
            assert count == 1
        finally:
            db.close()

    def test_has_column_detects_present_column(self, tmp_path):
        db_path = tmp_path / "current_state.db"
        _create_current_db(db_path)

        db = SessionDB(db_path=db_path, read_only=True)
        try:
            assert db._has_column("sessions", "archived") is True
        finally:
            db.close()

    def test_has_column_detects_missing_column(self, tmp_path):
        db_path = tmp_path / "legacy_state.db"
        _create_legacy_db(db_path)

        db = SessionDB(db_path=db_path, read_only=True)
        try:
            assert db._has_column("sessions", "archived") is False
        finally:
            db.close()

    def test_has_column_returns_false_for_nonexistent_table(self, tmp_path):
        db_path = tmp_path / "legacy_state.db"
        _create_legacy_db(db_path)

        db = SessionDB(db_path=db_path, read_only=True)
        try:
            assert db._has_column("nonexistent_table", "archived") is False
        finally:
            db.close()

    def test_current_schema_read_only_still_filters_archived(self, tmp_path):
        """With a current-schema DB the archived filter is applied as before."""
        db_path = tmp_path / "current_state.db"
        _create_current_db(db_path)

        db = SessionDB(db_path=db_path, read_only=True)
        try:
            count = db.session_count()
            assert count == 1
        finally:
            db.close()

    def test_list_sessions_rich_order_by_last_active_legacy(self, tmp_path):
        """order_by_last_active path must also tolerate legacy schemas."""
        db_path = tmp_path / "legacy_state.db"
        _create_legacy_db(db_path)

        db = SessionDB(db_path=db_path, read_only=True)
        try:
            rows = db.list_sessions_rich(limit=50, order_by_last_active=True)
            assert len(rows) == 1
            assert rows[0]["id"] == "sess_legacy_1"
        finally:
            db.close()
