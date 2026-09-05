"""Transitional import of the frozen Claude SDK session-id column.

These tests use only task-local SQLite files.  The legacy column is added after
the current host schema is created to model a database written by the frozen
candidate without making that provider-specific column part of ``SCHEMA_SQL``.
"""

from __future__ import annotations

from agent.runtime_api import RuntimeStateEnvelope
from hermes_state import SessionDB


_CLAUDE_RUNTIME_ID = "hermes-claude-agent-sdk"


def _legacy_db(tmp_path, name: str = "state.db") -> SessionDB:
    db = SessionDB(db_path=tmp_path / name)
    with db._lock:
        db._conn.execute(
            "ALTER TABLE sessions ADD COLUMN claude_sdk_session_id TEXT"
        )
        db._conn.commit()
    return db


def _set_legacy_id(db: SessionDB, session_id: str, value: str) -> None:
    with db._lock:
        db._conn.execute(
            "UPDATE sessions SET claude_sdk_session_id = ? WHERE id = ?",
            (value, session_id),
        )
        db._conn.commit()


def _legacy_id(db: SessionDB, session_id: str) -> str | None:
    row = db._conn.execute(
        "SELECT claude_sdk_session_id FROM sessions WHERE id = ?",
        (session_id,),
    ).fetchone()
    return row[0] if row is not None else None


def test_legacy_id_imports_to_one_generic_envelope_and_stays_unchanged(tmp_path):
    db = _legacy_db(tmp_path)
    try:
        db.create_session("legacy-session", source="cli")
        _set_legacy_id(db, "legacy-session", "synthetic-claude-session-1")

        state = db.get_runtime_state("legacy-session", _CLAUDE_RUNTIME_ID)

        assert state == RuntimeStateEnvelope(
            runtime_id=_CLAUDE_RUNTIME_ID,
            schema_version=1,
            state={"external_session_id": "synthetic-claude-session-1"},
        )
        assert _legacy_id(db, "legacy-session") == "synthetic-claude-session-1"
        assert db.get_runtime_state("legacy-session", _CLAUDE_RUNTIME_ID) == state
        assert db._conn.execute(
            "SELECT COUNT(*) FROM runtime_session_state"
        ).fetchone()[0] == 1
    finally:
        db.close()


def test_existing_generic_state_wins_without_touching_legacy_id(tmp_path):
    db = _legacy_db(tmp_path)
    try:
        db.create_session("existing-state", source="cli")
        _set_legacy_id(db, "existing-state", "synthetic-legacy-session")
        generic = RuntimeStateEnvelope(
            runtime_id=_CLAUDE_RUNTIME_ID,
            schema_version=1,
            state={"external_session_id": "synthetic-generic-session"},
        )
        db.update_runtime_state("existing-state", generic)

        assert db.get_runtime_state("existing-state", _CLAUDE_RUNTIME_ID) == generic
        assert _legacy_id(db, "existing-state") == "synthetic-legacy-session"
    finally:
        db.close()


def test_current_schema_without_legacy_column_remains_compatible(tmp_path):
    db = SessionDB(db_path=tmp_path / "current.db")
    try:
        db.create_session("current-session", source="cli")
        columns = {
            row[1]
            for row in db._conn.execute("PRAGMA table_info(sessions)").fetchall()
        }
        assert "claude_sdk_session_id" not in columns
        assert db.get_runtime_state("current-session", _CLAUDE_RUNTIME_ID) is None
        assert db._conn.execute(
            "SELECT COUNT(*) FROM runtime_session_state"
        ).fetchone()[0] == 0
    finally:
        db.close()


def test_non_claude_runtime_leaves_legacy_state_inert_without_plugin(tmp_path):
    db = _legacy_db(tmp_path)
    try:
        db.create_session("legacy-session", source="cli")
        _set_legacy_id(db, "legacy-session", "synthetic-legacy-session")

        assert db.get_runtime_state("legacy-session", "hermes-codex") is None
        assert db._conn.execute(
            "SELECT COUNT(*) FROM runtime_session_state"
        ).fetchone()[0] == 0
        assert _legacy_id(db, "legacy-session") == "synthetic-legacy-session"
    finally:
        db.close()


def test_read_only_open_does_not_import_or_mutate_legacy_state(tmp_path):
    path = tmp_path / "readonly.db"
    db = _legacy_db(tmp_path, path.name)
    db.create_session("legacy-session", source="cli")
    _set_legacy_id(db, "legacy-session", "synthetic-legacy-session")
    db.close()

    readonly = SessionDB(db_path=path, read_only=True)
    try:
        assert readonly.get_runtime_state("legacy-session", _CLAUDE_RUNTIME_ID) is None
        assert _legacy_id(readonly, "legacy-session") == "synthetic-legacy-session"
        assert readonly._conn.execute(
            "SELECT COUNT(*) FROM runtime_session_state"
        ).fetchone()[0] == 0
    finally:
        readonly.close()
