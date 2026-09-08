"""Session-owned GLOBAL policy snapshots persist independently of prompt blobs."""

from __future__ import annotations

import sqlite3

from hermes_state import SessionDB
from hermes_state_common import SCHEMA_SQL


# This test builds a writable database with the immediately previous sessions
# definition, rather than using SessionDB to initialize a current store first.
_LEGACY_SESSIONS_SCHEMA_SQL = SCHEMA_SQL.replace(
    "    global_policy_snapshot TEXT,\n", ""
)


def test_writable_legacy_sessions_table_is_reconciled_with_null_snapshot(tmp_path):
    """Opening a pre-snapshot store adds the nullable column without rewriting rows."""
    path = tmp_path / "legacy-state.db"
    conn = sqlite3.connect(path)
    try:
        conn.executescript(_LEGACY_SESSIONS_SCHEMA_SQL)
        conn.execute(
            "INSERT INTO sessions (id, source, started_at) VALUES (?, ?, ?)",
            ("legacy-session", "cli", 1.0),
        )
        conn.commit()
    finally:
        conn.close()

    db = SessionDB(db_path=path)
    try:
        columns = {
            row["name"]
            for row in db._conn.execute("PRAGMA table_info(sessions)").fetchall()
        }
        assert "global_policy_snapshot" in columns
        assert db.get_session("legacy-session")["global_policy_snapshot"] is None
    finally:
        db.close()


def test_global_policy_snapshot_distinguishes_value_empty_and_null_across_reopen(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(db_path=path)
    try:
        db.create_session(
            "snapshot-session",
            "cli",
            system_prompt="first prompt",
            global_policy_snapshot="A",
        )
        assert db.get_session("snapshot-session")["system_prompt"] == "first prompt"
        assert db.get_session("snapshot-session")["global_policy_snapshot"] == "A"

        db.update_system_prompt(
            "snapshot-session", "second prompt", global_policy_snapshot=""
        )
        assert db.get_session("snapshot-session")["system_prompt"] == "second prompt"
        assert db.get_session("snapshot-session")["global_policy_snapshot"] == ""

        # Omission preserves the explicit empty snapshot rather than treating it
        # as an instruction to reread or clear the policy.
        db.update_system_prompt("snapshot-session", "third prompt")
        assert db.get_session("snapshot-session")["global_policy_snapshot"] == ""

        # NULL remains the legacy/uninitialized state and differs from "".
        db.update_system_prompt(
            "snapshot-session", "fourth prompt", global_policy_snapshot=None
        )
    finally:
        db.close()

    reopened = SessionDB(db_path=path)
    try:
        session = reopened.get_session("snapshot-session")
        assert session["system_prompt"] == "fourth prompt"
        assert session["global_policy_snapshot"] is None
    finally:
        reopened.close()
