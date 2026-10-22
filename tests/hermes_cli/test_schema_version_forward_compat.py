"""Current-upstream state stores remain valid on this build."""
from __future__ import annotations

import sqlite3

from hermes_cli.session_recovery import _verify_structure
from hermes_cli.web_server_sessions import _open_session_db_at_path
from hermes_state import SessionDB


UPSTREAM_SCHEMA_VERSION = 32


def test_current_upstream_store_browses_and_verifies_for_recovery(tmp_path):
    path = tmp_path / "state.db"
    with SessionDB(db_path=path) as db:
        db.create_session("kept", source="cli")

    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE schema_version SET version = ?", (UPSTREAM_SCHEMA_VERSION,))

    db = _open_session_db_at_path(path, read_only=True)
    try:
        assert [row["id"] for row in db.list_sessions_rich(limit=10, compact_rows=True)] == ["kept"]
    finally:
        db.close()

    verification = {"errors": []}
    with sqlite3.connect(path) as conn:
        _verify_structure(conn, verification)
    assert verification["schema_version"] == UPSTREAM_SCHEMA_VERSION
    assert not [error for error in verification["errors"] if error.startswith("schema version is")]
