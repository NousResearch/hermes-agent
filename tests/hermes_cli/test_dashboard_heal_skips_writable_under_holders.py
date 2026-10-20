"""Dashboard browsing never repairs a store held by a live gateway."""

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from hermes_cli.web_server_sessions import _open_session_db_at_path
from hermes_state_common import DEFERRED_INDEX_SQL, SCHEMA_SQL


def _store(path: Path, *, stale: bool) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.executescript(DEFERRED_INDEX_SQL)
        if stale:
            conn.execute("DROP INDEX IF EXISTS idx_sessions_effective_activity")
            conn.execute("ALTER TABLE sessions DROP COLUMN last_activity_at")
        conn.commit()
    finally:
        conn.close()


@pytest.mark.parametrize("stale", [False, True], ids=["current", "stale"])
def test_dashboard_heal_skips_writable_open_when_gateway_holds_store(tmp_path, stale):
    db_path = tmp_path / "state.db"
    _store(db_path, stale=stale)
    before = db_path.read_bytes()
    with patch("hermes_state_registry.acquire") as acquire, patch(
        "hermes_state_holders.foreign_state_db_holders",
        return_value=[(99, "hermes gateway run --replace")],
    ), patch(
        "hermes_state_holders._read_proc_argv",
        return_value=["/usr/bin/hermes", "gateway", "run", "--replace"],
    ):
        if stale:
            with pytest.raises(HTTPException, match="running gateway owns it") as raised:
                _open_session_db_at_path(db_path, read_only=True)
            assert raised.value.status_code == 503
            assert "Restart that gateway" in raised.value.detail
        else:
            db = _open_session_db_at_path(db_path, read_only=True)
            try:
                assert db.read_only is True
            finally:
                db.close()
        acquire.assert_not_called()
    assert db_path.read_bytes() == before
    if stale:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
        assert "last_activity_at" not in columns
