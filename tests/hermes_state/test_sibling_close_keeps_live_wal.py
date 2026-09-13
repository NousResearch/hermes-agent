"""Closing a short-lived SessionDB must not unlink WAL under a live sibling writer.

Herder / TUI tab close is the field shape: one Hermes process exits while another
tab still holds ``state.db``. sqlite3's last-connection close then unlinks
``state.db-wal`` / ``-shm``, and the surviving session sticky-halts with
``DeletedWalGenerationError`` (#109727).
"""

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_state import SessionDB
from tests.hermes_state._wal_generation_harness import gateway_writer, make_db, pin_wal, require_wal


def test_healthy_writer_close_disables_close_time_wal_reset(tmp_path, monkeypatch):
    """A healthy close must arm NO_CKPT_ON_CLOSE before sqlite3_close (3.12+)."""
    flag = getattr(sqlite3, "SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE", None)
    if flag is None or not hasattr(sqlite3.Connection, "setconfig"):
        pytest.skip("Connection.setconfig / SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE unavailable")

    pin_wal(monkeypatch)
    db = make_db(tmp_path / "state.db", "s", "before-close")
    require_wal(db)
    setconfig = db._conn.setconfig
    with patch.object(db._conn, "setconfig", wraps=setconfig) as mock_setconfig:
        db.close()
        armed = [
            call for call in mock_setconfig.call_args_list
            if call[0] and call[0][0] == flag and call[0][1] is True
        ]
        assert armed, "healthy close did not disable SQLite's last-connection WAL reset"


def test_sibling_writer_close_does_not_poison_live_holder(tmp_path, monkeypatch):
    pin_wal(monkeypatch)
    with gateway_writer(tmp_path) as gw:
        gw.next_event("ready")
        wal = Path(str(gw.path) + "-wal")
        shm = Path(str(gw.path) + "-shm")
        assert wal.exists() and shm.exists()

        sibling = SessionDB(db_path=gw.path)
        sibling.create_session("tui-tab", "cli")
        sibling.append_message("tui-tab", role="user", content="closing this tab")
        sibling.close()

        assert wal.exists(), "sibling close unlinked state.db-wal under the live holder"
        assert shm.exists(), "sibling close unlinked state.db-shm under the live holder"

        gw.send("write")
        event = gw.next_event("write")
        assert event["refused"] is False, "live holder was poisoned by the sibling close"
