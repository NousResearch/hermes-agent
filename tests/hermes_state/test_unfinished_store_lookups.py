"""#153048: a state.db left 0-byte or table-less by an interrupted init reads as "no store".

SQLite opens both and then fails every query with "no such table". Lookups that already treat a
missing state.db as "nothing here" must treat these the same way, while a non-empty damaged file
still raises its real error (a corrupt store must stay loud)."""
from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest


def _zero_byte(path):
    path.write_bytes(b"")


def _table_less(path):
    conn = sqlite3.connect(path)  # what an init killed between connect and schema commit leaves
    conn.execute("PRAGMA journal_mode=wal")
    conn.close()


def _lookups(home):
    db = home / "state.db"
    from hermes_cli.approvals_suggest import scan_approval_history
    from hermes_cli.gateway_session_store import GatewaySessionStore
    from hermes_cli.sessions_repair_profiles import _Session
    from tools import bot_live_delivery

    store = GatewaySessionStore.__new__(GatewaySessionStore)
    store._reader = None
    return {
        "bot chat tip": lambda: bot_live_delivery._bot_chat_tip(home),
        "approvals history": lambda: scan_approval_history(db),
        "repair-profiles store": lambda: _Session(read_only=True).db(
            SimpleNamespace(db_path=db)).find_crossed_profile_sessions("default"),
        "CLI session reader": lambda: store.get_session("missing"),
    }


@pytest.mark.parametrize("make", [_zero_byte, _table_less], ids=["zero-byte", "table-less"])
def test_unfinished_state_db_reads_as_no_store(tmp_path, monkeypatch, make):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    make(tmp_path / "state.db")
    results = {name: call() for name, call in _lookups(tmp_path).items()}
    assert results == {
        "bot chat tip": None,
        "approvals history": [],
        "repair-profiles store": {"mislabelled": [], "foreign": [], "crossed_parents": []},
        "CLI session reader": None,
    }


def test_only_missing_or_unfinished_files_read_as_no_store(tmp_path):
    from hermes_state_dbfile import state_db_holds_store

    assert state_db_holds_store(tmp_path / "absent.db") is False
    (tmp_path / "dir.db").mkdir()  # a bad path must reach the real open and report itself
    assert state_db_holds_store(tmp_path / "dir.db") is True
    conn = sqlite3.connect(tmp_path / "real.db")
    conn.execute("CREATE TABLE sessions (id TEXT)")
    conn.close()
    assert state_db_holds_store(tmp_path / "real.db") is True


def test_damaged_non_empty_state_db_stays_loud(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "state.db").write_bytes(b"not a database " * 400)
    for name, call in _lookups(tmp_path).items():
        if name == "CLI session reader":
            continue  # main's reader logs and retries every open failure; unchanged here
        with pytest.raises(sqlite3.DatabaseError):
            call()
