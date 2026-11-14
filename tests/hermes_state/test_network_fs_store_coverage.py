"""Every Hermes SQLite store shares the network-filesystem journal policy (one helper, not per-store copies).

The field report saw the cron scheduler parked on ``cron/executions.db-shm`` too: converting state.db
alone left ``cron/executions.db``, ``cron/deliveries.db``, ``response_store.db`` and ``projects.db`` in
WAL. Each store opens through ``apply_wal_with_fallback`` (directly or via ``hermes_cli.sqlite_util.open_db``),
so a fresh one on a network mount starts in DELETE like state.db does.
"""

import sqlite3

import pytest

import hermes_state_wal


@pytest.fixture()
def network_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / "cron").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(hermes_state_wal, "is_sqlite_wal_reset_vulnerable", lambda *a, **k: False)
    monkeypatch.setattr(hermes_state_wal, "_path_on_cross_vm_fs", lambda p: False)
    monkeypatch.setattr(hermes_state_wal, "_detect_network_fs", lambda directory, *a, **k: True, raising=False)
    getattr(hermes_state_wal, "_network_fs_cache", {}).clear()
    getattr(hermes_state_wal, "_network_fs_new_store_warned_paths", set()).clear()
    return home


def _header_mode(path):
    with sqlite3.connect(str(path)) as conn:
        return str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()


def test_session_db_on_network_fs_starts_in_delete(network_home):
    from hermes_state import SessionDB
    with SessionDB(network_home / "state.db") as db:
        db.create_session("s", source="test")
        assert db._wal_active is False
    assert _header_mode(network_home / "state.db") == "delete"


@pytest.mark.parametrize("store", ["cron/executions.db", "cron/deliveries.db", "projects.db"])
def test_open_db_stores_on_network_fs_start_in_delete(network_home, store):
    from hermes_cli.sqlite_util import open_db
    conn = open_db(network_home / store, db_label=store)
    conn.close()
    assert _header_mode(network_home / store) == "delete"


def test_response_store_on_network_fs_starts_in_delete(network_home):
    from gateway.platforms.api_server_response_store import ResponseStore
    path = network_home / "response_store.db"
    store = ResponseStore(db_path=str(path))
    store.close()
    assert _header_mode(path) == "delete"
