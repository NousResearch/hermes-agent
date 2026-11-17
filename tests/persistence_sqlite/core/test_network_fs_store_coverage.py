"""Every Hermes SQLite store shares the network-filesystem journal policy (one helper, not per-store copies).

Converting state.db alone left ``cron/executions.db``, ``cron/deliveries.db``, ``response_store.db`` and
``projects.db`` in WAL. Each store opens through ``sqlite_support.apply_wal_with_fallback`` (directly or via
``sqlite_support.open_db``), so a fresh one on a network mount starts in DELETE like state.db does.
"""

import sqlite3

import pytest

import sqlite_support
from persistence.sqlite._holder import StoreHolder
from persistence.sqlite.conversation._store.sessions import create_session


@pytest.fixture()
def network_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / "cron").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(sqlite_support, "is_sqlite_wal_reset_vulnerable", lambda *a, **k: False)
    monkeypatch.setattr(sqlite_support, "path_on_cross_vm_fs", lambda p: False)
    monkeypatch.setattr(sqlite_support, "_detect_network_fs", lambda directory, *a, **k: True, raising=False)
    getattr(sqlite_support, "_network_fs_cache", {}).clear()
    getattr(sqlite_support, "_network_fs_new_store_warned_paths", set()).clear()
    return home


def _header_mode(path):
    with sqlite3.connect(str(path)) as conn:
        return str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()


def test_session_store_on_network_fs_starts_in_delete(network_home):
    db = StoreHolder(network_home / "state.db")
    try:
        create_session(db, "s", "test")
        assert db._wal_active is False
    finally:
        db.close()
    assert _header_mode(network_home / "state.db") == "delete"


@pytest.mark.parametrize("store", ["cron/executions.db", "cron/deliveries.db", "projects.db"])
def test_open_db_stores_on_network_fs_start_in_delete(network_home, store):
    conn = sqlite_support.open_db(network_home / store, db_label=store)
    conn.close()
    assert _header_mode(network_home / store) == "delete"


def test_response_store_on_network_fs_starts_in_delete(network_home):
    from gateway.platforms.api_server_response_store import ResponseStore
    path = network_home / "response_store.db"
    store = ResponseStore(db_path=str(path))
    store.close()
    assert _header_mode(path) == "delete"
