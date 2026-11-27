"""Stores that open through ``sqlite_support.open_db`` share the network-filesystem journal policy.

A fresh store on a network mount starts in DELETE like ``state.db`` does: the generic ``open_db`` path and the
hosted-room ``shared-state.db`` write connection both reach ``apply_wal_with_fallback``.
"""

import sqlite3

import pytest

import sqlite_support


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


@pytest.mark.parametrize("store", ["cron/executions.db", "cron/deliveries.db", "projects.db"])
def test_open_db_stores_on_network_fs_start_in_delete(network_home, store):
    conn = sqlite_support.open_db(network_home / store, db_label=store)
    conn.close()
    assert _header_mode(network_home / store) == "delete"


def test_hosted_room_shared_state_on_network_fs_starts_in_delete(network_home):
    from persistence.sqlite.gateway._hosted_store import _connect_shared
    path = network_home / "shared-state.db"
    _connect_shared(path, ()).close()
    assert _header_mode(path) == "delete"
