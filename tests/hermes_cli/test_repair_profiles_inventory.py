"""Read-only profile repair inventories missing-default and unsafe physical stores."""

import sqlite3

import pytest

from hermes_cli.sessions_repair_profiles import Store, scan_stores
from hermes_state import SessionDB


@pytest.mark.parametrize("named", [False, True])
def test_full_read_only_scan_keeps_an_absent_default_database_absent(tmp_path, monkeypatch, named):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    stores = [Store("default", home, routing=True)]
    if named:
        profile = home / "profiles" / "science"
        profile.mkdir(parents=True)
        SessionDB(profile / "state.db").close()
        stores.append(Store("science", profile))
    plan, session = scan_stores(stores, read_only=True)
    try:
        assert not plan.findings
        assert not (home / "state.db").exists()
    finally:
        session.close()


def test_existing_unreadable_database_does_not_produce_an_incomplete_repair_plan(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    profile = home / "profiles" / "science"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    (profile / "state.db").write_bytes(b"not a sqlite database")
    with pytest.raises(sqlite3.DatabaseError):
        scan_stores([Store("default", home, routing=True)], read_only=True)
    assert not (home / "state.db").exists()
