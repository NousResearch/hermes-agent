"""The rename and delete warnings name the error the store raised underneath: the SQLite error for a store failure,
``ValueError`` for an identity collision, the classes ``hermes profile rename`` and ``delete`` print."""
import json
import sqlite3
from pathlib import Path

import pytest

from hermes_cli.profile_identity import _migrate_profile_identity, _purge_profile_identity
from persistence.sqlite._backend import translated
from persistence.sqlite.housekeeping.profile_identity import SQLiteProfileIdentityRepository
from persistence_assembly import open_persistence, profile_scope


@pytest.fixture()
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    with open_persistence(profile_scope(home=home), eager=True):
        pass
    return home


def _disk_io_error(*_args):
    with translated():
        raise sqlite3.OperationalError("disk I/O error")


def test_rename_warning_names_the_sqlite_error(root, monkeypatch, capsys):
    monkeypatch.setattr(SQLiteProfileIdentityRepository, "rekey_profile", _disk_io_error)

    assert _migrate_profile_identity("old", "new", False) is False
    assert (f"identity migration failed for {root / 'state.db'}: OperationalError: disk I/O error"
            in capsys.readouterr().err)


def test_delete_warning_names_the_sqlite_error(root, monkeypatch, capsys):
    monkeypatch.setattr(SQLiteProfileIdentityRepository, "purge_profile", _disk_io_error)

    assert _purge_profile_identity("gone", False) is False
    assert (f"identity purge failed for {root / 'state.db'}: OperationalError: disk I/O error"
            in capsys.readouterr().err)


def test_rename_collision_warning_names_value_error(root, capsys):
    scope = str(root / "sessions")
    with open_persistence(profile_scope(home=root)) as store:
        for key in ("agent:old:feishu:dm:chatA", "agent:new:feishu:dm:chatA"):
            store.routing.upsert_routing_entry(key, json.dumps({"session_key": key}), routing_scope=scope)

    assert _migrate_profile_identity("old", "new", False) is False
    assert (f"identity migration failed for {root / 'state.db'}: ValueError: profile routing collision in scope"
            in capsys.readouterr().err)
