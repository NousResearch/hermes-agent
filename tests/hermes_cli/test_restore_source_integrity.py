"""Restore must validate the snapshot before mutating an existing database."""

from argparse import Namespace
from contextlib import closing
from pathlib import Path
import sqlite3
import zipfile

import pytest

from hermes_cli import backup


@pytest.fixture(params=["snapshot", "import"])
def restore_case(tmp_path, monkeypatch, request):
    import hermes_cli.gateway as gateway

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(gateway, "ensure_gateway_service", lambda **kwargs: False)
    monkeypatch.setattr(gateway, "_is_service_running", lambda: False)
    live = home / "state.db"
    with closing(sqlite3.connect(live)) as db:
        db.execute("CREATE TABLE evidence(value TEXT)")
        db.execute("INSERT INTO evidence VALUES ('snapshot')")
        db.commit()
    snapshot_id = backup.create_quick_snapshot(hermes_home=home)
    assert snapshot_id
    source = home / "state-snapshots" / snapshot_id / "state.db"
    with closing(sqlite3.connect(live)) as db:
        db.execute("UPDATE evidence SET value='live'")
        db.commit()

    def restore():
        if request.param == "snapshot":
            return backup.restore_quick_snapshot(snapshot_id, hermes_home=home)
        archive = tmp_path / "backup.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.write(source, "state.db")
        return backup.run_import(Namespace(zipfile=str(archive), force=True)) != 1

    return source, live, restore


@pytest.mark.parametrize("damage", ["header", "btree", "truncated"])
def test_corrupt_source_cannot_replace_a_healthy_database(restore_case, damage):
    source, live, restore = restore_case
    with closing(sqlite3.connect(source)) as db:
        page_size = db.execute("PRAGMA page_size").fetchone()[0]
        root_page = db.execute(
            "SELECT rootpage FROM sqlite_master WHERE name='evidence'"
        ).fetchone()[0]
    contents = bytearray(source.read_bytes())
    if damage == "header":
        contents[:16] = b"not a database!!"
    elif damage == "btree":
        contents[(root_page - 1) * page_size] = 0  # Invalid b-tree page type.
    else:
        del contents[page_size:]
    source.write_bytes(contents)
    assert not backup.verify_sqlite_integrity(source)["valid"]
    before = live.read_bytes()
    inode = live.stat().st_ino

    success = restore()

    assert live.read_bytes() == before
    assert live.stat().st_ino == inode
    assert not success
    with closing(sqlite3.connect(live)) as db:
        assert db.execute("SELECT value FROM evidence").fetchall() == [("live",)]


@pytest.mark.parametrize("destination", ["healthy", "corrupt", "held"])
def test_valid_source_still_restores_existing_destinations(restore_case, destination):
    source, live, restore = restore_case
    if destination == "corrupt":
        contents = bytearray(live.read_bytes())
        contents[:16] = b"not a database!!"
        live.write_bytes(contents)
    held = sqlite3.connect(live) if destination == "held" else None
    try:
        if held is not None:
            assert held.execute("SELECT value FROM evidence").fetchall() == [("live",)]
        assert restore()
        with closing(sqlite3.connect(live)) as db:
            assert db.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
            assert db.execute("SELECT value FROM evidence").fetchall() == [
                ("snapshot",)
            ]
        if held is not None:
            assert held.execute("SELECT value FROM evidence").fetchall() == [
                ("snapshot",)
            ]
    finally:
        if held is not None:
            held.close()
