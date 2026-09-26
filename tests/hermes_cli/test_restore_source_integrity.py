"""Restore must read the literal source path and validate it before mutating a live database."""

import json
import sqlite3
import zipfile
from argparse import Namespace
from contextlib import closing

import pytest

from hermes_cli import backup

# Unescaped in a file: URI, '#' truncates the path and '%23' decodes to a different one.
HOME_NAMES = ["x#y", "x%23y", "x y"]


def _database(path, count):
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE sessions(id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE messages(id INTEGER PRIMARY KEY)")
        conn.executemany("INSERT INTO sessions VALUES (?)", [(n,) for n in range(count)])
        conn.executemany("INSERT INTO messages VALUES (?)", [(n,) for n in range(count)])
        conn.commit()


def _assert_no_decoy(tmp_path, home_name):
    assert {p.name for p in tmp_path.iterdir()} <= {home_name, "native", "backup.zip", "hermes_test"}


@pytest.fixture(params=HOME_NAMES)
def home(tmp_path, monkeypatch, request):
    import hermes_cli.gateway as gateway

    home = tmp_path / request.param
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    # An existing default install keeps import from installing/starting a gateway.
    native = tmp_path / "native"
    native.mkdir()
    (native / "config.yaml").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(backup, "_get_platform_default_hermes_home", lambda: native)
    monkeypatch.setattr(gateway, "ensure_gateway_service", lambda **kwargs: False)
    monkeypatch.setattr(gateway, "_is_service_running", lambda: False)
    return home


def _restore(entry, home, snapshot_id, source):
    if entry == "snapshot":
        return backup.restore_quick_snapshot(snapshot_id, hermes_home=home)
    archive = home.parent / "backup.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(source, "state.db")
    return backup.run_import(Namespace(zipfile=str(archive), force=True)) != 1


@pytest.mark.parametrize("entry", ["snapshot", "import"])
def test_restore_uses_literal_paths_and_preserves_live_connection(
    tmp_path, capsys, home, entry
):
    from hermes_cli.sqlite_safe_read import connect_tracked

    target = home / "state.db"
    _database(target, 3)
    snapshot = home / "state-snapshots" / "saved"
    snapshot.mkdir(parents=True)
    source = snapshot / "state.db"
    _database(source, 1)
    (snapshot / "manifest.json").write_text(
        json.dumps({"files": {"state.db": source.stat().st_size}}), encoding="utf-8"
    )
    holder = connect_tracked(target)
    try:
        assert _restore(entry, home, "saved", source)
        if entry == "import":
            assert "3 session(s) / 3 message(s) -> 1 / 1" in capsys.readouterr().out
        assert holder.execute("SELECT id FROM messages").fetchall() == [(0,)]
        with closing(sqlite3.connect(target)) as reopened:
            assert reopened.execute("SELECT id FROM messages").fetchall() == [(0,)]
    finally:
        holder.close()
    _assert_no_decoy(tmp_path, home.name)


@pytest.mark.parametrize("entry", ["snapshot", "import"])
@pytest.mark.parametrize("damage", ["header", "btree", "truncated"])
def test_corrupt_source_cannot_replace_a_healthy_database(
    tmp_path, capsys, home, entry, damage
):
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

    assert not _restore(entry, home, snapshot_id, source)
    if entry == "import":
        captured = capsys.readouterr()
        assert "failed its integrity check" in captured.out + captured.err

    assert live.read_bytes() == before
    assert live.stat().st_ino == inode
    with closing(sqlite3.connect(live)) as db:
        assert db.execute("SELECT value FROM evidence").fetchall() == [("live",)]
    _assert_no_decoy(tmp_path, home.name)
