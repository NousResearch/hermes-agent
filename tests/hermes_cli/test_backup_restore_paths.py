"""Backup readers must address the literal filesystem path, including URI characters."""

import json
import sqlite3
import zipfile
from argparse import Namespace

import pytest


def _database(path, count):
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE sessions(id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE messages(id INTEGER PRIMARY KEY)")
        for number in range(count):
            conn.execute("INSERT INTO sessions VALUES (?)", (number,))
            conn.execute("INSERT INTO messages VALUES (?)", (number,))


@pytest.mark.parametrize("home_name", ["home#saved", "home%23saved", "home space"])
@pytest.mark.parametrize("entry", ["snapshot", "import"])
def test_restore_uses_literal_paths_and_preserves_live_connection(
    tmp_path, monkeypatch, capsys, home_name, entry
):
    home = tmp_path / home_name
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli import backup
    from hermes_cli.sqlite_safe_read import connect_tracked

    # Keep service discovery inside the disposable home, with an existing default
    # installation so import does not install/start a gateway for this test profile.
    native = tmp_path / "native"
    native.mkdir()
    (native / "config.yaml").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(backup, "_get_platform_default_hermes_home", lambda: native)

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
        if entry == "snapshot":
            assert backup.restore_quick_snapshot("saved", hermes_home=home)
        else:
            archive = tmp_path / "backup.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.write(source, "state.db")
            assert (
                backup.run_import(Namespace(zipfile=str(archive), force=True)) is None
            )
            assert "3 session(s) / 3 message(s) -> 1 / 1" in capsys.readouterr().out
        assert holder.execute("SELECT id FROM messages").fetchall() == [(0,)]
        with sqlite3.connect(target) as reopened:
            assert reopened.execute("SELECT id FROM messages").fetchall() == [(0,)]
    finally:
        holder.close()
    assert not (tmp_path / "home").exists()
    if home_name == "home%23saved":
        assert not (tmp_path / "home#saved").exists()


@pytest.mark.parametrize(
    "name", ["state#saved.db", "state%23saved.db", "state space.db"]
)
def test_row_counts_read_the_literal_database_without_creating_files(tmp_path, name):
    from hermes_cli.backup_restore import _count_session_rows

    source = tmp_path / name
    _database(source, 2)
    before = set(tmp_path.iterdir())
    assert _count_session_rows(source) == (2, 2)
    assert set(tmp_path.iterdir()) == before
