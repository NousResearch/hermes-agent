"""The manual and automatic archive paths use the same WAL-safe serialization."""
import sqlite3
import zipfile
from argparse import Namespace
from contextlib import closing
from pathlib import Path

import pytest

from hermes_cli import backup


@pytest.mark.parametrize("automatic", [False, True])
def test_zip_captures_live_wal_and_cleans_failed_staging(tmp_path, monkeypatch, automatic):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    output = tmp_path / "archive"
    output.mkdir()
    archive = output / "backup.zip"

    def run():
        if automatic:
            return backup._write_full_zip_backup(archive, home)
        return backup.run_backup(Namespace(output=str(archive)))

    with closing(sqlite3.connect(home / "state.db")) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("CREATE TABLE messages (body TEXT)")
        writer.commit()
        writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        writer.execute("INSERT INTO messages VALUES ('WAL-only row')")
        writer.commit()
        run()
        with zipfile.ZipFile(archive) as zipped:
            assert not any(name.endswith(('-wal', '-shm')) for name in zipped.namelist())
            member = next(name for name in zipped.namelist() if name.endswith('state.db'))
            restored = tmp_path / "restored.db"
            restored.write_bytes(zipped.read(member))
        with closing(sqlite3.connect(restored)) as snapshot:
            assert snapshot.execute("SELECT body FROM messages").fetchall() == [("WAL-only row",)]

        def refuse_write(self, filename, arcname=None, **kwargs):
            raise OSError("archive device full")

        monkeypatch.setattr(zipfile.ZipFile, "write", refuse_write)
        run()
        assert sorted(output.iterdir()) == [archive], "a failed write must not leak a private database snapshot"
