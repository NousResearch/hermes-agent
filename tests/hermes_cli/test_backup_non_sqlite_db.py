"""A readable ``*.db`` without the SQLite magic is archived verbatim, not a backup abort (#146016).

``_write_zip_entries`` routes by suffix; a fixture/download/truncated ``.db`` made
``_safe_copy_db`` raise ``file is not a database``, which aborted the whole archive.
A readable non-SQLite header now downgrades the entry to a plain file; an unreadable
header (permissions, live connection) keeps the fail-loud WAL-safe snapshot.
"""

import sqlite3
import zipfile
from argparse import Namespace
from contextlib import closing
from pathlib import Path

import pytest

from hermes_cli import backup, sqlite_safe_read

GARBAGE = b"test fixture, not a database\n" * 8


def _prepare_home(tmp_path, monkeypatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    junk_dir = home / "projects" / "environment"
    junk_dir.mkdir(parents=True)
    (junk_dir / "trunc.db").write_bytes(GARBAGE)
    return home


def _run(automatic: bool, archive: Path, home: Path):
    if automatic:
        return backup._write_full_zip_backup(archive, home)
    return backup.run_backup(Namespace(output=str(archive)))


@pytest.mark.parametrize("automatic", [False, True])
def test_non_sqlite_db_suffix_archives_verbatim(tmp_path, monkeypatch, automatic):
    home = _prepare_home(tmp_path, monkeypatch)
    with closing(sqlite3.connect(home / "state.db")) as db:
        db.execute("CREATE TABLE messages (body TEXT)")
        db.execute("INSERT INTO messages VALUES ('row')")
        db.commit()
    archive = tmp_path / "archive" / "backup.zip"
    archive.parent.mkdir()

    result = _run(automatic, archive, home)

    assert result is True or result == archive, (
        "one junk *.db must not abort the backup"
    )
    with zipfile.ZipFile(archive) as zipped:
        names = zipped.namelist()
        junk_member = next(name for name in names if name.endswith("trunc.db"))
        assert zipped.read(junk_member) == GARBAGE, (
            "the entry is archived byte-for-byte"
        )
        restored = tmp_path / "restored.db"
        restored.write_bytes(
            zipped.read(next(name for name in names if name.endswith("state.db")))
        )
    with closing(sqlite3.connect(restored)) as db:
        assert db.execute("SELECT body FROM messages").fetchall() == [("row",)]


@pytest.mark.parametrize("automatic", [False, True])
def test_unreadable_header_keeps_fail_loud_snapshot(tmp_path, monkeypatch, automatic):
    home = _prepare_home(tmp_path, monkeypatch)
    monkeypatch.setattr(
        sqlite_safe_read, "read_header_bytes_preopen", lambda *a, **k: None
    )
    archive = tmp_path / "archive" / "backup.zip"
    archive.parent.mkdir()

    result = _run(automatic, archive, home)

    if automatic:
        assert result is None, "an unreadable header keeps the aborting snapshot path"
    else:
        assert result is False, "the junk *.db must be reported as a failed entry"
        with zipfile.ZipFile(archive) as zipped:
            assert not any(name.endswith("trunc.db") for name in zipped.namelist())
