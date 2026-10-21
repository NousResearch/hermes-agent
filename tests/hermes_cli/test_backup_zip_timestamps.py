"""A file whose mtime is outside the ZIP date range must not break a backup.

ZIP stores DOS dates (1980-01-01 .. 2107-12-31). With the stdlib's strict default,
``ZipFile.write`` raises ``ValueError`` for a pre-1980 file (the file was skipped and the backup
reported incomplete) and ``struct.error`` for a file after 2107 (uncaught: the whole archive
aborted). Both archive writers now open the zip with ``strict_timestamps=False``: the entry date
is clamped, the contents are archived, and the source file's on-disk times stay untouched.
"""

import os
import time
import zipfile
from argparse import Namespace
from datetime import UTC, datetime

PRE_1980 = datetime(1970, 1, 2, tzinfo=UTC).timestamp()
PRE_EPOCH = -86400 * 400  # 1968
ORDINARY = datetime(2024, 5, 6, 7, 8, 10, tzinfo=UTC).timestamp()
FAR_FUTURE = datetime(2200, 1, 1, tzinfo=UTC).timestamp()


def _home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("model: {}\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    files = {
        "old-root.txt": PRE_1980,
        "logs/old/app.log.1": PRE_1980,
        "deep/a/b/pre-epoch.bin": PRE_EPOCH,
        "notes/ordinary.md": ORDINARY,
        "future/far.txt": FAR_FUTURE,
    }
    for rel, ts in files.items():
        p = home / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(f"content of {rel}\n".encode())
        os.utime(p, (ts, ts))
    return home, files


def _mtimes(home, files):
    return {rel: (home / rel).stat().st_mtime_ns for rel in files}


def _check_archive(zip_path, home, files):
    with zipfile.ZipFile(zip_path) as zf:
        for rel in files:
            assert zf.read(rel) == (home / rel).read_bytes(), rel
        assert zf.getinfo("old-root.txt").date_time == (1980, 1, 1, 0, 0, 0)
        assert zf.getinfo("deep/a/b/pre-epoch.bin").date_time == (1980, 1, 1, 0, 0, 0)
        assert zf.getinfo("future/far.txt").date_time[0] == 2107
        assert zf.getinfo("notes/ordinary.md").date_time == time.localtime(ORDINARY)[:6]
        assert zf.testzip() is None


def test_hermes_backup_archives_out_of_range_mtimes(tmp_path, monkeypatch, capsys):
    home, files = _home(tmp_path, monkeypatch)
    before = _mtimes(home, files)
    out = tmp_path / "out" / "state.zip"
    out.parent.mkdir()
    from hermes_cli import backup as backup_mod

    assert backup_mod.run_backup(Namespace(output=str(out), keep=0)) is True
    text = capsys.readouterr().out
    assert "could not be added" not in text and "Backup complete" in text
    _check_archive(out, home, files)
    assert _mtimes(home, files) == before


def test_automatic_full_zip_backup_archives_out_of_range_mtimes(tmp_path, monkeypatch):
    home, files = _home(tmp_path, monkeypatch)
    before = _mtimes(home, files)
    out = tmp_path / "auto.zip"
    from hermes_cli import backup as backup_mod

    assert backup_mod._write_full_zip_backup(out, home) == out
    _check_archive(out, home, files)
    assert _mtimes(home, files) == before
