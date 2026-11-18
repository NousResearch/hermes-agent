"""#147246: the pre-update full backup must report what it actually did.

``hermes update`` with ``updates.pre_update_backup: full`` used to print "Backup skipped (no files
found or write failed)" for an archive it had just published: an incomplete-but-importable salvage
(``*.incomplete.zip``) and a genuine failure looked identical from the caller.
"""
import zipfile
from pathlib import Path

import pytest

from hermes_cli import backup as backup_mod
from hermes_cli import update_cmd_maint


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("model: test\n", encoding="utf-8")
    (home / "notes.txt").write_text("hello\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return home


def _fail_one_entry(monkeypatch, rel: str) -> None:
    """Make the archive take every entry except ``rel``, which the walk found but cannot read."""
    real = backup_mod._write_zip_entries

    def flaky(zf, files_to_add, out_path, *, on_db_failure, on_error, on_progress, track_bytes):
        real(zf, files_to_add, out_path, on_db_failure=on_db_failure, on_error=on_error,
             on_progress=on_progress, track_bytes=track_bytes)
        on_error(Path(rel), FileNotFoundError(2, "No such file or directory", rel))

    monkeypatch.setattr(backup_mod, "_write_zip_entries", flaky)


def test_full_backup_reports_the_archive_it_wrote(home, capsys):
    update_cmd_maint._run_full_backup()

    out = capsys.readouterr().out
    archives = list((home / "backups").glob("pre-update-*.zip"))
    assert len(archives) == 1, archives
    assert archives[0].name in out
    assert "Saved:" in out
    # The reported count must be the archive's own content — a hardcoded number here would assert
    # the harness (whose log files also land in HERMES_HOME), not what the update reported.
    with zipfile.ZipFile(archives[0]) as zf:
        entries = [n for n in zf.namelist() if not n.endswith("/")]
    assert f"{len(entries)} file(s)" in out
    assert {"config.yaml", "notes.txt"} <= set(entries)
    assert "Backup skipped" not in out
    assert "Backup failed" not in out
    assert "skipped" not in out.lower()


def test_full_backup_says_nothing_to_copy_when_there_is_nothing(tmp_path, monkeypatch, capsys):
    empty = tmp_path / ".hermes"
    empty.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(empty))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # An isolated HERMES_HOME is never literally empty by the time the backup runs: this very
    # session opens its log inside it. Isolate the branch at its owner — the walk — so the test
    # asserts "nothing to copy" rather than the harness's own log file.
    monkeypatch.setattr(backup_mod, "_iter_backup_files", lambda *a, **k: iter(()))

    update_cmd_maint._run_full_backup()

    out = capsys.readouterr().out
    assert "nothing to copy" in out
    assert "Saved:" not in out
    assert "Backup failed" not in out


def test_full_backup_reports_a_write_failure_as_failed(home, monkeypatch, capsys):
    def boom(_root, _out_path):
        raise OSError("scan exploded")

    monkeypatch.setattr(backup_mod, "_iter_backup_files", boom)

    update_cmd_maint._run_full_backup()

    out = capsys.readouterr().out
    assert "Backup failed" in out
    assert "scan exploded" in out
    assert "nothing to copy" not in out
    assert "Saved:" not in out


def test_incomplete_backup_returns_its_salvage_path(home, monkeypatch):
    _fail_one_entry(monkeypatch, "cron/gone.stderr")

    outcome = backup_mod.create_pre_update_backup_outcome()

    assert outcome.status == "incomplete"
    assert outcome.path is not None and outcome.path.exists()
    assert outcome.path.name.endswith(".incomplete.zip")
    assert outcome.files == 1
    assert outcome.skipped == 1
    assert any("cron/gone.stderr" in line for line in outcome.errors)
    assert zipfile.is_zipfile(outcome.path)


def test_incomplete_backup_is_reported_with_its_path_not_as_skipped(home, monkeypatch, capsys):
    _fail_one_entry(monkeypatch, "cron/gone.stderr")

    update_cmd_maint._run_full_backup()

    out = capsys.readouterr().out
    archives = list((home / "backups").glob("pre-update-*.incomplete.zip"))
    assert len(archives) == 1, archives
    assert "Backup incomplete" in out
    assert "1 file(s) could not be read" in out
    assert archives[0].name in out
    assert "hermes import" in out
    assert "cron/gone.stderr" in out
    # The two lies go away: the archive is not "skipped", and it is not a failure either.
    assert "Backup skipped" not in out
    assert "Backup failed" not in out
