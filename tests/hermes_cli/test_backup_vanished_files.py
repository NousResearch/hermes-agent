"""A file removed between the backup scan and its write must not fail the backup (homelab #882).

Per-job cron output pruning deletes old ``cron/output/<job>/*.md`` files every few minutes.
A prune that lands inside the archive phase made ``hermes backup`` exit 1 with
``errors=1`` every night. The file no longer exists, so there is nothing to archive and the
archive is still complete for the state that exists.
"""

from argparse import Namespace

import pytest


def _make_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "skills").mkdir(parents=True)
    (home / "config.yaml").write_text("model: {}\n")
    (home / "skills" / "keep.txt").write_text("keep me\n")
    (home / "skills" / "second.txt").write_text("second\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _backup(home_root):
    import hermes_cli.backup as backup_mod

    out = home_root / "out.zip"
    return backup_mod, backup_mod.run_backup(Namespace(output=str(out))), out


def _vanish(monkeypatch, backup_mod, doomed):
    """Make the first write of *doomed* remove it, so the writer sees FileNotFoundError."""
    real = backup_mod._write_zip_file
    state = {"done": False}

    def fake(zf, path, arcname):
        if str(path) == str(doomed) and not state["done"]:
            state["done"] = True
            doomed.unlink()
        return real(zf, path, arcname)

    monkeypatch.setattr(backup_mod, "_write_zip_file", fake)


def test_vanished_file_does_not_fail_backup(tmp_path, monkeypatch, capsys):
    home = _make_home(tmp_path, monkeypatch)
    import hermes_cli.backup as backup_mod

    doomed = home / "skills" / "keep.txt"
    _vanish(monkeypatch, backup_mod, doomed)

    ok = backup_mod.run_backup(Namespace(output=str(tmp_path / "out.zip")))
    out = capsys.readouterr().out

    assert ok is True
    assert "could not be added" not in out
    assert "Backup complete" in out
    assert "Skipped 1 file(s) removed during the backup." in out
    with backup_mod.zipfile.ZipFile(tmp_path / "out.zip") as zf:
        names = zf.namelist()
    assert not any(name.endswith("keep.txt") for name in names)
    assert any(name.endswith("second.txt") for name in names)


def test_three_vanished_files_still_complete(tmp_path, monkeypatch, capsys):
    home = _make_home(tmp_path, monkeypatch)
    extra = [home / "skills" / f"extra{i}.txt" for i in range(2)]
    for path in extra:
        path.write_text("x\n")
    import hermes_cli.backup as backup_mod

    doomed = {str(p) for p in [home / "skills" / "keep.txt", *extra]}
    real = backup_mod._write_zip_file

    def fake(zf, path, arcname):
        if str(path) in doomed:
            doomed.discard(str(path))
            path.unlink()
        return real(zf, path, arcname)

    monkeypatch.setattr(backup_mod, "_write_zip_file", fake)

    assert backup_mod.run_backup(Namespace(output=str(tmp_path / "out.zip"))) is True
    out = capsys.readouterr().out
    assert "could not be added" not in out
    assert "Skipped 3 file(s) removed during the backup." in out


def test_read_failure_still_fails_backup(tmp_path, monkeypatch, capsys):
    home = _make_home(tmp_path, monkeypatch)
    import hermes_cli.backup as backup_mod

    doomed = str(home / "skills" / "keep.txt")
    real = backup_mod._write_zip_file

    def fake(zf, path, arcname):
        if str(path) == doomed:
            raise PermissionError(13, "denied")
        return real(zf, path, arcname)

    monkeypatch.setattr(backup_mod, "_write_zip_file", fake)

    assert backup_mod.run_backup(Namespace(output=str(tmp_path / "out.zip"))) is False
    out = capsys.readouterr().out
    assert "could not be added" in out
    assert "keep.txt" in out


def test_vanished_database_is_skipped_not_a_db_failure(tmp_path, monkeypatch, capsys):
    home = _make_home(tmp_path, monkeypatch)
    db = home / "state.db"
    db.write_bytes(b"SQLite format 3\x00" + b"\x00" * 100)
    import hermes_cli.backup as backup_mod

    def boom(*_args, **_kwargs):
        raise FileNotFoundError(2, "gone")

    monkeypatch.setattr(backup_mod, "_zip_sqlite_snapshot", boom)

    assert backup_mod.run_backup(Namespace(output=str(tmp_path / "out.zip"))) is True
    out = capsys.readouterr().out
    assert "SQLite safe copy failed" not in out
    assert "could not be added" not in out
    assert "Skipped 1 file(s) removed during the backup." in out


def test_normal_backup_is_unchanged(tmp_path, monkeypatch, capsys):
    home = _make_home(tmp_path, monkeypatch)
    import hermes_cli.backup as backup_mod

    assert backup_mod.run_backup(Namespace(output=str(tmp_path / "out.zip"))) is True
    out = capsys.readouterr().out
    assert "Backup complete" in out
    assert "Skipped" not in out
    assert "Restore with:" in out
    with backup_mod.zipfile.ZipFile(tmp_path / "out.zip") as zf:
        names = zf.namelist()
    assert any(name.endswith("keep.txt") for name in names)
