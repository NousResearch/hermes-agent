"""Regression test for files removed after backup enumeration."""

import zipfile
from argparse import Namespace
from pathlib import Path

import hermes_cli.backup as backup_mod


def test_backup_completes_when_member_vanishes_after_scan(tmp_path, monkeypatch, caplog):
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    source = hermes_home / "logs" / "rotated-receipt.json"
    source.parent.mkdir()
    source.write_text("receipt", encoding="utf-8")
    archive_path = tmp_path / "backup.zip"
    monkeypatch.setattr(
        backup_mod,
        "_iter_backup_files",
        lambda *_args, **_kwargs: iter([(source, Path("logs/rotated-receipt.json"))]),
    )
    original_write = backup_mod._write_zip_file

    def remove_before_write(zf, path, arcname):
        path.unlink()
        return original_write(zf, path, arcname)

    monkeypatch.setattr(backup_mod, "_write_zip_file", remove_before_write)

    from hermes_cli.main import cmd_backup

    assert cmd_backup(Namespace(output=str(archive_path), quick=False)) is None
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        assert archive.namelist() == []
    assert "vanished" in caplog.text.lower()
