"""Invalid full-backup retention must never delete a recovery archive."""

import argparse
import zipfile

import pytest


def test_full_backup_refuses_negative_retention_before_touching_archives(tmp_path, monkeypatch):
    from hermes_cli.backup import run_backup

    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("memory:\n  provider: null\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "managed"))
    output = tmp_path / "exports"
    output.mkdir()
    archive = output / "hermes-backup-2000-01-01-000000.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("config.yaml", "previous recovery settings")
    original = archive.read_bytes()

    with pytest.raises(ValueError, match="keep"):
        run_backup(argparse.Namespace(output=str(output), keep=-1))

    assert list(output.iterdir()) == [archive]
    assert archive.read_bytes() == original


def test_backup_parser_rejects_negative_retention_and_preserves_supported_values():
    from hermes_cli.subcommands.backup import build_backup_parser

    parser = argparse.ArgumentParser()
    build_backup_parser(parser.add_subparsers(dest="command"), cmd_backup=lambda args: None)

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["backup", "--keep", "-1"])
    assert exc.value.code == 2
    assert parser.parse_args(["backup", "--keep", "0"]).keep == 0
    assert parser.parse_args(["backup", "--keep", "1"]).keep == 1
    assert parser.parse_args(["backup"]).keep == 3
