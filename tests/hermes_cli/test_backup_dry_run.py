"""`hermes backup --dry-run` — preview files without creating an archive.

With --dry-run, backup scans the hermes home directory and prints a summary
of what would be backed up (file count, total size) without creating any zip.
"""

import pytest


@pytest.fixture()
def tmp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Create some files so the backup has something to scan
    (tmp_path / "config.yaml").write_text("model: test\n")
    (tmp_path / "state.db").write_bytes(b"fake-db")
    (tmp_path / ".env").write_text("TEST_KEY=test\n")
    sub = tmp_path / "skills" / "myskill"
    sub.mkdir(parents=True)
    (sub / "SKILL.md").write_text("# Test\n")
    return tmp_path


def test_dry_run_prints_summary(tmp_home, monkeypatch, capsys):
    from hermes_cli import backup as mod
    import types

    args = types.SimpleNamespace(quick=False, output=None, dry_run=True, label=None)
    mod.run_backup(args)
    out = capsys.readouterr().out
    assert "Dry run" in out
    assert "Files:" in out


def test_dry_run_no_zip_created(tmp_home, monkeypatch, capsys):
    from hermes_cli import backup as mod
    import types

    args = types.SimpleNamespace(quick=False, output=None, dry_run=True, label=None)
    mod.run_backup(args)
    import os
    zips = list(tmp_home.parent.glob("hermes-backup-*.zip"))
    assert zips == [], "dry run should not create any zip files"


def test_dry_run_shows_file_count(tmp_home, monkeypatch, capsys):
    from hermes_cli import backup as mod
    import types

    args = types.SimpleNamespace(quick=False, output=None, dry_run=True, label=None)
    mod.run_backup(args)
    out = capsys.readouterr().out
    assert "Files:" in out
    # Should have at least 1 file
    for line in out.splitlines():
        if line.strip().startswith("Files:"):
            count = int(line.split(":")[1].strip())
            assert count >= 1
            break
