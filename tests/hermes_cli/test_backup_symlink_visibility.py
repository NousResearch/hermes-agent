"""`hermes backup` must report symlinks it deliberately drops.

Symlinks are skipped for security — zipfile.write() follows a file symlink and
os.walk won't descend a symlinked directory, so a link could otherwise pull
data from outside HERMES_HOME into the archive (or silently omit a whole
content tree). That skip is correct and stays. The bug these tests pin is that
the skip used to be *silent*: a symlinked ``profiles/<name>/skills`` tree
vanished from the backup while the run still printed "Backup complete", so the
loss only surfaced at restore time. The backup must now name what it dropped.
"""

import logging
import zipfile
from argparse import Namespace
from pathlib import Path

import pytest


def _make_minimal_home(root: Path) -> None:
    """A backup-valid HERMES_HOME with no symlinks of its own."""
    (root / "config.yaml").write_text("model:\n  provider: openrouter\n")
    (root / ".env").write_text("OPENROUTER_API_KEY=sk-test\n")
    (root / "skills").mkdir()
    (root / "skills" / "my-skill").mkdir()
    (root / "skills" / "my-skill" / "SKILL.md").write_text("# My Skill\n")


def _symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except OSError as exc:  # pragma: no cover - platform-dependent
        pytest.skip(f"symlinks unavailable in test environment: {exc}")


def _run_backup(hermes_home: Path, tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    out_zip = tmp_path / "backup.zip"
    from hermes_cli.backup import run_backup
    run_backup(Namespace(output=str(out_zip)))
    return out_zip


def test_symlinked_file_is_skipped_and_reported(tmp_path, monkeypatch, capsys):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    _make_minimal_home(hermes_home)

    outside = tmp_path / "outside-secret.txt"
    outside.write_text("outside secret\n")
    _symlink_or_skip(hermes_home / "skills" / "outside-link.txt", outside)

    out_zip = _run_backup(hermes_home, tmp_path, monkeypatch)

    out = capsys.readouterr().out
    assert "Symlinks skipped (not archived): 1" in out
    assert "skills/outside-link.txt" in out
    # Reporting the skip must not turn a policy skip into a failure.
    assert "Backup complete" in out
    assert "Backup incomplete" not in out

    with zipfile.ZipFile(out_zip) as zf:
        names = zf.namelist()
        assert "skills/outside-link.txt" not in names
        assert all(zf.read(n) != b"outside secret\n" for n in names)


def test_symlinked_directory_subtree_is_skipped_and_reported(tmp_path, monkeypatch, capsys):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    _make_minimal_home(hermes_home)

    # A real content tree that lives outside HERMES_HOME, linked in as a whole
    # subtree — the exact "symlinked profile skills tree" failure case.
    outside_tree = tmp_path / "shared-skills"
    outside_tree.mkdir()
    (outside_tree / "shared.md").write_text("shared skill body\n")
    (outside_tree / "nested").mkdir()
    (outside_tree / "nested" / "deep.md").write_text("deep body\n")

    (hermes_home / "profiles").mkdir()
    (hermes_home / "profiles" / "coder").mkdir()
    _symlink_or_skip(hermes_home / "profiles" / "coder" / "skills", outside_tree)

    out_zip = _run_backup(hermes_home, tmp_path, monkeypatch)

    out = capsys.readouterr().out
    assert "Symlinks skipped (not archived): 1" in out
    assert "profiles/coder/skills" in out
    assert "Backup complete" in out

    with zipfile.ZipFile(out_zip) as zf:
        names = zf.namelist()
        # The subtree and its contents are genuinely absent from the archive.
        assert not any(n.startswith("profiles/coder/skills") for n in names)
        assert "shared skill body\n" not in [
            zf.read(n).decode("utf-8", "ignore") for n in names
        ]


def test_no_symlinks_prints_no_symlink_block(tmp_path, monkeypatch, capsys):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    _make_minimal_home(hermes_home)

    _run_backup(hermes_home, tmp_path, monkeypatch)

    out = capsys.readouterr().out
    assert "Symlinks skipped" not in out
    assert "Backup complete" in out


def test_pre_update_backup_reports_skipped_subtree_via_log(tmp_path, caplog):
    """The pre-update/pre-migration full-zip path (_write_full_zip_backup) has
    no interactive summary, so it must surface the same dropped subtree through
    its logger instead of dropping it silently."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    _make_minimal_home(hermes_home)

    outside_tree = tmp_path / "shared-skills"
    outside_tree.mkdir()
    (outside_tree / "shared.md").write_text("shared skill body\n")
    (hermes_home / "profiles").mkdir()
    (hermes_home / "profiles" / "coder").mkdir()
    _symlink_or_skip(hermes_home / "profiles" / "coder" / "skills", outside_tree)

    from hermes_cli.backup import create_pre_update_backup

    with caplog.at_level(logging.WARNING, logger="hermes_cli.backup"):
        out = create_pre_update_backup(hermes_home=hermes_home)

    assert out is not None and out.exists()

    messages = "\n".join(r.getMessage() for r in caplog.records)
    assert "symlink(s) skipped (not archived)" in messages
    assert "profiles/coder/skills" in messages

    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        assert not any(n.startswith("profiles/coder/skills") for n in names)
        assert "shared skill body\n" not in [
            zf.read(n).decode("utf-8", "ignore") for n in names
        ]
