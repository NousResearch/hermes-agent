"""Behavioral tests for the profile archive CI guard."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "ci"
    / "check_profile_archive_boundary.py"
)


def _run(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_clean_root_passes(tmp_path):
    result = _run(tmp_path)

    assert result.returncode == 0
    assert "No profile export archives" in result.stdout


def test_root_profile_archive_fails_without_printing_contents(tmp_path):
    default_archive = tmp_path / "default.tar.gz"
    alternate_archive = tmp_path / "backup.TGZ"
    default_archive.write_bytes(b"profile webhook secret must never be printed")
    alternate_archive.write_bytes(b"another archive")

    result = _run(tmp_path)

    assert result.returncode == 1
    assert "default.tar.gz" in result.stdout
    assert "backup.TGZ" in result.stdout
    assert "profile webhook secret" not in result.stdout
    assert "another archive" not in result.stdout


def test_nested_profile_archive_is_also_rejected(tmp_path):
    nested = tmp_path / "fixtures"
    nested.mkdir()
    (nested / "fixture.tar.gz").write_bytes(b"test fixture")

    result = _run(tmp_path)

    assert result.returncode == 1
    assert "fixtures/fixture.tar.gz" in result.stdout


def _git_root(root: Path, ignores: str) -> None:
    """A checkout whose ``.gitignore`` is *ignores*. Nothing is committed, so every file on disk
    is "others" and only the ignore rules decide what the walk skips."""
    subprocess.run(["git", "init", "-q"], cwd=root, check=True, capture_output=True)
    (root / ".gitignore").write_text(ignores, encoding="utf-8")


def test_archives_inside_a_gitignored_directory_are_local_state_not_offenders(tmp_path):
    """The desktop packager unpacks its build inputs under ``.cache/`` - gitignored, and absent
    from the image build context. Those archives never were profile exports."""
    _git_root(tmp_path, "*.tar.gz\n.cache/\n")
    builder = tmp_path / ".cache" / "desktop-inputs" / "packager" / "builder"
    builder.mkdir(parents=True)
    (builder / "7zip-darwin-arm64.tar.gz").write_bytes(b"packager input")

    result = _run(tmp_path)

    assert result.returncode == 0
    assert "7zip-darwin-arm64.tar.gz" not in result.stdout


def test_archive_matching_an_ignore_pattern_is_still_rejected(tmp_path):
    """``.gitignore`` also ignores ``*.tar.gz`` repo-wide. Honouring that would switch the guard
    off entirely: only a whole ignored DIRECTORY is local state, while a stray export sitting in
    the checkout is precisely the incident (#92457) this check is the boundary for."""
    _git_root(tmp_path, "*.tar.gz\n")
    (tmp_path / "profile-export.tar.gz").write_bytes(b"profile webhook secret must never be printed")

    result = _run(tmp_path)

    assert result.returncode == 1
    assert "profile-export.tar.gz" in result.stdout
    assert "profile webhook secret" not in result.stdout


def test_force_added_archive_inside_a_gitignored_directory_is_still_rejected(tmp_path):
    """The bypass this guard exists for: ``git add -f`` puts the archive in the index, so it is no
    longer "others" and the ignore set must not hide it."""
    _git_root(tmp_path, ".cache/\n")
    (tmp_path / ".cache").mkdir()
    (tmp_path / ".cache" / "export.tgz").write_bytes(b"forced")
    subprocess.run(["git", "add", "-f", ".cache/export.tgz"], cwd=tmp_path, check=True, capture_output=True)

    result = _run(tmp_path)

    assert result.returncode == 1
    assert ".cache/export.tgz" in result.stdout
