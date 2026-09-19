"""Tests for hermes_cli.dump._get_git_commit — git SHA resolution for ``hermes dump``.

``hermes dump`` prints the running commit so support bug reports identify the
exact version. Source installs resolve it live via git; packaged builds
(Docker, Nix) use the install stamp via ``version_info`` — but ONLY when the
requested project_root IS the running install, because version_info has no
per-root API and would otherwise answer with an unrelated install's identity.

These tests cover both paths plus the failure modes (no stamp, no git, and a
project_root the running install cannot speak for).
"""

from pathlib import Path
from subprocess import run as _run
from unittest.mock import patch

import pytest

from hermes_cli.version_info import VersionInfo, _reset_version_info_cache


def setup_function():
    _reset_version_info_cache()





@pytest.mark.parametrize('sha,timestamp,expected,date', [
    ('cafef00d' * 5, 1718662620, 'cafef00d', '2024-06-17'),
    (None, None, '(unknown)', ''),
])
def test_dump_uses_running_install_fallback_only(tmp_path, monkeypatch, sha, timestamp, expected, date):
    from hermes_cli import dump

    monkeypatch.setattr(dump, 'get_project_root', lambda: tmp_path)
    info = VersionInfo('1.0', '1.0', None, sha, None, 'docker', False, timestamp)
    monkeypatch.setattr('hermes_cli.version_info.get_version_info', lambda: info)
    assert dump._get_git_commit(tmp_path) == expected
    assert dump._get_git_commit_date(tmp_path) == date


# --------------------------------------------------------------------------
# Authority: the requested project_root decides, never the running install
# --------------------------------------------------------------------------


def test_get_git_commit_unknown_for_other_root_without_git(tmp_path):
    """A non-git project_root that is NOT the running install → '(unknown)'.

    No mocks: the running install (this checkout) HAS git provenance, so the
    pre-authority fallback would have answered with THIS repo's sha — an
    unrelated install's identity — instead of admitting unknown.
    """
    from hermes_cli import dump

    other_root = tmp_path / "plain-dir"
    other_root.mkdir()
    assert (other_root / ".git").exists() is False

    commit = dump._get_git_commit(other_root)
    assert commit == "(unknown)"


def test_get_git_commit_authoritative_for_real_temp_repo(tmp_path):
    """A real temp git repo reports ITS OWN commit, not the running install's."""
    from hermes_cli import dump

    repo_dir = tmp_path / "scratch-repo"
    repo_dir.mkdir()
    (repo_dir / "file.txt").write_text("x", encoding="utf-8")
    for args in (
        ["init", "-b", "main"],
        ["-c", "user.email=t@example.com", "-c", "user.name=t", "add", "."],
        ["-c", "user.email=t@example.com", "-c", "user.name=t",
         "commit", "-m", "scratch", "--no-gpg-sign"],
    ):
        _run(["git", *args], cwd=repo_dir, capture_output=True, check=True)
    expected = _run(["git", "rev-parse", "--short=8", "HEAD"], cwd=repo_dir,
                    capture_output=True, text=True, check=True).stdout.strip()

    commit = dump._get_git_commit(repo_dir)
    assert commit == expected
    assert commit != dump._get_git_commit(Path(dump.__file__).parent.parent)


def test_get_git_commit_date_unknown_for_other_root_without_git(tmp_path):
    """Same authority rule for the date helper: other root without git → ''."""
    from hermes_cli import dump

    other_root = tmp_path / "plain-dir"
    other_root.mkdir()

    assert dump._get_git_commit_date(other_root) == ""
