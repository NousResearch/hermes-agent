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

from hermes_cli.version_info import VersionInfo, _reset_version_info_cache


def setup_function():
    _reset_version_info_cache()


def _patch_running_install(repo_dir):
    """Make the requested repo_dir the running install for the fallback path."""
    return patch("hermes_cli.dump.get_project_root", return_value=repo_dir.resolve())


def test_get_git_commit_uses_live_git_when_available(tmp_path):
    """Source install: version_info resolves commit from live git."""
    from hermes_cli import dump

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    with _patch_running_install(repo_dir), \
         patch("hermes_cli.version_info._resolve_stamp_file", lambda: None), \
         patch("hermes_cli.version_info._resolve_repo_dir", lambda: repo_dir), \
         patch("hermes_cli.version_info._git_version_info",
               return_value=VersionInfo("0.19.0", "0.19.0+3", 3, "deadbeef" * 5, "main", "git")):
        commit = dump._get_git_commit(repo_dir)

    assert commit == "deadbeef"


def test_get_git_commit_uses_stamp_when_no_git(tmp_path):
    """Docker/Nix: version_info resolves commit from the install stamp."""
    from hermes_cli import dump

    repo_dir = tmp_path / "no-git-here"
    repo_dir.mkdir()

    with _patch_running_install(repo_dir), \
         patch("hermes_cli.version_info._resolve_stamp_file", lambda: tmp_path / "stamp.json"), \
         patch("hermes_cli.version_info._resolve_repo_dir", lambda: None), \
         patch("hermes_cli.version_info._stamp_version_info",
               return_value=VersionInfo("0.19.0", "0.19.0", None, "cafef00d" * 5, None, "docker")):
        commit = dump._get_git_commit(repo_dir)

    assert commit == "cafef00d"


def test_get_git_commit_returns_unknown_when_neither_source_available(tmp_path):
    """Pip-installed wheel: no stamp, no git → '(unknown)'."""
    from hermes_cli import dump

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    with _patch_running_install(repo_dir), \
         patch("hermes_cli.version_info._resolve_stamp_file", lambda: None), \
         patch("hermes_cli.version_info._resolve_repo_dir", lambda: None):
        commit = dump._get_git_commit(repo_dir)

    assert commit == "(unknown)"


def test_get_git_commit_output_format_identical_between_sources(tmp_path):
    """Regression guard: live-git and stamp outputs share the same shape."""
    from hermes_cli import dump

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    # Live-git path.
    with _patch_running_install(repo_dir), \
         patch("hermes_cli.version_info._resolve_stamp_file", lambda: None), \
         patch("hermes_cli.version_info._resolve_repo_dir", lambda: repo_dir), \
         patch("hermes_cli.version_info._git_version_info",
               return_value=VersionInfo("0.19.0", "0.19.0+3", 3, "b2f477a3" * 5, "main", "git")):
        _reset_version_info_cache()
        live = dump._get_git_commit(repo_dir)

    # Stamp path.
    with _patch_running_install(repo_dir), \
         patch("hermes_cli.version_info._resolve_stamp_file", lambda: tmp_path / "stamp.json"), \
         patch("hermes_cli.version_info._resolve_repo_dir", lambda: None), \
         patch("hermes_cli.version_info._stamp_version_info",
               return_value=VersionInfo("0.19.0", "0.19.0", None, "b2f477a3" * 5, None, "docker")):
        _reset_version_info_cache()
        baked = dump._get_git_commit(repo_dir)

    assert live == baked == "b2f477a3"
    assert len(live) == 8
    assert all(c in "0123456789abcdef" for c in live)


def test_get_git_commit_date_uses_version_info(tmp_path):
    """Source install: version_info carries the commit date from live git."""
    from hermes_cli import dump

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    with _patch_running_install(repo_dir), \
         patch("hermes_cli.version_info._resolve_stamp_file", lambda: None), \
         patch("hermes_cli.version_info._resolve_repo_dir", lambda: repo_dir), \
         patch("hermes_cli.version_info._git_version_info",
               return_value=VersionInfo("0.19.0", "0.19.0+3", 3, "deadbeef" * 5, "main", "git", False, 1718662620)):
        _reset_version_info_cache()
        date = dump._get_git_commit_date(repo_dir)

    assert date == "2024-06-17"


def test_get_git_commit_date_empty_when_unknown(tmp_path):
    """Docker/pip: no git, no stamp → '' so the dump line drops the date."""
    from hermes_cli import dump

    repo_dir = tmp_path / "no-git-here"
    repo_dir.mkdir()

    with _patch_running_install(repo_dir), \
         patch("hermes_cli.version_info._resolve_stamp_file", lambda: None), \
         patch("hermes_cli.version_info._resolve_repo_dir", lambda: None):
        _reset_version_info_cache()
        date = dump._get_git_commit_date(repo_dir)

    assert date == ""


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
