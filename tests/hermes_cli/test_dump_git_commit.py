"""Tests for hermes_cli.dump._get_git_commit — git SHA resolution for ``hermes dump``.

``hermes dump`` prints the running commit so support bug reports identify the
exact version.  Source installs resolve it live via ``git rev-parse``; the
published Docker image excludes ``.git`` and falls back to the baked SHA
written by the Dockerfile's ``HERMES_GIT_SHA`` build-arg.

These tests cover both paths plus the failure modes (no git, no baked file).
"""

from unittest.mock import MagicMock, patch


def test_get_git_commit_uses_live_git_when_available(tmp_path):
    """Source install: ``git rev-parse --short=8 HEAD`` wins; no fallback."""
    from hermes_cli import dump

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    git_result = MagicMock(returncode=0, stdout="deadbeef\n")
    # build_info should NOT be consulted when live git succeeds.
    with patch("hermes_cli.dump.subprocess.run", return_value=git_result) as mock_run, \
         patch("hermes_cli.build_info.get_build_sha") as mock_build:
        commit = dump._get_git_commit(repo_dir)

    assert commit == "deadbeef"
    mock_run.assert_called_once()
    mock_build.assert_not_called()


def test_get_git_commit_output_format_identical_between_sources(tmp_path):
    """Regression guard: live-git and baked-SHA outputs share the same shape.

    Ben explicitly asked for identical output between Docker and source installs
    so support tooling that parses ``hermes dump`` doesn't have to special-case
    container builds.  Both paths must return a bare 8-char SHA — no prefix,
    no suffix, no annotation.
    """
    from hermes_cli import dump

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    # Live-git path.
    git_result = MagicMock(returncode=0, stdout="b2f477a3\n")
    with patch("hermes_cli.dump.subprocess.run", return_value=git_result):
        live = dump._get_git_commit(repo_dir)

    # Baked-SHA path.
    failed = MagicMock(returncode=128, stdout="")
    with patch("hermes_cli.dump.subprocess.run", return_value=failed), \
         patch("hermes_cli.build_info.get_build_sha", return_value="b2f477a3"):
        baked = dump._get_git_commit(repo_dir)

    assert live == baked == "b2f477a3"
    # Same length, same charset — no decoration in either branch.
    assert len(live) == 8
    assert all(c in "0123456789abcdef" for c in live)


def test_get_git_commit_date_empty_when_git_fails(tmp_path):
    """Docker image / pip wheel: no git → '' so the dump line drops the date."""
    from hermes_cli import dump

    repo_dir = tmp_path / "no-git-here"
    repo_dir.mkdir()

    failed = MagicMock(returncode=128, stdout="")
    with patch("hermes_cli.dump.subprocess.run", return_value=failed):
        date = dump._get_git_commit_date(repo_dir)

    assert date == ""


