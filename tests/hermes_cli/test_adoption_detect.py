"""Tests for ``hermes_cli.adoption.detect_legacy_install`` — the legacy-layout
detector for the adoption funnel.

These tests exercise real git repos created in temp directories so we test
behavior, not mocks (per AGENTS.md: E2E validation over green unit mocks).
``detect_install_method`` is mocked only for the non-git-method cases where
we need to simulate docker/nixos.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

from hermes_cli.adoption import LegacyInfo, detect_legacy_install

OFFICIAL_CANONICAL = "github.com/nousresearch/hermes-agent"


# ---------------------------------------------------------------------------
# Git fixture helpers
# ---------------------------------------------------------------------------

def _git(args: list[str], cwd: Path, **kw) -> str:
    """Run git in ``cwd`` and return stdout, asserting success."""
    result = subprocess.run(
        ["git"] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
        **kw,
    )
    assert result.returncode == 0, f"git {args} failed: {result.stderr}"
    return result.stdout.strip()


def _make_repo(path: Path, origin_url: str = "https://github.com/NousResearch/hermes-agent.git") -> Path:
    """Create a clean git repo at ``path`` with ``origin`` set to ``origin_url``.

    The repo has one initial commit on ``main`` with a placeholder file,
    and ``origin/main`` is set up so the ahead-count check works.
    """
    path.mkdir(parents=True, exist_ok=True)
    _git(["init", "-b", "main"], path)
    _git(["config", "user.email", "test@example.com"], path)
    _git(["config", "user.name", "Test"], path)
    (path / "README.md").write_text("hello\n")
    _git(["add", "-A"], path)
    _git(["commit", "-m", "initial"], path)

    # Set up origin so rev-parse --abbrev-ref and rev-list HEAD..origin/main work.
    _git(["remote", "add", "origin", origin_url], path)
    # Create origin/main ref pointing at HEAD.
    _git(["update-ref", "refs/remotes/origin/main", "HEAD"], path)
    return path


def _make_repo_with_remote(path: Path, origin_url: str) -> Path:
    """Create a repo where ``origin`` is a real bare remote, so fetch/push work.

    This lets us simulate commits-ahead by pushing then committing locally.
    """
    remote = path.parent / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], capture_output=True)
    path.mkdir(parents=True, exist_ok=True)
    _git(["init", "-b", "main"], path)
    _git(["config", "user.email", "test@example.com"], path)
    _git(["config", "user.name", "Test"], path)
    (path / "README.md").write_text("hello\n")
    _git(["add", "-A"], path)
    _git(["commit", "-m", "initial"], path)
    _git(["remote", "add", "origin", str(remote)], path)
    _git(["push", "-u", "origin", "main"], path)
    # Ensure origin/main ref exists locally
    _git(["fetch", "origin"], path)
    return path


# ---------------------------------------------------------------------------
# None-returning cases (no adoption)
# ---------------------------------------------------------------------------

class TestReturnsNone:
    """detect_legacy_install returns None when adoption doesn't apply."""

    def test_returns_none_for_slot_layout(self, tmp_path: Path):
        """Running from a path under versions/ is a managed slot, not legacy."""
        slot = tmp_path / "versions" / "v1" / "hermes-agent"
        slot.mkdir(parents=True)
        (slot / ".git").mkdir()
        # Even if detect_install_method says "git", the slot check returns None.
        with patch("hermes_cli.adoption.detect_install_method", return_value="git"):
            result = detect_legacy_install(slot, tmp_path / "home")
        assert result is None

    def test_returns_none_for_docker(self, tmp_path: Path):
        """Docker installs cannot adopt."""
        with patch("hermes_cli.adoption.detect_install_method", return_value="docker"):
            result = detect_legacy_install(tmp_path, tmp_path / "home")
        assert result is None

    def test_returns_none_for_nixos(self, tmp_path: Path):
        """NixOS installs cannot adopt."""
        with patch("hermes_cli.adoption.detect_install_method", return_value="nixos"):
            result = detect_legacy_install(tmp_path, tmp_path / "home")
        assert result is None

    def test_returns_none_for_homebrew(self, tmp_path: Path):
        """Homebrew installs cannot adopt."""
        with patch("hermes_cli.adoption.detect_install_method", return_value="homebrew"):
            result = detect_legacy_install(tmp_path, tmp_path / "home")
        assert result is None

    def test_returns_none_for_pip(self, tmp_path: Path):
        """Pip installs cannot adopt."""
        with patch("hermes_cli.adoption.detect_install_method", return_value="pip"):
            result = detect_legacy_install(tmp_path, tmp_path / "home")
        assert result is None


# ---------------------------------------------------------------------------
# Pristine git checkout
# ---------------------------------------------------------------------------

class TestPristine:
    """A clean checkout with official origin on main, no commits ahead → pristine."""

    def test_pristine_checkout(self, tmp_path: Path):
        repo = _make_repo(tmp_path / "repo")
        with patch("hermes_cli.adoption.detect_install_method", return_value="git"):
            result = detect_legacy_install(repo, tmp_path / "home")
        assert result is not None
        assert result.pristine is True
        assert result.reasons == []


# ---------------------------------------------------------------------------
# Dirty cohorts
# ---------------------------------------------------------------------------

class TestDirty:
    """A dirty working tree → not pristine, reason mentions 'dirty'."""

    def test_dirty_tree(self, tmp_path: Path):
        repo = _make_repo(tmp_path / "repo")
        # Make the tree dirty.
        (repo / "README.md").write_text("changed\n")
        with patch("hermes_cli.adoption.detect_install_method", return_value="git"):
            result = detect_legacy_install(repo, tmp_path / "home")
        assert result is not None
        assert result.pristine is False
        assert any("dirty" in r.lower() for r in result.reasons)


# ---------------------------------------------------------------------------
# Fork remote
# ---------------------------------------------------------------------------

class TestForkRemote:
    """An origin pointing at a fork → not pristine, reason mentions 'fork'."""

    def test_fork_remote(self, tmp_path: Path):
        repo = _make_repo(
            tmp_path / "repo",
            origin_url="https://github.com/someoneelse/hermes-agent.git",
        )
        with patch("hermes_cli.adoption.detect_install_method", return_value="git"):
            result = detect_legacy_install(repo, tmp_path / "home")
        assert result is not None
        assert result.pristine is False
        assert any("fork" in r.lower() for r in result.reasons)

    def test_ssh_official_remote_is_recognized(self, tmp_path: Path):
        """SSH form of the official URL should be recognized as official."""
        repo = _make_repo(
            tmp_path / "repo",
            origin_url="git@github.com:NousResearch/hermes-agent.git",
        )
        with patch("hermes_cli.adoption.detect_install_method", return_value="git"):
            result = detect_legacy_install(repo, tmp_path / "home")
        assert result is not None
        assert result.pristine is True


# ---------------------------------------------------------------------------
# Commits ahead
# ---------------------------------------------------------------------------

class TestCommitsAhead:
    """Local commits ahead of origin/main → not pristine, reason mentions 'ahead'."""

    def test_commits_ahead_of_origin(self, tmp_path: Path):
        repo = _make_repo_with_remote(tmp_path / "repo", "https://github.com/NousResearch/hermes-agent.git")
        # Make an extra commit locally that is NOT pushed to origin.
        (repo / "extra.txt").write_text("extra\n")
        _git(["add", "-A"], repo)
        _git(["commit", "-m", "local commit"], repo)
        with patch("hermes_cli.adoption.detect_install_method", return_value="git"):
            result = detect_legacy_install(repo, tmp_path / "home")
        assert result is not None
        assert result.pristine is False
        assert any("ahead" in r.lower() for r in result.reasons)


# ---------------------------------------------------------------------------
# Unknown branch
# ---------------------------------------------------------------------------

class TestUnknownBranch:
    """On a branch other than main → not pristine."""

    def test_not_on_main(self, tmp_path: Path):
        repo = _make_repo(tmp_path / "repo")
        _git(["checkout", "-b", "feature-branch"], repo)
        with patch("hermes_cli.adoption.detect_install_method", return_value="git"):
            result = detect_legacy_install(repo, tmp_path / "home")
        assert result is not None
        assert result.pristine is False
        assert any("branch" in r.lower() for r in result.reasons)


# ---------------------------------------------------------------------------
# Crash-proof
# ---------------------------------------------------------------------------

class TestCrashProof:
    """The detector never raises; git errors produce pristine=False."""

    def test_git_unavailable(self, tmp_path: Path):
        """If git binary is missing, returns LegacyInfo(pristine=False) not raise."""
        with patch("hermes_cli.adoption.detect_install_method", return_value="git"), \
             patch("hermes_cli.adoption.subprocess.run", side_effect=FileNotFoundError("git not found")):
            result = detect_legacy_install(tmp_path, tmp_path / "home")
        assert result is not None
        assert result.pristine is False
        assert len(result.reasons) > 0

    def test_not_a_git_repo(self, tmp_path: Path):
        """A directory that is not a git repo → pristine=False, no exception."""
        repo = tmp_path / "repo"
        repo.mkdir()
        with patch("hermes_cli.adoption.detect_install_method", return_value="git"):
            result = detect_legacy_install(repo, tmp_path / "home")
        assert result is not None
        assert result.pristine is False
        assert len(result.reasons) > 0


# ---------------------------------------------------------------------------
# Remote canonicalization (ported from update-remote.ts)
# ---------------------------------------------------------------------------

class TestCanonicalRemote:
    """Tests for the ported canonicalGitHubRemote logic."""

    def test_https_official(self):
        from hermes_cli.adoption import _canonical_github_remote
        assert _canonical_github_remote("https://github.com/NousResearch/hermes-agent.git") == OFFICIAL_CANONICAL

    def test_ssh_official(self):
        from hermes_cli.adoption import _canonical_github_remote
        assert _canonical_github_remote("git@github.com:NousResearch/hermes-agent.git") == OFFICIAL_CANONICAL

    def test_ssh_scheme_official(self):
        from hermes_cli.adoption import _canonical_github_remote
        assert _canonical_github_remote("ssh://git@github.com/NousResearch/hermes-agent.git") == OFFICIAL_CANONICAL

    def test_https_no_git_suffix(self):
        from hermes_cli.adoption import _canonical_github_remote
        assert _canonical_github_remote("https://github.com/NousResearch/hermes-agent") == OFFICIAL_CANONICAL

    def test_fork_url_not_official(self):
        from hermes_cli.adoption import _is_official_remote
        assert not _is_official_remote("https://github.com/somefork/hermes-agent.git")

    def test_case_insensitive(self):
        from hermes_cli.adoption import _canonical_github_remote
        assert _canonical_github_remote("https://github.com/NOUSRESEARCH/Hermes-Agent.git") == OFFICIAL_CANONICAL

    def test_empty_url(self):
        from hermes_cli.adoption import _canonical_github_remote
        assert _canonical_github_remote("") == ""
