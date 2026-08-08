"""Tests for the update check mechanism in hermes_cli.banner."""

import json
import os
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest




def test_check_for_updates_uses_cache(tmp_path, monkeypatch):
    """When cache is fresh, check_for_updates should return cached value without calling git."""
    from hermes_cli.banner import check_for_updates
    from hermes_cli import __version__

    # Create a fake git repo and fresh cache
    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    cache_file = tmp_path / ".update_check"
    cache_file.write_text(json.dumps({"ts": time.time(), "behind": 3, "ver": __version__}))

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with patch("hermes_cli.banner.subprocess.run") as mock_run:
        result = check_for_updates()

    assert result == 3
    mock_run.assert_not_called()






def test_prefetch_non_blocking():
    """prefetch_update_check() should return immediately without blocking."""
    import hermes_cli.banner as banner

    # Reset module state
    banner._update_result = None
    banner._update_check_done = threading.Event()

    with patch.object(banner, "check_for_updates", return_value=5):
        start = time.monotonic()
        banner.prefetch_update_check()
        elapsed = time.monotonic() - start

        # Should return almost immediately (well under 1 second)
        assert elapsed < 1.0

        # Wait for the background thread to finish
        banner._update_check_done.wait(timeout=5)
        assert banner._update_result == 5


def test_detached_latest_release_tag_is_current_in_full_clone(tmp_path):
    """Latest stable tag is current even when main has moved ahead (#39771)."""
    import hermes_cli.banner as banner

    repo_dir = tmp_path / "repo"
    git_dir = repo_dir / ".git"
    git_dir.mkdir(parents=True)
    (git_dir / "HEAD").write_text("local-sha\n", encoding="utf-8")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd == ["git", "remote", "get-url", "origin"]:
            return MagicMock(
                returncode=0,
                stdout="git@github.com:NousResearch/hermes-agent.git\n",
            )
        if cmd == ["git", "rev-parse", "--is-shallow-repository"]:
            return MagicMock(returncode=0, stdout="false\n")
        if cmd == ["git", "describe", "--tags", "--exact-match", "--match", "v[0-9]*", "HEAD"]:
            return MagicMock(returncode=0, stdout="v2026.5.29.2\n")
        if cmd[:2] == ["git", "ls-remote"]:
            return MagicMock(
                returncode=0,
                stdout=(
                    "new-sha\trefs/tags/v2026.5.29.2\n"
                    "old-sha\trefs/tags/v2026.5.29\n"
                ),
            )
        raise AssertionError(f"unexpected command: {cmd}")

    with patch("hermes_cli.banner.subprocess.run", side_effect=fake_run):
        result = banner._check_via_local_git(repo_dir)

    assert result == 0
    assert not any(cmd[:2] == ["git", "fetch"] for cmd in calls)
    assert not any(cmd[:3] == ["git", "rev-list", "--count"] for cmd in calls)


def test_detached_older_release_tag_reports_update_without_count(tmp_path):
    import hermes_cli.banner as banner

    repo_dir = tmp_path / "repo"
    git_dir = repo_dir / ".git"
    git_dir.mkdir(parents=True)
    (git_dir / "HEAD").write_text("local-sha\n", encoding="utf-8")

    def fake_run(cmd, **kwargs):
        if cmd == ["git", "remote", "get-url", "origin"]:
            return MagicMock(returncode=0, stdout=f"{banner._UPSTREAM_REPO_URL}\n")
        if cmd == ["git", "rev-parse", "--is-shallow-repository"]:
            return MagicMock(returncode=0, stdout="false\n")
        if cmd == ["git", "describe", "--tags", "--exact-match", "--match", "v[0-9]*", "HEAD"]:
            return MagicMock(returncode=0, stdout="v2026.5.29\n")
        if cmd[:2] == ["git", "ls-remote"]:
            return MagicMock(
                returncode=0,
                stdout=(
                    "new-sha\trefs/tags/v2026.5.29.2\n"
                    "old-sha\trefs/tags/v2026.5.29\n"
                ),
            )
        raise AssertionError(f"unexpected command: {cmd}")

    with patch("hermes_cli.banner.subprocess.run", side_effect=fake_run):
        result = banner._check_via_local_git(repo_dir)

    assert result == banner.UPDATE_AVAILABLE_NO_COUNT


@pytest.mark.parametrize(
    ("origin", "head", "expected"),
    [
        ("https://github.com/someone/hermes-agent.git", "local-sha\n", 7),
        ("https://github.com/NousResearch/hermes-agent.git", "ref: refs/heads/main\n", 7),
        ("git@github.com:NousResearch/hermes-agent.git", "ref: refs/heads/main\n", 1),
    ],
)
def test_nonrelease_checkout_keeps_full_clone_path(tmp_path, origin, head, expected):
    """Forks and attached branches retain their origin/main comparison."""
    import hermes_cli.banner as banner

    repo_dir = tmp_path / "repo"
    git_dir = repo_dir / ".git"
    git_dir.mkdir(parents=True)
    (git_dir / "HEAD").write_text(head, encoding="utf-8")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd == ["git", "remote", "get-url", "origin"]:
            return MagicMock(returncode=0, stdout=f"{origin}\n")
        if cmd == ["git", "rev-parse", "--is-shallow-repository"]:
            return MagicMock(returncode=0, stdout="false\n")
        if cmd == ["git", "rev-parse", "HEAD"]:
            return MagicMock(returncode=0, stdout="local-sha\n")
        if cmd == [
            "git",
            "ls-remote",
            banner._UPSTREAM_REPO_URL,
            "refs/heads/main",
        ]:
            return MagicMock(returncode=0, stdout="upstream-sha\trefs/heads/main\n")
        if cmd == ["git", "fetch", "origin", "main", "--quiet"]:
            return MagicMock(returncode=0, stdout="")
        if cmd == ["git", "rev-list", "--count", "HEAD..origin/main"]:
            return MagicMock(returncode=0, stdout="7\n")
        raise AssertionError(f"unexpected command: {cmd}")

    with patch("hermes_cli.banner.subprocess.run", side_effect=fake_run):
        result = banner._check_via_local_git(repo_dir)

    assert result == expected
    assert not any(len(cmd) > 1 and cmd[1] == "describe" for cmd in calls)
    if "someone" in origin:
        assert not any(cmd[:2] == ["git", "ls-remote"] for cmd in calls)


def test_detached_nonrelease_tag_keeps_full_clone_count_path(tmp_path):
    """Ad-hoc v-tags are not treated as published releases."""
    import hermes_cli.banner as banner

    repo_dir = tmp_path / "repo"
    git_dir = repo_dir / ".git"
    git_dir.mkdir(parents=True)
    (git_dir / "HEAD").write_text("local-sha\n", encoding="utf-8")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd == ["git", "remote", "get-url", "origin"]:
            return MagicMock(returncode=0, stdout=f"{banner._UPSTREAM_REPO_URL}\n")
        if cmd == ["git", "rev-parse", "--is-shallow-repository"]:
            return MagicMock(returncode=0, stdout="false\n")
        if cmd == ["git", "describe", "--tags", "--exact-match", "--match", "v[0-9]*", "HEAD"]:
            return MagicMock(returncode=1, stdout="")
        if cmd == ["git", "fetch", "origin", "main", "--quiet"]:
            return MagicMock(returncode=0, stdout="")
        if cmd == ["git", "rev-list", "--count", "HEAD..origin/main"]:
            return MagicMock(returncode=0, stdout="4\n")
        raise AssertionError(f"unexpected command: {cmd}")

    with patch("hermes_cli.banner.subprocess.run", side_effect=fake_run):
        result = banner._check_via_local_git(repo_dir)

    assert result == 4
    assert not any(cmd[:2] == ["git", "ls-remote"] for cmd in calls)


def test_shallow_detached_release_keeps_presence_only_path(tmp_path):
    """Shallow installer checkouts must not switch to release-tag probing."""
    import hermes_cli.banner as banner

    repo_dir = tmp_path / "repo"
    git_dir = repo_dir / ".git"
    git_dir.mkdir(parents=True)
    (git_dir / "HEAD").write_text("local-sha\n", encoding="utf-8")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd == ["git", "remote", "get-url", "origin"]:
            return MagicMock(returncode=0, stdout=f"{banner._UPSTREAM_REPO_URL}\n")
        if cmd == ["git", "rev-parse", "--is-shallow-repository"]:
            return MagicMock(returncode=0, stdout="true\n")
        if cmd == ["git", "fetch", "origin", "main", "--depth", "1", "--quiet"]:
            return MagicMock(returncode=0, stdout="")
        if cmd == ["git", "rev-parse", "HEAD"]:
            return MagicMock(returncode=0, stdout="local-sha\n")
        if cmd == ["git", "rev-parse", "FETCH_HEAD"]:
            return MagicMock(returncode=0, stdout="upstream-sha\n")
        raise AssertionError(f"unexpected command: {cmd}")

    with patch("hermes_cli.banner.subprocess.run", side_effect=fake_run):
        result = banner._check_via_local_git(repo_dir)

    assert result == banner.UPDATE_AVAILABLE_NO_COUNT
    assert not any(len(cmd) > 1 and cmd[1] in {"describe", "ls-remote"} for cmd in calls)


