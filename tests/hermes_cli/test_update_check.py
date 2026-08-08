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
    import hermes_cli.banner as banner
    from hermes_cli import __version__

    # Create a fake git repo and fresh cache
    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    cache_file = tmp_path / ".update_check"
    cache_file.write_text(
        json.dumps(
            {
                "ts": time.time(),
                "behind": 3,
                "ver": __version__,
                "schema": banner._UPDATE_CHECK_CACHE_VERSION,
            }
        )
    )

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with patch("hermes_cli.banner.subprocess.run") as mock_run:
        result = banner.check_for_updates()

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


def test_local_git_update_check_prefers_upstream_for_shallow_checkout(tmp_path):
    """A shallow fork must compare against Nous upstream, not the personal fork."""
    import hermes_cli.banner as banner

    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)
    fetch_head: dict[str, str | None] = {"value": None}
    fetches = []

    def fake_git_stdout(args, *, cwd, timeout=5):
        values = {
            ("remote", "get-url", "origin"): "https://github.com/Sahil-SS9/KenseiAgent.git",
            ("remote", "get-url", "upstream"): "https://github.com/NousResearch/hermes-agent.git",
            ("rev-parse", "--is-shallow-repository"): "true",
            ("rev-parse", "HEAD"): "local-tip",
            ("rev-parse", "FETCH_HEAD"): fetch_head["value"],
        }
        return values.get(tuple(args))

    def fake_run(cmd, **kwargs):
        if cmd[:2] == ["git", "fetch"]:
            fetches.append(list(cmd))
            remote = next(remote for remote in ("upstream", "origin") if remote in cmd)
            fetch_head["value"] = "nous-tip" if remote == "upstream" else "fork-tip"
            return MagicMock(returncode=0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    with (
        patch.object(banner, "_git_stdout", side_effect=fake_git_stdout),
        patch.object(banner.subprocess, "run", side_effect=fake_run),
    ):
        result = banner._check_via_local_git(repo_dir)

    assert result == banner.UPDATE_AVAILABLE_NO_COUNT
    assert fetches, "the update check must fetch a remote"
    assert all("upstream" in cmd for cmd in fetches)
    assert not any("origin" in cmd for cmd in fetches)


def test_check_for_updates_ignores_cache_from_old_checker(tmp_path, monkeypatch):
    """The old fork-based result must not survive the upstream-checker fix."""
    import hermes_cli.banner as banner
    from hermes_cli import __version__

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".update_check").write_text(
        json.dumps({"ts": time.time(), "behind": 0, "rev": None, "ver": __version__})
    )

    with patch.object(banner, "_check_via_local_git", return_value=-1) as check:
        result = banner.check_for_updates()

    assert result == -1
    check.assert_called_once()


def test_local_git_update_check_keeps_unknown_count_for_official_ssh(tmp_path):
    """An official SSH checkout must not turn unknown status into "1 behind"."""
    import hermes_cli.banner as banner

    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    def fake_git_stdout(args, *, cwd, timeout=5):
        if tuple(args) == ("remote", "get-url", "origin"):
            return "git@github.com:nousresearch/hermes-agent.git"
        if tuple(args) == ("rev-parse", "HEAD"):
            return "local-tip"
        return None

    with (
        patch.object(banner, "_git_stdout", side_effect=fake_git_stdout),
        patch.object(banner, "_check_via_rev", return_value=banner.UPDATE_AVAILABLE_NO_COUNT),
    ):
        result = banner._check_via_local_git(repo_dir)

    assert result == banner.UPDATE_AVAILABLE_NO_COUNT




