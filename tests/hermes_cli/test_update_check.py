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


def test_format_update_notice_diverged_is_not_a_tip_count():
    """UPDATE_DIVERGED must not render as 'N commits behind' or generic available."""
    from hermes_cli.banner import UPDATE_DIVERGED, _format_update_notice

    line = _format_update_notice(UPDATE_DIVERGED)
    assert "diverged" in line.lower()
    assert "commits behind" not in line
    assert "commit behind" not in line


def test_format_update_notice_positive_behind_still_counts():
    from hermes_cli.banner import _format_update_notice

    line = _format_update_notice(3)
    assert "3 commits behind" in line


def test_fast_version_diverged_does_not_print_up_to_date(capsys, monkeypatch):
    """hermes --version must not imply a fast-forward when the tree diverged."""
    from hermes_cli.banner import UPDATE_DIVERGED
    from hermes_cli._startup_fast import print_fast_version_info

    monkeypatch.setattr(
        "hermes_cli.banner.check_for_updates", lambda: UPDATE_DIVERGED
    )
    print_fast_version_info(check_updates=True)
    out = capsys.readouterr().out.lower()
    assert "diverged" in out
    assert "up to date" not in out


def test_diverged_with_positive_behind_returns_diverged(tmp_path, monkeypatch):
    """behind>0 without HEAD ancestor of origin/main => UPDATE_DIVERGED."""
    from unittest.mock import MagicMock

    from hermes_cli.banner import UPDATE_DIVERGED, _check_via_local_git

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()

    def fake_run(cmd, **kwargs):
        cmd = list(cmd)
        if cmd[:3] == ["git", "rev-list", "--count"]:
            return MagicMock(returncode=0, stdout="1\n")
        if cmd[:3] == ["git", "merge-base", "--is-ancestor"]:
            return MagicMock(returncode=1, stdout="")
        if cmd[:2] == ["git", "fetch"]:
            return MagicMock(returncode=0, stdout="")
        return MagicMock(returncode=0, stdout="")

    monkeypatch.setattr("hermes_cli.banner.subprocess.run", fake_run)
    monkeypatch.setattr(
        "hermes_cli.banner._git_stdout",
        lambda args, cwd=None: (
            "https://github.com/NousResearch/hermes-agent.git"
            if list(args)[:2] == ["remote", "get-url"]
            else (
                "false"
                if list(args)[:2] == ["rev-parse", "--is-shallow-repository"]
                else ""
            )
        ),
    )
    assert _check_via_local_git(repo) == UPDATE_DIVERGED


def test_diverged_with_zero_behind_returns_diverged(tmp_path, monkeypatch):
    """behind==0 can still be diverged when neither tip is ancestor."""
    from unittest.mock import MagicMock

    from hermes_cli.banner import UPDATE_DIVERGED, _check_via_local_git

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()

    def fake_run(cmd, **kwargs):
        cmd = list(cmd)
        if cmd[:3] == ["git", "rev-list", "--count"]:
            return MagicMock(returncode=0, stdout="0\n")
        if cmd[:3] == ["git", "merge-base", "--is-ancestor"]:
            return MagicMock(returncode=1, stdout="")
        if cmd[:2] == ["git", "fetch"]:
            return MagicMock(returncode=0, stdout="")
        return MagicMock(returncode=0, stdout="")

    monkeypatch.setattr("hermes_cli.banner.subprocess.run", fake_run)
    monkeypatch.setattr(
        "hermes_cli.banner._git_stdout",
        lambda args, cwd=None: (
            "https://github.com/NousResearch/hermes-agent.git"
            if list(args)[:2] == ["remote", "get-url"]
            else (
                "false"
                if list(args)[:2] == ["rev-parse", "--is-shallow-repository"]
                else ""
            )
        ),
    )
    assert _check_via_local_git(repo) == UPDATE_DIVERGED

