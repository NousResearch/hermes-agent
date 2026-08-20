"""Tests for the update check mechanism in hermes_cli.banner."""

import json
import os
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest




def test_check_for_updates_uses_cache(tmp_path, monkeypatch):
    """A fresh cache stamped with the current HEAD skips the remote check."""
    import hermes_cli.banner as banner

    # Create a fake git repo and fresh cache
    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    cache_file = tmp_path / ".update_check"
    cache_file.write_text(json.dumps({
        "ts": time.time(),
        "behind": 3,
        "rev": "current-head",
        "ver": banner.VERSION,
    }))

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_REVISION", raising=False)
    with (
        patch("hermes_cli.banner._resolve_repo_dir", return_value=repo_dir),
        patch("hermes_cli.banner._git_stdout", return_value="current-head"),
        patch("hermes_cli.banner._check_via_local_git") as mock_git_check,
    ):
        result = banner.check_for_updates()

    assert result == 3
    mock_git_check.assert_not_called()


def test_check_for_updates_invalidates_cache_when_head_changes(tmp_path, monkeypatch):
    """A branch switch or external pull must invalidate a fresh cache."""
    import hermes_cli.banner as banner

    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()
    cache_file = tmp_path / ".update_check"
    cache_file.write_text(json.dumps({
        "ts": time.time(),
        "behind": 4,
        "rev": "old-head",
        "ver": banner.VERSION,
    }))

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_REVISION", raising=False)
    with (
        patch("hermes_cli.banner._resolve_repo_dir", return_value=repo_dir),
        patch("hermes_cli.banner._git_stdout", return_value="new-head"),
        patch(
            "hermes_cli.banner._check_via_local_git", return_value=0
        ) as mock_git_check,
    ):
        result = banner.check_for_updates()

    assert result == 0
    mock_git_check.assert_called_once_with(repo_dir)
    written = json.loads(cache_file.read_text(encoding="utf-8"))
    assert written["behind"] == 0
    assert written["rev"] == "new-head"


def test_check_for_updates_does_not_trust_cache_when_head_is_unknown(
    tmp_path, monkeypatch
):
    """A checkout whose HEAD cannot be resolved has no valid cache identity."""
    import hermes_cli.banner as banner

    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()
    cache_file = tmp_path / ".update_check"
    cache_file.write_text(json.dumps({
        "ts": time.time(),
        "behind": 4,
        "rev": "previous-head",
        "ver": banner.VERSION,
    }))

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_REVISION", raising=False)
    with (
        patch("hermes_cli.banner._resolve_repo_dir", return_value=repo_dir),
        patch("hermes_cli.banner._git_stdout", return_value=None),
        patch(
            "hermes_cli.banner._check_via_local_git", return_value=0
        ) as mock_git_check,
    ):
        result = banner.check_for_updates()

    assert result == 0
    mock_git_check.assert_called_once_with(repo_dir)


def test_check_for_updates_keeps_embedded_revision_cache(tmp_path, monkeypatch):
    """Nix builds keep using HERMES_REVISION without probing a checkout."""
    import hermes_cli.banner as banner

    cache_file = tmp_path / ".update_check"
    cache_file.write_text(json.dumps({
        "ts": time.time(),
        "behind": 0,
        "rev": "embedded-revision",
        "ver": banner.VERSION,
    }))

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_REVISION", "embedded-revision")
    with (
        patch("hermes_cli.banner._resolve_repo_dir") as mock_resolve,
        patch("hermes_cli.banner._check_via_rev") as mock_rev_check,
    ):
        result = banner.check_for_updates()

    assert result == 0
    mock_resolve.assert_not_called()
    mock_rev_check.assert_not_called()


def test_check_for_updates_keeps_cache_without_checkout(tmp_path, monkeypatch):
    """A packaged install with no git identity keeps its time-based cache."""
    import hermes_cli.banner as banner

    cache_file = tmp_path / ".update_check"
    cache_file.write_text(json.dumps({
        "ts": time.time(),
        "behind": None,
        "rev": None,
        "ver": banner.VERSION,
    }))

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_REVISION", raising=False)
    with (
        patch("hermes_cli.banner._resolve_repo_dir", return_value=None),
        patch("hermes_cli.banner._git_stdout") as mock_git_stdout,
        patch("hermes_cli.banner._check_via_local_git") as mock_git_check,
    ):
        result = banner.check_for_updates()

    assert result is None
    mock_git_stdout.assert_not_called()
    mock_git_check.assert_not_called()






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


