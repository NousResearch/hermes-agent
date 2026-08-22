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


def test_compare_ref_prefers_only_official_upstream(tmp_path):
    import hermes_cli.banner as banner

    origin = "https://github.com/example/hermes-agent.git"

    def official_stdout(args, **_kwargs):
        if args[-1] == "upstream":
            return "https://github.com/NousResearch/hermes-agent.git"
        raise AssertionError(args)

    with patch.object(banner, "_git_stdout", side_effect=official_stdout):
        assert banner._resolve_local_git_compare_ref(tmp_path, origin) == (
            "upstream",
            "upstream/main",
            "https://github.com/NousResearch/hermes-agent.git",
        )

    with patch.object(
        banner,
        "_git_stdout",
        return_value="https://github.com/other/hermes-agent.git",
    ):
        assert banner._resolve_local_git_compare_ref(tmp_path, origin) == (
            "origin",
            "origin/main",
            origin,
        )


def test_compare_ref_keeps_official_origin(tmp_path):
    import hermes_cli.banner as banner

    origin = "git@github.com:NousResearch/hermes-agent.git"
    with patch.object(banner, "_git_stdout") as git_stdout:
        assert banner._resolve_local_git_compare_ref(tmp_path, origin) == (
            "origin",
            "origin/main",
            origin,
        )
    git_stdout.assert_not_called()
