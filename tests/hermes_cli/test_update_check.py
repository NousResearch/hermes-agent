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


def test_check_for_updates_stable_tags_returns_no_count_and_context(tmp_path, monkeypatch):
    """Stable-tag mode reports release availability without branch commit counts."""
    import hermes_cli.banner as banner
    import hermes_cli.stable_update as stable_update

    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        banner,
        "_load_update_check_settings",
        lambda: (
            "stable-tags",
            {
                "pattern": "v20*",
                "remote": "origin",
                "command": "stable-update switch",
            },
        ),
    )
    monkeypatch.setattr(banner, "_resolve_repo_dir", lambda: repo_dir)
    monkeypatch.setattr(
        stable_update,
        "stable_update_status",
        lambda *args, **kwargs: {
            "mode": "stable-tags",
            "current_tag": "v2026.5.7",
            "latest_tag": "v2026.5.16",
            "target_tag": "v2026.5.16",
            "up_to_date": False,
            "update_available": True,
            "error": None,
        },
    )

    banner._update_context = {}
    result = banner.check_for_updates()

    assert result == banner.UPDATE_AVAILABLE_NO_COUNT
    assert banner.get_update_context()["target_tag"] == "v2026.5.16"
    assert banner.get_update_context()["update_command"] == "stable-update switch"
