"""Tests for the update check mechanism in hermes_cli.banner."""

import json
import os
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest




def test_check_for_updates_uses_cache(tmp_path, monkeypatch):
    """When cache is fresh and HEAD is unchanged, return the cached value without fetching.

    A fresh cache for a source install still runs one local `git rev-parse HEAD`
    (to confirm HEAD hasn't moved, see #40944) but must NOT run the network
    `git fetch` / `git rev-list` recheck.
    """
    from hermes_cli.banner import check_for_updates
    from hermes_cli import __version__

    # Create a fake git repo and fresh cache stamped with the current HEAD.
    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    cache_file = tmp_path / ".update_check"
    cache_file.write_text(json.dumps(
        {"ts": time.time(), "behind": 3, "rev": None, "ver": __version__, "head": "cafef00d"}
    ))

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_REVISION", raising=False)
    # `git rev-parse HEAD` reports the same hash the cache was stamped with.
    mock_result = MagicMock(returncode=0, stdout="cafef00d\n")
    with patch("hermes_cli.banner.subprocess.run", return_value=mock_result) as mock_run:
        result = check_for_updates()

    assert result == 3
    assert mock_run.call_count == 1  # only rev-parse HEAD, no fetch/rev-list


def test_check_for_updates_invalidates_on_head_change(tmp_path, monkeypatch):
    """A fresh cache from a different local HEAD must be re-checked, not reused.

    Regression for #40944: after a manual `git pull --ff-only` on a source
    install, HEAD moves but VERSION and the embedded rev are unchanged, so the
    6h-TTL cache kept reporting the stale 'behind' count. The HEAD guard forces
    a recheck, which now reports 0 commits behind.
    """
    import hermes_cli.banner as banner

    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()
    # Point _resolve_repo_dir() at our fake checkout (its __file__ preference
    # would otherwise resolve to the real repo running the tests).
    fake_banner = repo_dir / "hermes_cli" / "banner.py"
    fake_banner.parent.mkdir(parents=True, exist_ok=True)
    fake_banner.touch()
    monkeypatch.setattr(banner, "__file__", str(fake_banner))

    # Fresh (within TTL) cache that says "behind 81", stamped with the OLD HEAD.
    cache_file = tmp_path / ".update_check"
    cache_file.write_text(json.dumps(
        {"ts": time.time(), "behind": 81, "rev": None, "ver": banner.VERSION, "head": "0ldc0mmit"}
    ))

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_REVISION", raising=False)

    def fake_run(cmd, *args, **kwargs):
        # rev-parse reports the NEW post-pull HEAD; rev-list reports 0 behind.
        if "rev-parse" in cmd:
            return MagicMock(returncode=0, stdout="new00000\n")
        if "rev-list" in cmd:
            return MagicMock(returncode=0, stdout="0\n")
        return MagicMock(returncode=0, stdout="")  # git fetch

    with patch("hermes_cli.banner.subprocess.run", side_effect=fake_run):
        result = banner.check_for_updates()

    # Stale-HEAD cache rejected -> fresh check ran -> up-to-date result.
    assert result == 0
    written = json.loads(cache_file.read_text())
    assert written["behind"] == 0
    assert written["head"] == "new00000"


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




