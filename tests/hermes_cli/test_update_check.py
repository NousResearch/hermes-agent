"""Tests for the update check mechanism in hermes_cli.banner."""

import json
import os
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_version_string_no_v_prefix():
    """__version__ should be bare semver without a 'v' prefix."""
    from hermes_cli import __version__
    assert not __version__.startswith("v"), f"__version__ should not start with 'v', got {__version__!r}"


def test_check_for_updates_uses_cache(tmp_path, monkeypatch):
    """When cache is fresh, check_for_updates should return cached value without calling git."""
    from hermes_cli.banner import check_for_updates

    # Create a fake git repo and fresh cache
    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    cache_rev = f"git:{repo_dir}:abc123"
    cache_file = tmp_path / ".update_check"
    cache_file.write_text(json.dumps({"ts": time.time(), "behind": 3, "rev": cache_rev}))

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with patch("hermes_cli.banner._local_git_cache_rev", return_value=cache_rev), \
         patch("hermes_cli.banner.subprocess.run") as mock_run:
        result = check_for_updates()

    assert result == 3
    mock_run.assert_not_called()


def test_check_for_updates_expired_cache(tmp_path, monkeypatch):
    """When cache is expired, check_for_updates should call git fetch."""
    from hermes_cli.banner import check_for_updates

    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    cache_rev = f"git:{repo_dir}:abc123"

    # Write an expired cache (timestamp far in the past)
    cache_file = tmp_path / ".update_check"
    cache_file.write_text(json.dumps({"ts": 0, "behind": 1, "rev": cache_rev}))

    mock_result = MagicMock(returncode=0, stdout="5\n")

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with patch("hermes_cli.banner._local_git_cache_rev", return_value=cache_rev), \
         patch("hermes_cli.banner.subprocess.run", return_value=mock_result) as mock_run:
        result = check_for_updates()

    assert result == 5
    assert mock_run.call_count == 2  # git fetch + git rev-list


def test_check_for_updates_invalidates_stale_local_cache(tmp_path, monkeypatch):
    """Fresh local-git caches without the current HEAD must not be trusted."""
    from hermes_cli.banner import check_for_updates

    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    # This is the historical stale cache shape that caused `hermes --version`
    # to keep reporting a false commits-behind notice after the checkout updated.
    cache_file = tmp_path / ".update_check"
    cache_file.write_text(json.dumps({"ts": time.time(), "behind": 174}))

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with patch("hermes_cli.banner._local_git_cache_rev", return_value=f"git:{repo_dir}:current"), \
         patch("hermes_cli.banner.subprocess.run", return_value=MagicMock(returncode=0, stdout="0\n")) as mock_run:
        result = check_for_updates()

    assert result == 0
    assert mock_run.call_count == 2  # git fetch + git rev-list
    cached = json.loads(cache_file.read_text())
    assert cached["behind"] == 0
    assert cached["rev"] == f"git:{repo_dir}:current"


def test_check_for_updates_invalidates_when_upstream_ref_changes(tmp_path, monkeypatch):
    """Fresh cache with same HEAD but old origin/main must not be trusted."""
    from hermes_cli.banner import check_for_updates

    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    old_cache_rev = f"git:{repo_dir}:HEAD=current:origin/main=old-upstream"
    new_cache_rev = f"git:{repo_dir}:HEAD=current:origin/main=new-upstream"
    cache_file = tmp_path / ".update_check"
    cache_file.write_text(json.dumps({"ts": time.time(), "behind": 0, "rev": old_cache_rev}))

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with patch("hermes_cli.banner._local_git_cache_rev", return_value=new_cache_rev), \
         patch("hermes_cli.banner.subprocess.run", return_value=MagicMock(returncode=0, stdout="20\n")) as mock_run:
        result = check_for_updates()

    assert result == 20
    assert mock_run.call_count == 2  # git fetch + git rev-list
    cached = json.loads(cache_file.read_text())
    assert cached["behind"] == 20
    assert cached["rev"] == new_cache_rev


def test_check_for_updates_no_git_dir(tmp_path, monkeypatch):
    """Returns None when .git directory doesn't exist anywhere."""
    import hermes_cli.banner as banner

    # Create a fake banner.py so the fallback path also has no .git
    fake_banner = tmp_path / "hermes_cli" / "banner.py"
    fake_banner.parent.mkdir(parents=True, exist_ok=True)
    fake_banner.touch()

    monkeypatch.setattr(banner, "__file__", str(fake_banner))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with patch("hermes_cli.banner.subprocess.run") as mock_run:
        result = banner.check_for_updates()
    assert result is None
    mock_run.assert_not_called()


def test_check_for_updates_fallback_to_project_root(tmp_path, monkeypatch):
    """Dev install: falls back to Path(__file__).parent.parent when HERMES_HOME has no git repo."""
    import hermes_cli.banner as banner

    project_root = Path(banner.__file__).parent.parent.resolve()
    if not (project_root / ".git").exists():
        pytest.skip("Not running from a git checkout")

    # Point HERMES_HOME at a temp dir with no hermes-agent/.git
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with patch("hermes_cli.banner.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="0\n")
        result = banner.check_for_updates()
    # Should have fallen back to project root and run git commands
    assert mock_run.call_count >= 1


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


def test_invalidate_update_cache_clears_all_profiles(tmp_path):
    """_invalidate_update_cache() should delete .update_check from ALL profiles."""
    from hermes_cli.main import _invalidate_update_cache

    # Build a fake ~/.hermes with default + two named profiles
    default_home = tmp_path / ".hermes"
    default_home.mkdir()
    (default_home / ".update_check").write_text('{"ts":1,"behind":50}')

    profiles_root = default_home / "profiles"
    for name in ("ops", "dev"):
        p = profiles_root / name
        p.mkdir(parents=True)
        (p / ".update_check").write_text('{"ts":1,"behind":50}')

    with patch.object(Path, "home", return_value=tmp_path), \
         patch.dict(os.environ, {"HERMES_HOME": str(default_home)}):
        _invalidate_update_cache()

    # All three caches should be gone
    assert not (default_home / ".update_check").exists(), "default profile cache not cleared"
    assert not (profiles_root / "ops" / ".update_check").exists(), "ops profile cache not cleared"
    assert not (profiles_root / "dev" / ".update_check").exists(), "dev profile cache not cleared"


def test_invalidate_update_cache_no_profiles_dir(tmp_path):
    """Works fine when no profiles directory exists (single-profile setup)."""
    from hermes_cli.main import _invalidate_update_cache

    default_home = tmp_path / ".hermes"
    default_home.mkdir()
    (default_home / ".update_check").write_text('{"ts":1,"behind":5}')

    with patch.object(Path, "home", return_value=tmp_path), \
         patch.dict(os.environ, {"HERMES_HOME": str(default_home)}):
        _invalidate_update_cache()

    assert not (default_home / ".update_check").exists()
