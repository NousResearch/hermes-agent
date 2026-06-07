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

    assert not __version__.startswith(
        "v"
    ), f"__version__ should not start with 'v', got {__version__!r}"


def test_check_for_updates_uses_cache(tmp_path, monkeypatch):
    """When cache is fresh and HEAD matches, check_for_updates should return cached value.

    This prevents stale "behind" results after a local fast-forward update that would
    otherwise require another full git check within the TTL window.
    """
    from hermes_cli.banner import check_for_updates

    # Create a fake git repo and fresh cache with a HEAD fingerprint
    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    cache_file = tmp_path / ".update_check"
    cache_file.write_text(
        json.dumps({"ts": time.time(), "behind": 3, "head": "abc12345"})
    )

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with patch("hermes_cli.banner.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="abc12345\n")
        result = check_for_updates()

    assert result == 3
    assert mock_run.call_count == 1
    assert mock_run.call_args_list[0].args[0][:3] == ["git", "rev-parse", "--short=8"]


def test_check_for_updates_refreshes_cache_when_head_changes(tmp_path, monkeypatch):
    """If cached HEAD doesn't match current HEAD, check_for_updates refreshes.

    This handles fast-forward git pulls where package version stays unchanged but
    cached update status should be invalidated.
    """
    from hermes_cli.banner import check_for_updates

    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    cache_file = tmp_path / ".update_check"
    cache_file.write_text(
        json.dumps({"ts": time.time(), "behind": 3, "head": "oldhash1"})
    )

    mock_results = [
        MagicMock(returncode=0, stdout="newhash2\n"),  # current HEAD mismatch
        MagicMock(returncode=0, stdout=""),  # git fetch
        MagicMock(returncode=0, stdout="5\n"),  # git rev-list
    ]

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with patch(
        "hermes_cli.banner.subprocess.run", side_effect=mock_results
    ) as mock_run:
        result = check_for_updates()

    assert result == 5
    assert mock_run.call_count == 3  # git rev-parse HEAD + git fetch + git rev-list
    # cache was rewritten with the current HEAD hash
    reloaded = json.loads(cache_file.read_text())
    assert reloaded.get("head") == "newhash2"


def test_check_for_updates_uses_legacy_cache_when_revision_present(
    tmp_path, monkeypatch
):
    """Legacy cache with an explicit embedded revision should still be used."""
    from hermes_cli.banner import check_for_updates

    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    cache_file = tmp_path / ".update_check"
    cache_file.write_text(
        json.dumps({"ts": time.time(), "behind": 7, "rev": "some-embedded-rev"})
    )

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with patch("hermes_cli.banner.subprocess.run") as mock_run:
        result = check_for_updates()

    assert result == 7
    assert mock_run.call_count == 0


def test_check_for_updates_refreshes_legacy_cache_when_revision_missing(
    tmp_path, monkeypatch
):
    """Legacy cache with unknown embedded revision (None) should be invalidated."""
    from hermes_cli.banner import check_for_updates

    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    cache_file = tmp_path / ".update_check"
    cache_file.write_text(
        json.dumps({"ts": time.time(), "behind": 7, "rev": None, "head": None})
    )

    mock_results = [
        MagicMock(returncode=0, stdout="newhash2\n"),  # current HEAD
        MagicMock(returncode=0, stdout=""),  # git fetch
        MagicMock(returncode=0, stdout="6\n"),  # git rev-list
    ]

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with patch(
        "hermes_cli.banner.subprocess.run", side_effect=mock_results
    ) as mock_run:
        result = check_for_updates()

    assert result == 6
    assert mock_run.call_count == 3


def test_check_for_updates_expired_cache(tmp_path, monkeypatch):
    """When cache is expired, check_for_updates should call git fetch."""
    from hermes_cli.banner import check_for_updates

    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    # Write an expired cache (timestamp far in the past)
    cache_file = tmp_path / ".update_check"
    cache_file.write_text(json.dumps({"ts": 0, "behind": 1}))

    mock_result = MagicMock(returncode=0, stdout="5\n")

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with patch(
        "hermes_cli.banner.subprocess.run", return_value=mock_result
    ) as mock_run:
        result = check_for_updates()

    assert result == 5
    assert mock_run.call_count == 3  # git rev-parse HEAD + git fetch + git rev-list


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

    with patch.object(Path, "home", return_value=tmp_path), patch.dict(
        os.environ, {"HERMES_HOME": str(default_home)}
    ):
        _invalidate_update_cache()

    # All three caches should be gone
    assert not (
        default_home / ".update_check"
    ).exists(), "default profile cache not cleared"
    assert not (
        profiles_root / "ops" / ".update_check"
    ).exists(), "ops profile cache not cleared"
    assert not (
        profiles_root / "dev" / ".update_check"
    ).exists(), "dev profile cache not cleared"


def test_invalidate_update_cache_no_profiles_dir(tmp_path):
    """Works fine when no profiles directory exists (single-profile setup)."""
    from hermes_cli.main import _invalidate_update_cache

    default_home = tmp_path / ".hermes"
    default_home.mkdir()
    (default_home / ".update_check").write_text('{"ts":1,"behind":5}')

    with patch.object(Path, "home", return_value=tmp_path), patch.dict(
        os.environ, {"HERMES_HOME": str(default_home)}
    ):
        _invalidate_update_cache()

    assert not (default_home / ".update_check").exists()
