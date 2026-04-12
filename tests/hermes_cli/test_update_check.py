"""Tests for the update check mechanism in hermes_cli.banner."""

import json
import os
import sys
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

    cache_file = tmp_path / ".update_check"
    cache_file.write_text(json.dumps({"ts": time.time(), "behind": 3}))

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with patch("hermes_cli.banner.subprocess.run") as mock_run:
        result = check_for_updates()

    assert result == 3
    mock_run.assert_not_called()


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
    with patch("hermes_cli.banner.subprocess.run", return_value=mock_result) as mock_run:
        result = check_for_updates()

    assert result == 5
    assert mock_run.call_count == 2  # git fetch + git rev-list


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


def test_get_update_result_timeout():
    """get_update_result() returns None when check hasn't completed within timeout."""
    import hermes_cli.banner as banner

    # Reset module state — don't set the event
    banner._update_result = None
    banner._update_check_done = threading.Event()

    start = time.monotonic()
    result = banner.get_update_result(timeout=0.1)
    elapsed = time.monotonic() - start

    # Should have waited ~0.1s and returned None
    assert result is None
    assert elapsed < 0.5


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


def test_startup_update_prompt_accepts_and_reexecs(monkeypatch):
    import hermes_cli.main as main

    monkeypatch.delenv("HERMES_AUTO_UPDATE_REEXECED", raising=False)
    monkeypatch.setattr(sys, "argv", ["hermes", "chat"])
    monkeypatch.setattr(main, "__file__", "/tmp/hermes_cli/main.py")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)

    branch = MagicMock(returncode=0, stdout="main\n")
    clean = MagicMock(returncode=0, stdout="")

    with patch("hermes_cli.config.is_managed", return_value=False), \
         patch("hermes_cli.config.load_config", return_value={"startup": {"update_on_launch": "ask"}}), \
         patch("hermes_cli.banner.check_for_updates", return_value=2), \
         patch("hermes_cli.main.subprocess.run", side_effect=[branch, clean]) as mock_run, \
         patch("builtins.input", return_value="y"), \
         patch.object(main, "cmd_update") as mock_update, \
         patch("hermes_cli.main.os.execv") as mock_execv:
        main._maybe_auto_update_before_chat_launch()

    mock_update.assert_called_once()
    update_args = mock_update.call_args.args[0]
    assert update_args.gateway is False
    assert update_args.auto_startup is True
    mock_execv.assert_called_once_with(
        sys.executable,
        [sys.executable, str(Path("/tmp/hermes_cli/main.py").resolve()), "chat"],
    )
    assert os.environ["HERMES_AUTO_UPDATE_REEXECED"] == "1"
    assert mock_run.call_count == 2


def test_startup_update_prompt_decline_skips_update(monkeypatch):
    import hermes_cli.main as main

    monkeypatch.delenv("HERMES_AUTO_UPDATE_REEXECED", raising=False)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)

    branch = MagicMock(returncode=0, stdout="main\n")
    clean = MagicMock(returncode=0, stdout="")

    with patch("hermes_cli.config.is_managed", return_value=False), \
         patch("hermes_cli.config.load_config", return_value={"startup": {"update_on_launch": "ask"}}), \
         patch("hermes_cli.banner.check_for_updates", return_value=2), \
         patch("hermes_cli.main.subprocess.run", side_effect=[branch, clean]), \
         patch("builtins.input", return_value="n"), \
         patch.object(main, "cmd_update") as mock_update, \
         patch("hermes_cli.main.os.execv") as mock_execv:
        main._maybe_auto_update_before_chat_launch()

    mock_update.assert_not_called()
    mock_execv.assert_not_called()


def test_startup_auto_policy_runs_update_and_reexec(monkeypatch):
    import hermes_cli.main as main

    monkeypatch.delenv("HERMES_AUTO_UPDATE_REEXECED", raising=False)
    monkeypatch.setattr(sys, "argv", ["hermes", "chat"])
    monkeypatch.setattr(main, "__file__", "/tmp/hermes_cli/main.py")

    branch = MagicMock(returncode=0, stdout="main\n")
    clean = MagicMock(returncode=0, stdout="")

    with patch("hermes_cli.config.is_managed", return_value=False), \
         patch("hermes_cli.config.load_config", return_value={"startup": {"update_on_launch": "auto"}}), \
         patch("hermes_cli.banner.check_for_updates", return_value=2), \
         patch("hermes_cli.main.subprocess.run", side_effect=[branch, clean]) as mock_run, \
         patch.object(main, "cmd_update") as mock_update, \
         patch("hermes_cli.main.os.execv") as mock_execv:
        main._maybe_auto_update_before_chat_launch()

    mock_update.assert_called_once()
    update_args = mock_update.call_args.args[0]
    assert update_args.gateway is False
    assert update_args.auto_startup is True
    mock_execv.assert_called_once_with(
        sys.executable,
        [sys.executable, str(Path("/tmp/hermes_cli/main.py").resolve()), "chat"],
    )
    assert os.environ["HERMES_AUTO_UPDATE_REEXECED"] == "1"
    assert mock_run.call_count == 2


def test_startup_update_off_skips_update(monkeypatch):
    import hermes_cli.main as main

    monkeypatch.delenv("HERMES_AUTO_UPDATE_REEXECED", raising=False)

    with patch("hermes_cli.config.is_managed", return_value=False), \
         patch("hermes_cli.config.load_config", return_value={"startup": {"update_on_launch": "off"}}), \
         patch("hermes_cli.banner.check_for_updates") as mock_check, \
         patch.object(main, "cmd_update") as mock_update, \
         patch("hermes_cli.main.os.execv") as mock_execv:
        main._maybe_auto_update_before_chat_launch()

    mock_check.assert_not_called()
    mock_update.assert_not_called()
    mock_execv.assert_not_called()


def test_startup_update_legacy_true_maps_to_auto(monkeypatch):
    import hermes_cli.main as main

    monkeypatch.delenv("HERMES_AUTO_UPDATE_REEXECED", raising=False)
    monkeypatch.setattr(sys, "argv", ["hermes", "chat"])
    monkeypatch.setattr(main, "__file__", "/tmp/hermes_cli/main.py")

    branch = MagicMock(returncode=0, stdout="main\n")
    clean = MagicMock(returncode=0, stdout="")

    with patch("hermes_cli.config.is_managed", return_value=False), \
         patch("hermes_cli.config.load_config", return_value={"startup": {"auto_update_on_launch": True}}), \
         patch("hermes_cli.banner.check_for_updates", return_value=1), \
         patch("hermes_cli.main.subprocess.run", side_effect=[branch, clean]), \
         patch.object(main, "cmd_update") as mock_update, \
         patch("hermes_cli.main.os.execv") as mock_execv:
        main._maybe_auto_update_before_chat_launch()

    mock_update.assert_called_once()
    mock_execv.assert_called_once()


def test_startup_update_legacy_false_maps_to_off(monkeypatch):
    import hermes_cli.main as main

    monkeypatch.delenv("HERMES_AUTO_UPDATE_REEXECED", raising=False)

    with patch("hermes_cli.config.is_managed", return_value=False), \
         patch("hermes_cli.config.load_config", return_value={"startup": {"auto_update_on_launch": False}}), \
         patch("hermes_cli.banner.check_for_updates") as mock_check, \
         patch.object(main, "cmd_update") as mock_update, \
         patch("hermes_cli.main.os.execv") as mock_execv:
        main._maybe_auto_update_before_chat_launch()

    mock_check.assert_not_called()
    mock_update.assert_not_called()
    mock_execv.assert_not_called()


def test_startup_update_skips_dirty_repo(monkeypatch):
    import hermes_cli.main as main

    monkeypatch.delenv("HERMES_AUTO_UPDATE_REEXECED", raising=False)

    branch = MagicMock(returncode=0, stdout="main\n")
    dirty = MagicMock(returncode=0, stdout=" M cli.py\n")

    with patch("hermes_cli.config.is_managed", return_value=False), \
         patch("hermes_cli.config.load_config", return_value={"startup": {"update_on_launch": "auto"}}), \
         patch("hermes_cli.banner.check_for_updates", return_value=3), \
         patch("hermes_cli.main.subprocess.run", side_effect=[branch, dirty]), \
         patch.object(main, "cmd_update") as mock_update, \
         patch("hermes_cli.main.os.execv") as mock_execv:
        main._maybe_auto_update_before_chat_launch()

    mock_update.assert_not_called()
    mock_execv.assert_not_called()
