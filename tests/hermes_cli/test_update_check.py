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


def test_check_for_updates_invalidates_on_version_change(tmp_path, monkeypatch):
    """A fresh cache from a different installed version must be re-checked, not reused.

    Regression for #34491: after `pip install --upgrade`, VERSION changes but the
    cache's 6h TTL hadn't expired and rev was unchanged (both None), so the stale
    'behind' count survived the upgrade. The version guard forces a recheck.
    """
    import hermes_cli.banner as banner

    # No local git checkout -> the PyPI path is exercised (pip-install class).
    fake_banner = tmp_path / "hermes_cli" / "banner.py"
    fake_banner.parent.mkdir(parents=True, exist_ok=True)
    fake_banner.touch()
    monkeypatch.setattr(banner, "__file__", str(fake_banner))

    # Fresh (within TTL) cache that says "behind", but stamped with an OLD version.
    cache_file = tmp_path / ".update_check"
    cache_file.write_text(
        json.dumps({"ts": time.time(), "behind": 1, "rev": None, "ver": "0.0.1-old"})
    )

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_REVISION", raising=False)
    with patch("hermes_cli.banner.subprocess.run") as mock_run, \
         patch("hermes_cli.banner.check_via_pypi", return_value=0) as mock_pypi:
        result = banner.check_for_updates()

    # Stale-version cache rejected -> fresh check ran -> up-to-date result.
    assert result == 0
    mock_pypi.assert_called_once()
    mock_run.assert_not_called()

    # Cache rewritten with the current installed version.
    written = json.loads(cache_file.read_text())
    assert written["ver"] == banner.VERSION


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
    # tracking ref + remote probe + is-shallow probe + git fetch + git rev-list
    assert mock_run.call_count == 5


def test_check_for_updates_official_ssh_origin_uses_https_probe(tmp_path):
    """Passive update checks must not trigger SSH auth for official installs."""
    import hermes_cli.banner as banner

    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd == ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"]:
            return MagicMock(returncode=0, stdout="origin/main\n")
        if cmd == ["git", "remote", "get-url", "origin"]:
            return MagicMock(returncode=0, stdout="git@github.com:NousResearch/hermes-agent.git\n")
        if cmd == ["git", "rev-parse", "HEAD"]:
            return MagicMock(returncode=0, stdout="local-sha\n")
        if cmd == [
            "git",
            "ls-remote",
            "https://github.com/NousResearch/hermes-agent.git",
            "refs/heads/main",
        ]:
            return MagicMock(returncode=0, stdout="upstream-sha\trefs/heads/main\n")
        raise AssertionError(f"unexpected git command: {cmd!r}")

    with patch("hermes_cli.banner.subprocess.run", side_effect=fake_run):
        result = banner._check_via_local_git(repo_dir)

    assert result == 1
    assert ["git", "fetch", "origin", "--quiet"] not in calls


def test_check_via_local_git_shallow_clone_behind_reports_no_count(tmp_path):
    """Shallow installer clones must report presence-only, never a bogus count.

    On a ``git clone --depth 1`` checkout the history stops at one commit, so
    counting ``HEAD..origin/main`` across the shallow boundary yields a huge
    nonsense number (the "12492 commits behind" banner). The shallow path must
    compare tip SHAs and return UPDATE_AVAILABLE_NO_COUNT instead, and must
    never run ``git rev-list --count``.
    """
    import hermes_cli.banner as banner

    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd == ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"]:
            return MagicMock(returncode=0, stdout="origin/main\n")
        if cmd == ["git", "remote", "get-url", "origin"]:
            return MagicMock(returncode=0, stdout="https://github.com/NousResearch/hermes-agent.git\n")
        if cmd == ["git", "rev-parse", "--is-shallow-repository"]:
            return MagicMock(returncode=0, stdout="true\n")
        if cmd[:2] == ["git", "fetch"]:
            return MagicMock(returncode=0, stdout="")
        if cmd == ["git", "rev-parse", "HEAD"]:
            return MagicMock(returncode=0, stdout="local-sha\n")
        if cmd == ["git", "rev-parse", "FETCH_HEAD"]:
            return MagicMock(returncode=0, stdout="upstream-sha\n")
        if cmd[:3] == ["git", "rev-list", "--count"]:
            raise AssertionError("shallow path must not count across the boundary")
        raise AssertionError(f"unexpected git command: {cmd!r}")

    with patch("hermes_cli.banner.subprocess.run", side_effect=fake_run):
        result = banner._check_via_local_git(repo_dir)

    assert result == banner.UPDATE_AVAILABLE_NO_COUNT
    # The shallow fetch must preserve the boundary (--depth 1), not unshallow.
    assert ["git", "fetch", "origin", "--depth", "1", "--quiet"] in calls


def test_check_via_local_git_shallow_clone_up_to_date(tmp_path):
    """Shallow clone whose tip matches upstream reports up-to-date (0)."""
    import hermes_cli.banner as banner

    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    def fake_run(cmd, **kwargs):
        if cmd == ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"]:
            return MagicMock(returncode=0, stdout="origin/main\n")
        if cmd == ["git", "remote", "get-url", "origin"]:
            return MagicMock(returncode=0, stdout="https://github.com/NousResearch/hermes-agent.git\n")
        if cmd == ["git", "rev-parse", "--is-shallow-repository"]:
            return MagicMock(returncode=0, stdout="true\n")
        if cmd[:2] == ["git", "fetch"]:
            return MagicMock(returncode=0, stdout="")
        if cmd == ["git", "rev-parse", "HEAD"]:
            return MagicMock(returncode=0, stdout="same-sha\n")
        if cmd == ["git", "rev-parse", "FETCH_HEAD"]:
            return MagicMock(returncode=0, stdout="same-sha\n")
        raise AssertionError(f"unexpected git command: {cmd!r}")

    with patch("hermes_cli.banner.subprocess.run", side_effect=fake_run):
        result = banner._check_via_local_git(repo_dir)

    assert result == 0


def test_check_via_local_git_full_clone_keeps_exact_count(tmp_path):
    """Full (non-shallow) clones keep the exact rev-list count path."""
    import hermes_cli.banner as banner

    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    def fake_run(cmd, **kwargs):
        if cmd == ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"]:
            return MagicMock(returncode=0, stdout="origin/main\n")
        if cmd == ["git", "remote", "get-url", "origin"]:
            return MagicMock(returncode=0, stdout="https://github.com/NousResearch/hermes-agent.git\n")
        if cmd == ["git", "rev-parse", "--is-shallow-repository"]:
            return MagicMock(returncode=0, stdout="false\n")
        if cmd[:2] == ["git", "fetch"]:
            return MagicMock(returncode=0, stdout="")
        if cmd[:3] == ["git", "rev-list", "--count"]:
            return MagicMock(returncode=0, stdout="7\n")
        raise AssertionError(f"unexpected git command: {cmd!r}")

    with patch("hermes_cli.banner.subprocess.run", side_effect=fake_run):
        result = banner._check_via_local_git(repo_dir)

    assert result == 7


def test_check_for_updates_uses_current_tracking_ref(tmp_path, monkeypatch):
    """Fork installs should compare to their tracking remote, not origin/main."""
    from hermes_cli.banner import check_for_updates

    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    cache_file = tmp_path / ".update_check"
    cache_file.write_text(json.dumps({"ts": 0, "behind": 22}))

    calls: list[tuple[str, ...]] = []

    def fake_run(cmd, **kwargs):
        calls.append(tuple(cmd))
        if cmd[:4] == ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name"]:
            return MagicMock(returncode=0, stdout="fork/main\n")
        if cmd[:3] == ["git", "remote", "get-url"]:
            return MagicMock(returncode=0, stdout="git@github.com:OmarB97/hermes-agent.git\n")
        if cmd == ["git", "rev-parse", "--is-shallow-repository"]:
            return MagicMock(returncode=0, stdout="false\n")
        if cmd[:3] == ["git", "fetch", "fork"]:
            return MagicMock(returncode=0, stdout="")
        if cmd[:3] == ["git", "rev-list", "--count"]:
            return MagicMock(returncode=0, stdout="0\n")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with patch("hermes_cli.banner.subprocess.run", side_effect=fake_run):
        result = check_for_updates()

    assert result == 0
    assert ("git", "remote", "get-url", "fork") in calls
    assert ("git", "fetch", "fork", "--quiet") in calls
    assert ("git", "rev-list", "--count", "HEAD..fork/main") in calls
    assert ("git", "rev-list", "--count", "HEAD..origin/main") not in calls


def test_check_for_updates_no_git_dir(tmp_path, monkeypatch):
    """Falls back to PyPI check when .git directory doesn't exist anywhere."""
    import hermes_cli.banner as banner

    # Create a fake banner.py so the fallback path also has no .git
    fake_banner = tmp_path / "hermes_cli" / "banner.py"
    fake_banner.parent.mkdir(parents=True, exist_ok=True)
    fake_banner.touch()

    monkeypatch.setattr(banner, "__file__", str(fake_banner))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with patch("hermes_cli.banner.subprocess.run") as mock_run:
        with patch("hermes_cli.banner.check_via_pypi", return_value=0):
            result = banner.check_for_updates()
    assert result == 0
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


def test_check_for_updates_docker_returns_none(tmp_path, monkeypatch):
    """Inside the Docker image, check_for_updates() must short-circuit to None.

    Regression: the published image excludes .git (.dockerignore) and sets no
    HERMES_REVISION (nix-only), so without a docker guard check_for_updates()
    falls through to check_via_pypi(), whose version-mismatch flag (1) gets
    rendered by both the Rich banner and the Ink TUI badge as a phantom
    "1 commit behind" — despite there being no git repo or commit math in the
    container, and `hermes update` correctly refusing to run there. The guard
    must return None (so the > 0 render guards stay false) AND not reach the
    git/pypi probes or write a cache entry.
    """
    import hermes_cli.banner as banner

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cache_file = tmp_path / ".update_check"

    with patch("hermes_cli.config.detect_install_method", return_value="docker"), \
         patch("hermes_cli.banner.subprocess.run") as mock_run, \
         patch("hermes_cli.banner.check_via_pypi") as mock_pypi:
        result = banner.check_for_updates()

    assert result is None
    # Neither the git probe nor the PyPI probe should have run.
    mock_run.assert_not_called()
    mock_pypi.assert_not_called()
    # And no phantom "behind" count should be cached for the next 6h.
    assert not cache_file.exists()


def test_check_for_updates_non_docker_still_checks(tmp_path, monkeypatch):
    """The docker guard must NOT over-broaden: a pip install still version-checks.

    Invariant guarding against the guard firing for non-docker methods — pip
    installs legitimately reach check_via_pypi() and surface a real update.
    """
    import hermes_cli.banner as banner

    # No local git checkout -> the PyPI (pip-install) path is exercised.
    fake_banner = tmp_path / "hermes_cli" / "banner.py"
    fake_banner.parent.mkdir(parents=True, exist_ok=True)
    fake_banner.touch()
    monkeypatch.setattr(banner, "__file__", str(fake_banner))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_REVISION", raising=False)

    with patch("hermes_cli.config.detect_install_method", return_value="pip"), \
         patch("hermes_cli.banner.subprocess.run") as mock_run, \
         patch("hermes_cli.banner.check_via_pypi", return_value=1) as mock_pypi:
        result = banner.check_for_updates()

    assert result == 1
    mock_pypi.assert_called_once()
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
