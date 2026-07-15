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
    """When cache is fresh and heads match, check_for_updates returns cached value.

    The cache-hit path now probes local HEAD and origin/main to detect stale
    cache entries (cheap rev-parse, no network). Those two calls are expected.
    No fetch, ls-remote, rev-list, or pypi should happen.
    """
    from hermes_cli.banner import check_for_updates
    from hermes_cli import __version__

    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    fake_banner = repo_dir / "hermes_cli" / "banner.py"
    fake_banner.parent.mkdir(parents=True, exist_ok=True)
    fake_banner.touch()

    import hermes_cli.banner as banner
    monkeypatch.setattr(banner, "__file__", str(fake_banner))
    monkeypatch.delenv("HERMES_REVISION", raising=False)

    # Cache with matching heads — should be accepted
    cache_file = tmp_path / ".update_check"
    cache_file.write_text(json.dumps({
        "ts": time.time(),
        "behind": 3,
        "rev": None,
        "ver": __version__,
        "upstream_head": "live-upstream",
        "local_head": "live-local",
    }))

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    rev_parse_calls = 0

    def fake_run(cmd, **kwargs):
        nonlocal rev_parse_calls
        if cmd == ["git", "rev-parse", "HEAD"]:
            rev_parse_calls += 1
            return MagicMock(returncode=0, stdout="live-local\n")
        if cmd == ["git", "rev-parse", "origin/main"]:
            rev_parse_calls += 1
            return MagicMock(returncode=0, stdout="live-upstream\n")
        raise AssertionError(f"unexpected network command: {cmd}")

    with patch("hermes_cli.banner.subprocess.run", side_effect=fake_run), \
         patch("hermes_cli.banner.check_via_pypi") as mock_pypi:
        result = check_for_updates()

    assert result == 3
    # Two rev-parse probes are fine; no network operations
    assert rev_parse_calls == 2
    mock_pypi.assert_not_called()


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
    # remote get-url + is-shallow + fetch + rev-list + 2× rev-parse (cache write)
    assert mock_run.call_count == 6


def test_check_for_updates_official_ssh_origin_uses_https_probe(tmp_path):
    """Passive update checks must not trigger SSH auth for official installs."""
    import hermes_cli.banner as banner

    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
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


# =========================================================================
# Head-movement cache invalidation tests
# =========================================================================


def test_check_for_updates_cache_invalidates_on_upstream_head_movement(tmp_path, monkeypatch):
    """Cache hit with stale upstream_head must be rejected and re-checked.

    When upstream has advanced (``git fetch`` landed new commits) but
    VERSION and rev are unchanged, the old cache payload's ``upstream_head``
    no longer matches the live ``origin/main`` SHA. The probe must detect
    this, reject the cache, and run a fresh check. Without this guard the
    banner and ``hermes version`` would report "Up to date" for up to the
    TTL window despite real upstream movement.
    """
    import hermes_cli.banner as banner
    from hermes_cli import __version__

    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    # Set __file__ to point inside repo_dir so _resolve_repo_dir finds it
    fake_banner = repo_dir / "hermes_cli" / "banner.py"
    fake_banner.parent.mkdir(parents=True, exist_ok=True)
    fake_banner.touch()
    monkeypatch.setattr(banner, "__file__", str(fake_banner))

    # Populate a cache with OLD upstream_head and local_head
    cache_file = tmp_path / ".update_check"
    cache_file.write_text(
        json.dumps({
            "ts": time.time(),
            "behind": 0,
            "rev": None,
            "ver": __version__,
            "upstream_head": "old-upstream-sha",
            "local_head": "local-sha",
        })
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_REVISION", raising=False)

    call_log = []

    def fake_run(cmd, **kwargs):
        call_log.append(cmd)
        if cmd[:2] == ["git", "rev-parse"] and "--is-shallow-repository" in cmd:
            return MagicMock(returncode=0, stdout="false\n")
        if cmd == ["git", "remote", "get-url", "origin"]:
            return MagicMock(returncode=0, stdout="https://github.com/NousResearch/hermes-agent.git\n")
        if cmd[:2] == ["git", "fetch"]:
            return MagicMock(returncode=0, stdout="")
        if cmd[:3] == ["git", "rev-list", "--count"]:
            return MagicMock(returncode=0, stdout="3\n")
        if cmd == ["git", "rev-parse", "HEAD"]:
            return MagicMock(returncode=0, stdout="local-sha\n")
        if cmd == ["git", "rev-parse", "origin/main"]:
            return MagicMock(returncode=0, stdout="new-upstream-sha\n")
        raise AssertionError(f"unexpected command: {cmd}")

    with patch("hermes_cli.banner.subprocess.run", side_effect=fake_run), \
         patch("hermes_cli.config.detect_install_method", return_value="git"):
        result = banner.check_for_updates()

    # Upstream moved → cache rejected → full check ran → 3 behind
    assert result == 3
    # Fresh check proceeded: remote get-url, is-shallow, fetch, rev-list
    assert any(c[:3] == ["git", "rev-list", "--count"] for c in call_log), (
        "should have run rev-list --count on a cache miss"
    )

    # Cache was rewritten with the new upstream_head
    written = json.loads(cache_file.read_text())
    assert written["upstream_head"] == "new-upstream-sha"
    assert written["local_head"] == "local-sha"


def test_check_for_updates_cache_invalidates_on_local_head_movement(tmp_path, monkeypatch):
    """Cache hit with stale local_head must be rejected and re-checked.

    A ``git pull`` or ``git checkout`` can change local HEAD while
    VERSION stays pinned. Without the local_head guard, the cache
    would return a stale "behind" count until the TTL expired.
    """
    import hermes_cli.banner as banner
    from hermes_cli import __version__

    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    fake_banner = repo_dir / "hermes_cli" / "banner.py"
    fake_banner.parent.mkdir(parents=True, exist_ok=True)
    fake_banner.touch()
    monkeypatch.setattr(banner, "__file__", str(fake_banner))

    # Cache has OLD local_head but current upstream_head
    cache_file = tmp_path / ".update_check"
    cache_file.write_text(
        json.dumps({
            "ts": time.time(),
            "behind": 0,
            "rev": None,
            "ver": __version__,
            "upstream_head": "upstream-sha",
            "local_head": "old-local-sha",
        })
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_REVISION", raising=False)

    def fake_run(cmd, **kwargs):
        if cmd[:2] == ["git", "rev-parse"] and "--is-shallow-repository" in cmd:
            return MagicMock(returncode=0, stdout="false\n")
        if cmd == ["git", "remote", "get-url", "origin"]:
            return MagicMock(returncode=0, stdout="https://github.com/NousResearch/hermes-agent.git\n")
        if cmd[:2] == ["git", "fetch"]:
            return MagicMock(returncode=0, stdout="")
        if cmd[:3] == ["git", "rev-list", "--count"]:
            return MagicMock(returncode=0, stdout="0\n")
        if cmd == ["git", "rev-parse", "HEAD"]:
            return MagicMock(returncode=0, stdout="new-local-sha\n")
        if cmd == ["git", "rev-parse", "origin/main"]:
            return MagicMock(returncode=0, stdout="upstream-sha\n")
        raise AssertionError(f"unexpected command: {cmd}")

    with patch("hermes_cli.banner.subprocess.run", side_effect=fake_run), \
         patch("hermes_cli.config.detect_install_method", return_value="git"):
        result = banner.check_for_updates()

    # Local HEAD moved → cache rejected → fresh check → up to date (0)
    assert result == 0

    written = json.loads(cache_file.read_text())
    assert written["local_head"] == "new-local-sha"


def test_check_for_updates_cache_honours_stable_heads(tmp_path, monkeypatch):
    """When both local and upstream heads match the cache, the cached value is returned.

    On a stable checkout (no ``git pull``, no upstream movement) the
    git rev-parse probes should match the cached values and the behind
    count should be served from cache without any network operation.
    """
    import hermes_cli.banner as banner
    from hermes_cli import __version__

    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    fake_banner = repo_dir / "hermes_cli" / "banner.py"
    fake_banner.parent.mkdir(parents=True, exist_ok=True)
    fake_banner.touch()
    monkeypatch.setattr(banner, "__file__", str(fake_banner))

    cache_file = tmp_path / ".update_check"
    cache_file.write_text(
        json.dumps({
            "ts": time.time(),
            "behind": 0,
            "rev": None,
            "ver": __version__,
            "upstream_head": "stable-upstream",
            "local_head": "stable-local",
        })
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_REVISION", raising=False)

    # Only rev-parse calls should happen — no fetch, no ls-remote, no pypi
    rev_parse_count = 0

    def fake_run(cmd, **kwargs):
        nonlocal rev_parse_count
        if cmd == ["git", "rev-parse", "HEAD"]:
            rev_parse_count += 1
            return MagicMock(returncode=0, stdout="stable-local\n")
        if cmd == ["git", "rev-parse", "origin/main"]:
            rev_parse_count += 1
            return MagicMock(returncode=0, stdout="stable-upstream\n")
        raise AssertionError(f"unexpected network command: {cmd}")

    with patch("hermes_cli.banner.subprocess.run", side_effect=fake_run), \
         patch("hermes_cli.config.detect_install_method", return_value="git"), \
         patch("hermes_cli.banner.check_via_pypi") as mock_pypi:
        result = banner.check_for_updates()

    assert result == 0
    # Exactly two rev-parse calls (local + upstream), no network
    assert rev_parse_count == 2
    mock_pypi.assert_not_called()


def test_check_for_updates_cache_accepts_legacy_payload(tmp_path, monkeypatch):
    """A legacy cache payload (without upstream_head/local_head keys) is
    treated as a cache miss — the mismatch between cached None and live
    SHA forces a fresh check.

    Before the head-guard change (PR #9670) the cache was
    ``{ts, behind, rev, ver}`` with no head fields. An upgraded client
    must not accept a stale "behind" from a pre-head-guard cache, because
    that cached value may have been written hours ago and upstream may
    have moved since. The probe reads ``cached.get("upstream_head")``
    which returns None for legacy payloads, while live values are real
    SHAs — the mismatch causes a fresh check.
    """
    import hermes_cli.banner as banner
    from hermes_cli import __version__

    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    fake_banner = repo_dir / "hermes_cli" / "banner.py"
    fake_banner.parent.mkdir(parents=True, exist_ok=True)
    fake_banner.touch()
    monkeypatch.setattr(banner, "__file__", str(fake_banner))

    # Legacy payload — no upstream_head or local_head keys
    cache_file = tmp_path / ".update_check"
    cache_file.write_text(
        json.dumps({
            "ts": time.time(),
            "behind": 0,
            "rev": None,
            "ver": __version__,
        })
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_REVISION", raising=False)

    def fake_run(cmd, **kwargs):
        if cmd[:2] == ["git", "rev-parse"] and "--is-shallow-repository" in cmd:
            return MagicMock(returncode=0, stdout="false\n")
        if cmd == ["git", "remote", "get-url", "origin"]:
            return MagicMock(returncode=0, stdout="https://github.com/NousResearch/hermes-agent.git\n")
        if cmd[:2] == ["git", "fetch"]:
            return MagicMock(returncode=0, stdout="")
        if cmd[:3] == ["git", "rev-list", "--count"]:
            return MagicMock(returncode=0, stdout="1\n")
        if cmd == ["git", "rev-parse", "HEAD"]:
            return MagicMock(returncode=0, stdout="some-local-sha\n")
        if cmd == ["git", "rev-parse", "origin/main"]:
            return MagicMock(returncode=0, stdout="some-upstream-sha\n")
        raise AssertionError(f"unexpected command: {cmd}")

    with patch("hermes_cli.banner.subprocess.run", side_effect=fake_run), \
         patch("hermes_cli.config.detect_install_method", return_value="git"):
        result = banner.check_for_updates()

    # Legacy cache rejected → fresh check ran
    assert result == 1

    # Cache rewritten with head fields
    written = json.loads(cache_file.read_text())
    assert written["upstream_head"] == "some-upstream-sha"
    assert written["local_head"] == "some-local-sha"


# =========================================================================
# _print_version_info output tests
# =========================================================================


def test_print_version_info_behind_none(tmp_path, monkeypatch, capsys):
    """When check_for_updates() returns None, _print_version_info() prints
    no update message (the check was inconclusive).
    """
    from hermes_cli.main import _print_version_info

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with patch("hermes_cli.banner.check_for_updates", return_value=None):
        _print_version_info(check_updates=True)

    captured = capsys.readouterr()
    assert "Up to date" not in captured.out
    assert "Update available" not in captured.out


def test_print_version_info_behind_zero(tmp_path, monkeypatch, capsys):
    """When check_for_updates() returns 0, print 'Up to date'."""
    from hermes_cli.main import _print_version_info

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with patch("hermes_cli.banner.check_for_updates", return_value=0):
        _print_version_info(check_updates=True)

    captured = capsys.readouterr()
    assert "Up to date" in captured.out
    assert "Update available" not in captured.out


def test_print_version_info_behind_positive(tmp_path, monkeypatch, capsys):
    """When check_for_updates() returns a positive count, print the exact
    number and the update command.
    """
    from hermes_cli.main import _print_version_info

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with patch("hermes_cli.banner.check_for_updates", return_value=5), \
         patch("hermes_cli.config.recommended_update_command", return_value="hermes update"):
        _print_version_info(check_updates=True)

    captured = capsys.readouterr()
    assert "Update available: 5 commits behind" in captured.out
    assert "hermes update" in captured.out


def test_print_version_info_behind_update_available_no_count(tmp_path, monkeypatch, capsys):
    """When check_for_updates() returns UPDATE_AVAILABLE_NO_COUNT (-1),
    print a plain 'Update available' message noting the shallow checkout.
    """
    from hermes_cli.main import _print_version_info
    from hermes_cli.banner import UPDATE_AVAILABLE_NO_COUNT

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with patch("hermes_cli.banner.check_for_updates", return_value=UPDATE_AVAILABLE_NO_COUNT), \
         patch("hermes_cli.config.recommended_update_command", return_value="hermes update"):
        _print_version_info(check_updates=True)

    captured = capsys.readouterr()
    assert "Update available" in captured.out
    assert "shallow checkout" in captured.out
    assert "hermes update" in captured.out
    # No bogus commit count in the shallow message
    assert "0 commits" not in captured.out
    assert "-1" not in captured.out
