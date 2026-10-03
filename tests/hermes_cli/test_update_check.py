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
    # Release lookup unavailable → falls back to the main-tip comparison.
    with patch("hermes_cli.banner._fetch_latest_release_sha", return_value=None):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        with patch("hermes_cli.banner.subprocess.run", return_value=mock_result) as mock_run:
            result = check_for_updates()

    assert result == 5
    # exact-match describe + git fetch + git rev-list
    assert mock_run.call_count == 3


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
# Tag-pinned checkout: compare against the latest GitHub release, not main
# =========================================================================

_HEAD_SHA = "f" * 39 + "0"
_RELEASE_SHA = "f" * 39 + "1"


def _make_git_run(head_sha):
    """Build a subprocess.run dispatcher whose rev-parse reports ``head_sha``."""
    def _git_run_by_argv(argv, **kwargs):
        """Dispatch patched subprocess.run calls by command shape."""
        if argv[:3] == ["git", "describe", "--tags"]:
            return MagicMock(returncode=0, stdout="v2026.9.24\n")
        if argv[:2] == ["git", "fetch"]:
            raise AssertionError("git fetch must not run in the release-tag path")
        if argv[:2] == ["git", "rev-parse"]:
            return MagicMock(returncode=0, stdout=head_sha + "\n")
        if argv[:2] == ["git", "rev-list"]:
            return MagicMock(returncode=0, stdout="7000\n")
        raise AssertionError(f"unexpected subprocess call: {argv}")
    return _git_run_by_argv


def test_check_for_updates_tag_pinned_at_latest_release(tmp_path, monkeypatch):
    """HEAD exactly at a tag whose SHA matches the latest release → 0, not N behind.

    A tag-pinned editable checkout (HEAD exactly at release tag v2026.9.24) is
    thousands of commits behind main by construction, but that is not an
    available update: the comparison target must be the latest GitHub release
    tag's commit SHA instead.
    """
    import hermes_cli.banner as banner

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_REVISION", raising=False)
    cache_file = tmp_path / ".update_check"

    with patch("hermes_cli.banner.subprocess.run", side_effect=_make_git_run(_RELEASE_SHA)) as mock_run, \
         patch("hermes_cli.banner._fetch_latest_release_sha", return_value=_RELEASE_SHA):
        result = banner.check_for_updates()

    assert result == 0
    # Passive behavior preserved: GitHub API only, never a git fetch.
    for call in mock_run.call_args_list:
        assert call.args[0][:2] != ["git", "fetch"]
    # The tag is recorded so a moved HEAD tag invalidates the cached verdict.
    written = json.loads(cache_file.read_text())
    assert written["head"] == "v2026.9.24"
    assert written["behind"] == 0


def test_check_for_updates_tag_pinned_older_release(tmp_path, monkeypatch):
    """HEAD at a tag that is not the latest release → UPDATE_AVAILABLE_NO_COUNT."""
    import hermes_cli.banner as banner

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_REVISION", raising=False)

    with patch("hermes_cli.banner.subprocess.run", side_effect=_make_git_run(_HEAD_SHA)), \
         patch("hermes_cli.banner._fetch_latest_release_sha", return_value=_RELEASE_SHA):
        result = banner.check_for_updates()

    assert result == banner.UPDATE_AVAILABLE_NO_COUNT


def test_check_for_updates_no_exact_tag_falls_back_to_main(tmp_path, monkeypatch):
    """When HEAD is not exactly at a tag, keep the origin/main commit count."""
    import hermes_cli.banner as banner

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_REVISION", raising=False)

    def no_tag_run(argv, **kwargs):
        if argv[:3] == ["git", "describe", "--tags"]:
            return MagicMock(returncode=128, stdout="", stderr="no tag exactly on HEAD")
        return _make_git_run(_HEAD_SHA)(argv, **kwargs)

    with patch("hermes_cli.banner.subprocess.run", side_effect=no_tag_run) as mock_run, \
         patch("hermes_cli.banner._fetch_latest_release_sha") as mock_release:
        result = banner.check_for_updates()

    assert result == 7000
    # HEAD is not at a tag → no release lookup at all.
    mock_release.assert_not_called()
    commands = [call.args[0][:2] for call in mock_run.call_args_list]
    assert ["git", "fetch"] in commands and ["git", "rev-list"] in commands


def test_check_for_updates_release_info_unavailable_falls_back(tmp_path, monkeypatch):
    """Tag at HEAD but GitHub release lookup fails → main-tip comparison."""
    import hermes_cli.banner as banner

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_REVISION", raising=False)

    with patch("hermes_cli.banner.subprocess.run", side_effect=_make_git_run(_HEAD_SHA)) as mock_run, \
         patch("hermes_cli.banner._fetch_latest_release_sha", return_value=None):
        result = banner.check_for_updates()

    assert result == 7000
    commands = [call.args[0][:2] for call in mock_run.call_args_list]
    assert ["git", "fetch"] in commands and ["git", "rev-list"] in commands


def test_check_for_updates_tag_cache_invalidated_on_tag_move(tmp_path, monkeypatch):
    """A fresh cache stamped with a different HEAD tag must be re-checked."""
    import hermes_cli.banner as banner

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_REVISION", raising=False)
    cache_file = tmp_path / ".update_check"
    cache_file.write_text(
        json.dumps({"ts": time.time(), "behind": 0, "rev": None,
                    "ver": banner.VERSION, "head": "v2026.9.23"})
    )

    with patch("hermes_cli.banner.subprocess.run", side_effect=_make_git_run(_RELEASE_SHA)), \
         patch("hermes_cli.banner._fetch_latest_release_sha", return_value=_RELEASE_SHA):
        result = banner.check_for_updates()

    # HEAD has since moved to v2026.9.24 → cached verdict for the old tag is stale.
    assert result == 0
    assert json.loads(cache_file.read_text())["head"] == "v2026.9.24"


def test_check_for_updates_tag_cache_hit(tmp_path, monkeypatch):
    """Fresh cache with a matching HEAD tag returns the cached verdict."""
    import hermes_cli.banner as banner

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_REVISION", raising=False)
    cache_file = tmp_path / ".update_check"
    cache_file.write_text(
        json.dumps({"ts": time.time(), "behind": 0, "rev": None,
                    "ver": banner.VERSION, "head": "v2026.9.24"})
    )

    with patch("hermes_cli.banner.subprocess.run", side_effect=_make_git_run(_HEAD_SHA)) as mock_run, \
         patch("hermes_cli.banner._fetch_latest_release_sha") as mock_release:
        result = banner.check_for_updates()

    assert result == 0
    # Tag matches the cached one → no fresh check, no release lookup.
    mock_release.assert_not_called()
    commands = [call.args[0][:2] for call in mock_run.call_args_list]
    assert ["git", "fetch"] not in commands and ["git", "rev-list"] not in commands


def test_fetch_latest_release_sha_prefers_target_commitish_sha():
    """target_commitish carrying a full SHA is used directly."""
    import hermes_cli.banner as banner

    with patch.object(
        banner, "_github_api_json",
        return_value={"tag_name": "v2026.9.24", "target_commitish": _RELEASE_SHA},
    ) as api:
        assert banner._fetch_latest_release_sha() == _RELEASE_SHA
        # Single API call — no tag peeling needed.
        assert api.call_count == 1


def test_fetch_latest_release_sha_peels_annotated_tag():
    """target_commitish naming a branch → resolve and peel the tag via the API."""
    import hermes_cli.banner as banner

    def api_json(path):
        if path == "releases/latest":
            return {"tag_name": "v2026.9.24", "target_commitish": "main"}
        if path == "git/ref/tags/v2026.9.24":
            return {"object": {"sha": "a" * 40, "type": "tag"}}
        if path == "git/tags/" + "a" * 40:
            return {"object": {"sha": _RELEASE_SHA, "type": "commit"}}
        raise AssertionError(f"unexpected API path: {path}")

    with patch.object(banner, "_github_api_json", side_effect=api_json):
        assert banner._fetch_latest_release_sha() == _RELEASE_SHA


def test_fetch_latest_release_sha_returns_none_without_release():
    """API failure or missing tag_name → None (caller falls back to main tip)."""
    import hermes_cli.banner as banner

    with patch.object(banner, "_github_api_json", return_value=None):
        assert banner._fetch_latest_release_sha() is None

    with patch.object(banner, "_github_api_json", return_value={"tag_name": ""}):
        assert banner._fetch_latest_release_sha() is None


def test_git_exact_tag_requires_exact_match():
    """describe --exact-match failing (HEAD not on a tag) returns None."""
    import hermes_cli.banner as banner

    with patch("hermes_cli.banner.subprocess.run",
               return_value=MagicMock(returncode=128, stdout="")):
        assert banner._git_exact_tag(Path("/tmp/fake-repo")) is None
