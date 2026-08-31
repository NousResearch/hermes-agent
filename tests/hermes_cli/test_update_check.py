"""Tests for the update check mechanism in hermes_cli.banner.

Passive checks go through the GitHub REST API — never ``git fetch``. Every CLI, TUI and desktop
start used to fetch; across the install base that was tens of millions of fetch requests a day
and GitHub asked us to poll the API instead. These tests pin that contract plus the cache
policy that keeps the API traffic to one request a day per install.
"""

import json
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

import hermes_cli.banner as banner

SHA_A = "a" * 40
SHA_B = "b" * 40



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


def _stub_git(monkeypatch, *, head=SHA_A, origin="https://github.com/NousResearch/hermes-agent.git"):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        sub = args[1]
        if sub == "rev-parse":
            return MagicMock(returncode=0, stdout=f"{head}\n")
        if sub == "remote":
            return MagicMock(returncode=0, stdout=f"{origin}\n")
        if sub == "merge-base":
            return MagicMock(returncode=1, stdout="")
        raise AssertionError(f"passive check must not run git {sub}: {args}")

    monkeypatch.setattr(banner.subprocess, "run", fake_run)
    return calls


def test_passive_check_uses_the_api_and_never_fetches(git_repo, monkeypatch):
    """The whole point: no ``git fetch`` / ``ls-remote`` for a GitHub origin, exact count via compare."""
    calls = _stub_git(monkeypatch, head=SHA_A)
    tip = MagicMock(return_value=SHA_B)
    monkeypatch.setattr(banner, "_github_branch_tip", tip)
    monkeypatch.setattr(banner, "_github_compare_behind", lambda cur, tgt: 61)

    assert banner.check_for_updates() == 61
    tip.assert_called_once_with("nousresearch/hermes-agent", "main")
    assert not any(c[1] in {"fetch", "ls-remote"} for c in calls)

    cached = json.loads((git_repo.parent / ".update_check").read_text())
    assert (cached["head"], cached["target"], cached["behind"]) == (SHA_A, SHA_B, 61)


def test_cache_is_daily_but_invalidated_when_head_moves(git_repo, monkeypatch):
    """A fresh cache answers without any network; ``hermes update`` moving HEAD busts it at once;
    an inconclusive (None) result is retried after the shorter failure window, not never."""
    from hermes_cli import __version__

    cache_file = git_repo.parent / ".update_check"
    _stub_git(monkeypatch, head=SHA_A)
    tip = MagicMock(return_value=None)
    monkeypatch.setattr(banner, "_github_branch_tip", tip)

    def write_cache(*, ts, head, behind):
        cache_file.write_text(json.dumps(
            {"ts": ts, "behind": behind, "rev": None, "ver": __version__, "head": head}))

    write_cache(ts=time.time() - banner._UPDATE_CHECK_CACHE_SECONDS + 60, head=SHA_A, behind=3)
    assert banner.check_for_updates() == 3
    tip.assert_not_called()

    write_cache(ts=time.time(), head=SHA_B, behind=3)  # cached for a different HEAD
    assert banner.check_for_updates() is None  # API unreachable → inconclusive, re-asked
    tip.assert_called_once()

    tip.reset_mock()
    write_cache(ts=time.time() - banner._UPDATE_CHECK_FAILURE_CACHE_SECONDS + 60, head=SHA_A, behind=None)
    assert banner.check_for_updates() is None
    tip.assert_not_called()

    write_cache(ts=time.time() - banner._UPDATE_CHECK_FAILURE_CACHE_SECONDS - 1, head=SHA_A, behind=None)
    banner.check_for_updates()
    tip.assert_called_once()


def test_prefetch_non_blocking(monkeypatch):
    """prefetch_update_check() should return immediately without blocking."""
    # Reset module state; force the real (non-pytest) thread path.
    banner._update_result = None
    banner._update_check_done = threading.Event()
    monkeypatch.setattr(banner, "_skip_background_prefetch", lambda: False)

    with patch.object(banner, "check_for_updates", return_value=5):
        start = time.monotonic()
        banner.prefetch_update_check()
        assert time.monotonic() - start < 1.0
        banner._update_check_done.wait(timeout=5)
        assert banner._update_result == 5


def test_check_via_local_git_fetch_failure_returns_none(tmp_path, monkeypatch):
    """When git fetch fails and the stale origin/main ref is not ahead,
    _check_via_local_git must return None (#82166).

    A stale tracking ref cannot prove *currentness* (rev-list 0 just means
    the ref hasn't caught up), so returning None is the honest inconclusive
    result — and the caller must not cache it as "up to date".
    """
    from hermes_cli import banner

    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    # Simulate a non-shallow, non-SSH-remote checkout
    def mock_git_stdout(args, *, cwd, timeout=5):
        if args[:2] == ["remote", "get-url"]:
            return "https://github.com/NousResearch/hermes-agent.git"
        if args[:2] == ["rev-parse", "--is-shallow-repository"]:
            return "false"
        return None

    # Fetch fails (returncode != 0); stale rev-list reports 0 behind
    failed_proc = MagicMock()
    failed_proc.returncode = 1
    failed_proc.stdout = ""
    failed_proc.stderr = "fatal: could not reach remote"

    stale_zero_proc = MagicMock()
    stale_zero_proc.returncode = 0
    stale_zero_proc.stdout = "0"

    def mock_run(args, **kwargs):
        if args[:2] == ["git", "fetch"]:
            return failed_proc
        if args[:2] == ["git", "rev-list"]:
            return stale_zero_proc
        raise AssertionError(f"unexpected subprocess.run: {args}")

    monkeypatch.setattr(banner, "_git_stdout", mock_git_stdout)
    monkeypatch.setattr(banner.subprocess, "run", mock_run)

    result = banner._check_via_local_git(repo_dir)
    assert result is None, (
        "Fetch failure with stale 0-behind must return None, not 'up to date'"
    )


def test_check_via_local_git_fetch_failure_keeps_positive_stale_count(tmp_path, monkeypatch):
    """A failed fetch must preserve sound evidence: if the stale origin/main
    ref already shows HEAD behind, that positive count is still an update
    signal and must be returned (review #92578)."""
    from hermes_cli import banner

    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    def mock_git_stdout(args, *, cwd, timeout=5):
        if args[:2] == ["remote", "get-url"]:
            return "https://github.com/NousResearch/hermes-agent.git"
        if args[:2] == ["rev-parse", "--is-shallow-repository"]:
            return "false"
        return None

    failed_proc = MagicMock()
    failed_proc.returncode = 1
    failed_proc.stdout = ""
    failed_proc.stderr = "fatal: could not reach remote"

    stale_behind_proc = MagicMock()
    stale_behind_proc.returncode = 0
    stale_behind_proc.stdout = "5"

    def mock_run(args, **kwargs):
        if args[:2] == ["git", "fetch"]:
            return failed_proc
        if args[:2] == ["git", "rev-list"]:
            return stale_behind_proc
        raise AssertionError(f"unexpected subprocess.run: {args}")

    monkeypatch.setattr(banner, "_git_stdout", mock_git_stdout)
    monkeypatch.setattr(banner.subprocess, "run", mock_run)

    result = banner._check_via_local_git(repo_dir)
    assert result == 5, "Stale positive behind-count must be preserved on fetch failure"


def test_check_via_local_git_fetch_failure_rev_list_error_returns_none(tmp_path, monkeypatch):
    """If the stale rev-list itself fails, the check stays inconclusive (None)."""
    from hermes_cli import banner

    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    def mock_git_stdout(args, *, cwd, timeout=5):
        if args[:2] == ["remote", "get-url"]:
            return "https://github.com/NousResearch/hermes-agent.git"
        if args[:2] == ["rev-parse", "--is-shallow-repository"]:
            return "false"
        return None

    failed_proc = MagicMock()
    failed_proc.returncode = 1
    failed_proc.stdout = ""
    failed_proc.stderr = "fatal: could not reach remote"

    bad_rev_list = MagicMock()
    bad_rev_list.returncode = 128
    bad_rev_list.stdout = ""
    bad_rev_list.stderr = "fatal: ambiguous argument 'HEAD..origin/main'"

    def mock_run(args, **kwargs):
        if args[:2] == ["git", "fetch"]:
            return failed_proc
        if args[:2] == ["git", "rev-list"]:
            return bad_rev_list
        raise AssertionError(f"unexpected subprocess.run: {args}")

    monkeypatch.setattr(banner, "_git_stdout", mock_git_stdout)
    monkeypatch.setattr(banner.subprocess, "run", mock_run)

    result = banner._check_via_local_git(repo_dir)
    assert result is None


def test_check_for_updates_does_not_cache_none(tmp_path, monkeypatch):
    """check_for_updates must not cache None results so a transient fetch
    failure doesn't suppress retries for the full 6-hour cache window (#82166).

    Instead of mocking the full Path resolution chain, we verify the cache-write
    guard directly: call check_for_updates with a mocked _check_via_local_git
    that returns None, and confirm no cache file is created.
    """
    import hermes_cli.banner as banner

    cache_file = tmp_path / ".update_check"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_REVISION", raising=False)

    # Create a fake repo dir so the .git check passes
    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()

    # Mock the internal functions to force the local-git path returning None
    monkeypatch.setattr(banner, "_check_via_local_git", lambda rd: None)
    monkeypatch.setattr(
        "hermes_cli.config.detect_install_method", lambda root: "git"
    )
    monkeypatch.setattr(
        "hermes_cli.config.get_project_root", lambda: repo_dir
    )

    # Patch __file__ resolution by monkeypatching the module's Path calls.
    # check_for_updates does: Path(__file__).parent.parent.resolve()
    # We intercept by making the resolve() return our fake repo_dir.
    original_init = Path.__init__

    def patched_path_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)

    # Simpler: just patch the get_hermes_home and the repo_dir resolution
    # by making check_for_updates find our fake repo via hermes_home fallback.
    # The code checks Path(__file__).parent.parent/.git first, then falls
    # back to hermes_home / "hermes-agent". We ensure the fallback hits.
    # To do this, we make Path(__file__).parent.parent.resolve() return
    # a path without .git, so it falls through to hermes_home / "hermes-agent".
    real_resolve = Path.resolve

    def fake_resolve(self, *args, **kwargs):
        s = str(self)
        if "banner.py" in s or s.endswith("hermes_cli"):
            # Return a path that has no .git, forcing the fallback
            return tmp_path / "no-git-here"
        return real_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", fake_resolve)

    result = banner.check_for_updates()
    assert result is None

    # The cache file must NOT have been written with a None result
    assert not cache_file.exists(), "None result must not be cached"




