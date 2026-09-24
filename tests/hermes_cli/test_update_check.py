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


@pytest.fixture
def git_repo(tmp_path, monkeypatch):
    """A fake checkout the update check resolves to, with git calls stubbed out."""
    repo_dir = tmp_path / "hermes-agent"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_REVISION", raising=False)
    monkeypatch.setattr(banner, "_resolve_repo_dir", lambda: repo_dir)
    monkeypatch.setattr("hermes_cli.config.detect_install_method", lambda root: "git")
    monkeypatch.setattr("hermes_cli.config.get_project_root", lambda: repo_dir)
    return repo_dir


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


def test_prefetch_update_check_is_noop_under_pytest():
    """Under pytest the prefetch must NOT start the git-spawning daemon
    thread: a process-wide ``patch("subprocess.run")`` in an unrelated test
    can capture the thread's ``git fetch``/``rev-parse`` spawns, flaking the
    unrelated test's call_args assertions (seen in
    tests/tui_gateway/test_subprocess_encoding.py and
    test_bot_relay_methods.py on CI, 2026-08-28)."""
    banner._update_result = None
    banner._update_check_done = threading.Event()

    before = {t.ident for t in threading.enumerate()}
    with patch.object(banner, "check_for_updates") as mock_check:
        banner.prefetch_update_check()
        # The done event is set synchronously so get_update_result() callers
        # don't burn their timeout waiting on a check that will never run.
        assert banner._update_check_done.is_set()
        mock_check.assert_not_called()
    after = {t.ident for t in threading.enumerate()}
    assert after <= before, "prefetch_update_check spawned a thread under pytest"


def test_prefetch_banner_data_is_noop_under_pytest(monkeypatch):
    """Same stray-git-spawn class: prefetch_banner_data must not start its
    daemon thread under pytest."""
    monkeypatch.setattr(banner, "_banner_data_prefetch_started", False)
    with patch.object(banner, "get_git_banner_state") as mock_state:
        banner.prefetch_banner_data()
        # Give a hypothetical stray thread a beat to run — nothing should.
        time.sleep(0.05)
        mock_state.assert_not_called()
    assert banner._banner_data_prefetch_started is True


def test_upstream_main_sha_ls_remote_fallback_disables_git_prompts(monkeypatch):
    """When the API is unreachable the HTTPS ls-remote fallback must never inherit the terminal."""
    monkeypatch.setattr(banner, "_github_branch_tip", lambda slug, branch: None)
    completed = MagicMock(returncode=1, stdout="", stderr="auth required")
    run = MagicMock(return_value=completed)
    monkeypatch.setattr(banner.subprocess, "run", run)

    assert banner._upstream_main_sha() is None
    kwargs = run.call_args.kwargs
    assert kwargs["stdin"] is banner.subprocess.DEVNULL
    assert kwargs["env"]["GIT_TERMINAL_PROMPT"] == "0"
    assert kwargs["env"]["GCM_INTERACTIVE"] == "Never"


def test_tips_behind_diverged_when_merge_base_exists_and_neither_is_ancestor(tmp_path, monkeypatch):
    """Full clone: shared history but neither tip is ancestor → UPDATE_DIVERGED, not a count."""
    repo = tmp_path / "repo"
    repo.mkdir()

    def fake_ok(args, cwd=None, **kw):
        if args[:2] == ["merge-base", "--is-ancestor"]:
            return False
        if args[:1] == ["merge-base"]:
            return True
        return False

    monkeypatch.setattr(banner, "_git_ok", fake_ok)
    monkeypatch.setattr(banner, "_github_compare_behind", lambda *a, **k: 99)
    assert banner._tips_behind(SHA_A, SHA_B, repo) == banner.UPDATE_DIVERGED


def test_tips_behind_shallow_no_merge_base_uses_compare(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()

    def fake_ok(args, cwd=None, **kw):
        return False

    monkeypatch.setattr(banner, "_git_ok", fake_ok)
    monkeypatch.setattr(banner, "_github_compare_behind", lambda *a, **k: 61)
    assert banner._tips_behind(SHA_A, SHA_B, repo) == 61


def test_format_update_notice_diverged_is_not_a_tip_count(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.recommended_update_command", lambda: "hermes update")
    text = banner._format_update_notice(banner.UPDATE_DIVERGED)
    assert "diverged" in text.lower()
    assert "+-2" not in text
    assert "2 commit" not in text
