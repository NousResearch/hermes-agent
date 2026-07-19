"""Tests for hermes -z -w — worktree isolation in one-shot mode (#67458).

The flag was previously accepted but silently ignored: run_oneshot() never
received it, so the agent ran in the current checkout and its commits landed
on the live branch. These tests pin the forwarded behavior: TERMINAL_CWD is
retargeted at the disposable worktree for the duration of the run, cleanup
runs afterward, and a requested-but-failed setup refuses to run instead of
silently dropping isolation.
"""

import logging
import os
from unittest.mock import patch

import pytest

from hermes_cli.oneshot import run_oneshot


@pytest.fixture(autouse=True)
def _restore_global_state(monkeypatch):
    """run_oneshot mutates process-global state; keep it test-local."""
    monkeypatch.setenv("HERMES_YOLO_MODE", "0")
    monkeypatch.setenv("HERMES_ACCEPT_HOOKS", "0")
    monkeypatch.setenv("TERMINAL_CWD", "/original/cwd")
    yield
    logging.disable(logging.NOTSET)


def test_worktree_success_retargets_terminal_cwd_and_cleans_up(tmp_path, capsys):
    wt_path = str(tmp_path / "wt")
    wt_info = {"path": wt_path, "branch": "hermes/wt-test"}
    seen = {}
    cleaned = []

    def fake_run_agent(prompt, **kwargs):
        seen["terminal_cwd"] = os.environ.get("TERMINAL_CWD")
        return "done", {}

    def fake_cleanup(info):
        cleaned.append(info)
        # The real helper print()s its outcome — including the
        # "has unpushed commits, keeping" notice in -w's main use case.
        print(f"⚠ Worktree has unpushed commits, keeping: {info['path']}")

    with patch("cli.CLI_CONFIG", {}, create=True), \
         patch("cli._git_repo_root", return_value=str(tmp_path)), \
         patch("cli._prune_stale_worktrees") as prune, \
         patch("cli._setup_worktree", return_value=wt_info), \
         patch("cli._cleanup_worktree", side_effect=fake_cleanup), \
         patch("hermes_cli.oneshot._run_agent", side_effect=fake_run_agent):
        rc = run_oneshot("make a commit", worktree=True)

    assert rc == 0
    assert seen["terminal_cwd"] == wt_path, "agent did not run inside the worktree"
    assert cleaned == [wt_info], "worktree was not cleaned up after the run"
    prune.assert_called_once()
    # One-shot's stdout contract: ONLY the final response. The cleanup
    # helper's output must land on stderr, never before the response on
    # stdout.
    captured = capsys.readouterr()
    assert captured.out == "done\n"
    assert "unpushed commits" in captured.err
    # The dangling (now-deleted) worktree path must not leak into the
    # caller's env. (The exact restored value is whatever cli.py's config
    # bridge exported at import time, so pin only the non-dangling part.)
    assert os.environ.get("TERMINAL_CWD") != wt_path


def test_worktree_setup_failure_refuses_to_run(capsys):
    with patch("cli.CLI_CONFIG", {}, create=True), \
         patch("cli._git_repo_root", return_value=None), \
         patch("cli._prune_stale_worktrees"), \
         patch("cli._setup_worktree", return_value=None), \
         patch("cli._cleanup_worktree"), \
         patch("hermes_cli.oneshot._run_agent") as run_agent:
        rc = run_oneshot("make a commit", worktree=True)

    assert rc == 2
    run_agent.assert_not_called()
    err = capsys.readouterr().err
    assert "Refusing to run without" in err


def test_worktree_setup_exception_is_a_hard_error(capsys):
    with patch("cli.CLI_CONFIG", {}, create=True), \
         patch("cli._git_repo_root", side_effect=RuntimeError("git exploded")), \
         patch("hermes_cli.oneshot._run_agent") as run_agent:
        rc = run_oneshot("make a commit", worktree=True)

    assert rc == 2
    run_agent.assert_not_called()
    assert "failed to create worktree" in capsys.readouterr().err


def test_default_run_does_not_touch_worktree_machinery():
    with patch("cli._setup_worktree") as setup, \
         patch("hermes_cli.oneshot._run_agent", return_value=("done", {})):
        rc = run_oneshot("hello")

    assert rc == 0
    setup.assert_not_called()
    assert os.environ.get("TERMINAL_CWD") == "/original/cwd"
