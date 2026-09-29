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

    def fake_setup(**kwargs):
        # The real helper print()s its setup progress too.
        print(f"✓ Worktree created: {wt_path}")
        return wt_info

    with patch("cli.CLI_CONFIG", {}, create=True), \
         patch("cli._git_repo_root", return_value=str(tmp_path)), \
         patch("cli._prune_stale_worktrees") as prune, \
         patch("cli._setup_worktree", side_effect=fake_setup), \
         patch("cli._cleanup_worktree", side_effect=fake_cleanup), \
         patch("hermes_cli.oneshot._run_agent", side_effect=fake_run_agent):
        # Entering the patch context imported `cli`, whose config bridge
        # force-exports TERMINAL_CWD; re-pin the caller's value afterward so
        # the exact-restore assertion below is meaningful. (In production the
        # first `import cli` happens inside run_oneshot, AFTER its capture —
        # which is exactly why the capture sits before the import.)
        os.environ["TERMINAL_CWD"] = "/original/cwd"
        rc = run_oneshot("make a commit", worktree=True)

    assert rc == 0
    assert seen["terminal_cwd"] == wt_path, "agent did not run inside the worktree"
    assert cleaned == [wt_info], "worktree was not cleaned up after the run"
    prune.assert_called_once()
    # One-shot's stdout contract: ONLY the final response. Both the setup
    # helper's "Worktree created" print and the cleanup helper's output must
    # land on stderr, never on stdout.
    captured = capsys.readouterr()
    assert captured.out == "done\n"
    assert "Worktree created" in captured.err
    assert "unpushed commits" in captured.err
    # TERMINAL_CWD is captured BEFORE the cli import (whose config bridge
    # force-exports it), so the caller's original value is restored exactly.
    assert os.environ.get("TERMINAL_CWD") == "/original/cwd"


def test_worktree_setup_failure_refuses_to_run(capsys):
    with patch("cli.CLI_CONFIG", {}, create=True), \
         patch("cli._git_repo_root", return_value=None), \
         patch("cli._prune_stale_worktrees"), \
         patch("cli._setup_worktree", return_value=None), \
         patch("cli._cleanup_worktree"), \
         patch("hermes_cli.oneshot._run_agent") as run_agent:
        os.environ["TERMINAL_CWD"] = "/original/cwd"  # re-pin after cli import
        rc = run_oneshot("make a commit", worktree=True)

    assert rc == 2
    run_agent.assert_not_called()
    err = capsys.readouterr().err
    assert "Refusing to run without" in err
    # The early-return path must also restore the caller's TERMINAL_CWD —
    # the cli import already force-exported it before setup failed.
    assert os.environ.get("TERMINAL_CWD") == "/original/cwd"


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
        os.environ["TERMINAL_CWD"] = "/original/cwd"  # re-pin after cli import
        rc = run_oneshot("hello")

    assert rc == 0
    setup.assert_not_called()
    assert os.environ.get("TERMINAL_CWD") == "/original/cwd"


@pytest.mark.parametrize("original_cwd", ["/orig/cwd", None])
def test_keyboard_interrupt_during_setup_restores_terminal_cwd(monkeypatch, original_cwd):
    def fake_setup(**kwargs):
        os.environ["TERMINAL_CWD"] = "/mutated/by/import"
        raise KeyboardInterrupt

    with patch("cli.CLI_CONFIG", {}, create=True), \
         patch("cli._git_repo_root", return_value="/repo"), \
         patch("cli._prune_stale_worktrees"), \
         patch("cli._setup_worktree", side_effect=fake_setup), \
         patch("cli._cleanup_worktree"), \
         patch("hermes_cli.oneshot._run_agent") as run_agent:
        if original_cwd is None:
            monkeypatch.delenv("TERMINAL_CWD", raising=False)
        else:
            monkeypatch.setenv("TERMINAL_CWD", original_cwd)

        with pytest.raises(KeyboardInterrupt):
            run_oneshot("hello", worktree=True)

    assert os.environ.get("TERMINAL_CWD") == original_cwd
    if original_cwd is None:
        assert "TERMINAL_CWD" not in os.environ
    run_agent.assert_not_called()


def test_keyboard_interrupt_before_run_restores_terminal_cwd_and_cleans_up():
    wt_info = {"path": "/wt/path", "branch": "hermes/wt-test"}

    def interrupt_before_run():
        assert os.environ.get("TERMINAL_CWD") == wt_info["path"]
        raise KeyboardInterrupt

    with patch("cli.CLI_CONFIG", {}, create=True), \
         patch("cli._git_repo_root", return_value="/repo"), \
         patch("cli._prune_stale_worktrees"), \
         patch("cli._setup_worktree", return_value=wt_info), \
         patch("cli._cleanup_worktree") as cleanup, \
         patch("hermes_cli.oneshot.declare_stateless_channel", side_effect=interrupt_before_run), \
         patch("hermes_cli.oneshot._run_agent") as run_agent:
        os.environ["TERMINAL_CWD"] = "/orig/cwd"

        with pytest.raises(KeyboardInterrupt):
            run_oneshot("hello", worktree=True)

    run_agent.assert_not_called()
    cleanup.assert_called_once_with(wt_info)
    assert os.environ.get("TERMINAL_CWD") == "/orig/cwd"


def test_keyboard_interrupt_during_cleanup_restores_terminal_cwd(monkeypatch, tmp_path):
    wt_info = {"path": str(tmp_path / "wt"), "branch": "hermes/wt-test"}

    def fake_setup(**kwargs):
        os.environ["TERMINAL_CWD"] = "/mutated/by/import"
        return wt_info

    with patch("cli.CLI_CONFIG", {}, create=True), \
         patch("cli._git_repo_root", return_value=str(tmp_path)), \
         patch("cli._prune_stale_worktrees"), \
         patch("cli._setup_worktree", side_effect=fake_setup), \
         patch("cli._cleanup_worktree", side_effect=KeyboardInterrupt), \
         patch("hermes_cli.oneshot._run_agent", return_value=("done", {})) as run_agent:
        monkeypatch.setenv("TERMINAL_CWD", "/orig/cwd")

        with pytest.raises(KeyboardInterrupt):
            run_oneshot("hello", worktree=True)

    run_agent.assert_called_once()
    assert os.environ.get("TERMINAL_CWD") == "/orig/cwd"
