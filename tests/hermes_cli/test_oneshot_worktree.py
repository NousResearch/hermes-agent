"""`hermes -z -w` (one-shot) must run inside a disposable git worktree.

One-shot mode used to silently drop `-w`: the agent ran in the caller's cwd and
its commits landed on the checked-out branch. It now mirrors interactive
`hermes -w` — set up a worktree, chdir into it, run, and tear it down. See
issue #67458.
"""

import os

import pytest


def _stub_agent(monkeypatch, oneshot_mod, sink):
    """Make `_run_agent` record the cwd it saw and return a canned response."""

    def _fake_run_agent(prompt, **_kwargs):
        sink["agent_cwd"] = os.getcwd()
        return "done", {}

    monkeypatch.setattr(oneshot_mod, "_run_agent", _fake_run_agent)


def test_worktree_runs_agent_in_worktree_and_cleans_up(monkeypatch, tmp_path, capsys):
    import hermes_cli.oneshot as oneshot_mod

    wt_path = tmp_path / "wt"
    wt_path.mkdir()
    wt_info = {"path": str(wt_path), "branch": "hermes/test", "repo_root": str(tmp_path)}
    sink = {}
    cleaned = []

    monkeypatch.setattr(oneshot_mod, "_setup_oneshot_worktree", lambda: wt_info)
    monkeypatch.setattr(
        oneshot_mod, "_cleanup_oneshot_worktree", lambda info: cleaned.append(info)
    )
    _stub_agent(monkeypatch, oneshot_mod, sink)

    start_cwd = os.getcwd()
    try:
        rc = oneshot_mod.run_oneshot("hello", worktree=True)
    finally:
        os.chdir(start_cwd)

    assert rc == 0
    # Agent ran inside the worktree, not the caller's cwd.
    assert os.path.realpath(sink["agent_cwd"]) == os.path.realpath(str(wt_path))
    # cwd is restored and the worktree is torn down exactly once.
    assert os.getcwd() == start_cwd
    assert cleaned == [wt_info]
    # stdout carries only the final response.
    assert capsys.readouterr().out.strip() == "done"


def test_worktree_setup_failure_returns_2_without_running_agent(monkeypatch):
    import hermes_cli.oneshot as oneshot_mod

    ran = {"agent": False}

    def _boom_agent(*_args, **_kwargs):
        ran["agent"] = True
        return "should-not-run", {}

    monkeypatch.setattr(oneshot_mod, "_setup_oneshot_worktree", lambda: None)
    monkeypatch.setattr(oneshot_mod, "_run_agent", _boom_agent)
    # If setup fails we must not attempt cleanup on a nonexistent worktree.
    monkeypatch.setattr(
        oneshot_mod,
        "_cleanup_oneshot_worktree",
        lambda info: pytest.fail("cleanup ran without a worktree"),
    )

    start_cwd = os.getcwd()
    rc = oneshot_mod.run_oneshot("hello", worktree=True)

    assert rc == 2
    assert ran["agent"] is False
    assert os.getcwd() == start_cwd


def test_worktree_cleanup_runs_even_when_agent_fails(monkeypatch, tmp_path, capsys):
    import hermes_cli.oneshot as oneshot_mod

    wt_path = tmp_path / "wt"
    wt_path.mkdir()
    wt_info = {"path": str(wt_path), "branch": "hermes/test", "repo_root": str(tmp_path)}
    cleaned = []

    monkeypatch.setattr(oneshot_mod, "_setup_oneshot_worktree", lambda: wt_info)
    monkeypatch.setattr(
        oneshot_mod, "_cleanup_oneshot_worktree", lambda info: cleaned.append(info)
    )

    def _boom(*_args, **_kwargs):
        raise RuntimeError("agent exploded")

    monkeypatch.setattr(oneshot_mod, "_run_agent", _boom)

    start_cwd = os.getcwd()
    try:
        rc = oneshot_mod.run_oneshot("hello", worktree=True)
    finally:
        os.chdir(start_cwd)

    assert rc == 1
    # Worktree is still torn down and cwd restored on the failure path.
    assert cleaned == [wt_info]
    assert os.getcwd() == start_cwd


def test_no_worktree_by_default_does_not_touch_cwd(monkeypatch):
    import hermes_cli.oneshot as oneshot_mod

    sink = {}
    monkeypatch.setattr(
        oneshot_mod,
        "_setup_oneshot_worktree",
        lambda: pytest.fail("worktree setup ran without -w"),
    )
    _stub_agent(monkeypatch, oneshot_mod, sink)

    start_cwd = os.getcwd()
    rc = oneshot_mod.run_oneshot("hello")

    assert rc == 0
    assert sink["agent_cwd"] == start_cwd
    assert os.getcwd() == start_cwd
