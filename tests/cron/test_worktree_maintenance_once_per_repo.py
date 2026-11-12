"""Worktree maintenance must not multiply whole-``.worktrees/`` disk walks.

A multiplexed gateway ticks cron once per served profile, and every profile resolves the hermes
checkout; each prune pass ends in a footprint probe that used to ``du`` the entire ``.worktrees/``
(millions of node_modules inodes) even when the tree count alone decided the warning.
"""

import logging
import os
import stat
import subprocess

import pytest


def _git_repo(path):
    path.mkdir()
    subprocess.run(["git", "init", "--initial-branch=main", str(path)], check=True,
                   capture_output=True, timeout=15)
    (path / ".worktrees").mkdir()
    return path


def test_profiles_sharing_a_repo_prune_it_once_per_interval(tmp_path, monkeypatch):
    import cron.scheduler as sched
    from hermes_cli import worktree_ops
    from hermes_cli.worktree_registry import record_worktree_repo
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    shared = _git_repo(tmp_path / "shared")
    private = _git_repo(tmp_path / "private")
    homes = {name: tmp_path / name for name in ("default", "ops", "lab")}
    for name, home in homes.items():
        home.mkdir()
        token = set_hermes_home_override(home)
        try:
            record_worktree_repo(shared)
            if name == "lab":
                record_worktree_repo(private)
        finally:
            reset_hermes_home_override(token)

    pruned = []
    monkeypatch.setattr(worktree_ops, "_prune_stale_worktrees", lambda repo: pruned.append(repo))
    monkeypatch.setattr(sched, "_last_worktree_maintenance_at", None)
    from cron import scheduler_worktree_maintenance as claims
    monkeypatch.setattr(claims, "_last_worktree_prune_at", {})

    class _InlineThread:
        def __init__(self, target, args=(), **_kw):
            self._run = lambda: target(*args)

        def start(self):
            self._run()

    monkeypatch.setattr(sched.threading, "Thread", _InlineThread)

    for home in homes.values():  # one multiplexed tick: each served profile in its own scope
        token = set_hermes_home_override(home)
        try:
            sched._maybe_run_worktree_maintenance()
        finally:
            reset_hermes_home_override(token)

    ours = [os.path.realpath(r) for r in pruned if os.path.realpath(r) in
            {os.path.realpath(shared), os.path.realpath(private)}]
    assert sorted(ours) == sorted([os.path.realpath(shared), os.path.realpath(private)])


@pytest.mark.platforms("posix")
def test_footprint_warning_sizes_the_tree_only_when_the_count_does_not_decide(
        tmp_path, monkeypatch, caplog):
    from hermes_cli import worktree_ops

    shim_dir = tmp_path / "bin"
    shim_dir.mkdir()
    du_log = tmp_path / "du.log"
    du = shim_dir / "du"
    du.write_text(f'#!/bin/sh\necho "$@" >> "{du_log}"\necho "6000\t$2"\n')
    du.chmod(du.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{shim_dir}{os.pathsep}{os.environ['PATH']}")

    crowded = _git_repo(tmp_path / "crowded")
    for i in range(10):
        (crowded / ".worktrees" / f"lane-{i}").mkdir()
    sparse = _git_repo(tmp_path / "sparse")
    for i in range(2):
        (sparse / ".worktrees" / f"lane-{i}").mkdir()

    with caplog.at_level(logging.WARNING, logger=worktree_ops.logger.name):
        worktree_ops._warn_worktree_footprint(str(crowded))
        assert not du_log.exists(), "the count alone decides: no walk of .worktrees/"
        assert "holds 10 tree(s)" in caplog.text

        caplog.clear()
        worktree_ops._warn_worktree_footprint(str(sparse))
    assert du_log.read_text().count("-sm") == 1
    assert "holds 2 tree(s) (5.9GB)" in caplog.text
