"""Cron worktree discovery must not launch Apple's uninstalled Git stub."""

import os

import pytest

from cron import jobs, scheduler
from cron.scheduler_worktree_maintenance import run_sweep
from hermes_cli import worktree_ops, worktree_registry
from hermes_platform.resolver import git as git_resolver


@pytest.fixture
def stub_git_log(tmp_path, monkeypatch):
    """Make the PATH's only git a trap that logs each launch, marked as the CLT stub."""
    spawn_log = tmp_path / "git_spawns.log"
    trap_git = tmp_path / "apple_usr_bin" / "git"
    trap_git.parent.mkdir()
    trap_git.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$*\" >> '{spawn_log}'\nexit 1\n", encoding="utf-8")
    trap_git.chmod(0o755)
    # The trap stands in for /usr/bin/git without the Command Line Tools; the resolver's own
    # darwin/xcode-select verdict is covered in tests/hermes_platform/test_resolver_git.py.
    real_check = git_resolver.is_usable_git_executable
    trap = os.path.realpath(trap_git)
    monkeypatch.setattr(
        git_resolver, "is_usable_git_executable",
        lambda exe, env=None, **kw: bool(exe) and os.path.realpath(exe) != trap and real_check(exe, env, **kw),
    )
    monkeypatch.setenv("PATH", str(trap_git.parent))
    monkeypatch.setattr(scheduler, "project_root_str", lambda: str(tmp_path))
    assert git_resolver.inspect_git(cached=False).status == "clt_stub"
    return spawn_log


@pytest.mark.platforms("posix")
def test_cron_discovery_does_not_launch_uninstalled_apple_git(tmp_path, monkeypatch, stub_git_log):
    workdir = tmp_path / "job"
    workdir.mkdir()
    monkeypatch.setattr(worktree_registry, "registered_repos", list)
    monkeypatch.setattr(jobs, "load_jobs", lambda: [{"workdir": str(workdir)}])

    assert scheduler._worktree_maintenance_repos() == []
    assert not stub_git_log.exists(), f"git stub was spawned: {stub_git_log.read_text()}"


@pytest.mark.platforms("posix")
def test_cron_sweep_of_a_registered_repo_does_not_launch_uninstalled_apple_git(tmp_path, monkeypatch, stub_git_log):
    """A repo a worktree was made in earlier (before a macOS upgrade dropped the Command Line Tools)
    reaches the six-hourly prune without any job probe. The git-free reclaim of orphaned worktree
    dependency states still runs."""
    repo = tmp_path / "repo"
    (repo / ".worktrees").mkdir(parents=True)
    monkeypatch.setattr(worktree_registry, "registered_repos", lambda: [str(repo)])
    monkeypatch.setattr(jobs, "load_jobs", list)
    reclaims: list = []
    monkeypatch.setattr(worktree_ops, "_reclaim_orphan_install_states", lambda: reclaims.append(1))

    run_sweep(scheduler._worktree_maintenance_repos, interval_seconds=3600.0)
    assert not stub_git_log.exists(), f"git stub was spawned: {stub_git_log.read_text()}"
    assert reclaims == [1]
