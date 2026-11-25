"""Cron worktree discovery must not launch Apple's uninstalled Git stub."""

import os

import pytest

from cron import jobs, scheduler
from hermes_cli import worktree_registry
from hermes_platform.resolver import git as git_resolver


@pytest.mark.platforms("posix")
def test_cron_discovery_does_not_launch_uninstalled_apple_git(tmp_path, monkeypatch):
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
    workdir = tmp_path / "job"
    workdir.mkdir()
    monkeypatch.setattr(worktree_registry, "registered_repos", lambda: [])
    monkeypatch.setattr(jobs, "load_jobs", lambda: [{"workdir": str(workdir)}])
    monkeypatch.setattr(scheduler, "project_root_str", lambda: str(tmp_path))

    assert git_resolver.inspect_git(cached=False).status == "clt_stub"
    assert scheduler._worktree_maintenance_repos() == []
    assert not spawn_log.exists(), f"git stub was spawned: {spawn_log.read_text()}"
