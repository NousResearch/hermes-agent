"""Cron worktree discovery must not launch Apple's uninstalled Git stub."""

import os
import subprocess
from types import SimpleNamespace

from cron import jobs, scheduler
from hermes_cli import worktree_registry
from hermes_platform.resolver import git as git_resolver


def test_cron_discovery_does_not_launch_uninstalled_apple_git(tmp_path, monkeypatch):
    monkeypatch.setattr(git_resolver, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.delenv("DEVELOPER_DIR", raising=False)
    monkeypatch.setattr(git_resolver, "_CLT_CACHE", git_resolver._CltGitCache())
    monkeypatch.setattr(worktree_registry, "registered_repos", lambda: [])
    monkeypatch.setattr(jobs, "load_jobs", lambda: [{"workdir": str(tmp_path)}])
    monkeypatch.setattr(scheduler, "project_root_str", lambda: str(tmp_path))

    xcode_select = tmp_path / "xcode-select"
    xcode_select.write_text("#!/bin/sh\nexit 1\n")
    xcode_select.chmod(0o755)
    original_run = subprocess.run
    git_launches = []

    def run(argv, **kwargs):
        if argv == ["/usr/bin/xcode-select", "-p"]:
            return original_run([str(xcode_select), "-p"], **kwargs)
        if os.path.basename(argv[0]) == "git":
            git_launches.append(argv[1:])
            return subprocess.CompletedProcess(argv, 1, "", "")
        return original_run(argv, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    assert git_resolver.inspect_git(cached=False).status == "clt_stub"
    assert scheduler._worktree_maintenance_repos() == []
    assert not git_launches
