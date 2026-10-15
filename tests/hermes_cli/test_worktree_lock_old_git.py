"""Worktree lock state without ``git worktree list --porcelain -z`` (git < 2.36, e.g. Ubuntu 22.04).

The pruner read every lock through ``list -z``; on an older git that fails and the fail-safe
``"live"`` answer kept every ``.worktrees`` checkout forever ("in use by a running hermes session").
"""
import os
import shutil
import subprocess
import sys

import pytest

from hermes_cli import worktree_ops
from tests.hermes_cli import test_worktree


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX git shim")
def test_stale_worktrees_are_reaped_on_git_without_list_z(tmp_path, monkeypatch):
    import cli

    git_repo = tmp_path / "repo"
    for args in (["init", str(git_repo)], ["-C", str(git_repo), "commit", "--allow-empty", "-m", "init"],
                 ["-C", str(git_repo), "update-ref", "refs/remotes/origin/main", "HEAD"]):
        subprocess.run(["git", "-c", "user.name=T", "-c", "user.email=t@t", *args], capture_output=True, check=True)

    real_git = shutil.which("git")
    shim = tmp_path / "old-git"
    shim.mkdir()
    (shim / "git").write_text(
        "#!/bin/sh\n"
        'case " $* " in *" worktree list "*" -z "*) echo "error: unknown switch \\`z\'" >&2; exit 129;; esac\n'
        f'exec "{real_git}" "$@"\n')
    (shim / "git").chmod(0o755)
    dead = test_worktree.TestWorktreeLockReaping._mk(cli, git_repo, "hermes-dead", pid=999999)
    unlocked = test_worktree.TestWorktreeLockReaping._mk(cli, git_repo, "hermes-nolock")
    live = test_worktree.TestWorktreeLockReaping._mk(cli, git_repo, "hermes-live", pid=os.getpid())
    monkeypatch.setenv("PATH", f"{shim}:{os.environ['PATH']}")
    assert subprocess.run(["git", "worktree", "list", "--porcelain", "-z"], cwd=git_repo,
                          capture_output=True, check=False).returncode == 129

    assert worktree_ops._worktree_lock_is_live(str(git_repo), str(dead)) == "dead"
    assert worktree_ops._worktree_lock_is_live(str(git_repo), str(unlocked)) is None
    cli._prune_stale_worktrees(str(git_repo))
    assert not dead.exists() and not unlocked.exists()
    assert live.exists(), "a lock whose owner runs is still live"
