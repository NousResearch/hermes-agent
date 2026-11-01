"""live_process_pids on psutil hosts: which live processes work inside a worktree.

The scan visits every process on the machine, so it must not read other processes' directories:
on macOS psutil already reports a cwd as the kernel resolves it.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

from hermes_cli.worktree_sweep import live_process_pids
from tests.home_io_guard import HomeIOGuard


def _sleeper(cwd) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], cwd=str(cwd))


@pytest.mark.platforms("macos")
def test_finds_a_tree_cwd_without_reading_other_processes_directories(tmp_path, monkeypatch):
    tree = tmp_path / "tree"
    tree.mkdir()
    # The worktree is named through a symlink, and the process inside entered it through one.
    link = tmp_path / "tree-link"
    link.symlink_to(tree, target_is_directory=True)
    # Another process works in a directory the scan has no business reading (another user's home,
    # a live Hermes home on a developer machine); the guard refuses Python I/O under it.
    foreign = tmp_path / "foreign"
    (foreign / "work").mkdir(parents=True)
    guarded = os.path.realpath(foreign)
    inside, outside = _sleeper(link), _sleeper(foreign / "work")
    try:
        HomeIOGuard(lambda: [guarded]).install(monkeypatch)
        hits = live_process_pids([link], exclude=(os.getpid(),))
    finally:
        for proc in (inside, outside):
            proc.kill()
            proc.wait()

    assert hits == {str(link): [inside.pid]}
