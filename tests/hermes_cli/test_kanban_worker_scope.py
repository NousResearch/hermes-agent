"""Process-scope regressions for Kanban worker termination."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import pytest

from hermes_cli import kanban_db as kb


def _assert_group_kill_reaps_descendant() -> None:
    child_source = "import time; time.sleep(60)"
    leader_source = (
        "import subprocess, sys, time\n"
        f"child = subprocess.Popen([sys.executable, '-c', {child_source!r}])\n"
        "print(child.pid, flush=True)\n"
        "time.sleep(60)\n"
    )
    leader = subprocess.Popen(
        [sys.executable, "-c", leader_source],
        start_new_session=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    pgid = os.getpgid(leader.pid)
    try:
        assert leader.stdout is not None
        child_pid = int(leader.stdout.readline().strip())
        members = kb._group_member_pids(pgid)
        assert leader.pid in members
        assert child_pid in members
        assert kb._worker_alive(leader.pid, pgid)

        # Reproduce the original PID-only failure mode: the leader dies while
        # its descendant remains in the worker's session.
        os.kill(leader.pid, signal.SIGKILL)
        leader.wait(timeout=5)
        assert kb._group_alive(pgid)

        assert kb._kill_worker_scope(leader.pid, pgid, signal.SIGKILL)
        deadline = time.monotonic() + 5
        while kb._group_alive(pgid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not kb._group_alive(pgid)
    finally:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        if leader.poll() is None:
            leader.kill()
            leader.wait(timeout=5)


@pytest.mark.linux_only
def test_linux_worker_group_kill_reaps_descendant():
    _assert_group_kill_reaps_descendant()


@pytest.mark.macos_only
def test_macos_worker_group_kill_reaps_descendant():
    _assert_group_kill_reaps_descendant()
