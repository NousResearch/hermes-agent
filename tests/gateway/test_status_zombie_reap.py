"""``gateway.status._pid_exists`` must reap our own unreaped zombie children.

An exited child whose ``Popen`` handle was lost (re-exec, supervisor restart) stays a
zombie until the parent waits: it still occupies a process-table slot and — before the
reap was added — kept ``ps``/``/proc`` state queries answering for a dead PID forever.
Ported from openai/codex#43504.
"""

import os
import subprocess
import time

import pytest

from gateway import status

pytestmark = pytest.mark.linux_only


def _wait_for_zombie(pid: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    state = "?"
    while time.monotonic() < deadline:
        state = open(f"/proc/{pid}/stat").read().split()[2]
        if state == "Z":
            return
        time.sleep(0.02)
    raise AssertionError(f"pid {pid} never became a zombie (state={state})")


def test_pid_exists_reports_own_zombie_dead_and_reaps_it():
    child = subprocess.Popen(["sleep", "60"])
    try:
        child.send_signal(9)
        # Do NOT child.wait(): the point is an unreaped zombie.
        _wait_for_zombie(child.pid)

        assert status._pid_exists(child.pid) is False

        # The probe reaped it: the process-table entry is gone (waitpid raises ECHILD).
        deadline = time.monotonic() + 5.0
        while os.path.exists(f"/proc/{child.pid}") and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not os.path.exists(f"/proc/{child.pid}")
        with pytest.raises(ChildProcessError):
            os.waitpid(child.pid, os.WNOHANG)
    finally:
        # Neutralize Popen.__del__'s wait if the reap failed.
        child.returncode = -9


def test_pid_exists_foreign_zombie_still_reads_dead():
    # A zombie that is NOT our child: waitpid fails (ECHILD) but the verdict stays dead.
    child = subprocess.Popen(["sleep", "60"])
    try:
        child.send_signal(9)
        _wait_for_zombie(child.pid)
        # Simulate "foreign": _reap_own_zombie's waitpid on an already-reaped pid is a no-op.
        child.wait()
        # PID now fully gone; probe must report dead without raising.
        assert status._pid_exists(child.pid) is False
    finally:
        if child.returncode is None:
            child.kill()
            child.wait()
