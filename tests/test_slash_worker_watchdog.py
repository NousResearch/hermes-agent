import os
import subprocess
import sys
import time

import psutil
import pytest

from tui_gateway import slash_worker


def test_is_orphaned_true_when_ppid_changes():
    # Our parent went away and we were reparented to a subreaper/init.
    assert slash_worker._is_orphaned(1234, 1.0, getppid=lambda: 999999) is True


def test_is_orphaned_true_when_parent_create_time_mismatch():
    # Same ppid but a different create_time means the PID was reused.
    me = psutil.Process()
    assert slash_worker._is_orphaned(me.pid, 0.0, getppid=lambda: me.pid) is True


def test_is_orphaned_false_when_parent_alive_and_matches():
    me = psutil.Process()
    assert (
        slash_worker._is_orphaned(me.pid, me.create_time(), getppid=lambda: me.pid) is False
    )


def _gone(pid: int) -> bool:
    try:
        return not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    except psutil.Error:
        return True


# Worker (the grandchild): arm the watchdog against its parent, THEN announce its
# pid — so the test can't race in and kill the parent before the watchdog has
# captured the real ppid (otherwise it would anchor to init and never fire).
# Its stdout is inherited from the gateway, which is the test's pipe.
_WORKER = (
    "import os, time, psutil;"
    "from tui_gateway.slash_worker import _start_parent_death_watchdog;"
    "ppid = os.getppid();"
    "_start_parent_death_watchdog(ppid, psutil.Process(ppid).create_time());"
    "print(os.getpid(), flush=True);"
    "time.sleep(60)"
)
# Gateway stand-in (the child): spawn the worker, then idle. We SIGKILL THIS so
# the worker is genuinely orphaned (its parent gone, no atexit).
_GATEWAY = (
    "import sys, subprocess, time;"
    f"subprocess.Popen([sys.executable, '-c', {_WORKER!r}]);"
    "time.sleep(60)"
)


@pytest.mark.integration
@pytest.mark.live_system_guard_bypass  # genuinely spawns + SIGKILLs its own subtree
def test_watchdog_exits_real_worker_when_parent_dies():
    fast = {**os.environ, "HERMES_SLASH_WATCHDOG_POLL_S": "0.1", "HERMES_SLASH_WATCHDOG_GRACE_S": "0.1"}
    gateway = subprocess.Popen([sys.executable, "-c", _GATEWAY], stdout=subprocess.PIPE, text=True, env=fast)
    worker_pid = None
    try:
        worker_pid = int(gateway.stdout.readline())  # printed only after arming
        assert psutil.pid_exists(worker_pid)

        gateway.kill()  # crash the parent — no graceful atexit
        gateway.wait(timeout=5)

        deadline = time.monotonic() + 5  # generous vs the 0.2s poll+grace
        while not _gone(worker_pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert _gone(worker_pid), "orphaned worker did not self-terminate"
    finally:
        for pid in (worker_pid, gateway.pid):
            if pid:
                try:
                    os.kill(pid, 9)
                except OSError:
                    pass
