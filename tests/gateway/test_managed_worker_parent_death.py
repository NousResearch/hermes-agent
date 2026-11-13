"""A managed worker dies with its owner, descendants included, even when its turn ignores the
interrupt (andrexibiza 29 / P3.g). The owner spawns it in its own session; the owner's death
closes the control pipe, and the worker hard-kills its process group after a short grace."""
import contextlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import textwrap
import time

import psutil
import pytest

ROOT = Path(__file__).resolve().parents[2]

# A worker past bootstrap whose agent ignores interrupt and whose turn blocks forever in a tool
# that has started a descendant (the real WorkerControls reads the owner's pipe).
WORKER = textwrap.dedent('''
    import subprocess, sys, threading, time
    import agent.managed_worker as mw
    mw.ORPHAN_GRACE_SECONDS = 1
    grandchild = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
    print(grandchild.pid, flush=True)
    controls = mw.WorkerControls(channel=None, route='r')
    controls.agent = type('Stubborn', (), {'interrupt': lambda self, *a, **k: None})()
    while True:
        time.sleep(1)
''')

# The owner side: the real spawn options, then it is killed without any cleanup.
OWNER = textwrap.dedent('''
    import subprocess, sys
    from gateway.session_managed_worker import WORKER_POPEN
    worker = subprocess.Popen([sys.executable, '-c', sys.argv[1]], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, **WORKER_POPEN)
    print(worker.pid, worker.stdout.readline().decode().strip(), flush=True)
    worker.wait()
''')


def _gone(pid, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                return True
        except psutil.NoSuchProcess:
            return True
        time.sleep(.1)
    return False


@pytest.mark.platforms("linux", "macos")
@pytest.mark.live_system_guard_bypass  # cleanup may signal our own descendants after init adopted them
def test_worker_and_its_descendants_die_when_the_owner_is_killed():
    env = {**os.environ, 'PYTHONPATH': str(ROOT)}
    owner = subprocess.Popen([sys.executable, '-c', OWNER, WORKER], cwd=ROOT, env=env,
                             stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
    worker_pid = grandchild_pid = None
    try:
        worker_pid, grandchild_pid = map(int, owner.stdout.readline().split())
        assert os.getpgid(worker_pid) == worker_pid  # its own session, not the owner's group
        owner.kill()
        owner.wait(timeout=5)
        assert _gone(worker_pid, 15), 'the orphaned worker outlived its owner'
        assert _gone(grandchild_pid, 5), "the worker's descendant outlived it"
    finally:
        owner.stdout.close()
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=5)
        for pid in (worker_pid, grandchild_pid):
            if pid is not None and not _gone(pid, 0):
                with contextlib.suppress(ProcessLookupError):
                    os.kill(pid, signal.SIGKILL)  # windows-footgun: ok — POSIX-only test
