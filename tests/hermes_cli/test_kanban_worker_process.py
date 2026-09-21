from __future__ import annotations

import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import psutil
import pytest

from hermes_cli import kanban_worker_process as worker_process


def test_worker_identity_requires_hermes_entry_and_exact_task_query(monkeypatch):
    for argv, expected in (
        ([sys.executable, "-m", "hermes_cli.main", "-p", "worker", "--cli", "chat", "-q", "work kanban task t_one"], True),
        ([sys.executable, "other.py", "hermes", "work kanban task t_one"], False),
        ([sys.executable, "-m", "hermes_cli.main", "chat", "-q", "work kanban task t_one_extra"], False),
    ):
        monkeypatch.setattr(psutil, "Process", lambda _pid: SimpleNamespace(cmdline=lambda: argv))
        assert worker_process.pid_matches_task_worker(123, "t_one") is expected


@pytest.mark.macos_only
def test_worker_reclaim_signal_reaches_real_child_process():
    command = (
        "import subprocess,sys,time; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        "print(child.pid,flush=True); time.sleep(30)"
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", command], stdout=subprocess.PIPE,
        text=True, start_new_session=True,
    )
    try:
        child_pid = int(parent.stdout.readline())
        worker_process.signal_worker_tree(parent.pid, signal.SIGTERM)
        parent.wait(timeout=5)
        deadline = time.monotonic() + 5
        while psutil.pid_exists(child_pid):
            try:
                if psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE:
                    break
            except psutil.NoSuchProcess:
                break
            assert time.monotonic() < deadline, "worker child survived group termination"
            time.sleep(0.05)
    finally:
        try:
            worker_process.signal_worker_tree(parent.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        parent.wait(timeout=5)
        parent.stdout.close()
