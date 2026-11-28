"""A Linux thread id is never a live recorded process.

``psutil.Process(tid)`` succeeds for any thread id (``/proc/<tid>`` resolves even though it is not
listed), so a recorded PID that the kernel recycled as another process's thread used to read as a
live, matching process in ``_pid_alive_matches``.
"""

import io
import os
import sys
import threading
import types
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from hermes_cli import process_identity as pi


@contextmanager
def _live_thread_id():
    """Native id of a running non-leader thread of this process."""
    ready, release = threading.Event(), threading.Event()
    tid = []

    def run():
        tid.append(threading.get_native_id())
        ready.set()
        release.wait(10)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    assert ready.wait(10)
    try:
        yield tid[0]
    finally:
        release.set()
        worker.join(10)


def _fake_psutil_alive():
    """A psutil whose Process() says every pid is a running, non-zombie process."""

    class _Proc:
        def __init__(self, pid):
            self.pid = pid

        def create_time(self):
            return 1.0

        def is_running(self):
            return True

        def status(self):
            return "running"

    class _NoSuchProcess(Exception):
        pass

    return types.SimpleNamespace(
        Process=_Proc, NoSuchProcess=_NoSuchProcess, STATUS_ZOMBIE="zombie")


def _status_open(tgid: int):
    real_open = open

    def fake_open(path, *args, **kwargs):
        if str(path).startswith("/proc/") and str(path).endswith("/status"):
            return io.StringIO(f"Name:\tsomething\nTgid:\t{tgid}\nPid:\t4242\n")
        return real_open(path, *args, **kwargs)

    return fake_open


@pytest.mark.platforms("linux")
def test_real_thread_id_is_not_a_live_process():
    psutil = pytest.importorskip("psutil")
    with _live_thread_id() as tid:
        assert tid != os.getpid()
        # The premise: psutil itself does not list the tid, yet Process(tid) resolves.
        assert psutil.pid_exists(tid) is False
        birth = pi.process_birth(psutil.Process(tid))

        assert pi._is_linux_thread_id(tid) is True
        assert pi._pid_alive_matches(tid, None) is False
        assert pi._pid_alive_matches(tid, birth) is False
        assert pi._pid_alive_matches(tid, birth, strict=True) is False


@pytest.mark.platforms("linux")
def test_own_pid_still_matches():
    psutil = pytest.importorskip("psutil")
    pid = os.getpid()
    birth = pi.process_birth(psutil.Process(pid))

    assert pi._is_linux_thread_id(pid) is False
    assert pi._pid_alive_matches(pid, None) is True
    assert pi._pid_alive_matches(pid, birth, strict=True) is True


@pytest.mark.platforms("linux")
def test_mismatched_tgid_is_rejected_even_when_psutil_says_alive():
    with patch("builtins.open", _status_open(tgid=4241)), \
         patch.dict(sys.modules, {"psutil": _fake_psutil_alive()}):
        assert pi._is_linux_thread_id(4242) is True
        assert pi._pid_alive_matches(4242, None) is False


@pytest.mark.platforms("linux")
def test_matching_tgid_keeps_psutil_verdict():
    with patch("builtins.open", _status_open(tgid=4242)), \
         patch.dict(sys.modules, {"psutil": _fake_psutil_alive()}):
        assert pi._is_linux_thread_id(4242) is False
        assert pi._pid_alive_matches(4242, None) is True


@pytest.mark.platforms("linux")
def test_unreadable_proc_status_is_no_proof():
    def no_proc(path, *args, **kwargs):
        raise PermissionError(path)

    with patch("builtins.open", no_proc):
        assert pi._is_linux_thread_id(4242) is False


@pytest.mark.platforms("macos", "windows")
def test_non_linux_never_reads_proc():
    def must_not_open(path, *args, **kwargs):
        raise AssertionError(f"opened {path}")

    with patch("builtins.open", must_not_open):
        assert pi._is_linux_thread_id(os.getpid()) is False
