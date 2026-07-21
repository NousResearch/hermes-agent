"""Tests for agent/process_killer.py - process tree termination."""
import os
import signal
import subprocess
import sys
import time

import pytest

from agent.process_killer import kill_process_tree, kill_processes, get_remaining_processes

pytestmark = pytest.mark.live_system_guard_bypass


def _spawn_sleep_process(seconds=30):
    """Spawn a process that sleeps, returning its PID."""
    proc = subprocess.Popen(
        ["sleep", str(seconds)] if sys.platform != "win32" else ["timeout", str(seconds)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc


def _spawn_nested_process(seconds=30):
    """Spawn a process that spawns a child, returning the parent PID."""
    if sys.platform == "win32":
        # Windows: cmd /c "start /b timeout 30 & timeout 30"
        proc = subprocess.Popen(
            ["cmd", "/c", f"start /b timeout {seconds} & timeout {seconds}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        # Unix: bash -c "sleep 30 & sleep 30 & wait"
        proc = subprocess.Popen(
            ["bash", "-c", f"sleep {seconds} & sleep {seconds} & wait"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return proc


def _is_process_alive(pid):
    """Check if a process is alive."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False


class TestKillProcessTree:
    def test_kill_single_process(self):
        proc = _spawn_sleep_process()
        time.sleep(0.3)  # let it start
        assert _is_process_alive(proc.pid)

        result = kill_process_tree(proc.pid, timeout=3.0)
        time.sleep(0.3)

        assert not _is_process_alive(proc.pid)
        assert result["method"] in ("psutil", "fallback")
        assert proc.pid in result["killed_pids"] or not _is_process_alive(proc.pid)

    def test_kill_nonexistent_process(self):
        # PID 999999 is very unlikely to exist
        result = kill_process_tree(999999, timeout=1.0)
        assert result["survived"] == []
        assert result["killed_pids"] == []

    def test_kill_nested_process_tree(self):
        proc = _spawn_nested_process()
        time.sleep(0.5)  # let children start
        assert _is_process_alive(proc.pid)

        result = kill_process_tree(proc.pid, timeout=5.0)
        time.sleep(0.5)

        assert not _is_process_alive(proc.pid)
        assert len(result["killed_pids"]) >= 1

    @pytest.mark.skipif(sys.platform == "win32", reason="Process group tests are Unix-only")
    def test_kill_process_group_unix(self):
        proc = subprocess.Popen(
            ["bash", "-c", "sleep 30 & sleep 30 & wait"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,  # new process group
        )
        time.sleep(0.5)
        pgid = os.getpgid(proc.pid)
        assert _is_process_alive(proc.pid)

        result = kill_process_tree(proc.pid, timeout=5.0)
        time.sleep(0.5)

        assert not _is_process_alive(proc.pid)


class TestKillProcesses:
    def test_kill_multiple_processes(self):
        proc1 = _spawn_sleep_process()
        proc2 = _spawn_sleep_process()
        time.sleep(0.3)

        results = kill_processes([proc1.pid, proc2.pid], timeout=3.0)
        time.sleep(0.3)

        assert len(results) == 2
        assert not _is_process_alive(proc1.pid)
        assert not _is_process_alive(proc2.pid)

    def test_kill_empty_list(self):
        results = kill_processes([], timeout=1.0)
        assert results == []


class TestGetRemainingProcesses:
    def test_no_remaining(self):
        result = get_remaining_processes([999999])
        assert result == []

    def test_remaining_alive(self):
        proc = _spawn_sleep_process()
        time.sleep(0.3)
        try:
            remaining = get_remaining_processes([proc.pid])
            assert len(remaining) == 1
            assert remaining[0]["pid"] == proc.pid
        finally:
            kill_process_tree(proc.pid, timeout=3.0)
            time.sleep(0.3)
