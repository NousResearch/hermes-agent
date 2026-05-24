"""Tests for gateway.file_lock — cross-platform file locking utility."""

import os
import sys
import threading
import time
from pathlib import Path

import pytest

from gateway.file_lock import FileLock, LockTimeout


def test_acquire_and_release_non_blocking(tmp_path: Path) -> None:
    lock = FileLock(tmp_path / "test.lock")
    with lock(timeout=0):
        assert lock.path.exists()
    # After release, another acquire should succeed immediately
    with lock(timeout=0):
        pass


def test_timeout_raises_when_lock_held(tmp_path: Path) -> None:
    lock1 = FileLock(tmp_path / "test.lock")
    lock2 = FileLock(tmp_path / "test.lock")

    lock1.acquire(timeout=0)
    try:
        with pytest.raises(LockTimeout):
            lock2.acquire(timeout=0)
    finally:
        lock1.release()


def test_timeout_waits_and_acquires(tmp_path: Path) -> None:
    lock1 = FileLock(tmp_path / "test.lock")
    lock2 = FileLock(tmp_path / "test.lock")

    lock1.acquire(timeout=0)
    acquired = threading.Event()
    waiter_result = {"acquired": False, "error": None}

    def waiter() -> None:
        try:
            with lock2(timeout=2, retry_interval=0.05):
                waiter_result["acquired"] = True
        except Exception as e:
            waiter_result["error"] = str(e)
        finally:
            acquired.set()

    t = threading.Thread(target=waiter)
    t.start()

    time.sleep(0.3)
    lock1.release()
    acquired.wait(timeout=5)

    assert waiter_result["acquired"], f"Waiter failed: {waiter_result['error']}"
    t.join(timeout=5)


def test_auto_release_on_exception(tmp_path: Path) -> None:
    lock = FileLock(tmp_path / "test.lock")

    with pytest.raises(ValueError):
        with lock(timeout=0):
            raise ValueError("boom")

    # Should be able to re-acquire immediately after exception release
    with lock(timeout=0):
        pass


def test_lock_file_created_in_parent(tmp_path: Path) -> None:
    lock_path = tmp_path / "sub" / "dir" / "test.lock"
    lock = FileLock(lock_path)
    with lock(timeout=0):
        assert lock_path.exists()


def test_repr_shows_state(tmp_path: Path) -> None:
    lock = FileLock(tmp_path / "test.lock")
    assert "unlocked" in repr(lock)
    lock.acquire(timeout=0)
    assert "locked" in repr(lock)
    lock.release()
    assert "unlocked" in repr(lock)


def test_concurrent_processes_only_one_holds_lock(tmp_path: Path) -> None:
    """Verify that two processes cannot hold the lock simultaneously."""
    import subprocess

    lock_path = tmp_path / "concurrent.lock"
    result_path = tmp_path / "result.txt"
    worker = tmp_path / "worker.py"
    cwd = os.getcwd()
    # Worker: try to acquire, write 'GOT' to result file if successful,
    # hold for a short time, then release.
    worker.write_text(
        f"""
import sys, time, os
sys.path.insert(0, {cwd!r})
os.environ["HERMES_HOME"] = {str(tmp_path)!r}
from pathlib import Path
from gateway.file_lock import FileLock, LockTimeout
lock = FileLock({str(lock_path)!r})
try:
    with lock(timeout=0):
        Path({str(result_path)!r}).write_text("GOT", encoding="utf-8")
        time.sleep(0.5)  # hold the lock briefly
except LockTimeout:
        Path({str(result_path)!r}).write_text("BLOCKED", encoding="utf-8")
""",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = cwd

    # Start first process — it should acquire the lock and hold for 0.5s
    p1 = subprocess.Popen(
        [sys.executable, str(worker)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=cwd,
    )
    time.sleep(0.2)  # let p1 acquire and hold

    # Start second process WHILE p1 still holds — it should be BLOCKED
    p2 = subprocess.Popen(
        [sys.executable, str(worker)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=cwd,
    )
    # Wait for both to finish
    _, stderr1 = p1.communicate(timeout=15)
    _, stderr2 = p2.communicate(timeout=15)

    if not result_path.exists():
        # Debug: show subprocess stderr to diagnose import failures
        raise AssertionError(
            f"result.txt not created. p1 stderr: {stderr1.decode()!r}, "
            f"p2 stderr: {stderr2.decode()!r}"
        )

    result = result_path.read_text(encoding="utf-8")
    # p1 wrote GOT, then p2 wrote BLOCKED — last write wins
    assert "BLOCKED" in result, f"Expected BLOCKED but got: {result}"


@pytest.mark.skipif(sys.platform == "win32", reason="fcntl is POSIX-only")
def test_lock_released_on_process_death(tmp_path: Path) -> None:
    """Verify that a lock held by a child process is released when it dies."""
    import subprocess

    script = tmp_path / "holder.py"
    script.write_text(
        f"""
import time, os, sys
sys.path.insert(0, {os.getcwd()!r})
from gateway.file_lock import FileLock
lock = FileLock({str(tmp_path / 'death.lock')!r})
lock.acquire(timeout=0)
# Write PID so parent can verify we held it
(Path({str(tmp_path)!r}) / "holder.pid").write_text(str(os.getpid()))
time.sleep(60)  # hold lock until killed
""",
        encoding="utf-8",
    )

    proc = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(0.5)  # let child acquire lock
    proc.kill()
    proc.wait(timeout=5)

    # Parent should be able to acquire immediately
    lock = FileLock(tmp_path / "death.lock")
    with lock(timeout=0):
        pass  # success — lock was auto-released
