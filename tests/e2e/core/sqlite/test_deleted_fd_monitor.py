"""Kernel FD monitor contract without a database or Hermes process.

Only Linux exposes the ``/proc/<pid>/fd`` links inspected by Chamber. The child
opens ordinary named files so these tests isolate observer timing from SQLite.
"""

from __future__ import annotations

import os
import select
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from tests.e2e.core.sqlite._helpers import Chamber, SHM_CLOSE_GRACE_SECONDS


pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="requires /proc/<pid>/fd")

_CHILD = r"""
import os
import sys

path = sys.argv[1]
fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
print(f"ready:{fd}:{os.fstat(fd).st_ino}", flush=True)
for command in sys.stdin:
    command = command.strip()
    if command == "unlink":
        os.unlink(path)
        print(f"unlinked:{fd}:{os.fstat(fd).st_ino}", flush=True)
    elif command == "close":
        os.close(fd)
        fd = -1
        print("closed", flush=True)
    elif command == "reopen_unlink":
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        os.unlink(path)
        print(f"reopened_unlinked:{fd}:{os.fstat(fd).st_ino}", flush=True)
    else:
        raise AssertionError(f"unknown command: {command}")
"""


def _line(proc: subprocess.Popen[str]) -> str:
    ready, _, _ = select.select([proc.stdout], [], [], 5.0)
    assert ready, "owned FD child did not acknowledge command"
    assert proc.stdout is not None
    line = proc.stdout.readline().strip()
    assert line, f"owned FD child exited unexpectedly (rc={proc.poll()})"
    return line


def _command(proc: subprocess.Popen[str], command: str) -> str:
    assert proc.stdin is not None
    proc.stdin.write(command + "\n")
    proc.stdin.flush()
    return _line(proc)


@contextmanager
def _observer(path: Path):
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", _CHILD, str(path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert _line(proc).startswith("ready:")
        chamber = Chamber.__new__(Chamber)
        chamber.db = path.with_name("state.db")
        chamber.mode = "wal"
        chamber.procs = {"owned-fd-child": proc}
        chamber._lock = threading.Lock()
        chamber.deleted_hits = []
        yield chamber, proc
    finally:
        if proc.poll() is None:
            proc.kill()  # Only the child created by this fixture.
        proc.wait(timeout=5)
        if proc.stdin is not None:
            proc.stdin.close()
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.stderr is not None:
            proc.stderr.close()


def test_transient_shm_close_and_fd_reuse_do_not_inherit_observation(tmp_path: Path) -> None:
    path = tmp_path / "state.db-shm"
    pending: dict = {}
    with _observer(path) as (chamber, proc):
        # Keep the original inode alive so the reopened file must be a new inode.
        original = os.open(path, os.O_RDONLY)
        try:
            first_fd = int(_command(proc, "unlink").split(":")[1])
            chamber._scan_deleted_fds(pending)
            assert not chamber.deleted_hits_snapshot()
            assert _command(proc, "close") == "closed"
            reopened = _command(proc, "reopen_unlink").split(":")
            assert reopened[0] == "reopened_unlinked"
            assert int(reopened[1]) == first_fd
            assert int(reopened[2]) != os.fstat(original).st_ino
            # The old observation has aged, but the reused FD names a new inode.
            time.sleep(SHM_CLOSE_GRACE_SECONDS * 1.5)
            chamber._scan_deleted_fds(pending)
            assert not chamber.deleted_hits_snapshot(), "a new inode inherited the old FD's grace period"
            assert _command(proc, "close") == "closed"
            time.sleep(SHM_CLOSE_GRACE_SECONDS * 1.5)
            chamber._scan_deleted_fds(pending)
            assert not chamber.deleted_hits_snapshot(), "a transient SHM close became a durable hit"
        finally:
            os.close(original)


@pytest.mark.parametrize(
    ("suffix", "requires_grace"),
    [("", False), ("-wal", False), ("-shm", True)],
)
def test_persistent_deleted_fd_is_reported(tmp_path: Path, suffix: str, requires_grace: bool) -> None:
    path = tmp_path / f"state.db{suffix}"
    pending: dict = {}
    with _observer(path) as (chamber, proc):
        assert _command(proc, "unlink").startswith("unlinked:")
        chamber._scan_deleted_fds(pending)
        if requires_grace:
            assert not chamber.deleted_hits_snapshot()
            time.sleep(SHM_CLOSE_GRACE_SECONDS * 1.5)
            chamber._scan_deleted_fds(pending)
        expected_hits = [
            ("owned-fd-child", proc.pid, f"{path} (deleted)")
        ]
        assert chamber.deleted_hits_snapshot() == expected_hits
        proc.kill()  # Reap only this test's child; a confirmed hit remains evidence.
        proc.wait(timeout=5)
        chamber._scan_deleted_fds(pending)
        assert chamber.deleted_hits_snapshot() == expected_hits
