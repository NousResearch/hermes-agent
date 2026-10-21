"""A live worker is not reclaimed when its recorded start time is ~1 s off the current reading.

Since #140967 the start time is read from the kernel birth on macOS, so two dispatchers agree on the
same PID. A worker fingerprinted before that change (psutil's per-process boot-time frame), or by a
psutil build without the monotonic ``create_time``, can still be recorded 1 s off in either
direction. ``_pid_recycled`` compared the ``"<epoch>|<start>"`` fingerprint by exact string
equality, so such a live worker was reclaimed as ``crashed`` and the card re-dispatched beside it.
The start time now uses ``start_time_fingerprints_match`` (2 s, like every other liveness check);
the boot epoch stays exact.
"""

import os
import time

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    conn = kbc.connect(tmp_path / "kanban.db")
    try:
        yield conn
    finally:
        conn.close()


def _fp_shifted(pid: int, centis: int) -> str:
    epoch, start = kbd._process_fingerprint(pid).rsplit("|", 1)
    return f"{epoch}|{int(start) + centis}"


def _running(conn, *, pid: int, fingerprint: str) -> str:
    tid = kb.create_task(conn, title="job", assignee="worker")
    kb.claim_task(conn, tid)
    kbd._set_worker_pid(conn, tid, pid)
    old = int(time.time()) - 3600
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET worker_started_at = ?, started_at = ? WHERE id = ?",
                     (fingerprint, old, tid))
    return tid


@pytest.mark.parametrize("drift", [-100, 100, -200, 200])
def test_start_time_within_tolerance_is_the_same_worker(drift):
    pid = os.getpid()
    assert kbd._worker_alive(pid, _fp_shifted(pid, drift)) is True
    assert kbd._pid_recycled(pid, _fp_shifted(pid, drift)) is False


def test_far_start_time_foreign_epoch_or_junk_is_still_a_recycle():
    pid = os.getpid()
    epoch, start = kbd._process_fingerprint(pid).rsplit("|", 1)
    assert kbd._pid_recycled(pid, _fp_shifted(pid, 10_000)) is True
    assert kbd._pid_recycled(pid, _fp_shifted(pid, 201)) is True
    assert kbd._pid_recycled(pid, f"other-boot:1|{start}") is True
    assert kbd._pid_recycled(pid, f"{epoch}|junk") is True


def test_drifted_live_worker_is_not_reclaimed_or_charged(board):
    conn = board
    tid = _running(conn, pid=os.getpid(), fingerprint=_fp_shifted(os.getpid(), 100))
    assert kbd.detect_crashed_workers(conn) == []
    task = kb.get_task(conn, tid)
    assert task.status == "running" and task.worker_pid == os.getpid()
    assert task.consecutive_failures == 0
    assert "crashed" not in [e.kind for e in kb.list_events(conn, tid)]


def test_true_recycle_is_still_reclaimed(board):
    conn = board
    tid = _running(conn, pid=os.getpid(), fingerprint=_fp_shifted(os.getpid(), 10_000))
    assert tid in kbd.detect_crashed_workers(conn)
    assert kb.get_task(conn, tid).status != "running"
