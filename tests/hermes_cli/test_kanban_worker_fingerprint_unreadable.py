"""An unreadable worker fingerprint is "unknown", never "dead".

Regression for the 2026-10-22 double-dispatch incident: ``get_process_start_time`` returned ``None``
for a live worker (transient psutil failure on macOS, where there is no ``/proc``), ``_pid_recycled``
read that as "recycled", ``_worker_alive`` reported the worker dead, and the crash sweep released the
claim beside the still-running worker — a second worker was spawned for the same card. Only a read
that SUCCEEDS and DIFFERS may count as recycled; an unreadable identity holds the claim and refuses
to signal, exactly like an UNVERIFIED spawn.
"""

import os
import signal
import subprocess
import time

import pytest

from gateway import status
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd

pytestmark = pytest.mark.platforms("posix")


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    # The bounded re-read is exercised by call counting, not by waiting.
    monkeypatch.setattr(kbd, "FINGERPRINT_READ_RETRY_SECONDS", 0, raising=False)
    conn = kbc.connect(tmp_path / "kanban.db")
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def sleeper():
    proc = subprocess.Popen(["sleep", "600"])
    try:
        yield proc
    finally:
        proc.kill()
        proc.wait()


def _claimed_running(conn, *, pid: int, started_at, max_runtime=None) -> str:
    tid = kb.create_task(conn, title="job", assignee="worker", max_runtime_seconds=max_runtime)
    kb.claim_task(conn, tid)
    kbd._set_worker_pid(conn, tid, pid)
    old = int(time.time()) - 3600
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET worker_started_at = ?, started_at = ?, claim_expires = ? WHERE id = ?",
                     (started_at, old, old, tid))
        conn.execute("UPDATE task_runs SET started_at = ? WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                     (old, tid))
    return tid


def _unreadable(monkeypatch):
    calls = []

    def read(pid):
        calls.append(pid)  # the read "fails": the fake returns None like psutil on ZombieProcess

    monkeypatch.setattr(status, "get_process_start_time", read)
    return calls


def test_live_worker_with_unreadable_fingerprint_is_alive_and_not_reclaimed(board, sleeper, monkeypatch):
    """The incident shape: a live child, a legitimate stored fingerprint, and a start-time read that
    fails. The worker is reported alive, the crash sweep leaves the claim intact, and the stale-claim
    sweep extends rather than releases it."""
    conn = board
    fingerprint = kbd._process_fingerprint(sleeper.pid)
    assert fingerprint is not None
    tid = _claimed_running(conn, pid=sleeper.pid, started_at=fingerprint)

    calls = _unreadable(monkeypatch)
    assert kbd._pid_recycled(sleeper.pid, fingerprint) is False
    # Bounded: the read was retried, not spun on.
    assert len(calls) == kbd.FINGERPRINT_READ_ATTEMPTS
    assert kbd._worker_alive(sleeper.pid, fingerprint) is True

    assert kbd.detect_crashed_workers(conn) == []
    killed = []
    assert kb.release_stale_claims(conn, signal_fn=lambda pid, sig: killed.append((pid, sig))) == 0
    assert killed == []
    task = kb.get_task(conn, tid)
    assert task.status == "running" and task.worker_pid == sleeper.pid
    assert sleeper.poll() is None


def test_transient_read_failure_recovers_within_the_retry_budget(board, sleeper, monkeypatch):
    """One failed read followed by a good one resolves to a positive match — no hold needed."""
    fingerprint = kbd._process_fingerprint(sleeper.pid)
    real_read = status.get_process_start_time
    outcomes = iter([None])

    def flaky(pid):
        try:
            return next(outcomes)
        except StopIteration:
            return real_read(pid)

    monkeypatch.setattr(status, "get_process_start_time", flaky)
    assert kbd._fingerprint_matches(sleeper.pid, fingerprint) is True
    assert kbd._worker_alive(sleeper.pid, fingerprint) is True


def test_unreadable_fingerprint_is_never_signalled_and_holds_the_claim(board, sleeper, monkeypatch):
    """Reclaim paths that would signal: the live PID is neither SIGTERMed nor released beside."""
    conn = board
    fingerprint = kbd._process_fingerprint(sleeper.pid)
    tid = _claimed_running(conn, pid=sleeper.pid, started_at=fingerprint, max_runtime=1)
    _unreadable(monkeypatch)

    killed = []
    sig = lambda pid, s: killed.append((pid, s))
    termination = kbd._terminate_reclaimed_worker(
        sleeper.pid, kb.get_task(conn, tid).claim_lock, signal_fn=sig, started_at=fingerprint)
    assert killed == []
    assert termination["signal_refused"] is True and termination["terminated"] is False
    assert termination.get("pid_recycled") is None
    assert kbd._worker_survived_termination(termination) is True

    assert kbd.enforce_max_runtime(conn, signal_fn=sig) == []
    assert killed == []
    assert kb.get_task(conn, tid).status == "running"
    assert sleeper.poll() is None


def test_positively_recycled_pid_is_still_detected(board, sleeper):
    """The other direction holds: a read that succeeds and differs IS a recycled PID. The claim is
    released, nothing is signalled, and the stranger keeps running."""
    conn = board
    live = kbd._process_fingerprint(sleeper.pid)
    epoch, _, start = live.partition("|")
    stranger = f"{epoch}|{int(start) - 10_000_000}"  # a process born ~28 h earlier under this PID
    tid = _claimed_running(conn, pid=sleeper.pid, started_at=stranger, max_runtime=1)

    assert kbd._fingerprint_matches(sleeper.pid, stranger) is False
    assert kbd._pid_recycled(sleeper.pid, stranger) is True
    assert kbd._worker_alive(sleeper.pid, stranger) is False

    killed = []
    sig = lambda pid, s: killed.append((pid, s))
    assert tid in kbd.enforce_max_runtime(conn, signal_fn=sig)
    assert killed == []
    task = kb.get_task(conn, tid)
    assert task.status == "ready" and task.worker_pid is None
    assert sleeper.poll() is None

    tid2 = _claimed_running(conn, pid=sleeper.pid, started_at=stranger)
    assert tid2 in kbd.detect_crashed_workers(conn)
    assert kb.get_task(conn, tid2).status == "ready"


def test_matching_fingerprint_is_still_signalled_on_timeout(board, sleeper):
    """Being conservative about unreadable identities must not make a verified worker unkillable."""
    conn = board
    fingerprint = kbd._process_fingerprint(sleeper.pid)
    tid = _claimed_running(conn, pid=sleeper.pid, started_at=fingerprint, max_runtime=1)
    killed = []
    assert tid in kbd.enforce_max_runtime(conn, signal_fn=lambda pid, s: killed.append((pid, s)) or os.kill(pid, s))
    assert killed and killed[0] == (sleeper.pid, signal.SIGTERM)
