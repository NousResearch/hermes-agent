"""Transient identity-read failures must never book a crash of a live worker (fleet R5).

The crash sweep (``_reclaim_dead_workers``) declared live workers dead whenever the spawn
fingerprint could not be re-read: ``_pid_recycled`` failed CLOSED, so one process-table / psutil
hiccup on a loaded dispatch host released the claim and counted a failure (42 of 43 'pid not
alive' reclaims were exactly that; heartbeats were 2-70 s fresh and are never consulted here).

The fix splits the question into verdicts: only a PROVEN recycle (MISMATCH) or a GONE pid may
book a crash; a live-but-UNREADABLE worker is held across a confirmation window (two probes) and
is never signalled — missing identity is refusal, not permission (#99558).
"""

import json
import os
import signal
import subprocess
import sys
import time

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    # Default: the pre-fix single-probe contract; hold tests opt into a window explicitly.
    monkeypatch.setenv("HERMES_KANBAN_DEAD_PROBE_CONFIRM_SECONDS", "0")
    kbd._DEAD_PROBE_MEMORY.clear()
    conn = kbc.connect(tmp_path / "kanban.db")
    try:
        yield conn
    finally:
        conn.close()
        kbd._DEAD_PROBE_MEMORY.clear()


def _running_with_pid(conn, *, pid, started_at, max_runtime=None) -> str:
    tid = kb.create_task(conn, title="job", assignee="worker", max_runtime_seconds=max_runtime)
    kb.claim_task(conn, tid)
    kbd._set_worker_pid(conn, tid, pid)
    old = int(time.time()) - 3600
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET worker_started_at = ?, started_at = ?, claim_expires = ? WHERE id = ?",
            (started_at, old, old, tid),
        )
        conn.execute(
            "UPDATE task_runs SET started_at = ? WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
            (old, tid),
        )
    return tid


def _real_fingerprint(pid: int) -> str:
    from gateway.drain_control import current_instantiation_epoch
    from gateway.status import get_process_start_time

    return f"{current_instantiation_epoch()}|{get_process_start_time(pid)}"


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait(timeout=15)
    return proc.pid


def _ledger_entry(conn, task_id):
    path = kbd._dead_probe_ledger_path(None)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get(task_id)


# --- verdict semantics -------------------------------------------------------


def test_unreadable_fingerprint_reads_as_alive(board, monkeypatch):
    """Our own live PID with a fingerprint whose birth CANNOT be re-read is alive-but-unverified:
    the claim is extended, not reclaimed, and nothing is signalled."""
    conn = board
    killed: list = []
    # A closed confirmation window: the default here is the single-probe escape hatch, and this
    # test is about what a transient must NOT do (reclaim / signal) while it is unconfirmed.
    monkeypatch.setenv("HERMES_KANBAN_DEAD_PROBE_CONFIRM_SECONDS", "3600")
    tid = _running_with_pid(conn, pid=os.getpid(), started_at="9e-fake|123456789")
    monkeypatch.setattr(kbd, "_process_fingerprint", lambda pid: None)

    assert kbd._worker_alive(os.getpid(), "9e-fake|123456789") is True
    assert kbd._pid_recycled(os.getpid(), "9e-fake|123456789") is False
    assert kbd._worker_is_signal_eligible(os.getpid(), "9e-fake|123456789") is False
    assert kbd.detect_crashed_workers(conn) == []
    assert kb.get_task(conn, tid).status == "running"
    assert kb.release_stale_claims(conn, signal_fn=lambda pid, sig: killed.append((pid, sig))) == 0
    assert killed == []


def test_start_time_read_failure_reads_as_alive(board, monkeypatch):
    """Legacy integer fingerprint: a read failure (None) is UNREADABLE, not 'recycled'."""
    conn = board
    tid = _running_with_pid(conn, pid=os.getpid(), started_at=1)
    monkeypatch.setattr("gateway.status.get_process_start_time", lambda pid: None)

    state, probe = kbd.probe_worker_identity(os.getpid(), 1)
    assert state == kbd.WORKER_IDENTITY_UNREADABLE
    assert probe["read_failure"] == "start_time_unreadable"
    assert kbd._worker_alive(os.getpid(), 1) is True
    assert kb.get_task(conn, tid).status == "running"


def test_proven_mismatch_still_reclaims_immediately(board):
    """A live PID with a DIFFERENT verifiable fingerprint is a proven recycle: single probe,
    immediate reclaim, never signalled (behaviour preserved from the fingerprint contract)."""
    conn = board
    killed: list = []
    stranger = _real_fingerprint(os.getpid()) + "-foreign"
    tid = _running_with_pid(conn, pid=os.getpid(), started_at=stranger, max_runtime=1)

    assert kbd.detect_crashed_workers(conn) == [tid]
    task = kb.get_task(conn, tid)
    assert task.status == "ready" and task.worker_pid is None
    assert kbd.enforce_max_runtime(conn, signal_fn=lambda pid, sig: killed.append((pid, sig))) == []
    assert killed == []


def test_gone_pid_reclaims_immediately(board):
    """A pid that is verifiably NOT in the process table books on the first probe (control)."""
    conn = board
    tid = _running_with_pid(conn, pid=_dead_pid(), started_at=_real_fingerprint(os.getpid()))
    assert kbd.detect_crashed_workers(conn) == [tid]
    assert kb.get_task(conn, tid).status == "ready"


# --- two-probe confirmation ----------------------------------------------------


def test_unreadable_worker_is_held_then_reclaimed_on_confirmation(board, monkeypatch):
    """One transient read failure defers the reclaim; the second verdict past the confirmation
    window books it, with probe diagnostics on the event."""
    conn = board
    monkeypatch.setenv("HERMES_KANBAN_DEAD_PROBE_CONFIRM_SECONDS", "3600")
    monkeypatch.setattr(kbd, "_process_fingerprint", lambda pid: None)
    tid = _running_with_pid(conn, pid=os.getpid(), started_at="9e-fake|123456789")

    assert kbd.detect_crashed_workers(conn) == []          # tick 1: hold
    assert kb.get_task(conn, tid).status == "running"
    assert _ledger_entry(conn, tid) is not None
    assert kbd.detect_crashed_workers(conn) == []          # tick 2 inside the window: still held
    assert kb.get_task(conn, tid).status == "running"

    # Age the hold past the window (what tick N+1 ~60 s later amounts to).
    entry = _ledger_entry(conn, tid)
    entry["first_seen"] = time.time() - 3601
    path = kbd._dead_probe_ledger_path(None)
    path.write_text(json.dumps({tid: entry}), encoding="utf-8")

    assert kbd.detect_crashed_workers(conn) == [tid]       # confirmed: book it
    assert kb.get_task(conn, tid).status == "ready"
    assert _ledger_entry(conn, tid) is None                # hold consumed

    events = [e for e in kb.list_events(conn, tid) if e.kind == "crashed"]
    assert events, [e.kind for e in kb.list_events(conn, tid)]
    probe = events[-1].payload["probe"]
    assert probe["recorded"] == "9e-fake|123456789"
    assert probe["recomputed"] is None
    assert probe["read_failure"] == "fingerprint_unreadable"


def test_read_recovery_drops_the_hold(board, monkeypatch):
    """When the birth read recovers and matches, the worker is simply alive: no crash was ever
    booked and the stale hold is dropped."""
    conn = board
    monkeypatch.setenv("HERMES_KANBAN_DEAD_PROBE_CONFIRM_SECONDS", "3600")
    real = _real_fingerprint(os.getpid())
    tid = _running_with_pid(conn, pid=os.getpid(), started_at=real)
    monkeypatch.setattr(kbd, "_process_fingerprint", lambda pid: None)

    assert kbd.detect_crashed_workers(conn) == []
    assert _ledger_entry(conn, tid) is not None

    monkeypatch.setattr(kbd, "_process_fingerprint", lambda pid: _real_fingerprint(pid))
    assert kbd.detect_crashed_workers(conn) == []          # alive again: nothing to book
    task = kb.get_task(conn, tid)
    assert task.status == "running" and task.worker_pid is not None
    assert _ledger_entry(conn, tid) is None                # hold dropped on recovery


def test_confirmation_survives_a_fresh_dispatcher_process(board, monkeypatch):
    """The ledger is on disk: a separate ``hermes kanban dispatch`` process (empty memory) can
    confirm the previous tick's hold."""
    conn = board
    monkeypatch.setenv("HERMES_KANBAN_DEAD_PROBE_CONFIRM_SECONDS", "3600")
    monkeypatch.setattr(kbd, "_process_fingerprint", lambda pid: None)
    tid = _running_with_pid(conn, pid=os.getpid(), started_at="9e-fake|123456789")

    assert kbd.detect_crashed_workers(conn) == []
    entry = _ledger_entry(conn, tid)
    entry["first_seen"] = time.time() - 3601
    kbd._dead_probe_ledger_path(None).write_text(json.dumps({tid: entry}), encoding="utf-8")
    kbd._DEAD_PROBE_MEMORY.clear()                          # simulated new process

    assert kbd.detect_crashed_workers(conn) == [tid]


def test_stale_hold_for_a_previous_incarnation_never_confirms(board, monkeypatch):
    """A hold recorded against an older pid/fingerprint must not confirm the current worker's
    verdict at its first sighting."""
    conn = board
    monkeypatch.setenv("HERMES_KANBAN_DEAD_PROBE_CONFIRM_SECONDS", "3600")
    tid = _running_with_pid(conn, pid=os.getpid(), started_at="9e-fake|123456789")
    old = {"first_seen": time.time() - 9999, "pid": 4242, "fingerprint": "old"}
    kbd._dead_probe_ledger_path(None).parent.mkdir(parents=True, exist_ok=True)
    kbd._dead_probe_ledger_path(None).write_text(json.dumps({tid: old}), encoding="utf-8")
    monkeypatch.setattr(kbd, "_process_fingerprint", lambda pid: None)

    assert kbd.detect_crashed_workers(conn) == []          # re-noted against THIS incarnation
    entry = _ledger_entry(conn, tid)
    assert entry["pid"] == os.getpid() and entry["fingerprint"] == "9e-fake|123456789"


def test_window_zero_restores_single_probe_booking(board, monkeypatch):
    """The escape hatch (tests / operators) books an unreadable dead verdict on one tick, and the
    event still carries the probe diagnostics."""
    conn = board
    monkeypatch.setattr(kbd, "_process_fingerprint", lambda pid: None)
    tid = _running_with_pid(conn, pid=os.getpid(), started_at="9e-fake|123456789")

    assert kbd.detect_crashed_workers(conn) == [tid]
    events = [e for e in kb.list_events(conn, tid) if e.kind == "crashed"]
    assert events and events[-1].payload["probe"]["read_failure"] == "fingerprint_unreadable"


# --- signal safety on the unreadable path -------------------------------------


def test_termination_refuses_unreadable_identity(board, monkeypatch):
    """The fail-open reclaim must NOT turn into a licence to signal by number: an unreadable live
    worker is reported as surviving (signal_refused), never signalled."""
    killed: list = []
    monkeypatch.setattr(kbd, "_process_fingerprint", lambda pid: None)
    info = kbd._terminate_reclaimed_worker(
        os.getpid(), f"{kb._host_prefix()}:x",
        signal_fn=lambda pid, sig: killed.append((pid, sig)),
        started_at="9e-fake|123456789",
    )
    assert killed == []
    assert info["termination_attempted"] is False
    assert info.get("signal_refused") is True
    assert info.get("identity_unreadable") is True
    assert info["terminated"] is False


def test_timeout_path_never_signals_an_unreadable_worker(board, monkeypatch):
    """max-runtime enforcement over a live-but-unreadable worker: no SIGTERM, no SIGKILL, the card
    stays held for the reclaim sweep to confirm."""
    conn = board
    monkeypatch.setenv("HERMES_KANBAN_DEAD_PROBE_CONFIRM_SECONDS", "3600")
    monkeypatch.setattr(kbd, "_process_fingerprint", lambda pid: None)
    killed: list = []
    tid = _running_with_pid(
        conn, pid=os.getpid(), started_at="9e-fake|123456789", max_runtime=1,
    )
    assert kbd.enforce_max_runtime(conn, signal_fn=lambda pid, sig: killed.append((pid, sig))) == []
    assert killed == []
    assert kb.get_task(conn, tid).status == "running"
