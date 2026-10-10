"""A live worker whose recorded start time drifted is still OUR worker (#117505 follow-up).

On macOS ``psutil.Process.create_time()`` (backed by ``kern.boottime``) drifts by up to ~2 s
between two reads of the same live process, so the ``"<epoch>|<start>"`` fingerprint captured at
spawn stops comparing byte-equal. ``_pid_recycled`` compared the two strings with ``!=``, so a
verifiably live worker was reported as a recycled stranger: ``_worker_alive`` went False,
``release_stale_claims`` released the claim instead of extending it, and ``_reclaim_dead_workers``
released the claim and respawned a second worker beside it — two writers on one card, both
performing the card's external side effects.

These tests exercise the real code path against a REAL subprocess and a real temp board. Nothing
here mocks liveness: the worker is a live ``/bin/sleep`` and the fingerprints come from the same
``get_process_start_time`` the dispatcher uses.

``gateway.status.START_TIME_DRIFT_TOLERANCE`` / ``start_time_fingerprints_match`` already encode
the tolerance for the other three identity call sites; this is the fourth, which ignored them.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from gateway.status import (
    START_TIME_DRIFT_TOLERANCE,
    start_time_fingerprints_match,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def board(tmp_path, monkeypatch):
    """An isolated board with no crash grace, so reclaim decisions are unblocked."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    conn = kbc.connect(tmp_path / "kanban.db")
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def live_worker():
    """A real, unrelated-but-ours child process, reaped on teardown."""
    proc = subprocess.Popen(["/bin/sleep", "300"])
    try:
        yield proc
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()


def _drifted_fingerprint(pid: int, drift_cs: int) -> str:
    """The fingerprint the spawn recorded, read ``drift_cs`` centiseconds' worth of drift ago.

    Models ``kern.boottime`` moving under a live process: the recorded value differs from what a
    later read returns, but both describe the SAME incarnation.
    """
    recorded = kbd._process_fingerprint(pid)
    assert recorded is not None
    epoch, _, start = recorded.partition("|")
    return f"{epoch}|{int(start) - drift_cs}"


def _claimed_running(conn, *, pid: int, started_at, max_runtime=None) -> str:
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


def test_pid_recycled_tolerates_same_inception_drift(live_worker):
    """The dispatcher must classify a drifted fingerprint as OUR worker, not a stranger."""
    pid = live_worker.pid
    exact = kbd._process_fingerprint(pid)
    assert exact is not None

    # Sanity: this is the comparator contract the dispatcher is supposed to honour.
    assert start_time_fingerprints_match(
        int(exact.split("|", 1)[1]) - START_TIME_DRIFT_TOLERANCE,
        int(exact.split("|", 1)[1]),
    )

    for drift_cs in (1, 50, START_TIME_DRIFT_TOLERANCE):
        drifted = _drifted_fingerprint(pid, drift_cs)
        assert drifted != exact, "the fixture must actually drift"
        assert kbd._pid_recycled(pid, drifted) is False, (
            f"{drift_cs}cs of drift was misread as PID reuse"
        )
        assert kbd._worker_alive(pid, drifted) is True


def test_live_worker_with_drift_is_never_declared_dead(live_worker):
    """The headless symptom: a live PID plus a drifted fingerprint must read as alive."""
    pid = live_worker.pid
    for drift_cs in (1, 50, START_TIME_DRIFT_TOLERANCE):
        assert kbd._worker_alive(pid, _drifted_fingerprint(pid, drift_cs)) is True


def test_true_pid_reuse_is_still_refused(board):
    """A genuine stranger is still never signalled: a wildly different start time is not drift."""
    conn = board
    killed: list[tuple[int, int]] = []
    # A value this small was a real tick on Linux; on macOS a centisecond value this small means
    # "another epoch" — and either way it is not this process.
    tid = _claimed_running(conn, pid=os.getpid(), started_at="deadbeef-boot:1|1", max_runtime=1)

    assert kbd._worker_alive(os.getpid(), "deadbeef-boot:1|1") is False
    assert tid in kbd.enforce_max_runtime(conn, signal_fn=lambda pid, sig: killed.append((pid, sig)))
    assert killed == []
    task = kb.get_task(board, tid)
    assert task.status == "ready" and task.worker_pid is None


def test_same_epoch_far_apart_still_foreign(board, live_worker):
    """Tolerance must not become a blanket 'alive': past the window it is a stranger again.

    Guards the fix from over-correcting into a PID-reuse blind spot.
    """
    pid = live_worker.pid
    epoch, _, start = kbd._process_fingerprint(pid).partition("|")
    beyond = f"{epoch}|{int(start) - (START_TIME_DRIFT_TOLERANCE * 50)}"
    assert abs(int(start) - int(beyond.split('|', 1)[1])) > START_TIME_DRIFT_TOLERANCE
    assert kbd._pid_recycled(pid, beyond) is True


def test_expired_claim_with_live_drifted_worker_is_extended_not_released(board, live_worker):
    """REGRESSION (the reported bug): a slow-but-healthy worker's drifted fingerprint must keep its
    claim. Releasing it is what let a second writer run beside the first."""
    conn = board
    pid = live_worker.pid
    tid = _claimed_running(conn, pid=pid, started_at=_drifted_fingerprint(pid, START_TIME_DRIFT_TOLERANCE))

    assert kb.release_stale_claims(conn) == 0
    task = kb.get_task(conn, tid)
    assert task.status == "running", "a live worker must keep its claim"
    assert task.worker_pid == pid
    assert "claim_extended" in [e.kind for e in kb.list_events(conn, tid)]


def test_reclaim_dead_workers_does_not_respawn_beside_live_drifted_worker(board, live_worker):
    """REGRESSION (the reported bug): the crash sweep must not release a live worker's claim.

    ``_reclaim_dead_workers`` is the path that ends the run and respawns — the duplicate-writer
    mechanism. With a drifted-but-live fingerprint the card must stay ``running``.
    """
    conn = board
    pid = live_worker.pid
    tid = _claimed_running(conn, pid=pid, started_at=_drifted_fingerprint(pid, START_TIME_DRIFT_TOLERANCE))

    sweep = kbd._reclaim_dead_workers(conn)
    assert sweep.crashed == [], "a live worker was classified as crashed"
    assert kb.get_task(conn, tid).status == "running"


def test_real_crash_still_reclaimed_and_signalled(board):
    """The other direction: a genuinely dead worker is still reclaimed (no lost crash retry)."""
    conn = board
    proc = subprocess.Popen(["/bin/sleep", "300"])
    pid, fingerprint = proc.pid, kbd._process_fingerprint(proc.pid)
    tid = _claimed_running(conn, pid=pid, started_at=fingerprint)
    killed: list[tuple[int, int]] = []

    proc.kill()
    proc.wait()  # reaped: genuinely gone, not a zombie

    sweep = kbd._reclaim_dead_workers(conn)
    assert tid in sweep.crashed
    assert kb.get_task(conn, tid).status == "ready"
    # Dead worker: nothing left to signal.
    assert killed == []


def test_single_writer_guard_end_to_end_across_simulated_supervisor_restart(board, live_worker, tmp_path):
    """The full lifecycle in one assertion set: a worker spawned under one supervisor, whose
    recorded fingerprint drifts while a *second* supervisor instance re-derives the live value,
    never yields a second writer or a signal to a stranger.

    The two supervisors are simulated by reading the live start time through two separate
    interpreters, exactly as two gateway processes would; the DB, not the simulator, is what
    guarantees single-writer.
    """
    conn = board
    pid = live_worker.pid
    spawn_fp = kbd._process_fingerprint(pid)
    tid = _claimed_running(conn, pid=pid, started_at=spawn_fp)

    # Supervisor B comes up after the restart: it re-derives the live fingerprint in its own
    # process (its own psutil read, its own boottime), then evaluates the row A wrote.
    # ``sys.executable`` is the interpreter already running pytest, so the child necessarily has
    # psutil and this checkout on its path. Do NOT read HERMES_TEST_PYTHON here: the canonical
    # runner exports it as ``__HERMES_TEST_PYTHON`` (scripts/run_tests.sh:75) and then re-execs
    # under ``env -i`` with an explicit allowlist that omits it, so under CI neither spelling
    # survives and a bare ``python3`` fallback resolves to an interpreter without psutil.
    read_b = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, %r);"
            "from hermes_cli.kanban_db_dispatch import _process_fingerprint;"
            "print(_process_fingerprint(%d))" % (str(REPO_ROOT), pid),
        ],
        capture_output=True, text=True, timeout=30,
    )
    live_fp_b = read_b.stdout.strip()
    assert live_fp_b, f"supervisor B could not read the fingerprint: {read_b.stderr[-400:]}"

    stored = conn.execute(
        "SELECT worker_started_at FROM tasks WHERE id = ?", (tid,)
    ).fetchone()["worker_started_at"]
    assert stored == spawn_fp

    # Whatever small drift the second reader saw, it must still be OUR worker.
    assert kbd._pid_recycled(pid, live_fp_b) is False, "supervisor B saw a different process"
    assert kbd._worker_alive(pid, live_fp_b) is True

    # And the reclaim path agrees: one writer, no signal, card still owned by its live worker.
    killed: list[tuple[int, int]] = []
    assert kb.release_stale_claims(conn, signal_fn=lambda p, s: killed.append((p, s))) == 0
    assert killed == []
    assert kb.get_task(conn, tid).status == "running"
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM tasks WHERE id = ? AND status = 'running'", (tid,)
    ).fetchone()["n"] == 1

    # Cleanup proof: the harness never leaked the child.
    assert live_worker.poll() is None


def test_timeout_path_does_not_kill_a_drifted_live_worker(board, live_worker):
    """``enforce_max_runtime`` signals our worker when its fingerprint matches; with drift it must
    still recognise the process as ours and be willing to terminate it (only a stranger is refused).
    """
    conn = board
    pid = live_worker.pid
    tid = _claimed_running(
        conn, pid=pid, started_at=_drifted_fingerprint(pid, START_TIME_DRIFT_TOLERANCE), max_runtime=1
    )
    killed: list[tuple[int, int]] = []
    assert tid in kbd.enforce_max_runtime(conn, signal_fn=lambda p, s: killed.append((p, s)))
    # max_runtime is a legitimate kill of OUR OWN worker — assert it was recognised as ours.
    assert killed, "a drifted live worker must still be recognised as ours for legitimate timeout kill"
    assert killed[0] == (pid, signal.SIGTERM)
