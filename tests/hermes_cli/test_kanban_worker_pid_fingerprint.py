"""A recycled worker PID is never mistaken for our worker.

``tasks.worker_pid`` survives a reboot; the number can then belong to an unrelated process. Every
liveness decision (extend/defer the claim) and every kill (SIGTERM/SIGKILL on timeout or reclaim)
must require the spawn-time start fingerprint to match, never bare PID existence.
"""

import os
import signal
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


def test_recycled_pid_is_reclaimed_without_being_signalled(board):
    """Our own live PID with a foreign fingerprint models a post-reboot recycle: the claim is released
    (dead worker), no signal is sent, and max-runtime enforcement does not SIGTERM the stranger either."""
    conn = board
    killed = []
    stranger_fingerprint = 1  # no live process started at tick 1
    tid = _claimed_running(conn, pid=os.getpid(), started_at=stranger_fingerprint, max_runtime=1)

    assert kbd._worker_alive(os.getpid(), stranger_fingerprint) is False
    assert tid in kbd.enforce_max_runtime(conn, signal_fn=lambda pid, sig: killed.append((pid, sig)))
    assert killed == []
    task = kb.get_task(conn, tid)
    assert task.status == "ready" and task.worker_pid is None

    tid2 = _claimed_running(conn, pid=os.getpid(), started_at=stranger_fingerprint)
    assert kb.release_stale_claims(conn, signal_fn=lambda pid, sig: killed.append((pid, sig))) == 1
    assert killed == []
    assert kb.get_task(conn, tid2).status == "ready"


def test_matching_fingerprint_keeps_the_live_worker(board):
    """The same PID with ITS OWN fingerprint (recorded at spawn) is our worker: the expired claim is
    extended rather than reclaimed, and the timeout path signals it."""
    from gateway.status import get_process_start_time

    conn = board
    killed = []
    tid = _claimed_running(conn, pid=os.getpid(), started_at=get_process_start_time(os.getpid()))
    assert kbd._worker_alive(os.getpid(), get_process_start_time(os.getpid())) is True
    assert kb.release_stale_claims(conn) == 0
    assert kb.get_task(conn, tid).status == "running"
    kinds = [e.kind for e in kb.list_events(conn, tid)]
    assert "claim_extended" in kinds

    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET max_runtime_seconds = 1 WHERE id = ?", (tid,))
    kbd.enforce_max_runtime(conn, signal_fn=lambda pid, sig: killed.append((pid, sig)))
    assert killed and killed[0] == (os.getpid(), signal.SIGTERM)


def test_same_pid_and_start_tick_on_another_boot_is_foreign(board, monkeypatch):
    """A row that survived a reboot: the PID AND the boot-relative start tick both match a process on
    this boot (the Linux start time is clock ticks since boot, so that recurs), but the persisted
    witness does not. The worker is foreign: claim released, zero signals."""
    from gateway import drain_control

    # A Linux-style witness is pinned rather than read: a macOS host has no ``/proc`` epoch and
    # names its boot with the session UUID instead (see the Darwin test below).
    monkeypatch.setattr(drain_control, "current_instantiation_epoch", lambda: "boot-a:1")

    conn = board
    killed = []
    live_fingerprint = kbd._process_fingerprint(os.getpid())
    assert live_fingerprint is not None and live_fingerprint.split("|", 1)[1] == str(
        __import__("gateway.status", fromlist=["x"]).get_process_start_time(os.getpid()))
    tid = _claimed_running(conn, pid=os.getpid(), started_at=live_fingerprint, max_runtime=1)
    assert kbd._worker_alive(os.getpid(), live_fingerprint) is True

    # Same PID, same start tick, different boot identity.
    other_boot = "deadbeef-boot:1|" + live_fingerprint.split("|", 1)[1]
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET worker_started_at = ? WHERE id = ?", (other_boot, tid))
    assert kbd._worker_alive(os.getpid(), other_boot) is False
    assert tid in kbd.enforce_max_runtime(conn, signal_fn=lambda pid, sig: killed.append((pid, sig)))
    assert killed == []
    task = kb.get_task(conn, tid)
    assert task.status == "ready" and task.worker_pid is None

    # The same value re-derived on THIS boot still identifies our worker (the witness is stable
    # within a boot, unlike the recorded epoch of a previous one).
    assert kbd._process_fingerprint(os.getpid()) == live_fingerprint


def test_a_host_without_proc_names_its_boot_with_the_session_uuid(board, monkeypatch):
    """macOS: no ``/proc``, so ``current_instantiation_epoch()`` is ``""`` and the fingerprint's
    witness half used to be empty -- a live spawn recorded a bare ``"|179009989536"`` (M78).

    Darwin publishes a real per-boot identity, so the witness is the boot session UUID: stable for
    every read on this boot, different after a reboot. A row from another boot is foreign; the
    live worker under its own boot is ours, with no signal and its claim intact.
    """
    from gateway import drain_control

    conn = board
    killed = []
    monkeypatch.setattr(drain_control, "current_instantiation_epoch", lambda: "")
    monkeypatch.setattr(kbd, "_darwin_boot_session_uuid", lambda: "FAKE-BOOT-SESSION-UUID")

    live_fingerprint = kbd._process_fingerprint(os.getpid())
    assert live_fingerprint is not None
    assert live_fingerprint.startswith("bootsession:FAKE-BOOT-SESSION-UUID|")
    assert kbd._process_fingerprint(os.getpid()) == live_fingerprint, "the witness moved within a boot"

    tid = _claimed_running(conn, pid=os.getpid(), started_at=live_fingerprint)
    assert kbd._worker_alive(os.getpid(), live_fingerprint) is True
    assert kb.release_stale_claims(conn, signal_fn=lambda pid, sig: killed.append((pid, sig))) == 0
    assert killed == []
    assert kb.get_task(conn, tid).status == "running"

    # Same PID and same start tick after a reboot: the session UUID is what gives it away.
    other_boot = "bootsession:ANOTHER-BOOT-SESSION-UUID|" + live_fingerprint.split("|", 1)[1]
    assert kbd._worker_alive(os.getpid(), other_boot) is False


def test_a_host_that_cannot_name_its_boot_holds_the_claim_instead_of_releasing_it(board, monkeypatch):
    """No witness anywhere (no ``/proc`` epoch, no boot session UUID): the start time alone is the
    value the fingerprint docstring calls unsafe, so it is reported unreadable rather than composed.

    A spawn then records ``UNVERIFIED_WORKER_FINGERPRINT``: the claim is held beside the live PID,
    nothing is signalled, and a dead PID is still released -- missing identity is refusal, not
    permission (#99558).
    """
    from gateway import drain_control

    conn = board
    killed = []
    monkeypatch.setattr(drain_control, "current_instantiation_epoch", lambda: "")
    monkeypatch.setattr(kbd, "_darwin_boot_session_uuid", lambda: "")

    assert kbd._process_fingerprint(os.getpid()) is None
    tid = kb.create_task(conn, title="job", assignee="worker")
    kb.claim_task(conn, tid)
    kbd._set_worker_pid(conn, tid, os.getpid())
    row = conn.execute("SELECT worker_started_at FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert row["worker_started_at"] == kbd.UNVERIFIED_WORKER_FINGERPRINT
    event = [e for e in kb.list_events(conn, tid) if e.kind == "spawned"][-1]
    assert (event.payload or {})["started_at"] == kbd.UNVERIFIED_WORKER_FINGERPRINT

    old = int(time.time()) - 3600
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET started_at = ?, claim_expires = ? WHERE id = ?", (old, old, tid))
    sig = lambda pid, s: killed.append((pid, s))
    assert kb.release_stale_claims(conn, signal_fn=sig) == 0
    assert killed == []
    assert kb.get_task(conn, tid).status == "running"


def test_an_unre_derivable_fingerprint_never_reads_as_a_recycled_pid(board, monkeypatch):
    """A row that carries a composed fingerprint while the host can no longer re-derive one (the
    start time became unreadable) is unverifiable, not foreign.

    Comparing ``None`` against the row is what declared live workers dead and released their claims
    to a duplicate worker on the same worktree. Hold the claim, signal nothing.
    """
    from gateway import status
    from gateway.status import get_process_start_time

    conn = board
    killed = []
    start = get_process_start_time(os.getpid())
    assert start is not None
    tid = _claimed_running(conn, pid=os.getpid(), started_at=f"boot-a:1|{start}")
    monkeypatch.setattr(status, "_get_process_start_time", lambda pid: None)

    assert kbd._pid_recycled(os.getpid(), f"boot-a:1|{start}") is False
    assert kbd._worker_alive(os.getpid(), f"boot-a:1|{start}") is True
    assert kb.release_stale_claims(conn, signal_fn=lambda pid, sig: killed.append((pid, sig))) == 0
    assert killed == []
    assert kb.get_task(conn, tid).status == "running"


def test_unverified_fingerprint_capture_never_authorizes_a_signal(board, monkeypatch):
    """Fingerprint capture fails for a new spawn: the row is NOT a legacy NULL row. A live PID under
    it is never SIGTERM/SIGKILLed by any reclaim/timeout path, and the claim is held (not released
    beside the live process); once the PID is gone the claim is reclaimed normally."""
    from gateway import status

    conn = board
    killed = []
    monkeypatch.setattr(status, "_get_process_start_time", lambda pid: None)
    tid = kb.create_task(conn, title="job", assignee="worker", max_runtime_seconds=1)
    kb.claim_task(conn, tid)
    kbd._set_worker_pid(conn, tid, os.getpid())
    row = conn.execute("SELECT worker_started_at FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert row["worker_started_at"] == kbd.UNVERIFIED_WORKER_FINGERPRINT
    monkeypatch.undo()
    old = int(time.time()) - 3600
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET started_at = ?, claim_expires = ? WHERE id = ?", (old, old, tid))
        conn.execute("UPDATE task_runs SET started_at = ? WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                     (old, tid))

    sig = lambda pid, s: killed.append((pid, s))
    assert kbd.enforce_max_runtime(conn, signal_fn=sig) == []
    assert kb.release_stale_claims(conn, signal_fn=sig) == 0
    assert killed == []
    assert kb.get_task(conn, tid).status == "running"
    # An explicit operator reclaim releases the claim (human override) but still sends nothing.
    assert kb.reclaim_task(conn, tid, reason="operator", signal_fn=sig) is True
    assert killed == []

    # The process is gone (a dead PID): the row is reclaimed like any dead worker, still no signal.
    tid2 = kb.create_task(conn, title="job2", assignee="worker")
    kb.claim_task(conn, tid2)
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET worker_pid = ?, worker_started_at = ?, claim_expires = ? WHERE id = ?",
                     (os.getpid(), kbd.UNVERIFIED_WORKER_FINGERPRINT, old, tid2))
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)
    assert kb.release_stale_claims(conn, signal_fn=sig) == 1
    assert killed == [] and kb.get_task(conn, tid2).status == "ready"
