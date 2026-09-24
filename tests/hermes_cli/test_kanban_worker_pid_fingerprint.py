"""A recycled worker PID is never mistaken for our worker.

``tasks.worker_pid`` survives a reboot; the number can then belong to an unrelated process. Every
liveness decision (extend/defer the claim) and every kill (SIGTERM/SIGKILL on timeout or reclaim)
must require the spawn-time start fingerprint to match, never bare PID existence.
"""

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


def test_worker_registration_survives_dispatcher_exit_and_fences_stale_runs(board, tmp_path, monkeypatch):
    conn = board
    tid = kb.create_task(conn, title="orphan window", assignee="worker")
    claimed = kb.claim_task(conn, tid)
    marker = tmp_path / "model-entered"
    release = tmp_path / "release"
    env = os.environ.copy()
    env.pop("HERMES_DELEGATED_CHILD_CONTEXT", None)
    env.update(HERMES_KANBAN_DB=str(tmp_path / "kanban.db"),
               HERMES_KANBAN_TASK=tid, HERMES_KANBAN_RUN_ID=str(claimed.current_run_id),
               HERMES_KANBAN_CLAIM_LOCK=claimed.claim_lock,
               MODEL_MARKER=str(marker), MODEL_RELEASE=str(release))
    root = Path(__file__).resolve().parents[2]
    # Invoke the -q boundary in a separate interpreter: no direct call to the PID helper here.
    worker_code = """
import os
import time
from pathlib import Path
import cli
from hermes_cli.cli_single_query import _run_single_query_mode

cli._should_seed_interactive = lambda *args: True
cli._collect_query_images = lambda *args: ('job', [])
class Worker:
    def run(self):
        Path(os.environ['MODEL_MARKER']).write_text(str(os.getpid()))
        while not Path(os.environ['MODEL_RELEASE']).exists():
            time.sleep(0.05)
_run_single_query_mode(Worker(), 'job', None, False, True)
"""
    def launch():
        return subprocess.Popen([sys.executable, "-c", worker_code], cwd=root, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    worker = launch()
    try:
        deadline = time.monotonic() + 25
        while not marker.exists() and worker.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert marker.exists(), f"worker did not enter the model boundary (rc={worker.poll()})"
        child_pid = int(marker.read_text())
        registered = kb.get_task(conn, tid)
        assert registered.worker_pid == child_pid
        fingerprint = conn.execute("SELECT worker_started_at FROM tasks WHERE id = ?",
                                   (tid,)).fetchone()[0]
        assert fingerprint == kbd._process_fingerprint(child_pid)
        run = conn.execute("SELECT worker_pid, worker_started_at FROM task_runs WHERE id = ?",
                           (claimed.current_run_id,)).fetchone()
        assert (run["worker_pid"], run["worker_started_at"]) == (child_pid, fingerprint)
        # The original dispatcher has gone; a fresh connection must not spawn the live worker twice.
        conn.execute("UPDATE tasks SET claim_expires = ? WHERE id = ?", (int(time.time()) - 3600, tid))
        monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: None)
        with kbc.connect(tmp_path / "kanban.db") as restarted:
            spawns = []
            tick = kbd.dispatch_once(restarted, spawn_fn=lambda *args: spawns.append(args),
                                     max_in_progress=20, max_spawn=1)
        assert tick.spawned == [] and spawns == []
        assert kb.get_task(conn, tid).status == "running"
    finally:
        release.touch()
        try:
            worker.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.communicate()
    assert worker.returncode == 0
    assert len([e for e in kb.list_events(conn, tid) if e.kind == "spawned"]) == 1
    # The same environment after a successor claims the card must stop before model work.
    marker.unlink()
    conn.execute("UPDATE tasks SET claim_lock = ?, current_run_id = current_run_id + 1, "
                 "worker_pid = NULL WHERE id = ?", ("superseded", tid))
    stale = launch()
    try:
        stale.communicate(timeout=25)
    except subprocess.TimeoutExpired:
        stale.kill()
        stale.communicate()
        pytest.fail("superseded worker did not exit")
    assert stale.returncode != 0 and not marker.exists()


def test_registration_during_reclaim_selection_prevents_duplicate(board, monkeypatch):
    conn = board
    tid = kb.create_task(conn, title="late registration", assignee="worker")
    claimed = kb.claim_task(conn, tid)
    conn.execute("UPDATE tasks SET claim_expires = ? WHERE id = ?", (int(time.time()) - 3600, tid))

    def register_during_reclaim(*args, **kwargs):
        assert kbd._set_worker_pid(conn, tid, os.getpid(),
                                   expected_run_id=claimed.current_run_id, claim_lock=claimed.claim_lock)
        return {"status": "not_started"}

    monkeypatch.setattr(kb, "_terminate_reclaimed_worker", register_during_reclaim)
    assert kb.release_stale_claims(conn) == 0
    assert kb.get_task(conn, tid).worker_pid == os.getpid()
    assert kb.get_task(conn, tid).status == "running"


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
    instantiation epoch does not. The worker is foreign: claim released, zero signals."""
    from gateway import drain_control

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
    drain_control.current_instantiation_epoch.cache_clear()
    assert kbd._process_fingerprint(os.getpid()) == live_fingerprint


def test_unverified_fingerprint_capture_never_authorizes_a_signal(board, monkeypatch):
    """Fingerprint capture fails for a new spawn: the row is NOT a legacy NULL row. A live PID under
    it is never SIGTERM/SIGKILLed by any reclaim/timeout path, and the claim is held (not released
    beside the live process); once the PID is gone the claim is reclaimed normally."""
    import gateway.status as status

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

    sig = lambda pid, s: killed.append((pid, s))  # noqa: E731
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
