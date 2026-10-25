"""A live, heartbeating worker is never declared ``crashed`` by the pid probe alone.

Regression for the 2026-10-25 gmantra incident (card t_3d89954a): the in-gateway dispatcher
declared seven live workers ``crashed: pid N not alive`` in two sweeps. Root cause: the worker
fingerprint ``"<epoch>|<round(psutil create_time*100)>"`` was compared with exact string equality,
and on macOS psutil >= 7 shifts ``create_time()`` by ``INIT_BOOT_TIME - boot_time()`` — the
``kern.boottime`` drift accumulated since psutil was imported into THAT process. A day-old gateway
read every worker +2 s off its own spawn record (and off a one-shot dispatcher's record), so the
live pid read as "recycled" -> dead -> re-spawned beside the still-running worker, which then raced
its own replacement in the same worktree.

Three layers now stand between a live worker and a false crash verdict:

1. ``gateway.status_process_identity.raw_create_time`` reads the kernel's unadjusted start time on macOS.
2. ``_fingerprint_matches`` tolerates ``START_TIME_DRIFT_TOLERANCE`` on the start component.
3. ``_worker_alive_for_task``: a pid that exists and either carries the card in its environment/argv or
   heartbeated inside ``HEARTBEAT_LIVENESS_WINDOW_SECONDS`` is held, whatever the fingerprint says.

And a worker whose run WAS closed as dead can re-attach (``reattach_orphaned_run``) instead of being
locked out with ``unknown id or not running``.
"""

import os
import signal
import subprocess
import sys
import time

import pytest

from gateway import status
from gateway import status_process_identity as spi
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_dispatch_liveness as kbl

pytestmark = pytest.mark.platforms("posix")


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
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


def _spawn_worker_like(task_id: str, cwd=None) -> subprocess.Popen:
    """A process whose argv carries the dispatcher's worker prompt (``_worker_argv`` shape)."""
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(600)", "chat", "-q", f"work kanban task {task_id}"],
        cwd=cwd, start_new_session=True,
    )


def _claimed_running(conn, *, pid: int, started_at, last_heartbeat_at=None, max_runtime=None) -> str:
    tid = kb.create_task(conn, title="job", assignee="worker", max_runtime_seconds=max_runtime)
    kb.claim_task(conn, tid)
    kbd._set_worker_pid(conn, tid, pid)
    old = int(time.time()) - 3600
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET worker_started_at = ?, started_at = ?, claim_expires = ?, last_heartbeat_at = ? "
            "WHERE id = ?", (started_at, old, old, last_heartbeat_at, tid))
        conn.execute("UPDATE task_runs SET started_at = ? WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                     (old, tid))
    return tid


def _drifted(fingerprint: str, centiseconds: int) -> str:
    epoch, _, start = fingerprint.partition("|")
    return f"{epoch}|{int(start) + centiseconds}"


def _stored_fingerprint(conn, tid):
    return conn.execute("SELECT worker_started_at FROM tasks WHERE id = ?", (tid,)).fetchone()[0]


def _events(conn, tid, kind):
    return conn.execute("SELECT payload FROM task_events WHERE task_id = ? AND kind = ?", (tid, kind)).fetchall()


# --- layer 1: the identity source -------------------------------------------------------------

@pytest.mark.skipif(sys.platform != "darwin", reason="psutil's boot-time correction is macOS-only")
def test_raw_create_time_ignores_psutil_boot_time_correction(sleeper, monkeypatch):
    """The incident mechanism, reproduced: shifting psutil's import-time boot witness by 2 s moves
    ``create_time()`` by 2 s, but the fingerprint source stays put."""
    import psutil
    from psutil import _psosx

    proc = psutil.Process(sleeper.pid)
    baseline_adjusted = proc.create_time()
    baseline_raw = spi.raw_create_time(proc)
    before = status.get_process_start_time(sleeper.pid)

    monkeypatch.setattr(_psosx, "INIT_BOOT_TIME", _psosx.INIT_BOOT_TIME + 2)
    shifted = psutil.Process(sleeper.pid)  # fresh object: create_time() is cached per instance
    assert shifted.create_time() == pytest.approx(baseline_adjusted + 2, abs=0.01)
    assert spi.raw_create_time(shifted) == pytest.approx(baseline_raw, abs=0.0001)
    assert status.get_process_start_time(sleeper.pid) == before


# --- layer 2: tolerant comparison -------------------------------------------------------------

def test_fingerprint_within_drift_tolerance_matches(sleeper):
    live = kbd._process_fingerprint(sleeper.pid)
    assert live is not None
    # The incident's exact offset: +2.00 s (200 centiseconds / ticks).
    assert kbd._fingerprint_matches(sleeper.pid, _drifted(live, 200)) is True
    assert kbd._fingerprint_matches(sleeper.pid, _drifted(live, -200)) is True
    assert kbd._worker_alive(sleeper.pid, _drifted(live, 200)) is True
    # Beyond the tolerance it is still a stranger.
    assert kbd._fingerprint_matches(sleeper.pid, _drifted(live, status.START_TIME_DRIFT_TOLERANCE + 1)) is False
    assert kbd._fingerprint_matches(sleeper.pid, _drifted(live, -10_000_000)) is False


def test_fingerprint_epoch_component_is_still_exact(sleeper):
    live = kbd._process_fingerprint(sleeper.pid)
    _, _, start = live.partition("|")
    assert kbd._fingerprint_matches(sleeper.pid, f"other-boot|{start}") is False


def test_drifted_record_does_not_crash_the_worker(board, sleeper):
    """End to end on the sweep: a +2 s stale record (the incident's) keeps the card running."""
    conn = board
    live = kbd._process_fingerprint(sleeper.pid)
    tid = _claimed_running(conn, pid=sleeper.pid, started_at=_drifted(live, 200))
    assert kbd.detect_crashed_workers(conn) == []
    task = kb.get_task(conn, tid)
    assert task.status == "running" and task.worker_pid == sleeper.pid
    assert kb.release_stale_claims(conn, signal_fn=lambda *_: None) == 0
    assert kb.get_task(conn, tid).status == "running"


# --- layer 3: card-level witnesses ------------------------------------------------------------

def test_fresh_heartbeat_blocks_crashed_even_when_fingerprint_says_stranger(board, sleeper):
    """The fingerprint is positively different (beyond tolerance) but the card heartbeated 30 s
    ago: the sweep must hold it, not re-spawn beside it."""
    conn = board
    live = kbd._process_fingerprint(sleeper.pid)
    stranger = _drifted(live, -10_000_000)
    tid = _claimed_running(conn, pid=sleeper.pid, started_at=stranger,
                           last_heartbeat_at=int(time.time()) - 30)

    assert kbd._worker_alive(sleeper.pid, stranger) is False  # fingerprint alone would kill it
    alive, hold = kbl._worker_alive_for_task(
        sleeper.pid, stranger, task_id=tid, last_heartbeat_at=int(time.time()) - 30)
    assert (alive, hold) == (True, "heartbeat")

    assert kbd.detect_crashed_workers(conn) == []
    task = kb.get_task(conn, tid)
    assert task.status == "running" and task.worker_pid == sleeper.pid
    assert kb.claim_task(conn, tid) is None  # no second worker can take the card
    holds = _events(conn, tid, "liveness_hold")
    assert len(holds) == 1 and '"reason": "heartbeat"' in holds[0]["payload"]
    # Re-evaluated every tick, recorded once.
    assert kbd.detect_crashed_workers(conn) == []
    assert len(_events(conn, tid, "liveness_hold")) == 1
    # A heartbeat-only hold never rewrites the identity (nothing proved which process this is).
    assert _stored_fingerprint(conn, tid) == stranger
    assert sleeper.poll() is None


def test_stale_heartbeat_and_no_witness_is_still_reclaimed(board, sleeper):
    """The other direction: beyond the window with no process witness, a positively different
    fingerprint IS a recycled pid and the card is released (nothing signalled)."""
    conn = board
    live = kbd._process_fingerprint(sleeper.pid)
    stranger = _drifted(live, -10_000_000)
    tid = _claimed_running(conn, pid=sleeper.pid, started_at=stranger,
                           last_heartbeat_at=int(time.time()) - kbl.HEARTBEAT_LIVENESS_WINDOW_SECONDS - 60)
    assert tid in kbd.detect_crashed_workers(conn)
    assert kb.get_task(conn, tid).status == "ready"
    assert sleeper.poll() is None


def test_process_witness_holds_and_rebinds_fingerprint(board, tmp_path):
    """The process names the card on its argv (no HERMES_KANBAN_TASK in its environment, so the
    argv fallback answers): held with no heartbeat at all, and the stale stored fingerprint is
    re-recorded so later probes match without the witness."""
    conn = board
    tid = kb.create_task(conn, title="job", assignee="worker")
    worker = _spawn_worker_like(tid, cwd=tmp_path)
    try:
        live = kbd._process_fingerprint(worker.pid)
        stranger = _drifted(live, -10_000_000)
        kb.claim_task(conn, tid)
        kbd._set_worker_pid(conn, tid, worker.pid)
        old = int(time.time()) - 3600
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET worker_started_at = ?, started_at = ?, last_heartbeat_at = NULL WHERE id = ?",
                         (stranger, old, tid))
        assert kbl._process_names_task(worker.pid, tid) is True
        assert kbl._process_names_task(worker.pid, "t_00000000") is False
        assert kbl._worker_alive_for_task(worker.pid, stranger, task_id=tid, last_heartbeat_at=None) == (True, "process")

        assert kbd.detect_crashed_workers(conn) == []
        task = kb.get_task(conn, tid)
        assert task.status == "running"
        assert _stored_fingerprint(conn, tid) == live  # rebound
        run = kb.latest_run(conn, tid)
        assert conn.execute("SELECT worker_started_at FROM task_runs WHERE id = ?", (run.id,)).fetchone()[0] == live
        holds = _events(conn, tid, "liveness_hold")
        assert len(holds) == 1 and '"rebound": true' in holds[0]["payload"]
        # Now the plain fingerprint path agrees and no further witness is needed.
        assert kbd._worker_alive(worker.pid, _stored_fingerprint(conn, tid)) is True
        assert worker.poll() is None
    finally:
        worker.kill()
        worker.wait()


def test_environment_witness_is_read_before_argv(board, tmp_path):
    """The dispatcher pins HERMES_KANBAN_TASK into the worker's environment; that record decides
    even when argv says nothing (a tool shell, a re-exec) — and a foreign pin is a stranger."""
    conn = board
    tid = kb.create_task(conn, title="job", assignee="worker")
    env = {**os.environ, "HERMES_KANBAN_TASK": tid}
    quiet = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"], env=env, cwd=tmp_path)
    foreign = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(600)", "chat", "-q", f"work kanban task {tid}"],
        env={**os.environ, "HERMES_KANBAN_TASK": "t_00000000"}, cwd=tmp_path)
    try:
        assert kbl._process_names_task(quiet.pid, tid) is True
        assert kbl._process_names_task(quiet.pid, "t_00000000") is False
        # Environment outranks argv: the argv names the card but the pinned task is another one.
        assert kbl._process_names_task(foreign.pid, tid) is False
    finally:
        for p in (quiet, foreign):
            p.kill()
            p.wait()


def test_max_runtime_kill_still_reaches_a_drifted_worker(board, tmp_path):
    """A timed-out worker whose fingerprint drifted must be ended, not merely released beside."""
    conn = board
    tid = kb.create_task(conn, title="job", assignee="worker", max_runtime_seconds=1)
    worker = _spawn_worker_like(tid, cwd=tmp_path)
    try:
        live = kbd._process_fingerprint(worker.pid)
        kb.claim_task(conn, tid)
        kbd._set_worker_pid(conn, tid, worker.pid)
        old = int(time.time()) - 3600
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET worker_started_at = ?, started_at = ? WHERE id = ?",
                         (_drifted(live, -10_000_000), old, tid))
            conn.execute("UPDATE task_runs SET started_at = ? WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                         (old, tid))
        killed = []
        assert tid in kbd.enforce_max_runtime(
            conn, signal_fn=lambda pid, s: killed.append((pid, s)) or os.kill(pid, s))
        assert killed and killed[0] == (worker.pid, signal.SIGTERM)
        assert worker.wait(timeout=10) is not None
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait()


# --- duplicate-worker prevention: the dead leader's group ------------------------------------

@pytest.mark.live_system_guard_bypass  # the leader's orphans reparent to init: real signals, real pids
def test_dead_leader_group_members_in_the_workspace_are_terminated(board, tmp_path):
    """Leader exits, its tool shell keeps running in the workspace: the sweep ends the shell before the
    card is re-spawned there. A process in the same group but outside the workspace and not naming
    the card is left alone."""
    conn = board
    workspace = tmp_path / "ws"
    workspace.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    # Leader: starts two sleepers in its session (same pgid), then exits.
    leader = subprocess.Popen(
        [sys.executable, "-c",
         "import subprocess, sys, os\n"
         f"a = subprocess.Popen(['sleep', '600'], cwd={str(workspace)!r})\n"
         f"b = subprocess.Popen(['sleep', '600'], cwd={str(elsewhere)!r})\n"
         "print(a.pid, b.pid, flush=True)\n"],
        stdout=subprocess.PIPE, start_new_session=True, text=True,
    )
    ws_pid, other_pid = (int(x) for x in leader.stdout.readline().split())
    leader.wait(timeout=10)
    try:
        assert os.getpgid(ws_pid) == leader.pid and os.getpgid(other_pid) == leader.pid
        tid = kb.create_task(conn, title="job", assignee="worker")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET workspace_path = ? WHERE id = ?", (str(workspace), tid))
        kb.claim_task(conn, tid)
        with kb.write_txn(conn):
            old = int(time.time()) - 3600
            conn.execute("UPDATE tasks SET worker_pid = ?, worker_started_at = NULL, started_at = ? WHERE id = ?",
                         (leader.pid, old, tid))

        assert tid in kbd.detect_crashed_workers(conn)
        crashed = _events(conn, tid, "crashed")
        assert len(crashed) == 1 and f'"group_terminated": [{ws_pid}]' in crashed[0]["payload"]
        # The workspace member got SIGTERM; the unrelated group member did not.
        deadline = time.time() + 5
        while time.time() < deadline and kbd._pid_alive(ws_pid):
            time.sleep(0.1)
        assert not kbd._pid_alive(ws_pid)
        assert kbd._pid_alive(other_pid)
    finally:
        for p in (ws_pid, other_pid):
            try:
                os.kill(p, signal.SIGKILL)
            except ProcessLookupError:
                pass


# --- re-attach: a wrongly-closed run can still land its handoff -----------------------------

def _crash_sweep_closes(conn, pid, stale_hb):
    """Drive the sweep to the incident's false verdict: positively different fingerprint, no witness."""
    live = kbd._process_fingerprint(pid)
    tid = _claimed_running(conn, pid=pid, started_at=_drifted(live, -10_000_000), last_heartbeat_at=stale_hb)
    run_id = kb.get_task(conn, tid).current_run_id
    assert tid in kbd.detect_crashed_workers(conn)
    task = kb.get_task(conn, tid)
    assert task.status == "ready" and task.current_run_id is None
    assert conn.execute("SELECT outcome FROM task_runs WHERE id = ?", (run_id,)).fetchone()[0] == "crashed"
    return tid, run_id


def test_reattach_reopens_a_run_closed_as_crashed(board, sleeper):
    conn = board
    stale = int(time.time()) - kbl.HEARTBEAT_LIVENESS_WINDOW_SECONDS - 60
    tid, run_id = _crash_sweep_closes(conn, sleeper.pid, stale)
    failures_after_crash = kb.get_task(conn, tid).consecutive_failures
    assert failures_after_crash >= 1

    assert kbl.reattach_orphaned_run(conn, tid, run_id, pid=sleeper.pid, claim_lock="host:1") == "reattached"
    task = kb.get_task(conn, tid)
    assert task.status == "running" and task.current_run_id == run_id and task.worker_pid == sleeper.pid
    assert _stored_fingerprint(conn, tid) == kbd._process_fingerprint(sleeper.pid)
    assert task.consecutive_failures == failures_after_crash - 1
    run = conn.execute("SELECT status, outcome, ended_at FROM task_runs WHERE id = ?", (run_id,)).fetchone()
    assert (run["status"], run["outcome"], run["ended_at"]) == ("running", None, None)
    assert len(_events(conn, tid, "run_reattached")) == 1
    # Idempotent while live, and the handoff now lands under the original run identity.
    assert kbl.reattach_orphaned_run(conn, tid, run_id, pid=sleeper.pid) == "reattached"
    assert kbd.heartbeat_worker(conn, tid, expected_run_id=run_id) is True
    assert kb.complete_task(conn, tid, summary="done after all", expected_run_id=run_id) is True
    assert kb.get_task(conn, tid).status == "done"
    assert kbl.reattach_orphaned_run(conn, tid, run_id, pid=sleeper.pid) == "not_orphaned"


def test_reattach_refuses_when_a_successor_owns_the_card(board, sleeper):
    conn = board
    stale = int(time.time()) - kbl.HEARTBEAT_LIVENESS_WINDOW_SECONDS - 60
    tid, run_id = _crash_sweep_closes(conn, sleeper.pid, stale)
    assert kb.claim_task(conn, tid) is not None  # the dispatcher spawned a successor
    successor = kb.get_task(conn, tid).current_run_id
    assert successor != run_id

    assert kbl.reattach_orphaned_run(conn, tid, run_id, pid=sleeper.pid) == "successor_live"
    task = kb.get_task(conn, tid)
    assert task.status == "running" and task.current_run_id == successor
    refused = _events(conn, tid, "orphan_worker_refused")
    assert len(refused) == 1 and f'"successor_run_id": {successor}' in refused[0]["payload"]
    assert kbl.reattach_orphaned_run(conn, tid, run_id, pid=sleeper.pid) == "successor_live"
    assert len(_events(conn, tid, "orphan_worker_refused")) == 1  # alerted once
    # The orphan's own CAS still fails closed.
    assert kbd.heartbeat_worker(conn, tid, expected_run_id=run_id) is False
    assert kb.complete_task(conn, tid, summary="x", expected_run_id=run_id) is False


def test_reattach_leaves_intentionally_closed_runs_alone(board, sleeper):
    conn = board
    live = kbd._process_fingerprint(sleeper.pid)
    tid = _claimed_running(conn, pid=sleeper.pid, started_at=live)
    run_id = kb.get_task(conn, tid).current_run_id
    assert kb.block_task(conn, tid, reason="need input", expected_run_id=run_id) is True
    assert kbl.reattach_orphaned_run(conn, tid, run_id, pid=sleeper.pid) == "not_orphaned"
    assert kb.get_task(conn, tid).status == "blocked"
    assert kbl.reattach_orphaned_run(conn, tid, 999_999, pid=sleeper.pid) == "unknown"


def test_worker_tool_reattaches_before_completing(board, sleeper, monkeypatch, tmp_path):
    """The tool path: a worker whose run was closed as crashed calls kanban_complete and lands it."""
    conn = board
    stale = int(time.time()) - kbl.HEARTBEAT_LIVENESS_WINDOW_SECONDS - 60
    tid, run_id = _crash_sweep_closes(conn, sleeper.pid, stale)
    conn.close()

    from tools import kanban_tools
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "kanban.db"))
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "host:1")
    monkeypatch.setattr(kanban_tools, "_goal_gate", lambda *a, **k: None)
    out = kanban_tools._handle_complete({"summary": "landed after a false crash verdict"})
    assert '"error"' not in out, out
    conn = kbc.connect(tmp_path / "kanban.db")
    try:
        task = kb.get_task(conn, tid)
        assert task.status == "done"
        assert len(_events(conn, tid, "run_reattached")) == 1
        run = conn.execute("SELECT outcome FROM task_runs WHERE id = ?", (run_id,)).fetchone()
        assert run["outcome"] == "completed"
    finally:
        conn.close()
