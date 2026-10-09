"""No-model regression for false crash reconciliation of a possibly-live worker.

Run 321 exposed a crash declaration beside a visibly live worker; its exact
probe failure was not retained. Controlled start-time readings below reproduce
substantiated failure mechanisms, not a reconstruction of that live incident.
All process observations and signals are doubles; the board is scratch-only.
"""

from pathlib import Path

import pytest

from gateway import drain_control, status
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as dispatch


PID = 424242
START = 179157293635
EPOCH = "test-boot:1"
FINGERPRINT = f"{EPOCH}|{START}"
NOW = 1800000000


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(dispatch.time, "time", lambda: NOW)
    monkeypatch.setattr(kb, "_host_prefix", lambda: "test-host:")
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: pid == PID)
    monkeypatch.setattr(drain_control, "current_instantiation_epoch", lambda: EPOCH)
    monkeypatch.setattr(status, "get_process_start_time", lambda pid: START)
    # Dead-worker diagnostics cannot consult a real board log or reap host children.
    monkeypatch.setattr(dispatch, "_classify_worker_exit", lambda pid: ("unknown", None))
    monkeypatch.setattr(dispatch, "_worker_log_exit_code", lambda *a, **k: None)
    monkeypatch.setattr(dispatch, "_worker_final_output", lambda *a, **k: "")
    with kbc.connect(tmp_path / "kanban.db") as conn:
        yield conn


def running_worker(conn, *, fingerprint=FINGERPRINT, max_runtime=None):
    tid = kb.create_task(conn, title="liveness fixture", assignee="fixture-worker",
                         max_runtime_seconds=max_runtime)
    assert kb.claim_task(conn, tid, claimer="test-host:dispatcher") is not None
    dispatch._set_worker_pid(conn, tid, PID)
    task = kb.get_task(conn, tid)
    assert task is not None
    run_id = task.current_run_id
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET worker_started_at = ?, started_at = ?, claim_expires = ? WHERE id = ?",
                     (fingerprint, NOW - 31, NOW - 1, tid))
        conn.execute("UPDATE task_runs SET worker_started_at = ?, started_at = ?, claim_expires = ? WHERE id = ?",
                     (fingerprint, NOW - 31, NOW - 1, run_id))
    return tid, run_id


@pytest.mark.parametrize("reading", [START - 1, START + 1, None],
                         ids=["minus-one-centisecond", "plus-one-centisecond", "unreadable-start"])
def test_crash_sweep_retains_possibly_live_worker(board, monkeypatch, reading):
    tid, run_id = running_worker(board)
    monkeypatch.setattr(status, "get_process_start_time", lambda pid: reading)

    sweep = dispatch._reclaim_dead_workers(board)

    assert sweep.crashed == [], "a live PID with no material identity mismatch must not be declared crashed"
    assert sweep.exited_hook_payloads == []
    task = kb.get_task(board, tid)
    assert task is not None
    assert task.status == "running" and task.worker_pid == PID
    assert task.current_run_id == run_id
    assert board.execute("SELECT worker_started_at FROM tasks WHERE id = ?", (tid,)).fetchone()[0] == FINGERPRINT
    run = board.execute("SELECT ended_at, outcome, worker_pid, worker_started_at FROM task_runs WHERE id = ?",
                        (run_id,)).fetchone()
    assert run["ended_at"] is None and run["outcome"] is None
    assert run["worker_pid"] == PID and run["worker_started_at"] == FINGERPRINT
    assert "crashed" not in [event.kind for event in kb.list_events(board, tid)]
    # Liveness reconciliation is not authority to signal this numeric PID.
    assert dispatch._pid_recycled(PID, FINGERPRINT) is True


@pytest.mark.parametrize("reading", [START + 1, None], ids=["drift", "unreadable"])
def test_reaper_preserves_witness_when_live_identity_cannot_authorize_signal(board, monkeypatch, reading):
    tid, run_id = running_worker(board)
    assert kb.complete_task(board, tid, result="fixture done", expected_run_id=run_id)
    board.execute("UPDATE task_runs SET ended_at = ? WHERE id = ?", (NOW - 600, run_id))
    monkeypatch.setattr(status, "get_process_start_time", lambda pid: reading)
    signals = []

    assert dispatch.reap_terminal_workers(board, signal_fn=lambda *args: signals.append(args)) == []

    assert signals == []
    run = board.execute("SELECT worker_pid, worker_started_at FROM task_runs WHERE id = ?", (run_id,)).fetchone()
    assert run["worker_pid"] == PID and run["worker_started_at"] == FINGERPRINT
    assert "terminal_worker_reaped" not in [event.kind for event in kb.list_events(board, tid)]


@pytest.mark.parametrize("reading", [START + 1, None], ids=["drift", "unreadable"])
def test_timeout_holds_possibly_live_claim_without_signalling(board, monkeypatch, reading):
    tid, run_id = running_worker(board, max_runtime=1)
    monkeypatch.setattr(status, "get_process_start_time", lambda pid: reading)
    signals = []

    assert dispatch.enforce_max_runtime(board, signal_fn=lambda *args: signals.append(args)) == []

    assert signals == []
    assert kb.get_task(board, tid).status == "running"
    assert board.execute("SELECT ended_at FROM task_runs WHERE id = ?", (run_id,)).fetchone()[0] is None


@pytest.mark.parametrize("path", ["reclaim", "timeout"])
@pytest.mark.parametrize("reading", [START + 1, None], ids=["drift", "unreadable"])
def test_sigkill_requires_strict_identity_recheck_after_sigterm(board, monkeypatch, path, reading):
    tid, run_id = running_worker(board, max_runtime=1)
    monkeypatch.setattr(dispatch.time, "sleep", lambda seconds: None)
    signals = []

    def signal_fn(pid, sig):
        signals.append((pid, sig))
        monkeypatch.setattr(status, "get_process_start_time", lambda pid: reading)

    if path == "reclaim":
        termination = dispatch._terminate_reclaimed_worker(PID, "test-host:dispatcher",
                                                         started_at=FINGERPRINT, signal_fn=signal_fn)
        assert termination["terminated"] is False
        assert termination.get("signal_refused") is True
    else:
        assert dispatch.enforce_max_runtime(board, signal_fn=signal_fn) == []
        assert kb.get_task(board, tid).status == "running"
        assert board.execute("SELECT ended_at FROM task_runs WHERE id = ?", (run_id,)).fetchone()[0] is None
    assert signals == [(PID, dispatch.signal.SIGTERM)], "tolerant liveness must never authorize SIGKILL"


def test_timeout_holds_claim_when_probe_becomes_unreadable_before_sigterm(board, monkeypatch):
    tid, run_id = running_worker(board, max_runtime=1)
    readings = iter([START, None])
    monkeypatch.setattr(status, "get_process_start_time", lambda pid: next(readings, None))
    signals = []
    assert dispatch.enforce_max_runtime(board, signal_fn=lambda *args: signals.append(args)) == []
    assert signals == []
    assert board.execute("SELECT ended_at FROM task_runs WHERE id = ?", (run_id,)).fetchone()[0] is None
    assert kb.get_task(board, tid).status == "running"


@pytest.mark.parametrize("fingerprint,reading,epoch,alive,strict_refusal", [
    (FINGERPRINT, START, EPOCH, True, False),
    (FINGERPRINT, START - 200, EPOCH, True, True),
    (FINGERPRINT, START + 200, EPOCH, True, True),
    (FINGERPRINT, START - 201, EPOCH, False, True),
    (FINGERPRINT, START + 201, EPOCH, False, True),
    (FINGERPRINT, START, "another-boot:1", False, True),
    (FINGERPRINT, None, "another-boot:1", False, True),
    (f"|{START}", START + 1, "", True, True),
    (START, START, EPOCH, True, False),
    (START, START + 1, EPOCH, True, True),
    (START, None, EPOCH, True, True),
    (START, START + 201, EPOCH, False, True),
    (None, START, EPOCH, True, False),
    (dispatch.UNVERIFIED_WORKER_FINGERPRINT, START, EPOCH, True, True),
    (f"{EPOCH}|junk", START, EPOCH, False, True),
    (f"{EPOCH}|junk", None, EPOCH, False, True),
    (f"{EPOCH}|0", START, EPOCH, False, True),
])
def test_liveness_and_strict_signal_identity_are_separate(board, monkeypatch, fingerprint, reading, epoch,
                                                        alive, strict_refusal):
    monkeypatch.setattr(status, "get_process_start_time", lambda pid: reading)
    monkeypatch.setattr(drain_control, "current_instantiation_epoch", lambda: epoch)
    assert dispatch._worker_alive(PID, fingerprint) is alive
    assert dispatch._pid_recycled(PID, fingerprint) is strict_refusal


@pytest.mark.parametrize("cause", ["dead", "reboot", "reboot-unreadable", "recycled"])
def test_definitively_gone_worker_is_reclaimed_and_terminal_witness_cleared(board, monkeypatch, cause):
    tid, run_id = running_worker(board)
    if cause == "dead":
        monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)
    elif cause.startswith("reboot"):
        monkeypatch.setattr(drain_control, "current_instantiation_epoch", lambda: "another-boot:1")
        if cause == "reboot-unreadable":
            monkeypatch.setattr(status, "get_process_start_time", lambda pid: None)
    else:
        monkeypatch.setattr(status, "get_process_start_time", lambda pid: START + 201)

    sweep = dispatch._reclaim_dead_workers(board)

    assert sweep.crashed == [tid]
    assert kb.get_task(board, tid).status == "ready"
    run = board.execute("SELECT ended_at, outcome, worker_pid FROM task_runs WHERE id = ?", (run_id,)).fetchone()
    assert run["ended_at"] == NOW and run["outcome"] == "crashed" and run["worker_pid"] == PID
    board.execute("UPDATE task_runs SET ended_at = ? WHERE id = ?", (NOW - 600, run_id))
    signals = []
    assert dispatch.reap_terminal_workers(board, signal_fn=lambda *args: signals.append(args)) == []
    assert signals == []
    assert board.execute("SELECT worker_pid FROM task_runs WHERE id = ?", (run_id,)).fetchone()[0] is None


@pytest.mark.parametrize("reading", [START + 1, None], ids=["drift", "unreadable"])
def test_expired_claim_is_extended_beside_possibly_live_worker(board, monkeypatch, reading):
    tid, run_id = running_worker(board)
    monkeypatch.setattr(status, "get_process_start_time", lambda pid: reading)
    signals = []
    assert kb.release_stale_claims(board, signal_fn=lambda *args: signals.append(args)) == 0
    assert signals == []
    task = kb.get_task(board, tid)
    assert task.status == "running" and task.current_run_id == run_id and task.claim_expires > NOW
    assert "claim_extended" in [event.kind for event in kb.list_events(board, tid)]


def test_unreadable_probe_does_not_hold_claim_after_pid_exits(board, monkeypatch):
    tid, run_id = running_worker(board)
    monkeypatch.setattr(status, "get_process_start_time", lambda pid: None)
    assert dispatch._reclaim_dead_workers(board).crashed == []
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)
    assert dispatch._reclaim_dead_workers(board).crashed == [tid]
    assert board.execute("SELECT outcome FROM task_runs WHERE id = ?", (run_id,)).fetchone()[0] == "crashed"


@pytest.mark.parametrize("path", ["reclaim", "timeout", "terminal-reap"])
def test_exact_signal_identity_still_permits_sigterm_and_sigkill(board, monkeypatch, path):
    tid, run_id = running_worker(board, max_runtime=1)
    if path == "terminal-reap":
        assert kb.complete_task(board, tid, result="fixture done", expected_run_id=run_id)
        board.execute("UPDATE task_runs SET ended_at = ? WHERE id = ?", (NOW - 600, run_id))
    monkeypatch.setattr(dispatch.time, "sleep", lambda seconds: None)
    signals = []

    def signal_fn(pid, sig):
        signals.append((pid, sig))
        if sig == dispatch.signal.SIGKILL:
            monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)

    if path == "reclaim":
        termination = dispatch._terminate_reclaimed_worker(PID, "test-host:dispatcher",
                                                         started_at=FINGERPRINT, signal_fn=signal_fn)
        assert termination["terminated"] and termination["sigkill"]
    elif path == "timeout":
        assert dispatch.enforce_max_runtime(board, signal_fn=signal_fn) == [tid]
    else:
        assert dispatch.reap_terminal_workers(board, signal_fn=signal_fn) == [tid]
        assert board.execute("SELECT worker_pid FROM task_runs WHERE id = ?", (run_id,)).fetchone()[0] is None
    assert signals == [(PID, dispatch.signal.SIGTERM), (PID, dispatch.signal.SIGKILL)]
