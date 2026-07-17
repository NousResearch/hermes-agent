"""Lifecycle tests for the detached Kanban worker supervisor."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_worker_supervisor as supervisor


def _wait_until_gone(pid: int, *, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.02)
    return False


def _wait_for_pid_file(path: Path, *, timeout: float = 5.0) -> int:
    """Wait until *path* holds a complete positive PID; open() precedes write()."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            pid = int(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            pid = 0
        if pid > 0:
            return pid
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for a parseable pid in {path}")


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group semantics only")
def _start_frozen_term_ignoring_supervisor(tmp_path, monkeypatch, *, max_runtime=None):
    """Create the adversarial boundary: stopped supervisor + live stubborn child."""
    from hermes_cli import kanban_db as kb

    child_pid_path = tmp_path / "child.pid"
    spec_path = tmp_path / "worker-spec.json"
    handshake_path = tmp_path / "supervisor.pid"
    log_path = tmp_path / "worker.log"
    child_code = (
        "import os, signal, sys, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "open(sys.argv[1], 'w').write(str(os.getpid())); time.sleep(60)"
    )
    spec_path.write_text(
        json.dumps(
            {
                "command": [sys.executable, "-c", child_code, str(child_pid_path)],
                "cwd": None,
                "log_path": str(log_path),
                "handshake_path": str(handshake_path),
                "task_id": "placeholder",
                "run_id": 1,
                "claim_lock": "placeholder",
                "board": "default",
            },
        ),
        encoding="utf-8",
    )

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="stubborn tree", assignee="worker",
            max_runtime_seconds=max_runtime,
        )
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None
        assert claimed.claim_lock is not None
        assert claimed.current_run_id is not None

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec.update(
        task_id=task_id,
        run_id=claimed.current_run_id,
        claim_lock=claimed.claim_lock,
    )
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    repo_root = Path(__file__).parents[2]
    proc = subprocess.Popen(
        [sys.executable, "-m", "hermes_cli.kanban_worker_supervisor", str(spec_path)],
        cwd=repo_root,
        start_new_session=True,
    )
    try:
        assert _wait_for_pid_file(handshake_path, timeout=5) == proc.pid
        child_pid = _wait_for_pid_file(child_pid_path, timeout=5)
        # Freeze after the supervisor has published and spawned.  A direct
        # supervisor SIGKILL at this interleaving leaves the stubborn child
        # orphaned; group SIGTERM/KILL must remove both.
        os.kill(proc.pid, signal.SIGSTOP)
        monkeypatch.setattr(kb, "_WORKER_TREE_TERMINATE_GRACE_SECONDS", 0.12)
        monkeypatch.setattr(kb, "_WORKER_TREE_TERMINATE_POLL_SECONDS", 0.01)
        return kb, task_id, claimed, proc, child_pid
    except BaseException:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        raise


def _cleanup_process_group(proc):
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group semantics only")
def test_generic_crash_sweeper_defers_dead_supervisor_with_live_child(tmp_path, monkeypatch):
    """A dead group leader is not enough proof to release its live worker tree."""
    from hermes_cli import kanban_db as kb

    real_killpg = os.killpg
    group_checks = []

    def observe_group_probe(pgid, sig):
        group_checks.append((pgid, sig))
        return real_killpg(pgid, sig)

    monkeypatch.setattr(kb.os, "killpg", observe_group_probe)

    ready_path = tmp_path / "supervisor.ready"
    release_path = tmp_path / "supervisor.release"
    child_pid_path = tmp_path / "child.pid"
    child_code = (
        "import os, signal, sys, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "open(sys.argv[1], 'w').write(str(os.getpid())); time.sleep(60)"
    )
    supervisor_code = """
import os
import subprocess
import sys
import time

ready, release, child_pid, child_code = sys.argv[1:5]
open(ready, "w").write(str(os.getpid()))
while not os.path.exists(release):
    time.sleep(0.01)
subprocess.Popen([sys.executable, "-c", child_code, child_pid])
"""
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            supervisor_code,
            str(ready_path),
            str(release_path),
            str(child_pid_path),
            child_code,
        ],
        start_new_session=True,
    )
    child_pid = None
    try:
        assert _wait_for_pid_file(ready_path, timeout=5) == proc.pid

        with kb.connect() as conn:
            task_id = kb.create_task(conn, title="orphaned child", assignee="worker")
            claimed = kb.claim_task(conn, task_id)
            assert claimed is not None
            assert claimed.claim_lock is not None
            assert claimed.current_run_id is not None
            assert kb._set_worker_pid(conn, task_id, proc.pid)
            original_identity = kb.get_task(conn, task_id).worker_identity
            assert original_identity is not None

            release_path.touch()
            proc.wait(timeout=5)
            child_pid = _wait_for_pid_file(child_pid_path, timeout=5)
            assert os.getpgid(child_pid) == proc.pid

            # Age the launch beyond the generic sweeper's normal grace period
            # and expire both leases, so only the defer's own extension can
            # keep the stale-claim sweep away from this live tree.
            old = int(time.time()) - 120
            expired = int(time.time()) - 30
            conn.execute(
                "UPDATE tasks SET started_at = ?, claim_expires = ? WHERE id = ?",
                (old, expired, task_id),
            )
            conn.execute(
                "UPDATE task_runs SET started_at = ?, claim_expires = ? WHERE id = ?",
                (old, expired, claimed.current_run_id),
            )

            sweep_time = int(time.time())
            assert kb.detect_crashed_workers(conn) == []
            # The defer must have renewed the expired leases; a stale-claim
            # sweep inside that grace window must not release the live tree.
            assert kb.release_stale_claims(conn) == 0
            task = kb.get_task(conn, task_id)
            run = kb.latest_run(conn, task_id)
            events = kb.list_events(conn, task_id)
            spawned = []
            dispatch = kb.dispatch_once(
                conn,
                spawn_fn=lambda *args: spawned.append(args) or None,
            )

        assert task is not None
        assert task.status == "running"
        assert task.claim_lock == claimed.claim_lock
        assert task.current_run_id == claimed.current_run_id
        assert task.worker_pid == proc.pid
        assert task.worker_identity == original_identity
        assert task.claim_expires is not None
        assert task.claim_expires >= sweep_time + kb.RECLAIM_DEFER_GRACE_SECONDS
        assert run is not None
        assert run.claim_lock == claimed.claim_lock
        assert run.worker_pid == proc.pid
        assert run.worker_identity == original_identity
        assert run.claim_expires is not None
        assert run.claim_expires >= sweep_time + kb.RECLAIM_DEFER_GRACE_SECONDS
        assert child_pid is not None
        os.kill(child_pid, 0)
        assert spawned == []
        assert task_id not in dispatch.crashed
        assert dispatch.deferred == [task_id]
        assert group_checks and all(sig == 0 for _pgid, sig in group_checks)
        deferred = [event for event in events if event.kind == "reclaim_deferred"]
        assert len(deferred) == 1
        assert deferred[0].payload is not None
        assert deferred[0].payload["reason"] == "dead_supervisor_process_group_live"
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if child_pid is not None:
            assert _wait_until_gone(child_pid), "failed to clean up test child"


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group semantics only")
def test_manual_reclaim_releases_dead_supervisor_with_gone_group(tmp_path, monkeypatch):
    """An operator reclaim recovers a reaped supervisor via the signal-free probe.

    The supervisor registers its PID/incarnation while provably alive, then
    exits with no children and is reaped.  ``_terminate_reclaimed_worker`` can
    then only report ``leader_gone`` — manual reclaim must apply the same
    dead-group proof as operator complete/block/archive/delete instead of
    deferring until a dispatcher tick.
    """
    from hermes_cli import kanban_db as kb

    real_killpg = os.killpg
    group_checks = []

    def observe_group_probe(pgid, sig):
        group_checks.append((pgid, sig))
        return real_killpg(pgid, sig)

    monkeypatch.setattr(kb.os, "killpg", observe_group_probe)

    ready_path = tmp_path / "supervisor.ready"
    release_path = tmp_path / "supervisor.release"
    supervisor_code = """
import os
import sys
import time

ready, release = sys.argv[1:3]
open(ready, "w").write(str(os.getpid()))
while not os.path.exists(release):
    time.sleep(0.01)
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", supervisor_code, str(ready_path), str(release_path)],
        start_new_session=True,
    )
    try:
        assert _wait_for_pid_file(ready_path, timeout=5) == proc.pid

        with kb.connect() as conn:
            task_id = kb.create_task(conn, title="reaped leader", assignee="worker")
            claimed = kb.claim_task(conn, task_id)
            assert claimed is not None
            assert claimed.claim_lock is not None
            assert claimed.current_run_id is not None
            assert kb._set_worker_pid(conn, task_id, proc.pid)
            registered = kb.get_task(conn, task_id)
            assert registered is not None
            assert registered.worker_identity is not None

            # The leader exits childless and is reaped: the PID-named group
            # is gone while the persisted registration still points at it.
            release_path.touch()
            proc.wait(timeout=5)

            assert kb.reclaim_task(conn, task_id, reason="operator recovery")
            task = kb.get_task(conn, task_id)
            run = kb.latest_run(conn, task_id)
            events = kb.list_events(conn, task_id)

        assert task is not None
        assert task.status == "ready"
        assert task.claim_lock is None
        assert task.claim_expires is None
        assert task.worker_pid is None
        assert task.worker_identity is None
        assert task.current_run_id is None
        assert run is not None
        assert run.id == claimed.current_run_id
        assert run.status == "reclaimed"
        assert run.outcome == "reclaimed"
        assert run.ended_at is not None
        assert run.claim_lock is None
        assert run.claim_expires is None
        assert run.worker_pid is None
        assert run.worker_identity is None
        assert group_checks and all(sig == 0 for _pgid, sig in group_checks)
        reclaimed = [event for event in events if event.kind == "reclaimed"]
        assert len(reclaimed) == 1
        assert reclaimed[0].payload is not None
        assert reclaimed[0].payload["manual"] is True
        assert reclaimed[0].payload["identity"] == "leader_gone"
        assert reclaimed[0].payload["tree_target"] == "dead_supervisor_group_probe"
        assert (
            reclaimed[0].payload["dead_supervisor_group_state"]
            == "process_group_gone"
        )
    finally:
        if proc.poll() is None:
            try:
                real_killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group semantics only")
def test_manual_reclaim_kills_frozen_supervisor_tree_before_release(tmp_path, monkeypatch):
    kb, task_id, _claimed, proc, child_pid = _start_frozen_term_ignoring_supervisor(
        tmp_path, monkeypatch,
    )
    try:
        with kb.connect() as conn:
            assert kb._set_worker_pid(conn, task_id, proc.pid)
            assert kb.reclaim_task(conn, task_id, reason="real tree regression")
            assert _wait_until_gone(child_pid), "manual reclaim orphaned child"
            task = kb.get_task(conn, task_id)
            assert task is not None
            assert task.status == "ready"
            assert task.claim_lock is None
            assert task.worker_pid is None
    finally:
        _cleanup_process_group(proc)


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group semantics only")
def test_worker_schedule_uses_exact_capability_without_signalling_itself(tmp_path, monkeypatch):
    """A valid worker schedule closes only its own exact run, without killpg."""
    kb, task_id, claimed, proc, child_pid = _start_frozen_term_ignoring_supervisor(
        tmp_path, monkeypatch,
    )
    try:
        signals = []
        with monkeypatch.context() as signal_observer:
            with kb.connect() as conn:
                assert kb._set_worker_pid(conn, task_id, proc.pid)
                signal_observer.setattr(kb.os, "killpg", lambda *args: signals.append(args))
                assert kb.schedule_task(
                    conn,
                    task_id,
                    reason="worker reschedule",
                    expected_run_id=claimed.current_run_id,
                    expected_claim_lock=claimed.claim_lock,
                )
                task = kb.get_task(conn, task_id)
                run = kb.latest_run(conn, task_id)

            assert signals == []

        assert task is not None
        assert task.status == "scheduled"
        assert task.claim_lock is None
        assert task.worker_pid is None
        assert run is not None
        assert run.status == "scheduled"
        assert run.worker_pid is None
        os.kill(child_pid, 0)
    finally:
        _cleanup_process_group(proc)


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group semantics only")
@pytest.mark.parametrize("operation", ["complete", "block"])
def test_operator_completion_transition_kills_detached_tree_before_clearing_claim(
    tmp_path, monkeypatch, operation,
):
    """Operator complete/block cannot lose a stopped supervisor's worker tree."""
    kb, task_id, _claimed, proc, child_pid = _start_frozen_term_ignoring_supervisor(
        tmp_path, monkeypatch,
    )
    try:
        with kb.connect() as conn:
            assert kb._set_worker_pid(conn, task_id, proc.pid)
            if operation == "complete":
                assert kb.complete_task(conn, task_id, summary="operator completion")
                expected_status, expected_outcome = "done", "completed"
            else:
                assert kb.block_task(conn, task_id, reason="operator block")
                expected_status, expected_outcome = "blocked", "blocked"
            assert _wait_until_gone(child_pid), f"{operation} orphaned detached child"
            task = kb.get_task(conn, task_id)
            run = kb.latest_run(conn, task_id)

        assert task is not None
        assert task.status == expected_status
        assert task.claim_lock is None
        assert task.worker_pid is None
        assert task.worker_identity is None
        assert task.current_run_id is None
        assert run is not None
        assert run.outcome == expected_outcome
        assert run.worker_pid is None
        assert run.worker_identity is None
    finally:
        _cleanup_process_group(proc)


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group semantics only")
@pytest.mark.parametrize("operation", ["complete", "block"])
def test_operator_completion_transition_defers_unregistered_launch(tmp_path, operation):
    """No-PID launch windows cannot be terminally transitioned blind."""
    from hermes_cli import kanban_db as kb

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="launch window", assignee="worker")
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None
        assert claimed.current_run_id is not None
        if operation == "complete":
            assert not kb.complete_task(conn, task_id, summary="must defer")
        else:
            assert not kb.block_task(conn, task_id, reason="must defer")

        task = kb.get_task(conn, task_id)
        run = kb.latest_run(conn, task_id)
        deferred = [event for event in kb.list_events(conn, task_id) if event.kind == "reclaim_deferred"]

    assert task is not None
    assert task.status == "running"
    assert task.current_run_id == claimed.current_run_id
    assert task.claim_lock is not None
    assert task.worker_pid is None
    assert run is not None and run.ended_at is None
    assert len(deferred) == 1
    assert deferred[0].payload is not None
    assert deferred[0].payload["identity"] == "worker_pid_missing"


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group semantics only")
@pytest.mark.parametrize("operation", ["archive", "delete"])
def test_operator_terminal_transition_kills_detached_tree_before_mutating(
    tmp_path, monkeypatch, operation,
):
    """Archive/delete wait for verified KILL/reap before destroying ownership."""
    kb, task_id, _claimed, proc, child_pid = _start_frozen_term_ignoring_supervisor(
        tmp_path, monkeypatch,
    )
    try:
        with kb.connect() as conn:
            assert kb._set_worker_pid(conn, task_id, proc.pid)
            if operation == "archive":
                assert kb.archive_task(conn, task_id)
            else:
                assert kb.delete_task(conn, task_id)
            assert _wait_until_gone(
                child_pid,
            ), f"{operation} orphaned detached child"
            task = kb.get_task(conn, task_id)
            run_count = conn.execute(
                "SELECT COUNT(*) FROM task_runs WHERE task_id = ?", (task_id,),
            ).fetchone()[0]

        if operation == "archive":
            assert task is not None
            assert task.status == "archived"
            assert task.claim_lock is None
            assert task.worker_pid is None
            assert run_count == 1
        else:
            assert task is None
            assert run_count == 0
    finally:
        _cleanup_process_group(proc)


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group semantics only")
@pytest.mark.parametrize("operation", ["complete", "block", "schedule", "archive", "delete"])
def test_operator_transition_identity_mismatch_defers_without_mutation(
    tmp_path, monkeypatch, operation,
):
    """A replacement detached leader is never signalled or stripped of ownership.

    Literal PID reuse is nondeterministic, so the identity probe represents the
    persisted leader's start ticks changing underneath this real detached
    supervisor/TERM-ignoring-child tree.
    """
    kb, task_id, claimed, proc, child_pid = _start_frozen_term_ignoring_supervisor(
        tmp_path, monkeypatch,
    )
    real_killpg = os.killpg
    signals = []
    try:
        with kb.connect() as conn:
            assert kb._set_worker_pid(conn, task_id, proc.pid)
            task_before = kb.get_task(conn, task_id)
            assert task_before is not None
            assert task_before.worker_identity is not None
            assert task_before.claim_expires is not None
            run_before = kb.latest_run(conn, task_id)
            assert run_before is not None
            monkeypatch.setattr(
                kb, "_supervisor_process_identity", lambda _pid: "linux-start-ticks:replacement",
            )
            monkeypatch.setattr(kb.os, "killpg", lambda *args: signals.append(args))

            if operation == "complete":
                assert not kb.complete_task(conn, task_id, summary="identity replacement")
            elif operation == "block":
                assert not kb.block_task(conn, task_id, reason="identity replacement")
            elif operation == "schedule":
                assert not kb.schedule_task(conn, task_id)
            elif operation == "archive":
                assert not kb.archive_task(conn, task_id)
            else:
                assert not kb.delete_task(conn, task_id)

            task = kb.get_task(conn, task_id)
            run = kb.latest_run(conn, task_id)
            deferred = [
                event for event in kb.list_events(conn, task_id)
                if event.kind == "reclaim_deferred"
            ]

        assert task is not None
        assert task.status == "running"
        assert task.claim_lock == task_before.claim_lock
        assert task.worker_pid == proc.pid
        assert task.worker_identity == task_before.worker_identity
        # An unproven (identity-mismatched) tree must not have its lease
        # renewed: the claim stays bound but claim_expires is untouched.
        assert task.claim_expires == task_before.claim_expires
        assert run is not None
        assert run.claim_lock == task_before.claim_lock
        assert run.worker_pid == proc.pid
        assert run.worker_identity == task_before.worker_identity
        assert run.claim_expires == run_before.claim_expires
        assert signals == []
        assert len(deferred) == 1
        assert deferred[0].payload is not None
        assert deferred[0].payload["identity"] == "identity_mismatch"
        assert deferred[0].payload["reason"] == f"operator_{operation}_worker_alive"
        os.kill(child_pid, 0)
    finally:
        monkeypatch.undo()
        try:
            real_killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        _cleanup_process_group(proc)


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group semantics only")
def test_stale_reclaim_kills_frozen_supervisor_tree_before_release(tmp_path, monkeypatch):
    kb, task_id, _claimed, proc, child_pid = _start_frozen_term_ignoring_supervisor(
        tmp_path, monkeypatch,
    )
    try:
        with kb.connect() as conn:
            assert kb._set_worker_pid(conn, task_id, proc.pid)
            conn.execute(
                "UPDATE tasks SET claim_expires = ?, last_heartbeat_at = ? WHERE id = ?",
                (int(time.time()) - 1, int(time.time()) - 7200, task_id),
            )
            assert kb.release_stale_claims(conn) == 1
            assert _wait_until_gone(child_pid), "stale reclaim orphaned child"
            task = kb.get_task(conn, task_id)
            assert task is not None
            assert task.status == "ready"
            assert task.claim_lock is None
    finally:
        _cleanup_process_group(proc)


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group semantics only")
@pytest.mark.parametrize("reclaim_path", ["ttl", "heartbeat_stale"])
def test_stale_sweepers_hold_no_pid_publication_window_with_live_worker_tree(
    tmp_path, monkeypatch, reclaim_path,
):
    """No registered PID is never evidence that the detached tree is gone."""
    kb, task_id, claimed, proc, child_pid = _start_frozen_term_ignoring_supervisor(
        tmp_path, monkeypatch,
    )
    try:
        with kb.connect() as conn:
            # Deliberately leave the freshly launched supervisor unregistered:
            # this is the dispatcher's post-spawn/pre-PID-publication window.
            assert kb.get_task(conn, task_id).worker_pid is None
            old = int(time.time()) - 7200
            if reclaim_path == "ttl":
                conn.execute(
                    "UPDATE tasks SET claim_expires = ?, last_heartbeat_at = ? WHERE id = ?",
                    (int(time.time()) - 1, old, task_id),
                )
                assert kb.release_stale_claims(conn) == 0
            else:
                conn.execute("UPDATE tasks SET started_at = ? WHERE id = ?", (old, task_id))
                conn.execute(
                    "UPDATE task_runs SET started_at = ? WHERE id = ?",
                    (old, claimed.current_run_id),
                )
                assert kb.detect_stale_running(conn, stale_timeout_seconds=1) == []

            task = kb.get_task(conn, task_id)
            run = kb.get_run(conn, claimed.current_run_id)
            events = kb.list_events(conn, task_id)

        assert task is not None
        assert task.status == "running"
        assert task.claim_lock == claimed.claim_lock
        assert task.worker_pid is None
        assert task.current_run_id == claimed.current_run_id
        assert run is not None and run.ended_at is None
        assert proc.poll() is None
        os.kill(child_pid, 0)
        deferred = [event for event in events if event.kind == "reclaim_deferred"]
        assert len(deferred) == 1
        assert deferred[0].payload is not None
        assert deferred[0].payload["identity"] == "worker_pid_missing"
    finally:
        _cleanup_process_group(proc)


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group semantics only")
def test_max_runtime_kills_frozen_supervisor_tree_before_timeout_release(tmp_path, monkeypatch):
    kb, task_id, claimed, proc, child_pid = _start_frozen_term_ignoring_supervisor(
        tmp_path, monkeypatch, max_runtime=1,
    )
    try:
        with kb.connect() as conn:
            assert kb._set_worker_pid(conn, task_id, proc.pid)
            old = int(time.time()) - 10
            conn.execute("UPDATE tasks SET started_at = ? WHERE id = ?", (old, task_id))
            conn.execute(
                "UPDATE task_runs SET started_at = ? WHERE id = ?",
                (old, claimed.current_run_id),
            )
            assert kb.enforce_max_runtime(conn) == [task_id]
            assert _wait_until_gone(child_pid), "max runtime orphaned child"
            task = kb.get_task(conn, task_id)
            assert task is not None
            assert task.status == "ready"
            assert task.claim_lock is None
            assert task.consecutive_failures == 1
    finally:
        _cleanup_process_group(proc)


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group semantics only")
def test_reclaim_identity_mismatch_defers_without_signalling_unrelated_group(tmp_path):
    """A reused/non-leader PID is never converted into killpg(pid, ...)."""
    from hermes_cli import kanban_db as kb

    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        with kb.connect() as conn:
            task_id = kb.create_task(conn, title="pid reuse", assignee="worker")
            assert kb.claim_task(conn, task_id) is not None
            assert kb._set_worker_pid(conn, task_id, unrelated.pid)
            assert not kb.reclaim_task(conn, task_id, reason="identity check")
            assert not kb.reassign_task(
                conn, task_id, "other-worker", reclaim_first=True,
            )
            task = kb.get_task(conn, task_id)
            events = kb.list_events(conn, task_id)
        assert unrelated.poll() is None
        assert task is not None
        assert task.status == "running"
        assert task.assignee == "worker"
        deferred = [event for event in events if event.kind == "reclaim_deferred"]
        assert len(deferred) == 2
        for event in deferred:
            assert event.payload is not None
            assert event.payload["identity"] == "process_group_mismatch"
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)


@pytest.mark.skipif(os.name == "nt", reason="POSIX detached-session fixture only")
@pytest.mark.parametrize("reclaim_path", ["manual", "ttl", "max_runtime", "heartbeat_stale"])
def test_replacement_session_leader_identity_mismatch_defers_every_reclaim_path(
    monkeypatch, reclaim_path,
):
    """A replacement session leader never receives a reclaim tree signal.

    Literal PID reuse is impractical to force deterministically.  The injected
    identity probe models the old PID's registered start ticks changing to a
    replacement process while the real subprocess proves the dangerous shape:
    an unrelated process that is itself a fresh session/group leader.
    """
    from hermes_cli import kanban_db as kb

    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    identity = {"value": "linux-start-ticks:registered"}
    signals = []
    real_killpg = os.killpg
    monkeypatch.setattr(kb, "_supervisor_process_identity", lambda _pid: identity["value"])
    monkeypatch.setattr(kb.os, "killpg", lambda *args: signals.append(args))
    try:
        assert os.getpgid(unrelated.pid) == unrelated.pid
        assert os.getsid(unrelated.pid) == unrelated.pid
        with kb.connect() as conn:
            task_id = kb.create_task(
                conn,
                title="replacement session leader",
                assignee="worker",
                max_runtime_seconds=1 if reclaim_path == "max_runtime" else None,
            )
            claimed = kb.claim_task(conn, task_id)
            assert claimed is not None
            assert claimed.current_run_id is not None
            assert kb._set_worker_pid(conn, task_id, unrelated.pid)

            # Simulate PID reuse after registration. The new leader is not our
            # supervisor even though its PID/PGID/SID shape passes the old test.
            identity["value"] = "linux-start-ticks:replacement"
            old = int(time.time()) - 7200
            if reclaim_path == "ttl":
                conn.execute(
                    "UPDATE tasks SET claim_expires = ?, last_heartbeat_at = ? WHERE id = ?",
                    (int(time.time()) - 1, old, task_id),
                )
            elif reclaim_path in ("max_runtime", "heartbeat_stale"):
                conn.execute("UPDATE tasks SET started_at = ? WHERE id = ?", (old, task_id))
                conn.execute(
                    "UPDATE task_runs SET started_at = ? WHERE id = ?",
                    (old, claimed.current_run_id),
                )

            expires_before = (
                kb.get_task(conn, task_id).claim_expires,
                kb.latest_run(conn, task_id).claim_expires,
            )
            if reclaim_path == "manual":
                assert not kb.reclaim_task(conn, task_id, reason="identity replacement")
            elif reclaim_path == "ttl":
                assert kb.release_stale_claims(conn) == 0
            elif reclaim_path == "max_runtime":
                assert kb.enforce_max_runtime(conn) == []
            else:
                assert kb.detect_stale_running(conn, stale_timeout_seconds=1) == []

            task = kb.get_task(conn, task_id)
            run = kb.latest_run(conn, task_id)
            events = kb.list_events(conn, task_id)
        assert task is not None
        assert task.status == "running"
        assert task.worker_pid == unrelated.pid
        assert task.worker_identity == "linux-start-ticks:registered"
        assert run is not None
        assert run.worker_identity == "linux-start-ticks:registered"
        # Deferred ownership stays bound, but an unproven replacement leader
        # must never receive a lease renewal on the task or its open run.
        assert (task.claim_expires, run.claim_expires) == expires_before
        assert unrelated.poll() is None
        assert signals == []
        deferred = [event for event in events if event.kind == "reclaim_deferred"]
        assert len(deferred) == 1
        assert deferred[0].payload is not None
        assert deferred[0].payload["identity"] == "identity_mismatch"
    finally:
        real_killpg(unrelated.pid, signal.SIGKILL)
        unrelated.wait(timeout=5)


def test_windows_reclaim_tree_cleanup_fails_closed_without_incarnation_proof(monkeypatch):
    """Windows cannot safely taskkill a PID without an incarnation token."""
    from hermes_cli import kanban_db as kb

    calls = []
    monkeypatch.setattr(kb, "_IS_WINDOWS", True)
    monkeypatch.setattr(
        kb.subprocess,
        "run",
        lambda args, **kwargs: calls.append((args, kwargs)),
    )
    info = kb._terminate_reclaimed_worker(
        91015,
        f"{kb._claimer_id().split(':', 1)[0]}:x",
        worker_identity="unverifiable-on-windows",
        run_worker_identity="unverifiable-on-windows",
        run_worker_pid=91015,
        run_claim_lock=f"{kb._claimer_id().split(':', 1)[0]}:x",
    )
    assert info["identity"] == "identity_unavailable_platform"
    assert info["termination_deferred"] is True
    assert calls == []


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups only")
@pytest.mark.parametrize("supervisor_mode", ["timeout", "early_exit"])
def test_parent_handshake_failure_kills_real_supervisor_tree_and_cleans_artifacts(
    tmp_path, monkeypatch, supervisor_mode,
):
    """A no-handshake supervisor must not leave its real child behind."""
    from hermes_cli import kanban_db as kb

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(kb, "_WORKER_PID_HANDSHAKE_TIMEOUT_SECONDS", 0.08)
    db_path = kb.kanban_db_path()
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()

    child_pid_path = tmp_path / "child.pid"
    child_code = (
        "import os, signal, sys, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "open(sys.argv[1], 'w').write(str(os.getpid())); time.sleep(60)"
    )
    supervisor_code = (
        "import subprocess, sys, time; "
        "subprocess.Popen([sys.executable, '-c', sys.argv[2], sys.argv[1]]); "
        + ("sys.exit(23)" if supervisor_mode == "early_exit" else "time.sleep(60)")
    )
    real_popen = subprocess.Popen
    spawned = {}

    def spawn_no_handshake(_cmd, **_kwargs):
        proc = real_popen(
            [
                sys.executable,
                "-c",
                supervisor_code,
                str(child_pid_path),
                child_code,
            ],
            start_new_session=True,
        )
        spawned["child_pid"] = _wait_for_pid_file(child_pid_path, timeout=2)
        return proc

    monkeypatch.setattr(kb.subprocess, "Popen", spawn_no_handshake)
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="tree cleanup", assignee="worker")
        task = kb.claim_task(conn, task_id)
        assert task is not None
        with pytest.raises(RuntimeError, match=(
            "timed out" if supervisor_mode == "timeout" else "exited with code 23"
        )):
            kb._default_spawn(task, str(tmp_path))

    child_pid = spawned["child_pid"]
    assert _wait_until_gone(child_pid), "handshake cleanup orphaned the worker"
    log_dir = home / "kanban" / "logs"
    assert not list(log_dir.glob("*.spawn.json"))
    assert not list(log_dir.glob("*.worker.pid"))


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal semantics only")
def test_real_pid_publication_failure_kills_and_reaps_worker_tree(tmp_path):
    """Supervisor-side publication failure kills a real TERM-ignoring worker."""
    child_pid_path = tmp_path / "child.pid"
    spec_path = tmp_path / "worker-spec.json"
    log_path = tmp_path / "worker.log"
    handshake_path = tmp_path / "pid-directory"
    handshake_path.mkdir()
    child_code = (
        "import os, signal, sys, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "open(sys.argv[1], 'w').write(str(os.getpid())); time.sleep(60)"
    )
    spec_path.write_text(
        json.dumps(
            {
                "command": [sys.executable, "-c", child_code, str(child_pid_path)],
                "cwd": None,
                "log_path": str(log_path),
                "handshake_path": str(handshake_path),
                "task_id": "t_worker",
                "run_id": 1,
                "claim_lock": "host:worker",
                "board": "default",
            },
        ),
        encoding="utf-8",
    )

    real_popen = subprocess.Popen
    captured = {}

    def capture_worker(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        captured["pid"] = proc.pid
        return proc

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(supervisor.subprocess, "Popen", capture_worker)
    try:
        with pytest.raises(OSError):
            supervisor.run(spec_path)
    finally:
        monkeypatch.undo()

    child_pid = captured["pid"]
    assert _wait_until_gone(child_pid), "publication failure orphaned the worker"


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group semantics only")
def test_pid_publication_failure_reaps_term_ignoring_grandchild_tree(tmp_path):
    """The supervisor's own failed publication path kills its full session."""
    direct_pid_path = tmp_path / "direct.pid"
    grandchild_pid_path = tmp_path / "grandchild.pid"
    ready_path = tmp_path / "worker.ready"
    spec_path = tmp_path / "worker-spec.json"
    log_path = tmp_path / "worker.log"
    handshake_path = tmp_path / "worker.pid"
    grandchild_code = (
        "import os, signal, sys, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "open(sys.argv[1], 'w').write(str(os.getpid())); time.sleep(60)"
    )
    worker_code = (
        "import os, signal, subprocess, sys, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "subprocess.Popen([sys.executable, '-c', sys.argv[3], sys.argv[1]]); "
        "open(sys.argv[2], 'w').write(str(os.getpid())); "
        "open(sys.argv[4], 'w').write('ready'); time.sleep(60)"
    )
    spec_path.write_text(
        json.dumps(
            {
                "command": [
                    sys.executable, "-c", worker_code,
                    str(grandchild_pid_path), str(direct_pid_path),
                    grandchild_code, str(ready_path),
                ],
                "cwd": None,
                "log_path": str(log_path),
                "handshake_path": str(handshake_path),
                "task_id": "t_worker",
                "run_id": 1,
                "claim_lock": "host:worker",
                "board": "default",
            },
        ),
        encoding="utf-8",
    )
    launcher = """\
import sys
import time
from pathlib import Path
from hermes_cli import kanban_worker_supervisor as s

ready = Path(sys.argv[2])

def fail(_path, _pid):
    deadline = time.monotonic() + 5
    while not ready.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    if not ready.exists():
        raise RuntimeError("worker never became ready")
    raise OSError("forced publication failure")

s._write_supervisor_pid = fail
s.run(Path(sys.argv[1]))
"""
    repo_root = Path(__file__).parents[2]
    proc = subprocess.Popen(
        [sys.executable, "-c", launcher, str(spec_path), str(ready_path)],
        cwd=repo_root,
        start_new_session=True,
    )
    try:
        assert proc.wait(timeout=10) != 0
        direct_pid = int(direct_pid_path.read_text(encoding="utf-8"))
        grandchild_pid = int(grandchild_pid_path.read_text(encoding="utf-8"))
        assert _wait_until_gone(direct_pid), "publication failure orphaned worker"
        assert _wait_until_gone(grandchild_pid), "publication failure orphaned grandchild"
    finally:
        _cleanup_process_group(proc)


def test_windows_publication_failure_cleanup_uses_taskkill_tree(monkeypatch):
    """Windows uses taskkill /T /F before reaping an unpublished worker."""
    calls = []

    class Worker:
        pid = 91004

        def wait(self, timeout=None):
            calls.append(("wait", timeout))
            return -9

        def terminate(self):
            calls.append(("terminate",))

        def kill(self):
            calls.append(("kill",))

    monkeypatch.setattr(supervisor, "_IS_WINDOWS", True, raising=False)
    monkeypatch.setattr(
        supervisor.subprocess,
        "run",
        lambda args, **kwargs: calls.append(("taskkill", args, kwargs)),
    )

    supervisor._terminate_and_reap_worker(Worker())

    assert calls[0][0] == "taskkill"
    assert calls[0][1] == ["taskkill", "/PID", "91004", "/T", "/F"]
    assert calls[1] == ("wait", supervisor._WORKER_TERMINATE_TIMEOUT_SECONDS)


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal semantics only")
def test_direct_supervisor_sigterm_forwards_to_and_reaps_real_worker(tmp_path):
    """Stop/reclaim signaling the registered supervisor PID reaches its child."""
    child_pid_path = tmp_path / "child.pid"
    spec_path = tmp_path / "worker-spec.json"
    handshake_path = tmp_path / "supervisor.pid"
    log_path = tmp_path / "worker.log"
    child_code = (
        "import os, sys, time; "
        "open(sys.argv[1], 'w').write(str(os.getpid())); time.sleep(60)"
    )
    spec_path.write_text(
        json.dumps(
            {
                "command": [sys.executable, "-c", child_code, str(child_pid_path)],
                "cwd": None,
                "log_path": str(log_path),
                "handshake_path": str(handshake_path),
                "task_id": "t_worker",
                "run_id": 1,
                "claim_lock": "host:worker",
                "board": "default",
            },
        ),
        encoding="utf-8",
    )
    repo_root = Path(__file__).parents[2]
    proc = subprocess.Popen(
        [sys.executable, "-m", "hermes_cli.kanban_worker_supervisor", str(spec_path)],
        cwd=repo_root,
        start_new_session=True,
    )
    try:
        assert _wait_for_pid_file(handshake_path, timeout=5) == proc.pid
        child_pid = _wait_for_pid_file(child_pid_path, timeout=5)

        proc.terminate()
        proc.wait(timeout=5)
        assert _wait_until_gone(child_pid), "SIGTERM left the real worker alive"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_windows_tree_cleanup_uses_taskkill_and_reaps_supervisor(monkeypatch):
    """Windows takes the explicit taskkill /T path rather than POSIX signals."""
    from hermes_cli import kanban_db as kb

    calls = []

    class FakeSupervisor:
        def wait(self, timeout=None):
            calls.append(("wait", timeout))
            return 0

    monkeypatch.setattr(kb, "_IS_WINDOWS", True)
    monkeypatch.setattr(
        kb.subprocess,
        "run",
        lambda args, **kwargs: calls.append(("taskkill", args, kwargs)),
    )

    kb._terminate_supervisor_tree(91013, supervisor=FakeSupervisor())

    assert calls[0][0] == "taskkill"
    assert calls[0][1] == ["taskkill", "/PID", "91013", "/T", "/F"]
    assert calls[1] == ("wait", 5)


def test_handshake_timeout_preserves_original_error_when_cleanup_raises(
    tmp_path, monkeypatch,
):
    """A cleanup fault cannot replace the handshake timeout reported to dispatch."""
    from hermes_cli import kanban_db as kb

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    db_path = kb.kanban_db_path()
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    monkeypatch.setattr(kb, "_WORKER_PID_HANDSHAKE_TIMEOUT_SECONDS", 0.01)

    class FakeSupervisor:
        pid = 91014
        returncode = None

        def poll(self):
            return None

    monkeypatch.setattr(kb.subprocess, "Popen", lambda *_args, **_kwargs: FakeSupervisor())
    monkeypatch.setattr(
        kb,
        "_terminate_supervisor_tree",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("cleanup failure")),
    )
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="preserve timeout", assignee="worker")
        task = kb.claim_task(conn, task_id)
        assert task is not None
        with pytest.raises(RuntimeError, match="timed out before reporting the supervisor pid"):
            kb._default_spawn(task, str(tmp_path))


def test_pid_publication_failure_terminates_kills_and_reaps_worker(tmp_path, monkeypatch):
    """Never leave a worker alive when its PID cannot be published to dispatch."""
    spec_path = tmp_path / "worker-spec.json"
    log_path = tmp_path / "worker.log"
    handshake_path = tmp_path / "worker.pid"
    spec_path.write_text(
        json.dumps(
            {
                "command": ["worker"],
                "cwd": None,
                "log_path": str(log_path),
                "handshake_path": str(handshake_path),
                "task_id": "t_worker",
                "run_id": 1,
                "claim_lock": "host:worker",
                "board": "default",
            },
        ),
        encoding="utf-8",
    )

    class Worker:
        pid = 91004

        def __init__(self):
            self.terminated = 0
            self.killed = 0
            self.wait_timeouts: list[float | None] = []

        def terminate(self):
            self.terminated += 1

        def kill(self):
            self.killed += 1

        def wait(self, timeout=None):
            self.wait_timeouts.append(timeout)
            if self.killed == 0:
                raise subprocess.TimeoutExpired(["worker"], timeout or 0.0)
            return -9

    worker = Worker()
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *_args, **_kwargs: worker)

    def fail_pid_publication(_path, _pid):
        raise OSError("handshake disk failure")

    monkeypatch.setattr(supervisor, "_write_supervisor_pid", fail_pid_publication)

    with pytest.raises(OSError, match="handshake disk failure"):
        supervisor.run(spec_path)

    assert worker.terminated == 1
    assert worker.killed == 1
    assert len(worker.wait_timeouts) == 2
    assert worker.wait_timeouts[0] is not None
    assert worker.wait_timeouts[1] is None
