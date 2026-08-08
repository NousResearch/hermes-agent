"""Regression coverage for terminal timeout worker teardown ordering."""

from __future__ import annotations

import signal
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _profile: True)
    kb.init_db()
    return home


@pytest.mark.live_system_guard_bypass
def test_terminal_timeout_kills_previous_worker_before_retry_spawn(kanban_home):
    """A worker-side timeout must not release a live writer into a retry race."""
    previous_worker = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"]
    )
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="timeout retry", assignee="developer")
        assert kb.claim_task(conn, task_id) is not None
        first_run = kb.latest_run(conn, task_id)
        assert first_run is not None
        kb._set_worker_pid(conn, task_id, previous_worker.pid)

        kb._record_task_failure(
            conn,
            task_id,
            error="Iteration budget exhausted (90/90)",
            outcome="timed_out",
            release_claim=True,
            end_run=True,
            event_payload_extra={"budget_used": 90, "budget_max": 90},
        )

        task = kb.get_task(conn, task_id)
        closed_run = kb.get_run(conn, first_run.id)
        assert task is not None
        assert closed_run is not None
        assert task.status == "ready"
        assert task.claim_lock is None
        assert task.current_run_id is None
        assert closed_run.outcome == "timed_out"
        assert closed_run.ended_at is not None

        alive_at_spawn = []

        def spawn_retry(_task, _workspace):
            alive_at_spawn.append(previous_worker.poll() is None)
            return None

        result = kb.dispatch_once(conn, spawn_fn=spawn_retry)

        assert result.spawned and result.spawned[0][0] == task_id
        assert alive_at_spawn == [False]
        previous_worker.wait(timeout=5)
        retry_run = kb.latest_run(conn, task_id)
        assert retry_run is not None
        assert retry_run.id != first_run.id
        assert retry_run.ended_at is None
    finally:
        conn.close()
        if previous_worker.poll() is None:
            previous_worker.terminate()
            previous_worker.wait(timeout=5)


def test_terminal_timeout_without_process_fingerprint_fails_closed(
    kanban_home, monkeypatch
):
    """An unverifiable live PID must block retry rather than risk overlap."""
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="timeout retry", assignee="developer")
        assert kb.claim_task(conn, task_id) is not None
        kb._set_worker_pid(conn, task_id, 424242)
        monkeypatch.setattr(kb, "_worker_process_start_time", lambda _pid: None)

        kb._record_task_failure(
            conn,
            task_id,
            error="Iteration budget exhausted (90/90)",
            outcome="timed_out",
            release_claim=True,
            end_run=True,
        )

        monkeypatch.setattr(kb, "_pid_alive", lambda _pid: True)
        termination_calls = []

        def terminate_worker(pid, claimer):
            termination_calls.append((pid, claimer))
            return {
                "host_local": True,
                "termination_attempted": True,
                "terminated": True,
            }

        monkeypatch.setattr(kb, "_terminate_reclaimed_worker", terminate_worker)
        spawn_calls = []
        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda task, _workspace: spawn_calls.append(task.id),
        )

        assert spawn_calls == []
        assert termination_calls == []
        assert result.respawn_guarded == [
            (task_id, "timed_out_worker_identity_unverified")
        ]
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "ready"
        assert task.claim_lock is None
        assert task.current_run_id is None
    finally:
        conn.close()


@pytest.mark.live_system_guard_bypass
def test_direct_claim_reaps_terminal_timeout_worker_first(kanban_home):
    """Manual/control-plane claims must use the same teardown boundary."""
    previous_worker = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"]
    )
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="direct timeout retry")
        assert kb.claim_task(conn, task_id) is not None
        kb._set_worker_pid(conn, task_id, previous_worker.pid)
        kb._record_task_failure(
            conn,
            task_id,
            error="Iteration budget exhausted (90/90)",
            outcome="timed_out",
            release_claim=True,
            end_run=True,
        )

        retry = kb.claim_task(conn, task_id)

        assert retry is not None
        assert previous_worker.poll() is not None
    finally:
        conn.close()
        if previous_worker.poll() is None:
            previous_worker.terminate()
            previous_worker.wait(timeout=5)


def test_stale_timeout_report_cannot_close_a_newer_run(kanban_home):
    """A delayed finalizer must be bound to its original run and claim."""
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="stale finalizer")
        first_claim = kb.claim_task(conn, task_id)
        assert first_claim is not None
        first_run = kb.latest_run(conn, task_id)
        assert first_run is not None
        first_claim_lock = first_claim.claim_lock
        assert first_claim_lock is not None

        kb._record_task_failure(
            conn,
            task_id,
            error="first attempt ended",
            outcome="crashed",
            release_claim=True,
            end_run=True,
        )
        second_claim = kb.claim_task(conn, task_id)
        assert second_claim is not None
        second_run = kb.latest_run(conn, task_id)
        assert second_run is not None

        kb._record_task_failure(
            conn,
            task_id,
            error="late timeout from first attempt",
            outcome="timed_out",
            release_claim=True,
            end_run=True,
            expected_run_id=first_run.id,
            expected_claim_lock=first_claim_lock,
        )

        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "running"
        assert task.current_run_id == second_run.id
        assert task.claim_lock == second_claim.claim_lock
        active_run = kb.get_run(conn, second_run.id)
        assert active_run is not None
        assert active_run.ended_at is None
    finally:
        conn.close()


@pytest.mark.live_system_guard_bypass
def test_unblocked_gave_up_timeout_reaps_worker_before_claim(kanban_home):
    """A breaker outcome must retain the timeout teardown requirement."""
    previous_worker = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"]
    )
    conn = kb.connect()
    try:
        task_id = kb.create_task(
            conn,
            title="gave-up timeout retry",
            max_retries=1,
        )
        assert kb.claim_task(conn, task_id) is not None
        kb._set_worker_pid(conn, task_id, previous_worker.pid)
        blocked = kb._record_task_failure(
            conn,
            task_id,
            error="Iteration budget exhausted (90/90)",
            outcome="timed_out",
            release_claim=True,
            end_run=True,
        )
        assert blocked is True
        assert kb.unblock_task(conn, task_id) is True

        retry = kb.claim_task(conn, task_id)

        assert retry is not None
        assert previous_worker.poll() is not None
    finally:
        conn.close()
        if previous_worker.poll() is None:
            previous_worker.terminate()
            previous_worker.wait(timeout=5)


def test_pid_reuse_after_sigterm_never_receives_sigkill(kanban_home, monkeypatch):
    """Revalidate the process fingerprint throughout the termination grace."""
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="pid reuse")
        assert kb.claim_task(conn, task_id) is not None
        kb._set_worker_pid(conn, task_id, 424242)
        monkeypatch.setattr(kb, "_worker_process_start_time", lambda _pid: 100)
        kb._record_task_failure(
            conn,
            task_id,
            error="Iteration budget exhausted (90/90)",
            outcome="timed_out",
            release_claim=True,
            end_run=True,
        )

        process_starts = iter([100, 100, 200])
        monkeypatch.setattr(
            kb,
            "_worker_process_start_time",
            lambda _pid: next(process_starts),
        )
        monkeypatch.setattr(kb, "_pid_alive", lambda _pid: True)
        monkeypatch.setattr(kb.time, "sleep", lambda _seconds: None)
        signals = []
        monkeypatch.setattr(kb.os, "kill", lambda pid, sig: signals.append((pid, sig)))

        retry = kb.claim_task(conn, task_id)

        assert retry is not None
        assert signals == [(424242, signal.SIGTERM)]
        assert all(sig != signal.SIGKILL for _, sig in signals)
    finally:
        conn.close()


def test_timeout_escalation_uses_sigterm_when_sigkill_is_unavailable(
    kanban_home, monkeypatch
):
    """Windows-style signal modules must still terminate before retry."""
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="portable escalation")
        assert kb.claim_task(conn, task_id) is not None
        kb._set_worker_pid(conn, task_id, 424242)
        monkeypatch.setattr(kb, "_worker_process_start_time", lambda _pid: 100)
        kb._record_task_failure(
            conn,
            task_id,
            error="Iteration budget exhausted (90/90)",
            outcome="timed_out",
            release_claim=True,
            end_run=True,
        )

        alive = {"value": True}
        signals = []

        def signal_worker(pid, sig):
            signals.append((pid, sig))
            if len(signals) == 2:
                alive["value"] = False

        monkeypatch.setattr(kb, "_pid_alive", lambda _pid: alive["value"])
        monkeypatch.setattr(kb.time, "sleep", lambda _seconds: None)
        monkeypatch.setattr(kb.os, "kill", signal_worker)
        monkeypatch.delattr(signal, "SIGKILL", raising=False)

        retry = kb.claim_task(conn, task_id)

        assert retry is not None
        assert signals == [
            (424242, signal.SIGTERM),
            (424242, signal.SIGTERM),
        ]
    finally:
        conn.close()


def test_remote_timeout_worker_fails_closed_at_claim_boundary(
    kanban_home, monkeypatch
):
    """Never infer remote worker death from this host's PID namespace."""
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="remote timeout")
        first_claim = kb.claim_task(conn, task_id, claimer="remote-host:claim")
        assert first_claim is not None
        kb._set_worker_pid(conn, task_id, 424242)
        monkeypatch.setattr(kb, "_worker_process_start_time", lambda _pid: 100)
        kb._record_task_failure(
            conn,
            task_id,
            error="Iteration budget exhausted (90/90)",
            outcome="timed_out",
            release_claim=True,
            end_run=True,
        )

        monkeypatch.setattr(kb, "_pid_alive", lambda _pid: True)
        signals = []
        monkeypatch.setattr(kb.os, "kill", lambda pid, sig: signals.append((pid, sig)))

        retry = kb.claim_task(conn, task_id)

        assert retry is None
        assert signals == []
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "ready"
        assert task.claim_lock is None
    finally:
        conn.close()
