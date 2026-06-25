"""Regression tests for the per-profile protocol-violation circuit breaker."""

import json
import os
import time

import pytest

from hermes_cli import kanban_db as kb


def _make_protocol_violation(conn, task_id, profile="worker", fake_pid=999999):
    """Simulate a clean-exit protocol violation for ``task_id``."""
    import os as _os
    _os.environ["HERMES_KANBAN_CRASH_GRACE_SECONDS"] = "0"
    host_prefix = kb._claimer_id().split(":", 1)[0]
    lock = f"{host_prefix}:mock"
    kb.claim_task(conn, task_id, claimer=lock)
    kb._set_worker_pid(conn, task_id, fake_pid)
    kb._record_worker_exit(fake_pid, 0)
    original_alive = kb._pid_alive
    kb._pid_alive = lambda p: False
    try:
        kb.detect_crashed_workers(conn)
    finally:
        kb._pid_alive = original_alive


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Point kanban at an isolated temp home."""
    home = tmp_path / "kanban_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    # Speed up tests by defaulting threshold to 3 with no env override.
    yield home


def test_resolve_profile_protocol_violation_threshold(monkeypatch):
    """Threshold resolution follows documented precedence."""
    monkeypatch.setenv(
        "HERMES_KANBAN_PROFILE_PROTOCOL_VIOLATION_THRESHOLD_WORKER", "5"
    )
    assert kb._resolve_profile_protocol_violation_threshold("worker") == 5
    monkeypatch.setenv(
        "HERMES_KANBAN_PROFILE_PROTOCOL_VIOLATION_THRESHOLD_WORKER", "0"
    )
    assert kb._resolve_profile_protocol_violation_threshold("worker") == 0
    assert kb._resolve_profile_protocol_violation_threshold("other") > 0


def test_profile_circuit_opens_after_n_protocol_violations(kanban_home, monkeypatch):
    """After N distinct tasks of the same profile violate protocol, the
    circuit opens for that profile."""
    threshold = 3
    conn = kb.connect()
    try:
        task_ids = []
        for i in range(threshold):
            tid = kb.create_task(
                conn, title=f"quiet-{i}", assignee="worker",
            )
            task_ids.append(tid)
            _make_protocol_violation(conn, tid, fake_pid=999900 + i)

        is_open, effective, open_ids = kb._check_profile_protocol_violation_circuit(
            conn, "worker",
        )
        assert is_open, "expected circuit to open after threshold violations"
        assert effective == threshold
        assert set(open_ids) == set(task_ids)
    finally:
        conn.close()


def test_profile_circuit_counts_distinct_tasks_not_events(kanban_home):
    """Repeated protocol violations on the SAME task should not open the
    circuit by themselves."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="single-noisy", assignee="worker")
        for i in range(5):
            # Reclaim and re-trigger violation on the same task.
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET status='running', worker_pid=? "
                    "WHERE id=?",
                    (999990 + i, tid),
                )
            kb._record_worker_exit(999990 + i, 0)
            original_alive = kb._pid_alive
            kb._pid_alive = lambda p: False
            try:
                kb.detect_crashed_workers(conn)
            finally:
                kb._pid_alive = original_alive

        is_open, _, _ = kb._check_profile_protocol_violation_circuit(conn, "worker")
        assert not is_open, "same-task violations should not open circuit"
    finally:
        conn.close()


def test_profile_circuit_respects_window(kanban_home, monkeypatch):
    """Events older than the monitoring window are ignored."""
    threshold = 2
    conn = kb.connect()
    try:
        tid_old = kb.create_task(conn, title="old", assignee="worker")
        tid_new = kb.create_task(conn, title="new", assignee="worker")

        _make_protocol_violation(conn, tid_old, fake_pid=999800)
        # Age the old event beyond the window.
        monkeypatch.setattr(kb, "PROFILE_PROTOCOL_VIOLATION_WINDOW_SECONDS", 10)
        conn.execute(
            "UPDATE task_events SET created_at = ? WHERE task_id=? AND kind='protocol_violation'",
            (int(time.time()) - 100, tid_old),
        )

        _make_protocol_violation(conn, tid_new, fake_pid=999801)

        is_open, _, _ = kb._check_profile_protocol_violation_circuit(
            conn, "worker",
        )
        assert not is_open, "only one in-window violation should not open circuit"
    finally:
        conn.close()


def test_dispatch_once_records_open_circuits(kanban_home, monkeypatch):
    """dispatch_once exposes profile circuits in DispatchResult and skips
    spawning for the affected profile."""
    monkeypatch.setenv("HERMES_KANBAN_PROFILE_PROTOCOL_VIOLATION_THRESHOLD_WORKER", "2")
    # Force the fake profile to be treated as spawnable by mocking profile_exists.
    import hermes_cli.profiles as _profiles
    monkeypatch.setattr(_profiles, "profile_exists", lambda p: True)
    conn = kb.connect()
    try:
        t1 = kb.create_task(conn, title="q1", assignee="worker")
        t2 = kb.create_task(conn, title="q2", assignee="worker")
        t3 = kb.create_task(conn, title="pending", assignee="worker")

        _make_protocol_violation(conn, t1, fake_pid=999700)
        _make_protocol_violation(conn, t2, fake_pid=999701)

        # The third task is ready; with the circuit open it should not spawn.
        res = kb.dispatch_once(conn, spawn_fn=lambda task, workspace: 12345)

        profiles_open = {c["profile"] for c in res.profile_circuits_open}
        assert "worker" in profiles_open
        ready = {r["id"] for r in conn.execute(
            "SELECT id FROM tasks WHERE status='ready'"
        )}
        assert t3 in ready, "pending task should stay ready"
        spawned = [s[0] for s in res.spawned]
        assert t3 not in spawned, "circuit-open profile must not be spawned"
        guarded = {task_id for task_id, reason in res.respawn_guarded
                   if reason == "profile_protocol_violation_circuit_open"}
        assert t3 in guarded
    finally:
        conn.close()


def test_profile_incidents_telemetry_counts_violations_and_denominator(
    kanban_home,
):
    """profile_protocol_violation_telemetry returns violations, denominator,
    and rate for the profile within the configured window."""
    conn = kb.connect()
    try:
        t_done = kb.create_task(conn, title="ok", assignee="worker")
        kb.claim_task(conn, t_done)
        kb.complete_task(conn, t_done, result="done")

        t_pv = kb.create_task(conn, title="bad", assignee="worker")
        _make_protocol_violation(conn, t_pv, fake_pid=999600)

        tel = kb.profile_protocol_violation_telemetry(conn, "worker")
        assert tel["profile"] == "worker"
        assert tel["violations"] == 1
        assert tel["denominator"] == 2  # one done + one violation
        assert tel["rate_pct"] == 50
        assert tel["circuit_open"] is False  # threshold default is 3
        assert tel["window_seconds"] == kb.PROFILE_PROTOCOL_VIOLATION_WINDOW_SECONDS
    finally:
        conn.close()


def test_rate_limited_does_not_count_as_protocol_violation(kanban_home, monkeypatch):
    """rate_limited events are quota-wall requeues, not protocol
    violations, and must never contribute to the per-profile circuit."""
    import hermes_cli.kanban_db as _kb
    monkeypatch.setenv(
        "HERMES_KANBAN_PROFILE_PROTOCOL_VIOLATION_THRESHOLD", "1"
    )
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="quota", assignee="worker")
        host_prefix = _kb._claimer_id().split(":", 1)[0]
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status='running', worker_pid=?, "
                "claim_lock=? WHERE id=?",
                (12345, f"{host_prefix}:q", tid),
            )
        _kb._record_worker_exit(12345, _kb.KANBAN_RATE_LIMIT_EXIT_CODE << 8)
        original_alive = _kb._pid_alive
        _kb._pid_alive = lambda p: False
        try:
            kb.detect_crashed_workers(conn)
        finally:
            _kb._pid_alive = original_alive

        events = kb.list_events(conn, tid)
        kinds = [e.kind for e in events]
        assert "rate_limited" in kinds
        assert "protocol_violation" not in kinds

        is_open, _, _ = kb._check_profile_protocol_violation_circuit(
            conn, "worker",
        )
        assert not is_open, "rate-limited requeue must not open PV circuit"
    finally:
        conn.close()


def test_profile_circuit_respects_min_rate_guard(kanban_home, monkeypatch):
    """When min_rate_pct is set, a profile with very few terminations and
    threshold violations should NOT open the circuit."""
    monkeypatch.setenv(
        "HERMES_KANBAN_PROFILE_PROTOCOL_VIOLATION_THRESHOLD_WORKER", "1"
    )
    monkeypatch.setenv(
        "HERMES_KANBAN_PROFILE_PROTOCOL_VIOLATION_MIN_RATE_PCT", "51"
    )
    conn = kb.connect()
    try:
        t_done = kb.create_task(conn, title="ok", assignee="worker")
        kb.claim_task(conn, t_done)
        kb.complete_task(conn, t_done, result="done")

        t_pv = kb.create_task(conn, title="bad", assignee="worker")
        _make_protocol_violation(conn, t_pv, fake_pid=999500)

        is_open, threshold, _ = kb._check_profile_protocol_violation_circuit(
            conn, "worker",
        )
        assert threshold == 1
        assert not is_open, "rate below min_rate_pct should keep circuit closed"
    finally:
        conn.close()


def test_profile_circuit_respects_min_denominator_guard(kanban_home, monkeypatch):
    """When min_denominator is set, a profile without enough terminated
    tasks in the window should NOT open the circuit."""
    monkeypatch.setenv(
        "HERMES_KANBAN_PROFILE_PROTOCOL_VIOLATION_THRESHOLD_WORKER", "1"
    )
    monkeypatch.setenv(
        "HERMES_KANBAN_PROFILE_PROTOCOL_VIOLATION_MIN_DENOMINATOR", "5"
    )
    conn = kb.connect()
    try:
        t_pv = kb.create_task(conn, title="bad", assignee="worker")
        _make_protocol_violation(conn, t_pv, fake_pid=999400)

        is_open, _, _ = kb._check_profile_protocol_violation_circuit(
            conn, "worker",
        )
        assert not is_open, "denominator below min_denominator should keep circuit closed"
    finally:
        conn.close()


def test_profile_circuit_open_event_includes_telemetry(kanban_home, monkeypatch):
    """profile_circuit_open events carry telemetry so dashboards and
    incident reports can show rate/denominator context."""
    monkeypatch.setenv(
        "HERMES_KANBAN_PROFILE_PROTOCOL_VIOLATION_THRESHOLD_WORKER", "1"
    )
    conn = kb.connect()
    try:
        t_pv = kb.create_task(conn, title="bad", assignee="worker")
        _make_protocol_violation(conn, t_pv, fake_pid=999300)

        open_circuits = kb._list_open_profile_protocol_violation_circuits(
            conn, emit_on_open=True,
        )
        assert len(open_circuits) == 1
        assert "telemetry" in open_circuits[0]
        tel = open_circuits[0]["telemetry"]
        assert tel["violations"] >= 1
        assert "denominator" in tel

        events = kb.list_events(conn, t_pv)
        pv_open_events = [e for e in events if e.kind == "profile_circuit_open"]
        assert pv_open_events
        assert "telemetry" in (pv_open_events[0].payload or {})
    finally:
        conn.close()
