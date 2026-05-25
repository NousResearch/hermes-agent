"""PR3 kanban liveness & reconciliation tests.

Covers the three PR3 pillars:

1. ``DispatchResult.claim_extended`` — TTL-expired claims extended because
   the host-local PID is still alive (prevents false-ready transitions).
2. ``DispatchResult.live_pid_reclaim_skipped`` — stuck PIDs that survived
   SIGTERM+SIGKILL are not set to ``ready`` (no duplicate spawns).
3. Clean-exit auto-reconciliation — a worker that exits rc=0 with a
   recent heartbeat is transitioned to ``done`` rather than auto-blocked
   as a protocol violation.

Plus board/reality-drift diagnostic rules:
4. ``stuck_pid_liveness`` diagnostic — fires when events show the dispatcher
   could not kill the worker.
5. ``ttl_extensions_without_heartbeat`` diagnostic — fires when the claim
   TTL has been extended N times without a worker heartbeat.

And the enriched heartbeat tool:
6. Expired-claim specific error message from ``kanban_heartbeat``.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_diagnostics as kd


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _task_dict(**overrides):
    base = {
        "id": "t_demo00",
        "title": "demo",
        "assignee": "demo",
        "status": "running",
        "consecutive_failures": 0,
        "last_failure_error": None,
        "worker_pid": 12345,
        "claim_lock": "host:demo:12345",
        "claim_expires": int(time.time()) + 900,
        "last_heartbeat_at": int(time.time()) - 30,
        "created_at": int(time.time()) - 120,
    }
    base.update(overrides)
    return base


def _event_dict(kind, ts=None, **payload):
    return {
        "kind": kind,
        "created_at": int(ts if ts is not None else time.time()),
        "payload": payload or None,
    }


# ---------------------------------------------------------------------------
# 1. DispatchResult.claim_extended — PID-alive extension tracking
# ---------------------------------------------------------------------------

def test_claim_extended_populated_when_pid_alive(kanban_home, monkeypatch):
    """release_stale_claims extends an expired claim when PID is alive.
    dispatch_once surfaces this in DispatchResult.claim_extended."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="long-runner", assignee="demo")
        # Claim the task so it's running
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None

        # Set a fake worker PID and force the claim to appear expired.
        # worker_pid is normally set by _set_worker_pid after spawn;
        # we set it directly here to simulate a running dispatcher worker.
        fake_pid = 54321
        past = int(time.time()) - 10
        conn.execute(
            "UPDATE tasks SET claim_expires = ?, worker_pid = ? WHERE id = ?",
            (past, fake_pid, tid),
        )
        conn.commit()

        # Simulate a live PID: _pid_alive → True
        with (
            patch.object(kb, "_pid_alive", return_value=True),
            patch.object(kb, "_default_spawn", return_value=99999),
        ):
            result = kb.dispatch_once(conn, spawn_fn=lambda *a, **kw: None)

        # The task should NOT have been reclaimed to ready
        task = kb.get_task(conn, tid)
        assert task.status == "running", f"expected running, got {task.status}"

        # claim_extended must carry this task id
        assert tid in result.claim_extended, (
            f"expected {tid} in claim_extended, got {result.claim_extended}"
        )

    finally:
        conn.close()


def test_claim_not_extended_when_pid_dead(kanban_home, monkeypatch):
    """release_stale_claims reclaims an expired claim when PID is dead.
    claim_extended must NOT contain the task id."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="dead-worker", assignee="demo")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None

        fake_pid = 65432
        past = int(time.time()) - 10
        conn.execute(
            "UPDATE tasks SET claim_expires = ?, worker_pid = ? WHERE id = ?",
            (past, fake_pid, tid),
        )
        conn.commit()

        with (
            patch.object(kb, "_pid_alive", return_value=False),
            patch.object(kb, "_terminate_reclaimed_worker", return_value={
                "prev_pid": fake_pid, "host_local": True,
                "termination_attempted": True, "terminated": True,
                "sigkill": False,
            }),
        ):
            result = kb.dispatch_once(conn, spawn_fn=lambda *a, **kw: None)

        task = kb.get_task(conn, tid)
        assert task.status == "ready", f"expected ready, got {task.status}"
        assert tid not in result.claim_extended


    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 2. live_pid_reclaim_skipped — unkillable PID protection
# ---------------------------------------------------------------------------

def test_live_pid_reclaim_skipped_when_sigkill_fails(kanban_home):
    """If SIGTERM+SIGKILL fail and PID still alive, the task must NOT be
    set to ready.  live_pid_reclaim_skipped must contain the task id."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="unkillable", assignee="demo")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None

        past = int(time.time()) - 10
        # Also set worker_pid so the PID-alive checks have a non-null value
        conn.execute(
            "UPDATE tasks SET claim_expires = ?, worker_pid = 12345 WHERE id = ?",
            (past, tid),
        )
        conn.commit()

        # First call to _pid_alive (extension check) → False (PID dead for
        # the extension branch, but alive after termination attempt).
        # Simulate: not host-local extension path, termination fails + alive.
        pid_check_calls = []

        def _fake_pid_alive(pid):
            pid_check_calls.append(pid)
            # Second call (after termination): pid still alive
            if len(pid_check_calls) >= 2:
                return True
            return False  # first call: not the extension branch

        termination_result = {
            "prev_pid": 12345,
            "host_local": True,
            "termination_attempted": True,
            "terminated": False,  # SIGTERM failed
            "sigkill": True,      # SIGKILL attempted
        }

        with (
            patch.object(kb, "_pid_alive", side_effect=_fake_pid_alive),
            patch.object(kb, "_terminate_reclaimed_worker",
                         return_value=termination_result),
        ):
            result = kb.release_stale_claims(conn)

        task = kb.get_task(conn, tid)
        # Task must stay running — not set to ready
        assert task.status == "running", (
            f"expected running (unkillable PID), got {task.status}"
        )

        # Verify the stuck_pid_liveness event was emitted
        events = kb.list_events(conn, tid)
        stuck_evs = [e for e in events if e.kind == "stuck_pid_liveness"]
        assert stuck_evs, "expected stuck_pid_liveness event"

        # Check side-channel
        last_stuck = getattr(kb.release_stale_claims, "_last_stuck_pid", [])
        assert tid in last_stuck

    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 3. Clean-exit auto-reconciliation
# ---------------------------------------------------------------------------

def test_clean_exit_with_recent_heartbeat_auto_reconciles(kanban_home):
    """A worker that exits rc=0 within _CLEAN_EXIT_RECONCILE_WINDOW of its
    last heartbeat should be auto-reconciled to done, not auto-blocked."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="nearly-done", assignee="demo")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None

        # Set a very recent heartbeat (30 seconds ago)
        recent_hb = int(time.time()) - 30
        conn.execute(
            "UPDATE tasks SET last_heartbeat_at = ?, worker_pid = 55555 "
            "WHERE id = ?",
            (recent_hb, tid),
        )
        conn.commit()

        # Fake the worker PID as dead (clean exit, rc=0, recent reap)
        kb._record_worker_exit(55555, 0)  # record rc=0 in exit registry

        with patch.object(kb, "_pid_alive", return_value=False):
            crashed = kb.detect_crashed_workers(conn)

        task = kb.get_task(conn, tid)
        assert task.status == "done", (
            f"expected done (auto-reconciled), got {task.status}"
        )

        # crashed list must NOT include this task (it was reconciled, not crashed)
        assert tid not in crashed

        # reconciled list must include it
        reconciled = getattr(kb.detect_crashed_workers, "_last_reconciled", [])
        assert tid in reconciled

        # No 'gave_up' / 'protocol_violation' events
        events = kb.list_events(conn, tid)
        bad_kinds = {e.kind for e in events if e.kind in {
            "protocol_violation", "gave_up",
        }}
        assert not bad_kinds, f"unexpected events: {bad_kinds}"

    finally:
        conn.close()


def test_clean_exit_without_recent_heartbeat_is_protocol_violation(kanban_home):
    """A worker that exits rc=0 long after its last heartbeat triggers the
    existing protocol-violation auto-block (no reconciliation)."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="silent-worker", assignee="demo")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None

        # Heartbeat was 10 minutes ago (well outside reconcile window)
        old_hb = int(time.time()) - 600
        conn.execute(
            "UPDATE tasks SET last_heartbeat_at = ?, worker_pid = 66666 "
            "WHERE id = ?",
            (old_hb, tid),
        )
        conn.commit()

        kb._record_worker_exit(66666, 0)  # rc=0 in exit registry

        with patch.object(kb, "_pid_alive", return_value=False):
            crashed = kb.detect_crashed_workers(conn)

        task = kb.get_task(conn, tid)
        # Should be blocked (protocol violation → auto-block on first occurrence)
        assert task.status in ("ready", "blocked"), (
            f"expected ready or blocked for protocol violation, got {task.status}"
        )

        # NOT auto-reconciled
        reconciled = getattr(kb.detect_crashed_workers, "_last_reconciled", [])
        assert tid not in reconciled

    finally:
        conn.close()


def test_clean_exit_no_heartbeat_ever_is_protocol_violation(kanban_home):
    """A worker that exits rc=0 with no heartbeat at all is a protocol
    violation (existing behavior preserved)."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="never-heartbeat", assignee="demo")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None

        conn.execute(
            "UPDATE tasks SET last_heartbeat_at = NULL, worker_pid = 77777 "
            "WHERE id = ?",
            (tid,),
        )
        conn.commit()

        kb._record_worker_exit(77777, 0)

        with patch.object(kb, "_pid_alive", return_value=False):
            kb.detect_crashed_workers(conn)

        task = kb.get_task(conn, tid)
        assert task.status in ("ready", "blocked")

        reconciled = getattr(kb.detect_crashed_workers, "_last_reconciled", [])
        assert tid not in reconciled

    finally:
        conn.close()


def test_dispatch_result_reconciled_populated(kanban_home):
    """dispatch_once propagates detect_crashed_workers._last_reconciled into
    DispatchResult.reconciled."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="dispatch-reconcile", assignee="demo")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None

        recent_hb = int(time.time()) - 20
        conn.execute(
            "UPDATE tasks SET last_heartbeat_at = ?, worker_pid = 88888 "
            "WHERE id = ?",
            (recent_hb, tid),
        )
        conn.commit()

        kb._record_worker_exit(88888, 0)

        with patch.object(kb, "_pid_alive", return_value=False):
            result = kb.dispatch_once(conn, spawn_fn=lambda *a, **kw: None)

        assert tid in result.reconciled, (
            f"expected {tid} in result.reconciled, got {result.reconciled}"
        )

    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 4. stuck_pid_liveness diagnostic
# ---------------------------------------------------------------------------

def test_diagnostic_stuck_pid_liveness_fires(kanban_home):
    """_rule_stuck_pid_liveness fires critical when stuck_pid_liveness events
    are present on a running task."""
    now = int(time.time())
    task = _task_dict(status="running")
    events = [
        _event_dict("stuck_pid_liveness", ts=now - 120,
                    worker_pid=12345, reason="termination_failed_pid_alive"),
        _event_dict("stuck_pid_liveness", ts=now - 60,
                    worker_pid=12345, reason="termination_failed_pid_alive"),
    ]
    diags = kd.compute_task_diagnostics(task, events, [], now=now)
    stuck = [d for d in diags if d.kind == "stuck_pid_liveness"]
    assert len(stuck) == 1
    d = stuck[0]
    assert d.severity == "critical"
    assert d.count == 2
    assert d.data["worker_pid"] == 12345
    # At least one suggested action
    assert any(a.suggested for a in d.actions)


def test_diagnostic_stuck_pid_liveness_clears_when_not_running(kanban_home):
    """_rule_stuck_pid_liveness does NOT fire when the task is no longer running."""
    now = int(time.time())
    task = _task_dict(status="done")
    events = [
        _event_dict("stuck_pid_liveness", ts=now - 60,
                    worker_pid=12345, reason="termination_failed_pid_alive"),
    ]
    diags = kd.compute_task_diagnostics(task, events, [], now=now)
    assert not any(d.kind == "stuck_pid_liveness" for d in diags)


# ---------------------------------------------------------------------------
# 5. ttl_extensions_without_heartbeat diagnostic
# ---------------------------------------------------------------------------

def test_diagnostic_ttl_extensions_without_heartbeat_fires(kanban_home):
    """_rule_ttl_extensions_without_heartbeat fires warning after N
    consecutive claim_extended events with no intervening heartbeat."""
    now = int(time.time())
    task = _task_dict(status="running")
    events = [
        _event_dict("claim_extended", ts=now - 300,
                    reason="pid_alive", worker_pid=12345),
        _event_dict("claim_extended", ts=now - 200,
                    reason="pid_alive", worker_pid=12345),
        _event_dict("claim_extended", ts=now - 100,
                    reason="pid_alive", worker_pid=12345),
    ]
    config = {"ttl_extension_count_threshold": 3}
    diags = kd.compute_task_diagnostics(task, events, [], now=now, config=config)
    ext_diags = [d for d in diags if d.kind == "ttl_extensions_without_heartbeat"]
    assert len(ext_diags) == 1
    d = ext_diags[0]
    assert d.severity == "warning"
    assert d.data["consecutive_extensions"] == 3


def test_diagnostic_ttl_extensions_clears_on_heartbeat(kanban_home):
    """A heartbeat event between claim_extended events resets the streak."""
    now = int(time.time())
    task = _task_dict(status="running")
    events = [
        _event_dict("claim_extended", ts=now - 600,
                    reason="pid_alive", worker_pid=12345),
        _event_dict("claim_extended", ts=now - 500,
                    reason="pid_alive", worker_pid=12345),
        _event_dict("heartbeat", ts=now - 400, note="still working"),
        _event_dict("claim_extended", ts=now - 300,
                    reason="pid_alive", worker_pid=12345),
        _event_dict("claim_extended", ts=now - 200,
                    reason="pid_alive", worker_pid=12345),
    ]
    # threshold=3 → only 2 extensions since last heartbeat → should not fire
    config = {"ttl_extension_count_threshold": 3}
    diags = kd.compute_task_diagnostics(task, events, [], now=now, config=config)
    assert not any(d.kind == "ttl_extensions_without_heartbeat" for d in diags)


def test_diagnostic_ttl_extensions_below_threshold(kanban_home):
    """Fewer extensions than the threshold → no diagnostic."""
    now = int(time.time())
    task = _task_dict(status="running")
    events = [
        _event_dict("claim_extended", ts=now - 200,
                    reason="pid_alive", worker_pid=12345),
        _event_dict("claim_extended", ts=now - 100,
                    reason="pid_alive", worker_pid=12345),
    ]
    diags = kd.compute_task_diagnostics(task, events, [], now=now)
    assert not any(d.kind == "ttl_extensions_without_heartbeat" for d in diags)


def test_diagnostic_ttl_extensions_not_running(kanban_home):
    """Only fires for running tasks."""
    now = int(time.time())
    task = _task_dict(status="ready")
    events = [
        _event_dict("claim_extended", ts=now - i * 100, reason="pid_alive")
        for i in range(5)
    ]
    diags = kd.compute_task_diagnostics(task, events, [], now=now)
    assert not any(d.kind == "ttl_extensions_without_heartbeat" for d in diags)


# ---------------------------------------------------------------------------
# 6. Heartbeat tool — expired claim warning
# ---------------------------------------------------------------------------

def test_heartbeat_tool_expired_claim_returns_specific_error(kanban_home,
                                                              monkeypatch):
    """When heartbeat_claim returns False (claim expired), kanban_heartbeat
    returns a specific diagnostic message rather than the generic one."""
    from tools import kanban_tools as kt

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_demo00")
    monkeypatch.delenv("HERMES_KANBAN_CLAIM_LOCK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)

    # Simulate: heartbeat_claim → False (expired), heartbeat_worker → False
    def fake_connect(board=None):
        class FakeKB:
            def heartbeat_claim(self, *a, **kw):
                return False
            def heartbeat_worker(self, *a, **kw):
                return False
        class FakeConn:
            def close(self):
                pass
        return FakeKB(), FakeConn()

    monkeypatch.setattr(kt, "_connect", fake_connect)

    result = kt._handle_heartbeat({"task_id": "t_demo00"})
    assert '"ok": false' in result or '"ok":false' in result or "claim" in result.lower()
    # Must mention claim expiry, not just the generic error
    assert "expired" in result.lower() or "claim" in result.lower()


def test_heartbeat_tool_generic_error_when_not_claim_expiry(kanban_home,
                                                             monkeypatch):
    """When heartbeat_claim succeeds but heartbeat_worker fails, return the
    generic 'not running' error (not the expired-claim message)."""
    from tools import kanban_tools as kt

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_demo00")
    monkeypatch.delenv("HERMES_KANBAN_CLAIM_LOCK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)

    def fake_connect(board=None):
        class FakeKB:
            def heartbeat_claim(self, *a, **kw):
                return True  # claim is still live
            def heartbeat_worker(self, *a, **kw):
                return False  # but worker update failed
        class FakeConn:
            def close(self):
                pass
        return FakeKB(), FakeConn()

    monkeypatch.setattr(kt, "_connect", fake_connect)

    result = kt._handle_heartbeat({"task_id": "t_demo00"})
    assert "not running" in result or "unknown id" in result


# ---------------------------------------------------------------------------
# 7. Board/reality drift — integration: open run not closed when task done
# ---------------------------------------------------------------------------

def test_no_drift_after_complete(kanban_home):
    """After complete_task, there must be no orphaned open run (ended_at IS
    NULL) for the same task — verifies there is no board/reality drift."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="normal-complete", assignee="demo")
        kb.claim_task(conn, tid)
        kb.complete_task(conn, tid, summary="all done")

        task = kb.get_task(conn, tid)
        assert task.status == "done"
        assert task.current_run_id is None

        # Verify no open run exists
        open_runs = conn.execute(
            "SELECT id FROM task_runs WHERE task_id = ? AND ended_at IS NULL",
            (tid,),
        ).fetchall()
        assert not open_runs, f"orphaned open runs: {[r['id'] for r in open_runs]}"

    finally:
        conn.close()


def test_no_drift_after_block(kanban_home):
    """After block_task, the run must be closed with outcome='blocked'."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="blocked-task", assignee="demo")
        kb.claim_task(conn, tid)
        kb.block_task(conn, tid, reason="need more info")

        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        assert task.current_run_id is None

        open_runs = conn.execute(
            "SELECT id FROM task_runs WHERE task_id = ? AND ended_at IS NULL",
            (tid,),
        ).fetchall()
        assert not open_runs

        runs = kb.list_runs(conn, tid)
        assert any(r.outcome == "blocked" for r in runs)

    finally:
        conn.close()
