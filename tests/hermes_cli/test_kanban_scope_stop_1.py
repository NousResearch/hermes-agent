from __future__ import annotations

import hermes_cli.kanban_db as _owner_kanban_db

import hermes_cli.kanban_db_boards as _owner_kanban_db_boards

import hermes_cli.kanban_claims as _owner_kanban_claims
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
import pytest
from hermes_cli import kanban_db as kb
from gateway import kanban_watchers as _kw
from tests.hermes_cli.kanban_scope_support import (
    _stable_module_identity,
    _SYSTEMD_RUN_SHIM,
    _SYSTEMCTL_SHIM,
    _CHILD_WAIT_PROGRAM,
    _STUBBORN_CHILD_WAIT_PROGRAM,
    Shims,
    shims,
    kanban_home,
    conn,
    _make_task,
    _patch_systemd_available,
    _patch_systemd_run_binary,
    _fake_popen_capture,
    _write_kanban_config,
    _capture_worker_argv,
    _assert_plain_argv_shape,
    _scoped_task_row,
    _patch_managed_gateway,
    _fake_refused_launch_popen,
    _refused_launch_setup,
    _spawnable_profile,
    _running_row,
    _deferred_handoff_row,
    _breaker_shaped_row,
    _max_runtime_row,
    _timed_out_payload,
    _load_dashboard_plugin,
    _untracked_running_row,
    _PRE_CHANGE_TASKS_SQL,
    _PRE_CHANGE_TASK_RUNS_SQL,
)


def test_registration_race_mid_stop_keeps_task_running(shims, conn, monkeypatch):
    """F (new a), the race: the worker registers BETWEEN the sweep's
    snapshot and its verified stop. The write-lock re-check plus the CAS
    in the failure record keep the row running — no spawn_failed, no
    failure counted, the registration survives."""
    pid = shims.sleeper()
    unit = _kanban_worker_scope._kanban_worker_scope_unit("t_race", 1)
    shims.write_unit(unit, [pid])
    tid = _scoped_task_row(
        conn,
        scope=unit,
        pid=os.getpid(),
        started_delta=-_kanban_worker_scope.WORKER_REGISTRATION_GRACE_SECONDS - 300,
    )
    run_id = conn.execute(
        "SELECT current_run_id FROM tasks WHERE id = ?", (tid,)
    ).fetchone()["current_run_id"]

    # The worker's first heartbeat lands while the sweep is stopping the
    # scope — exactly the review's snapshot-vs-stop window.
    real_stop = _kanban_worker_stop.request_worker_scope_stop

    def register_then_stop(unit_name, *, task_id=None, **kwargs):
        _kanban_worker_identity.register_worker_pid(
            conn,
            task_id,
            expected_run_id=run_id,
            pid=pid,
        )
        return real_stop(unit_name, task_id=task_id, **kwargs)

    monkeypatch.setattr(
        _kanban_worker_stop, "request_worker_scope_stop", register_then_stop
    )

    assert _kanban_worker_recovery.fail_unregistered_workers(conn) == []
    row = conn.execute(
        "SELECT status, consecutive_failures, worker_pid, "
        "       worker_registered_at FROM tasks WHERE id = ?",
        (tid,),
    ).fetchone()
    assert row["status"] == "running"
    assert row["consecutive_failures"] == 0
    assert row["worker_pid"] == pid  # the registration was not overwritten
    assert row["worker_registered_at"] is not None
    run = _kanban_stats.latest_run(conn, tid)
    assert run.outcome is None  # the open run was never closed as failed


def test_queued_stop_cas_wins_registration_self_aborts(
    shims,
    conn,
    monkeypatch,
):
    """R, marker wins: the worker's first heartbeat lands AFTER the
    drain's re-check but BEFORE the signal — the exact window a plain
    read cannot close. The stop-pending CAS has already committed by
    then, so the registration self-aborts (no half-registered row) and
    the stop proceeds on the unregistered-launch verdict. The marker
    served its purpose for exactly the signal window: once the stop
    CONFIRMS (pass 8, AC) the service clears it, so a later re-adoption
    of the run can register again."""
    import threading

    monkeypatch.setattr(_kanban_worker_stop, "_scope_stop_inline", False)
    gate = threading.Event()

    pid = shims.sleeper()
    unit = _kanban_worker_scope._kanban_worker_scope_unit("t_casrace", 1)
    shims.write_unit(unit, [pid])
    tid = _scoped_task_row(
        conn,
        scope=unit,
        pid=os.getpid(),
        started_delta=-_kanban_worker_scope.WORKER_REGISTRATION_GRACE_SECONDS - 300,
    )
    run_id = conn.execute(
        "SELECT current_run_id FROM tasks WHERE id = ?", (tid,)
    ).fetchone()["current_run_id"]

    registration: dict = {}
    stops: list[str] = []

    def register_then_confirm(unit_name, **kwargs):
        # The heartbeat lands at the exact instant between the drain's
        # re-check and the signal. It runs on the service thread, so it
        # takes its own connection (sqlite connections are per-thread).
        stops.append(unit_name)
        with _kanban_db_connect.connect() as c:
            registration["ok"] = _kanban_worker_identity.register_worker_pid(
                c,
                tid,
                expected_run_id=run_id,
                pid=pid,
            )
        gate.set()
        return True

    monkeypatch.setattr(
        _kanban_worker_scope, "_stop_kanban_worker_scope", register_then_confirm
    )
    assert not _kanban_worker_stop.request_worker_scope_stop(
        unit,
        task_id=tid,
        skip_if_registered=True,
    )
    _kanban_worker_stop.join_scope_stop_service(timeout=5.0)

    assert gate.is_set()  # the stop (and thus the race) ran
    assert stops == [unit]
    assert registration["ok"] is False  # self-aborted on the marker
    row = conn.execute(
        "SELECT status, worker_pid, worker_registered_at FROM tasks WHERE id = ?",
        (tid,),
    ).fetchone()
    assert row["status"] == "running"
    assert row["worker_pid"] == os.getpid()  # launcher pid kept
    assert row["worker_registered_at"] is None  # no half-registration
    marked = conn.execute(
        "SELECT stop_pending FROM task_runs WHERE id = ?", (run_id,)
    ).fetchone()["stop_pending"]
    # Pass 8 (AC): the fake stop reported verified, so the confirmed
    # stop retired the marker it had just set — the signal window is
    # over, and the run row must not carry the marker past it.
    assert marked == 0
    _kanban_worker_stop.reset_scope_stop_service_for_tests()


def test_queued_stop_stands_down_when_registration_wins_cas(
    shims,
    conn,
    monkeypatch,
):
    """R, registration wins: it commits AFTER the drain's re-check read
    (the read missed it by a hair) but BEFORE the stop-pending CAS. The
    CAS excludes registered rows, matches nothing, and the stop stands
    down — the worker lives on despite the stale read."""
    monkeypatch.setattr(_kanban_worker_stop, "_scope_stop_inline", False)
    pid = shims.sleeper()
    unit = _kanban_worker_scope._kanban_worker_scope_unit("t_caslost", 1)
    shims.write_unit(unit, [pid])
    tid = _scoped_task_row(
        conn,
        scope=unit,
        pid=os.getpid(),
        started_delta=-_kanban_worker_scope.WORKER_REGISTRATION_GRACE_SECONDS - 300,
    )
    run_id = conn.execute(
        "SELECT current_run_id FROM tasks WHERE id = ?", (tid,)
    ).fetchone()["current_run_id"]

    # Registration commits first (its own connection, as the worker's
    # heartbeat would)…
    with _kanban_db_connect.connect() as c:
        assert _kanban_worker_identity.register_worker_pid(
            c,
            tid,
            expected_run_id=run_id,
            pid=pid,
        )
    # …but the drain's read re-check misses it — the race the CAS closes.
    monkeypatch.setattr(
        _kanban_worker_recovery, "_task_has_registered_worker", lambda _tid: False
    )

    stops: list[str] = []

    def never_stop(unit_name, **kwargs):
        stops.append(unit_name)
        return True

    monkeypatch.setattr(_kanban_worker_scope, "_stop_kanban_worker_scope", never_stop)
    assert not _kanban_worker_stop.request_worker_scope_stop(
        unit,
        task_id=tid,
        skip_if_registered=True,
    )
    _kanban_worker_stop.join_scope_stop_service(timeout=5.0)

    assert stops == []  # the CAS stood the stop down
    assert _kanban_db_dispatch._pid_alive(pid)
    row = conn.execute(
        "SELECT status, worker_registered_at FROM tasks WHERE id = ?",
        (tid,),
    ).fetchone()
    assert row["status"] == "running"
    assert row["worker_registered_at"] is not None
    marked = conn.execute(
        "SELECT stop_pending FROM task_runs WHERE id = ?", (run_id,)
    ).fetchone()["stop_pending"]
    assert marked is None  # registered rows never get marked
    _kanban_worker_stop.reset_scope_stop_service_for_tests()


def test_verified_stop_escalates_to_sigkill_on_stop_timeout(shims):
    """A SIGTERM-immune descendant (the stop "times out" server-side)
    forces the SIGKILL escalation, and the stop is only confirmed once
    the cgroup is actually empty."""
    from tools import process_registry_scope as pr

    stubborn = shims.stubborn_sleeper()
    unit = _kanban_worker_scope._kanban_worker_scope_unit("t_stubborn", 1)
    shims.write_unit(unit, [stubborn])

    assert pr._stop_systemd_unit_verified(unit) is True
    actions = [s["action"] for s in shims.stops() if s["unit"] == unit]
    assert actions[0] == "stop" and "kill" in actions
    assert shims.wait_for(lambda: not _kanban_db_dispatch._pid_alive(stubborn)), (
        "SIGKILL escalation must reap the SIGTERM-immune descendant"
    )
    assert not shims.cgroup_pids(unit)


def test_verified_stop_cancel_event_aborts_mid_stop(shims, monkeypatch):
    """Y: a cancel event reaches INSIDE an in-flight verified stop —
    after the TERM wait, before the SIGKILL wait, and before the final
    verify — instead of only between units. Every cancelled stop returns
    False ("still stopping") without paying for escalation the caller
    will never read."""
    import threading

    from tools import process_registry_scope as pr

    # --- 1. cancelled after the TERM wait: no SIGKILL, no verify -----
    pid = shims.sleeper()
    unit = _kanban_worker_scope._kanban_worker_scope_unit("t_cxl1", 1)
    shims.write_unit(unit, [pid])
    cancel = threading.Event()

    def slow_term(u, **kwargs):
        # The stop job is slow; the caller's budget dies mid-wait.
        # (**kwargs: the verified stop threads its cancel plumbing
        # into the stop call itself since pass 9, AH.)
        cancel.set()
        return True

    # A private MonkeyPatch: undoing the shared fixture-level one would
    # take the shim PATH down with it.
    mp1 = pytest.MonkeyPatch()
    mp1.setattr(pr, "_stop_systemd_unit", slow_term)
    assert pr._stop_systemd_unit_verified(unit, cancel_event=cancel) is False
    actions = [s["action"] for s in shims.stops() if s["unit"] == unit]
    assert actions == [], "cancelled stop must not escalate to SIGKILL"
    assert _kanban_db_dispatch._pid_alive(pid), (
        "nothing was signalled — cgroup still live"
    )
    mp1.undo()

    # --- 2. cancelled before the SIGKILL wait ------------------------
    stubborn = shims.stubborn_sleeper()  # ignores SIGTERM
    unit2 = _kanban_worker_scope._kanban_worker_scope_unit("t_cxl2", 1)
    shims.write_unit(unit2, [stubborn])
    cancel2 = threading.Event()
    real_liveness = pr._scope_unit_liveness
    calls = {"n": 0}

    def liveness_that_cancels(u):
        state = real_liveness(u)
        calls["n"] += 1
        if state == "alive":
            cancel2.set()  # budget dies between TERM and KILL
        return state

    mp2 = pytest.MonkeyPatch()
    mp2.setattr(pr, "_scope_unit_liveness", liveness_that_cancels)
    assert pr._stop_systemd_unit_verified(unit2, cancel_event=cancel2) is False
    actions2 = [s["action"] for s in shims.stops() if s["unit"] == unit2]
    assert actions2 == ["stop"], "no SIGKILL after the pre-KILL cancel"
    assert _kanban_db_dispatch._pid_alive(stubborn)
    mp2.undo()

    # --- 3. cancelled before the final verify ------------------------
    stubborn2 = shims.stubborn_sleeper()
    unit3 = _kanban_worker_scope._kanban_worker_scope_unit("t_cxl3", 1)
    shims.write_unit(unit3, [stubborn2])
    cancel3 = threading.Event()
    real_popen = pr.subprocess.Popen

    def popen_then_cancel(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        cmd = args[0] if args else kwargs.get("args")
        if cmd and "kill" in cmd:
            # Budget dies the instant the SIGKILL is on the wire: the
            # next verify probe belongs to a caller that moved on.
            # (Pass 9, AH: the escalation now runs through the
            # cancellable Popen helper, not subprocess.run.)
            cancel3.set()
        return proc

    mp3 = pytest.MonkeyPatch()
    mp3.setattr(pr.subprocess, "Popen", popen_then_cancel)
    assert pr._stop_systemd_unit_verified(unit3, cancel_event=cancel3) is False
    # The escalation DID run (cancel came too late to spare it), but the
    # verdict is still "still stopping": a cancelled caller must re-read
    # state, and the next stop confirms the already-empty cgroup for free.
    actions3 = [s["action"] for s in shims.stops() if s["unit"] == unit3]
    assert "kill" in actions3
    assert shims.wait_for(lambda: not _kanban_db_dispatch._pid_alive(stubborn2))


def test_verified_stop_deadline_aborts_like_cancel_event(shims, monkeypatch):
    """Y: a monotonic deadline is the event's twin — a stop whose
    deadline already passed never even fires the TERM."""
    from tools import process_registry_scope as pr

    pid = shims.sleeper()
    unit = _kanban_worker_scope._kanban_worker_scope_unit("t_dx", 1)
    shims.write_unit(unit, [pid])
    fired: list[str] = []

    def no_term(u, **kwargs):
        fired.append("term")
        return True

    monkeypatch.setattr(pr, "_stop_systemd_unit", no_term)
    assert (
        pr._stop_systemd_unit_verified(
            unit,
            deadline=time.monotonic() - 1.0,
        )
        is False
    )
    # The TERM was faked (never reached the shim); the point is the
    # verdict — False even though nothing was wrong with the unit —
    # and that no escalation followed.
    actions = [s["action"] for s in shims.stops() if s["unit"] == unit]
    assert actions == []
    assert _kanban_db_dispatch._pid_alive(pid)


def test_verified_stop_cancel_kills_slow_stop_helper(shims, monkeypatch):
    """Pass 9 (AH): the cancel event must reach INSIDE the blocking
    systemctl stop client itself. ``subprocess.run(..., timeout=15)``
    could not observe cancellation, so a shutdown past its budget kept
    the wedged stop client (and then the SIGKILL client) signalling
    after the dispatcher lock was released. The Popen+poll helper is
    killed mid-flight instead, and the stop reports still-stopping
    without paying the client timeout or escalating."""
    import threading

    from tools import process_registry_scope as pr

    stubborn = shims.stubborn_sleeper()
    unit = _kanban_worker_scope._kanban_worker_scope_unit("t_slowcxl", 1)
    shims.write_unit(unit, [stubborn])
    shims.arm_slow_op(unit, "stop", seconds=30)
    cancel = threading.Event()

    # Cancel from another thread only once the helper is PROVABLY
    # mid-hang (its pidfile exists) — the exact window the old blocking
    # call could not escape, without racing the shim's interpreter boot.
    def cancel_when_deep():
        if shims.wait_for(
            lambda: shims.slow_op_pid(unit, "stop") is not None,
            timeout=8.0,
        ):
            time.sleep(0.1)  # squarely inside the 30 s hang
            cancel.set()

    watcher = threading.Thread(target=cancel_when_deep, daemon=True)
    watcher.start()
    started = time.monotonic()
    assert (
        pr._stop_systemd_unit_verified(
            unit,
            cancel_event=cancel,
        )
        is False
    )
    elapsed = time.monotonic() - started
    assert elapsed < 10.0, "cancelled stop must not pay the 15 s timeout"
    helper_pid = shims.slow_op_pid(unit, "stop")
    assert helper_pid is not None
    assert shims.wait_for(lambda: not _kanban_db_dispatch._pid_alive(helper_pid)), (
        "the in-flight systemctl stop helper must be killed on cancel"
    )
    actions = [s["action"] for s in shims.stops() if s["unit"] == unit]
    assert actions == ["stop"], "no SIGKILL escalation after a cancelled stop"
    assert _kanban_db_dispatch._pid_alive(stubborn), (
        "nothing was signalled past the cancel"
    )


def test_verified_stop_cancel_kills_slow_sigkill_helper(shims, monkeypatch):
    """Pass 9 (AH): the same cancellation covers the SIGKILL escalation
    client — the second uncancellable subprocess the finding named."""
    import threading

    from tools import process_registry_scope as pr

    stubborn = shims.stubborn_sleeper()  # survives TERM, forces escalation
    unit = _kanban_worker_scope._kanban_worker_scope_unit("t_slowkill", 1)
    shims.write_unit(unit, [stubborn])
    shims.arm_slow_op(unit, "kill", seconds=30)
    cancel = threading.Event()

    # The TERM wait completes (stubborn ignores it), liveness says
    # alive, the SIGKILL helper hangs — budget dies inside it. A
    # watcher thread cancels only once the kill helper is PROVABLY
    # mid-hang (its pidfile exists), without racing interpreter boot.
    def cancel_when_deep():
        if shims.wait_for(
            lambda: shims.slow_op_pid(unit, "kill") is not None,
            timeout=8.0,
        ):
            time.sleep(0.1)  # squarely inside the 30 s hang
            cancel.set()

    watcher = threading.Thread(target=cancel_when_deep, daemon=True)
    watcher.start()
    started = time.monotonic()
    assert pr._stop_systemd_unit_verified(unit, cancel_event=cancel) is False
    elapsed = time.monotonic() - started
    assert elapsed < 10.0, "cancelled escalation must not pay its timeout"
    helper_pid = shims.slow_op_pid(unit, "kill")
    assert helper_pid is not None
    assert shims.wait_for(lambda: not _kanban_db_dispatch._pid_alive(helper_pid)), (
        "the in-flight systemctl kill helper must be killed on cancel"
    )
    assert _kanban_db_dispatch._pid_alive(stubborn), "the cgroup was never drained"
    actions = [s["action"] for s in shims.stops() if s["unit"] == unit]
    assert actions == ["stop", "kill"]


def test_join_scope_stop_service_cancels_inflight_stop(shims, monkeypatch):
    """Pass 9 (AH): a service-thread stop already inside a slow systemctl
    could not observe shutdown cancellation — join_scope_stop_service
    only waited, so the old gateway kept signalling after its budget and
    after the dispatcher lock release. The join must now CANCEL the
    in-flight unit within its budget: the helper subprocess is killed,
    the unit is reported as leftover and requeued for re-adoption."""
    import threading

    monkeypatch.setattr(
        _kanban_worker_stop, "_scope_stop_inline", False
    )  # real service thread

    stubborn = shims.stubborn_sleeper()
    unit = _kanban_worker_scope._kanban_worker_scope_unit("t_joincxl", 1)
    shims.write_unit(unit, [stubborn])
    shims.arm_slow_op(unit, "stop", seconds=30)

    with _kanban_worker_stop._scope_stop_lock:
        _kanban_worker_stop._scope_stop_pending[unit] = (
            _kanban_worker_stop._ScopeStopRequest(unit=unit)
        )
    _kanban_worker_stop._ensure_scope_stop_thread()
    _kanban_worker_stop._scope_stop_wake.set()
    assert shims.wait_for(lambda: _kanban_worker_stop._scope_stop_inflight == unit)

    shutdown_cancel = threading.Event()
    started = time.monotonic()
    leftover = _kanban_worker_stop.join_scope_stop_service(
        timeout=1.0,
        cancel_event=shutdown_cancel,
    )
    elapsed = time.monotonic() - started

    assert elapsed < 10.0, "join must return within its budget"
    assert unit in leftover, "the cancelled unit is listed as leftover"
    assert shutdown_cancel.is_set(), "the caller's cancel event fired too"
    helper_pid = shims.slow_op_pid(unit, "stop")
    assert helper_pid is not None
    assert shims.wait_for(lambda: not _kanban_db_dispatch._pid_alive(helper_pid)), (
        "the join's cancel must kill the in-flight stop helper"
    )
    # The service stood down and handed the unit back to the queue.
    assert shims.wait_for(
        lambda: (
            _kanban_worker_stop._scope_stop_pending.get(unit) is not None
            and _kanban_worker_stop._scope_stop_inflight is None
        )
    )
    # Nothing was signalled past the cancel: no SIGKILL escalation.
    actions = [s["action"] for s in shims.stops() if s["unit"] == unit]
    assert actions == ["stop"]
    assert _kanban_db_dispatch._pid_alive(stubborn)


def test_scope_stop_service_unlatches_after_cancelled_join(
    shims,
    monkeypatch,
):
    """Pass 10 (AL): a timed-out join used to leave the service cancel
    event set forever — the thread stayed alive but every later drain
    returned immediately, so each subsequent in-process stop request
    queued behind a permanent latch. Cancellation must be per join/run:
    after the cancelled join, a NEWLY enqueued stop is serviced (stop →
    SIGKILL escalation → confirmed) by the same service thread."""
    monkeypatch.setattr(_kanban_worker_stop, "_scope_stop_inline", False)  # real thread

    # Unit 1: a slow stop the join's budget will cancel mid-flight.
    stubborn1 = shims.stubborn_sleeper()
    unit1 = _kanban_worker_scope._kanban_worker_scope_unit("t_latch1", 1)
    shims.write_unit(unit1, [stubborn1])
    shims.arm_slow_op(unit1, "stop", seconds=30)
    with _kanban_worker_stop._scope_stop_lock:
        _kanban_worker_stop._scope_stop_pending[unit1] = (
            _kanban_worker_stop._ScopeStopRequest(unit=unit1)
        )
    _kanban_worker_stop._ensure_scope_stop_thread()
    _kanban_worker_stop._scope_stop_wake.set()
    assert shims.wait_for(lambda: _kanban_worker_stop._scope_stop_inflight == unit1)

    leftover = _kanban_worker_stop.join_scope_stop_service(timeout=1.0)
    assert unit1 in leftover, "the cancelled unit is reported as leftover"
    # The cancelled drain requeued unit 1 and stood down.
    assert shims.wait_for(
        lambda: (
            _kanban_worker_stop._scope_stop_pending.get(unit1) is not None
            and _kanban_worker_stop._scope_stop_inflight is None
        )
    )
    # Disarm the slow op so the requeued unit 1 can drain quickly later.
    shims.arm_slow_op(unit1, "stop", seconds=0)

    # Unit 2: requested AFTER the cancelled join — the latch test. With
    # a per-run token the next drain serves it: TERM, SIGKILL
    # escalation, verified dead.
    stubborn2 = shims.stubborn_sleeper()
    unit2 = _kanban_worker_scope._kanban_worker_scope_unit("t_latch2", 1)
    shims.write_unit(unit2, [stubborn2])
    with _kanban_worker_stop._scope_stop_lock:
        _kanban_worker_stop._scope_stop_pending[unit2] = (
            _kanban_worker_stop._ScopeStopRequest(unit=unit2)
        )
    _kanban_worker_stop._scope_stop_wake.set()

    assert shims.wait_for(
        lambda: unit2 in _kanban_worker_stop._scope_stop_confirmed,
        timeout=15.0,
    ), "a stop enqueued after the cancelled join must still be serviced"
    actions2 = [s["action"] for s in shims.stops() if s["unit"] == unit2]
    assert actions2[:2] == ["stop", "kill"], (
        "the new stop ran the full verified sequence, SIGKILL included"
    )
    assert shims.wait_for(lambda: not _kanban_db_dispatch._pid_alive(stubborn2))

    # The queue fully drains afterwards (unit 1's requeue included).
    assert _kanban_worker_stop.join_scope_stop_service(timeout=15.0) == []


def test_release_stale_claims_stops_worker_scope(shims, conn):
    """TTL-expired reclaim of a scoped worker stops the whole unit before
    the pid kill backstop, and clears the scope bookkeeping."""
    host = kb._claimer_id().split(":", 1)[0]
    pid = shims.sleeper()
    unit = _kanban_worker_scope._kanban_worker_scope_unit("t_stale", 1)
    shims.write_unit(unit, [pid])
    tid = kb.create_task(conn, title="stale", assignee="w")
    _owner_kanban_claims.claim_task(conn, tid, claimer=f"{host}:4194304")
    conn.execute(
        "UPDATE tasks SET status='running', worker_pid=?, worker_scope=?, "
        "claim_expires=?, last_heartbeat_at=? WHERE id=?",
        (pid, unit, int(time.time()) - 60, int(time.time()) - 7200, tid),
    )
    conn.commit()

    reclaimed = _owner_kanban_claims.release_stale_claims(conn)
    assert reclaimed == 1
    assert any(s["unit"] == unit and s["action"] == "stop" for s in shims.stops())
    row = conn.execute(
        "SELECT status, worker_scope FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row["status"] != "running"
    assert row["worker_scope"] is None


def test_crash_cleanup_defers_until_scope_stop_verified(shims, conn):
    """Crash reclamation of a scoped run waits for the VERIFIED scope
    stop (Gate B review, crash-cleanup ordering): a deactivating unit
    makes the worker look dead (its pid is gone) but the stop cannot yet
    be confirmed — the claim is held, a ``scope_stopping`` event records
    the hold, and the requeue happens only on a later tick once the
    verified stop lands. Nothing is released beside an unconfirmed
    cgroup, so no duplicate worker can spawn."""
    straggler = shims.sleeper()  # live process inside the worker's cgroup
    launcher = subprocess.Popen(["true"])
    launcher.wait()  # the recorded worker pid: already gone
    unit = _kanban_worker_scope._kanban_worker_scope_unit("t_crashstop", 1)
    shims.write_unit(unit, [straggler])
    shims.arm_deactivating(unit)
    tid = kb.create_task(conn, title="crash stop", assignee="w")
    _owner_kanban_claims.claim_task(conn, tid, claimer=kb._claimer_id())
    now = int(time.time())
    conn.execute(
        "UPDATE tasks SET status='running', worker_pid=?, "
        "worker_pid_started_at=?, worker_registered_at=?, worker_scope=?, "
        "claim_expires=?, last_heartbeat_at=? WHERE id=?",
        (
            launcher.pid,
            _kanban_worker_identity._worker_pid_start_time(launcher.pid),
            now,
            unit,
            now,
            now,
            tid,
        ),
    )
    conn.execute("UPDATE tasks SET started_at = started_at - 9999 WHERE id=?", (tid,))
    conn.commit()

    # Tick 1: the worker pid is gone and deactivating is NOT "alive", so
    # the run classifies as dead — but the stop job is still draining, so
    # NOTHING is released and the crash is retried next tick.
    assert _kanban_worker_recovery.detect_crashed_workers(conn) == []
    row = conn.execute(
        "SELECT status, claim_lock, worker_scope FROM tasks WHERE id=?",
        (tid,),
    ).fetchone()
    assert row["status"] == "running"
    assert row["claim_lock"] == kb._claimer_id()
    assert row["worker_scope"] == unit
    stopping = conn.execute(
        "SELECT count(*) AS n FROM task_events WHERE task_id=? "
        "AND kind='scope_stopping'",
        (tid,),
    ).fetchone()
    assert stopping["n"] == 1  # the hold is auditable, once per run

    # Still exactly one marker if the stop keeps failing: ticks repeat
    # without flooding the timeline.
    assert _kanban_worker_recovery.detect_crashed_workers(conn) == []
    stopping = conn.execute(
        "SELECT count(*) AS n FROM task_events WHERE task_id=? "
        "AND kind='scope_stopping'",
        (tid,),
    ).fetchone()
    assert stopping["n"] == 1

    # Tick 3: the stop job completes — scope verified dead, so the crash
    # requeue goes through.
    shims.clear_deactivating(unit)
    assert _kanban_worker_recovery.detect_crashed_workers(conn) == [tid]
    row = conn.execute(
        "SELECT status, worker_scope FROM tasks WHERE id=?", (tid,)
    ).fetchone()
    assert row["status"] != "running"
    assert row["worker_scope"] is None
    # The verified stop killed the straggler the dead worker left behind.
    assert not _kanban_db_dispatch._pid_alive(straggler)


def test_stop_timeout_does_not_release_claim_then_completes(shims, conn):
    """A wedged stop (killproof scope) must NOT release the claim — that
    would spawn a duplicate beside a live worker. The reclaim defers and
    completes on the next tick once the stop can be confirmed."""
    pid = shims.sleeper()
    unit = _kanban_worker_scope._kanban_worker_scope_unit("t_wedged", 1)
    shims.write_unit(unit, [pid])
    shims.arm_killproof(unit)
    tid = kb.create_task(conn, title="wedged stop", assignee="w")
    _owner_kanban_claims.claim_task(conn, tid, claimer=kb._claimer_id())
    now = int(time.time())
    conn.execute(
        "UPDATE tasks SET status='running', worker_pid=?, worker_scope=?, "
        "worker_registered_at=?, claim_expires=?, last_heartbeat_at=? "
        "WHERE id=?",
        (pid, unit, now, now - 60, now - 7200, tid),
    )
    conn.commit()

    # Tick 1: stop refuses (still stopping) — claim held, task still running.
    assert _owner_kanban_claims.release_stale_claims(conn) == 0
    row = conn.execute(
        "SELECT status, claim_lock, worker_scope FROM tasks WHERE id = ?",
        (tid,),
    ).fetchone()
    assert row["status"] == "running"
    assert row["claim_lock"] == kb._claimer_id()
    assert row["worker_scope"] == unit
    event = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? "
        "AND kind = 'reclaim_deferred'",
        (tid,),
    ).fetchone()
    assert event is not None  # the hold is auditable

    # Tick 2 (after the defer grace expires): the stop completes — the
    # reclaim goes through. The defer extended claim_expires, so age it.
    shims.clear_killproof(unit)
    conn.execute(
        "UPDATE tasks SET claim_expires = ? WHERE id = ?",
        (int(time.time()) - 60, tid),
    )
    conn.commit()
    assert _owner_kanban_claims.release_stale_claims(conn) == 1
    row = conn.execute(
        "SELECT status, worker_scope FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row["status"] != "running"
    assert row["worker_scope"] is None


def test_reclaim_task_stops_worker_scope(shims, conn):
    """Operator reclaim of a scoped worker stops its scope too."""
    pid = shims.sleeper()
    unit = _kanban_worker_scope._kanban_worker_scope_unit("t_manual", 1)
    shims.write_unit(unit, [pid])
    tid = kb.create_task(conn, title="manual", assignee="w")
    _owner_kanban_claims.claim_task(conn, tid, claimer=kb._claimer_id())
    conn.execute(
        "UPDATE tasks SET status='running', worker_pid=?, worker_scope=? WHERE id=?",
        (pid, unit, tid),
    )
    conn.commit()

    assert _owner_kanban_claims.reclaim_task(conn, tid, reason="operator abort") is True
    assert any(s["unit"] == unit for s in shims.stops())
    row = conn.execute(
        "SELECT status, worker_scope FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row["worker_scope"] is None


def test_archive_and_schedule_stop_scope(shims, conn):
    """archive_task / schedule_task stop a scoped worker's scope and clear
    the bookkeeping — the finding-5 surfaces."""
    pid = shims.sleeper()
    unit_a = _kanban_worker_scope._kanban_worker_scope_unit("t_arch", 1)
    shims.write_unit(unit_a, [pid])
    tid_a = kb.create_task(conn, title="to archive", assignee="w")
    _owner_kanban_claims.claim_task(conn, tid_a, claimer=kb._claimer_id())
    conn.execute(
        "UPDATE tasks SET status='running', worker_pid=?, worker_scope=? WHERE id=?",
        (pid, unit_a, tid_a),
    )
    conn.commit()

    assert _kanban_transitions.archive_task(conn, tid_a) is True
    assert any(s["unit"] == unit_a for s in shims.stops())
    row = conn.execute(
        "SELECT worker_scope FROM tasks WHERE id = ?", (tid_a,)
    ).fetchone()
    assert row["worker_scope"] is None

    pid_s = shims.sleeper()
    unit_s = _kanban_worker_scope._kanban_worker_scope_unit("t_sched", 1)
    shims.write_unit(unit_s, [pid_s])
    tid_s = kb.create_task(conn, title="to schedule", assignee="w")
    _owner_kanban_claims.claim_task(conn, tid_s, claimer=kb._claimer_id())
    conn.execute(
        "UPDATE tasks SET status='running', worker_pid=?, worker_scope=? WHERE id=?",
        (pid_s, unit_s, tid_s),
    )
    conn.commit()

    assert _kanban_transitions.schedule_task(conn, tid_s, reason="later") is True
    assert any(s["unit"] == unit_s for s in shims.stops())
    row = conn.execute(
        "SELECT worker_scope FROM tasks WHERE id = ?", (tid_s,)
    ).fetchone()
    assert row["worker_scope"] is None


def test_invalidate_descendants_stops_scope(shims, conn):
    """Ancestor reopen invalidation stops a running scoped descendant."""
    pid = shims.sleeper()
    unit = _kanban_worker_scope._kanban_worker_scope_unit("t_inval", 1)
    shims.write_unit(unit, [pid])
    parent = kb.create_task(conn, title="ancestor", assignee="planner")
    assert _kanban_completion.complete_task(conn, parent)
    child = kb.create_task(
        conn,
        title="running child",
        assignee="builder",
        parents=[parent],
    )
    _owner_kanban_claims.claim_task(conn, child, claimer=kb._claimer_id())
    conn.execute(
        "UPDATE tasks SET status='running', worker_pid=?, worker_scope=? WHERE id=?",
        (pid, unit, child),
    )
    conn.commit()

    with _kanban_db_connect.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status = 'todo', completed_at = NULL WHERE id = ?",
            (parent,),
        )
    result = _kanban_transitions.invalidate_descendants_for_parent_reopen(
        conn,
        parent,
        author="operator",
    )
    # Scoped descendants are verified stopped BEFORE the demotion — no
    # post-commit termination tuple exists for an already-empty cgroup
    # (E: a spawnable 'todo' never lands beside a live scope).
    assert result["terminations"] == []
    assert any(s["unit"] == unit for s in shims.stops())
    assert shims.wait_for(lambda: not _kanban_db_dispatch._pid_alive(pid))
    row = conn.execute(
        "SELECT status, worker_scope FROM tasks WHERE id = ?", (child,)
    ).fetchone()
    assert row["status"] == "todo"
    assert row["worker_scope"] is None


def test_completion_stops_scope_and_reaps_leaked_descendant(shims, conn, kanban_home):
    """Normal completion must not leak the worker's descendants: the
    worker-side stop is detached (it cannot wait on its own teardown), the
    audit sweep does the verified kill — and a stubborn descendant that
    ignores SIGTERM dies to the SIGKILL escalation."""
    leaked = shims.stubborn_sleeper()
    _spawnable_profile(kanban_home)
    tid = kb.create_task(conn, title="done soon", assignee="elias")
    _kanban_db_dispatch.dispatch_once(conn, dry_run=False)

    row = conn.execute(
        "SELECT worker_pid, worker_scope, current_run_id FROM tasks WHERE id = ?",
        (tid,),
    ).fetchone()
    unit = row["worker_scope"]
    worker_pid = shims.unit_json(unit)["pids"][0]
    assert _kanban_worker_identity.register_worker_pid(
        conn,
        tid,
        expected_run_id=row["current_run_id"],
        pid=worker_pid,
    )
    # A descendant the worker "spawned" that outlives it in the cgroup.
    shims.arm_sticky(unit, leaked)

    assert _kanban_completion.complete_task(conn, tid, result="done") is True

    # The detached stop fired (worker-side terminal path).
    assert shims.wait_for(lambda: any(s["unit"] == unit for s in shims.stops())), (
        "complete_task never attempted to stop the scope"
    )
    # The task row no longer claims the unit, so the audit sweep reaps it —
    # and the VERIFIED stop escalates: the SIGTERM-immune descendant dies to
    # the SIGKILL pass instead of outliving the task.
    _kanban_worker_recovery.reap_orphaned_worker_scopes(conn)
    assert shims.wait_for(
        lambda: not _kanban_db_dispatch._pid_alive(leaked), timeout=8.0
    ), "leaked descendant survived the SIGKILL escalation"


def test_dashboard_direct_running_to_ready_refuses_unverified_stop(
    shims,
    conn,
    kanban_home,
):
    """E: with the scope stop unconfirmed (stop job wedged), the drag
    does NOT flip the task to spawnable 'ready' — the row stays running
    with its claim held, a scope_stopping marker records why, and the
    refusal names the reason. Once the stop can confirm, the retry
    lands."""
    mod = _load_dashboard_plugin()
    pid = shims.sleeper()
    unit = _kanban_worker_scope._kanban_worker_scope_unit("t_dash_refuse", 1)
    shims.write_unit(unit, [pid])
    shims.arm_deactivating(unit)
    shims.arm_killproof(unit)
    tid = _scoped_task_row(conn, scope=unit, pid=pid, registered=True)

    with pytest.raises(mod._StatusTransitionRefused):
        mod._set_status_direct(conn, tid, "ready")
    row = conn.execute(
        "SELECT status, claim_lock, worker_scope FROM tasks WHERE id=?",
        (tid,),
    ).fetchone()
    assert row["status"] == "running"  # non-spawnable: no duplicate window
    assert row["claim_lock"]  # claim held
    assert row["worker_scope"] == unit
    kinds = [
        r["kind"]
        for r in conn.execute(
            "SELECT kind FROM task_events WHERE task_id=? ORDER BY id",
            (tid,),
        ).fetchall()
    ]
    assert "scope_stopping" in kinds
    assert "reclaim_deferred" in kinds
    # The pid-kill backstop may well have killed the worker directly —
    # that is fine; what must NOT happen is confirming the SCOPE or
    # releasing the row while its cgroup state is unknown.
    assert any(
        a["action"] in {"stop", "kill"} and a["unit"] == unit for a in shims.stops()
    )

    # Unwedged: the retry verified-stops first, then flips the status.
    shims.clear_deactivating(unit)
    shims.clear_killproof(unit)
    assert mod._set_status_direct(conn, tid, "ready") is True
    row = conn.execute(
        "SELECT status, worker_pid, worker_scope FROM tasks WHERE id=?",
        (tid,),
    ).fetchone()
    assert row["status"] == "ready"
    assert row["worker_pid"] is None
    assert row["worker_scope"] is None
    assert shims.wait_for(lambda: not _kanban_db_dispatch._pid_alive(pid))
    assert shims.cgroup_pids(unit) == []


def test_stop_all_scoped_workers_is_host_local(shims, conn):
    """The shutdown policy stops every scoped worker THIS host claims and
    leaves other hosts' workers to their own gateways."""
    mine_pid = shims.sleeper()
    mine_unit = _kanban_worker_scope._kanban_worker_scope_unit("t_mine", 1)
    shims.write_unit(mine_unit, [mine_pid])
    _scoped_task_row(conn, scope=mine_unit, pid=mine_pid)

    theirs_pid = shims.sleeper()
    theirs_unit = _kanban_worker_scope._kanban_worker_scope_unit("t_theirs", 1)
    shims.write_unit(theirs_unit, [theirs_pid])
    _scoped_task_row(
        conn,
        scope=theirs_unit,
        pid=theirs_pid,
        claimer="otherhost:99",
    )

    stopped = _kanban_worker_recovery.stop_all_scoped_workers(conn)
    assert stopped == [mine_unit]
    assert shims.wait_for(lambda: not _kanban_db_dispatch._pid_alive(mine_pid))
    assert _kanban_db_dispatch._pid_alive(theirs_pid)


def test_shutdown_policy_knob_runs_on_watcher_exit(shims, conn, kanban_home):
    """The gateway dispatcher watcher honours
    ``kanban.worker_isolation_stop_on_shutdown`` on graceful exit: knob true
    → workers stopped; default (unset) → they keep running for re-adoption."""
    import asyncio

    from gateway.kanban_watchers import GatewayKanbanWatchersMixin

    _write_kanban_config(
        Path(kanban_home), "  worker_isolation_stop_on_shutdown: true\n"
    )
    _kanban_db_connect._INITIALIZED_PATHS.clear()
    pid = shims.sleeper()
    unit = _kanban_worker_scope._kanban_worker_scope_unit("t_policy", 1)
    shims.write_unit(unit, [pid])
    _scoped_task_row(conn, scope=unit, pid=pid)

    class Harness(GatewayKanbanWatchersMixin):
        def __init__(self):
            self._running = False
            self._kanban_dispatcher_lock_handle = None

    asyncio.run(Harness()._kanban_dispatcher_watcher())
    assert shims.wait_for(lambda: not _kanban_db_dispatch._pid_alive(pid), timeout=8.0)


def test_shutdown_waits_for_cleanup_before_releasing_lock(
    shims,
    conn,
    kanban_home,
    monkeypatch,
):
    """H (new g): with the knob on, the dispatcher lock is not released
    while the scoped-worker cleanup is still running inside its budget —
    the watcher waits for the cleanup thread, then releases."""
    import asyncio
    import threading

    import gateway.kanban_watchers as kw
    from gateway.kanban_watchers import GatewayKanbanWatchersMixin

    _write_kanban_config(
        Path(kanban_home), "  worker_isolation_stop_on_shutdown: true\n"
    )
    _kanban_db_connect._INITIALIZED_PATHS.clear()
    pid = shims.sleeper()
    unit = _kanban_worker_scope._kanban_worker_scope_unit("t_hwait", 1)
    shims.write_unit(unit, [pid])
    _scoped_task_row(conn, scope=unit, pid=pid)

    # Shrink the budget knobs so the test is fast, but keep the scaled
    # math: base 0.3s + 1 unit x (0.3s bound + 2s drain margin).
    monkeypatch.setattr(_shutdown, "_SHUTDOWN_STOP_BASE_SECONDS", 0.3)
    monkeypatch.setattr(
        "tools.process_registry_scope.SCOPE_STOP_VERIFY_BOUND_SECONDS",
        0.3,
    )

    cleanup_started = threading.Event()
    gate = threading.Event()
    released = threading.Event()
    released_before_gate = []

    real_stop = _kanban_worker_recovery.stop_all_scoped_workers

    def gated_stop(c, should_abort=None, **kwargs):
        cleanup_started.set()
        gate.wait(timeout=10.0)
        return real_stop(c)

    monkeypatch.setattr(_kanban_worker_recovery, "stop_all_scoped_workers", gated_stop)

    class Harness(GatewayKanbanWatchersMixin):
        def __init__(self):
            self._running = False
            self._kanban_dispatcher_lock_handle = None

        def _release_kanban_dispatcher_lock(self) -> None:
            released_before_gate.append(gate.is_set())
            released.set()

    # The watcher on its own thread so this thread can observe the
    # lock NOT being released while cleanup is mid-flight.
    watcher_thread = threading.Thread(
        target=lambda: asyncio.run(Harness()._kanban_dispatcher_watcher()),
        daemon=True,
    )
    watcher_thread.start()

    # The watcher's own startup (lock + setup) takes a few seconds
    # before the graceful-exit path runs — the budget only starts once
    # the cleanup does.
    assert cleanup_started.wait(timeout=15.0)
    # Inside the budget, cleanup still running: bounded negative poll —
    # a correct watcher never releases here, so the poll simply runs
    # out; a buggy one (releasing without waiting) trips it. No blind
    # wall-clock sleep (item K).
    assert not shims.wait_for(released.is_set, timeout=0.5)
    gate.set()
    assert released.wait(timeout=5.0)
    assert released_before_gate == [True]  # released only AFTER cleanup
    watcher_thread.join(timeout=5.0)


def test_shutdown_budget_expiry_logs_leftovers_and_releases(
    shims,
    conn,
    kanban_home,
    monkeypatch,
    caplog,
):
    """H (new g), the bound: when cleanup exceeds its budget the lock IS
    released (shutdown must not hang) — but only after logging exactly
    which units were left stopping."""
    import asyncio
    import threading

    import gateway.kanban_watchers as kw
    from gateway.kanban_watchers import GatewayKanbanWatchersMixin

    _write_kanban_config(
        Path(kanban_home), "  worker_isolation_stop_on_shutdown: true\n"
    )
    _kanban_db_connect._INITIALIZED_PATHS.clear()
    pid = shims.sleeper()
    unit = _kanban_worker_scope._kanban_worker_scope_unit("t_hslow", 1)
    shims.write_unit(unit, [pid])
    _scoped_task_row(conn, scope=unit, pid=pid)

    monkeypatch.setattr(_shutdown, "_SHUTDOWN_STOP_BASE_SECONDS", 0.1)
    monkeypatch.setattr(
        "tools.process_registry_scope.SCOPE_STOP_VERIFY_BOUND_SECONDS",
        0.1,
    )

    gate = threading.Event()
    released = threading.Event()

    def wedged_stop(c, should_abort=None, **kwargs):
        gate.wait(timeout=10.0)  # "worse than any budget"
        return []

    monkeypatch.setattr(_kanban_worker_recovery, "stop_all_scoped_workers", wedged_stop)

    class Harness(GatewayKanbanWatchersMixin):
        def __init__(self):
            self._running = False
            self._kanban_dispatcher_lock_handle = None

        def _release_kanban_dispatcher_lock(self) -> None:
            released.set()

    import logging

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        asyncio.run(Harness()._kanban_dispatcher_watcher())
        gate.set()  # let the daemon thread finish for teardown

    assert released.wait(timeout=1.0) or released.is_set()
    warnings = [r for r in caplog.records if "still stopping" in r.message]
    assert warnings, "expected the leftover-units warning"
    assert unit in warnings[0].getMessage()


def test_join_scope_stop_service_reports_inflight_unit(
    shims,
    monkeypatch,
):
    """I: a unit mid verified-stop (popped off the queue, still being
    stopped on the service thread) is reported by a draining join — an
    in-flight stop is as "still stopping" as one still queued, and the
    old return (pending only) omitted it."""
    import threading

    gate = threading.Event()
    real_stop = _kanban_worker_scope._stop_kanban_worker_scope

    def wedged_stop(unit, **kwargs):
        # **kwargs: the service threads its cancel event into every
        # stop since pass 9 (AH).
        gate.wait(timeout=10.0)
        return real_stop(unit)

    monkeypatch.setattr(_kanban_worker_scope, "_stop_kanban_worker_scope", wedged_stop)
    monkeypatch.setattr(_kanban_worker_stop, "_scope_stop_inline", False)
    unit = _kanban_worker_scope._kanban_worker_scope_unit("t_inflight", 1)
    shims.write_unit(unit, [shims.sleeper()])
    try:
        assert not _kanban_worker_stop.request_worker_scope_stop(unit)
        leftover = _kanban_worker_stop.join_scope_stop_service(timeout=0.5)
        # Whichever side of the pop the service thread is on, the unit
        # must be reported: queued OR in flight.
        assert unit in leftover
    finally:
        gate.set()
        _kanban_worker_stop.join_scope_stop_service(timeout=5.0)
        _kanban_worker_stop.reset_scope_stop_service_for_tests()


def test_join_scope_stop_service_returns_fast_when_drained(shims):
    """I: "joined" means DRAINED, not thread-exit. The service thread is
    immortal (daemon for the process lifetime), so a plain Thread.join
    always burned the full timeout; an empty queue must return
    immediately."""
    t0 = time.monotonic()
    assert _kanban_worker_stop.join_scope_stop_service(timeout=5.0) == []
    assert time.monotonic() - t0 < 1.0


def test_scope_stop_intents_are_connection_local_not_thread_local(
    shims,
    conn,
    kanban_home,
    tmp_path,
    monkeypatch,
):
    """AA: the commit-conditional intent stack is keyed by connection, not
    thread. A shared ``check_same_thread=False`` connection that is mid
    transaction on one thread must COLLECT (not immediately queue) a stop
    requested from another thread — the thread-local stack let that
    request bypass the transaction and fire a kill the rollback could not
    recall. And a commit on one connection must never flush another
    connection's intents."""
    import sqlite3
    import threading

    monkeypatch.setattr(_kanban_worker_stop, "_scope_stop_inline", False)
    monkeypatch.setattr(_kanban_worker_stop, "_ensure_scope_stop_thread", lambda: None)
    # A service thread left alive by an EARLIER test is parked on the
    # module wake event; the flush below sets it, and the thread would
    # drain (and pop) the queued entry before the asserts look at it.
    # Parking that thread on the OLD event keeps this test the only
    # observer of the queue.
    monkeypatch.setattr(_kanban_worker_stop, "_scope_stop_wake", threading.Event())
    monkeypatch.setattr(
        _kanban_worker_scope, "_kanban_scope_state", lambda unit: "unknown"
    )
    _kanban_worker_stop.reset_scope_stop_service_for_tests()

    unit_shared = _kanban_worker_scope._kanban_worker_scope_unit("t_shared", 1)
    unit_a = _kanban_worker_scope._kanban_worker_scope_unit("t_a", 1)
    unit_b = _kanban_worker_scope._kanban_worker_scope_unit("t_b", 1)

    # Shared connection: thread 2 requests a stop while thread 1 holds the
    # transaction open.
    shared = sqlite3.connect(
        _owner_kanban_db.kanban_db_path(board="default"),
        timeout=5.0,
        check_same_thread=False,
    )
    # A second, independent database file so two write transactions can be
    # open at once on this thread.
    other = _kanban_db_connect.connect(db_path=tmp_path / "board2.db")
    try:
        requested = threading.Event()

        def request_from_other_thread():
            _kanban_worker_stop.request_worker_scope_stop(unit_shared, conn=shared)
            requested.set()

        with _kanban_db_connect.write_txn(shared):
            t = threading.Thread(target=request_from_other_thread)
            t.start()
            assert requested.wait(timeout=5.0)
            t.join(timeout=5.0)
            # Collected as an intent of THIS transaction — not queued.
            assert unit_shared not in _kanban_worker_stop._scope_stop_pending
        # The outermost commit flushed exactly that intent.
        assert unit_shared in _kanban_worker_stop._scope_stop_pending

        _kanban_worker_stop.reset_scope_stop_service_for_tests()
        with _kanban_db_connect.write_txn(conn):
            with _kanban_db_connect.write_txn(other):
                _kanban_worker_stop.request_worker_scope_stop(unit_a, conn=conn)
                _kanban_worker_stop.request_worker_scope_stop(unit_b, conn=other)
            # The inner (other) transaction committed first: only its
            # intent reached the queue; conn's stays collected.
            assert unit_b in _kanban_worker_stop._scope_stop_pending
            assert unit_a not in _kanban_worker_stop._scope_stop_pending
        # conn's outermost commit flushes its own intent.
        assert unit_a in _kanban_worker_stop._scope_stop_pending
    finally:
        shared.close()
        other.close()
        _kanban_worker_stop.reset_scope_stop_service_for_tests()


def test_committed_txn_flushes_stops_when_invariant_raises(conn, monkeypatch):
    """Z: the outermost intent level is popped and flushed BEFORE the
    post-commit file-length invariant check. An invariant exception used
    to leave committed DB state with no queued stop (the check raised
    first); the flush now runs first, so a committed transaction always
    queues its stops."""
    import sqlite3

    def boom(_conn):
        raise sqlite3.DatabaseError("torn-extend detected (test)")

    tid = kb.create_task(conn, title="invariant probe", assignee="w")
    monkeypatch.setattr(_kanban_db_connect, "_check_file_length_invariant", boom)
    monkeypatch.setattr(_kanban_worker_stop, "_scope_stop_inline", False)
    monkeypatch.setattr(_kanban_worker_stop, "_ensure_scope_stop_thread", lambda: None)
    # Park any lingering service thread on the OLD wake event (see the
    # connection-local intents test) — the flush must not let it drain
    # the queued entry before the assert reads it.
    import threading as _threading

    monkeypatch.setattr(_kanban_worker_stop, "_scope_stop_wake", _threading.Event())
    monkeypatch.setattr(
        _kanban_worker_scope, "_kanban_scope_state", lambda unit: "unknown"
    )
    _kanban_worker_stop.reset_scope_stop_service_for_tests()
    unit = _kanban_worker_scope._kanban_worker_scope_unit("t_inv", 3)
    try:
        with pytest.raises(sqlite3.DatabaseError):
            with _kanban_db_connect.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET title='flushed anyway' WHERE id=?",
                    (tid,),
                )
                assert not _kanban_worker_stop.request_worker_scope_stop(
                    unit, conn=conn
                )
                assert unit not in _kanban_worker_stop._scope_stop_pending
        # The transaction COMMITTED (the raise came after COMMIT) and its
        # stop reached the queue despite the invariant exception.
        assert (
            conn.execute(
                "SELECT title FROM tasks WHERE id=?",
                (tid,),
            ).fetchone()["title"]
            == "flushed anyway"
        )
        assert unit in _kanban_worker_stop._scope_stop_pending
    finally:
        _kanban_worker_stop.reset_scope_stop_service_for_tests()


def test_stop_pending_cleared_on_run_end_and_confirmed_stop(shims, conn):
    """AC: the stop-pending marker has clearing paths. ``_end_run`` wipes
    it when the run closes (a respawn's fresh registration must never see
    a stale marker), and the scope-stop service wipes it once a
    registration-sensitive stop CONFIRMS (a verified-empty cgroup means
    nothing is left to signal, so an adopted run must be registrable
    again)."""
    # Half 1 — run end: a marked run closes, the task respawns, the new
    # run registers cleanly.
    tid = kb.create_task(conn, title="ac respawn", assignee="w")
    claimed = _owner_kanban_claims.claim_task(conn, tid, claimer=kb._claimer_id())
    assert claimed is not None
    run_n = kb._current_run_id(conn, tid)
    with _kanban_db_connect.write_txn(conn):
        conn.execute(
            "UPDATE task_runs SET stop_pending=1 WHERE id=?",
            (run_n,),
        )
        kb._end_run(conn, tid, outcome="crashed", status="crashed")
        conn.execute(
            "UPDATE tasks SET status='ready', claim_lock=NULL, "
            "claim_expires=NULL, worker_pid=NULL, "
            "worker_pid_started_at=NULL, worker_registered_at=NULL, "
            "worker_scope=NULL WHERE id=?",
            (tid,),
        )
    assert (
        conn.execute(
            "SELECT stop_pending FROM task_runs WHERE id=?",
            (run_n,),
        ).fetchone()["stop_pending"]
        == 0
    )
    claimed2 = _owner_kanban_claims.claim_task(conn, tid, claimer=kb._claimer_id())
    assert claimed2 is not None
    run_m = kb._current_run_id(conn, tid)
    assert run_m != run_n
    assert _kanban_worker_identity.register_worker_pid(conn, tid, expected_run_id=run_m)

    # Half 2 — confirmed stop: the service marks the run right before
    # signalling, the stop verifies, the marker clears.
    pid = shims.sleeper()
    unit = _kanban_worker_scope._kanban_worker_scope_unit("t_ac", 1)
    shims.write_unit(unit, [pid])
    tid2 = _scoped_task_row(
        conn,
        scope=unit,
        pid=pid,
        started_delta=-3600,
    )
    run2 = kb._current_run_id(conn, tid2)
    with _kanban_db_connect.write_txn(conn):
        conn.execute(
            "UPDATE task_runs SET stop_pending=1 WHERE id=?",
            (run2,),
        )
    # The drain re-marks (CAS) then signals; verified True clears again.
    assert _kanban_worker_stop.request_worker_scope_stop(
        unit,
        task_id=tid2,
        skip_if_registered=True,
    )
    assert shims.wait_for(lambda: not _kanban_db_dispatch._pid_alive(pid))
    assert (
        conn.execute(
            "SELECT stop_pending FROM task_runs WHERE id=?",
            (run2,),
        ).fetchone()["stop_pending"]
        == 0
    )


def test_queue_coalescing_never_reenables_skipping(monkeypatch):
    """AB: two queued requests for one unit coalesce into one entry and
    ``skip_if_registered`` composes with AND. A terminal request (False —
    the completed task's scope must be reaped regardless of registration)
    makes the coalesced entry False in BOTH arrival orders: a later True
    can never re-enable skipping past a terminal stop."""
    monkeypatch.setattr(_kanban_worker_stop, "_scope_stop_inline", False)
    monkeypatch.setattr(_kanban_worker_stop, "_ensure_scope_stop_thread", lambda: None)
    # Park any lingering service thread on the OLD wake event (see the
    # connection-local intents test) so nothing drains the entries the
    # asserts inspect.
    import threading as _threading

    monkeypatch.setattr(_kanban_worker_stop, "_scope_stop_wake", _threading.Event())
    monkeypatch.setattr(
        _kanban_worker_scope, "_kanban_scope_state", lambda unit: "unknown"
    )
    _kanban_worker_stop.reset_scope_stop_service_for_tests()
    unit = _kanban_worker_scope._kanban_worker_scope_unit("t_coal", 2)
    try:
        # Registration-sensitive first, terminal second.
        assert not _kanban_worker_stop.request_worker_scope_stop(
            unit, skip_if_registered=True
        )
        assert not _kanban_worker_stop.request_worker_scope_stop(
            unit,
            task_id="t_coal",
            skip_if_registered=False,
        )
        assert _kanban_worker_stop._scope_stop_pending[unit].skip_if_registered is False
        _kanban_worker_stop.reset_scope_stop_service_for_tests()

        # Terminal first, registration-sensitive second.
        assert not _kanban_worker_stop.request_worker_scope_stop(
            unit,
            task_id="t_coal",
            skip_if_registered=False,
        )
        assert not _kanban_worker_stop.request_worker_scope_stop(
            unit, skip_if_registered=True
        )
        assert _kanban_worker_stop._scope_stop_pending[unit].skip_if_registered is False
    finally:
        _kanban_worker_stop.reset_scope_stop_service_for_tests()


def test_shutdown_single_budget_bounds_both_joins(
    shims,
    conn,
    kanban_home,
    monkeypatch,
    caplog,
):
    """I: base + N x per-unit is a CEILING on the whole stop. The old
    code joined the wedged worker for the full budget and THEN stacked a
    whole extra per-unit drain timeout for the service join — nearly
    double the stated budget. One deadline now bounds both joins."""
    import asyncio
    import logging
    import threading

    import gateway.kanban_watchers as kw
    from gateway.kanban_watchers import GatewayKanbanWatchersMixin

    _write_kanban_config(
        Path(kanban_home), "  worker_isolation_stop_on_shutdown: true\n"
    )
    _kanban_db_connect._INITIALIZED_PATHS.clear()
    pid = shims.sleeper()
    unit = _kanban_worker_scope._kanban_worker_scope_unit("t_hbudget", 1)
    shims.write_unit(unit, [pid])
    _scoped_task_row(conn, scope=unit, pid=pid)

    monkeypatch.setattr(_shutdown, "_SHUTDOWN_STOP_BASE_SECONDS", 0.1)
    monkeypatch.setattr(
        "tools.process_registry_scope.SCOPE_STOP_VERIFY_BOUND_SECONDS",
        0.1,
    )

    gate = threading.Event()
    timings: dict[str, float] = {}

    def wedged_stop(c, should_abort=None, **kwargs):
        timings["stop_entered"] = time.time()
        gate.wait(timeout=10.0)  # worse than any budget
        return []

    monkeypatch.setattr(_kanban_worker_recovery, "stop_all_scoped_workers", wedged_stop)

    class Harness(GatewayKanbanWatchersMixin):
        def __init__(self):
            self._running = False
            self._kanban_dispatcher_lock_handle = None

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        asyncio.run(Harness()._kanban_dispatcher_watcher())
        gate.set()  # let the daemon thread finish for teardown

    warnings = [r for r in caplog.records if "still stopping" in r.message]
    assert warnings, "expected the leftover-units warning"
    # budget = 0.1 base + 1 unit x (0.1 bound + 2.0 margin) = 2.2 s. The
    # old stacked joins cost ~budget + per-unit + margin = ~4.3 s; the
    # single deadline costs the budget alone.
    assert "stop_entered" in timings
    elapsed = warnings[0].created - timings["stop_entered"]
    assert elapsed <= 3.0, f"shutdown stop took {elapsed:.1f}s (budget 2.2s)"


def test_stop_all_scoped_workers_honors_abort_between_units(
    shims,
    conn,
    monkeypatch,
):
    """Q, unit half: the per-unit abort check stands the stop loop down
    BETWEEN units — a cancelled shutdown never proceeds to the next
    worker, and the units it did not reach are simply not reported
    stopped."""
    u1 = _kanban_worker_scope._kanban_worker_scope_unit("t_abort1", 1)
    u2 = _kanban_worker_scope._kanban_worker_scope_unit("t_abort2", 2)
    p1, p2 = shims.sleeper(), shims.sleeper()
    shims.write_unit(u1, [p1])
    shims.write_unit(u2, [p2])
    _scoped_task_row(conn, scope=u1, pid=p1)
    _scoped_task_row(conn, scope=u2, pid=p2)
    stopped_calls: list[str] = []
    monkeypatch.setattr(
        _kanban_worker_scope,
        "_stop_kanban_worker_scope",
        lambda unit, **kw: (stopped_calls.append(unit), True)[1],
    )

    def abort_after_first() -> bool:
        return len(stopped_calls) >= 1

    stopped = _kanban_worker_recovery.stop_all_scoped_workers(
        conn, should_abort=abort_after_first
    )

    assert stopped == [u1]
    assert stopped_calls == [u1]  # u2 was never attempted


from hermes_cli import kanban_boards as _kanban_boards
from hermes_cli import kanban_completion as _kanban_completion
from hermes_cli import kanban_db_connect as _kanban_db_connect
from hermes_cli import kanban_db_dispatch as _kanban_db_dispatch
from hermes_cli import kanban_stats as _kanban_stats
from hermes_cli import kanban_transitions as _kanban_transitions
from hermes_cli import kanban_worker_identity as _kanban_worker_identity
from hermes_cli import kanban_worker_recovery as _kanban_worker_recovery
from hermes_cli import kanban_worker_scope as _kanban_worker_scope
from hermes_cli import kanban_worker_stop as _kanban_worker_stop

from gateway import kanban_watchers_shutdown as _shutdown
