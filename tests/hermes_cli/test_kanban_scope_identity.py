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


def test_dispatch_records_launcher_pid_and_worker_self_registers(
    shims, conn, kanban_home
):
    """End-to-end with the shims: the dispatcher records the LAUNCHER pid
    and the run-suffixed scope; the unit's cgroup holds a DIFFERENT (worker)
    pid; ``register_worker_pid`` (what the heartbeat bridge calls from
    inside the worker) overwrites the launcher pid with the worker's and
    flips ``worker_registered_at`` with a ``worker_registered`` event."""
    _spawnable_profile(kanban_home)
    tid = kb.create_task(conn, title="scoped", assignee="elias")
    result = _kanban_db_dispatch.dispatch_once(conn, dry_run=False)

    assert [s[0] for s in result.spawned] == [tid]
    row = conn.execute(
        "SELECT worker_pid, worker_pid_started_at, worker_scope, "
        "       worker_registered_at, current_run_id FROM tasks WHERE id = ?",
        (tid,),
    ).fetchone()
    launcher_pid = row["worker_pid"]
    unit = row["worker_scope"]
    assert unit == _kanban_worker_scope._kanban_worker_scope_unit(
        tid, row["current_run_id"]
    )
    assert "-r" in unit  # run-suffixed: unique per attempt
    assert row["worker_registered_at"] is None  # starting, not registered

    # The launcher survived its probe window (no spawn failure recorded).
    assert result.late_spawn_failed == []

    # The unit's cgroup holds the worker pid — a different process.
    data = shims.unit_json(unit)
    assert data is not None
    worker_pids = [p for p in data["pids"] if p != launcher_pid]
    assert len(worker_pids) == 1
    worker_pid = worker_pids[0]
    assert worker_pid != launcher_pid
    assert _kanban_db_dispatch._pid_alive(worker_pid)

    # Worker-side self-registration (the heartbeat bridge's call).
    assert _kanban_worker_identity.register_worker_pid(
        conn,
        tid,
        expected_run_id=row["current_run_id"],
        pid=worker_pid,
    )
    after = conn.execute(
        "SELECT worker_pid, worker_registered_at FROM tasks WHERE id = ?",
        (tid,),
    ).fetchone()
    assert after["worker_pid"] == worker_pid
    assert after["worker_registered_at"] is not None
    event = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? "
        "AND kind = 'worker_registered'",
        (tid,),
    ).fetchone()
    assert event and json.loads(event["payload"])["pid"] == worker_pid


def test_heartbeat_bridge_registers_worker_pid(shims, conn, kanban_home, monkeypatch):
    """The auto-heartbeat bridge (run from INSIDE the worker process on
    first activity) registers the calling process's own pid."""
    from tools import kanban_tools as kt

    tid = kb.create_task(conn, title="bridge", assignee="w")
    _owner_kanban_claims.claim_task(conn, tid, claimer=kb._claimer_id())
    run_id = conn.execute(
        "SELECT current_run_id FROM tasks WHERE id = ?", (tid,)
    ).fetchone()["current_run_id"]

    db_path = _owner_kanban_db.kanban_db_path(board="default")
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kt._auto_heartbeat_last_attempt = 0.0
    assert kt.heartbeat_current_worker_from_env() is True

    row = conn.execute(
        "SELECT worker_pid, worker_pid_started_at, worker_registered_at, "
        "       last_heartbeat_at FROM tasks WHERE id = ?",
        (tid,),
    ).fetchone()
    assert row["worker_pid"] == os.getpid()  # THIS process = the worker
    assert row["worker_registered_at"] is not None
    assert row[
        "worker_pid_started_at"
    ] == _kanban_worker_identity._worker_pid_start_time(os.getpid())
    assert row["last_heartbeat_at"] is not None

    # The explicit tool path registers too (fresh task, direct call).
    tid2 = kb.create_task(conn, title="tool", assignee="w")
    _owner_kanban_claims.claim_task(conn, tid2, claimer=kb._claimer_id())
    run2 = conn.execute(
        "SELECT current_run_id FROM tasks WHERE id = ?", (tid2,)
    ).fetchone()["current_run_id"]
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid2)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run2))
    out = kt._handle_heartbeat({"task_id": tid2})
    assert '"ok"' in out or '"status"' in out
    row2 = conn.execute(
        "SELECT worker_pid, worker_registered_at FROM tasks WHERE id = ?",
        (tid2,),
    ).fetchone()
    assert row2["worker_pid"] == os.getpid()
    assert row2["worker_registered_at"] is not None


def test_adopt_surviving_worker_rewrites_claim_and_run_continues(conn):
    """A live, freshly-heartbeating worker owned by the previous gateway
    pid is re-adopted: claim moves to this claimer, run stays running, no
    failure counted, and crash detection leaves it alone."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="survivor", assignee="w")
    _owner_kanban_claims.claim_task(conn, tid, claimer=f"{host}:4194304")
    live = os.getpid()
    _running_row(
        conn,
        tid,
        claimer=f"{host}:4194304",
        pid=live,
        pid_started=_kanban_worker_identity._worker_pid_start_time(live),
        heartbeat=int(time.time()),
    )

    adopted = _kanban_worker_recovery.adopt_surviving_running_workers(conn)
    assert adopted == [tid]

    row = conn.execute(
        "SELECT status, claim_lock, claim_expires, consecutive_failures "
        "FROM tasks WHERE id = ?",
        (tid,),
    ).fetchone()
    assert row["status"] == "running"
    assert row["claim_lock"] == kb._claimer_id()
    assert row["claim_expires"] > int(time.time())
    assert row["consecutive_failures"] == 0

    event = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'adopted'",
        (tid,),
    ).fetchone()
    payload = json.loads(event["payload"])
    assert payload["previous_claimer"] == f"{host}:4194304"
    assert payload["claimer"] == kb._claimer_id()

    # The adopted run is not a crash: detection must skip it entirely.
    assert _kanban_worker_recovery.detect_crashed_workers(conn) == []
    # Idempotent: a second pass finds nothing to adopt.
    assert _kanban_worker_recovery.adopt_surviving_running_workers(conn) == []


def test_adoption_skips_stale_heartbeat(conn):
    """Alive pid but no observable progress for > 1h → NOT adopted; the
    existing stale paths own that case."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="wedged", assignee="w")
    _owner_kanban_claims.claim_task(conn, tid, claimer=f"{host}:4194304")
    live = os.getpid()
    _running_row(
        conn,
        tid,
        claimer=f"{host}:4194304",
        pid=live,
        pid_started=_kanban_worker_identity._worker_pid_start_time(live),
        heartbeat=int(time.time()) - 7200,
    )

    assert _kanban_worker_recovery.adopt_surviving_running_workers(conn) == []
    row = conn.execute("SELECT claim_lock FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert row["claim_lock"] == f"{host}:4194304"


def test_dead_pid_still_crashes_and_counts_failure(conn):
    """Adoption must not rescue a genuinely dead worker: crash detection
    still fires, marks the run crashed, and counts the failure."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="crashed", assignee="w")
    _owner_kanban_claims.claim_task(conn, tid, claimer=f"{host}:4194304")
    dead = subprocess.Popen(["true"])
    dead.wait()
    _running_row(
        conn,
        tid,
        claimer=f"{host}:4194304",
        pid=dead.pid,
        pid_started=None,
        heartbeat=int(time.time()),
    )
    conn.execute("UPDATE tasks SET started_at = started_at - 9999 WHERE id=?", (tid,))
    conn.commit()

    assert _kanban_worker_recovery.adopt_surviving_running_workers(conn) == []
    crashed = _kanban_worker_recovery.detect_crashed_workers(conn)
    assert tid in crashed
    row = conn.execute(
        "SELECT status, consecutive_failures FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row["status"] != "running"
    assert row["consecutive_failures"] >= 1


def test_recycled_pid_is_not_mistaken_for_the_worker(conn):
    """PID-reuse guard: a live but different process at the recorded pid is
    a dead worker, not a survivor — crash detection fires with the reuse
    flag, and adoption never claims it."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="reused", assignee="w")
    _owner_kanban_claims.claim_task(conn, tid, claimer=f"{host}:4194304")
    # A very much alive process (this test) whose start-time fingerprint
    # differs from the recorded one — exactly what pid reuse looks like.
    _running_row(
        conn,
        tid,
        claimer=f"{host}:4194304",
        pid=os.getpid(),
        pid_started=12345,
        heartbeat=int(time.time()),
    )
    conn.execute("UPDATE tasks SET started_at = started_at - 9999 WHERE id=?", (tid,))
    conn.commit()

    assert _kanban_worker_recovery.adopt_surviving_running_workers(conn) == []
    crashed = _kanban_worker_recovery.detect_crashed_workers(conn)
    assert tid in crashed

    event = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'crashed'",
        (tid,),
    ).fetchone()
    payload = json.loads(event["payload"])
    assert payload.get("pid_reused") is True


def test_unregistered_within_grace_is_left_alone(shims, conn):
    """A scoped run inside its launch grace window is 'starting', not dead —
    the sweep must not fail it."""
    pid = shims.sleeper()
    unit = _kanban_worker_scope._kanban_worker_scope_unit("t_grace", 1)
    shims.write_unit(unit, [pid])
    tid = _scoped_task_row(conn, scope=unit, pid=pid, started_delta=0)
    assert _kanban_worker_recovery.fail_unregistered_workers(conn) == []
    row = conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert row["status"] == "running"


def test_queued_stop_skips_when_worker_registers_first(
    shims,
    conn,
    monkeypatch,
):
    """J: the worker registers AFTER the sweep's pre-stop check but
    BEFORE the queued verified stop runs on the service thread. The stop
    re-checks registration immediately before acting and stands down —
    the CAS already prevented the spawn_failed record; without this the
    queued stop still killed the legitimate worker."""
    import threading

    monkeypatch.setattr(_kanban_worker_stop, "_scope_stop_inline", False)
    pid = shims.sleeper()
    unit = _kanban_worker_scope._kanban_worker_scope_unit("t_qrace", 1)
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

    # Hold the service on a decoy unit so the real request sits QUEUED
    # (not in flight) while the worker registers — exactly the finding's
    # window between the pre-stop check and the queued stop running.
    gate = threading.Event()
    stops: list[str] = []

    def gated_stop(unit_name, **kwargs):
        stops.append(unit_name)
        gate.wait(timeout=5.0)
        return True

    monkeypatch.setattr(_kanban_worker_scope, "_stop_kanban_worker_scope", gated_stop)
    decoy = _kanban_worker_scope._kanban_worker_scope_unit("t_decoy", 1)
    shims.write_unit(decoy, [shims.sleeper()])  # active, so it queues
    assert not _kanban_worker_stop.request_worker_scope_stop(
        decoy
    )  # queued, now in flight
    assert shims.wait_for(lambda: stops == [decoy])

    # The sweep: pre-stop check sees an unregistered row, queues its
    # stop behind the decoy, and (not confirmed this tick) fails nothing.
    assert _kanban_worker_recovery.fail_unregistered_workers(conn) == []

    # The worker's first heartbeat lands while the stop is still queued.
    _kanban_worker_identity.register_worker_pid(
        conn,
        tid,
        expected_run_id=run_id,
        pid=pid,
    )

    gate.set()  # release the decoy — the service reaches the real unit
    _kanban_worker_stop.join_scope_stop_service(timeout=5.0)

    # The queued stop stood down: the decoy was stopped, the real unit's
    # verified stop never ran, and the worker lives on.
    assert stops == [decoy]
    assert _kanban_db_dispatch._pid_alive(pid)
    row = conn.execute(
        "SELECT status, consecutive_failures, worker_registered_at "
        "FROM tasks WHERE id = ?",
        (tid,),
    ).fetchone()
    assert row["status"] == "running"
    assert row["consecutive_failures"] == 0
    assert row["worker_registered_at"] is not None
    # And the next sweep agrees: a registered row is not its business.
    assert _kanban_worker_recovery.fail_unregistered_workers(conn) == []
    _kanban_worker_stop.reset_scope_stop_service_for_tests()


def test_reregistration_same_pid_new_fingerprint_rejected(
    shims,
    conn,
    monkeypatch,
):
    """F (new a), the reused pid: a second registration presenting the
    SAME numeric pid but a DIFFERENT start fingerprint is the kernel
    having recycled the number — rejected with a log, and the recorded
    registration is left exactly as the real worker wrote it."""
    pid = shims.sleeper()
    unit = _kanban_worker_scope._kanban_worker_scope_unit("t_reuse", 1)
    shims.write_unit(unit, [pid])
    tid = _scoped_task_row(conn, scope=unit, pid=os.getpid())
    run_id = conn.execute(
        "SELECT current_run_id FROM tasks WHERE id = ?", (tid,)
    ).fetchone()["current_run_id"]

    fingerprints = iter([111, 222, 111])
    monkeypatch.setattr(
        _kanban_worker_identity,
        "_worker_pid_start_time",
        lambda _pid: next(fingerprints),
    )

    assert _kanban_worker_identity.register_worker_pid(
        conn, tid, expected_run_id=run_id, pid=pid
    )
    assert not _kanban_worker_identity.register_worker_pid(
        conn,
        tid,
        expected_run_id=run_id,
        pid=pid,
    )  # same pid number, new fingerprint — a recycled pid, not our worker
    row = conn.execute(
        "SELECT worker_pid_started_at FROM tasks WHERE id = ?",
        (tid,),
    ).fetchone()
    assert row["worker_pid_started_at"] == 111
    # A true re-registration (same pid, same fingerprint) stays accepted.
    assert _kanban_worker_identity.register_worker_pid(
        conn, tid, expected_run_id=run_id, pid=pid
    )


def test_release_stale_claims_reservation_cas_blocks_inflight_heartbeat(
    shims,
    conn,
    monkeypatch,
):
    """Pass 11 (AN): a heartbeat committing between the sweep's fresh
    re-read and the reservation UPDATE — the exact window AK's lock-free
    pre-check could not cover — must make the optimistic CAS miss and
    stand the row down for the tick: no signal, no reclaim, and the
    heartbeat's TTL (not a defer grace) owns the row afterwards."""
    worker = shims.sleeper()
    unit = _kanban_worker_scope._kanban_worker_scope_unit("t_anhb", 1)
    shims.write_unit(unit, [worker])
    tid = kb.create_task(conn, title="reservation cas", assignee="w")
    _owner_kanban_claims.claim_task(conn, tid, claimer=kb._claimer_id())
    now = int(time.time())
    conn.execute(
        "UPDATE tasks SET status='running', worker_pid=?, "
        "worker_pid_started_at=?, worker_registered_at=?, worker_scope=?, "
        "claim_expires=?, last_heartbeat_at=? WHERE id=?",
        (
            worker,
            _kanban_worker_identity._worker_pid_start_time(worker),
            now,
            unit,
            now - 60,
            now - 7200,
            tid,
        ),  # heartbeat past the 1h backstop
    )
    conn.commit()

    real_reread = _owner_kanban_claims._reread_stale_claim_for_reclaim

    def reread_then_inflight_heartbeat(conn_arg, task_id, claim_lock):
        row = real_reread(conn_arg, task_id, claim_lock)
        # The interleaving under test: the worker's heartbeat commits
        # between the sweep's re-read and its reservation CAS, i.e.
        # after the decision transaction opened but before its UPDATE.
        landed = int(time.time())
        conn_arg.execute(
            "UPDATE tasks SET claim_expires = ?, last_heartbeat_at = ? "
            "WHERE id = ? AND status = 'running'",
            (landed + kb.DEFAULT_CLAIM_TTL_SECONDS, landed, task_id),
        )
        return row

    monkeypatch.setattr(
        _owner_kanban_claims,
        "_reread_stale_claim_for_reclaim",
        reread_then_inflight_heartbeat,
    )
    signalled: list[tuple] = []

    def record_signal(pid, sig):
        signalled.append((pid, sig))

    assert _owner_kanban_claims.release_stale_claims(conn, signal_fn=record_signal) == 0

    row = conn.execute(
        "SELECT status, claim_lock, claim_expires FROM tasks WHERE id=?",
        (tid,),
    ).fetchone()
    assert row["status"] == "running"
    assert row["claim_lock"] == kb._claimer_id()
    # The heartbeat's TTL owns the row — not a defer grace.
    assert row["claim_expires"] > int(time.time()) + 600
    assert signalled == [], "a heartbeat that beat the CAS is never signalled"
    assert _kanban_db_dispatch._pid_alive(worker), "the live worker was never signalled"
    assert [s["action"] for s in shims.stops() if s["unit"] == unit] == []
    assert (
        conn.execute(
            "SELECT count(*) AS n FROM task_events WHERE task_id=? "
            "AND kind='reclaimed'",
            (tid,),
        ).fetchone()["n"]
        == 0
    )


def test_release_stale_claims_reservation_refuses_late_heartbeat_and_signals(
    shims,
    conn,
    monkeypatch,
):
    """Pass 12 (AP), worker side: a heartbeat committing AFTER the
    reclaim reservation commits (but before the signal) is refused as
    claim-lost — ``heartbeat_claim`` never extends a reserved row — so
    the re-check still sees the untouched reservation and the sweep
    signals a row that provably stopped heartbeating."""
    worker = shims.sleeper()
    unit = _kanban_worker_scope._kanban_worker_scope_unit("t_aphb", 1)
    shims.write_unit(unit, [worker])
    tid = kb.create_task(conn, title="reserved heartbeat", assignee="w")
    _owner_kanban_claims.claim_task(conn, tid, claimer=kb._claimer_id())
    now = int(time.time())
    conn.execute(
        "UPDATE tasks SET status='running', worker_pid=?, "
        "worker_pid_started_at=?, worker_registered_at=?, worker_scope=?, "
        "claim_expires=?, last_heartbeat_at=? WHERE id=?",
        (
            worker,
            _kanban_worker_identity._worker_pid_start_time(worker),
            now,
            unit,
            now - 60,
            now - 7200,
            tid,
        ),  # heartbeat past the 1h backstop
    )
    conn.commit()

    refused: list[bool] = []
    real_recheck = _owner_kanban_claims._recheck_reclaim_reservation

    def recheck_after_reservation(conn_arg, task_id, claim_lock, grace, fresh):
        # The interleaving under test: the worker's heartbeat lands
        # AFTER the reservation transaction committed (the marker is
        # already on the row) but BEFORE the re-check / signal.
        refused.append(
            _owner_kanban_claims.heartbeat_claim(conn_arg, task_id, claimer=claim_lock)
        )
        return real_recheck(conn_arg, task_id, claim_lock, grace, fresh)

    monkeypatch.setattr(
        _owner_kanban_claims,
        "_recheck_reclaim_reservation",
        recheck_after_reservation,
    )

    assert _owner_kanban_claims.release_stale_claims(conn) == 1

    # The heartbeat was refused — claim lost, never an extension.
    assert refused == [False]
    row = conn.execute(
        "SELECT status, claim_lock, claim_expires, reclaim_reserved_at, "
        "worker_scope FROM tasks WHERE id=?",
        (tid,),
    ).fetchone()
    # ... and because it was refused, the signal proceeded and the row
    # was reclaimed cleanly (marker cleared with the bookkeeping).
    assert row["status"] == "ready"
    assert row["claim_lock"] is None
    assert row["claim_expires"] is None
    assert row["reclaim_reserved_at"] is None
    assert row["worker_scope"] is None
    assert (
        conn.execute(
            "SELECT count(*) AS n FROM task_events WHERE task_id=? "
            "AND kind='reclaimed'",
            (tid,),
        ).fetchone()["n"]
        == 1
    )
    # The provably-not-heartbeating worker was stopped by the sweep.
    assert shims.wait_for(lambda: not _kanban_db_dispatch._pid_alive(worker))
    assert [s for s in shims.stops() if s["unit"] == unit]


def test_release_stale_claims_reservation_recheck_stands_down_on_raced_heartbeat(
    shims,
    conn,
    monkeypatch,
):
    """Pass 12 (AP), sweep side: a heartbeat that raced in before the
    re-check (an in-flight writer committing after the reservation —
    simulated by a direct DB write, as from an older worker binary) is
    never signalled: the re-check sees the row moved, releases the
    reservation to the live values, and the heartbeat's TTL owns the
    row."""
    worker = shims.sleeper()
    unit = _kanban_worker_scope._kanban_worker_scope_unit("t_apraced", 1)
    shims.write_unit(unit, [worker])
    tid = kb.create_task(conn, title="raced heartbeat", assignee="w")
    _owner_kanban_claims.claim_task(conn, tid, claimer=kb._claimer_id())
    now = int(time.time())
    conn.execute(
        "UPDATE tasks SET status='running', worker_pid=?, "
        "worker_pid_started_at=?, worker_registered_at=?, worker_scope=?, "
        "claim_expires=?, last_heartbeat_at=? WHERE id=?",
        (
            worker,
            _kanban_worker_identity._worker_pid_start_time(worker),
            now,
            unit,
            now - 60,
            now - 7200,
            tid,
        ),  # heartbeat past the 1h backstop
    )
    conn.commit()

    real_recheck = _owner_kanban_claims._recheck_reclaim_reservation

    def recheck_after_raced_heartbeat(conn_arg, task_id, claim_lock, grace, fresh):
        # The interleaving under test: a heartbeat that was already in
        # flight when the reservation committed lands now — between the
        # reservation transaction and the re-check — rewriting the claim
        # TTL and the heartbeat timestamp on the reserved row.
        landed = int(time.time())
        conn_arg.execute(
            "UPDATE tasks SET claim_expires = ?, last_heartbeat_at = ? "
            "WHERE id = ? AND status = 'running'",
            (landed + kb.DEFAULT_CLAIM_TTL_SECONDS, landed, task_id),
        )
        conn_arg.commit()
        return real_recheck(conn_arg, task_id, claim_lock, grace, fresh)

    monkeypatch.setattr(
        _owner_kanban_claims,
        "_recheck_reclaim_reservation",
        recheck_after_raced_heartbeat,
    )
    signalled: list[tuple] = []

    def record_signal(pid, sig):
        signalled.append((pid, sig))

    assert _owner_kanban_claims.release_stale_claims(conn, signal_fn=record_signal) == 0

    row = conn.execute(
        "SELECT status, claim_lock, claim_expires, reclaim_reserved_at "
        "FROM tasks WHERE id=?",
        (tid,),
    ).fetchone()
    assert row["status"] == "running"
    assert row["claim_lock"] == kb._claimer_id()
    # The heartbeat's TTL owns the row — the reservation was released to
    # the live values (marker dropped), not held over them.
    assert row["claim_expires"] > int(time.time()) + 600
    assert row["reclaim_reserved_at"] is None
    assert signalled == [], "a heartbeat that beat the re-check is never signalled"
    assert _kanban_db_dispatch._pid_alive(worker), "the live worker was never signalled"
    assert [s["action"] for s in shims.stops() if s["unit"] == unit] == []
    assert (
        conn.execute(
            "SELECT count(*) AS n FROM task_events WHERE task_id=? "
            "AND kind='reclaimed'",
            (tid,),
        ).fetchone()["n"]
        == 0
    )


def test_enforce_max_runtime_scoped_never_signals_the_pid(shims, conn):
    """E (finding 3, scoped half): a scoped max-runtime run is ended by a
    VERIFIED scope stop and the recorded pid is never signalled — once
    the cgroup is confirmed empty nothing of the run survives, and the
    pid number may already have been handed to an unrelated process."""
    pid = shims.sleeper()
    unit = _kanban_worker_scope._kanban_worker_scope_unit("t_maxrt_scoped", 1)
    shims.write_unit(unit, [pid])
    tid = _max_runtime_row(
        conn,
        pid,
        unit,
        pid_started_at=_kanban_worker_identity._worker_pid_start_time(pid),
    )
    signals: list[tuple[int, int]] = []

    def recording_kill(p, s):
        signals.append((p, s))

    assert _kanban_worker_recovery.enforce_max_runtime(
        conn, signal_fn=recording_kill
    ) == [tid]
    assert signals == []  # the unit was stopped; the pid itself untouched
    assert not _kanban_db_dispatch._pid_alive(pid)  # teardown verified, not assumed
    assert _timed_out_payload(conn, tid)["scope_stopped"] == unit


def test_enforce_max_runtime_never_signals_a_recycled_pid(shims, conn):
    """E (finding 3, unscoped half): when the recorded start fingerprint no
    longer matches the live pid (the worker died and the kernel reused its
    number), the run times out WITHOUT signalling — a bare liveness kill
    would murder an unrelated process that inherited the pid."""
    pid = shims.sleeper()  # stands in for the unrelated impostor process
    tid = _max_runtime_row(conn, pid, None, pid_started_at=111111)
    signals: list[tuple[int, int]] = []

    def recording_kill(p, s):
        signals.append((p, s))

    assert _kanban_worker_recovery.enforce_max_runtime(
        conn, signal_fn=recording_kill
    ) == [tid]
    assert signals == []
    assert _kanban_db_dispatch._pid_alive(pid)  # the impostor was never touched
    payload = _timed_out_payload(conn, tid)
    assert payload.get("pid_reused") is True


def test_enforce_max_runtime_legacy_row_with_predating_pid_not_signalled(
    shims, conn, monkeypatch
):
    """B (finding, legacy half): a row written before the fingerprint
    column existed has no pid identity to check — but a live process
    that was running BEFORE the run started cannot be that run's worker,
    so the pid must not be signalled; the run still times out."""
    pid = shims.sleeper()  # stands in for the unrelated impostor
    tid = _max_runtime_row(conn, pid, None, pid_started_at=None)
    started = conn.execute(
        "SELECT COALESCE(r.started_at, t.started_at) AS s "
        "FROM tasks t LEFT JOIN task_runs r ON r.id = t.current_run_id "
        "WHERE t.id = ?",
        (tid,),
    ).fetchone()["s"]
    # The live pid began long before the run row existed.
    monkeypatch.setattr(
        _kanban_worker_identity,
        "_worker_pid_epoch_start",
        lambda _pid: float(started) - 3600.0,
    )
    signals: list[tuple[int, int]] = []

    def recording_kill(p, s):
        signals.append((p, s))

    assert _kanban_worker_recovery.enforce_max_runtime(
        conn, signal_fn=recording_kill
    ) == [tid]
    assert signals == []
    assert _kanban_db_dispatch._pid_alive(pid)  # the impostor was never touched
    payload = _timed_out_payload(conn, tid)
    assert payload.get("pid_reused") is True


def test_reap_sweep_adopts_a_live_untracked_restart_safe_unit(
    monkeypatch,
    shims,
    conn,
):
    """The deploy moment for this branch: on a managed gateway with
    isolation 'none', a worker spawned by the PRE-FIX build runs in
    ``hermes-worker-kanban-<task>-run-<n>.scope`` while its row records no
    scope at all. The widened sweep now lists that unit, so without
    adoption the first tick after the upgrade would reap a LIVE worker.

    The unit is instead recognised through its own name, left running, and
    written back to the run — after which it is ordinary tracked state."""
    monkeypatch.setattr(
        _kanban_worker_scope, "_resolve_worker_isolation", lambda *a, **k: "none"
    )
    _patch_managed_gateway(monkeypatch, managed=True)
    live_pid = shims.sleeper()
    tid, run_id = _untracked_running_row(conn, pid=live_pid, age=600)
    live_unit = f"hermes-worker-kanban-{tid}-run-{run_id}.scope"
    shims.write_unit(live_unit, [live_pid])
    # A unit of the same shape whose task does not exist at all: the
    # adoption must not turn the whole prefix into a no-reap zone.
    orphan_pid = shims.sleeper()
    orphan_unit = "hermes-worker-kanban-t_ghost-run-1.scope"
    shims.write_unit(orphan_unit, [orphan_pid])

    assert _kanban_worker_recovery.reap_orphaned_worker_scopes(conn) == [orphan_unit]
    assert _kanban_db_dispatch._pid_alive(
        live_pid
    )  # the live worker survived the sweep
    assert shims.wait_for(lambda: not _kanban_db_dispatch._pid_alive(orphan_pid))
    # Backfilled on both rows, so the normal teardown covers the worker.
    assert (
        conn.execute("SELECT worker_scope FROM tasks WHERE id = ?", (tid,)).fetchone()[
            0
        ]
        == live_unit
    )
    assert (
        conn.execute(
            "SELECT worker_scope FROM task_runs WHERE id = ?", (run_id,)
        ).fetchone()[0]
        == live_unit
    )
    # Recording a scope must not hand the adopted row to the
    # never-registered launch grace: the row predates scope tracking, so
    # the same write normalises it the way an unscoped row is normalised.
    assert (
        conn.execute(
            "SELECT worker_registered_at FROM tasks WHERE id = ?", (tid,)
        ).fetchone()[0]
        is not None
    )
    assert _kanban_worker_recovery.fail_unregistered_workers(conn) == []
    assert _kanban_db_dispatch._pid_alive(live_pid)

    # Adoption is not immortality: once the task is no longer running, the
    # very same unit is an orphan again and the next sweep reaps it.
    conn.execute("UPDATE tasks SET status='done', worker_scope=NULL WHERE id=?", (tid,))
    conn.execute("UPDATE task_runs SET status='done' WHERE id=?", (run_id,))
    conn.commit()
    assert _kanban_worker_recovery.reap_orphaned_worker_scopes(conn) == [live_unit]
    assert shims.wait_for(lambda: not _kanban_db_dispatch._pid_alive(live_pid))


def test_reap_sweep_does_not_adopt_a_stale_earlier_attempt_unit(
    monkeypatch,
    shims,
    conn,
):
    """Adoption is keyed to the run's OWN attempt name. A unit left over
    from an EARLIER attempt of the same task names the same task id, so a
    task-id-only match would keep leaked scopes alive forever; only the
    current attempt's unit is adopted."""
    monkeypatch.setattr(
        _kanban_worker_scope, "_resolve_worker_isolation", lambda *a, **k: "none"
    )
    _patch_managed_gateway(monkeypatch, managed=True)
    tid, run_id = _untracked_running_row(conn, pid=shims.sleeper())
    stale_unit = f"hermes-worker-kanban-{tid}-run-{run_id - 1}.scope"
    stale_pid = shims.sleeper()
    shims.write_unit(stale_unit, [stale_pid])

    assert _kanban_worker_recovery.reap_orphaned_worker_scopes(conn) == [stale_unit]
    assert shims.wait_for(lambda: not _kanban_db_dispatch._pid_alive(stale_pid))
    assert (
        conn.execute("SELECT worker_scope FROM tasks WHERE id = ?", (tid,)).fetchone()[
            0
        ]
        is None
    )


from hermes_cli import kanban_boards as _kanban_boards
from hermes_cli import kanban_db_dispatch as _kanban_db_dispatch
from hermes_cli import kanban_worker_identity as _kanban_worker_identity
from hermes_cli import kanban_worker_recovery as _kanban_worker_recovery
from hermes_cli import kanban_worker_scope as _kanban_worker_scope
from hermes_cli import kanban_worker_stop as _kanban_worker_stop
