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

def test_managed_gateway_isolation_none_records_restart_safe_scope(
    monkeypatch, tmp_path,
):
    """isolation 'none' on a systemd-MANAGED gateway still runs the worker
    in a scope — upstream's restart-safe wrap, which a plain child needs or
    the next gateway restart kills it with the service cgroup. 'none' turns
    off KANBAN's isolation, not that guarantee, so the spawn must record the
    wrap's own unit as the run's scope. Recording "" for it (the pre-fix
    behaviour) marked the run registered instantly, left every terminal path
    with no scope to stop, and hid the unit from the audit sweep."""
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    _write_kanban_config(home, "  worker_isolation: none\n")
    monkeypatch.setattr(_kanban_worker_spawn, "_resolve_hermes_argv", lambda: ["hermes"])
    _patch_managed_gateway(monkeypatch, managed=True)
    _patch_systemd_available(monkeypatch, True)
    _patch_systemd_run_binary(monkeypatch)
    captured: dict = {}
    _fake_popen_capture(monkeypatch, captured, pid=4242)

    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    task = _make_task(task_id="t_scope1", run_id=7)
    pid = _kanban_worker_spawn._default_spawn(task, str(workspace))

    argv = captured["cmds"][0]
    assert argv[0] == "/usr/bin/systemd-run"
    unit_arg = argv[argv.index("--unit") + 1]
    # The restart-safe wrap's naming, NOT kanban's isolation naming.
    assert unit_arg == f"hermes-worker-kanban-{_kanban_worker_scope._scope_task_key('t_scope1')}-run-7"
    assert _kanban_worker_scope._kanban_worker_scope_unit("t_scope1", 7) not in argv
    # One wrap only — a nested scope would show a second systemd-run.
    assert argv.count("--unit") == 1
    assert argv[argv.index("--") + 1] == "hermes"
    # ... and the run records the unit systemd actually creates, so
    # `_set_worker_pid` writes worker_scope, teardown has something to
    # stop, and the audit sweep's parser maps it back to the task.
    assert pid.scope_unit == unit_arg+".scope"
    assert captured["kwargs"]["env"]["HERMES_KANBAN_SCOPE"] == pid.scope_unit
    assert _kanban_worker_scope._task_id_from_kanban_scope_unit(pid.scope_unit) == "t_scope1"


def test_managed_gateway_missing_run_id_refuses_before_any_launch(
    monkeypatch, tmp_path,
):
    """Both launch paths name the unit after the attempt, so a managed
    dispatch of a task with no current run id is refused above the
    isolation branch. Previously only the restart-safe path refused and
    the isolation path minted the attempt-free
    ``hermes-kanban-<task>.scope`` — a name a retry can collide with."""
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    _write_kanban_config(home, "  worker_isolation: auto\n")
    monkeypatch.setattr(_kanban_worker_spawn, "_resolve_hermes_argv", lambda: ["hermes"])
    _patch_managed_gateway(monkeypatch, managed=True)
    _patch_systemd_available(monkeypatch, True)
    _patch_systemd_run_binary(monkeypatch)
    captured: dict = {}
    _fake_popen_capture(monkeypatch, captured)

    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    with pytest.raises(RuntimeError, match="no current run id"):
        _kanban_worker_spawn._default_spawn(_make_task(run_id=None), str(workspace))

    assert captured == {}


def test_managed_gateway_refused_isolation_launch_never_retries(
    monkeypatch, tmp_path,
):
    """A refused isolation launch on a managed gateway has no unisolated
    fallback to degrade to: the 'plain' argv is itself a systemd-run wrap,
    so 'auto' must NOT retry. The old retry re-entered systemd-run under a
    different unit with no launch probe and no stderr capture, then
    recorded scope "" for a worker that was in fact scoped. Now it fails
    closed on the refusal's own stderr, as strict mode does."""
    workspace = _refused_launch_setup(monkeypatch, tmp_path, managed=True)
    calls: list[list[str]] = []
    _fake_refused_launch_popen(
        monkeypatch, calls, b"systemd-run-test: user bus connection refused\n",
    )

    with pytest.raises(RuntimeError, match="user bus connection refused"):
        _kanban_worker_spawn._default_spawn(_make_task(), str(workspace))

    # Exactly one launch attempt — no second systemd-run.
    assert len(calls) == 1
    assert calls[0][0] == "/usr/bin/systemd-run"
    # The refusal's stderr is what the dispatcher records as spawn_failed.
    # The exception above is the per-call spawn error receipt; no shared channel.


def test_scope_audit_sweeps_restart_safe_units_on_managed_gateway(
    monkeypatch, conn,
):
    """The audit lists BOTH unit prefixes and runs even with kanban's own
    isolation off, because a managed gateway's workers live in
    ``hermes-worker-kanban-*`` units. Gated on isolation alone, the sweep
    returned early and those units leaked unseen."""
    monkeypatch.setattr(_kanban_worker_scope, "_resolve_worker_isolation", lambda *a, **k: "none")
    _patch_managed_gateway(monkeypatch, managed=True)
    patterns: list[str] = []

    def fake_list(pattern):
        patterns.append(pattern)
        return {}

    monkeypatch.setattr(_kanban_worker_scope, "_kanban_list_scope_units", fake_list)

    assert _kanban_worker_recovery.reap_orphaned_worker_scopes(conn) == []
    assert patterns == ["hermes-kanban-*", "hermes-worker-kanban-*"]


def test_killing_the_launcher_does_not_kill_the_run(shims, conn, kanban_home):
    """THE launcher-vs-worker contract, end-to-end: a gateway death kills
    the systemd-run launcher (a child of the gateway) while the scoped
    worker survives — re-adoption must follow the registered worker via
    scope truth, and crash detection must NOT fire on the dead launcher."""
    _spawnable_profile(kanban_home)
    tid = kb.create_task(conn, title="restart survivor", assignee="elias")
    _kanban_db_dispatch.dispatch_once(conn, dry_run=False)

    row = conn.execute(
        "SELECT worker_pid, worker_scope, current_run_id FROM tasks "
        "WHERE id = ?", (tid,)
    ).fetchone()
    launcher_pid = row["worker_pid"]
    unit = row["worker_scope"]
    worker_pid = shims.unit_json(unit)["pids"][0]
    assert worker_pid != launcher_pid

    # The worker self-registers (first activity after spawn).
    assert _kanban_worker_identity.register_worker_pid(
        conn, tid, expected_run_id=row["current_run_id"], pid=worker_pid,
    )
    # Simulate the gateway dying: only the LAUNCHER dies.
    os.kill(launcher_pid, signal.SIGKILL)
    assert shims.wait_for(
        lambda: not _kanban_db_dispatch._pid_alive(launcher_pid), timeout=5.0
    )
    assert _kanban_db_dispatch._pid_alive(worker_pid)  # the scoped worker survives

    # The claim now names a dead gateway; heartbeat is fresh.
    host = kb._claimer_id().split(":", 1)[0]
    now = int(time.time())
    conn.execute(
        "UPDATE tasks SET claim_lock=?, claim_expires=?, "
        "last_heartbeat_at=? WHERE id=?",
        (f"{host}:4194304", now - 60, now, tid),
    )
    conn.commit()

    assert _kanban_worker_recovery.detect_crashed_workers(conn) == []  # scope truth: alive
    adopted = _kanban_worker_recovery.adopt_surviving_running_workers(conn)
    assert adopted == [tid]
    event = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'adopted'",
        (tid,),
    ).fetchone()
    payload = json.loads(event["payload"])
    assert payload.get("verified_by") == "scope_active"
    row = conn.execute(
        "SELECT status, claim_lock, worker_pid FROM tasks WHERE id = ?",
        (tid,),
    ).fetchone()
    assert row["status"] == "running"
    assert row["claim_lock"] == kb._claimer_id()
    assert row["worker_pid"] == worker_pid


def test_unscoped_rows_are_normalized_not_failed(shims, conn):
    """Unscoped running rows (plain spawns — and every legacy row from
    before the isolation feature, which are unscoped by definition) get
    ``worker_registered_at`` backfilled: their recorded pid IS the worker.
    Nothing is failed."""
    tid = kb.create_task(conn, title="plain legacy", assignee="w")
    _owner_kanban_claims.claim_task(conn, tid, claimer=kb._claimer_id())
    now = int(time.time())
    conn.execute(
        "UPDATE tasks SET status='running', worker_pid=?, "
        "worker_pid_started_at=?, worker_registered_at=NULL, "
        "started_at=? WHERE id=?",
        (os.getpid(), _kanban_worker_identity._worker_pid_start_time(os.getpid()),
         now - 99999, tid),
    )
    conn.commit()

    assert _kanban_worker_recovery.fail_unregistered_workers(conn) == []
    row = conn.execute(
        "SELECT status, worker_registered_at FROM tasks WHERE id = ?",
        (tid,),
    ).fetchone()
    assert row["status"] == "running"
    assert row["worker_registered_at"] is not None

    # Idempotent: a second sweep finds nothing to do and fails nothing.
    assert _kanban_worker_recovery.fail_unregistered_workers(conn) == []


def test_scope_liveness_is_cgroup_procs_truth(shims):
    """Alive/dead comes from the unit's cgroup.procs, not ActiveState
    alone: a live pid → alive; the pid dying (kernel drops it from the
    cgroup) → dead even though the unit is still loaded; a missing unit
    without kernel evidence remains unknown."""
    from tools import process_registry_scope as pr

    never = _kanban_worker_scope._kanban_worker_scope_unit("t_never", 1)
    assert pr._scope_unit_liveness(never) == "unknown"
    assert pr._scope_unit_active_state(never) == "unknown"

    pid = shims.sleeper()
    unit = _kanban_worker_scope._kanban_worker_scope_unit("t_cg", 1)
    shims.write_unit(unit, [pid])
    assert pid in shims.cgroup_pids(unit)
    assert pr._scope_unit_liveness(unit) == "alive"
    assert pr._scope_unit_active_state(unit) == "active"

    # The pid dies outside any stop: the cgroup empties, the unit stays
    # loaded — still dead, because pids are the truth.
    os.kill(pid, signal.SIGKILL)
    assert shims.wait_for(
        lambda: pr._scope_unit_liveness(unit) == "dead"
    ), "kernel-dropped pid must read as dead via cgroup.procs"


def test_scope_liveness_deactivating_and_query_failure_are_not_dead(shims):
    """ActiveState=deactivating (stop job draining) and an unreachable
    systemctl are 'unknown' — never 'dead': releasing a claim on either
    would let the dispatcher spawn beside a still-draining scope."""
    from tools import process_registry_scope as pr

    pid = shims.sleeper()
    unit = _kanban_worker_scope._kanban_worker_scope_unit("t_drain", 1)
    shims.write_unit(unit, [pid])
    shims.arm_deactivating(unit)
    assert pr._scope_unit_liveness(unit) == "unknown"
    assert pr._scope_unit_active_state(unit) == "unknown"

    # systemctl gone entirely (a wedged user bus): unknown, and the
    # verified stop reports failure instead of assuming success.
    real_which = shutil.which

    def no_systemctl(name, *args, **kwargs):
        return None if name == "systemctl" else real_which(name, *args, **kwargs)

    shims.clear_deactivating(unit)
    monkey = pytest.MonkeyPatch()
    monkey.setattr(shutil, "which", no_systemctl)
    try:
        assert pr._scope_unit_liveness(unit) == "unknown"
        assert pr._stop_systemd_unit_verified(unit) is False
    finally:
        monkey.undo()


def test_scope_liveness_unreadable_procs_file_is_not_death(shims):
    """Gate B pass 4 finding C: a LOADED unit whose cgroup.procs path
    cannot be read (mis-derived prefix, custom cgroup mount, container
    or namespace layout) must classify as 'unknown', never 'dead' — the
    old FileNotFoundError->dead verdict released claims beside live
    scopes. Only the unit not being loaded, or a procs file that was
    actually READ and found empty, is verified death."""
    from tools import process_registry_scope as pr

    pid = shims.sleeper()
    unit = _kanban_worker_scope._kanban_worker_scope_unit("t_badcg", 1)
    shims.write_unit(unit, [pid])
    assert pr._scope_unit_liveness(unit) == "alive"

    shims.arm_bad_cgroup_path(unit)
    assert pr._scope_unit_liveness(unit) == "unknown"
    assert pr._scope_unit_active_state(unit) == "unknown"
    # An unreadable cgroup cannot CONFIRM a stop either (the stop still
    # fires — signalling a unit we want dead is correct — but the
    # verdict stays False so callers keep their claim and retry).
    assert pr._stop_systemd_unit_verified(unit) is False

    # Contrast: not-loaded units stay verified dead (cgroup gone with
    # the unit), and an EMPTY procs file that was read is real death.
    other = _kanban_worker_scope._kanban_worker_scope_unit("t_badcg", 2)
    assert pr._scope_unit_liveness(other) == "dead"
    shims.write_unit(other, [])
    assert pr._scope_unit_liveness(other) == "dead"


def test_dispatch_dead_scope_sweep_clears_and_promotes_in_same_tick(
    shims, conn, kanban_home,
):
    """Pass 12 (AQ), recovery by the sweep: the dead-scope pass runs
    BEFORE promotion, so a breaker row whose scope died since the last
    tick is cleared, promoted, and spawned exactly once — all in the
    same tick."""
    _spawnable_profile(kanban_home)
    worker = shims.sleeper()
    unit = _kanban_worker_scope._kanban_worker_scope_unit("t_aqsweep", 1)
    shims.write_unit(unit, [worker])
    tid = _breaker_shaped_row(conn, unit, title="breaker swept")

    # The scope dies before the tick (cgroup empties; LoadState stays
    # loaded but cgroup.procs reads empty — a verified dead verdict).
    os.kill(worker, signal.SIGKILL)
    assert shims.wait_for(lambda: not _kanban_db_dispatch._pid_alive(worker))

    spawns: list[str] = []

    def stub_spawn(task, workspace, board):
        spawns.append(task.id)
        return None

    result = _kanban_db_dispatch._dispatch_once_locked(conn, spawn_fn=stub_spawn)

    assert result.scopes_cleared == [tid]
    assert result.skipped_scope_live == []
    assert spawns == [tid]  # promoted and spawned, exactly once
    row = conn.execute(
        "SELECT status, worker_scope, claim_lock FROM tasks WHERE id=?",
        (tid,),
    ).fetchone()
    assert row["worker_scope"] is None
    cleared = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? "
        "AND kind='worker_scope_cleared'", (tid,),
    ).fetchone()
    assert cleared and json.loads(cleared["payload"])["scope"] == unit


def test_release_stale_claims_never_terminates_refreshed_claim(
    shims, conn, monkeypatch,
):
    """Pass 10 (AK), marker-free row: the same interleaving on the
    generic path — the heartbeat lands after the scan but before
    ``_terminate_reclaimed_worker`` — must stand the sweep down instead
    of firing the pid-kill backstop on a live worker whose claim was
    refreshed after the snapshot."""
    worker = shims.sleeper()
    unit = _kanban_worker_scope._kanban_worker_scope_unit("t_freshhb", 1)
    shims.write_unit(unit, [worker])
    tid = kb.create_task(conn, title="fresh heartbeat", assignee="w")
    _owner_kanban_claims.claim_task(conn, tid, claimer=kb._claimer_id())
    now = int(time.time())
    conn.execute(
        "UPDATE tasks SET status='running', worker_pid=?, "
        "worker_pid_started_at=?, worker_registered_at=?, worker_scope=?, "
        "claim_expires=?, last_heartbeat_at=? WHERE id=?",
        (worker, _kanban_worker_identity._worker_pid_start_time(worker), now, unit,
         now - 60, now - 7200, tid),  # heartbeat past the 1h backstop
    )
    conn.commit()

    real_alive = _kanban_worker_identity._run_worker_alive

    def alive_after_a_heartbeat(row_arg):
        # The heartbeat commits between the scan and the termination
        # call (the exact window the stale snapshot used to cover).
        _owner_kanban_claims.heartbeat_claim(conn, tid)
        return real_alive(row_arg)

    monkeypatch.setattr(_kanban_worker_identity, "_run_worker_alive", alive_after_a_heartbeat)
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
    assert row["claim_expires"] > int(time.time()) + 600
    assert signalled == [], "the refreshed claim is skipped before any signal"
    assert _kanban_db_dispatch._pid_alive(worker), "the live worker was never signalled"
    assert [s["action"] for s in shims.stops() if s["unit"] == unit] == []


def test_enforce_max_runtime_defers_then_completes(shims, conn):
    """Same deferral contract on the max-runtime path: unverified stop →
    skip this tick; once verifiable → timeout recorded, scope stopped."""
    pid = shims.sleeper()
    unit = _kanban_worker_scope._kanban_worker_scope_unit("t_maxrt", 1)
    shims.write_unit(unit, [pid])
    shims.arm_killproof(unit)
    tid = kb.create_task(conn, title="timeout", assignee="w")
    _owner_kanban_claims.claim_task(conn, tid, claimer=kb._claimer_id())
    now = int(time.time())
    conn.execute(
        "UPDATE tasks SET status='running', worker_pid=?, worker_scope=?, "
        "worker_registered_at=?, max_runtime_seconds=1, started_at=? "
        "WHERE id=?",
        (pid, unit, now, now - 100, tid),
    )
    conn.execute(
        "UPDATE task_runs SET started_at = started_at - 9999 "
        "WHERE id=(SELECT current_run_id FROM tasks WHERE id=?)", (tid,),
    )
    conn.commit()

    def fake_kill(pid_, sig):
        raise ProcessLookupError()

    assert _kanban_worker_recovery.enforce_max_runtime(conn, signal_fn=fake_kill) == []
    row = conn.execute(
        "SELECT status, worker_scope FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row["status"] == "running"
    assert row["worker_scope"] == unit

    shims.clear_killproof(unit)
    assert _kanban_worker_recovery.enforce_max_runtime(conn, signal_fn=fake_kill) == [tid]
    row = conn.execute(
        "SELECT status, worker_scope FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row["worker_scope"] is None


def test_enforce_max_runtime_legacy_row_matching_start_still_killed(
    shims, conn
):
    """B, control case: a legacy row whose live pid started AFTER the run
    began (the normal case — worker spawned at run start) still gets the
    bare-pid timeout; the gate must not regress legacy handling."""
    pid = shims.sleeper()
    tid = _max_runtime_row(conn, pid, None, pid_started_at=None)

    assert _kanban_worker_recovery.enforce_max_runtime(conn) == [tid]
    assert not _kanban_db_dispatch._pid_alive(pid)  # teardown verified, not assumed
    payload = _timed_out_payload(conn, tid)
    assert payload.get("pid") == pid
    assert "pid_reused" not in payload


def test_enforce_max_runtime_unknown_identity_retains_claim(conn, monkeypatch):
    """Unattributable live worker is neither signalled nor declared terminated."""
    tid = _max_runtime_row(conn, 12345, None, pid_started_at=None)
    before = conn.execute("SELECT current_run_id FROM tasks WHERE id=?", (tid,)).fetchone()[0]
    monkeypatch.setattr(_kanban_worker_identity, "_legacy_worker_fingerprint", lambda *a: None)
    monkeypatch.setattr(_kanban_worker_identity, "_worker_pid_identity_state", lambda *a: "unknown")
    signals = []
    assert _kanban_worker_recovery.enforce_max_runtime(conn, signal_fn=lambda *a: signals.append(a)) == []
    assert signals == []
    row = conn.execute("SELECT status,current_run_id,consecutive_failures FROM tasks WHERE id=?", (tid,)).fetchone()
    assert tuple(row) == ("running", before, 0)



def test_enforce_max_runtime_identity_unreadable_not_signalled(
    shims, conn, monkeypatch
):
    """P, fingerprint half: a row WITH a start fingerprint whose live
    process start time cannot be read is an unknown identity — no
    bare-pid signal, the run still times out."""
    import gateway.status as gateway_status

    pid = shims.sleeper()
    tid = _max_runtime_row(conn, pid, None, pid_started_at=111111)
    monkeypatch.setattr(
        gateway_status, "get_process_start_time", lambda _pid: None,
    )
    signals: list[tuple[int, int]] = []

    def recording_kill(p, s):
        signals.append((p, s))

    assert _kanban_worker_recovery.enforce_max_runtime(conn, signal_fn=recording_kill) == [tid]
    assert signals == []
    assert _kanban_db_dispatch._pid_alive(pid)
    payload = _timed_out_payload(conn, tid)
    assert payload.get("signal_skipped") == "pid_identity_unknown"


def test_reopen_demotion_checks_identity_for_every_descendant(
    shims, conn, monkeypatch,
):
    """W: the reopen demotion's run-identity guard must not depend on the
    row still having a worker_scope. Phase 0 can confirm the OLD scoped
    run's cgroup empty while a retry replaces it with a fresh UNSCOPED
    run (the spawn-fallback shape) before the demotion transaction — a
    stop verdict about the old run must not demote (or kill) the new
    one. Per-row guards, not a blanket stand-down: an unchanged unscoped
    running descendant still demotes with its kill queued."""
    import threading

    parent = kb.create_task(conn, title="ancestor", assignee="planner")
    assert _kanban_completion.complete_task(conn, parent)
    child = kb.create_task(
        conn, title="replaced child", assignee="builder", parents=[parent],
    )
    _owner_kanban_claims.claim_task(conn, child, claimer=kb._claimer_id())
    # The old run: scoped, cgroup already empty (the unit was never
    # written, so the Phase 0 probe confirms dead instantly).
    old_unit = _kanban_worker_scope._kanban_worker_scope_unit("t_w_old", 1)
    conn.execute(
        "UPDATE tasks SET status='running', worker_scope=? WHERE id=?",
        (old_unit, child),
    )
    # Positive control: an unscoped running descendant whose identity
    # stays stable across the two phases — still demoted, kill queued.
    steady = kb.create_task(
        conn, title="steady child", assignee="builder", parents=[parent],
    )
    _owner_kanban_claims.claim_task(conn, steady, claimer=kb._claimer_id())
    conn.execute(
        "UPDATE tasks SET status='running', worker_pid=424242 WHERE id=?",
        (steady,),
    )
    conn.commit()

    # Race the probe->txn window: right after the OLD scope's stop
    # confirms, replace the run with a fresh UNSCOPED worker.
    real_stop = _kanban_worker_stop.request_worker_scope_stop
    replaced = threading.Event()

    def stop_then_replace(unit, **kwargs):
        result = real_stop(unit, **kwargs)
        if result and unit == old_unit and not replaced.is_set():
            replaced.set()
            new_pid = shims.sleeper()
            with _kanban_db_connect.write_txn(conn):
                kb._end_run(conn, child, outcome="crashed", status="ready")
                conn.execute(
                    "UPDATE tasks SET status='ready', claim_lock=NULL, "
                    "claim_expires=NULL, worker_pid=NULL, "
                    "worker_pid_started_at=NULL, worker_registered_at=NULL, "
                    "worker_scope=NULL WHERE id=?",
                    (child,),
                )
            assert _owner_kanban_claims.claim_task(conn, child, claimer=kb._claimer_id())
            with _kanban_db_connect.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET status='running' WHERE id=?", (child,),
                )
            _kanban_db_dispatch._set_worker_pid(conn, child, new_pid)
            stop_then_replace.new_pid = new_pid
        return result

    monkeypatch.setattr(_kanban_worker_stop, "request_worker_scope_stop", stop_then_replace)

    result = _kanban_transitions.invalidate_descendants_for_parent_reopen(
        conn, parent, author="operator",
    )
    assert replaced.is_set()
    row = conn.execute(
        "SELECT status, worker_scope, worker_pid FROM tasks WHERE id=?",
        (child,),
    ).fetchone()
    # The new run is NOT demoted, never queued for a kill, and its
    # worker survives the retraction untouched.
    assert row["status"] == "running"
    assert row["worker_scope"] is None
    assert row["worker_pid"] == stop_then_replace.new_pid
    assert _kanban_db_dispatch._pid_alive(stop_then_replace.new_pid)
    assert all(entry["id"] != child for entry in result["invalidated"])
    assert all(
        t[0] != stop_then_replace.new_pid for t in result["terminations"]
    )
    # The stable unscoped sibling demotes as before.
    assert any(entry["id"] == steady for entry in result["invalidated"])
    assert any(t[0] == 424242 for t in result["terminations"])


def test_fast_worker_exit_is_not_a_launch_failure(
    shims, conn, kanban_home, monkeypatch,
):
    """A worker that legitimately finishes within the launch probe is NOT
    a failed systemd launch: rc=0 from the launcher means the scoped
    command ran and exited. Auto mode must NOT plain-spawn a duplicate
    beside it (the review's critical duplication bug) — the spawn stands,
    exactly one worker ever exists, and exit classification owns the
    outcome on the next tick."""
    _write_kanban_config(Path(kanban_home), "  worker_isolation: auto\n")
    _kanban_db_connect._INITIALIZED_PATHS.clear()
    _spawnable_profile(kanban_home)
    monkeypatch.setattr(
        _kanban_worker_spawn, "_resolve_hermes_argv",
        lambda: [sys.executable, "-c", "pass"],
    )
    tid = kb.create_task(conn, title="fast worker", assignee="elias")

    result = _kanban_db_dispatch.dispatch_once(conn, dry_run=False)
    assert [s[0] for s in result.spawned] == [tid]
    row = conn.execute(
        "SELECT worker_pid, worker_scope, status FROM tasks WHERE id = ?",
        (tid,),
    ).fetchone()
    assert row["worker_scope"] is not None  # no fallback re-spawn happened
    spawned_events = conn.execute(
        "SELECT count(*) AS n FROM task_events "
        "WHERE task_id=? AND kind='spawned'", (tid,),
    ).fetchone()
    assert spawned_events["n"] == 1  # exactly one worker, ever
    assert result.late_spawn_failed == []


def test_fast_worker_nonzero_exit_is_not_a_launch_failure(
    shims, conn, kanban_home, monkeypatch,
):
    """Same contract in strict mode with a non-zero rc: the unit WAS
    created, so the worker ran and exited 3 — that is a worker exit, not
    a launch failure. No spawn_failed, no duplicate spawn, no breaker
    tick; the exit registry classifies the run on a later tick."""
    _write_kanban_config(Path(kanban_home), "  worker_isolation: systemd-scope\n")
    _kanban_db_connect._INITIALIZED_PATHS.clear()
    _spawnable_profile(kanban_home)
    monkeypatch.setattr(
        _kanban_worker_spawn, "_resolve_hermes_argv",
        lambda: [sys.executable, "-c", "raise SystemExit(3)"],
    )
    tid = kb.create_task(conn, title="fast fail", assignee="elias")

    result = _kanban_db_dispatch.dispatch_once(conn, dry_run=False)
    assert [s[0] for s in result.spawned] == [tid]
    row = conn.execute(
        "SELECT worker_pid, worker_scope, status, consecutive_failures "
        "FROM tasks WHERE id = ?", (tid,),
    ).fetchone()
    assert row["status"] == "running"
    assert row["worker_scope"] is not None
    assert row["consecutive_failures"] == 0  # no spawn_failed recorded
    spawned_events = conn.execute(
        "SELECT count(*) AS n FROM task_events "
        "WHERE task_id=? AND kind='spawned'", (tid,),
    ).fetchone()
    assert spawned_events["n"] == 1
    # The load-bearing fact for this regression: without --collect the
    # FAILED unit stays loaded on the bus (inspectable), which is what
    # lets the probe read "ran and exited" instead of "never created"
    # (the shim models --collect unloading any completion; under the old
    # argv the unit was gone when the probe looked).  The probe SAW it —
    # and once the run is terminal the tick's sweep collects the unit
    # EXPLICITLY, which is observable in the shim's action log:
    unit = _kanban_worker_scope._kanban_worker_scope_unit(tid, 1)
    assert {"action": "reset-failed", "unit": unit} in shims.stops()


def test_dashboard_direct_running_to_ready_terminates_cleanly(
    shims, conn, kanban_home,
):
    """Dashboard drag running->ready must not crash the termination
    drain (Gate B review, finding 5): the direct path records the same
    four-field termination tuple as every other transition — with and
    without a scope — and a scoped worker's unit is stopped (verified)
    BEFORE the status lands, never beside it."""
    mod = _load_dashboard_plugin()

    # A scoped running row and an unscoped one.
    scoped_pid = shims.sleeper()
    unit = _kanban_worker_scope._kanban_worker_scope_unit("t_dash", 3)
    shims.write_unit(unit, [scoped_pid])
    tid_scoped = _scoped_task_row(
        conn, scope=unit, pid=scoped_pid, registered=True,
    )
    plain_pid = shims.sleeper()
    tid_plain = kb.create_task(conn, title="plain drag", assignee="w")
    _owner_kanban_claims.claim_task(conn, tid_plain, claimer=kb._claimer_id())
    now = int(time.time())
    conn.execute(
        "UPDATE tasks SET status='running', worker_pid=?, "
        "worker_pid_started_at=?, worker_registered_at=? WHERE id=?",
        (plain_pid, _kanban_worker_identity._worker_pid_start_time(plain_pid), now, tid_plain),
    )
    conn.commit()

    # The old bug: ValueError unpacking a two-tuple into four names.
    assert mod._set_status_direct(conn, tid_scoped, "ready") is True
    assert mod._set_status_direct(conn, tid_plain, "ready") is True

    for tid in (tid_scoped, tid_plain):
        row = conn.execute(
            "SELECT status, worker_pid, worker_pid_started_at, "
            "worker_registered_at, worker_scope FROM tasks WHERE id=?",
            (tid,),
        ).fetchone()
        assert row["status"] == "ready"
        assert row["worker_pid"] is None
        assert row["worker_pid_started_at"] is None
        assert row["worker_registered_at"] is None
        assert row["worker_scope"] is None
    # The scoped worker was terminated through its scope (the plain one
    # via the pid loop); both are gone.
    assert shims.wait_for(lambda: not _kanban_db_dispatch._pid_alive(scoped_pid))
    assert shims.cgroup_pids(unit) == []
    assert shims.wait_for(lambda: not _kanban_db_dispatch._pid_alive(plain_pid))


def test_ancestor_reopen_defers_scoped_running_descendant(shims, conn):
    """E, invalidation half: reopening an ancestor demotes a scoped
    running descendant only after its scope is verified dead. An
    unconfirmed stop defers the whole descendant (stays running, claim
    held, marker written) instead of parking a spawnable 'todo' beside a
    draining cgroup; a confirmed one demotes and the worker is dead."""
    # Wedged descendant.
    wedged_pid = shims.sleeper()
    wedged_unit = _kanban_worker_scope._kanban_worker_scope_unit("t_desc_wedged", 1)
    shims.write_unit(wedged_unit, [wedged_pid])
    shims.arm_deactivating(wedged_unit)
    shims.arm_killproof(wedged_unit)
    # Clean descendant.
    clean_pid = shims.sleeper()
    clean_unit = _kanban_worker_scope._kanban_worker_scope_unit("t_desc_clean", 1)
    shims.write_unit(clean_unit, [clean_pid])
    parent = kb.create_task(conn, title="ancestor", assignee="planner")
    assert _kanban_completion.complete_task(conn, parent)
    for scope, pid, title in (
        (wedged_unit, wedged_pid, "wedged child"),
        (clean_unit, clean_pid, "clean child"),
    ):
        child = kb.create_task(
            conn, title=title, assignee="builder", parents=[parent],
        )
        claimed = _owner_kanban_claims.claim_task(conn, child)
        assert claimed is not None and claimed.status == "running"
        _kanban_db_dispatch._set_worker_pid(conn, child, pid)
        conn.execute(
            "UPDATE tasks SET worker_scope=? WHERE id=?", (scope, child),
        )
        conn.commit()
    wedged_child, clean_child = (
        conn.execute(
            "SELECT id FROM tasks WHERE title=?", (t,),
        ).fetchone()["id"]
        for t in ("wedged child", "clean child")
    )

    result = _kanban_transitions.invalidate_descendants_for_parent_reopen(
        conn, parent, author="operator",
    )

    # Wedged: deferred whole — still running, claim held, marked.
    row = conn.execute(
        "SELECT status, claim_lock, worker_scope FROM tasks WHERE id=?",
        (wedged_child,),
    ).fetchone()
    assert row["status"] == "running"
    assert row["claim_lock"]
    assert row["worker_scope"] == wedged_unit
    kinds = [
        r["kind"] for r in conn.execute(
            "SELECT kind FROM task_events WHERE task_id=? ORDER BY id",
            (wedged_child,),
        ).fetchall()
    ]
    assert "scope_stopping" in kinds
    assert "reclaim_deferred" in kinds
    # Clean: verified dead first, then demoted — no termination tuple is
    # left for a post-commit kill of an already-empty cgroup.
    row = conn.execute(
        "SELECT status, worker_scope FROM tasks WHERE id=?",
        (clean_child,),
    ).fetchone()
    assert row["status"] == "todo"
    assert row["worker_scope"] is None
    assert result["terminations"] == []
    assert shims.wait_for(lambda: not _kanban_db_dispatch._pid_alive(clean_pid))
    assert _kanban_db_dispatch._pid_alive(wedged_pid)  # killproof held the wedge

    # Unwedge + re-run: the deferred descendant demotes on the retry.
    shims.clear_deactivating(wedged_unit)
    shims.clear_killproof(wedged_unit)
    _kanban_transitions.invalidate_descendants_for_parent_reopen(conn, parent, author="op")
    row = conn.execute(
        "SELECT status, worker_scope FROM tasks WHERE id=?",
        (wedged_child,),
    ).fetchone()
    assert row["status"] == "todo"
    assert row["worker_scope"] is None
    assert shims.wait_for(lambda: not _kanban_db_dispatch._pid_alive(wedged_pid))


def test_reap_orphaned_scope_sweep(shims, conn):
    """Active scopes with no running task claiming them are stopped; the
    scope a running task still owns is left alone."""
    orphan_pid = shims.sleeper()
    orphan_unit = _kanban_worker_scope._kanban_worker_scope_unit("t_orphan", 4)
    shims.write_unit(orphan_unit, [orphan_pid])
    # And a scope whose task moved on to a DIFFERENT unit name.
    stale_pid = shims.sleeper()
    stale_unit = _kanban_worker_scope._kanban_worker_scope_unit("t_moved", 1)
    shims.write_unit(stale_unit, [stale_pid])
    current_unit = _kanban_worker_scope._kanban_worker_scope_unit("t_moved", 2)
    live_pid = shims.sleeper()
    shims.write_unit(current_unit, [live_pid])
    tid = _scoped_task_row(conn, scope=current_unit, pid=live_pid)

    reaped = _kanban_worker_recovery.reap_orphaned_worker_scopes(conn)
    assert set(reaped) == {orphan_unit, stale_unit}
    assert shims.wait_for(lambda: not _kanban_db_dispatch._pid_alive(orphan_pid))
    assert shims.wait_for(lambda: not _kanban_db_dispatch._pid_alive(stale_pid))
    assert _kanban_db_dispatch._pid_alive(live_pid)  # the running task's worker survives
    row = conn.execute(
        "SELECT status FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row["status"] == "running"


def test_reap_sweep_bounds_synchronous_probes_per_tick(shims, conn, monkeypatch):
    """AD: the audit runs under the dispatch lock, so many orphans must
    not each cost a synchronous probe in ONE tick. At most
    _SCOPE_AUDIT_SYNCHRONOUS_UNITS_PER_TICK units are touched per tick
    (the lock hold stays bounded) and the rest are deferred — but over
    successive ticks all of them are swept."""
    units = []
    pids = []
    for i in range(10):
        pid = shims.sleeper()
        unit = _kanban_worker_scope._kanban_worker_scope_unit(f"t_adbound{i}", 1)
        shims.write_unit(unit, [pid])
        units.append(unit)
        pids.append(pid)

    real_state = _kanban_worker_scope._kanban_scope_state
    probes_per_tick: list[int] = []

    def counting_state(unit):
        probes_per_tick[-1] += 1
        return real_state(unit)

    monkeypatch.setattr(_kanban_worker_scope, "_kanban_scope_state", counting_state)

    all_reaped: list[str] = []
    first_tick_reaped: list[int] = []
    for tick in range(10):
        probes_per_tick.append(0)
        all_reaped.extend(_kanban_worker_recovery.reap_orphaned_worker_scopes(conn))
        # Bounded synchronous work: one listing (not counted) plus at
        # most one liveness probe per TOUCHED unit — never one per
        # orphan present. The bound is pinned as a LITERAL so editing
        # the constant cannot quietly widen this assertion with it.
        touched = probes_per_tick[-1]
        assert touched <= 3, (
            f"tick {tick}: {touched} synchronous probes under the "
            "dispatch lock"
        )
        first_tick_reaped.append(len(all_reaped))
        if len(all_reaped) == len(units):
            break

    assert sorted(all_reaped) == sorted(units), "every orphan swept eventually"
    # The deferral was real: the first tick left orphans untouched...
    assert first_tick_reaped[0] <= 3
    # ...and the sweep needed more than one tick for ten orphans.
    assert len(first_tick_reaped) >= 4
    assert shims.wait_for(
        lambda: all(not _kanban_db_dispatch._pid_alive(p) for p in pids), timeout=8.0
    ), "all ten orphan workers were eventually stopped"
    # And the sweep reached completion: a further tick finds nothing.
    probes_per_tick.append(0)
    assert _kanban_worker_recovery.reap_orphaned_worker_scopes(conn) == []


def test_reap_sweep_rotation_prevents_wedge_starvation(shims, conn):
    """AI: oldest-first alone let a handful of permanently wedged old
    scopes occupy the per-tick bound forever — every orphan behind them
    starved. The audit window must ROTATE across ticks, so newer
    orphans are probed (and reaped) within a bounded number of ticks
    regardless of wedged ones ahead of them in the age order."""
    # Three permanently wedged old scopes: a stop job that can never
    # complete (killproof) draining a process that never exits.
    wedged_units = []
    wedged_pids = []
    for i in range(3):
        pid = shims.sleeper()
        unit = _kanban_worker_scope._kanban_worker_scope_unit(f"t_wedge{i}", 1)
        shims.write_unit(unit, [pid])
        shims.arm_deactivating(unit)
        shims.arm_killproof(unit)
        wedged_units.append(unit)
        wedged_pids.append(pid)

    # Priming tick: the wedged scopes are seen first and take the whole
    # bound. They cannot be reaped, so they stay in the listing with an
    # older first-seen stamp than anything created after this point.
    assert _kanban_worker_recovery.reap_orphaned_worker_scopes(conn) == []

    # Two FRESH orphans appear behind the wedged ones in the age order.
    fresh_units = []
    fresh_pids = []
    for i in range(2):
        pid = shims.sleeper()
        unit = _kanban_worker_scope._kanban_worker_scope_unit(f"t_fresh{i}", 1)
        shims.write_unit(unit, [pid])
        fresh_units.append(unit)
        fresh_pids.append(pid)

    # Oldest-first would sit on the three wedged scopes every tick and
    # the fresh orphans would never be probed. Rotation must reach them
    # within a few ticks.
    all_reaped: list[str] = []
    for tick in range(4):
        all_reaped.extend(_kanban_worker_recovery.reap_orphaned_worker_scopes(conn))
        if set(fresh_units) <= set(all_reaped):
            break
    assert set(fresh_units) <= set(all_reaped), (
        "the fresh orphans starved behind the wedged scopes"
    )
    # The wedged scopes were never reaped (that is what makes them
    # wedged) and their workers are still alive.
    assert not (set(wedged_units) & set(all_reaped))
    assert all(_kanban_db_dispatch._pid_alive(p) for p in wedged_pids)
    # The fresh orphans were genuinely stopped, not merely listed.
    assert shims.wait_for(
        lambda: all(not _kanban_db_dispatch._pid_alive(p) for p in fresh_pids), timeout=8.0
    )


def test_reap_sweep_escalates_a_wedged_deactivating_orphan(shims, conn):
    """D: a deactivating orphan is not terminal — a stop job draining a
    stubborn process sits in deactivating forever.  The sweep must
    re-request the verified stop (whose SIGKILL escalation drains the
    wedge) instead of quietly collecting around it."""
    pid = shims.sleeper()
    unit = _kanban_worker_scope._kanban_worker_scope_unit("t_wedged", 1)
    shims.write_unit(unit, [pid])
    shims.arm_deactivating(unit)
    shims.arm_killproof(unit)  # the stop job never completes server-side

    # Wedged: stop re-requested, SIGKILL escalation fired, unit neither
    # confirmed nor collected.
    assert _kanban_worker_recovery.reap_orphaned_worker_scopes(conn) == []
    assert any(
        a["action"] == "stop" and a["unit"] == unit for a in shims.stops()
    )
    assert any(
        a["action"] == "kill" and a["unit"] == unit for a in shims.stops()
    )
    assert not any(
        a["action"] == "reset-failed" and a["unit"] == unit
        for a in shims.stops()
    )
    assert _kanban_db_dispatch._pid_alive(pid)  # killproof: the shim refused, unconfirmed

    # Unwedged: once the stop can complete the next sweep drains,
    # confirms, and collects the orphan like any other.
    shims.clear_deactivating(unit)
    shims.clear_killproof(unit)
    assert _kanban_worker_recovery.reap_orphaned_worker_scopes(conn) == [unit]
    assert shims.wait_for(lambda: not _kanban_db_dispatch._pid_alive(pid))
    assert any(
        a["action"] == "reset-failed" and a["unit"] == unit
        for a in shims.stops()
    )


def test_dispatch_tick_at_concurrency_cap_still_sweeps_orphan_scopes(
    shims, conn,
):
    """T: the cap guards return before the spawn loops, but the orphan
    scope audit must still fire — under sustained caps the tick used to
    return at the first guard and leaked scopes persisted indefinitely."""
    spawns: list[str] = []

    def never_spawn(task, workspace, board):
        spawns.append(task.id)
        return None

    orphan_pid = shims.sleeper()
    orphan_unit = _kanban_worker_scope._kanban_worker_scope_unit("t_cap_orphan", 1)
    shims.write_unit(orphan_unit, [orphan_pid])
    live_pid = shims.sleeper()
    live_unit = _kanban_worker_scope._kanban_worker_scope_unit("t_cap_live", 1)
    shims.write_unit(live_unit, [live_pid])
    tid = _scoped_task_row(conn, scope=live_unit, pid=live_pid)

    result = _kanban_db_dispatch._dispatch_once_locked(
        conn, spawn_fn=never_spawn, max_spawn=1,  # running_count == cap
    )

    assert spawns == []                       # the cap held
    assert result.scopes_reaped == [orphan_unit]  # but the audit ran
    assert shims.wait_for(lambda: not _kanban_db_dispatch._pid_alive(orphan_pid))
    assert _kanban_db_dispatch._pid_alive(live_pid)            # the claimed worker survives
    row = conn.execute(
        "SELECT status FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row["status"] == "running"


def test_dispatch_tick_under_critical_pressure_still_sweeps_orphan_scopes(
    shims, conn, monkeypatch,
):
    """T, the other early return: critical memory pressure stands the
    spawn side down but must not skip the orphan scope audit either."""
    monkeypatch.setattr(_kanban_db_dispatch, "_memory_pressure_level", lambda: "critical")

    spawns: list[str] = []

    def never_spawn(task, workspace, board):
        spawns.append(task.id)
        return None

    orphan_pid = shims.sleeper()
    orphan_unit = _kanban_worker_scope._kanban_worker_scope_unit("t_mem_orphan", 1)
    shims.write_unit(orphan_unit, [orphan_pid])

    result = _kanban_db_dispatch._dispatch_once_locked(conn, spawn_fn=never_spawn)

    assert spawns == []
    assert result.memory_pressure == "critical"
    assert result.scopes_reaped == [orphan_unit]
    assert shims.wait_for(lambda: not _kanban_db_dispatch._pid_alive(orphan_pid))


def test_generic_reclaim_never_signals_unreadable_identity(
    conn, monkeypatch, caplog,
):
    """X: the generic reclaim path treats an unreadable pid identity as a
    third state and never signals it — a signal we cannot attribute might
    hit an unrelated process. The row is reclaimed WITHOUT a signal (the
    same fail-safe stance enforce_max_runtime took in pass 4) and the
    stand-down warns once per run id."""
    import logging

    killed: list[tuple[int, int]] = []

    def fake_kill(pid, sig):
        killed.append((pid, sig))

    # Alive pid whose live start fingerprint cannot be read.
    monkeypatch.setattr(
        "gateway.status.get_process_start_time", lambda pid: None,
    )
    tid = kb.create_task(conn, title="unreadable identity", assignee="w")
    assert _owner_kanban_claims.claim_task(conn, tid, claimer=kb._claimer_id()) is not None
    _kanban_db_dispatch._set_worker_pid(conn, tid, os.getpid())
    run_id = kb._current_run_id(conn, tid)

    with caplog.at_level(logging.WARNING, logger="hermes_cli.kanban_db"):
        assert _owner_kanban_claims.reclaim_task(
            conn, tid, reason="operator reclaim", signal_fn=fake_kill,
        )
    assert killed == []  # unknown identity — no signal, ever
    row = conn.execute(
        "SELECT status, claim_lock FROM tasks WHERE id=?", (tid,),
    ).fetchone()
    assert row["status"] == "ready"
    assert row["claim_lock"] is None
    payload = json.loads(conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? "
        "AND kind='reclaimed'", (tid,),
    ).fetchone()["payload"])
    assert payload["termination_attempted"] is True
    assert payload["terminated"] is True
    assert payload["signal_skipped"] == "pid_identity_unknown"
    warnings = [
        r for r in caplog.records
        if "not signalling; reclaiming" in r.message
    ]
    assert len(warnings) == 1
    assert str(run_id) in warnings[0].message


def test_generic_reclaim_legacy_missing_fingerprint_not_signalled(
    conn, monkeypatch,
):
    """X, legacy half: a row with NO start fingerprint (pre-column spawn)
    is also 'unknown', never 'alive' — the old boolean helper folded
    missing into alive and signalled the bare pid. The row reclaims
    without a signal."""
    killed: list[tuple[int, int]] = []

    def fake_kill(pid, sig):
        killed.append((pid, sig))

    monkeypatch.setattr(_kanban_db_dispatch, "_pid_alive", lambda _pid: True)
    tid = kb.create_task(conn, title="legacy row", assignee="w")
    assert _owner_kanban_claims.claim_task(conn, tid, claimer=kb._claimer_id()) is not None
    with _kanban_db_connect.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET worker_pid=4242, worker_pid_started_at=NULL "
            "WHERE id=?",
            (tid,),
        )
    assert _owner_kanban_claims.reclaim_task(
        conn, tid, reason="operator reclaim", signal_fn=fake_kill,
    )
    assert killed == []
    payload = json.loads(conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? "
        "AND kind='reclaimed'", (tid,),
    ).fetchone()["payload"])
    assert payload["signal_skipped"] == "pid_identity_unknown"
    assert payload["terminated"] is True


def test_migration_adds_worker_columns_without_data_loss(tmp_path, monkeypatch):
    """A DB created with the pre-change schema opens cleanly: the three
    worker-lifecycle columns appear via the additive migration, and every
    legacy row survives intact."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    db_path = _owner_kanban_db.kanban_db_path(board="default")
    import sqlite3

    raw = sqlite3.connect(db_path)
    raw.executescript(_PRE_CHANGE_TASKS_SQL)
    now = int(time.time())
    raw.execute(
        "INSERT INTO tasks (id, title, assignee, status, created_by, "
        "created_at, started_at, worker_pid, claim_lock, claim_expires, "
        "current_run_id, consecutive_failures) "
        "VALUES ('t_legacy1', 'legacy run', 'elias', 'running', 'op', "
        "?, ?, 4242, 'oldhost:1', ?, 3, 2)",
        (now - 5000, now - 4000, now - 60),
    )
    raw.execute(
        "INSERT INTO tasks (id, title, assignee, status, created_by, "
        "created_at, completed_at) "
        "VALUES ('t_legacy2', 'legacy done', 'maya', 'done', 'op', ?, ?)",
        (now - 9000, now - 8000),
    )
    raw.commit()
    raw.close()

    conn = _kanban_db_connect.connect(db_path)
    try:
        cols = {
            r["name"] for r in conn.execute("PRAGMA table_info(tasks)")
        }
        assert {"worker_pid_started_at", "worker_scope",
                "worker_registered_at"} <= cols
        legacy = conn.execute(
            "SELECT title, assignee, status, worker_pid, claim_lock, "
            "       consecutive_failures, worker_pid_started_at, "
            "       worker_scope, worker_registered_at "
            "FROM tasks WHERE id = 't_legacy1'"
        ).fetchone()
        assert legacy["title"] == "legacy run"
        assert legacy["status"] == "running"
        assert legacy["worker_pid"] == 4242
        assert legacy["claim_lock"] == "oldhost:1"
        assert legacy["consecutive_failures"] == 2
        assert legacy["worker_pid_started_at"] is None
        assert legacy["worker_scope"] is None
        assert legacy["worker_registered_at"] is None
        done = conn.execute(
            "SELECT status, completed_at FROM tasks WHERE id = 't_legacy2'"
        ).fetchone()
        assert done["status"] == "done"
        assert done["completed_at"] == now - 8000
    finally:
        conn.close()


def test_migration_completes_a_partial_pre_existing_schema(
    tmp_path, monkeypatch,
):
    """I (new e): a DB caught mid-migration — tasks already carries TWO of
    the three worker-lifecycle columns (with live data in them) while
    task_runs predates the feature entirely — opens cleanly: the missing
    column is added, already-migrated values survive untouched, the run
    table gains its column, and a second open is a no-op."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    db_path = _owner_kanban_db.kanban_db_path(board="default")
    import sqlite3

    raw = sqlite3.connect(db_path)
    raw.executescript(_PRE_CHANGE_TASKS_SQL)
    raw.executescript(_PRE_CHANGE_TASK_RUNS_SQL)
    # Half-migrated tasks: the fingerprint and registration columns
    # shipped, worker_scope did not (a deploy interrupted mid-rollout).
    raw.execute("ALTER TABLE tasks ADD COLUMN worker_pid_started_at INTEGER")
    raw.execute("ALTER TABLE tasks ADD COLUMN worker_registered_at INTEGER")
    now = int(time.time())
    raw.execute(
        "INSERT INTO tasks (id, title, assignee, status, created_by, "
        "created_at, started_at, worker_pid, worker_pid_started_at, "
        "worker_registered_at, claim_lock, claim_expires, current_run_id) "
        "VALUES ('t_partial', 'half migrated', 'elias', 'running', 'op', "
        "?, ?, 4242, 1712345678, ?, 'oldhost:1', ?, 11)",
        (now - 5000, now - 4000, now - 3000, now - 60),
    )
    raw.execute(
        "INSERT INTO task_runs (id, task_id, profile, status, claim_lock, "
        "worker_pid, started_at) VALUES (11, 't_partial', 'elias', "
        "'running', 'oldhost:1', 4242, ?)",
        (now - 4000,),
    )
    raw.commit()
    raw.close()

    conn = _kanban_db_connect.connect(db_path)
    try:
        cols = {
            r["name"] for r in conn.execute("PRAGMA table_info(tasks)")
        }
        run_cols = {
            r["name"] for r in conn.execute("PRAGMA table_info(task_runs)")
        }
        assert {"worker_pid_started_at", "worker_scope",
                "worker_registered_at"} <= cols
        assert "worker_scope" in run_cols
        task = conn.execute(
            "SELECT worker_pid_started_at, worker_registered_at "
            "FROM tasks WHERE id = 't_partial'"
        ).fetchone()
        # The already-migrated half is preserved, not reset to defaults.
        assert task["worker_pid_started_at"] == 1712345678
        assert task["worker_registered_at"] == now - 3000
        run = conn.execute(
            "SELECT worker_pid, status, worker_scope FROM task_runs "
            "WHERE id = 11"
        ).fetchone()
        assert run["worker_pid"] == 4242
        assert run["status"] == "running"
        assert run["worker_scope"] is None
    finally:
        conn.close()

    # Idempotent: a second open changes nothing.
    conn = _kanban_db_connect.connect(db_path)
    try:
        task = conn.execute(
            "SELECT worker_pid_started_at, worker_registered_at, "
            "worker_scope FROM tasks WHERE id = 't_partial'"
        ).fetchone()
        assert task["worker_pid_started_at"] == 1712345678
        assert task["worker_registered_at"] == now - 3000
        assert task["worker_scope"] is None
    finally:
        conn.close()

from hermes_cli import kanban_boards as _kanban_boards
from hermes_cli import kanban_completion as _kanban_completion
from hermes_cli import kanban_db_connect as _kanban_db_connect
from hermes_cli import kanban_db_dispatch as _kanban_db_dispatch
from hermes_cli import kanban_transitions as _kanban_transitions
from hermes_cli import kanban_worker_identity as _kanban_worker_identity
from hermes_cli import kanban_worker_recovery as _kanban_worker_recovery
from hermes_cli import kanban_worker_scope as _kanban_worker_scope
from hermes_cli import kanban_worker_spawn as _kanban_worker_spawn
from hermes_cli import kanban_worker_stop as _kanban_worker_stop
