from __future__ import annotations

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

def test_default_spawn_wraps_argv_in_systemd_scope(monkeypatch, tmp_path):
    """Probe passes + isolation auto → systemd-run prefix with the
    run-suffixed unit name, description, per-worker memory properties, and
    the legacy argv intact after ``--``. The unwrapped baseline is captured
    from the same code path with the probe disabled, so the comparison is
    exact."""
    config = "  worker_isolation: auto\n  worker_memory_max_mb: 512\n"
    plain, plain_pid = _capture_worker_argv(
        monkeypatch, tmp_path, config, systemd_available=False
    )
    _assert_plain_argv_shape(plain)

    # Same config, probe now passes → same spawn, wrapped.
    wrapped, wrapped_pid = _capture_worker_argv(
        monkeypatch, tmp_path, config, systemd_available=True
    )

    assert wrapped[0] == "/usr/bin/systemd-run"
    # Flags before the command separator, in the builder's canonical order.
    head = wrapped[: wrapped.index("--")]
    assert head[1:5] == ["--user", "--scope", "--quiet", "--unit"]
    unit = _kanban_worker_scope._kanban_worker_scope_unit("t_scope1", 7)
    assert unit.startswith("hermes-kanban-b") and unit.endswith("--t_scope1-r7.scope")
    assert unit in head
    # No --collect: a fast nonzero worker exit must leave the failed
    # unit LOADED so the launch probe can tell "ran" from "refused"
    # (collection is explicit once the run is terminal).
    assert "--collect" not in head
    assert head[head.index("--description") + 1] == (
        "Hermes kanban worker t_scope1: build the widget"
    )
    assert head[head.index("--property") + 1] == "MemoryAccounting=yes"
    assert head[head.index("--property") + 3] == f"MemoryMax={512 * 1024 * 1024}"
    assert head[head.index("--property") + 5] == f"MemorySwapMax={512 * 1024 * 1024}"
    assert "OOMPolicy=kill" not in head  # scopes reject this service-only property
    # Legacy argv preserved verbatim after the separator.
    assert wrapped[wrapped.index("--") + 1:] == plain
    # The spawn published the unit on the returned pid — a per-call
    # channel, usable verbatim as an int by every existing caller.
    assert isinstance(wrapped_pid, int)
    assert wrapped_pid.scope_unit == unit
    assert plain_pid.scope_unit == ""


def test_default_spawn_memory_default_derives_both_bounds(monkeypatch, tmp_path):
    """No explicit worker_memory_max_mb → BOTH MemoryMax and MemorySwapMax
    come from the shared process-registry helper, at the same value."""
    wrapped = _capture_worker_argv(
        monkeypatch, tmp_path, "  worker_isolation: auto\n",
        systemd_available=True,
    )
    head = wrapped[0][: wrapped[0].index("--")]
    expected = _kanban_worker_scope._kanban_worker_memory_bytes()
    assert expected, "helper must return a positive bound on a normal host"
    props = [p for p in head if p.startswith("MemoryMax=")]
    assert props == [f"MemoryMax={expected}"], props
    swap = [p for p in head if p.startswith("MemorySwapMax=")]
    assert swap == [f"MemorySwapMax={expected}"], swap


def test_default_spawn_omits_memory_when_helper_falsy(monkeypatch, tmp_path):
    """A helper that cannot compute a bound (returns 0/None) → BOTH memory
    properties omitted. MemoryMax=0 means 'no limit' in systemd — the exact
    opposite of the intended bound — so emitting it is worse than nothing."""
    monkeypatch.setattr(
        'tools.process_registry_scope._worker_memory_max_bytes', lambda: 0
    )
    _kanban_worker_scope._memory_bound_omitted_warned = False
    try:
        wrapped = _capture_worker_argv(
            monkeypatch, tmp_path, "  worker_isolation: auto\n",
            systemd_available=True,
        )
    finally:
        _kanban_worker_scope._memory_bound_omitted_warned = False
    head = wrapped[0][: wrapped[0].index("--")]
    assert not [p for p in head if p.startswith("MemoryMax=")]
    assert not [p for p in head if p.startswith("MemorySwapMax=")]


def test_default_spawn_none_keeps_legacy_argv_exactly(monkeypatch, tmp_path):
    """isolation 'none' must produce today's argv byte-for-byte, even when
    systemd is fully available — the rollback contract."""
    none_cmd, none_pid = _capture_worker_argv(
        monkeypatch, tmp_path, "  worker_isolation: none\n",
        systemd_available=True,
    )
    _assert_plain_argv_shape(none_cmd)
    # 'none' ignores availability: a second capture with the probe down
    # (the classic macOS/container host) is byte-identical.
    fallback_cmd, fallback_pid = _capture_worker_argv(
        monkeypatch, tmp_path, "  worker_isolation: none\n",
        systemd_available=False,
    )
    assert fallback_cmd == none_cmd
    assert none_pid.scope_unit == ""
    assert fallback_pid.scope_unit == ""


def test_default_spawn_auto_without_systemd_keeps_legacy_argv(monkeypatch, tmp_path):
    """Unusable systemd (macOS / containers) with the default 'auto' mode
    silently falls back to the plain argv — no behavioural change."""
    cmd, pid = _capture_worker_argv(
        monkeypatch, tmp_path, "  worker_isolation: auto\n",
        systemd_available=False,
    )
    _assert_plain_argv_shape(cmd)
    assert pid.scope_unit == ""


def test_spawn_scope_unit_is_per_call_not_global(monkeypatch, tmp_path):
    """F: the scope unit travels on the spawn's RETURN VALUE, not on a
    process-global function attribute. Per-board dispatches run
    concurrently, so the old global let board B's spawn overwrite the
    unit board A's dispatcher was about to record. Here: spawn A, then
    spawn B, then read A's unit — under the old global that read
    returned B's unit; per-call it is stable and each pid works as a
    plain int for every caller contract."""
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    _write_kanban_config(home, "  worker_isolation: auto\n")
    monkeypatch.setattr(_kanban_worker_spawn, "_resolve_hermes_argv", lambda: ["hermes"])
    _patch_systemd_available(monkeypatch, True)
    _patch_systemd_run_binary(monkeypatch)
    captured: dict = {}
    _fake_popen_capture(monkeypatch, captured, pid=4242)

    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    pid_a = _kanban_worker_spawn._default_spawn(
        _make_task(task_id="t_board_a", run_id=11), str(workspace),
    )
    pid_b = _kanban_worker_spawn._default_spawn(
        _make_task(task_id="t_board_b", run_id=22), str(workspace),
    )

    unit_a = _kanban_worker_scope._kanban_worker_scope_unit("t_board_a", 11)
    unit_b = _kanban_worker_scope._kanban_worker_scope_unit("t_board_b", 22)
    assert unit_a != unit_b
    # A's unit survives B's spawn — the cross-wire the global allowed.
    assert pid_a.scope_unit == unit_a
    assert pid_b.scope_unit == unit_b
    # The annotated pid is a real int for every existing caller
    # contract (truthiness, int(), str(), arithmetic).
    assert pid_a == 4242
    assert int(pid_a) == 4242
    assert str(pid_a) == "4242"
    # And the old global channel no longer exists to cross-wire.
    assert not hasattr(_kanban_worker_spawn._default_spawn, "_last_scope_unit")


def test_spawn_reads_gateway_topology_once_and_keeps_it(
    monkeypatch, tmp_path,
):
    """The managed/unmanaged answer is snapshotted at the top of the spawn
    and reused, so a transient probe failure cannot flip the path mid-spawn.

    ``_is_supervised_gateway_process`` swallows every exception into False,
    so re-asking after the launch could downgrade a managed gateway to
    "unmanaged" and take the plain fallback — which on a managed host is
    itself a systemd-run wrap under a different unit name, i.e. a second
    worker recorded as unscoped. The probe here flips exactly at that
    moment: True while the spawn decides, False from the launch onwards."""
    workspace = _refused_launch_setup(monkeypatch, tmp_path, managed=True)
    supervised = {"value": True}
    monkeypatch.setattr(
        'tools.process_registry_scope._is_supervised_gateway_process',
        lambda: supervised["value"],
    )
    calls: list[list[str]] = []
    _fake_refused_launch_popen(
        monkeypatch, calls, b"systemd-run-test: user bus connection refused\n",
    )
    refusing_popen = subprocess.Popen  # the refusing fake installed above

    def flipping_popen(cmd, *args, **kwargs):
        supervised["value"] = False  # the transient failure lands here
        return refusing_popen(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", flipping_popen)

    with pytest.raises(RuntimeError, match="user bus connection refused"):
        _kanban_worker_spawn._default_spawn(_make_task(), str(workspace))

    # Still exactly one launch: the post-launch decision used the snapshot
    # taken before the flip, so the managed-gateway "no unisolated
    # fallback" rule held instead of plain-spawning a duplicate.
    assert len(calls) == 1
    assert calls[0][0] == "/usr/bin/systemd-run"


def test_unmanaged_host_refused_isolation_launch_retries_plain_argv(
    monkeypatch, tmp_path,
):
    """Off a managed gateway the auto fallback is still correct and still
    honest: the retry is the genuinely bare worker argv (not one systemd
    token), and the run records no scope unit."""
    workspace = _refused_launch_setup(monkeypatch, tmp_path, managed=False)
    calls: list[list[str]] = []
    _fake_refused_launch_popen(
        monkeypatch, calls, b"systemd-run-test: user bus connection refused\n",
    )

    pid = _kanban_worker_spawn._default_spawn(_make_task(), str(workspace))

    assert len(calls) == 2
    assert calls[0][0] == "/usr/bin/systemd-run"
    _assert_plain_argv_shape(calls[1])
    assert pid == 4242
    assert pid.scope_unit == ""


def test_forced_scope_without_systemd_refuses_spawn(monkeypatch, tmp_path):
    """H: 'systemd-scope' + unusable probe = REFUSED spawn, never a silent
    unisolated fallback. The operator pinned strict — "no worker" beats an
    unisolated worker; only 'auto' may fall back. The refusal raises with
    the operator-facing reason BEFORE any process is launched."""
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    _write_kanban_config(home, "  worker_isolation: systemd-scope\n")
    monkeypatch.setattr(_kanban_worker_spawn, "_resolve_hermes_argv", lambda: ["hermes"])
    _patch_systemd_available(monkeypatch, False)
    captured: dict = {}
    _fake_popen_capture(monkeypatch, captured)
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)

    with pytest.raises(RuntimeError, match="worker_isolation=systemd-scope"):
        _kanban_worker_spawn._default_spawn(_make_task(), str(workspace))

    # Nothing was ever launched — neither isolated nor unisolated.
    assert captured == {}


def test_forced_scope_vanished_binary_refuses_spawn(monkeypatch, tmp_path):
    """H, second gap: the probe passed but systemd-run disappeared from
    PATH before the argv build, so the builder returned the argv
    unwrapped. Strict mode refuses here too — an unwrapped argv must not
    become a silent unisolated spawn one code path away from the probe
    refusal."""
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    _write_kanban_config(home, "  worker_isolation: systemd-scope\n")
    monkeypatch.setattr(_kanban_worker_spawn, "_resolve_hermes_argv", lambda: ["hermes"])
    _patch_systemd_available(monkeypatch, True)
    real_which = shutil.which

    def gone_which(name, *args, **kwargs):
        if name == "systemd-run":
            return None
        return real_which(name, *args, **kwargs)

    monkeypatch.setattr(shutil, "which", gone_which)
    captured: dict = {}
    _fake_popen_capture(monkeypatch, captured)
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)

    with pytest.raises(RuntimeError, match="disappeared between"):
        _kanban_worker_spawn._default_spawn(_make_task(), str(workspace))

    assert captured == {}


def test_set_worker_pid_records_scope_and_start_fingerprint(conn):
    """The pid, its start-time fingerprint (PID-reuse guard), and the scope
    unit land on both the task row and the active run; the ``spawned``
    event carries the scope for operators. A scoped row is NOT registered
    (the pid is the launcher's); a plain spawn is registered immediately —
    its pid IS the worker."""
    tid = kb.create_task(conn, title="record", assignee="w")
    _owner_kanban_claims.claim_task(conn, tid, claimer=kb._claimer_id())

    live = os.getpid()
    unit = _kanban_worker_scope._kanban_worker_scope_unit(tid, None)
    _kanban_db_dispatch._set_worker_pid(conn, tid, live, scope_unit=unit)

    row = conn.execute(
        "SELECT worker_pid, worker_pid_started_at, worker_scope, "
        "       worker_registered_at FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row["worker_pid"] == live
    assert row["worker_pid_started_at"] == _kanban_worker_identity._worker_pid_start_time(live)
    assert row["worker_scope"] == unit
    assert row["worker_registered_at"] is None  # launcher pid, not worker

    run = conn.execute(
        "SELECT worker_pid, worker_scope FROM task_runs "
        "WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
        (tid,),
    ).fetchone()
    assert run["worker_pid"] == live
    assert run["worker_scope"] == unit

    event = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'spawned'",
        (tid,),
    ).fetchone()
    assert event and json.loads(event["payload"])["scope"] == unit

    # Plain spawn: no scope → registered at spawn time.
    tid2 = kb.create_task(conn, title="plain", assignee="w")
    _owner_kanban_claims.claim_task(conn, tid2, claimer=kb._claimer_id())
    _kanban_db_dispatch._set_worker_pid(conn, tid2, live, scope_unit="")
    row2 = conn.execute(
        "SELECT worker_registered_at FROM tasks WHERE id = ?", (tid2,)
    ).fetchone()
    assert row2["worker_registered_at"] is not None


def test_unregistered_past_grace_fails_as_spawn_failed(shims, conn):
    """A scoped run that never registered past the grace window is a silent
    launch failure: spawn_failed run, failure counted, scope stopped."""
    pid = shims.sleeper()
    unit = _kanban_worker_scope._kanban_worker_scope_unit("t_deadspawn", 1)
    shims.write_unit(unit, [pid])
    # The launcher pid is recorded (as every scoped spawn does) but the
    # worker never registered; the run started long before the grace
    # window closed.
    tid = _scoped_task_row(
        conn, scope=unit, pid=os.getpid(),
        started_delta=-_kanban_worker_scope.WORKER_REGISTRATION_GRACE_SECONDS - 300,
    )
    failed = _kanban_worker_recovery.fail_unregistered_workers(conn)
    assert failed == [tid]
    row = conn.execute(
        "SELECT status, consecutive_failures FROM tasks WHERE id = ?",
        (tid,),
    ).fetchone()
    assert row["status"] != "running"
    assert row["consecutive_failures"] >= 1
    run = _kanban_stats.latest_run(conn, tid)
    assert run is not None and run.outcome == "spawn_failed"


def test_breaker_row_with_live_scope_is_never_spawnable(
    shims, conn, kanban_home,
):
    """Pass 12 (AQ): a breaker row that still carries a LIVE worker scope
    is never spawnable — recompute_ready leaves it blocked, unblock_task
    refuses (the scope is alive, not verified dead), and the dispatch
    spawn loop skips a scope-carrying row even if one lands in ready."""
    _spawnable_profile(kanban_home)
    straggler = shims.sleeper()
    unit = _kanban_worker_scope._kanban_worker_scope_unit("t_aqlive", 1)
    shims.write_unit(unit, [straggler])
    tid = _breaker_shaped_row(conn, unit)

    # recompute_ready: no promotion beside a live scope (the row carries
    # no sticky 'blocked' event and no failures — without the skip this
    # is exactly the row the old code auto-promoted).
    assert _kanban_transitions.recompute_ready(conn) == 0
    row = conn.execute(
        "SELECT status, worker_scope FROM tasks WHERE id=?", (tid,),
    ).fetchone()
    assert row["status"] == "blocked"
    assert row["worker_scope"] == unit

    # unblock refuses: the scope is still alive, so the row must not
    # become spawnable by hand either.
    assert _kanban_transitions.unblock_task(conn, tid) is False
    row = conn.execute(
        "SELECT status, worker_scope FROM tasks WHERE id=?", (tid,),
    ).fetchone()
    assert row["status"] == "blocked"
    assert row["worker_scope"] == unit

    # Spawn-loop invariant: even a scope-carrying row parked in ready
    # (manual SQL, a racy writer) is skipped, never claimed or spawned.
    conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    conn.commit()
    spawns: list[str] = []

    def never_spawn(task, workspace, board):
        spawns.append(task.id)
        return None

    result = _kanban_db_dispatch._dispatch_once_locked(conn, spawn_fn=never_spawn)

    assert spawns == []
    assert result.skipped_scope_live == [(tid, unit)]
    assert result.spawned == []
    row = conn.execute(
        "SELECT status, claim_lock, worker_scope FROM tasks WHERE id=?",
        (tid,),
    ).fetchone()
    assert row["status"] == "ready"      # untouched: never claimed
    assert row["claim_lock"] is None
    assert row["worker_scope"] == unit


def test_breaker_row_dead_scope_unblocks_and_spawns_once(
    shims, conn, kanban_home,
):
    """Pass 12 (AQ), recovery by unblock: once the scope is verified dead,
    unblock_task verifies the death itself, clears the stale pointer,
    succeeds, and the dispatcher spawns exactly one worker for the row."""
    _spawnable_profile(kanban_home)
    # Retained loaded unit with verified empty kernel membership. Manager
    # absence alone cannot unblock a row with a persisted scope receipt.
    dead_unit = _kanban_worker_scope._kanban_worker_scope_unit("t_aqdead", 1)
    shims.write_unit(dead_unit, [])
    tid = _breaker_shaped_row(conn, dead_unit)

    assert _kanban_transitions.unblock_task(conn, tid) is True
    row = conn.execute(
        "SELECT status, worker_scope FROM tasks WHERE id=?", (tid,),
    ).fetchone()
    assert row["status"] == "ready"
    assert row["worker_scope"] is None

    spawns: list[str] = []

    def stub_spawn(task, workspace, board):
        spawns.append(task.id)
        return None

    result = _kanban_db_dispatch._dispatch_once_locked(conn, spawn_fn=stub_spawn)

    assert spawns == [tid]
    assert [s[0] for s in result.spawned] == [tid]
    assert result.skipped_scope_live == []
    assert conn.execute(
        "SELECT worker_scope FROM tasks WHERE id=?", (tid,),
    ).fetchone()["worker_scope"] is None


def test_retry_spawn_uses_a_new_unique_unit(shims, conn, kanban_home):
    """A respawned attempt gets a DIFFERENT unit name (run-suffixed), so a
    lingering half-dead scope from the previous attempt can never collide —
    and the audit sweep stops the orphaned old unit."""
    _spawnable_profile(kanban_home)
    tid = kb.create_task(conn, title="retry", assignee="elias")
    assert _kanban_db_dispatch.dispatch_once(conn, dry_run=False).spawned
    row = conn.execute(
        "SELECT worker_scope, current_run_id FROM tasks WHERE id = ?",
        (tid,),
    ).fetchone()
    unit_a = row["worker_scope"]
    assert unit_a.endswith(f"-r{row['current_run_id']}.scope")

    # Operator reclaims (verified stop) → task returns to its source lane.
    assert _owner_kanban_claims.reclaim_task(conn, tid, reason="retry") is True

    # Second attempt: new run id → new unit name.
    assert _kanban_db_dispatch.dispatch_once(conn, dry_run=False).spawned
    row = conn.execute(
        "SELECT worker_scope, current_run_id, status FROM tasks "
        "WHERE id = ?", (tid,)
    ).fetchone()
    unit_b = row["worker_scope"]
    assert unit_b != unit_a
    assert unit_b.endswith(f"-r{row['current_run_id']}.scope")
    assert _kanban_worker_scope._task_id_from_kanban_scope_unit(unit_b) == tid


def test_spawn_failure_auto_falls_back_to_plain_spawn(shims, conn, kanban_home):
    """A refused systemd-run launch in 'auto' mode falls back to a plain
    spawn for THIS run (with a warning), records the real pid (which IS the
    worker for a plain spawn, so it counts as registered), and keeps the
    board moving."""
    _write_kanban_config(Path(kanban_home), "  worker_isolation: auto\n")
    _kanban_db_connect._INITIALIZED_PATHS.clear()
    _spawnable_profile(kanban_home)
    shims.arm_fail_next(1)
    tid = kb.create_task(conn, title="degraded", assignee="elias")

    result = _kanban_db_dispatch.dispatch_once(conn, dry_run=False)
    assert [s[0] for s in result.spawned] == [tid]
    row = conn.execute(
        "SELECT worker_pid, worker_scope, worker_registered_at "
        "FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row["worker_scope"] is None
    assert row["worker_pid"] is not None
    shims.track(row["worker_pid"])
    assert row["worker_registered_at"] is not None  # plain pid == worker
    assert result.late_spawn_failed == []


def test_spawn_failure_systemd_scope_mode_fails_loudly(shims, conn, kanban_home):
    """'systemd-scope' never degrades: a refused launch raises into the
    dispatcher's failure recording — spawn_failed run with the systemd-run
    stderr, failure counted, nothing spawned."""
    _write_kanban_config(Path(kanban_home), "  worker_isolation: systemd-scope\n")
    _kanban_db_connect._INITIALIZED_PATHS.clear()
    _spawnable_profile(kanban_home)
    shims.arm_fail_next(1)
    tid = kb.create_task(conn, title="strict", assignee="elias")

    result = _kanban_db_dispatch.dispatch_once(conn, dry_run=False)
    assert result.spawned == []
    assert result.auto_blocked == []
    row = conn.execute(
        "SELECT status, consecutive_failures, last_failure_error, "
        "       worker_pid, worker_scope FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row["status"] != "running"
    assert row["consecutive_failures"] == 1
    assert "systemd-run launch failed" in (row["last_failure_error"] or "")
    assert row["worker_pid"] is None
    assert row["worker_scope"] is None
    run = _kanban_stats.latest_run(conn, tid)
    assert run is not None and run.outcome == "spawn_failed"
    assert "user bus connection refused" in (run.error or "")


def test_strict_probe_unavailable_surfaces_spawn_failed(
    shims, conn, kanban_home, monkeypatch,
):
    """H: the strict-mode probe refusal must reach the TASK, not just the
    dispatcher log. With the probe itself unusable (macOS / user bus gone)
    the spawn refuses BEFORE launching anything; dispatch_once records
    spawn_failed with the operator-facing reason on the row and the run,
    and nothing is ever spawned."""
    _write_kanban_config(Path(kanban_home), "  worker_isolation: systemd-scope\n")
    _kanban_db_connect._INITIALIZED_PATHS.clear()
    _spawnable_profile(kanban_home)
    monkeypatch.setattr(
        'tools.process_registry_scope._systemd_run_user_scope_available',
        lambda: False,
    )
    tid = kb.create_task(conn, title="no bus", assignee="elias")

    result = _kanban_db_dispatch.dispatch_once(conn, dry_run=False)

    assert result.spawned == []
    assert result.auto_blocked == []
    row = conn.execute(
        "SELECT status, consecutive_failures, last_failure_error, "
        "       worker_pid, worker_scope FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row["status"] != "running"
    assert row["consecutive_failures"] == 1
    assert "worker_isolation=systemd-scope is configured but" in (
        row["last_failure_error"] or ""
    )
    assert "refusing to spawn task" in (row["last_failure_error"] or "")
    assert row["worker_pid"] is None
    assert row["worker_scope"] is None
    run = _kanban_stats.latest_run(conn, tid)
    assert run is not None and run.outcome == "spawn_failed"
    assert "refusing to spawn task" in (run.error or "")
    event = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND "
        "kind='spawn_failed' ORDER BY id DESC LIMIT 1", (tid,),
    ).fetchone()
    assert event is not None
    assert "refusing to spawn task" in (event["payload"] or "")


def test_spawn_failure_with_uncleanable_unit_refuses_fallback(
    shims, conn, kanban_home, monkeypatch,
):
    """A failed launch whose scope cleanup CANNOT be verified must not
    plain-spawn a replacement beside the possibly-live half-created unit
    — auto mode included. The dispatcher records spawn_failed with the
    refusal instead."""
    _write_kanban_config(Path(kanban_home), "  worker_isolation: auto\n")
    _kanban_db_connect._INITIALIZED_PATHS.clear()
    _spawnable_profile(kanban_home)
    shims.arm_fail_next(1)
    cleanup_calls: list[str] = []

    def unverifiable_stop(unit, **kwargs):
        cleanup_calls.append(unit)
        return False  # cleanup could not be verified (wedged stop job)

    monkeypatch.setattr(_kanban_worker_scope, "_stop_kanban_worker_scope", unverifiable_stop)
    tid = kb.create_task(conn, title="dirty launch", assignee="elias")

    result = _kanban_db_dispatch.dispatch_once(conn, dry_run=False)
    assert result.spawned == []
    assert cleanup_calls == [_kanban_worker_scope._kanban_worker_scope_unit(tid, 1)]
    row = conn.execute(
        "SELECT status, worker_pid, worker_scope, consecutive_failures, "
        "last_failure_error FROM tasks WHERE id = ?", (tid,),
    ).fetchone()
    assert row["status"] != "running"
    assert row["worker_pid"] is None  # no plain-spawn duplicate beside it
    assert row["consecutive_failures"] == 1
    assert "refusing to spawn a replacement" in (
        row["last_failure_error"] or ""
    )


def test_set_worker_pid_refuses_terminal_task(conn):
    """The status guard: a task that left 'running' before its spawn was
    recorded (fast worker completing mid-spawn, crash reclaim racing the
    spawn loop) must not get worker bookkeeping reattached — the row
    keeps its terminal state and no spawned event claims a live run."""
    tid = kb.create_task(conn, title="done already", assignee="w")
    _owner_kanban_claims.claim_task(conn, tid, claimer=kb._claimer_id())
    conn.execute(
        "UPDATE tasks SET status='ready', claim_lock=NULL, worker_pid=NULL, "
        "worker_scope=NULL WHERE id=?", (tid,),
    )
    conn.commit()

    _kanban_db_dispatch._set_worker_pid(conn, tid, os.getpid(), scope_unit="u.scope")

    row = conn.execute(
        "SELECT status, worker_pid, worker_scope FROM tasks WHERE id=?",
        (tid,),
    ).fetchone()
    assert row["status"] == "ready"
    assert row["worker_pid"] is None
    assert row["worker_scope"] is None
    events = conn.execute(
        "SELECT count(*) AS n FROM task_events "
        "WHERE task_id=? AND kind='spawned'", (tid,),
    ).fetchone()
    assert events["n"] == 0

from hermes_cli import kanban_db_connect as _kanban_db_connect
from hermes_cli import kanban_db_dispatch as _kanban_db_dispatch
from hermes_cli import kanban_stats as _kanban_stats
from hermes_cli import kanban_transitions as _kanban_transitions
from hermes_cli import kanban_worker_identity as _kanban_worker_identity
from hermes_cli import kanban_worker_recovery as _kanban_worker_recovery
from hermes_cli import kanban_worker_scope as _kanban_worker_scope
from hermes_cli import kanban_worker_spawn as _kanban_worker_spawn
