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


def test_release_stale_claims_applies_pending_own_worker_handoff(shims, conn):
    """Pass 9 (AF): the TTL sweep runs BEFORE crash detection, and a
    deferred own-worker handoff extends the claim only by one defer
    grace — a scope drain outlasting it used to fall into the generic
    stale reclaim here, silently dropping the worker's requested review
    transition. The stale path must re-extend the claim while the stop
    is in progress and apply the handoff itself once the scope is
    verified dead: the row lands in review with the payload intact and
    is never reclaimed."""
    straggler = shims.sleeper()  # descendant keeping the drain running
    unit = _kanban_worker_scope._kanban_worker_scope_unit("t_stalehandoff", 1)
    shims.write_unit(unit, [straggler])
    shims.arm_deactivating(unit)  # stop job mid-drain: never confirms
    tid = kb.create_task(conn, title="stale handoff", assignee="w")
    _deferred_handoff_row(
        shims,
        conn,
        unit,
        handoff={
            "handoff": "review_requested",
            "implementer": "w",
            "reviewer": "r",
            "summary": "done building",
            "metadata": {"pr": 42},
        },
        tid=tid,
    )

    # Tick 1: the grace expired but the drain is still in progress —
    # the claim is re-extended behind the marker, nothing reclaimed.
    assert _owner_kanban_claims.release_stale_claims(conn) == 0
    row = conn.execute(
        "SELECT status, claim_lock, claim_expires FROM tasks WHERE id=?",
        (tid,),
    ).fetchone()
    assert row["status"] == "running"
    assert row["claim_lock"] == kb._claimer_id()
    assert row["claim_expires"] > int(time.time())
    ext = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? "
        "AND kind='claim_extended' ORDER BY id DESC",
        (tid,),
    ).fetchone()
    assert ext is not None
    assert json.loads(ext["payload"])["reason"] == "own_worker_handoff_draining"

    # Tick 2: the drain completes — scope verified dead, so the sweep
    # applies the deferred handoff (row to review, payload intact),
    # never the generic reclaim.
    shims.clear_deactivating(unit)
    conn.execute(
        "UPDATE tasks SET claim_expires = ? WHERE id = ?",
        (int(time.time()) - 60, tid),
    )
    conn.commit()
    assert (
        _owner_kanban_claims.release_stale_claims(conn) == 0
    )  # applied, not reclaimed
    row = conn.execute(
        "SELECT status, assignee, claim_lock, worker_scope FROM tasks WHERE id=?",
        (tid,),
    ).fetchone()
    assert row["status"] == "review"
    assert row["assignee"] == "r"  # reviewer carried by the payload
    assert row["claim_lock"] is None
    assert row["worker_scope"] is None
    event = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? "
        "AND kind='review_requested' ORDER BY id DESC",
        (tid,),
    ).fetchone()
    payload = json.loads(event["payload"])
    assert payload["deferred_handoff"] is True
    assert payload["implementer"] == "w"
    assert payload["reviewer"] == "r"
    assert "done building" in (payload["summary"] or "")
    assert (
        conn.execute(
            "SELECT count(*) AS n FROM task_events WHERE task_id=? "
            "AND kind='reclaimed'",
            (tid,),
        ).fetchone()["n"]
        == 0
    )


def test_release_stale_claims_own_worker_handoff_ceiling_escalates_when_scope_alive(
    shims,
    conn,
):
    """Pass 10 (AJ): the drain ceiling must NEVER apply a handoff beside
    a scope the cgroup proves alive. The old ceiling behaviour cleared
    the claim/scope while the unit was active, making the row spawnable
    — the same dispatch tick then spawned a duplicate worker beside the
    live one, the exact duplication loop this branch exists to prevent.
    At the ceiling with a live scope the sweep instead ESCALATES: the
    queued verified stop keeps its SIGKILL escalation, one
    ``handoff_stop_escalated`` event per run records it, and the claim +
    marker survive for a later tick."""
    straggler = shims.stubborn_sleeper()
    unit = _kanban_worker_scope._kanban_worker_scope_unit("t_ceiling", 1)
    shims.write_unit(unit, [straggler])
    shims.arm_killproof(unit)  # stop job wedged server-side, forever
    tid = kb.create_task(conn, title="ceiling handoff", assignee="w")
    _deferred_handoff_row(
        shims,
        conn,
        unit,
        handoff={
            "handoff": "changes_requested",
            "implementer": "w",
            "reviewer": "r",
            "reason": "redo the edges",
        },
        tid=tid,
    )
    # No live worker pid and a stale heartbeat: the generic live-scope
    # extension stands down, so the handoff branch owns the row. Age the
    # marker past the drain ceiling.
    now = int(time.time())
    conn.execute(
        "UPDATE tasks SET worker_pid=NULL, worker_pid_started_at=NULL, "
        "last_heartbeat_at=? WHERE id=?",
        (now - 7200, tid),
    )
    conn.execute(
        "UPDATE task_events SET created_at=? "
        "WHERE task_id=? AND kind='own_worker_handoff'",
        (now - _kanban_worker_handoff._OWN_WORKER_HANDOFF_MAX_DRAIN_SECONDS - 1, tid),
    )
    conn.commit()

    assert (
        _owner_kanban_claims.release_stale_claims(conn) == 0
    )  # held, not applied/reclaimed
    row = conn.execute(
        "SELECT status, claim_lock, claim_expires, worker_scope FROM tasks WHERE id=?",
        (tid,),
    ).fetchone()
    # No spawnable write: the row stays a held running claim.
    assert row["status"] == "running"
    assert row["claim_lock"] == kb._claimer_id()
    assert row["claim_expires"] > int(time.time())
    assert row["worker_scope"] == unit
    assert (
        conn.execute(
            "SELECT count(*) AS n FROM task_events WHERE task_id=? "
            "AND kind IN ('changes_requested', 'review_requested', "
            "'reclaimed')",
            (tid,),
        ).fetchone()["n"]
        == 0
    )
    # The teardown escalated through the verified stop's SIGKILL path.
    actions = [s["action"] for s in shims.stops() if s["unit"] == unit]
    assert "stop" in actions and "kill" in actions
    # One escalation event per run — a second ceiling tick does not
    # duplicate it and still does not apply anything.
    assert (
        conn.execute(
            "SELECT count(*) AS n FROM task_events WHERE task_id=? "
            "AND kind='handoff_stop_escalated'",
            (tid,),
        ).fetchone()["n"]
        == 1
    )
    conn.execute(
        "UPDATE tasks SET claim_expires=? WHERE id=?",
        (int(time.time()) - 60, tid),
    )
    conn.commit()
    assert _owner_kanban_claims.release_stale_claims(conn) == 0
    assert (
        conn.execute(
            "SELECT count(*) AS n FROM task_events WHERE task_id=? "
            "AND kind='handoff_stop_escalated'",
            (tid,),
        ).fetchone()["n"]
        == 1
    )
    assert (
        conn.execute(
            "SELECT status FROM tasks WHERE id=?",
            (tid,),
        ).fetchone()["status"]
        == "running"
    )


def test_release_stale_claims_own_worker_handoff_applies_once_scope_dead_after_ceiling(
    shims,
    conn,
):
    """Pass 10 (AJ), the other half of the ceiling contract: the handoff
    applies exactly once, on a later tick, once the scope CONFIRMS dead
    — never before, no matter the marker's age."""
    straggler = shims.stubborn_sleeper()
    unit = _kanban_worker_scope._kanban_worker_scope_unit("t_ceildone", 1)
    shims.write_unit(unit, [straggler])
    shims.arm_killproof(unit)
    tid = kb.create_task(conn, title="ceiling then drain", assignee="w")
    _deferred_handoff_row(
        shims,
        conn,
        unit,
        handoff={
            "handoff": "changes_requested",
            "implementer": "w",
            "reviewer": "r",
            "reason": "redo the edges",
        },
        tid=tid,
    )
    now = int(time.time())
    conn.execute(
        "UPDATE tasks SET worker_pid=NULL, worker_pid_started_at=NULL, "
        "last_heartbeat_at=? WHERE id=?",
        (now - 7200, tid),
    )
    conn.execute(
        "UPDATE task_events SET created_at=? "
        "WHERE task_id=? AND kind='own_worker_handoff'",
        (now - _kanban_worker_handoff._OWN_WORKER_HANDOFF_MAX_DRAIN_SECONDS - 1, tid),
    )
    conn.commit()

    # Tick 1: ceiling with a live (killproof) scope — held, escalated.
    assert _owner_kanban_claims.release_stale_claims(conn) == 0
    assert (
        conn.execute(
            "SELECT status FROM tasks WHERE id=?",
            (tid,),
        ).fetchone()["status"]
        == "running"
    )

    # The wedged stop clears server-side and the straggler dies: the
    # cgroup is now verifiably empty.
    shims.clear_killproof(unit)
    os.kill(straggler, signal.SIGKILL)
    assert shims.wait_for(lambda: not _kanban_db_dispatch._pid_alive(straggler))
    conn.execute(
        "UPDATE tasks SET claim_expires=? WHERE id=?",
        (int(time.time()) - 60, tid),
    )
    conn.commit()

    # Tick 2: the scope confirms dead — the handoff applies ONCE, per
    # the deferred payload, and the generic reclaim never saw the row.
    assert (
        _owner_kanban_claims.release_stale_claims(conn) == 0
    )  # applied, not reclaimed
    row = conn.execute(
        "SELECT status, claim_lock, worker_scope FROM tasks WHERE id=?",
        (tid,),
    ).fetchone()
    assert row["status"] == "ready"  # changes_requested landed per payload
    assert row["claim_lock"] is None
    assert row["worker_scope"] is None
    events = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? "
        "AND kind='changes_requested' ORDER BY id",
        (tid,),
    ).fetchall()
    assert len(events) == 1
    payload = json.loads(events[0]["payload"])
    assert payload["deferred_handoff"] is True
    assert payload["reason"] == "redo the edges"
    assert (
        conn.execute(
            "SELECT count(*) AS n FROM task_events WHERE task_id=? "
            "AND kind='reclaimed'",
            (tid,),
        ).fetchone()["n"]
        == 0
    )


def test_release_stale_claims_handoff_ceiling_breaker_blocks_killproof_scope(
    shims,
    conn,
):
    """Pass 11 (AO): a drain the SIGKILL escalation cannot clear is
    bounded. After _OWN_WORKER_HANDOFF_DRAIN_BREAKER_TICKS consecutive
    ceiling ticks with the scope still not provably empty, the task
    blocks (needs_input) with the run ended and the scope kept on the
    row for the operator; the row is not spawnable, exactly one
    ``handoff_scope_stuck`` event names the unit and worker pid, and
    later ticks neither extend nor warn."""
    straggler = shims.stubborn_sleeper()
    unit = _kanban_worker_scope._kanban_worker_scope_unit("t_breaker", 1)
    shims.write_unit(unit, [straggler])
    shims.arm_killproof(unit)  # stop job wedged server-side, forever
    tid = kb.create_task(conn, title="breaker", assignee="w")
    _deferred_handoff_row(
        shims,
        conn,
        unit,
        handoff={
            "handoff": "changes_requested",
            "implementer": "w",
            "reviewer": "r",
            "reason": "redo the edges",
        },
        tid=tid,
    )
    now = int(time.time())
    # A live worker pid for the reason to name; the stale heartbeat
    # keeps the generic live-scope extension out of the way so the
    # handoff branch owns the row, and the marker is aged past the
    # drain ceiling so every tick is a ceiling tick.
    conn.execute(
        "UPDATE tasks SET worker_pid=?, worker_pid_started_at=?, "
        "last_heartbeat_at=? WHERE id=?",
        (
            straggler,
            _kanban_worker_identity._worker_pid_start_time(straggler),
            now - 7200,
            tid,
        ),
    )
    conn.execute(
        "UPDATE task_events SET created_at=? "
        "WHERE task_id=? AND kind='own_worker_handoff'",
        (now - _kanban_worker_handoff._OWN_WORKER_HANDOFF_MAX_DRAIN_SECONDS - 1, tid),
    )
    conn.commit()

    def tick():
        conn.execute(
            "UPDATE tasks SET claim_expires=? WHERE id=?",
            (int(time.time()) - 60, tid),
        )
        conn.commit()
        return _owner_kanban_claims.release_stale_claims(conn)

    # Ceiling ticks 1 and 2: escalated and held, the row stays running.
    assert tick() == 0
    assert tick() == 0
    assert (
        conn.execute(
            "SELECT status FROM tasks WHERE id=?",
            (tid,),
        ).fetchone()["status"]
        == "running"
    )

    # Ceiling tick 3: the breaker fires — blocked, not spawnable.
    assert tick() == 0
    row = conn.execute(
        "SELECT status, block_kind, claim_lock, claim_expires, "
        "worker_pid, worker_scope FROM tasks WHERE id=?",
        (tid,),
    ).fetchone()
    assert row["status"] == "blocked"
    assert row["block_kind"] == "needs_input"
    assert row["claim_lock"] is None
    assert row["claim_expires"] is None
    assert row["worker_pid"] is None
    assert row["worker_scope"] == unit  # kept for the operator
    # The run ended terminally.
    run = conn.execute(
        "SELECT status, outcome, ended_at FROM task_runs "
        "WHERE task_id=? ORDER BY id DESC LIMIT 1",
        (tid,),
    ).fetchone()
    assert run["status"] == "blocked"
    assert run["outcome"] == "blocked"
    assert run["ended_at"] is not None
    # One stuck event naming the unit and the worker pid.
    stuck = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? "
        "AND kind='handoff_scope_stuck'",
        (tid,),
    ).fetchall()
    assert len(stuck) == 1
    payload = json.loads(stuck[0]["payload"])
    assert payload["scope"] == unit
    assert payload["worker_pid"] == straggler
    assert "scope will not drain" in payload["reason"]
    assert unit in payload["reason"]
    assert str(straggler) in payload["reason"]
    # The escalation happened first (once), and no handoff or reclaim
    # ever applied.
    assert (
        conn.execute(
            "SELECT count(*) AS n FROM task_events WHERE task_id=? "
            "AND kind='handoff_stop_escalated'",
            (tid,),
        ).fetchone()["n"]
        == 1
    )
    assert (
        conn.execute(
            "SELECT count(*) AS n FROM task_events WHERE task_id=? "
            "AND kind IN ('changes_requested', 'review_requested', "
            "'reclaimed')",
            (tid,),
        ).fetchone()["n"]
        == 0
    )
    # Not spawnable: a claim against the blocked row fails.
    assert _owner_kanban_claims.claim_task(conn, tid, claimer="other:host:2") is None
    # Later ticks do nothing: no more extensions, no second event.
    assert tick() == 0
    assert (
        conn.execute(
            "SELECT count(*) AS n FROM task_events WHERE task_id=? "
            "AND kind='handoff_scope_stuck'",
            (tid,),
        ).fetchone()["n"]
        == 1
    )


def test_release_stale_claims_handoff_ceiling_dies_before_breaker_applies_once(
    shims,
    conn,
):
    """Pass 11 (AO), the breaker's other half: a scope that verifies
    dead before the third ceiling tick never trips it — the existing
    contract holds and the handoff applies exactly once, per payload."""
    straggler = shims.stubborn_sleeper()
    unit = _kanban_worker_scope._kanban_worker_scope_unit("t_breakdone", 1)
    shims.write_unit(unit, [straggler])
    shims.arm_killproof(unit)
    tid = kb.create_task(conn, title="breaker averted", assignee="w")
    _deferred_handoff_row(
        shims,
        conn,
        unit,
        handoff={
            "handoff": "changes_requested",
            "implementer": "w",
            "reviewer": "r",
            "reason": "redo the edges",
        },
        tid=tid,
    )
    now = int(time.time())
    conn.execute(
        "UPDATE tasks SET worker_pid=NULL, worker_pid_started_at=NULL, "
        "last_heartbeat_at=? WHERE id=?",
        (now - 7200, tid),
    )
    conn.execute(
        "UPDATE task_events SET created_at=? "
        "WHERE task_id=? AND kind='own_worker_handoff'",
        (now - _kanban_worker_handoff._OWN_WORKER_HANDOFF_MAX_DRAIN_SECONDS - 1, tid),
    )
    conn.commit()

    # Two ceiling ticks: escalated and held.
    for _ in range(2):
        conn.execute(
            "UPDATE tasks SET claim_expires=? WHERE id=?",
            (int(time.time()) - 60, tid),
        )
        conn.commit()
        assert _owner_kanban_claims.release_stale_claims(conn) == 0
    assert (
        conn.execute(
            "SELECT status FROM tasks WHERE id=?",
            (tid,),
        ).fetchone()["status"]
        == "running"
    )

    # The wedged stop clears server-side and the straggler dies before
    # the third ceiling tick: the scope is verifiably empty.
    shims.clear_killproof(unit)
    os.kill(straggler, signal.SIGKILL)
    assert shims.wait_for(lambda: not _kanban_db_dispatch._pid_alive(straggler))
    conn.execute(
        "UPDATE tasks SET claim_expires=? WHERE id=?",
        (int(time.time()) - 60, tid),
    )
    conn.commit()

    # Tick 3: the handoff applies once; the breaker never fires.
    assert _owner_kanban_claims.release_stale_claims(conn) == 0
    row = conn.execute(
        "SELECT status, claim_lock, worker_scope FROM tasks WHERE id=?",
        (tid,),
    ).fetchone()
    assert row["status"] == "ready"  # changes_requested landed per payload
    assert row["claim_lock"] is None
    assert row["worker_scope"] is None
    events = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? "
        "AND kind='changes_requested' ORDER BY id",
        (tid,),
    ).fetchall()
    assert len(events) == 1
    payload = json.loads(events[0]["payload"])
    assert payload["deferred_handoff"] is True
    assert payload["reason"] == "redo the edges"
    assert (
        conn.execute(
            "SELECT count(*) AS n FROM task_events WHERE task_id=? "
            "AND kind='handoff_scope_stuck'",
            (tid,),
        ).fetchone()["n"]
        == 0
    )


def test_release_stale_claims_handoff_reservation_rejects_late_heartbeat(
    shims,
    conn,
    monkeypatch,
):
    """After stop intent commits, an in-flight heartbeat cannot revoke it."""
    straggler = shims.sleeper()
    unit = _kanban_worker_scope._kanban_worker_scope_unit("t_casmiss", 1)
    shims.write_unit(unit, [straggler])
    shims.arm_deactivating(unit)  # drain in progress: the extension runs
    tid = kb.create_task(conn, title="cas miss", assignee="w")
    _deferred_handoff_row(
        shims,
        conn,
        unit,
        handoff={
            "handoff": "review_requested",
            "implementer": "w",
            "reviewer": "r",
            "summary": "done building",
        },
        tid=tid,
    )

    def stop_that_loses_to_a_heartbeat(unit_arg, *, task_id=None, **kw):
        # The interleaving under test: the worker's heartbeat lands
        # between the sweep's stale scan and the extension CAS.
        assert not _owner_kanban_claims.heartbeat_claim(conn, tid)
        return False  # stop not confirmed — the extension branch runs

    monkeypatch.setattr(
        _kanban_worker_stop,
        "request_worker_scope_stop",
        stop_that_loses_to_a_heartbeat,
    )
    signalled: list[int] = []

    def record_signal(pid, sig):
        signalled.append((pid, sig))

    assert _owner_kanban_claims.release_stale_claims(conn, signal_fn=record_signal) == 0

    row = conn.execute(
        "SELECT status, claim_lock, claim_expires FROM tasks WHERE id=?",
        (tid,),
    ).fetchone()
    assert row["status"] == "running"
    assert row["claim_lock"] == kb._claimer_id()
    # Unconfirmed stop keeps the run held, with the bounded defer grace.
    assert (
        int(time.time())
        < row["claim_expires"]
        <= int(time.time()) + kb.RECLAIM_DEFER_GRACE_SECONDS
    )
    assert signalled == [], "a heartbeat-refreshed claim is never signalled"
    assert _kanban_db_dispatch._pid_alive(straggler), "nothing of the run was signalled"
    assert (
        conn.execute(
            "SELECT count(*) AS n FROM task_events WHERE task_id=? "
            "AND kind IN ('reclaimed', 'review_requested')",
            (tid,),
        ).fetchone()["n"]
        == 0
    )


from hermes_cli import kanban_db_dispatch as _kanban_db_dispatch
from hermes_cli import kanban_worker_identity as _kanban_worker_identity
from hermes_cli import kanban_worker_scope as _kanban_worker_scope
from hermes_cli import kanban_worker_stop as _kanban_worker_stop

from hermes_cli import kanban_worker_handoff as _kanban_worker_handoff
