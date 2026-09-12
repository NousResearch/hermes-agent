"""T005 Slice A behavioral coverage for active-router Kanban dispatch."""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import fields
from pathlib import Path

import pytest

from agent.execution_router import (
    CONTRACT_VERSION,
    ExecutionKind,
    ExecutionRouteDecisionKind,
    ExecutionRouteDecisionV1,
    ExecutionRouteEventType,
    ExecutionRouteEventV1,
    ExecutionRouteIdentityV1,
    ExecutionRouterProviderDescriptorV1,
    MAX_REASON_TEXT_UTF8_BYTES,
    canonical_json_bytes,
)
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.execution_router_runtime import ExecutionRouterRegistration


@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    with kbc.connect() as connection:
        yield connection


def _registration(callback):
    class Provider:
        descriptor = ExecutionRouterProviderDescriptorV1(
            plugin_id="router-test",
            plugin_version="1.0.0",
            provider_id="router-test",
            contract_version=CONTRACT_VERSION,
            supported_execution_kinds=tuple(ExecutionKind),
        )

        def resolve_execution_route(self, request, cancellation):
            return callback(request)

    return ExecutionRouterRegistration(Provider(), 1, lambda generation: generation == 1)


def _validated_public_event(raw_payload: str) -> ExecutionRouteEventV1:
    payload = json.loads(raw_payload)
    assert set(payload) == {field.name for field in fields(ExecutionRouteEventV1)}
    assert "router_plugin_version" not in payload
    payload["execution_kind"] = ExecutionKind(payload["execution_kind"])
    payload["event_type"] = ExecutionRouteEventType(payload["event_type"])
    if payload["decision_state"] is not None:
        payload["decision_state"] = ExecutionRouteDecisionKind(payload["decision_state"])
    for name in ("accepted_route", "actual_route"):
        if payload[name] is not None:
            payload[name] = ExecutionRouteIdentityV1(**payload[name])
    event = ExecutionRouteEventV1(**payload)
    assert raw_payload == canonical_json_bytes(event).decode("utf-8")
    return event


def _route_metadata(task_id: str, *, state: str = "route") -> dict:
    route = {
        "candidate_id": "native",
        "provider": "native-provider",
        "model": "native-model",
        "reasoning": None,
    }
    return {
        "state": state,
        **route,
        "accepted_route": route if state == "route" else None,
        "event_context": {
            "root_id": "root-old",
            "execution_id": "execution-old",
            "attempt_id": "attempt-old",
            "request_id": "request-old",
            "previous_attempt_id": None,
            "execution_kind": "kanban_worker",
            "surface_class": "kanban_dispatcher",
            "router_plugin_id": "router-test",
            "router_provider_id": "router-test",
            "router_contract_version": CONTRACT_VERSION,
        },
    }


def test_route_events_use_the_exact_public_envelope_and_child_actual_route(conn, monkeypatch):
    """FR-014/015 and ARC-004: requested, accepted, actual and terminal stay distinct."""
    task_id = kb.create_task(
        conn,
        title="exact envelope",
        assignee="worker",
        provider_override="accepted-provider",
        model_override="accepted-model",
    )
    registration = _registration(lambda request: ExecutionRouteDecisionV1.route(
        request_id=request.request_id,
        attempt_id=request.attempt_id,
        candidate_id="native",
        reason_code="selected",
    ))
    from hermes_cli.plugins import get_plugin_manager
    monkeypatch.setattr(
        get_plugin_manager(), "get_execution_router_registration", lambda: registration,
    )
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: None)

    kbd.dispatch_once(conn, spawn_fn=lambda *args, **kwargs: 4242)
    task = kb.get_task(conn, task_id)
    assert task is not None and task.current_run_id is not None and task.claim_lock
    run_id = task.current_run_id
    assert kb.record_routed_run_started(
        conn,
        task_id,
        run_id,
        task.claim_lock,
        provider="actual-provider",
        model="actual-model",
        reasoning="high",
    )
    assert kb.complete_task(conn, task_id, summary="done")

    rows = conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id = ? "
        "AND kind LIKE 'route_%' ORDER BY id",
        (task_id,),
    ).fetchall()
    events = [_validated_public_event(row["payload"]) for row in rows]
    assert [event.event_type.value for event in events] == [
        "route_requested", "route_accepted", "route_started", "route_finished",
    ]
    assert [event.sequence for event in events] == [1, 2, 3, 4]
    assert len({event.event_id for event in events}) == 4
    assert all(event.task_id == task_id for event in events)
    assert len({event.root_id for event in events}) == 1
    assert len({event.execution_id for event in events}) == 1
    assert len({event.attempt_id for event in events}) == 1
    assert len({event.request_id for event in events}) == 1
    assert all(event.timestamp_utc_ms > 0 for event in events)
    requested, accepted, started, finished = events
    assert requested.request_digest and requested.instruction_digest and requested.eligibility_revision
    assert all(
        event.request_digest is event.instruction_digest is event.eligibility_revision is None
        for event in events[1:]
    )
    expected_accepted = ExecutionRouteIdentityV1(
        "native", "accepted-provider", "accepted-model", None,
    )
    expected_actual = ExecutionRouteIdentityV1(
        "native", "actual-provider", "actual-model", "high",
    )
    assert accepted.accepted_route == expected_accepted and accepted.actual_route is None
    assert started.accepted_route == expected_accepted and started.actual_route == expected_actual
    assert finished.accepted_route == expected_accepted and finished.actual_route == expected_actual
    assert finished.terminal_state == "done"
    previous = kb.previous_terminal_routed_attempt(conn, task_id)
    assert previous is not None and previous.route == expected_actual


def test_active_router_reserves_then_atomically_opens_the_canonical_run(conn, monkeypatch):
    """T005 S001-S004: route only after same-lane reservation, then bind run/evidence."""
    task_id = kb.create_task(
        conn,
        title="route me",
        body="bounded worker instruction",
        assignee="worker",
        workspace_kind="scratch",
        model_override="native-model",
        provider_override="native-provider",
    )
    observed = {}

    def decide(request):
        with kbc.connect() as callback_conn:
            task = kb.get_task(callback_conn, task_id)
            assert task is not None
            observed.update(
                status=task.status,
                claim_lock=task.claim_lock,
                claim_expires=task.claim_expires,
                run_count=callback_conn.execute(
                    "SELECT COUNT(*) FROM task_runs WHERE task_id = ?", (task_id,)
                ).fetchone()[0],
                kind=request.execution_kind,
            )
        return ExecutionRouteDecisionV1.route(
            request_id=request.request_id,
            attempt_id=request.attempt_id,
            candidate_id="native",
        )

    registration = _registration(decide)
    from hermes_cli.plugins import get_plugin_manager
    monkeypatch.setattr(
        get_plugin_manager(), "get_execution_router_registration", lambda: registration,
    )
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: None)
    spawned = []

    def spawn(task, workspace, board=None):
        route_kinds = [row[0] for row in conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? AND kind LIKE 'route_%' ORDER BY id",
            (task_id,),
        ).fetchall()]
        assert route_kinds == ["route_requested", "route_accepted"]
        spawned.append(task)
        return 4242

    result = kbd.dispatch_once(conn, spawn_fn=spawn)

    assert observed, (result, kb.get_task(conn, task_id))
    assert observed["status"] == "ready"
    assert observed["claim_lock"]
    assert observed["claim_expires"]
    assert observed["run_count"] == 0
    assert observed["kind"] is ExecutionKind.KANBAN_WORKER
    assert [item.id for item in spawned] == [task_id]
    assert result.spawned[0][0] == task_id

    task = kb.get_task(conn, task_id)
    assert task is not None
    assert task.status == "running"
    run = conn.execute(
        "SELECT id, metadata FROM task_runs WHERE task_id = ?", (task_id,)
    ).fetchone()
    assert run["id"] == task.current_run_id
    route = json.loads(run["metadata"])["execution_router"]
    assert route["state"] == "route"
    assert route["candidate_id"] == "native"
    events = conn.execute(
        "SELECT kind, run_id, payload FROM task_events WHERE task_id = ? ORDER BY id", (task_id,)
    ).fetchall()
    route_events = [row for row in events if row["kind"].startswith("route_")]
    assert [row["kind"] for row in route_events] == [
        "route_requested",
        "route_accepted",
    ]
    assert route_events[1]["run_id"] == run["id"]
    requested = json.loads(route_events[0]["payload"])
    assert requested["request_digest"]
    assert requested["instruction_digest"]
    assert requested["eligibility_revision"]


@pytest.mark.parametrize("failure_at", ["workspace", "spawn"])
def test_admitted_route_failure_before_worker_start_never_records_started(
    conn, monkeypatch, failure_at,
):
    """T005 S003/S004: admitted pre-start failures close without a fabricated start."""
    task_id = kb.create_task(conn, title="fail before start", assignee="worker")
    registration = _registration(lambda request: ExecutionRouteDecisionV1.route(
        request_id=request.request_id,
        attempt_id=request.attempt_id,
        candidate_id="native",
    ))
    from hermes_cli.plugins import get_plugin_manager
    monkeypatch.setattr(
        get_plugin_manager(), "get_execution_router_registration", lambda: registration,
    )
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: None)
    if failure_at == "workspace":
        monkeypatch.setattr(
            kbd._kbw, "resolve_workspace",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("workspace unavailable")),
        )

    def spawn(*args, **kwargs):
        raise RuntimeError("spawn unavailable")

    result = kbd.dispatch_once(conn, spawn_fn=spawn)

    assert result.spawned == []
    task = kb.get_task(conn, task_id)
    assert task is not None
    assert task.status == "ready"
    run = conn.execute(
        "SELECT outcome, ended_at FROM task_runs WHERE task_id = ?", (task_id,),
    ).fetchone()
    assert run["outcome"] == "spawn_failed"
    assert run["ended_at"] is not None
    route_rows = conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id = ? AND kind LIKE 'route_%' ORDER BY id",
        (task_id,),
    ).fetchall()
    assert [row["kind"] for row in route_rows] == [
        "route_requested", "route_accepted", "route_not_started",
    ]
    assert all(_validated_public_event(row["payload"]) for row in route_rows)


@pytest.mark.parametrize("outcome", ["stop", "router_error"])
def test_stop_or_router_error_keeps_the_lane_and_defers_without_a_run(
    conn, monkeypatch, outcome,
):
    """T005 S003-S004: not-started outcomes retain a short claim without budget use."""
    task_id = kb.create_task(conn, title="defer me", assignee="worker")
    raw_secret = "sk-t005-review-secret-raw-value"

    def decide(request):
        if outcome == "router_error":
            raise RuntimeError(f"Authorization: Bearer {raw_secret}")
        return ExecutionRouteDecisionV1.stop(
            request_id=request.request_id,
            attempt_id=request.attempt_id,
            reason_code="policy_stop",
            reason_text=f"Authorization: Bearer {raw_secret}",
        )

    registration = _registration(decide)
    from hermes_cli.plugins import get_plugin_manager
    monkeypatch.setattr(
        get_plugin_manager(), "get_execution_router_registration", lambda: registration,
    )
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: None)
    spawned = []
    before = int(time.time())

    result = kbd.dispatch_once(
        conn, spawn_fn=lambda *args, **kwargs: spawned.append(args) or 4242,
    )

    task = kb.get_task(conn, task_id)
    assert task is not None
    assert task.status == "ready"
    assert task.claim_lock
    assert task.claim_expires is not None
    assert before < task.claim_expires <= before + 60
    assert task.current_run_id is None
    assert task.consecutive_failures == 0
    assert spawned == []
    assert result.spawned == []
    assert conn.execute(
        "SELECT COUNT(*) FROM task_runs WHERE task_id = ?", (task_id,)
    ).fetchone()[0] == 0
    event = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'route_not_started'",
        (task_id,),
    ).fetchone()
    route_event = _validated_public_event(event["payload"])
    assert route_event.decision_state.value == outcome
    assert len(route_event.reason_text.encode("utf-8")) <= MAX_REASON_TEXT_UTF8_BYTES
    assert raw_secret not in route_event.reason_text
    assert raw_secret not in event["payload"]


def test_expired_reserved_task_is_released_without_a_phantom_run(conn):
    """T005 S002/S005: recovery releases only routing-owned pre-run claims."""
    task_id = kb.create_task(conn, title="recover me", assignee="worker")
    foreign_id = kb.create_task(conn, title="foreign claim", assignee="worker")
    reserved = kb.reserve_task_for_routing(
        conn, task_id, "ready", ttl_seconds=1, claimer="router-reservation",
    )
    assert reserved is not None
    conn.execute(
        "UPDATE tasks SET claim_expires = ? WHERE id = ?",
        (int(time.time()) - 1, task_id),
    )
    conn.execute(
        "UPDATE tasks SET claim_lock = ?, claim_expires = ? WHERE id = ?",
        ("ordinary-control-plane", int(time.time()) - 1, foreign_id),
    )
    conn.commit()

    reclaimed = kb.release_stale_claims(conn)

    task = kb.get_task(conn, task_id)
    assert task is not None
    assert reclaimed == 1
    assert task.status == "ready"
    assert task.claim_lock is None
    assert task.claim_expires is None
    assert task.current_run_id is None
    assert conn.execute(
        "SELECT COUNT(*) FROM task_runs WHERE task_id = ?", (task_id,)
    ).fetchone()[0] == 0
    event = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'route_recovered'",
        (task_id,),
    ).fetchone()
    assert json.loads(event["payload"])["source_status"] == "ready"
    foreign = kb.get_task(conn, foreign_id)
    assert foreign is not None
    assert foreign.claim_lock == "ordinary-control-plane"
    assert conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = 'route_recovered'",
        (foreign_id,),
    ).fetchone()[0] == 0


@pytest.mark.parametrize(
    ("claimer", "expected_local", "expected_reclaimed"),
    [(None, True, 0), ("foreign-host:987", False, 1)],
)
def test_routed_run_claim_keeps_host_ownership_semantics_after_open(
    conn, monkeypatch, claimer, expected_local, expected_reclaimed,
):
    """T005 S005/S006: routed locks retain native local/foreign reclaim ownership."""
    task_id = kb.create_task(conn, title="routed ownership", assignee="worker")
    reserved = kb.reserve_task_for_routing(
        conn, task_id, "ready", ttl_seconds=60, claimer=claimer,
    )
    assert reserved is not None
    assert reserved.claim_lock is not None
    opened = kb.open_reserved_routed_run(
        conn,
        task_id,
        "ready",
        reserved.claim_lock,
        "request-old",
        "attempt-old",
        _route_metadata(task_id),
    )
    assert opened is not None
    conn.execute(
        "UPDATE tasks SET worker_pid = ?, claim_expires = ? WHERE id = ?",
        (4242, int(time.time()) - 1, task_id),
    )
    conn.commit()

    signals = []
    monkeypatch.setattr(kbd, "_poll_worker_exit", lambda _pid: True)
    ownership = kbd._terminate_reclaimed_worker(
        4242,
        reserved.claim_lock,
        signal_fn=lambda pid, sig: signals.append((pid, sig)),
    )
    assert ownership["host_local"] is expected_local
    assert bool(signals) is expected_local

    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: True)
    reclaimed = kb.release_stale_claims(conn, signal_fn=lambda *_args: None)

    task = kb.get_task(conn, task_id)
    assert task is not None
    assert reclaimed == expected_reclaimed
    assert task.status == ("running" if expected_local else "ready")
    assert (task.claim_lock is not None) is expected_local


@pytest.mark.parametrize("lane", ["ready", "review"])
def test_active_router_reclaims_a_dangling_run_before_reservation(
    conn, monkeypatch, lane,
):
    """T005 S002: native dangling-run recovery precedes routed reservation/open."""
    task_id = kb.create_task(conn, title="recover dangling run", assignee="worker")
    original = kb.claim_task(conn, task_id)
    assert original is not None
    leaked_run_id = original.current_run_id
    conn.execute(
        "UPDATE tasks SET status = ?, claim_lock = NULL, claim_expires = NULL, worker_pid = NULL "
        "WHERE id = ?",
        (lane, task_id),
    )
    conn.commit()
    registration = _registration(lambda request: ExecutionRouteDecisionV1.pass_through(
        request_id=request.request_id,
        attempt_id=request.attempt_id,
        reason_code="native",
    ))
    from hermes_cli.plugins import get_plugin_manager
    monkeypatch.setattr(
        get_plugin_manager(), "get_execution_router_registration", lambda: registration,
    )
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: None)

    result = kbd.dispatch_once(conn, spawn_fn=lambda *args, **kwargs: 4242)

    assert [item[0] for item in result.spawned] == [task_id]
    task = kb.get_task(conn, task_id)
    assert task is not None
    assert task.status == "running"
    assert task.current_run_id != leaked_run_id
    leaked = conn.execute(
        "SELECT outcome, ended_at FROM task_runs WHERE id = ?", (leaked_run_id,),
    ).fetchone()
    assert leaked["outcome"] == "reclaimed"
    assert leaked["ended_at"] is not None


def test_route_selects_an_existing_fallback_candidate_without_changing_task_pins(
    conn, monkeypatch,
):
    """T005 S003/S004: an accepted candidate drives only this spawned attempt."""
    home = Path(kb.kanban_home())
    profile_home = home / "profiles" / "worker"
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text(
        "model: native-model\n"
        "fallback_model:\n"
        "  provider: fallback-provider\n"
        "  model: fallback-model\n"
        "  reasoning_effort: high\n",
        encoding="utf-8",
    )
    task_id = kb.create_task(conn, title="choose route", assignee="worker")

    def decide(request):
        assert [candidate.candidate_id for candidate in request.eligible_candidates] == [
            "native", "fallback-0",
        ]
        return ExecutionRouteDecisionV1.route(
            request_id=request.request_id,
            attempt_id=request.attempt_id,
            candidate_id="fallback-0",
        )

    registration = _registration(decide)
    from hermes_cli.plugins import get_plugin_manager
    monkeypatch.setattr(
        get_plugin_manager(), "get_execution_router_registration", lambda: registration,
    )
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: None)
    spawned = []

    kbd.dispatch_once(
        conn, spawn_fn=lambda task, workspace, board=None: spawned.append(task) or 4242,
    )

    assert len(spawned) == 1
    assert spawned[0].provider_override == "fallback-provider"
    assert spawned[0].model_override == "fallback-model"
    assert spawned[0].reasoning_effort == "high"
    persisted = kb.get_task(conn, task_id)
    assert persisted is not None
    assert persisted.provider_override is None
    assert persisted.model_override is None
    assert persisted.reasoning_effort is None


@pytest.mark.parametrize("close_path", ["completion", "spawn_failure", "reclaim"])
def test_route_binding_survives_existing_run_close_metadata_updates(
    conn, monkeypatch, close_path,
):
    """T005 S004: the canonical close owner preserves durable route binding."""
    task_id = kb.create_task(conn, title="preserve route", assignee="worker")
    registration = _registration(lambda request: ExecutionRouteDecisionV1.route(
        request_id=request.request_id,
        attempt_id=request.attempt_id,
        candidate_id="native",
    ))
    from hermes_cli.plugins import get_plugin_manager
    monkeypatch.setattr(
        get_plugin_manager(), "get_execution_router_registration", lambda: registration,
    )
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: None)

    def spawn(*args, **kwargs):
        if close_path == "spawn_failure":
            raise RuntimeError("spawn failed")
        return None

    kbd.dispatch_once(conn, spawn_fn=spawn)
    if close_path == "completion":
        assert kb.complete_task(conn, task_id, summary="done", metadata={"result": "ok"})
    elif close_path == "reclaim":
        assert kb.reclaim_task(conn, task_id, reason="operator")

    run = conn.execute(
        "SELECT metadata FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1", (task_id,),
    ).fetchone()
    metadata = json.loads(run["metadata"])
    assert metadata["execution_router"]["candidate_id"] == "native"
    if close_path == "completion":
        assert metadata["result"] == "ok"


def test_review_pass_through_opens_from_the_reserved_review_lane(conn, monkeypatch):
    """T005 S002-S004: active pass-through preserves review identity and evidence."""
    task_id = kb.create_task(conn, title="review natively", assignee="reviewer")
    conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (task_id,))
    conn.commit()
    observed = {}

    def decide(request):
        with kbc.connect() as callback_conn:
            task = kb.get_task(callback_conn, task_id)
            assert task is not None
            observed["status"] = task.status
            observed["run_id"] = task.current_run_id
        return ExecutionRouteDecisionV1.pass_through(
            request_id=request.request_id,
            attempt_id=request.attempt_id,
            reason_code="keep_native",
        )

    registration = _registration(decide)
    from hermes_cli.plugins import get_plugin_manager
    monkeypatch.setattr(
        get_plugin_manager(), "get_execution_router_registration", lambda: registration,
    )
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: None)
    spawned = []

    kbd.dispatch_once(
        conn, spawn_fn=lambda task, workspace, board=None: spawned.append(task) or 4242,
    )

    assert observed == {"status": "review", "run_id": None}
    assert len(spawned) == 1
    assert "sdlc-review" in spawned[0].skills
    run = conn.execute(
        "SELECT metadata FROM task_runs WHERE task_id = ?", (task_id,)
    ).fetchone()
    evidence = json.loads(run["metadata"])["execution_router"]
    assert evidence["state"] == "pass_through"
    assert evidence["reason_code"] == "keep_native"
    task = kb.get_task(conn, task_id)
    assert task is not None and task.current_run_id is not None and task.claim_lock
    assert kb.record_routed_run_started(
        conn,
        task_id,
        task.current_run_id,
        task.claim_lock,
        provider="native-provider",
        model="native-model",
        reasoning=None,
    )
    route_rows = conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id = ? AND kind LIKE 'route_%' ORDER BY id",
        (task_id,),
    ).fetchall()
    assert [row["kind"] for row in route_rows] == ["route_requested", "route_started"]
    started = _validated_public_event(route_rows[-1]["payload"])
    assert started.decision_state is ExecutionRouteDecisionKind.PASS_THROUGH
    assert started.accepted_route is None
    assert started.actual_route == ExecutionRouteIdentityV1(
        "native", "native-provider", "native-model", None,
    )
    claimed = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'claimed'",
        (task_id,),
    ).fetchone()
    assert json.loads(claimed["payload"])["source_status"] == "review"


def test_active_router_reservation_preserves_parent_validation(conn, monkeypatch):
    """T005 S001/S002: routing reservation keeps the existing dependency gate."""
    parent = kb.create_task(conn, title="parent", assignee="worker")
    child = kb.create_task(conn, title="child", assignee="worker")
    kb.link_tasks(conn, parent, child)
    conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (child,))
    conn.commit()
    callbacks = []
    registration = _registration(lambda request: callbacks.append(request.task_id))
    from hermes_cli.plugins import get_plugin_manager
    monkeypatch.setattr(
        get_plugin_manager(), "get_execution_router_registration", lambda: registration,
    )
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: None)

    kbd.dispatch_once(conn, spawn_fn=lambda *args, **kwargs: 4242)

    task = kb.get_task(conn, child)
    assert task is not None
    assert task.status == "todo"
    assert task.claim_lock is None
    assert child not in callbacks


def test_no_router_keeps_native_claim_open_workspace_spawn_error_path(conn, monkeypatch):
    """T005 S001: disabled routing preserves native ordering and outward failure."""
    task_id = kb.create_task(conn, title="native baseline", assignee="worker")
    from hermes_cli.plugins import get_plugin_manager
    monkeypatch.setattr(
        get_plugin_manager(), "get_execution_router_registration", lambda: None,
    )
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: None)
    ordering = []
    resolve_workspace = kbd._kbw.resolve_workspace

    def resolve(task, *, board=None):
        persisted = kb.get_task(conn, task_id)
        assert persisted is not None
        assert persisted.status == "running"
        assert persisted.current_run_id is not None
        ordering.append("workspace")
        return resolve_workspace(task, board=board)

    def spawn(task, workspace, board=None):
        ordering.append("spawn")
        persisted = kb.get_task(conn, task_id)
        assert persisted is not None
        assert persisted.status == "running"
        raise RuntimeError("native spawn failure")

    monkeypatch.setattr(kbd._kbw, "resolve_workspace", resolve)
    result = kbd.dispatch_once(conn, spawn_fn=spawn)

    assert ordering == ["workspace", "spawn"]
    assert result.spawned == []
    task = kb.get_task(conn, task_id)
    assert task is not None
    assert task.status == "ready"
    assert task.claim_lock is None
    assert task.claim_expires is None
    assert task.current_run_id is None
    run = conn.execute(
        "SELECT outcome, error, metadata FROM task_runs WHERE task_id = ?", (task_id,),
    ).fetchone()
    assert run["outcome"] == "spawn_failed"
    assert run["error"] == "native spawn failure"
    assert json.loads(run["metadata"]) == {"failures": 1, "retry_status": "ready"}
    assert conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind LIKE 'route_%'", (task_id,),
    ).fetchone()[0] == 0


@pytest.mark.parametrize(
    ("router_outcome", "expected_routed"),
    [("route", True), ("pass_through", False), ("no_router", False)],
)
def test_only_route_selected_worker_attempt_reaches_the_agent_fallback_fence(
    conn, monkeypatch, router_outcome, expected_routed,
):
    """T005 S001/S006: the private fence marks only the selected worker attempt."""
    task_id = kb.create_task(conn, title="worker fence", assignee="worker")
    from hermes_cli.plugins import get_plugin_manager

    if router_outcome == "no_router":
        registration = None
    elif router_outcome == "route":
        registration = _registration(lambda request: ExecutionRouteDecisionV1.route(
            request_id=request.request_id,
            attempt_id=request.attempt_id,
            candidate_id="native",
        ))
    else:
        registration = _registration(lambda request: ExecutionRouteDecisionV1.pass_through(
            request_id=request.request_id,
            attempt_id=request.attempt_id,
            reason_code="native",
        ))
    monkeypatch.setattr(
        get_plugin_manager(), "get_execution_router_registration", lambda: registration,
    )
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: None)
    spawned = []
    kbd.dispatch_once(
        conn,
        spawn_fn=lambda task, workspace, board=None: spawned.append((task, workspace)) or 4242,
    )
    assert len(spawned) == 1
    spawned_task, workspace = spawned[0]
    assert bool(getattr(spawned_task, "_execution_router_selected_attempt", False)) is expected_routed

    captured = {}

    class FakeProc:
        pid = 4243

    def fake_popen(*args, **kwargs):
        captured["env"] = dict(kwargs["env"])
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(kbd, "_resolve_hermes_argv", lambda: ["hermes"])
    monkeypatch.setattr(kbd, "_restart_safe_worker_argv", lambda _task, command: command)
    monkeypatch.setattr(kbd, "_resolve_worker_cli_toolsets", lambda _home: None)
    monkeypatch.setattr(kbd, "_retag_legacy_worker_sessions", lambda _root: None)
    kbd._default_spawn(spawned_task, workspace)

    marker = "HERMES_KANBAN_ROUTED_ATTEMPT"
    assert captured["env"].get(marker) == (
        router_outcome if router_outcome != "no_router" else None
    )
    for key, value in captured["env"].items():
        monkeypatch.setenv(key, value)
    from run_agent import AIAgent
    agent = AIAgent(
        api_key="test",
        base_url="http://127.0.0.1:1/v1",
        provider="openai-compat",
        model="test-model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        enabled_toolsets=[],
    )
    assert agent._execution_router_selected_attempt is expected_routed
    route_kinds = [row[0] for row in conn.execute(
        "SELECT kind FROM task_events WHERE task_id = ? AND kind LIKE 'route_%' ORDER BY id",
        (task_id,),
    ).fetchall()]
    assert route_kinds == (
        ["route_requested", "route_accepted", "route_started"]
        if router_outcome == "route"
        else ["route_requested", "route_started"]
        if router_outcome == "pass_through"
        else []
    )

    from agent.delegation_context import delegated_child_subprocess_env
    child_env = delegated_child_subprocess_env(captured["env"])
    assert child_env is not None
    assert marker not in child_env


@pytest.mark.parametrize("child_receipt", [False, True])
def test_routed_worker_failure_accounts_only_after_child_start_receipt(
    conn, monkeypatch, child_receipt,
):
    """T005 S003/S006: a PID is not an executor receipt or failure-budget boundary."""
    task_id = kb.create_task(conn, title="retry routed worker", assignee="worker")
    requests = []

    def decide(request):
        requests.append(request)
        return ExecutionRouteDecisionV1.route(
            request_id=request.request_id,
            attempt_id=request.attempt_id,
            candidate_id="native",
        )

    registration = _registration(decide)
    from hermes_cli.plugins import get_plugin_manager
    monkeypatch.setattr(
        get_plugin_manager(), "get_execution_router_registration", lambda: registration,
    )
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: None)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    spawned = []
    pids = iter((111, 222))

    def spawn(task, workspace, board=None):
        spawned.append((task, workspace))
        return next(pids)

    first = kbd.dispatch_once(conn, spawn_fn=spawn, failure_limit=2)
    assert [item[0] for item in first.spawned] == [task_id]
    assert requests[0].previous_attempt is None
    first_task = kb.get_task(conn, task_id)
    assert first_task is not None
    first_run_id = first_task.current_run_id
    assert first_run_id is not None

    if child_receipt:
        assert kb.record_routed_run_started(
            conn,
            task_id,
            first_run_id,
            first_task.claim_lock,
            provider="native-provider",
            model="native-model",
            reasoning=None,
        )

    kbd._record_worker_exit(111, 1 << 8)
    recovery = kbd.dispatch_once(conn, spawn_fn=spawn, failure_limit=2)
    assert recovery.crashed == ([task_id] if child_receipt else [])
    failed = kb.get_task(conn, task_id)
    assert failed is not None
    assert failed.status == "ready"
    assert failed.consecutive_failures == (1 if child_receipt else 0)
    assert len(spawned) == 1
    assert len(requests) == 1
    old_finished = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND run_id = ? "
        "AND kind = 'route_finished'",
        (task_id, first_run_id),
    ).fetchall()
    assert len(old_finished) == (1 if child_receipt else 0)
    if child_receipt:
        assert json.loads(old_finished[0]["payload"])["terminal_state"] == "crashed"
    else:
        not_started = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND run_id = ? "
            "AND kind = 'route_not_started'",
            (task_id, first_run_id),
        ).fetchall()
        assert len(not_started) == 1
        assert _validated_public_event(not_started[0]["payload"]).actual_route is None

    retry = kbd.dispatch_once(conn, spawn_fn=spawn, failure_limit=2)
    assert [item[0] for item in retry.spawned] == [task_id]
    assert len(spawned) == 2
    assert len(requests) == 2
    assert requests[0].request_id != requests[1].request_id
    assert requests[0].attempt_id != requests[1].attempt_id
    previous = requests[1].previous_attempt
    if child_receipt:
        assert previous is not None
        assert previous.attempt_id == requests[0].attempt_id
        assert previous.terminal_state == "crashed"
        assert previous.route == ExecutionRouteIdentityV1(
            "native", "native-provider", "native-model", None,
        )
        assert previous.reason_text is None
    else:
        assert previous is None
    current = kb.get_task(conn, task_id)
    assert current is not None
    assert current.status == "running"
    assert current.current_run_id != first_run_id
    assert conn.execute(
        "SELECT COUNT(*) FROM task_runs WHERE task_id = ?", (task_id,),
    ).fetchone()[0] == 2
    assert conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = 'route_finished'",
        (task_id,),
    ).fetchone()[0] == (1 if child_receipt else 0)


@pytest.mark.parametrize("worker_route_state", ["route", "pass_through"])
def test_prerouted_kanban_worker_chat_does_not_route_a_second_main_turn(
    conn, monkeypatch, worker_route_state,
):
    """T005 S005: accepted worker admission bypasses only the second CLI decision."""
    task_id = kb.create_task(
        conn,
        title="one worker route",
        assignee="worker",
        provider_override="worker-provider",
        model_override="worker-model",
    )
    requests = []

    def decide(request):
        requests.append(request)
        if worker_route_state == "route":
            return ExecutionRouteDecisionV1.route(
                request_id=request.request_id,
                attempt_id=request.attempt_id,
                candidate_id="native",
            )
        return ExecutionRouteDecisionV1.pass_through(
            request_id=request.request_id,
            attempt_id=request.attempt_id,
            reason_code="native",
        )

    registration = _registration(decide)
    from hermes_cli.plugins import get_plugin_manager
    monkeypatch.setattr(
        get_plugin_manager(), "get_execution_router_registration", lambda: registration,
    )
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: None)
    kbd.dispatch_once(conn, spawn_fn=lambda *args, **kwargs: 4242)
    task = kb.get_task(conn, task_id)
    assert task is not None and task.current_run_id is not None and task.claim_lock
    assert len(requests) == 1
    assert requests[0].execution_kind is ExecutionKind.KANBAN_WORKER
    assert conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND run_id = ? "
        "AND kind = 'route_started'",
        (task_id, task.current_run_id),
    ).fetchone()[0] == 0

    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(task.current_run_id))
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", task.claim_lock)
    monkeypatch.setenv("HERMES_KANBAN_ROUTED_ATTEMPT", worker_route_state)

    from hermes_cli.cli_chat_turn_mixin import CLIChatTurnMixin

    class NativeAgent:
        provider = "worker-provider"
        model = "worker-model"
        reasoning_config = None
        _execution_router_selected_attempt = worker_route_state == "route"

        def run_conversation(self, **_kwargs):
            return {"messages": [], "final_response": "native", "completed": True}

        def release_clients(self):
            return None

    constructed = []

    class Shell(CLIChatTurnMixin):
        def __init__(self):
            self._secret_capture_callback = None
            self._sudo_password_callback = None
            self._approval_callback = None
            self._active_agent_route_signature = ("worker-provider", "worker-model")
            self.agent = None
            self.conversation_history = []
            self.session_id = "worker-session"
            self.model = "worker-model"
            self.provider = "worker-provider"
            self.requested_provider = "worker-provider"
            self.reasoning_config = None
            self._fallback_model = []
            self._explicit_model_override = False
            self._explicit_provider_override = False
            self._explicit_reasoning_override = False
            self._pending_model_switch_note = None
            self._pending_skills_reload_note = None
            self._pending_moa_config = None
            self._pending_one_turn_model_restore = None
            self._pending_moa_disable_after_turn = False
            self._pending_moa_restore_model = None

        def _ensure_runtime_credentials(self):
            return True

        def _resolve_turn_agent_config(self, _message):
            return {
                "model": self.model,
                "runtime": {},
                "signature": (self.provider, self.model),
            }

        def _init_agent(self, **_kwargs):
            self.agent = NativeAgent()
            constructed.append(self.agent)
            if "HERMES_KANBAN_ROUTED_ATTEMPT" in os.environ:
                from tools.kanban_tools import _record_kanban_worker_start_receipt

                try:
                    _record_kanban_worker_start_receipt(
                        self.agent, os.environ["HERMES_KANBAN_ROUTED_ATTEMPT"],
                    )
                except RuntimeError:
                    self.agent = None
                    return False
            return True

        def _chat_route_images(self, message, images):
            return message

        def _chat_expand_context_references(self, message):
            return message, None

        def _chat_stage_user_message(self, agent, message):
            self.conversation_history.append({"role": "user", "content": message})

        def _reset_stream_state(self):
            return None

        def _chat_setup_turn_audio(self, turn, message, voice_input):
            return None

        def _chat_monitor_agent_thread(self, turn, agent_thread):
            agent_thread.join()
            return None

        def _chat_settle_turn(self, turn):
            return None

        def _chat_render_turn(self, turn, agent_thread, interrupt_msg):
            return turn.result["final_response"]

        def _chat_release_turn_audio(self, turn):
            return None

        def _flush_credit_notices(self):
            return None

    worker = Shell()
    assert worker.chat("work") == "native"
    assert len(requests) == 1
    assert len(constructed) == 1
    assert worker.provider == "worker-provider"
    assert worker.model == "worker-model"
    assert worker.agent is not None
    assert worker.agent._execution_router_selected_attempt is (worker_route_state == "route")
    assert conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND run_id = ? "
        "AND kind = 'route_started'",
        (task_id, task.current_run_id),
    ).fetchone()[0] == 1

    for name in (
        "HERMES_KANBAN_TASK",
        "HERMES_KANBAN_RUN_ID",
        "HERMES_KANBAN_CLAIM_LOCK",
        "HERMES_KANBAN_ROUTED_ATTEMPT",
    ):
        monkeypatch.delenv(name, raising=False)
    ordinary = Shell()
    assert ordinary.chat("ordinary") == "native"
    assert len(requests) == 2
    assert requests[1].execution_kind is ExecutionKind.MAIN_TURN

    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(task.current_run_id))
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "forged-claim")
    monkeypatch.setenv("HERMES_KANBAN_ROUTED_ATTEMPT", worker_route_state)
    forged = Shell()
    assert forged.chat("forged") is None
    assert len(requests) == 3
    assert requests[2].execution_kind is ExecutionKind.MAIN_TURN
