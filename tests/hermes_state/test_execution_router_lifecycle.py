from __future__ import annotations

import dataclasses
import threading
from pathlib import Path

import pytest

from agent.execution_router import (
    ExecutionKind,
    ExecutionRouteDecisionV1,
    ExecutionRouteEventType,
    ExecutionRouteIdentityV1,
    ExecutionRouterProviderDescriptorV1,
)
from hermes_cli.execution_router_runtime import (
    ExecutionRouteResolutionState,
    ExecutionRouterRegistration,
    resolve_execution_route,
)
from hermes_state import SessionDB


def _request():
    from tests.agent.test_execution_router import _contract

    return _contract()[0]


def _registration():
    class Provider:
        descriptor = ExecutionRouterProviderDescriptorV1(
            plugin_id="plugin-a",
            plugin_version="1.0",
            provider_id="router-a",
            contract_version="1.0",
            supported_execution_kinds=tuple(ExecutionKind),
        )

        def resolve_execution_route(self, request, cancellation):
            return ExecutionRouteDecisionV1.route(
                request_id=request.request_id,
                attempt_id=request.attempt_id,
                candidate_id="candidate-1",
            )

    registration = None
    registration = ExecutionRouterRegistration(Provider(), 1, lambda generation: generation == 1)
    return registration


def _resolve(request, lifecycle):
    return resolve_execution_route(
        request,
        registration=_registration(),
        lifecycle=lifecycle,
        revalidate=lambda *_args: True,
    )


def test_session_lifecycle_is_transactional_idempotent_and_immutable(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.create_session("session-a", source="cli")
    lifecycle = db.execution_route_lifecycle("session-a")
    request = _request()

    result = _resolve(request, lifecycle)
    replay = _resolve(request, lifecycle)

    assert result.state is ExecutionRouteResolutionState.ROUTE
    assert replay == result
    events = lifecycle.read_events(limit=100)
    assert [event.event_type for event in events] == [
        ExecutionRouteEventType.REQUESTED,
        ExecutionRouteEventType.ACCEPTED,
    ]
    assert events[0].sequence == 1
    assert events[1].sequence == 2
    assert events[0].accepted_route is None
    assert events[0].request_digest == request.request_digest
    assert events[0].instruction_digest == request.instruction.digest
    assert events[0].eligibility_revision == request.eligibility_revision
    assert events[1].request_digest is None
    assert events[1].instruction_digest is None
    assert events[1].eligibility_revision is None
    assert events[1].accepted_route.provider == "provider-a"
    ids = {"request_id": request.request_id, "attempt_id": request.attempt_id}
    with pytest.raises(ValueError):
        lifecycle.record_finished("done", **ids)

    actual = ExecutionRouteIdentityV1("candidate-1", "provider-a", "model-a", "high")
    started = lifecycle.record_started(actual, **ids)
    assert lifecycle.record_started(actual, **ids) == started
    finished = lifecycle.record_finished("failed", **ids)
    assert lifecycle.record_finished("failed", **ids) == finished
    attempt_events = lifecycle.read_events(
        request_id=request.request_id, attempt_id=request.attempt_id, limit=100
    )
    assert [event.sequence for event in attempt_events] == [1, 2, 3, 4]
    db.close()


def test_concurrent_duplicate_request_invokes_provider_once(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("session-a", source="cli")
    lifecycle = db.execution_route_lifecycle("session-a")
    request = _request()
    registration = _registration()
    entered = threading.Event()
    release = threading.Event()
    calls = []
    provider = registration.provider

    def delayed(value, cancellation):
        calls.append(value.request_id)
        entered.set()
        release.wait(1)
        return ExecutionRouteDecisionV1.route(
            request_id=value.request_id,
            attempt_id=value.attempt_id,
            candidate_id="candidate-1",
        )

    object.__setattr__(provider, "resolve_execution_route", delayed)
    registration = ExecutionRouterRegistration(provider, 1, lambda generation: generation == 1)
    results = []
    workers = [threading.Thread(target=lambda: results.append(resolve_execution_route(
        request, registration=registration, lifecycle=lifecycle, revalidate=lambda *_args: True
    ))) for _ in range(2)]
    workers[0].start()
    assert entered.wait(1)
    workers[1].start()
    release.set()
    for worker in workers:
        worker.join(1)
    assert calls == [request.request_id]
    assert len(results) == 2 and results[0] == results[1]
    db.close()


def test_requested_binding_mismatch_is_conflicting_replay(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("session-a", source="cli")
    lifecycle = db.execution_route_lifecycle("session-a")
    request = _request()
    lifecycle.record_execution_route_requested(request, _registration())
    changed_instruction = dataclasses.replace(
        request,
        instruction=type(request.instruction).from_text("different"),
        request_digest=None,
    ).with_computed_digest()
    with pytest.raises(ValueError, match="bound differently"):
        lifecycle.record_execution_route_requested(changed_instruction, _registration())
    assert [event.sequence for event in lifecycle.read_events(limit=100)] == [1]
    db.close()


def test_settled_resolution_rejects_conflicting_request_binding_replay(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("session-a", source="cli")
    lifecycle = db.execution_route_lifecycle("session-a")
    request = _request()
    _resolve(request, lifecycle)
    changed = dataclasses.replace(
        request, eligibility_revision="eligibility-2", request_digest=None
    ).with_computed_digest()

    replay = _resolve(changed, lifecycle)

    assert replay.state is ExecutionRouteResolutionState.ROUTER_ERROR
    assert replay.reason_code == "conflicting_replay"
    assert [
        event.sequence
        for event in lifecycle.read_events(
            request_id=request.request_id,
            attempt_id=request.attempt_id,
            limit=100,
        )
    ] == [1, 2]
    db.close()


def test_started_route_must_equal_accepted_route(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("session-a", source="cli")
    lifecycle = db.execution_route_lifecycle("session-a")
    request = _request()
    _resolve(request, lifecycle)
    wrong = ExecutionRouteIdentityV1("candidate-x", "provider-x", "model-x", None)
    with pytest.raises(ValueError, match="actual route"):
        lifecycle.record_started(
            wrong,
            request_id=request.request_id,
            attempt_id=request.attempt_id,
        )
    assert [event.event_type for event in lifecycle.read_events(limit=100)] == [
        ExecutionRouteEventType.REQUESTED,
        ExecutionRouteEventType.ACCEPTED,
    ]
    db.close()


def test_fresh_lifecycle_cannot_mutate_an_implicitly_latest_attempt(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("session-a", source="cli")
    _resolve(_request(), db.execution_route_lifecycle("session-a"))
    fresh = db.execution_route_lifecycle("session-a")
    actual = ExecutionRouteIdentityV1("candidate-1", "provider-a", "model-a", "high")

    with pytest.raises(TypeError, match="request_id"):
        fresh.record_started(actual)
    db.close()


def test_shared_session_lifecycle_requires_explicit_attempt_for_transition(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("session-a", source="cli")
    lifecycle = db.execution_route_lifecycle("session-a")
    first = _request()
    second = dataclasses.replace(
        first,
        request_id="request-2",
        attempt_id="attempt-2",
        request_digest=None,
    ).with_computed_digest()
    _resolve(first, lifecycle)
    _resolve(second, lifecycle)
    actual = ExecutionRouteIdentityV1("candidate-1", "provider-a", "model-a", "high")

    lifecycle.record_started(
        actual,
        request_id=first.request_id,
        attempt_id=first.attempt_id,
    )

    first_events = lifecycle.read_events(
        request_id=first.request_id,
        attempt_id=first.attempt_id,
        limit=100,
    )
    second_events = lifecycle.read_events(
        request_id=second.request_id,
        attempt_id=second.attempt_id,
        limit=100,
    )
    assert first_events[-1].event_type is ExecutionRouteEventType.STARTED
    assert second_events[-1].event_type is ExecutionRouteEventType.ACCEPTED
    db.close()


def test_route_rows_recover_with_session_and_cascade_on_delete(tmp_path):
    path = Path(tmp_path) / "state.db"
    db = SessionDB(path)
    db.create_session("session-a", source="subagent")
    lifecycle = db.execution_route_lifecycle("session-a")
    _resolve(_request(), lifecycle)
    db.close()

    reopened = SessionDB(path)
    assert len(reopened.execution_route_lifecycle("session-a").read_events(limit=100)) == 2
    assert reopened.execution_route_lifecycle("other-session").read_events(limit=100) == ()
    assert reopened.delete_session("session-a") is True
    assert reopened.execution_route_lifecycle("session-a").read_events(limit=100) == ()
    reopened.close()


def test_offline_session_recovery_preserves_route_attempts_and_events(tmp_path):
    from hermes_cli.session_recovery import recover_session_database

    source = tmp_path / "source.db"
    db = SessionDB(source)
    db.create_session("session-a", source="cli")
    lifecycle = db.execution_route_lifecycle("session-a")
    request = _request()
    _resolve(request, lifecycle)
    db.close()

    output = tmp_path / "recovered.db"
    recover_session_database(source, output, work_dir=tmp_path)
    recovered = SessionDB(output)
    events = recovered.execution_route_lifecycle("session-a").read_events(limit=100)
    assert {event.event_type for event in events} == {
        ExecutionRouteEventType.REQUESTED,
        ExecutionRouteEventType.ACCEPTED,
    }
    requested = next(event for event in events if event.event_type is ExecutionRouteEventType.REQUESTED)
    assert requested.instruction_digest == request.instruction.digest
    recovered.close()


def test_event_reads_are_bounded_and_not_mutating(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("session-a", source="cli")
    lifecycle = db.execution_route_lifecycle("session-a")
    _resolve(_request(), lifecycle)
    with pytest.raises(ValueError):
        lifecycle.read_events(limit=101)
    assert isinstance(lifecycle.read_events(event_type=ExecutionRouteEventType.REQUESTED), tuple)
    assert not hasattr(lifecycle, "delete_event")
    assert not hasattr(lifecycle, "replay")
    db.close()


def test_attempt_sequence_bounds_require_exact_authoritative_identity(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("session-a", source="cli")
    lifecycle = db.execution_route_lifecycle("session-a")
    _resolve(_request(), lifecycle)
    with pytest.raises(ValueError, match="request_id and attempt_id"):
        lifecycle.read_events(after_sequence=1)
    with pytest.raises(ValueError, match="request_id and attempt_id"):
        lifecycle.read_events(request_id="request-1", before_sequence=2)
    with pytest.raises(ValueError, match="authoritative attempt"):
        lifecycle.read_events(
            request_id="request-1", attempt_id="wrong", after_sequence=0
        )
    events = lifecycle.read_events(
        request_id="request-1", attempt_id="attempt-1",
        after_sequence=1, before_sequence=3,
    )
    assert [event.sequence for event in events] == [2]
    db.close()


def test_session_pagination_is_opaque_stable_and_filter_bound(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("session-a", source="cli")
    db.create_session("session-b", source="cli")
    monkeypatch.setattr("hermes_state_execution_router.time.time", lambda: 1234.0)
    for number in range(3):
        request = dataclasses.replace(
            _request(),
            request_id=f"request-{number}",
            attempt_id=f"attempt-{number}",
            request_digest=None,
        ).with_computed_digest()
        _resolve(request, db.execution_route_lifecycle("session-a"))
    lifecycle = db.execution_route_lifecycle("session-a")
    first = lifecycle.read_event_page(limit=2)
    second = lifecycle.read_event_page(limit=2, page_token=first.next_page_token)
    third = lifecycle.read_event_page(limit=2, page_token=second.next_page_token)
    event_ids = [event.event_id for page in (first, second, third) for event in page.events]
    assert len(event_ids) == 6
    assert len(set(event_ids)) == 6
    assert event_ids == sorted(event_ids)
    assert lifecycle.read_event_page(page_token="altered").events == ()
    first_token = first.next_page_token
    assert first_token is not None
    altered = first_token[:4] + "!!!!" + first_token[4:]
    assert lifecycle.read_event_page(page_token=altered, limit=2).events == ()
    assert db.execution_route_lifecycle("session-b").read_event_page(
        page_token=first.next_page_token
    ).events == ()
    assert lifecycle.read_event_page(
        page_token=first.next_page_token,
        event_type=ExecutionRouteEventType.REQUESTED,
    ).events == ()
    with pytest.raises(ValueError, match="sequence"):
        lifecycle.read_event_page(page_token=first.next_page_token, after_sequence=1)
    db.close()


def test_filtered_continuation_uses_only_its_bound_page_token(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("session-a", source="cli")
    monkeypatch.setattr("hermes_state_execution_router.time.time", lambda: 1234.0)
    lifecycle = db.execution_route_lifecycle("session-a")
    for number in range(2):
        request = dataclasses.replace(
            _request(),
            request_id=f"request-{number}",
            attempt_id=f"attempt-{number}",
            request_digest=None,
        ).with_computed_digest()
        _resolve(request, lifecycle)
    first = lifecycle.read_event_page(
        event_type=ExecutionRouteEventType.REQUESTED,
        limit=1,
    )
    assert first.next_page_token is not None

    second = lifecycle.read_event_page(page_token=first.next_page_token, limit=1)

    assert len(second.events) == 1
    assert second.events[0].event_type is ExecutionRouteEventType.REQUESTED
    db.close()


def test_session_page_token_is_bound_to_owning_store_scope(tmp_path, monkeypatch):
    monkeypatch.setattr("hermes_state_execution_router.time.time", lambda: 1000.0)
    first_db = SessionDB(tmp_path / "profile-a" / "state.db")
    first_db.create_session("same-session", source="cli")
    first_lifecycle = first_db.execution_route_lifecycle("same-session")
    _resolve(_request(), first_lifecycle)
    token = first_lifecycle.read_event_page(limit=1).next_page_token
    assert token is not None

    monkeypatch.setattr("hermes_state_execution_router.time.time", lambda: 1001.0)
    second_db = SessionDB(tmp_path / "profile-b" / "state.db")
    second_db.create_session("same-session", source="cli")
    _resolve(_request(), second_db.execution_route_lifecycle("same-session"))

    assert second_db.execution_route_lifecycle("same-session").read_event_page(
        page_token=token,
        limit=1,
    ).events == ()
    first_db.close()
    second_db.close()


def test_route_rows_cascade_through_existing_prune_operation(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("session-a", source="cli")
    lifecycle = db.execution_route_lifecycle("session-a")
    _resolve(_request(), lifecycle)
    db.end_session("session-a", end_reason="done")

    assert db.prune_sessions(older_than_days=None, source="cli") == 1
    assert lifecycle.read_events(limit=100) == ()
    db.close()
