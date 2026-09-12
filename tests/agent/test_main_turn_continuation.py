from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.execution_router import (
    ExecutionKind,
    ExecutionRouteCandidateV1,
    ExecutionRouteDecisionV1,
    ExecutionRouterProviderDescriptorV1,
    RoutedAttemptRestartRequired,
)
from agent.context_compressor import _DB_PERSISTED_MARKER
from hermes_cli.execution_router_runtime import ExecutionRouterRegistration


def test_main_turn_continuation_uses_exact_transcript_without_user_replay(monkeypatch):
    from agent.main_turn_continuation import (
        _continue_main_turn_attempt,
        _seal_main_turn_continuation,
    )

    messages = [
        {"role": "user", "content": "do it", "_row_id": 1, _DB_PERSISTED_MARKER: True},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "fixture", "arguments": "{}"}}],
            "_row_id": 2,
            _DB_PERSISTED_MARKER: True,
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "done", "_row_id": 3, _DB_PERSISTED_MARKER: True},
    ]
    old_agent = SimpleNamespace(
        session_id="session-a",
        _last_flushed_db_idx=3,
        _routed_restart_required=SimpleNamespace(reason_code="failed", consumed_fallback_slot=0),
    )
    prepared = SimpleNamespace(
        request=SimpleNamespace(request_id="request-old", attempt_id="attempt-old"),
        route_signature=("provider-a", "model-a", None, 1, "1.0"),
    )
    record = _seal_main_turn_continuation(
        old_agent,
        prepared,
        {"messages": messages, "turn_id": "turn-1", "current_turn_user_idx": 0},
    )
    captured = {}

    class ContinuationAgent:
        session_id = "session-a"

    from agent import conversation_loop
    def continue_from_transcript(_agent, transcript, *, turn_id, current_turn_user_idx):
        captured.update(
            transcript=transcript,
            turn_id=turn_id,
            current_turn_user_idx=current_turn_user_idx,
        )
        transcript.append({"role": "assistant", "content": "finished"})
        return {"messages": transcript, "final_response": "finished", "completed": True}
    monkeypatch.setattr(
        conversation_loop, "_continue_main_turn_from_transcript", continue_from_transcript
    )

    result = _continue_main_turn_attempt(
        ContinuationAgent(), record, existing_surface_callbacks={}
    )

    assert [row["role"] for row in captured["transcript"][:3]] == ["user", "assistant", "tool"]
    assert captured["turn_id"] == "turn-1"
    assert captured["current_turn_user_idx"] == 0
    assert record.router_generation == 1
    assert record.router_contract_generation == "1.0"
    assert sum(row["role"] == "user" for row in result["messages"]) == 1
    assert result["messages"][-1] == {"role": "assistant", "content": "finished"}
    assert not hasattr(__import__("agent.main_turn_continuation", fromlist=["x"]), "__all__")
    assert "_continue_main_turn_from_transcript" not in conversation_loop.__all__


def test_main_turn_continuation_does_not_repeat_completed_tool_side_effect(monkeypatch):
    from agent.main_turn_continuation import _continue_main_turn_attempt

    side_effects = {"count": 1}
    record = _sealed_record(0, rows=[
        {"role": "user", "content": "go", _DB_PERSISTED_MARKER: True},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call-1"}], _DB_PERSISTED_MARKER: True},
        {"role": "tool", "content": "done", "tool_call_id": "call-1", _DB_PERSISTED_MARKER: True},
    ])

    class ContinuationAgent:
        session_id = "session-a"

    from agent import conversation_loop
    def continue_from_transcript(_agent, transcript, **_boundary):
        assert transcript[-1]["role"] == "tool"
        return {"messages": transcript, "final_response": "done", "completed": True}
    monkeypatch.setattr(
        conversation_loop, "_continue_main_turn_from_transcript", continue_from_transcript
    )

    _continue_main_turn_attempt(
        ContinuationAgent(), record, existing_surface_callbacks={}
    )

    assert side_effects["count"] == 1


class _CursorProvider:
    descriptor = ExecutionRouterProviderDescriptorV1(
        plugin_id="router-plugin",
        plugin_version="1.0",
        provider_id="router-provider",
        contract_version="1.0",
        supported_execution_kinds=(ExecutionKind.MAIN_TURN,),
    )

    def __init__(self):
        self.requests = []

    def resolve_execution_route(self, request, _cancellation):
        self.requests.append(request)
        return ExecutionRouteDecisionV1.route(
            request_id=request.request_id,
            attempt_id=request.attempt_id,
            candidate_id=request.native_candidate_id,
        )


def _sealed_record(consumed_slot=0, rows=None):
    from agent.main_turn_continuation import _seal_main_turn_continuation

    rows = rows or [{"role": "user", "content": "go", _DB_PERSISTED_MARKER: True}]
    agent = SimpleNamespace(
        session_id="session-a",
        _last_flushed_db_idx=len(rows),
        _routed_restart_required=RoutedAttemptRestartRequired("failed", consumed_slot),
    )
    prepared = SimpleNamespace(
        request=SimpleNamespace(request_id="old-request", attempt_id="old-attempt"),
        route_signature=("p0", "m0", None, 1, "1.0"),
    )
    return _seal_main_turn_continuation(
        agent,
        prepared,
        {"messages": rows, "turn_id": "turn-1", "current_turn_user_idx": 0},
    )


def test_main_turn_continuation_carries_spent_cursor_across_rebuild_and_primary_restore():
    from agent.execution_router import ExecutionRouteIdentityV1
    from hermes_cli.execution_router_runtime import (
        ExecutionRouteResolution,
        ExecutionRouteResolutionState,
        _prepare_main_turn_continuation_attempt,
        bind_main_turn_attempt,
    )

    provider = _CursorProvider()
    registration = ExecutionRouterRegistration(provider, 1, lambda generation: generation == 1)
    candidates = (
        ExecutionRouteCandidateV1("native", "p0", "m0", None),
        ExecutionRouteCandidateV1("fallback-0", "p1", "m1", None),
        ExecutionRouteCandidateV1("fallback-1", "p2", "m2", None),
    )

    prepared = _prepare_main_turn_continuation_attempt(
        _sealed_record(0),
        raw_instruction="go",
        surface_class="cli",
        eligible_candidates=candidates,
        registration=registration,
    )

    assert tuple(c.candidate_id for c in prepared.request.eligible_candidates) == ("fallback-1",)
    assert prepared.request.native_candidate_id == "fallback-1"
    assert provider.requests == [prepared.request]

    with pytest.raises(RuntimeError, match="exhausted"):
        _prepare_main_turn_continuation_attempt(
            _sealed_record(1),
            raw_instruction="go",
            surface_class="cli",
            eligible_candidates=candidates,
            registration=registration,
        )
    assert provider.requests == [prepared.request]

    selected = ExecutionRouteIdentityV1("fallback-1", "p2", "m2", None)
    bound = SimpleNamespace(
        resolution=ExecutionRouteResolution(
            ExecutionRouteResolutionState.ROUTE,
            accepted_route=selected,
        ),
        selected_route=selected,
        route_signature=("p2", "m2", None, 1, "1.0"),
    )
    agent = SimpleNamespace(_fallback_index=0)
    bind_main_turn_attempt(bound, agent, candidates)
    assert agent._fallback_index == 1
    assert agent._routed_route_signature == bound.route_signature


def test_main_turn_continuation_fresh_identity_and_lifecycle_order():
    from hermes_cli.execution_router_runtime import _prepare_main_turn_continuation_attempt

    provider = _CursorProvider()
    registration = ExecutionRouterRegistration(provider, 1, lambda generation: generation == 1)
    prepared = _prepare_main_turn_continuation_attempt(
        _sealed_record(0),
        raw_instruction="go",
        surface_class="gateway",
        eligible_candidates=(
            ExecutionRouteCandidateV1("native", "p0", "m0", None),
            ExecutionRouteCandidateV1("fallback-0", "p1", "m1", None),
            ExecutionRouteCandidateV1("fallback-1", "p2", "m2", None),
        ),
        registration=registration,
    )

    assert prepared.request.request_id != "old-request"
    assert prepared.request.attempt_id != "old-attempt"
    assert len(provider.requests) == 1
    _assert_main_turn_route_outcomes_pins_lifecycle_and_renderer_isolation()


def _assert_main_turn_route_outcomes_pins_lifecycle_and_renderer_isolation():
    from hermes_cli.execution_router_runtime import (
        ExecutionRouteResolutionState,
        finish_main_turn_attempt,
        prepare_main_turn_attempt,
        record_main_turn_not_started,
        start_main_turn_attempt,
    )
    from hermes_state import SessionDB
    from pathlib import Path
    import tempfile

    class Provider:
        descriptor = ExecutionRouterProviderDescriptorV1(
            plugin_id="router-plugin",
            plugin_version="1.0",
            provider_id="router-provider",
            contract_version="1.0",
            supported_execution_kinds=(ExecutionKind.MAIN_TURN,),
        )

        def __init__(self, outcome):
            self.outcome = outcome
            self.calls = 0

        def resolve_execution_route(self, request, _cancellation):
            self.calls += 1
            if self.outcome == "error":
                raise RuntimeError("router failed")
            if self.outcome == "route":
                return ExecutionRouteDecisionV1.route(
                    request_id=request.request_id,
                    attempt_id=request.attempt_id,
                    candidate_id="routed",
                )
            if self.outcome == "stop":
                return ExecutionRouteDecisionV1.stop(
                    request_id=request.request_id,
                    attempt_id=request.attempt_id,
                    reason_code="blocked",
                )
            return ExecutionRouteDecisionV1.pass_through(
                request_id=request.request_id,
                attempt_id=request.attempt_id,
            )

    candidates = (
        ExecutionRouteCandidateV1("native", "native-provider", "native-model", "medium"),
        ExecutionRouteCandidateV1("routed", "routed-provider", "routed-model", "high"),
    )
    outcomes = {
        "route": (ExecutionRouteResolutionState.ROUTE, "routed-model"),
        "pass": (ExecutionRouteResolutionState.PASS_THROUGH, "native-model"),
        "stop": (ExecutionRouteResolutionState.STOP, None),
        "error": (ExecutionRouteResolutionState.ROUTER_ERROR, None),
    }
    with tempfile.TemporaryDirectory() as directory:
        db = SessionDB(Path(directory) / "state.db")
        for outcome, (state, model) in outcomes.items():
            session_id = f"session-{outcome}"
            db.create_session(session_id, source="cli")
            lifecycle = db.execution_route_lifecycle(session_id)
            provider = Provider(outcome)

            def broken_renderer(_notice):
                raise RuntimeError("optional renderer failed")

            attempt = prepare_main_turn_attempt(
                raw_instruction="route this",
                surface_class="cli",
                session_id=session_id,
                native_candidate_id="native",
                eligible_candidates=candidates,
                registration=ExecutionRouterRegistration(
                    provider, 1, lambda generation: generation == 1
                ),
                lifecycle=lifecycle,
                render_notice=broken_renderer,
                revalidate=lambda _request, accepted: accepted in tuple(
                    candidate.identity() for candidate in candidates
                ),
            )
            assert provider.calls == 1
            assert attempt.resolution.state is state
            assert attempt.selected_model == model
            assert attempt.may_start is (model is not None)
            event_types = [event.event_type.value for event in lifecycle.read_events(
                request_id=attempt.request.request_id,
                attempt_id=attempt.request.attempt_id,
            )]
            assert event_types == (
                ["route_requested", "route_accepted"]
                if outcome == "route"
                else ["route_requested", "route_not_started"]
                if outcome in {"stop", "error"}
                else ["route_requested"]
            )
            if outcome == "route":
                start_main_turn_attempt(attempt, lifecycle, attempt.selected_route)
                finish_main_turn_attempt(attempt, lifecycle, "completed")
                assert [event.event_type.value for event in lifecycle.read_events(
                    request_id=attempt.request.request_id,
                    attempt_id=attempt.request.attempt_id,
                )] == [
                    "route_requested", "route_accepted", "route_started", "route_finished"
                ]

        native = candidates[0]
        pinned = prepare_main_turn_attempt(
            raw_instruction="unchanged",
            surface_class="cli",
            session_id="native-pinned",
            native_candidate_id="native",
            eligible_candidates=(native,),
            explicit_model_pin="native-model",
            explicit_provider_pin="native-provider",
            explicit_reasoning_pin="medium",
            registration=None,
        )
        assert pinned.resolution.state is ExecutionRouteResolutionState.NATIVE
        assert pinned.selected_route == native.identity()
        assert (
            pinned.request.pins.model,
            pinned.request.pins.provider,
            pinned.request.pins.reasoning,
        ) == ("native-model", "native-provider", "medium")

        db.create_session("construction", source="cli")
        provider = Provider("route")
        construction = prepare_main_turn_attempt(
            raw_instruction="construct",
            surface_class="cli",
            session_id="construction",
            native_candidate_id="native",
            eligible_candidates=candidates,
            registration=ExecutionRouterRegistration(
                provider, 1, lambda generation: generation == 1
            ),
            lifecycle=db.execution_route_lifecycle("construction"),
            revalidate=lambda _request, accepted: accepted in tuple(
                candidate.identity() for candidate in candidates
            ),
        )
        record_main_turn_not_started(construction, "credential_binding_failed")
        assert [event.event_type.value for event in construction.lifecycle.read_events(
            request_id=construction.request.request_id,
            attempt_id=construction.request.attempt_id,
        )] == ["route_requested", "route_accepted", "route_not_started"]
        db.close()
