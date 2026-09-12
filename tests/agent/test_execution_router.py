from __future__ import annotations

import dataclasses
import hashlib
import json
import threading
import time
from typing import Any

import pytest


def _contract():
    from agent.execution_router import (
        CONTRACT_VERSION,
        ExecutionKind,
        ExecutionRouteCapabilityAttributeV1,
        ExecutionRouteCandidateV1,
        ExecutionRouteDecisionKind,
        ExecutionRouteDecisionV1,
        ExecutionRouteInstructionV1,
        ExecutionRoutePinsV1,
        ExecutionRouteRequestV1,
        canonical_json_bytes,
        compute_request_digest,
        validate_decision,
    )

    attrs = (ExecutionRouteCapabilityAttributeV1("context_window", 128000),)
    candidate = ExecutionRouteCandidateV1(
        candidate_id="candidate-1",
        provider="provider-a",
        model="model-a",
        reasoning="high",
        attributes=attrs,
    )
    instruction = ExecutionRouteInstructionV1.from_text("héllo", original_utf8_bytes=6)
    request = ExecutionRouteRequestV1(
        contract_version=CONTRACT_VERSION,
        request_id="request-1",
        root_id="root-1",
        task_id=None,
        execution_id="execution-1",
        attempt_id="attempt-1",
        execution_kind=ExecutionKind.MAIN_TURN,
        surface_class="cli",
        instruction=instruction,
        pins=ExecutionRoutePinsV1(),
        native_candidate_id="candidate-1",
        eligibility_revision="revision-1",
        eligible_candidates=(candidate,),
        previous_attempt=None,
        request_digest=None,
    ).with_computed_digest()
    return request, candidate, ExecutionRouteDecisionV1, ExecutionRouteDecisionKind, canonical_json_bytes, compute_request_digest, validate_decision


def test_v1_contract_is_frozen_canonical_and_digest_bound():
    request, _, _, _, canonical_json_bytes, compute_request_digest, _ = _contract()

    assert json.loads(canonical_json_bytes(request))["execution_kind"] == "main_turn"
    assert canonical_json_bytes(request) == canonical_json_bytes(request)
    assert request.request_digest == compute_request_digest(request)
    assert request.instruction.digest == hashlib.sha256("héllo".encode()).hexdigest()
    with pytest.raises(dataclasses.FrozenInstanceError):
        request.request_id = "changed"


def test_candidate_attributes_and_json_safe_integers_are_strict():
    from agent.execution_router import ExecutionRouteCapabilityAttributeV1

    with pytest.raises(ValueError):
        ExecutionRouteCapabilityAttributeV1("z", 9007199254740992)
    with pytest.raises(TypeError):
        ExecutionRouteCapabilityAttributeV1("z", 1.5)
    with pytest.raises(ValueError):
        ExecutionRouteCapabilityAttributeV1("é", "x")


def test_instruction_limit_is_exact_utf8_bytes():
    from agent.execution_router import MAX_INSTRUCTION_UTF8_BYTES, ExecutionRouteInstructionV1

    exact = "é" * (MAX_INSTRUCTION_UTF8_BYTES // 2)
    instruction = ExecutionRouteInstructionV1.from_text(exact)
    assert instruction.text is not None
    assert len(instruction.text.encode("utf-8")) == MAX_INSTRUCTION_UTF8_BYTES
    with pytest.raises(ValueError):
        ExecutionRouteInstructionV1.from_text(exact + "é")


def test_decision_validation_is_exact_and_correlated():
    request, candidate, Decision, Kind, _, _, validate_decision = _contract()
    accepted = validate_decision(
        request,
        Decision.route(request_id=request.request_id, attempt_id=request.attempt_id, candidate_id=candidate.candidate_id),
    )
    assert accepted.kind is Kind.ROUTE
    with pytest.raises(ValueError):
        validate_decision(request, Decision.route(request_id="other", attempt_id=request.attempt_id, candidate_id=candidate.candidate_id))
    with pytest.raises(ValueError):
        validate_decision(request, Decision.route(request_id=request.request_id, attempt_id=request.attempt_id, candidate_id="missing"))


def test_host_capability_discovery_does_not_claim_surface_integration():
    from agent.execution_router import ExecutionKind, discover_execution_router_capabilities

    capabilities = discover_execution_router_capabilities()
    assert capabilities.supported_contract_versions == ("1.0",)
    assert capabilities.supported_execution_kinds == tuple(ExecutionKind)
    assert capabilities.operational_execution_kinds == ()
    assert capabilities.route_event_contract_available is True
    assert capabilities.route_event_read_available is True
    assert capabilities.host_release != "unknown"
    assert capabilities.host_build != "unknown"
    with pytest.raises(TypeError):
        discover_execution_router_capabilities(host_release="forged")


def test_host_request_admission_authorizes_redacts_truncates_and_binds_exact_bytes():
    from agent.execution_router import MAX_INSTRUCTION_UTF8_BYTES, ExecutionKind
    from hermes_cli.execution_router_runtime import prepare_execution_route_request

    base = _contract()[0]
    calls = []
    with pytest.raises(PermissionError):
        prepare_execution_route_request(
            base,
            raw_instruction="secret",
            authorized_execution_kinds=frozenset(),
            redact_instruction=lambda value: calls.append(value) or value,
        )
    assert calls == []

    admitted = prepare_execution_route_request(
        base,
        raw_instruction="secret " + ("é" * MAX_INSTRUCTION_UTF8_BYTES),
        authorized_execution_kinds=frozenset({ExecutionKind.MAIN_TURN}),
        redact_instruction=lambda value: calls.append(value) or value.replace("secret", "[redacted]"),
    )
    delivered = admitted.instruction.text.encode("utf-8")
    assert len(delivered) <= MAX_INSTRUCTION_UTF8_BYTES
    assert delivered.decode("utf-8").startswith("[redacted]")
    assert admitted.instruction.digest == hashlib.sha256(delivered).hexdigest()
    assert admitted.request_digest == admitted.with_computed_digest().request_digest


def test_event_contract_rejects_impossible_transition_shapes():
    from agent.execution_router import (
        CONTRACT_VERSION,
        ExecutionKind,
        ExecutionRouteDecisionKind,
        ExecutionRouteEventType,
        ExecutionRouteEventV1,
        ExecutionRouteIdentityV1,
    )

    common: dict[str, Any] = dict(
        contract_version=CONTRACT_VERSION,
        event_id="event-1",
        root_id="root-1",
        task_id=None,
        execution_id="execution-1",
        attempt_id="attempt-1",
        request_id="request-1",
        previous_attempt_id=None,
        sequence=1,
        timestamp_utc_ms=1,
        execution_kind=ExecutionKind.MAIN_TURN,
        surface_class="cli",
        router_plugin_id="plugin-a",
        router_provider_id="router-a",
        router_contract_version=CONTRACT_VERSION,
        requested_candidate_id=None,
        reason_code=None,
        reason_text=None,
        terminal_state=None,
    )
    route = ExecutionRouteIdentityV1("candidate-1", "provider-a", "model-a", None)
    with pytest.raises(ValueError, match="route_requested"):
        ExecutionRouteEventV1(
            **common,
            event_type=ExecutionRouteEventType.REQUESTED,
            decision_state=None,
            accepted_route=route,
            actual_route=None,
        )
    with pytest.raises(ValueError, match="route_started"):
        ExecutionRouteEventV1(
            **common,
            event_type=ExecutionRouteEventType.STARTED,
            decision_state=ExecutionRouteDecisionKind.ROUTE,
            accepted_route=route,
            actual_route=None,
        )


class _Lifecycle:
    def __init__(self):
        self.results = {}
        self.events = []

    def get_execution_route_resolution(self, request_id, attempt_id):
        return self.results.get((request_id, attempt_id))

    def record_execution_route_requested(self, request, registration):
        owns_resolution = (request.request_id, request.attempt_id) not in self.results
        if owns_resolution:
            self.events.append(("requested", request.request_id))
        return owns_resolution

    def record_execution_route_result(self, request, result, registration):
        self.events.append(("result", result.state.value))
        self.results[(request.request_id, request.attempt_id)] = result
        return result


def _registration(callback, generation=1, is_current=None):
    from agent.execution_router import ExecutionKind, ExecutionRouterProviderDescriptorV1
    from hermes_cli.execution_router_runtime import ExecutionRouterRegistration

    class Provider:
        descriptor = ExecutionRouterProviderDescriptorV1(
            plugin_id="plugin-a",
            plugin_version="1.2.3",
            provider_id="router-a",
            contract_version="1.0",
            supported_execution_kinds=tuple(ExecutionKind),
        )

        def resolve_execution_route(self, request, cancellation):
            return callback(request, cancellation)

    current = is_current or (lambda value: value == generation)
    return ExecutionRouterRegistration(Provider(), generation, current)


def test_runtime_no_router_is_native_and_has_no_synthetic_events():
    from hermes_cli.execution_router_runtime import ExecutionRouteResolutionState, resolve_execution_route

    request = _contract()[0]
    lifecycle = _Lifecycle()
    result = resolve_execution_route(request, lifecycle=lifecycle)
    assert result.state is ExecutionRouteResolutionState.NATIVE
    assert lifecycle.events == []


def test_runtime_accepts_once_and_reuses_durable_resolution():
    from agent.execution_router import ExecutionRouteDecisionV1
    from hermes_cli.execution_router_runtime import ExecutionRouteResolutionState, resolve_execution_route

    request = _contract()[0]
    calls = []
    registration = _registration(
        lambda value, _cancellation: calls.append(value.request_id)
        or ExecutionRouteDecisionV1.route(
            request_id=value.request_id,
            attempt_id=value.attempt_id,
            candidate_id="candidate-1",
        )
    )
    lifecycle = _Lifecycle()
    first = resolve_execution_route(
        request, registration=registration, lifecycle=lifecycle, revalidate=lambda *_args: True
    )
    second = resolve_execution_route(request, registration=registration, lifecycle=lifecycle)
    assert first.state is ExecutionRouteResolutionState.ROUTE
    assert second == first
    assert calls == [request.request_id]
    assert lifecycle.events == [("requested", request.request_id), ("result", "route")]


def test_runtime_timeout_is_fail_closed_and_late_result_is_discarded():
    from agent.execution_router import ExecutionRouteDecisionV1
    from hermes_cli.execution_router_runtime import ExecutionRouteResolutionState, resolve_execution_route

    request = _contract()[0]
    release = threading.Event()
    observed = {}

    def delayed(value, cancellation):
        observed["signal"] = cancellation
        release.wait(2)
        observed["cancelled"] = cancellation.is_cancelled()
        return ExecutionRouteDecisionV1.pass_through(
            request_id=value.request_id, attempt_id=value.attempt_id
        )

    registration = _registration(delayed)
    lifecycle = _Lifecycle()
    started = time.monotonic()
    result = resolve_execution_route(request, registration=registration, lifecycle=lifecycle)
    elapsed = time.monotonic() - started
    assert result.state is ExecutionRouteResolutionState.ROUTER_ERROR
    assert result.reason_code == "timeout"
    assert elapsed < 2
    release.set()
    time.sleep(0.05)
    assert observed["cancelled"] is True
    assert lifecycle.results[(request.request_id, request.attempt_id)] == result


def test_provider_contract_requires_two_arguments_and_signal_is_read_only():
    from agent.execution_router import ExecutionRouterProviderV1
    from hermes_cli.execution_router_runtime import resolve_execution_route

    request = _contract()[0]

    class OneArgumentProvider:
        descriptor = _registration(lambda *_args: None).provider.descriptor

        def resolve_execution_route(self, request):
            return None

    registration = _registration(
        lambda _request, cancellation: (
            pytest.raises(AttributeError, setattr, cancellation, "cancelled", True),
            pytest.raises(AttributeError, getattr, cancellation, "cancel"),
            None,
        )[-1]
    )
    from agent.execution_router import validate_provider
    with pytest.raises(TypeError, match="exactly request and cancellation"):
        validate_provider(OneArgumentProvider())
    assert resolve_execution_route(request, registration=registration).state.value == "pass_through"


def test_cancellation_signal_contains_no_provider_reachable_host_callable():
    from hermes_cli.execution_router_runtime import resolve_execution_route

    captured = []
    registration = _registration(
        lambda _request, cancellation: captured.append(cancellation)
    )
    assert resolve_execution_route(_contract()[0], registration=registration).state.value == "pass_through"
    signal = captured[0]
    private_prefix = f"_{type(signal).__name__}__"
    assert all(
        not callable(object.__getattribute__(signal, name))
        for name in dir(signal)
        if name.startswith(private_prefix)
    )


def test_closed_resolution_retires_private_cancellation_state():
    from agent.execution_router import _CANCELLATION_LOCK, _CANCELLATION_STATES
    from hermes_cli.execution_router_runtime import resolve_execution_route

    with _CANCELLATION_LOCK:
        before = set(_CANCELLATION_STATES)
    assert resolve_execution_route(
        _contract()[0],
        registration=_registration(lambda _request, _cancellation: None),
    ).state.value == "pass_through"
    with _CANCELLATION_LOCK:
        assert set(_CANCELLATION_STATES) == before


def test_generation_revocation_signals_and_rejects_late_output():
    from agent.execution_router import ExecutionRouteDecisionV1
    from hermes_cli.execution_router_runtime import resolve_execution_route

    request = _contract()[0]
    entered = threading.Event()
    release = threading.Event()
    observed = {}

    def delayed(value, cancellation):
        entered.set()
        release.wait(2)
        observed["cancelled"] = cancellation.is_cancelled()
        return ExecutionRouteDecisionV1.pass_through(
            request_id=value.request_id, attempt_id=value.attempt_id
        )

    registration = _registration(delayed)
    outcome = []
    thread = threading.Thread(
        target=lambda: outcome.append(resolve_execution_route(request, registration=registration))
    )
    thread.start()
    assert entered.wait(1)
    registration.revoke()
    thread.join(1)
    assert not thread.is_alive()
    assert outcome[0].reason_code == "stale_generation"
    release.set()
    time.sleep(0.05)
    assert observed["cancelled"] is True


def test_live_generation_invalidation_signals_and_closes_open_resolution():
    from agent.execution_router import ExecutionRouteDecisionV1
    from hermes_cli.execution_router_runtime import resolve_execution_route

    current = threading.Event()
    current.set()
    entered = threading.Event()
    release = threading.Event()
    observed = {}

    def delayed(value, cancellation):
        entered.set()
        release.wait(2)
        observed["cancelled"] = cancellation.is_cancelled()
        return ExecutionRouteDecisionV1.pass_through(
            request_id=value.request_id,
            attempt_id=value.attempt_id,
        )

    registration = _registration(delayed, is_current=lambda _value: current.is_set())
    outcome = []
    worker = threading.Thread(
        target=lambda: outcome.append(
            resolve_execution_route(_contract()[0], registration=registration)
        )
    )
    worker.start()
    assert entered.wait(1)
    current.clear()
    worker.join(1)
    assert not worker.is_alive()
    assert outcome[0].reason_code == "stale_generation"
    release.set()
    time.sleep(0.05)
    assert observed["cancelled"] is True


def test_runtime_rejects_stale_generation_malformed_and_recursive_calls():
    from hermes_cli.execution_router_runtime import ExecutionRouteResolutionState, resolve_execution_route

    request = _contract()[0]
    stale = _registration(lambda _request, _cancellation: None, generation=2, is_current=lambda _value: False)
    assert resolve_execution_route(request, registration=stale).reason_code == "stale_generation"

    malformed = _registration(lambda _request, _cancellation: {"kind": "route"})
    assert resolve_execution_route(request, registration=malformed).reason_code == "malformed_decision"

    nested = []

    def recursive(value, _cancellation):
        nested.append(resolve_execution_route(value, registration=recursive_registration))
        return None

    recursive_registration = _registration(recursive)
    outer = resolve_execution_route(request, registration=recursive_registration)
    assert outer.state is ExecutionRouteResolutionState.PASS_THROUGH
    assert nested[0].reason_code == "recursion"


def test_runtime_provider_exception_and_awaitable_are_fail_closed():
    from hermes_cli.execution_router_runtime import ExecutionRouteResolutionState, resolve_execution_route

    request = _contract()[0]

    def fail(_request, _cancellation):
        raise RuntimeError("provider detail must not escape")

    error = resolve_execution_route(request, registration=_registration(fail))
    assert error.state is ExecutionRouteResolutionState.ROUTER_ERROR
    assert error.reason_code == "provider_error"
    assert "provider detail" not in (error.reason_text or "")

    class AwaitableDecision:
        def __await__(self):
            yield

    malformed = resolve_execution_route(
        request, registration=_registration(lambda _request, _cancellation: AwaitableDecision())
    )
    assert malformed.state is ExecutionRouteResolutionState.ROUTER_ERROR
    assert malformed.reason_code == "malformed_decision"


def test_runtime_revalidation_failure_is_fail_closed():
    from agent.execution_router import ExecutionRouteDecisionV1
    from hermes_cli.execution_router_runtime import ExecutionRouteResolutionState, resolve_execution_route

    request = _contract()[0]
    registration = _registration(lambda value, _cancellation: ExecutionRouteDecisionV1.route(
        request_id=value.request_id,
        attempt_id=value.attempt_id,
        candidate_id="candidate-1",
    ))
    result = resolve_execution_route(request, registration=registration, revalidate=lambda *_args: False)
    assert result.state is ExecutionRouteResolutionState.ROUTER_ERROR
    assert result.reason_code == "stale_eligibility"
    missing = resolve_execution_route(request, registration=registration)
    assert missing.state is ExecutionRouteResolutionState.ROUTER_ERROR
    assert missing.reason_code == "stale_eligibility"


def test_registration_captures_callback_and_descriptor_immutably():
    from agent.execution_router import ExecutionRouteDecisionV1
    from hermes_cli.execution_router_runtime import ExecutionRouteResolutionState, resolve_execution_route

    request = _contract()[0]
    registration = _registration(lambda value, _signal: ExecutionRouteDecisionV1.pass_through(
        request_id=value.request_id, attempt_id=value.attempt_id
    ))
    registration.provider.resolve_execution_route = lambda *_args: {"forged": True}
    registration.provider.descriptor = None
    result = resolve_execution_route(request, registration=registration)
    assert result.state is ExecutionRouteResolutionState.PASS_THROUGH


def test_runtime_renderer_failure_cannot_change_resolution():
    from hermes_cli.execution_router_runtime import ExecutionRouteResolutionState, resolve_execution_route

    def broken_renderer(_result):
        raise RuntimeError("renderer failed")

    result = resolve_execution_route(
        _contract()[0],
        registration=_registration(lambda _request, _cancellation: None),
        render_notice=broken_renderer,
    )
    assert result.state is ExecutionRouteResolutionState.PASS_THROUGH


def test_router_selected_fallback_emits_target_free_restart_before_runtime_mutation(monkeypatch):
    from types import SimpleNamespace

    from agent.chat_completion_helpers import try_activate_fallback
    from agent.execution_router import RoutedAttemptRestartRequired

    agent = SimpleNamespace(
        _execution_router_selected_attempt=True,
        _fallback_chain=[{"provider": "secret-provider", "model": "secret-model"}],
        _fallback_index=0,
        model="current-model",
        provider="current-provider",
        base_url="https://current.invalid",
        _rate_limited_until=0,
        _rate_limit_backoff_count=0,
    )
    monkeypatch.setattr("agent.fallback_cooldown._arm_rate_limit_cooldown", lambda *_args: None)

    assert try_activate_fallback(agent, "failed") is False
    signal = agent._routed_restart_required
    assert isinstance(signal, RoutedAttemptRestartRequired)
    assert signal.consumed_fallback_slot == 0
    assert signal.reason_code == "failed"
    assert not hasattr(signal, "provider")
    assert not hasattr(signal, "model")
    assert agent._fallback_index == 1
    assert (agent.model, agent.provider, agent.base_url) == (
        "current-model",
        "current-provider",
        "https://current.invalid",
    )
