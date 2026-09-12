"""Host-owned execution-router resolution without credential binding or dispatch."""

from __future__ import annotations

import inspect
import hashlib
import logging
import threading
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Protocol

from agent.execution_router import (
    ROUTER_DEADLINE_MS,
    MAX_INSTRUCTION_UTF8_BYTES,
    MAX_REASON_TEXT_UTF8_BYTES,
    ExecutionRouteDecisionKind,
    ExecutionRouteCandidateV1,
    ExecutionRouteDecisionV1,
    ExecutionRouteIdentityV1,
    ExecutionRouteInstructionV1,
    ExecutionRoutePinsV1,
    ExecutionRoutePreviousAttemptV1,
    ExecutionRouteRequestV1,
    ExecutionKind,
    CONTRACT_VERSION,
    canonical_json_bytes,
    ExecutionRouterProviderV1,
    ExecutionRouterCancellationSignalV1,
    _new_execution_router_cancellation,
    validate_decision,
    validate_provider,
    validate_request,
)

logger = logging.getLogger(__name__)
_IN_RESOLUTION: ContextVar[bool] = ContextVar("execution_router_resolution", default=False)


def _redact_router_instruction(value: str) -> str:
    try:
        from agent.redact import redact_sensitive_text

        return redact_sensitive_text(value, force=True, redact_url_credentials=True)
    except Exception:
        return "[REDACTED]"


def prepare_execution_route_request(
    request: ExecutionRouteRequestV1,
    *,
    raw_instruction: str | None,
    authorized_execution_kinds: frozenset,
    redact_instruction: Callable[[str], str],
) -> ExecutionRouteRequestV1:
    """Apply the common authorization/redaction/UTF-8 admission boundary."""
    if type(request) is not ExecutionRouteRequestV1:
        raise TypeError("request must be exact ExecutionRouteRequestV1")
    if request.execution_kind not in authorized_execution_kinds:
        raise PermissionError("instruction disclosure is not authorized for this execution kind")
    if raw_instruction is None:
        instruction = ExecutionRouteInstructionV1.absent()
    else:
        if type(raw_instruction) is not str:
            raise TypeError("raw_instruction must be str or None")
        redacted = redact_instruction(raw_instruction)
        if type(redacted) is not str:
            raise TypeError("instruction redactor must return str")
        encoded = redacted.encode("utf-8")
        delivered = encoded[:MAX_INSTRUCTION_UTF8_BYTES].decode("utf-8", errors="ignore")
        instruction = ExecutionRouteInstructionV1.from_text(
            delivered,
            original_utf8_bytes=len(encoded),
            truncated=len(encoded) > len(delivered.encode("utf-8")),
        )
    return replace(request, instruction=instruction, request_digest=None).with_computed_digest()


class ExecutionRouteResolutionState(str, Enum):
    NATIVE = "native"
    ROUTE = "route"
    PASS_THROUGH = "pass_through"
    STOP = "stop"
    ROUTER_ERROR = "router_error"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class ExecutionRouteResolution:
    state: ExecutionRouteResolutionState
    decision: ExecutionRouteDecisionV1 | None = None
    accepted_route: ExecutionRouteIdentityV1 | None = None
    reason_code: str | None = None
    reason_text: str | None = None


@dataclass(frozen=True)
class MainTurnAttemptPreparation:
    """Credential-free result consumed by existing main-turn construction paths."""

    request: ExecutionRouteRequestV1
    resolution: ExecutionRouteResolution
    selected_route: ExecutionRouteIdentityV1 | None
    route_signature: tuple[Any, ...] | None
    lifecycle: Any = field(default=None, repr=False, compare=False)

    @property
    def may_start(self) -> bool:
        return self.selected_route is not None

    @property
    def selected_model(self) -> str | None:
        return self.selected_route.model if self.selected_route is not None else None

    @property
    def selected_provider(self) -> str | None:
        return self.selected_route.provider if self.selected_route is not None else None

    @property
    def selected_reasoning(self) -> str | None:
        return self.selected_route.reasoning if self.selected_route is not None else None


@dataclass(frozen=True)
class _KanbanWorkerAttemptPreparation:
    request: ExecutionRouteRequestV1
    resolution: ExecutionRouteResolution
    selected_route: ExecutionRouteIdentityV1 | None

    @property
    def may_start(self) -> bool:
        return self.selected_route is not None


def _prepare_kanban_worker_attempt(
    *,
    raw_instruction: str | None,
    task_id: str,
    native_candidate: ExecutionRouteCandidateV1,
    eligible_candidates: tuple[ExecutionRouteCandidateV1, ...],
    explicit_model_pin: str | None,
    explicit_provider_pin: str | None,
    explicit_reasoning_pin: str | None,
    previous_attempt: ExecutionRoutePreviousAttemptV1 | None,
    registration: "ExecutionRouterRegistration",
    lifecycle: Any = None,
) -> _KanbanWorkerAttemptPreparation:
    """Resolve one reserved Kanban attempt before its run is opened."""
    pins = ExecutionRoutePinsV1(
        model=explicit_model_pin,
        provider=explicit_provider_pin,
        reasoning=explicit_reasoning_pin,
    )
    candidates = tuple(
        candidate for candidate in eligible_candidates
        if (pins.model is None or candidate.model == pins.model)
        and (pins.provider is None or candidate.provider == pins.provider)
        and (pins.reasoning is None or candidate.reasoning == pins.reasoning)
    )
    if native_candidate not in candidates:
        raise ValueError("native Kanban candidate must satisfy every explicit task pin")
    request = ExecutionRouteRequestV1(
        contract_version=CONTRACT_VERSION,
        request_id=uuid.uuid4().hex,
        root_id=hashlib.sha256(task_id.encode("utf-8")).hexdigest(),
        task_id=task_id,
        execution_id=uuid.uuid4().hex,
        attempt_id=uuid.uuid4().hex,
        execution_kind=ExecutionKind.KANBAN_WORKER,
        surface_class="kanban_dispatcher",
        instruction=ExecutionRouteInstructionV1.absent(),
        pins=pins,
        native_candidate_id=native_candidate.candidate_id,
        eligibility_revision=hashlib.sha256(
            canonical_json_bytes(candidates) + b"\n" + canonical_json_bytes(pins)
        ).hexdigest(),
        eligible_candidates=candidates,
        previous_attempt=previous_attempt,
        request_digest=None,
    )
    request = prepare_execution_route_request(
        request,
        raw_instruction=raw_instruction,
        authorized_execution_kinds=frozenset({ExecutionKind.KANBAN_WORKER}),
        redact_instruction=_redact_router_instruction,
    )
    resolution = resolve_execution_route(
        request,
        registration=registration,
        lifecycle=lifecycle,
        revalidate=lambda _request, accepted: accepted in tuple(
            candidate.identity() for candidate in candidates
        ),
    )
    selected = (
        resolution.accepted_route
        if resolution.state is ExecutionRouteResolutionState.ROUTE
        else native_candidate.identity()
        if resolution.state is ExecutionRouteResolutionState.PASS_THROUGH
        else None
    )
    return _KanbanWorkerAttemptPreparation(request, resolution, selected)


@dataclass(frozen=True)
class _NativeChildAttemptPreparation:
    request: ExecutionRouteRequestV1
    resolution: ExecutionRouteResolution
    selected_route: ExecutionRouteIdentityV1 | None
    lifecycle: Any = field(default=None, repr=False, compare=False)

    @property
    def may_start(self) -> bool:
        return self.selected_route is not None


def _prepare_native_child_attempt(
    *,
    raw_instruction: str,
    surface_class: str,
    session_id: str,
    task_index: int,
    native_candidate_id: str,
    eligible_candidates: tuple[ExecutionRouteCandidateV1, ...],
    explicit_model_pin: str | None = None,
    explicit_provider_pin: str | None = None,
    explicit_reasoning_pin: str | None = None,
    registration: ExecutionRouterRegistration,
    lifecycle: ExecutionRouteLifecycleAuthority | None = None,
    render_notice: Callable[[ExecutionRouteRequestV1, ExecutionRouteResolution], Any] | None = None,
) -> _NativeChildAttemptPreparation:
    """Resolve one initial native child before credentials or construction."""
    pins = ExecutionRoutePinsV1(
        model=explicit_model_pin,
        provider=explicit_provider_pin,
        reasoning=explicit_reasoning_pin,
    )
    candidates = tuple(
        candidate for candidate in eligible_candidates
        if (pins.model is None or candidate.model == pins.model)
        and (pins.provider is None or candidate.provider == pins.provider)
        and (pins.reasoning is None or candidate.reasoning == pins.reasoning)
    )
    native = next((item for item in candidates if item.candidate_id == native_candidate_id), None)
    if native is None:
        raise ValueError("native child candidate must satisfy every explicit delegation pin")
    eligibility_revision = hashlib.sha256(
        canonical_json_bytes(candidates) + b"\n" + canonical_json_bytes(pins)
    ).hexdigest()
    request = ExecutionRouteRequestV1(
        contract_version=CONTRACT_VERSION,
        request_id=uuid.uuid4().hex,
        root_id=hashlib.sha256(session_id.encode("utf-8")).hexdigest(),
        task_id=hashlib.sha256(f"{session_id}:{task_index}".encode("utf-8")).hexdigest(),
        execution_id=uuid.uuid4().hex,
        attempt_id=uuid.uuid4().hex,
        execution_kind=ExecutionKind.NATIVE_CHILD,
        surface_class=surface_class,
        instruction=ExecutionRouteInstructionV1.absent(),
        pins=pins,
        native_candidate_id=native_candidate_id,
        eligibility_revision=eligibility_revision,
        eligible_candidates=candidates,
        previous_attempt=None,
        request_digest=None,
    )
    request = prepare_execution_route_request(
        request,
        raw_instruction=raw_instruction,
        authorized_execution_kinds=frozenset({ExecutionKind.NATIVE_CHILD}),
        redact_instruction=_redact_router_instruction,
    )
    resolution = resolve_execution_route(
        request,
        registration=registration,
        lifecycle=lifecycle,
        revalidate=lambda _request, accepted: accepted in tuple(item.identity() for item in candidates),
    )
    if render_notice is not None:
        try:
            render_notice(request, resolution)
        except Exception:
            logger.warning("execution-router native-child notice renderer failed", exc_info=True)
    selected = (
        resolution.accepted_route
        if resolution.state is ExecutionRouteResolutionState.ROUTE
        else native.identity()
        if resolution.state is ExecutionRouteResolutionState.PASS_THROUGH
        else None
    )
    return _NativeChildAttemptPreparation(request, resolution, selected, lifecycle)


def _start_native_child_attempt(prepared: _NativeChildAttemptPreparation, child: Any) -> Any:
    """Persist the actual route only after the existing builder constructed the child."""
    if prepared.lifecycle is None or prepared.selected_route is None:
        raise RuntimeError("active native-child routing requires parent SessionDB lifecycle")
    reasoning_config = getattr(child, "reasoning_config", None)
    reasoning = (
        reasoning_config.get("effort")
        if isinstance(reasoning_config, dict) and reasoning_config.get("enabled", True)
        else None
    )
    actual = ExecutionRouteIdentityV1(
        candidate_id=prepared.selected_route.candidate_id,
        provider=str(getattr(child, "requested_provider", None) or getattr(child, "provider", None) or "auto"),
        model=str(getattr(child, "model", None) or "auto"),
        reasoning=reasoning,
    )
    started = prepared.lifecycle.record_started(
        actual,
        request_id=prepared.request.request_id,
        attempt_id=prepared.request.attempt_id,
    )
    child._execution_router_native_child_attempt = prepared
    if prepared.resolution.state is ExecutionRouteResolutionState.ROUTE:
        child._execution_router_selected_attempt = True
    return started


def _finish_native_child_attempt(
    prepared: _NativeChildAttemptPreparation,
    terminal_state: str,
) -> Any:
    """Finish one started active-router child before its normal cleanup."""
    if prepared.lifecycle is None:
        raise RuntimeError("active native-child routing requires parent SessionDB lifecycle")
    return prepared.lifecycle.record_finished(
        terminal_state,
        request_id=prepared.request.request_id,
        attempt_id=prepared.request.attempt_id,
    )


def start_main_turn_attempt(
    prepared: MainTurnAttemptPreparation,
    lifecycle: Any,
    actual_route: ExecutionRouteIdentityV1,
) -> Any:
    """Record executor construction for an active-router main-turn attempt."""
    if prepared.resolution.state not in {
        ExecutionRouteResolutionState.ROUTE,
        ExecutionRouteResolutionState.PASS_THROUGH,
    }:
        return None
    return lifecycle.record_started(
        actual_route,
        request_id=prepared.request.request_id,
        attempt_id=prepared.request.attempt_id,
    )


def finish_main_turn_attempt(
    prepared: MainTurnAttemptPreparation,
    lifecycle: Any,
    terminal_state: str,
) -> Any:
    """Record the terminal state for an active-router main-turn attempt."""
    if prepared.resolution.state not in {
        ExecutionRouteResolutionState.ROUTE,
        ExecutionRouteResolutionState.PASS_THROUGH,
    }:
        return None
    return lifecycle.record_finished(
        terminal_state,
        request_id=prepared.request.request_id,
        attempt_id=prepared.request.attempt_id,
    )


def main_turn_actual_route(
    prepared: MainTurnAttemptPreparation,
    agent: Any,
) -> ExecutionRouteIdentityV1:
    """Build the actual start receipt from the newly constructed or reused agent."""
    if prepared.selected_route is None:
        raise ValueError("main-turn attempt has no selected route")

    reasoning_config = getattr(agent, "reasoning_config", None)
    reasoning = (
        reasoning_config.get("effort")
        if isinstance(reasoning_config, dict) and reasoning_config.get("enabled", True)
        else None
    )
    actual = ExecutionRouteIdentityV1(
        candidate_id=prepared.selected_route.candidate_id,
        provider=str(getattr(agent, "requested_provider", None) or getattr(agent, "provider", None) or "auto"),
        model=str(getattr(agent, "model", None) or "auto"),
        reasoning=reasoning,
    )
    if actual.provider != prepared.selected_route.provider:
        raise ValueError("constructed agent does not match selected provider")
    if actual.model != prepared.selected_route.model:
        raise ValueError("constructed agent does not match selected model")
    if actual.reasoning != prepared.selected_route.reasoning:
        raise ValueError("constructed agent does not match selected reasoning")
    return actual


def record_main_turn_started(prepared: MainTurnAttemptPreparation | None, agent: Any) -> None:
    """Apply host persistence policy while recording the actual constructed route."""
    if prepared is None or prepared.lifecycle is None:
        return
    start_main_turn_attempt(prepared, prepared.lifecycle, main_turn_actual_route(prepared, agent))


def bind_main_turn_attempt(
    prepared: MainTurnAttemptPreparation | None,
    agent: Any,
    eligible_candidates: tuple[ExecutionRouteCandidateV1, ...],
) -> None:
    """Attach routed-attempt fencing and align existing fallback authority to its route."""
    if (
        prepared is None
        or prepared.resolution.state is not ExecutionRouteResolutionState.ROUTE
    ):
        return
    agent._execution_router_selected_attempt = True
    agent._routed_route_signature = prepared.route_signature
    agent._routed_restart_required = None
    if prepared.selected_route is None:
        return
    selected_id = prepared.selected_route.candidate_id
    fallback_candidates = tuple(
        item for item in eligible_candidates if item.candidate_id.startswith("fallback-")
    )
    for index, candidate in enumerate(fallback_candidates):
        if candidate.candidate_id == selected_id:
            agent._fallback_index = max(int(getattr(agent, "_fallback_index", 0)), index)
            return


def record_main_turn_not_started(
    prepared: MainTurnAttemptPreparation | None,
    reason_code: str,
    reason_text: str | None = None,
) -> None:
    """Record a post-decision credential/construction refusal without changing host policy."""
    if (
        prepared is None
        or prepared.lifecycle is None
        or prepared.resolution.state not in {
            ExecutionRouteResolutionState.ROUTE,
            ExecutionRouteResolutionState.PASS_THROUGH,
        }
    ):
        return
    try:
        prepared.lifecycle.record_not_started(
            reason_code,
            reason_text,
            request_id=prepared.request.request_id,
            attempt_id=prepared.request.attempt_id,
        )
    except Exception:
        logger.warning("execution-router route-not-started persistence failed", exc_info=True)


def record_main_turn_finished(
    prepared: MainTurnAttemptPreparation | None,
    result: Any,
) -> None:
    """Persist one terminal state without changing the surface's completion semantics."""
    if prepared is None or prepared.lifecycle is None:
        return
    terminal_state = (
        "interrupted" if isinstance(result, dict) and result.get("interrupted")
        else "completed" if isinstance(result, dict) and result.get("completed")
        else "failed"
    )
    try:
        finish_main_turn_attempt(prepared, prepared.lifecycle, terminal_state)
    except Exception:
        logger.warning("execution-router route-finish persistence failed", exc_info=True)


def record_main_turn_routed_restart(
    prepared: MainTurnAttemptPreparation | None,
) -> None:
    """Close the old route before a continuation attempt receives fresh identities."""
    if prepared is None or prepared.lifecycle is None:
        return
    finish_main_turn_attempt(prepared, prepared.lifecycle, "routed_restart_required")


def prepare_main_turn_attempt(
    *,
    raw_instruction: str | None,
    surface_class: str,
    session_id: str,
    native_candidate_id: str,
    eligible_candidates: tuple[ExecutionRouteCandidateV1, ...],
    explicit_model_pin: str | None = None,
    explicit_provider_pin: str | None = None,
    explicit_reasoning_pin: str | None = None,
    registration: ExecutionRouterRegistration | None = None,
    lifecycle: ExecutionRouteLifecycleAuthority | None = None,
    render_notice: Callable[[ExecutionRouteResolution], Any] | None = None,
    revalidate: Callable[[ExecutionRouteRequestV1, ExecutionRouteIdentityV1], bool] | None = None,
    previous_attempt: Any = None,
) -> MainTurnAttemptPreparation:
    """Resolve one main-turn attempt without credentials, agent creation, or dispatch."""
    pins = ExecutionRoutePinsV1(
        model=explicit_model_pin,
        provider=explicit_provider_pin,
        reasoning=explicit_reasoning_pin,
    )
    candidates = tuple(
        candidate for candidate in eligible_candidates
        if (pins.model is None or candidate.model == pins.model)
        and (pins.provider is None or candidate.provider == pins.provider)
        and (pins.reasoning is None or candidate.reasoning == pins.reasoning)
    )
    native = next((item for item in candidates if item.candidate_id == native_candidate_id), None)
    if native is None:
        raise ValueError("native candidate must satisfy every explicit main-turn pin")
    eligibility_revision = hashlib.sha256(
        canonical_json_bytes(candidates) + b"\n" + canonical_json_bytes(pins)
    ).hexdigest()
    request = ExecutionRouteRequestV1(
        contract_version=CONTRACT_VERSION,
        request_id=uuid.uuid4().hex,
        root_id=hashlib.sha256(session_id.encode("utf-8")).hexdigest(),
        task_id=None,
        execution_id=uuid.uuid4().hex,
        attempt_id=uuid.uuid4().hex,
        execution_kind=ExecutionKind.MAIN_TURN,
        surface_class=surface_class,
        instruction=ExecutionRouteInstructionV1.absent(),
        pins=pins,
        native_candidate_id=native_candidate_id,
        eligibility_revision=eligibility_revision,
        eligible_candidates=candidates,
        previous_attempt=previous_attempt,
        request_digest=None,
    )
    request = prepare_execution_route_request(
        request,
        raw_instruction=raw_instruction,
        authorized_execution_kinds=frozenset({ExecutionKind.MAIN_TURN}),
        redact_instruction=_redact_router_instruction,
    )
    resolution = resolve_execution_route(
        request,
        registration=registration,
        lifecycle=lifecycle,
        revalidate=revalidate,
        render_notice=render_notice,
    )
    selected = (
        resolution.accepted_route
        if resolution.state is ExecutionRouteResolutionState.ROUTE
        else native.identity()
        if resolution.state in {
            ExecutionRouteResolutionState.NATIVE,
            ExecutionRouteResolutionState.PASS_THROUGH,
        }
        else None
    )
    generation = registration.generation if registration is not None else None
    contract_generation = (
        registration.descriptor.contract_version if registration is not None else None
    )
    signature = (
        selected.provider,
        selected.model,
        selected.reasoning,
        generation,
        contract_generation,
    ) if selected is not None else None
    return MainTurnAttemptPreparation(request, resolution, selected, signature, lifecycle)


def _prepare_main_turn_continuation_attempt(
    continuation_record: Any,
    *,
    raw_instruction: str | None,
    surface_class: str,
    eligible_candidates: tuple[ExecutionRouteCandidateV1, ...],
    registration: ExecutionRouterRegistration | None,
    lifecycle: ExecutionRouteLifecycleAuthority | None = None,
    render_notice: Callable[[ExecutionRouteResolution], Any] | None = None,
) -> MainTurnAttemptPreparation:
    """Validate a carried fallback cursor before issuing fresh route identities."""
    cursor = getattr(continuation_record, "next_fallback_cursor", None)
    if type(cursor) is not int or cursor < 0:
        raise RuntimeError("main-turn fallback cursor is invalid")
    fallbacks = tuple(
        candidate for candidate in eligible_candidates
        if candidate.candidate_id.startswith("fallback-")
    )
    remaining = fallbacks[cursor:]
    if not remaining:
        raise RuntimeError("main-turn fallback budget is exhausted")
    native = remaining[0]
    previous = ExecutionRoutePreviousAttemptV1(
        attempt_id=str(continuation_record.old_attempt_id),
        terminal_state="routed_restart_required",
        route=None,
        reason_text=str(continuation_record.terminal_reason)[:1024],
    )
    return prepare_main_turn_attempt(
        raw_instruction=raw_instruction,
        surface_class=surface_class,
        session_id=str(continuation_record.session_id),
        native_candidate_id=native.candidate_id,
        eligible_candidates=remaining,
        registration=registration,
        lifecycle=lifecycle,
        render_notice=render_notice,
        revalidate=lambda _request, accepted: accepted in tuple(
            candidate.identity() for candidate in remaining
        ),
        previous_attempt=previous,
    )


@dataclass(frozen=True)
class ExecutionRouterRegistration:
    provider: ExecutionRouterProviderV1
    generation: int
    is_current: Callable[[int], bool]
    _revoked: threading.Event = field(default_factory=threading.Event, repr=False, compare=False)
    descriptor: Any = field(init=False)
    callback: Callable[..., ExecutionRouteDecisionV1 | None] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        validate_provider(self.provider)
        object.__setattr__(self, "descriptor", self.provider.descriptor)
        object.__setattr__(self, "callback", self.provider.resolve_execution_route)

    def revoke(self) -> None:
        self._revoked.set()

    def is_revoked(self) -> bool:
        return self._revoked.is_set()


class ExecutionRouteLifecycleAuthority(Protocol):
    def get_execution_route_resolution(
        self, request_id: str, attempt_id: str
    ) -> ExecutionRouteResolution | None: ...

    def record_execution_route_requested(
        self, request: ExecutionRouteRequestV1, registration: ExecutionRouterRegistration
    ) -> bool: ...

    def record_execution_route_result(
        self,
        request: ExecutionRouteRequestV1,
        result: ExecutionRouteResolution,
        registration: ExecutionRouterRegistration,
    ) -> ExecutionRouteResolution: ...


def _error(code: str, text: str) -> ExecutionRouteResolution:
    return ExecutionRouteResolution(
        ExecutionRouteResolutionState.ROUTER_ERROR,
        reason_code=code,
        reason_text=text,
    )


def _registration_is_live(registration: ExecutionRouterRegistration) -> bool:
    return (
        not registration.is_revoked()
        and registration.is_current(registration.generation)
    )


def _settle(
    request: ExecutionRouteRequestV1,
    result: ExecutionRouteResolution,
    registration: ExecutionRouterRegistration,
    lifecycle: ExecutionRouteLifecycleAuthority | None,
    render_notice: Callable[[ExecutionRouteResolution], Any] | None,
) -> ExecutionRouteResolution:
    updates = {}
    if result.reason_text is not None:
        redacted = _redact_router_instruction(result.reason_text)
        bounded = redacted.encode("utf-8")[:MAX_REASON_TEXT_UTF8_BYTES].decode("utf-8", errors="ignore")
        updates["reason_text"] = bounded
    if result.decision is not None and result.decision.reason_text is not None:
        redacted = _redact_router_instruction(result.decision.reason_text)
        bounded = redacted.encode("utf-8")[:MAX_REASON_TEXT_UTF8_BYTES].decode("utf-8", errors="ignore")
        updates["decision"] = replace(result.decision, reason_text=bounded)
    if updates:
        result = replace(result, **updates)
    if lifecycle is not None:
        result = lifecycle.record_execution_route_result(request, result, registration)
    if render_notice is not None:
        try:
            render_notice(result)
        except Exception:
            logger.warning("execution-router notice renderer failed", exc_info=True)
    return result


def _preflight(
    request: ExecutionRouteRequestV1,
    registration: ExecutionRouterRegistration,
) -> ExecutionRouteResolution | None:
    try:
        validate_request(request)
    except (TypeError, ValueError) as exc:
        return _error("invalid_request", str(exc))
    descriptor = registration.descriptor
    if request.execution_kind not in descriptor.supported_execution_kinds:
        return ExecutionRouteResolution(
            ExecutionRouteResolutionState.UNSUPPORTED,
            reason_code="unsupported_kind",
            reason_text="execution kind is not supported by the active router",
        )
    if not _registration_is_live(registration):
        return _error("stale_generation", "execution router generation is no longer active")
    return None


def resolve_execution_route(
    request: ExecutionRouteRequestV1,
    *,
    registration: ExecutionRouterRegistration | None = None,
    lifecycle: ExecutionRouteLifecycleAuthority | None = None,
    revalidate: Callable[[ExecutionRouteRequestV1, ExecutionRouteIdentityV1], bool] | None = None,
    render_notice: Callable[[ExecutionRouteResolution], Any] | None = None,
) -> ExecutionRouteResolution:
    """Resolve one request. This function never reads credentials or creates an executor."""
    if registration is None:
        return ExecutionRouteResolution(ExecutionRouteResolutionState.NATIVE)
    if _IN_RESOLUTION.get():
        return _error("recursion", "nested execution-route resolution is not allowed")

    preflight = _preflight(request, registration)
    if preflight is not None:
        return preflight
    if lifecycle is not None:
        try:
            owns_resolution = lifecycle.record_execution_route_requested(request, registration)
        except ValueError:
            return _error(
                "conflicting_replay",
                "execution route request identity is already bound differently",
            )
        if owns_resolution is False:
            existing = lifecycle.get_execution_route_resolution(request.request_id, request.attempt_id)
            if existing is not None:
                return existing
            wait_deadline = time.monotonic() + (ROUTER_DEADLINE_MS / 1000)
            while time.monotonic() < wait_deadline:
                existing = lifecycle.get_execution_route_resolution(request.request_id, request.attempt_id)
                if existing is not None:
                    return existing
                time.sleep(0.001)
            existing = lifecycle.get_execution_route_resolution(request.request_id, request.attempt_id)
            if existing is not None:
                return existing
            return _error("resolution_in_progress", "execution route resolution is still in progress")

    lock = threading.Lock()
    complete = threading.Event()
    state: dict[str, Any] = {"open": True}
    deadline = time.monotonic() + (ROUTER_DEADLINE_MS / 1000)
    cancellation, cancellation_controller = _new_execution_router_cancellation()

    def invoke() -> None:
        token = _IN_RESOLUTION.set(True)
        try:
            try:
                value = registration.callback(request, cancellation)
                payload = ("value", value)
            except BaseException as exc:
                payload = ("error", exc)
            with lock:
                if state["open"] and time.monotonic() < deadline and _registration_is_live(registration):
                    state["payload"] = payload
                    state["open"] = False
        finally:
            _IN_RESOLUTION.reset(token)
            complete.set()

    worker = threading.Thread(
        target=invoke,
        name=f"execution-router:{request.request_id}",
        daemon=True,
    )
    worker.start()
    while not complete.is_set() and _registration_is_live(registration):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        complete.wait(min(remaining, 0.01))
    with lock:
        if state["open"]:
            state["open"] = False
            payload = None
        else:
            payload = state.get("payload")

    if payload is None or not _registration_is_live(registration):
        cancellation_controller.cancel()
    cancellation_controller.close()

    if payload is None and not _registration_is_live(registration):
        return _settle(
            request,
            _error("stale_generation", "execution router generation changed during resolution"),
            registration,
            lifecycle,
            render_notice,
        )
    if payload is None:
        return _settle(
            request,
            _error("timeout", "execution router did not respond within 250 ms"),
            registration,
            lifecycle,
            render_notice,
        )
    if not _registration_is_live(registration):
        return _settle(
            request,
            _error("stale_generation", "execution router generation changed during resolution"),
            registration,
            lifecycle,
            render_notice,
        )
    kind, value = payload
    if kind == "error":
        return _settle(
            request,
            _error("provider_error", "execution router failed"),
            registration,
            lifecycle,
            render_notice,
        )
    if value is None:
        decision = ExecutionRouteDecisionV1.pass_through(
            request_id=request.request_id,
            attempt_id=request.attempt_id,
        )
    elif type(value) is not ExecutionRouteDecisionV1 or inspect.isawaitable(value):
        return _settle(
            request,
            _error("malformed_decision", "execution router returned an unsupported value"),
            registration,
            lifecycle,
            render_notice,
        )
    else:
        decision = value
    try:
        validate_decision(request, decision)
    except (TypeError, ValueError):
        return _settle(
            request,
            _error("malformed_decision", "execution router returned an invalid decision"),
            registration,
            lifecycle,
            render_notice,
        )

    if decision.kind is ExecutionRouteDecisionKind.ROUTE:
        candidate = next(
            item for item in request.eligible_candidates if item.candidate_id == decision.candidate_id
        )
        accepted = candidate.identity()
        if revalidate is not None:
            try:
                fresh = revalidate(request, accepted)
            except Exception:
                fresh = False
            if not fresh:
                return _settle(
                    request,
                    _error("stale_eligibility", "execution route eligibility is no longer current"),
                    registration,
                    lifecycle,
                    render_notice,
                )
        else:
            return _settle(
                request,
                _error("stale_eligibility", "execution route eligibility was not revalidated"),
                registration,
                lifecycle,
                render_notice,
            )
        result = ExecutionRouteResolution(
            ExecutionRouteResolutionState.ROUTE,
            decision=decision,
            accepted_route=accepted,
            reason_code=decision.reason_code,
            reason_text=decision.reason_text,
        )
    elif decision.kind is ExecutionRouteDecisionKind.PASS_THROUGH:
        result = ExecutionRouteResolution(
            ExecutionRouteResolutionState.PASS_THROUGH,
            decision=decision,
            reason_code=decision.reason_code,
            reason_text=decision.reason_text,
        )
    else:
        result = ExecutionRouteResolution(
            ExecutionRouteResolutionState.STOP,
            decision=decision,
            reason_code=decision.reason_code,
            reason_text=decision.reason_text,
        )
    return _settle(request, result, registration, lifecycle, render_notice)
