"""Public, policy-neutral execution-router contract version 1.0."""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import re
import secrets
import threading
import time
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Protocol, runtime_checkable

CONTRACT_VERSION = "1.0"
ROUTER_DEADLINE_MS = 250
MAX_INSTRUCTION_UTF8_BYTES = 16_384
MAX_SERIALIZED_REQUEST_BYTES = 65_536
MAX_ELIGIBLE_CANDIDATES = 128
MAX_OPAQUE_ID_ASCII_CHARS = 128
MAX_SURFACE_CLASS_ASCII_CHARS = 64
MAX_REASON_CODE_ASCII_CHARS = 64
MAX_REASON_TEXT_UTF8_BYTES = 512
MAX_EVENT_PAGE_RECORDS = 100
MAX_CAPABILITY_ATTRIBUTES = 32
MAX_CAPABILITY_ATTRIBUTE_NAME_ASCII_CHARS = 64
MAX_CAPABILITY_ATTRIBUTE_STRING_UTF8_BYTES = 256
MAX_JSON_SAFE_INTEGER = 9_007_199_254_740_991

_ASCII_TOKEN = re.compile(r"^[\x21-\x7e]+$")


class ExecutionKind(str, Enum):
    MAIN_TURN = "main_turn"
    NATIVE_CHILD = "native_child"
    KANBAN_WORKER = "kanban_worker"


class ExecutionRouteDecisionKind(str, Enum):
    ROUTE = "route"
    PASS_THROUGH = "pass_through"
    STOP = "stop"
    ROUTER_ERROR = "router_error"


class ExecutionRouteEventType(str, Enum):
    REQUESTED = "route_requested"
    ACCEPTED = "route_accepted"
    STARTED = "route_started"
    NOT_STARTED = "route_not_started"
    FINISHED = "route_finished"


@dataclass(frozen=True)
class RoutedAttemptRestartRequired:
    """Host-only receipt: existing fallback authority consumed one slot; no target is carried."""

    reason_code: str
    consumed_fallback_slot: int

    def __post_init__(self) -> None:
        _require_ascii(self.reason_code, "reason_code", MAX_REASON_CODE_ASCII_CHARS)
        _require_integer(self.consumed_fallback_slot, "consumed_fallback_slot")


def _require_exact_str(value: Any, field: str, *, max_bytes: int | None = None) -> None:
    if type(value) is not str:
        raise TypeError(f"{field} must be str")
    if max_bytes is not None and len(value.encode("utf-8")) > max_bytes:
        raise ValueError(f"{field} exceeds {max_bytes} UTF-8 bytes")


def _require_ascii(value: Any, field: str, maximum: int, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    _require_exact_str(value, field)
    if not value or len(value) > maximum or _ASCII_TOKEN.fullmatch(value) is None:
        raise ValueError(f"{field} must be 1..{maximum} printable ASCII characters")


def _require_integer(value: Any, field: str, maximum: int = MAX_JSON_SAFE_INTEGER) -> None:
    if type(value) is not int:
        raise TypeError(f"{field} must be an integer")
    if not 0 <= value <= maximum:
        raise ValueError(f"{field} must be in 0..{maximum}")


def _enum(value: Any, enum_type: type[Enum], field: str) -> None:
    if type(value) is not enum_type:
        raise TypeError(f"{field} must be {enum_type.__name__}")


@dataclass(frozen=True)
class ExecutionRouteCapabilityAttributeV1:
    name: str
    value: str | int | bool

    def __post_init__(self) -> None:
        _require_ascii(self.name, "attribute.name", MAX_CAPABILITY_ATTRIBUTE_NAME_ASCII_CHARS)
        if type(self.value) is str:
            _require_exact_str(
                self.value,
                "attribute.value",
                max_bytes=MAX_CAPABILITY_ATTRIBUTE_STRING_UTF8_BYTES,
            )
        elif type(self.value) is int:
            _require_integer(self.value, "attribute.value")
        elif type(self.value) is not bool:
            raise TypeError("attribute.value must be str, int, or bool")


@dataclass(frozen=True)
class ExecutionRouteIdentityV1:
    candidate_id: str
    provider: str
    model: str
    reasoning: str | None

    def __post_init__(self) -> None:
        _require_ascii(self.candidate_id, "candidate_id", MAX_OPAQUE_ID_ASCII_CHARS)
        _require_ascii(self.provider, "provider", MAX_OPAQUE_ID_ASCII_CHARS)
        _require_ascii(self.model, "model", MAX_OPAQUE_ID_ASCII_CHARS)
        if self.reasoning is not None:
            _require_ascii(self.reasoning, "reasoning", MAX_OPAQUE_ID_ASCII_CHARS)


@dataclass(frozen=True)
class ExecutionRouteCandidateV1:
    candidate_id: str
    provider: str
    model: str
    reasoning: str | None
    attributes: tuple[ExecutionRouteCapabilityAttributeV1, ...] = ()

    def __post_init__(self) -> None:
        ExecutionRouteIdentityV1(self.candidate_id, self.provider, self.model, self.reasoning)
        if type(self.attributes) is not tuple:
            raise TypeError("attributes must be a tuple")
        if len(self.attributes) > MAX_CAPABILITY_ATTRIBUTES:
            raise ValueError("too many capability attributes")
        if any(type(item) is not ExecutionRouteCapabilityAttributeV1 for item in self.attributes):
            raise TypeError("attributes must contain exact ExecutionRouteCapabilityAttributeV1 values")
        names = tuple(item.name for item in self.attributes)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ValueError("attribute names must be unique and lexicographically sorted")

    def identity(self) -> ExecutionRouteIdentityV1:
        return ExecutionRouteIdentityV1(self.candidate_id, self.provider, self.model, self.reasoning)


@dataclass(frozen=True)
class ExecutionRoutePinsV1:
    model: str | None = None
    provider: str | None = None
    reasoning: str | None = None

    def __post_init__(self) -> None:
        for name in ("model", "provider", "reasoning"):
            value = getattr(self, name)
            if value is not None:
                _require_ascii(value, f"pins.{name}", MAX_OPAQUE_ID_ASCII_CHARS)


@dataclass(frozen=True)
class ExecutionRouteInstructionV1:
    text: str | None
    truncated: bool
    original_utf8_bytes: int
    digest_algorithm: str
    digest: str | None

    def __post_init__(self) -> None:
        if type(self.truncated) is not bool:
            raise TypeError("instruction.truncated must be bool")
        _require_integer(self.original_utf8_bytes, "instruction.original_utf8_bytes")
        if self.digest_algorithm != "sha256":
            raise ValueError("instruction.digest_algorithm must be sha256")
        if self.text is None:
            if self.truncated or self.original_utf8_bytes != 0 or self.digest is not None:
                raise ValueError("absent instruction must use the canonical null shape")
            return
        _require_exact_str(self.text, "instruction.text", max_bytes=MAX_INSTRUCTION_UTF8_BYTES)
        delivered = self.text.encode("utf-8")
        expected = hashlib.sha256(delivered).hexdigest()
        if self.digest != expected:
            raise ValueError("instruction digest mismatch")
        if self.original_utf8_bytes < len(delivered):
            raise ValueError("instruction original size is smaller than delivered text")
        if self.truncated != (self.original_utf8_bytes > len(delivered)):
            raise ValueError("instruction truncation flag does not match byte sizes")

    @classmethod
    def absent(cls) -> "ExecutionRouteInstructionV1":
        return cls(None, False, 0, "sha256", None)

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        original_utf8_bytes: int | None = None,
        truncated: bool | None = None,
    ) -> "ExecutionRouteInstructionV1":
        _require_exact_str(text, "instruction.text")
        delivered = text.encode("utf-8")
        original = len(delivered) if original_utf8_bytes is None else original_utf8_bytes
        is_truncated = original > len(delivered) if truncated is None else truncated
        return cls(text, is_truncated, original, "sha256", hashlib.sha256(delivered).hexdigest())


@dataclass(frozen=True)
class ExecutionRoutePreviousAttemptV1:
    attempt_id: str
    terminal_state: str
    route: ExecutionRouteIdentityV1 | None
    reason_text: str | None

    def __post_init__(self) -> None:
        _require_ascii(self.attempt_id, "previous_attempt.attempt_id", MAX_OPAQUE_ID_ASCII_CHARS)
        _require_ascii(self.terminal_state, "previous_attempt.terminal_state", MAX_REASON_CODE_ASCII_CHARS)
        if self.route is not None and type(self.route) is not ExecutionRouteIdentityV1:
            raise TypeError("previous_attempt.route must be ExecutionRouteIdentityV1")
        if self.reason_text is not None:
            _require_exact_str(
                self.reason_text,
                "previous_attempt.reason_text",
                max_bytes=MAX_REASON_TEXT_UTF8_BYTES,
            )


@dataclass(frozen=True)
class ExecutionRouteRequestV1:
    contract_version: str
    request_id: str
    root_id: str
    task_id: str | None
    execution_id: str
    attempt_id: str
    execution_kind: ExecutionKind
    surface_class: str
    instruction: ExecutionRouteInstructionV1
    pins: ExecutionRoutePinsV1
    native_candidate_id: str | None
    eligibility_revision: str
    eligible_candidates: tuple[ExecutionRouteCandidateV1, ...]
    previous_attempt: ExecutionRoutePreviousAttemptV1 | None
    request_digest: str | None

    def __post_init__(self) -> None:
        if self.contract_version != CONTRACT_VERSION:
            raise ValueError("unsupported execution-router contract version")
        for field_name in ("request_id", "root_id", "execution_id", "attempt_id"):
            _require_ascii(getattr(self, field_name), field_name, MAX_OPAQUE_ID_ASCII_CHARS)
        _require_ascii(self.task_id, "task_id", MAX_OPAQUE_ID_ASCII_CHARS, nullable=True)
        _enum(self.execution_kind, ExecutionKind, "execution_kind")
        _require_ascii(self.surface_class, "surface_class", MAX_SURFACE_CLASS_ASCII_CHARS)
        if type(self.instruction) is not ExecutionRouteInstructionV1:
            raise TypeError("instruction must be exact ExecutionRouteInstructionV1")
        if type(self.pins) is not ExecutionRoutePinsV1:
            raise TypeError("pins must be exact ExecutionRoutePinsV1")
        _require_ascii(
            self.native_candidate_id,
            "native_candidate_id",
            MAX_OPAQUE_ID_ASCII_CHARS,
            nullable=True,
        )
        _require_ascii(self.eligibility_revision, "eligibility_revision", MAX_OPAQUE_ID_ASCII_CHARS)
        if type(self.eligible_candidates) is not tuple:
            raise TypeError("eligible_candidates must be a tuple")
        if len(self.eligible_candidates) > MAX_ELIGIBLE_CANDIDATES:
            raise ValueError("too many eligible candidates")
        if any(type(item) is not ExecutionRouteCandidateV1 for item in self.eligible_candidates):
            raise TypeError("eligible_candidates contains a non-candidate")
        candidate_ids = tuple(item.candidate_id for item in self.eligible_candidates)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate ids must be unique")
        if self.native_candidate_id is not None and self.native_candidate_id not in candidate_ids:
            raise ValueError("native candidate is not eligible")
        if self.previous_attempt is not None and type(self.previous_attempt) is not ExecutionRoutePreviousAttemptV1:
            raise TypeError("previous_attempt must be exact ExecutionRoutePreviousAttemptV1")
        if self.request_digest is not None:
            _require_ascii(self.request_digest, "request_digest", 64)
            if self.request_digest != compute_request_digest(self):
                raise ValueError("request digest mismatch")
        if len(canonical_json_bytes(self)) > MAX_SERIALIZED_REQUEST_BYTES:
            raise ValueError("serialized request exceeds contract limit")

    def with_computed_digest(self) -> "ExecutionRouteRequestV1":
        return replace(self, request_digest=compute_request_digest(self))


@dataclass(frozen=True)
class ExecutionRouteDecisionV1:
    contract_version: str
    kind: ExecutionRouteDecisionKind
    request_id: str
    attempt_id: str
    candidate_id: str | None = None
    reason_code: str | None = None
    reason_text: str | None = None

    def __post_init__(self) -> None:
        if self.contract_version != CONTRACT_VERSION:
            raise ValueError("unsupported execution-router contract version")
        _enum(self.kind, ExecutionRouteDecisionKind, "decision.kind")
        if self.kind is ExecutionRouteDecisionKind.ROUTER_ERROR:
            raise ValueError("router_error is host-owned and not a provider decision")
        _require_ascii(self.request_id, "decision.request_id", MAX_OPAQUE_ID_ASCII_CHARS)
        _require_ascii(self.attempt_id, "decision.attempt_id", MAX_OPAQUE_ID_ASCII_CHARS)
        _require_ascii(self.candidate_id, "decision.candidate_id", MAX_OPAQUE_ID_ASCII_CHARS, nullable=True)
        if (self.kind is ExecutionRouteDecisionKind.ROUTE) != (self.candidate_id is not None):
            raise ValueError("candidate_id is required only for route decisions")
        _require_ascii(self.reason_code, "decision.reason_code", MAX_REASON_CODE_ASCII_CHARS, nullable=True)
        if self.reason_text is not None:
            _require_exact_str(self.reason_text, "decision.reason_text", max_bytes=MAX_REASON_TEXT_UTF8_BYTES)

    @classmethod
    def route(cls, *, request_id: str, attempt_id: str, candidate_id: str, reason_code: str | None = None, reason_text: str | None = None) -> "ExecutionRouteDecisionV1":
        return cls(CONTRACT_VERSION, ExecutionRouteDecisionKind.ROUTE, request_id, attempt_id, candidate_id, reason_code, reason_text)

    @classmethod
    def pass_through(cls, *, request_id: str, attempt_id: str, reason_code: str | None = None, reason_text: str | None = None) -> "ExecutionRouteDecisionV1":
        return cls(CONTRACT_VERSION, ExecutionRouteDecisionKind.PASS_THROUGH, request_id, attempt_id, None, reason_code, reason_text)

    @classmethod
    def stop(cls, *, request_id: str, attempt_id: str, reason_code: str | None = None, reason_text: str | None = None) -> "ExecutionRouteDecisionV1":
        return cls(CONTRACT_VERSION, ExecutionRouteDecisionKind.STOP, request_id, attempt_id, None, reason_code, reason_text)


@dataclass(frozen=True)
class ExecutionRouterProviderDescriptorV1:
    plugin_id: str
    plugin_version: str
    provider_id: str
    contract_version: str
    supported_execution_kinds: tuple[ExecutionKind, ...]

    def __post_init__(self) -> None:
        for name in ("plugin_id", "plugin_version", "provider_id"):
            _require_ascii(getattr(self, name), name, MAX_OPAQUE_ID_ASCII_CHARS)
        if self.contract_version != CONTRACT_VERSION:
            raise ValueError("unsupported execution-router contract version")
        if type(self.supported_execution_kinds) is not tuple or any(
            type(kind) is not ExecutionKind for kind in self.supported_execution_kinds
        ):
            raise TypeError("supported_execution_kinds must be a tuple of ExecutionKind")
        if len(set(self.supported_execution_kinds)) != len(self.supported_execution_kinds):
            raise ValueError("supported execution kinds must be unique")


@runtime_checkable
class ExecutionRouterProviderV1(Protocol):
    descriptor: ExecutionRouterProviderDescriptorV1

    def resolve_execution_route(
        self,
        request: ExecutionRouteRequestV1,
        cancellation: "ExecutionRouterCancellationSignalV1",
    ) -> ExecutionRouteDecisionV1 | None: ...


_CANCELLATION_LOCK = threading.Lock()
_CANCELLATION_STATES: dict[str, tuple[bool, float]] = {}
_CANCELLATION_STATE_TTL_SECONDS = 60.0


def _read_cancellation_state(token: str) -> bool:
    with _CANCELLATION_LOCK:
        state = _CANCELLATION_STATES.get(token)
    return True if state is None else state[0]


class _ExecutionRouterCancellationController:
    __slots__ = ("_token",)

    def __init__(self, token: str) -> None:
        self._token = token

    def cancel(self) -> None:
        with _CANCELLATION_LOCK:
            state = _CANCELLATION_STATES.get(self._token)
            if state is not None:
                _CANCELLATION_STATES[self._token] = (True, state[1])

    def close(self) -> None:
        with _CANCELLATION_LOCK:
            _CANCELLATION_STATES.pop(self._token, None)


def _new_execution_router_cancellation(
) -> tuple["ExecutionRouterCancellationSignalV1", _ExecutionRouterCancellationController]:
    now = time.monotonic()
    token = secrets.token_hex(32)
    with _CANCELLATION_LOCK:
        expired = [
            key
            for key, (_cancelled, created) in _CANCELLATION_STATES.items()
            if now - created > _CANCELLATION_STATE_TTL_SECONDS
        ]
        for key in expired:
            del _CANCELLATION_STATES[key]
        _CANCELLATION_STATES[token] = (False, now)
    return ExecutionRouterCancellationSignalV1(token), _ExecutionRouterCancellationController(token)


class ExecutionRouterCancellationSignalV1:
    """Immutable provider-facing read side of host-owned cancellation."""

    __slots__ = ("__token",)

    def __init__(self, token: str) -> None:
        object.__setattr__(self, "_ExecutionRouterCancellationSignalV1__token", token)

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise AttributeError("execution-router cancellation signal is immutable")

    def is_cancelled(self) -> bool:
        return _read_cancellation_state(self.__token)


def validate_provider(provider: Any) -> ExecutionRouterProviderV1:
    """Validate the exact unreleased v1 synchronous callback shape."""
    if not isinstance(provider, ExecutionRouterProviderV1):
        raise TypeError("execution router must implement ExecutionRouterProviderV1")
    callback = getattr(provider, "resolve_execution_route", None)
    if callback is None or not callable(callback):
        raise TypeError("execution router callback is missing")
    if inspect.iscoroutinefunction(callback):
        raise TypeError("execution router callback must be synchronous")
    try:
        parameters = tuple(inspect.signature(callback).parameters.values())
    except (TypeError, ValueError) as exc:
        raise TypeError("execution router callback signature is not inspectable") from exc
    if len(parameters) != 2 or any(
        item.kind not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for item in parameters
    ):
        raise TypeError("execution router callback must accept exactly request and cancellation")
    return provider


@dataclass(frozen=True)
class ExecutionRouterHostCapabilitiesV1:
    supported_contract_versions: tuple[str, ...]
    supported_execution_kinds: tuple[ExecutionKind, ...]
    operational_execution_kinds: tuple[ExecutionKind, ...]
    router_deadline_ms: int
    max_instruction_utf8_bytes: int
    max_serialized_request_bytes: int
    max_eligible_candidates: int
    max_event_page_records: int
    route_event_contract_available: bool
    route_event_read_available: bool
    host_release: str
    host_build: str


@dataclass(frozen=True)
class ExecutionRouteEventV1:
    contract_version: str
    event_id: str
    root_id: str
    task_id: str | None
    execution_id: str
    attempt_id: str
    request_id: str
    previous_attempt_id: str | None
    sequence: int
    timestamp_utc_ms: int
    execution_kind: ExecutionKind
    surface_class: str
    router_plugin_id: str
    router_provider_id: str
    router_contract_version: str
    event_type: ExecutionRouteEventType
    decision_state: ExecutionRouteDecisionKind | None
    requested_candidate_id: str | None
    accepted_route: ExecutionRouteIdentityV1 | None
    actual_route: ExecutionRouteIdentityV1 | None
    reason_code: str | None
    reason_text: str | None
    terminal_state: str | None
    request_digest: str | None = None
    instruction_digest: str | None = None
    eligibility_revision: str | None = None

    def __post_init__(self) -> None:
        if self.contract_version != CONTRACT_VERSION or self.router_contract_version != CONTRACT_VERSION:
            raise ValueError("unsupported event contract version")
        for name in ("event_id", "root_id", "execution_id", "attempt_id", "request_id", "router_plugin_id", "router_provider_id"):
            _require_ascii(getattr(self, name), name, MAX_OPAQUE_ID_ASCII_CHARS)
        for name in ("task_id", "previous_attempt_id", "requested_candidate_id"):
            _require_ascii(getattr(self, name), name, MAX_OPAQUE_ID_ASCII_CHARS, nullable=True)
        _require_integer(self.sequence, "sequence")
        _require_integer(self.timestamp_utc_ms, "timestamp_utc_ms")
        _enum(self.execution_kind, ExecutionKind, "execution_kind")
        _require_ascii(self.surface_class, "surface_class", MAX_SURFACE_CLASS_ASCII_CHARS)
        _enum(self.event_type, ExecutionRouteEventType, "event_type")
        if self.decision_state is not None:
            _enum(self.decision_state, ExecutionRouteDecisionKind, "decision_state")
        for name in ("accepted_route", "actual_route"):
            value = getattr(self, name)
            if value is not None and type(value) is not ExecutionRouteIdentityV1:
                raise TypeError(f"{name} must be ExecutionRouteIdentityV1")
        _require_ascii(self.reason_code, "reason_code", MAX_REASON_CODE_ASCII_CHARS, nullable=True)
        if self.reason_text is not None:
            _require_exact_str(self.reason_text, "reason_text", max_bytes=MAX_REASON_TEXT_UTF8_BYTES)
        _require_ascii(self.terminal_state, "terminal_state", MAX_REASON_CODE_ASCII_CHARS, nullable=True)
        for name in ("request_digest", "instruction_digest"):
            value = getattr(self, name)
            if value is not None and re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        _require_ascii(self.eligibility_revision, "eligibility_revision", MAX_OPAQUE_ID_ASCII_CHARS, nullable=True)
        event = self.event_type
        if event is ExecutionRouteEventType.REQUESTED:
            if self.request_digest is None or self.eligibility_revision is None:
                raise ValueError("route_requested requires request and eligibility bindings")
            if any(value is not None for value in (
                self.decision_state, self.requested_candidate_id, self.accepted_route,
                self.actual_route, self.reason_code, self.reason_text, self.terminal_state,
            )):
                raise ValueError("route_requested contains result or terminal fields")
        elif any(value is not None for value in (
            self.request_digest, self.instruction_digest, self.eligibility_revision,
        )):
            raise ValueError(f"{event.value} cannot contain requested-event bindings")
        if event is ExecutionRouteEventType.ACCEPTED:
            if (
                self.decision_state is not ExecutionRouteDecisionKind.ROUTE
                or self.requested_candidate_id is None
                or self.accepted_route is None
                or self.actual_route is not None
                or self.terminal_state is not None
            ):
                raise ValueError("route_accepted requires a requested and accepted route only")
        elif event in {ExecutionRouteEventType.STARTED, ExecutionRouteEventType.FINISHED}:
            if self.decision_state not in {
                ExecutionRouteDecisionKind.ROUTE,
                ExecutionRouteDecisionKind.PASS_THROUGH,
            } or self.actual_route is None:
                raise ValueError(f"{event.value} requires a started route or pass-through identity")
            if self.decision_state is ExecutionRouteDecisionKind.ROUTE:
                if self.requested_candidate_id is None or self.accepted_route is None:
                    raise ValueError(f"{event.value} route is missing its accepted identity")
            elif self.requested_candidate_id is not None or self.accepted_route is not None:
                raise ValueError(f"{event.value} pass-through cannot contain an accepted route")
            if (event is ExecutionRouteEventType.FINISHED) != (self.terminal_state is not None):
                raise ValueError(f"{event.value} terminal state shape is invalid")
        elif event is ExecutionRouteEventType.NOT_STARTED:
            if self.decision_state is None or self.actual_route is not None or self.terminal_state is not None:
                raise ValueError("route_not_started requires a decision without actual or terminal route")
            if self.decision_state is ExecutionRouteDecisionKind.ROUTE:
                if self.requested_candidate_id is None or self.accepted_route is None:
                    raise ValueError("route_not_started route is missing its accepted identity")
            elif self.requested_candidate_id is not None or self.accepted_route is not None:
                raise ValueError("route_not_started non-route decision cannot contain an accepted route")


def _wire(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value):
        return {field.name: _wire(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if type(value) is tuple:
        return [_wire(item) for item in value]
    if value is None or type(value) in (str, int, bool):
        return value
    raise TypeError(f"unsupported canonical value type: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _wire(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def compute_request_digest(request: ExecutionRouteRequestV1) -> str:
    if type(request) is not ExecutionRouteRequestV1:
        raise TypeError("request must be exact ExecutionRouteRequestV1")
    return hashlib.sha256(canonical_json_bytes(replace(request, request_digest=None))).hexdigest()


def validate_request(request: ExecutionRouteRequestV1) -> ExecutionRouteRequestV1:
    if type(request) is not ExecutionRouteRequestV1:
        raise TypeError("request must be exact ExecutionRouteRequestV1")
    if request.request_digest is None or request.request_digest != compute_request_digest(request):
        raise ValueError("request digest mismatch")
    return request


def validate_decision(
    request: ExecutionRouteRequestV1,
    decision: ExecutionRouteDecisionV1,
) -> ExecutionRouteDecisionV1:
    validate_request(request)
    if type(decision) is not ExecutionRouteDecisionV1:
        raise TypeError("decision must be exact ExecutionRouteDecisionV1")
    if decision.request_id != request.request_id or decision.attempt_id != request.attempt_id:
        raise ValueError("decision correlation mismatch")
    if decision.kind is ExecutionRouteDecisionKind.ROUTE:
        candidates = {candidate.candidate_id: candidate for candidate in request.eligible_candidates}
        candidate = candidates.get(decision.candidate_id)
        if candidate is None:
            raise ValueError("decision candidate is not eligible")
        for field_name in ("model", "provider", "reasoning"):
            pin = getattr(request.pins, field_name)
            if pin is not None and getattr(candidate, field_name) != pin:
                raise ValueError(f"decision conflicts with explicit {field_name} pin")
    return decision


def discover_execution_router_capabilities() -> ExecutionRouterHostCapabilitiesV1:
    from hermes_cli.build_info import get_code_identity

    identity = get_code_identity()
    return ExecutionRouterHostCapabilitiesV1(
        supported_contract_versions=(CONTRACT_VERSION,),
        supported_execution_kinds=tuple(ExecutionKind),
        operational_execution_kinds=(),
        router_deadline_ms=ROUTER_DEADLINE_MS,
        max_instruction_utf8_bytes=MAX_INSTRUCTION_UTF8_BYTES,
        max_serialized_request_bytes=MAX_SERIALIZED_REQUEST_BYTES,
        max_eligible_candidates=MAX_ELIGIBLE_CANDIDATES,
        max_event_page_records=MAX_EVENT_PAGE_RECORDS,
        route_event_contract_available=True,
        route_event_read_available=True,
        host_release=identity.get("version") or "unavailable",
        host_build=identity.get("sha") or "unavailable",
    )
