"""Independent lifecycle states, audited events, and transition validation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Mapping, TypeVar

from .validation import (
    DomainValidationError,
    _ascii_trimmed_nfc,
    _reject_sensitive,
)
class TaskState(str, Enum):
    NEW = "NEW"
    PLANNED = "PLANNED"
    ROUTED = "ROUTED"
    DISPATCHED = "DISPATCHED"
    VERIFYING = "VERIFYING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    WAITING_FOR_CAPACITY = "WAITING_FOR_CAPACITY"
    CANCELLED = "CANCELLED"

class AttemptState(str, Enum):
    CREATED = "CREATED"
    RESERVED = "RESERVED"
    DISPATCHED = "DISPATCHED"
    RUNNING = "RUNNING"
    RESULT_RECORDED = "RESULT_RECORDED"
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

class RouteState(str, Enum):
    DISCOVERED = "DISCOVERED"
    ELIGIBLE = "ELIGIBLE"
    INELIGIBLE = "INELIGIBLE"
    COOLDOWN = "COOLDOWN"
    BREAKER_OPEN = "BREAKER_OPEN"
    RETIRED = "RETIRED"

class CredentialState(str, Enum):
    AVAILABLE = "AVAILABLE"
    EXHAUSTED = "EXHAUSTED"
    COOLDOWN = "COOLDOWN"
    DEAD = "DEAD"

class ReservationState(str, Enum):
    PENDING = "PENDING"
    HELD = "HELD"
    CONSUMED = "CONSUMED"
    RELEASED = "RELEASED"
    EXPIRED = "EXPIRED"
    RECONCILED = "RECONCILED"

class ReviewState(str, Enum):
    NOT_REQUIRED = "NOT_REQUIRED"
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    PASSED = "PASSED"
    FAILED = "FAILED"
    HUMAN_APPROVED = "HUMAN_APPROVED"
    HUMAN_REJECTED = "HUMAN_REJECTED"

E = TypeVar("E")

S = TypeVar("S", bound=Enum)

def _validate_audit_event(
    event: object,
    state_type: type[Enum],
    from_state: object,
    to_state: object,
    identity_field: str,
) -> None:
    if type(from_state) is not state_type or type(to_state) is not state_type:
        raise DomainValidationError(
            "state.domain_mismatch",
            f"transition states must both be {state_type.__name__}",
        )
    for name in (identity_field, "actor", "timestamp", "reason", "correlation_id"):
        object.__setattr__(
            event,
            name,
            _ascii_trimmed_nfc(
                getattr(event, name),
                field_name=name,
                code="state.audit_metadata_required",
            ),
        )
    _reject_sensitive(event, "state_event")

def _ensure_legal_transition(
    from_state: S,
    to_state: S,
    allowed: Mapping[S, frozenset[S]],
) -> None:
    if to_state not in allowed.get(from_state, frozenset()):
        raise DomainValidationError(
            "state.transition_illegal",
            f"illegal {type(from_state).__name__} transition {from_state.value}->{to_state.value}",
        )

@dataclass(frozen=True)
class TaskStateEvent:
    task_id: str
    from_state: TaskState
    to_state: TaskState
    actor: str
    timestamp: str
    reason: str
    correlation_id: str

    def __post_init__(self) -> None:
        _validate_audit_event(self, TaskState, self.from_state, self.to_state, "task_id")
        _ensure_legal_transition(self.from_state, self.to_state, _TASK_TRANSITIONS)

@dataclass(frozen=True)
class AttemptStateEvent:
    attempt_id: str
    from_state: AttemptState
    to_state: AttemptState
    actor: str
    timestamp: str
    reason: str
    correlation_id: str

    def __post_init__(self) -> None:
        _validate_audit_event(self, AttemptState, self.from_state, self.to_state, "attempt_id")
        _ensure_legal_transition(self.from_state, self.to_state, _ATTEMPT_TRANSITIONS)

@dataclass(frozen=True)
class RouteStateEvent:
    route_id: str
    from_state: RouteState
    to_state: RouteState
    actor: str
    timestamp: str
    reason: str
    correlation_id: str

    def __post_init__(self) -> None:
        _validate_audit_event(self, RouteState, self.from_state, self.to_state, "route_id")
        _ensure_legal_transition(self.from_state, self.to_state, _ROUTE_TRANSITIONS)

@dataclass(frozen=True)
class CredentialStateEvent:
    credential_id: str
    from_state: CredentialState
    to_state: CredentialState
    actor: str
    timestamp: str
    reason: str
    correlation_id: str

    def __post_init__(self) -> None:
        _validate_audit_event(
            self, CredentialState, self.from_state, self.to_state, "credential_id"
        )
        _ensure_legal_transition(self.from_state, self.to_state, _CREDENTIAL_TRANSITIONS)

@dataclass(frozen=True)
class ReservationStateEvent:
    reservation_id: str
    from_state: ReservationState
    to_state: ReservationState
    actor: str
    timestamp: str
    reason: str
    correlation_id: str

    def __post_init__(self) -> None:
        _validate_audit_event(
            self, ReservationState, self.from_state, self.to_state, "reservation_id"
        )
        _ensure_legal_transition(self.from_state, self.to_state, _RESERVATION_TRANSITIONS)

@dataclass(frozen=True)
class ReviewStateEvent:
    review_id: str
    from_state: ReviewState
    to_state: ReviewState
    actor: str
    timestamp: str
    reason: str
    correlation_id: str

    def __post_init__(self) -> None:
        _validate_audit_event(self, ReviewState, self.from_state, self.to_state, "review_id")
        _ensure_legal_transition(self.from_state, self.to_state, _REVIEW_TRANSITIONS)

_TASK_TRANSITIONS = {
    TaskState.NEW: frozenset({TaskState.PLANNED, TaskState.CANCELLED}),
    TaskState.PLANNED: frozenset(
        {TaskState.ROUTED, TaskState.WAITING_FOR_CAPACITY, TaskState.FAILED, TaskState.CANCELLED}
    ),
    TaskState.ROUTED: frozenset(
        {TaskState.DISPATCHED, TaskState.WAITING_FOR_CAPACITY, TaskState.FAILED, TaskState.CANCELLED}
    ),
    TaskState.DISPATCHED: frozenset(
        {TaskState.VERIFYING, TaskState.FAILED, TaskState.CANCELLED}
    ),
    TaskState.VERIFYING: frozenset(
        {
            TaskState.COMPLETED,
            TaskState.FAILED,
            TaskState.WAITING_FOR_CAPACITY,
            TaskState.CANCELLED,
        }
    ),
    TaskState.WAITING_FOR_CAPACITY: frozenset(
        {TaskState.PLANNED, TaskState.ROUTED, TaskState.FAILED, TaskState.CANCELLED}
    ),
}

_ATTEMPT_TRANSITIONS = {
    AttemptState.CREATED: frozenset(
        {AttemptState.RESERVED, AttemptState.FAILED, AttemptState.CANCELLED}
    ),
    AttemptState.RESERVED: frozenset(
        {AttemptState.DISPATCHED, AttemptState.FAILED, AttemptState.CANCELLED}
    ),
    AttemptState.DISPATCHED: frozenset(
        {AttemptState.RUNNING, AttemptState.FAILED, AttemptState.CANCELLED}
    ),
    AttemptState.RUNNING: frozenset(
        {AttemptState.RESULT_RECORDED, AttemptState.FAILED, AttemptState.CANCELLED}
    ),
    AttemptState.RESULT_RECORDED: frozenset(
        {AttemptState.VERIFIED, AttemptState.FAILED, AttemptState.CANCELLED}
    ),
}

_ROUTE_TRANSITIONS = {
    RouteState.DISCOVERED: frozenset(
        {
            RouteState.ELIGIBLE,
            RouteState.INELIGIBLE,
            RouteState.COOLDOWN,
            RouteState.BREAKER_OPEN,
            RouteState.RETIRED,
        }
    ),
    RouteState.ELIGIBLE: frozenset(
        {RouteState.INELIGIBLE, RouteState.COOLDOWN, RouteState.BREAKER_OPEN, RouteState.RETIRED}
    ),
    RouteState.INELIGIBLE: frozenset({RouteState.ELIGIBLE, RouteState.RETIRED}),
    RouteState.COOLDOWN: frozenset(
        {RouteState.ELIGIBLE, RouteState.INELIGIBLE, RouteState.BREAKER_OPEN, RouteState.RETIRED}
    ),
    RouteState.BREAKER_OPEN: frozenset(
        {RouteState.COOLDOWN, RouteState.ELIGIBLE, RouteState.RETIRED}
    ),
}

_CREDENTIAL_TRANSITIONS = {
    CredentialState.AVAILABLE: frozenset(
        {CredentialState.EXHAUSTED, CredentialState.COOLDOWN, CredentialState.DEAD}
    ),
    CredentialState.EXHAUSTED: frozenset(
        {CredentialState.AVAILABLE, CredentialState.COOLDOWN, CredentialState.DEAD}
    ),
    CredentialState.COOLDOWN: frozenset(
        {CredentialState.AVAILABLE, CredentialState.EXHAUSTED, CredentialState.DEAD}
    ),
}

_RESERVATION_TRANSITIONS = {
    ReservationState.PENDING: frozenset(
        {ReservationState.HELD, ReservationState.RELEASED, ReservationState.EXPIRED}
    ),
    ReservationState.HELD: frozenset(
        {
            ReservationState.CONSUMED,
            ReservationState.RELEASED,
            ReservationState.EXPIRED,
            ReservationState.RECONCILED,
        }
    ),
    ReservationState.CONSUMED: frozenset({ReservationState.RECONCILED}),
    ReservationState.RELEASED: frozenset({ReservationState.RECONCILED}),
    ReservationState.EXPIRED: frozenset({ReservationState.RECONCILED}),
}

_REVIEW_TRANSITIONS = {
    ReviewState.PENDING: frozenset(
        {ReviewState.IN_PROGRESS, ReviewState.HUMAN_APPROVED, ReviewState.HUMAN_REJECTED}
    ),
    ReviewState.IN_PROGRESS: frozenset(
        {
            ReviewState.PASSED,
            ReviewState.FAILED,
            ReviewState.HUMAN_APPROVED,
            ReviewState.HUMAN_REJECTED,
        }
    ),
}

def _validated_transition(
    event_type: Callable[..., E],
    allowed: Mapping[S, frozenset[S]],
    *,
    from_state: S,
    to_state: S,
    **audit_metadata: str,
) -> E:
    event = event_type(from_state=from_state, to_state=to_state, **audit_metadata)
    _ensure_legal_transition(from_state, to_state, allowed)
    return event

def validate_task_transition(
    *, from_state: TaskState, to_state: TaskState, **audit_metadata: str
) -> TaskStateEvent:
    return _validated_transition(
        TaskStateEvent,
        _TASK_TRANSITIONS,
        from_state=from_state,
        to_state=to_state,
        **audit_metadata,
    )

def validate_attempt_transition(
    *, from_state: AttemptState, to_state: AttemptState, **audit_metadata: str
) -> AttemptStateEvent:
    return _validated_transition(
        AttemptStateEvent,
        _ATTEMPT_TRANSITIONS,
        from_state=from_state,
        to_state=to_state,
        **audit_metadata,
    )

def validate_route_transition(
    *, from_state: RouteState, to_state: RouteState, **audit_metadata: str
) -> RouteStateEvent:
    return _validated_transition(
        RouteStateEvent,
        _ROUTE_TRANSITIONS,
        from_state=from_state,
        to_state=to_state,
        **audit_metadata,
    )

def validate_credential_transition(
    *, from_state: CredentialState, to_state: CredentialState, **audit_metadata: str
) -> CredentialStateEvent:
    return _validated_transition(
        CredentialStateEvent,
        _CREDENTIAL_TRANSITIONS,
        from_state=from_state,
        to_state=to_state,
        **audit_metadata,
    )

def validate_reservation_transition(
    *, from_state: ReservationState, to_state: ReservationState, **audit_metadata: str
) -> ReservationStateEvent:
    return _validated_transition(
        ReservationStateEvent,
        _RESERVATION_TRANSITIONS,
        from_state=from_state,
        to_state=to_state,
        **audit_metadata,
    )

def validate_review_transition(
    *, from_state: ReviewState, to_state: ReviewState, **audit_metadata: str
) -> ReviewStateEvent:
    return _validated_transition(
        ReviewStateEvent,
        _REVIEW_TRANSITIONS,
        from_state=from_state,
        to_state=to_state,
        **audit_metadata,
    )
