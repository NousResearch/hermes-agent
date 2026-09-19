"""Task-local Hermes Tag authority carried through gateway execution."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator

from .errors import LeaseError, LeaseTampered
from .model import CapabilityLease, DecisionOutcome, PolicyDecision, TurnAdmission


_CURRENT_ADMISSION: ContextVar[TurnAdmission | None] = ContextVar(
    "hermes_tag_admission", default=None
)
_CURRENT_DECISION: ContextVar[PolicyDecision | None] = ContextVar(
    "hermes_tag_decision", default=None
)
_CURRENT_LEASE: ContextVar[CapabilityLease | None] = ContextVar(
    "hermes_tag_lease", default=None
)


@dataclass(frozen=True, slots=True)
class RuntimeAuthority:
    admission: TurnAdmission
    decision: PolicyDecision | None = None
    lease: CapabilityLease | None = None

    def __post_init__(self) -> None:
        admission = self.admission
        decision = self.decision
        lease = self.lease
        scope_digest = admission.scope.digest

        if decision is not None and decision.scope_digest != scope_digest:
            raise LeaseTampered(
                "policy decision scope does not match the bound turn admission"
            )

        if lease is None:
            return
        if decision is None:
            raise LeaseTampered(
                "capability lease cannot be bound without its policy decision"
            )
        if decision.outcome is not DecisionOutcome.ALLOW:
            raise LeaseTampered(
                "capability lease cannot be paired with a non-allow decision"
            )
        if lease.principal_id != admission.principal.principal_id:
            raise LeaseTampered(
                "capability lease principal does not match the bound turn admission"
            )
        if lease.scope_digest != scope_digest:
            raise LeaseTampered(
                "capability lease scope does not match the bound turn admission"
            )
        if lease.continuity_id != admission.continuity_id:
            raise LeaseTampered(
                "capability lease continuity does not match the bound turn admission"
            )

        linked_fields = (
            ("decision_id", decision.decision_id, lease.decision_id),
            ("capability", decision.capability, lease.capability),
            ("intent_digest", decision.intent_digest, lease.intent_digest),
            ("scope_digest", decision.scope_digest, lease.scope_digest),
            ("obligations", decision.obligations, lease.obligations),
            (
                "budget_reservation_id",
                decision.budget_reservation_id,
                lease.budget_reservation_id,
            ),
            ("approval_id", decision.approval_id, lease.approval_id),
        )
        for name, expected, actual in linked_fields:
            if expected != actual:
                raise LeaseTampered(
                    f"capability lease {name} does not match its policy decision"
                )


@contextmanager
def bind_admission(admission: TurnAdmission) -> Iterator[TurnAdmission]:
    """Bind a new admission and clear all authority derived from an outer turn."""
    admission_token = _CURRENT_ADMISSION.set(admission)
    decision_token = _CURRENT_DECISION.set(None)
    lease_token = _CURRENT_LEASE.set(None)
    try:
        yield admission
    finally:
        _CURRENT_LEASE.reset(lease_token)
        _CURRENT_DECISION.reset(decision_token)
        _CURRENT_ADMISSION.reset(admission_token)


@contextmanager
def bind_decision(decision: PolicyDecision) -> Iterator[PolicyDecision]:
    """Bind a decision for the current admission and clear any outer lease."""
    admission = _CURRENT_ADMISSION.get()
    if admission is None:
        raise LeaseError(
            "cannot bind a Hermes Tag policy decision without a turn admission"
        )
    RuntimeAuthority(admission=admission, decision=decision)
    decision_token = _CURRENT_DECISION.set(decision)
    lease_token = _CURRENT_LEASE.set(None)
    try:
        yield decision
    finally:
        _CURRENT_LEASE.reset(lease_token)
        _CURRENT_DECISION.reset(decision_token)


@contextmanager
def bind_lease(lease: CapabilityLease) -> Iterator[CapabilityLease]:
    """Bind a lease only when its admission and policy decision are current."""
    admission = _CURRENT_ADMISSION.get()
    decision = _CURRENT_DECISION.get()
    if admission is None or decision is None:
        raise LeaseError(
            "cannot bind a Hermes Tag capability lease without admission and decision"
        )
    RuntimeAuthority(admission=admission, decision=decision, lease=lease)
    token = _CURRENT_LEASE.set(lease)
    try:
        yield lease
    finally:
        _CURRENT_LEASE.reset(token)


@contextmanager
def bind_authority(authority: RuntimeAuthority) -> Iterator[RuntimeAuthority]:
    """Atomically replace the complete task-local authority tuple."""
    admission_token = _CURRENT_ADMISSION.set(authority.admission)
    decision_token = _CURRENT_DECISION.set(authority.decision)
    lease_token = _CURRENT_LEASE.set(authority.lease)
    try:
        yield authority
    finally:
        _CURRENT_LEASE.reset(lease_token)
        _CURRENT_DECISION.reset(decision_token)
        _CURRENT_ADMISSION.reset(admission_token)


def current_admission(*, required: bool = False) -> TurnAdmission | None:
    value = _CURRENT_ADMISSION.get()
    if required and value is None:
        raise LeaseError("no Hermes Tag turn admission is bound")
    return value


def current_decision(*, required: bool = False) -> PolicyDecision | None:
    value = _CURRENT_DECISION.get()
    if required and value is None:
        raise LeaseError("no Hermes Tag policy decision is bound")
    return value


def current_lease(*, required: bool = False) -> CapabilityLease | None:
    value = _CURRENT_LEASE.get()
    if required and value is None:
        raise LeaseError("no Hermes Tag capability lease is bound")
    return value


def capture_authority() -> RuntimeAuthority:
    admission = current_admission(required=True)
    assert admission is not None
    return RuntimeAuthority(
        admission=admission,
        decision=current_decision(),
        lease=current_lease(),
    )


def clear_runtime_context() -> None:
    """Explicitly clear the current context for process lifecycle tests."""
    _CURRENT_ADMISSION.set(None)
    _CURRENT_DECISION.set(None)
    _CURRENT_LEASE.set(None)
