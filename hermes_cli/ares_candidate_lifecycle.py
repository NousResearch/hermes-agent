"""Typed lifecycle policy for immutable Ares candidates.

The state machine is intentionally independent of Context Governor.  It owns
custody progress, never artifact identity or activation policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class CandidateLifecycleState(StrEnum):
    BUILDING = "BUILDING"
    CORE_REPRODUCED = "CORE_REPRODUCED"
    CERTIFYING = "CERTIFYING"
    CERTIFIED = "CERTIFIED"
    SEALING = "SEALING"
    SEALED = "SEALED"
    AWAITING_HOSTILE_AUDIT = "AWAITING_HOSTILE_AUDIT"
    HOSTILE_AUDIT_IN_PROGRESS = "HOSTILE_AUDIT_IN_PROGRESS"
    AUDIT_BLOCKED = "AUDIT_BLOCKED"
    AUDIT_PASSED = "AUDIT_PASSED"
    AUDIT_FAILED = "AUDIT_FAILED"
    AWAITING_ACTIVATION = "AWAITING_ACTIVATION"
    ACTIVE = "ACTIVE"
    ROLLBACK_REQUIRED = "ROLLBACK_REQUIRED"
    ROLLED_BACK = "ROLLED_BACK"
    REJECTED = "REJECTED"
    SUPERSEDED = "SUPERSEDED"
    INCIDENT_HELD = "INCIDENT_HELD"
    GC_ELIGIBLE = "GC_ELIGIBLE"


@dataclass(frozen=True)
class LifecyclePolicy:
    incoming: frozenset[CandidateLifecycleState]
    required_artifacts: frozenset[str]
    audit_eligible: bool = False
    activation_eligible: bool = False
    gc_eligible: bool = False


_all = frozenset(CandidateLifecycleState)
_terminal = frozenset({
    CandidateLifecycleState.REJECTED,
    CandidateLifecycleState.SUPERSEDED,
    CandidateLifecycleState.ROLLED_BACK,
})

LIFECYCLE_POLICY: dict[CandidateLifecycleState, LifecyclePolicy] = {
    CandidateLifecycleState.BUILDING: LifecyclePolicy(frozenset(), frozenset()),
    CandidateLifecycleState.CORE_REPRODUCED: LifecyclePolicy(
        frozenset({CandidateLifecycleState.BUILDING}),
        frozenset({"candidate-core-manifest"}),
    ),
    CandidateLifecycleState.CERTIFYING: LifecyclePolicy(
        frozenset({CandidateLifecycleState.CORE_REPRODUCED}),
        frozenset({"candidate-core-manifest"}),
    ),
    CandidateLifecycleState.CERTIFIED: LifecyclePolicy(
        frozenset({CandidateLifecycleState.CERTIFYING}),
        frozenset({"candidate-core-manifest", "certification-set-manifest"}),
    ),
    CandidateLifecycleState.SEALING: LifecyclePolicy(
        frozenset({CandidateLifecycleState.CERTIFIED}),
        frozenset({
            "candidate-core-manifest",
            "certification-set-manifest",
            "sealed-candidate-manifest",
        }),
    ),
    CandidateLifecycleState.SEALED: LifecyclePolicy(
        frozenset({CandidateLifecycleState.SEALING}),
        frozenset({
            "archive",
            "candidate-core-manifest",
            "certification-set-manifest",
            "sealed-candidate-manifest",
        }),
    ),
    CandidateLifecycleState.AWAITING_HOSTILE_AUDIT: LifecyclePolicy(
        frozenset({
            CandidateLifecycleState.SEALED,
            CandidateLifecycleState.AUDIT_BLOCKED,
        }),
        frozenset({"archive", "hostile-audit-handoff"}),
        audit_eligible=True,
    ),
    CandidateLifecycleState.HOSTILE_AUDIT_IN_PROGRESS: LifecyclePolicy(
        frozenset({CandidateLifecycleState.AWAITING_HOSTILE_AUDIT}),
        frozenset({"archive", "hostile-audit-handoff"}),
        audit_eligible=True,
    ),
    CandidateLifecycleState.AUDIT_BLOCKED: LifecyclePolicy(
        frozenset({CandidateLifecycleState.HOSTILE_AUDIT_IN_PROGRESS}),
        frozenset({"archive", "hostile-audit-handoff"}),
        audit_eligible=True,
    ),
    CandidateLifecycleState.AUDIT_PASSED: LifecyclePolicy(
        frozenset({CandidateLifecycleState.HOSTILE_AUDIT_IN_PROGRESS}),
        frozenset({"archive", "hostile-audit-handoff"}),
        activation_eligible=True,
    ),
    CandidateLifecycleState.AUDIT_FAILED: LifecyclePolicy(
        frozenset({
            CandidateLifecycleState.HOSTILE_AUDIT_IN_PROGRESS,
            CandidateLifecycleState.AUDIT_BLOCKED,
        }),
        frozenset({"archive", "hostile-audit-handoff"}),
    ),
    CandidateLifecycleState.AWAITING_ACTIVATION: LifecyclePolicy(
        frozenset({CandidateLifecycleState.AUDIT_PASSED}),
        frozenset({"archive"}),
        activation_eligible=True,
    ),
    CandidateLifecycleState.ACTIVE: LifecyclePolicy(
        frozenset({CandidateLifecycleState.AWAITING_ACTIVATION}),
        frozenset({"archive"}),
        activation_eligible=True,
    ),
    CandidateLifecycleState.ROLLBACK_REQUIRED: LifecyclePolicy(
        frozenset({
            CandidateLifecycleState.ACTIVE,
            CandidateLifecycleState.AWAITING_ACTIVATION,
        }),
        frozenset({"archive"}),
    ),
    CandidateLifecycleState.ROLLED_BACK: LifecyclePolicy(
        frozenset({CandidateLifecycleState.ROLLBACK_REQUIRED}),
        frozenset({"archive"}),
        gc_eligible=True,
    ),
    CandidateLifecycleState.REJECTED: LifecyclePolicy(
        _all - frozenset({CandidateLifecycleState.ACTIVE}),
        frozenset(),
        gc_eligible=True,
    ),
    CandidateLifecycleState.SUPERSEDED: LifecyclePolicy(
        _terminal | frozenset({CandidateLifecycleState.AUDIT_FAILED}),
        frozenset(),
        gc_eligible=True,
    ),
    CandidateLifecycleState.INCIDENT_HELD: LifecyclePolicy(
        _all - frozenset({CandidateLifecycleState.GC_ELIGIBLE}), frozenset({"archive"})
    ),
    CandidateLifecycleState.GC_ELIGIBLE: LifecyclePolicy(
        _terminal, frozenset(), gc_eligible=True
    ),
}


def transition_allowed(
    old: CandidateLifecycleState, new: CandidateLifecycleState
) -> bool:
    """Return whether an explicit custody transition is permitted."""
    return old in LIFECYCLE_POLICY[new].incoming


def require_transition(
    old: CandidateLifecycleState, new: CandidateLifecycleState
) -> None:
    if not transition_allowed(old, new):
        raise ValueError(f"ILLEGAL_LIFECYCLE_TRANSITION: {old}->{new}")


def is_gc_protected(state: CandidateLifecycleState) -> bool:
    return not LIFECYCLE_POLICY[state].gc_eligible
