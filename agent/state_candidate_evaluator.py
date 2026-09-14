"""Pure evaluation of context-scoped state update candidates.

This module deliberately has no persistence, prompt, gateway, or model calls.
It turns a current state plus attributed evidence into a structured decision
that a later runtime seam may route or hold.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
from typing import Any, Iterable


class DeltaType(str, Enum):
    NONE = "none"
    REPLACE = "replace"
    SCOPE_CHANGE = "scope_change"
    CONFLICT = "conflict"


class CandidateStatus(str, Enum):
    CORROBORATION = "corroboration"
    PENDING = "pending"
    UNVERIFIED = "unverified"
    REJECTED = "rejected"
    CONFLICT = "conflict"
    DUPLICATE = "duplicate"


@dataclass(frozen=True)
class Evidence:
    value: str
    source_type: str
    source_ref: str
    scope: str


@dataclass(frozen=True)
class StateCandidateResult:
    candidate_id: str
    delta_type: DeltaType
    status: CandidateStatus
    state_key: str = ""
    old_value: str = ""
    new_value: str = ""
    scope: str = ""
    evidence_ref: str = ""
    reason: str = ""


def _candidate_id(state_key: str, scope: str, value: str, evidence_ref: str) -> str:
    raw = "|".join((state_key, scope, value, evidence_ref)).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _result(
    *,
    state_key: str = "",
    old_value: str = "",
    new_value: str = "",
    scope: str = "",
    evidence_ref: str = "",
    delta_type: DeltaType,
    status: CandidateStatus,
    reason: str = "",
) -> StateCandidateResult:
    return StateCandidateResult(
        candidate_id=_candidate_id(state_key, scope, new_value, evidence_ref),
        delta_type=delta_type,
        status=status,
        state_key=state_key,
        old_value=old_value,
        new_value=new_value,
        scope=scope,
        evidence_ref=evidence_ref,
        reason=reason,
    )


def evaluate_state_candidate(
    *,
    active_context: str,
    current_state: Any,
    evidence: Evidence,
    existing_candidates: Iterable[StateCandidateResult] = (),
) -> StateCandidateResult:
    """Evaluate one attributed evidence item against one scoped current state.

    The function is intentionally side-effect-free. A ``pending`` result is a
    candidate only; it does not authorize persistence or replacement of state.
    """
    if (
        not isinstance(evidence, Evidence)
        or not all(isinstance(value, str) for value in (
            evidence.value, evidence.source_type, evidence.source_ref, evidence.scope
        ))
        or not evidence.value.strip()
        or not evidence.source_type.strip()
        or not evidence.source_ref.strip()
        or not evidence.scope.strip()
        or not isinstance(active_context, str)
        or not active_context.strip()
    ):
        return _result(
            delta_type=DeltaType.CONFLICT,
            status=CandidateStatus.REJECTED,
            reason="malformed evidence",
        )
    try:
        prior = list(existing_candidates)
    except (TypeError, ValueError):
        return _result(
            delta_type=DeltaType.CONFLICT,
            status=CandidateStatus.REJECTED,
            reason="malformed existing candidates",
        )
    if any(not isinstance(item, StateCandidateResult) for item in prior):
        return _result(
            delta_type=DeltaType.CONFLICT,
            status=CandidateStatus.REJECTED,
            reason="malformed existing candidate",
        )

    required = ("key", "value", "scope", "source_ref")
    if (
        not isinstance(current_state, dict)
        or any(key not in current_state for key in required)
        or not all(isinstance(current_state[key], str) for key in required)
        or not all(current_state[key].strip() for key in required)
    ):
        return _result(
            delta_type=DeltaType.CONFLICT,
            status=CandidateStatus.REJECTED,
            reason="malformed current state",
        )
    if not evidence.source_ref:
        return _result(
            state_key=str(current_state["key"]),
            scope=evidence.scope,
            evidence_ref=evidence.source_ref,
            delta_type=DeltaType.CONFLICT,
            status=CandidateStatus.REJECTED,
            reason="missing source reference",
        )

    state_key = str(current_state["key"])
    current_scope = str(current_state["scope"])
    old_value = str(current_state["value"])
    new_value = str(evidence.value)
    candidate_id = _candidate_id(state_key, evidence.scope, new_value, evidence.source_ref)

    if current_scope != active_context or current_scope != evidence.scope:
        return _result(
            state_key=state_key,
            old_value=old_value,
            new_value=new_value,
            scope=evidence.scope,
            evidence_ref=evidence.source_ref,
            delta_type=DeltaType.SCOPE_CHANGE,
            status=CandidateStatus.UNVERIFIED,
            reason="current state, active context, and evidence scope disagree",
        )

    if any(item.candidate_id == candidate_id for item in prior):
        return _result(
            state_key=state_key,
            old_value=old_value,
            new_value=new_value,
            scope=evidence.scope,
            evidence_ref=evidence.source_ref,
            delta_type=DeltaType.REPLACE,
            status=CandidateStatus.DUPLICATE,
            reason="same candidate already exists",
        )

    if active_context != evidence.scope:
        return _result(
            state_key=state_key,
            old_value=old_value,
            new_value=new_value,
            scope=evidence.scope,
            evidence_ref=evidence.source_ref,
            delta_type=DeltaType.SCOPE_CHANGE,
            status=CandidateStatus.UNVERIFIED,
            reason="context and evidence scope disagree",
        )

    if evidence.source_type == "assistant_inference":
        return _result(
            state_key=state_key,
            old_value=old_value,
            new_value=new_value,
            scope=evidence.scope,
            evidence_ref=evidence.source_ref,
            delta_type=DeltaType.REPLACE,
            status=CandidateStatus.UNVERIFIED,
            reason="assistant inference cannot establish a state change",
        )

    if new_value == old_value:
        return _result(
            state_key=state_key,
            old_value=old_value,
            new_value=new_value,
            scope=evidence.scope,
            evidence_ref=evidence.source_ref,
            delta_type=DeltaType.NONE,
            status=CandidateStatus.CORROBORATION,
            reason="evidence matches current state",
        )

    if any(
        item.state_key == state_key
        and item.scope == evidence.scope
        and item.new_value != new_value
        and item.status in (CandidateStatus.PENDING, CandidateStatus.CONFLICT)
        for item in prior
    ):
        return _result(
            state_key=state_key,
            old_value=old_value,
            new_value=new_value,
            scope=evidence.scope,
            evidence_ref=evidence.source_ref,
            delta_type=DeltaType.CONFLICT,
            status=CandidateStatus.CONFLICT,
            reason="incompatible pending candidate exists",
        )

    return _result(
        state_key=state_key,
        old_value=old_value,
        new_value=new_value,
        scope=evidence.scope,
        evidence_ref=evidence.source_ref,
        delta_type=DeltaType.REPLACE,
        status=CandidateStatus.PENDING,
        reason="current state and evidence differ",
    )
