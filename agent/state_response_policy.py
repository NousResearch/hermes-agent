"""Pure routing from candidate evaluation to answer/persistence policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .state_candidate_evaluator import CandidateStatus, StateCandidateResult


class ResponsePolicy(str, Enum):
    CONTINUE = "continue"
    ASK_CONFIRMATION = "ask_confirmation"
    HOLD = "hold"
    CONFLICT_STOP = "conflict_stop"


class PersistencePolicy(str, Enum):
    NO_WRITE = "no_write"
    PENDING_ONLY = "pending_only"


@dataclass(frozen=True)
class ResponseDecision:
    response_policy: ResponsePolicy
    persistence_policy: PersistencePolicy
    reason: str


def route_candidate_result(
    result: Optional[StateCandidateResult],
    *,
    answer_depends_on_candidate: bool,
) -> ResponseDecision:
    """Route metadata without writing state or invoking a gateway/model.

    A candidate is not evidence of authorization. Dependency is explicit so an
    unrelated answer is not blocked merely because a pending candidate exists.
    """
    if result is None:
        return ResponseDecision(ResponsePolicy.CONTINUE, PersistencePolicy.NO_WRITE, "no candidate")

    if result.status in (CandidateStatus.CORROBORATION, CandidateStatus.DUPLICATE):
        return ResponseDecision(ResponsePolicy.CONTINUE, PersistencePolicy.NO_WRITE, result.status.value)

    if result.status == CandidateStatus.PENDING:
        policy = ResponsePolicy.ASK_CONFIRMATION if answer_depends_on_candidate else ResponsePolicy.CONTINUE
        return ResponseDecision(policy, PersistencePolicy.PENDING_ONLY, "candidate requires approval")

    if result.status == CandidateStatus.UNVERIFIED:
        policy = ResponsePolicy.ASK_CONFIRMATION if answer_depends_on_candidate else ResponsePolicy.CONTINUE
        return ResponseDecision(policy, PersistencePolicy.NO_WRITE, "candidate is unverified")

    if result.status == CandidateStatus.CONFLICT:
        policy = ResponsePolicy.CONFLICT_STOP if answer_depends_on_candidate else ResponsePolicy.CONTINUE
        return ResponseDecision(policy, PersistencePolicy.NO_WRITE, "candidate conflicts with existing state")

    # A rejected or malformed candidate cannot block unrelated work, but it
    # cannot authorize an answer that relies on the rejected state either.
    policy = ResponsePolicy.HOLD if answer_depends_on_candidate else ResponsePolicy.CONTINUE
    return ResponseDecision(policy, PersistencePolicy.NO_WRITE, "candidate was rejected")
