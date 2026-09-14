"""Answer-time state gate adapter.

This module decides whether a model call may proceed. It has no gateway,
persistence, or finalization side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .state_answer_dependency import answer_depends_on_candidate
from .state_candidate_evaluator import StateCandidateResult
from .state_response_policy import ResponseDecision, ResponsePolicy, route_candidate_result


@dataclass(frozen=True)
class StateAnswerGateResult:
    decision: ResponseDecision
    depends_on_candidate: bool
    model_call_allowed: bool


def evaluate_answer_gate(
    candidate: StateCandidateResult | None,
    *,
    requested_state_keys: Iterable[str],
    answer_scope: str,
) -> StateAnswerGateResult:
    """Compute the pre-model decision from explicit answer metadata."""
    depends = answer_depends_on_candidate(
        candidate,
        requested_state_keys=requested_state_keys,
        answer_scope=answer_scope,
    )
    decision = route_candidate_result(candidate, answer_depends_on_candidate=depends)
    return StateAnswerGateResult(
        decision=decision,
        depends_on_candidate=depends,
        model_call_allowed=decision.response_policy is ResponsePolicy.CONTINUE,
    )
