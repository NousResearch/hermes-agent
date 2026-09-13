"""Answer-time state gate adapter.

This module decides whether a model call may proceed. It has no gateway,
persistence, or finalization side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .state_answer_dependency import answer_depends_on_candidate
from .state_candidate_evaluator import StateCandidateResult
from .state_response_policy import (
    PersistencePolicy, ResponseDecision, ResponsePolicy, route_candidate_result,
)


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
    keys = ()
    malformed = (
        not isinstance(answer_scope, str) or not answer_scope.strip()
        or isinstance(requested_state_keys, (str, bytes))
    )
    if not malformed:
        try:
            keys = list(requested_state_keys)
            malformed = not all(isinstance(key, str) and key.strip() for key in keys)
        except (TypeError, ValueError):
            malformed = True
    if candidate is not None and (
        not isinstance(candidate.state_key, str) or not candidate.state_key.strip()
        or not isinstance(candidate.scope, str) or not candidate.scope.strip()
    ):
        malformed = True
    if malformed:
        return StateAnswerGateResult(
            decision=ResponseDecision(ResponsePolicy.HOLD, PersistencePolicy.NO_WRITE, "malformed answer metadata"),
            depends_on_candidate=True,
            model_call_allowed=False,
        )
    depends = answer_depends_on_candidate(
        candidate,
        requested_state_keys=keys,
        answer_scope=answer_scope,
    )
    decision = route_candidate_result(candidate, answer_depends_on_candidate=depends)
    return StateAnswerGateResult(
        decision=decision,
        depends_on_candidate=depends,
        model_call_allowed=decision.response_policy is ResponsePolicy.CONTINUE,
    )
