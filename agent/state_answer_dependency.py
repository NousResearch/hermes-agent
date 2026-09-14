"""Pure dependency checks between an answer request and a state candidate."""

from __future__ import annotations

from collections.abc import Iterable

from .state_candidate_evaluator import StateCandidateResult


def answer_depends_on_candidate(
    result: StateCandidateResult | None,
    *,
    requested_state_keys: Iterable[str],
    answer_scope: str,
) -> bool:
    """Return true only when the answer explicitly targets the candidate state.

    This deliberately does not infer semantic dependency from free-form text.
    Callers must provide the state keys they intend to use, and the scope must
    match exactly. A candidate's mere presence therefore cannot stop unrelated
    answers.
    """
    if result is None or not result.state_key or not result.scope:
        return False
    if result.scope != answer_scope:
        return False
    return result.state_key in set(requested_state_keys)
