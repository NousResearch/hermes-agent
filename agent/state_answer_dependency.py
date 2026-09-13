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
    if not isinstance(answer_scope, str) or not answer_scope.strip():
        return True
    if isinstance(requested_state_keys, (str, bytes)):
        return True
    try:
        requested = list(requested_state_keys)
    except (TypeError, ValueError):
        return True
    if not all(isinstance(key, str) and key.strip() for key in requested):
        return True
    if result is None:
        return False
    if (
        not isinstance(result.state_key, str) or not result.state_key.strip()
        or not isinstance(result.scope, str) or not result.scope.strip()
    ):
        return False
    if result.scope != answer_scope:
        return False
    return result.state_key in set(requested)
