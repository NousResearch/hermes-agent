"""Aggregate input budget of a detached background review fork.

A review fork with an explicit ``_review_input_token_budget`` reserves every provider request
against it before the request goes out (``_reserve_review_input_request``), rolls the
reservation back when an attempt ends without a completed response
(``_release_review_input_request``), and exits at the top of the next iteration once its real
usage or its reservations reach the budget (``_review_input_budget_exhausted``, read by
``turn_iteration_prep``). Only forks with such a budget are gated (#93057); a fork revoked by
the foreground's cancel escalation is exhausted whatever its budget.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _review_input_budget_exhausted(agent: Any) -> bool:
    """True when a detached review fork has replayed its aggregate input budget.

    Only forks with an explicit ``_review_input_token_budget`` are gated (#93057). Fires
    at the top of the NEXT iteration, so the budget-crossing request completes first. A fork
    revoked by the foreground's cancel escalation is exhausted whatever its budget."""
    if getattr(agent, "_review_revoked", False):
        return True
    budget = getattr(agent, "_review_input_token_budget", None)
    if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
        return False
    # ``session_prompt_tokens`` is the provider's complete input, including cache reads/writes.
    # Uncached-only ``session_input_tokens`` would make the warm-cache replay path nearly free.
    used = getattr(
        agent,
        "session_prompt_tokens",
        getattr(agent, "session_input_tokens", 0),
    )
    reserved = getattr(agent, "_review_input_tokens_reserved", 0)
    totals = [
        value
        for value in (used, reserved)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
    ]
    return bool(totals) and max(totals) >= budget


def _review_input_tokens_consumed(agent: Any) -> int:
    """Input tokens a detached review has used or reserved so far (0 before its first request)."""
    used = getattr(agent, "session_prompt_tokens", 0)
    reserved = getattr(agent, "_review_input_tokens_reserved", 0)
    valid_totals = [
        value
        for value in (used, reserved)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
    ]
    return max(valid_totals, default=0)


def _reserve_review_input_request(agent: Any, projected_tokens: Any) -> bool:
    """Reserve one provider request against a detached automatic review's hard budget."""
    if getattr(agent, "_review_revoked", False):
        return False
    budget = getattr(agent, "_review_input_token_budget", None)
    if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
        return True
    if (
        not isinstance(projected_tokens, int)
        or isinstance(projected_tokens, bool)
        or projected_tokens < 0
    ):
        return False
    consumed = _review_input_tokens_consumed(agent)
    if projected_tokens > budget - consumed:
        return False
    agent._review_input_tokens_reserved = consumed + projected_tokens
    return True


def _release_review_input_request(agent: Any, reserved_before: int) -> None:
    """Roll an attempt's reservation back when it ended without a completed response, so the
    retry of the same request reserves its projection once, not on top of the stale one. Real
    usage from a completed attempt lives in ``session_prompt_tokens`` and stays counted."""
    if hasattr(agent, "_review_input_tokens_reserved"):
        agent._review_input_tokens_reserved = reserved_before


def _log_refused_review_first_request(agent: Any) -> None:
    """One owner-tagged, body-free skip line when a detached review's FIRST request is refused
    by its aggregate input budget: the fork then exits with zero provider calls and no writes,
    and no other line names its owner. A later crossing is the budget's ordinary exhaustion
    (the ``Turn ended`` line covers it); a revoked fork logs its own reason."""
    owner = getattr(agent, "_review_owner_tag", None)
    if (
        not owner
        or getattr(agent, "_review_revoked", False)
        or _review_input_tokens_consumed(agent)
    ):
        return
    from agent.review_admission import REASON_INPUT_BUDGET_REFUSED

    logger.info(
        "Background review skipped (owner=%s): %s", owner, REASON_INPUT_BUDGET_REFUSED
    )
