"""Admission control for automatic review requests without rewriting their cached prefix."""
from typing import Any


def automatic_review_budget_exceeded(agent: Any, request_tokens: int) -> bool:
    """Compare the assembled request estimate plus spent input against the review budget.

    Cache reads/writes are billed input too. Explicit /refine and other forks keep
    their existing post-call aggregate gate; only unattended review is admitted here.
    """
    if getattr(agent, "_turn_origin", None) != "background_review" or getattr(agent, "_review_attended", False):
        return False
    budget = getattr(agent, "_review_input_token_budget", None)
    if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
        return False
    used = sum(
        value for key in ("input_tokens", "cache_read_tokens", "cache_write_tokens")
        if isinstance(value := getattr(agent, f"session_{key}", 0), int)
        and not isinstance(value, bool) and value > 0
    )
    return used + request_tokens > budget
