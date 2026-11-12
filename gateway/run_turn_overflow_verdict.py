"""The context-overflow verdict for a finished agent turn (#1630, #7100, #134261).

Split out of ``gateway.run_turn`` under the per-file line cap (root AGENTS.md). The symbol is
re-exported there, so existing call sites and tests keep importing it from ``gateway.run_turn``.
"""

_CONTEXT_OVERFLOW_ERROR_PHRASES = (
    "context length", "context size", "context window",
    "maximum context", "token limit", "too many tokens",
    "reduce the length", "exceeds the limit",
    "request entity too large", "prompt is too long",
    "payload too large", "input is too long",
)


def is_context_overflow_failure_result(agent_result: dict, history_len: int) -> bool:
    """One verdict for "this failed turn is a context overflow", shared by transcript persistence
    (#1630 skip) and the user-facing reply so the two can never disagree.

    Multi-word phrases (not bare "exceed"/"token") avoid matching "rate limit exceeded" or
    "invalid authentication token"; a bare 400 only counts on a long session."""
    if not agent_result.get("failed"):
        return False
    if agent_result.get("compression_exhausted"):
        return True
    # An already-classified non-overflow 400 beats the long-history guess: a content safety
    # refusal (#134261) or a rejected reasoning.effort setting (#100536) cannot be compressed away.
    if agent_result.get("failure_reason") in ("content_policy_blocked", "reasoning_mandatory"):
        return False
    err = str(agent_result.get("error") or "").lower()
    return any(p in err for p in _CONTEXT_OVERFLOW_ERROR_PHRASES) or ("400" in err and history_len > 50)
