"""Context-overflow failure verdict shared by transcript persistence and user-facing reply."""

from __future__ import annotations

from typing import Any

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
    err = str(agent_result.get("error") or "").lower()
    return any(p in err for p in _CONTEXT_OVERFLOW_ERROR_PHRASES) or ("400" in err and history_len > 50)
