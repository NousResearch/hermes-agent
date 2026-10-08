"""Gateway classification for failed turns that must not grow an oversized transcript."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_CONTEXT_OVERFLOW_ERROR_PHRASES = (
    "context length", "context size", "context window",
    "maximum context", "token limit", "too many tokens",
    "reduce the length", "exceeds the limit",
    "request entity too large", "prompt is too long",
    "payload too large", "input is too long",
)
_CONTEXT_PRESSURE_FAILURE_REASONS = frozenset({
    "context_overflow", "payload_too_large", "long_context_tier",
})


def is_context_overflow_failure_result(agent_result: dict, history_len: int) -> bool:
    """One verdict for transcript persistence and the user-facing reply.

    A stamped failure reason is authoritative: a classified content-policy or request-shape
    rejection must not be reclassified from the same long-session 400 heuristic.
    """
    if not agent_result.get("failed"):
        return False
    if agent_result.get("compression_exhausted"):
        return True
    err = str(agent_result.get("error") or "").lower()
    reason = str(agent_result.get("failure_reason") or "").strip()
    if reason:
        return reason in _CONTEXT_PRESSURE_FAILURE_REASONS
    return any(p in err for p in _CONTEXT_OVERFLOW_ERROR_PHRASES) or ("400" in err and history_len > 50)


def is_context_overflow_exception(
    error: BaseException, history_len: int, *, approx_tokens: int = 0, context_length: int = 200_000,
) -> bool:
    """Classify an escaped provider exception without letting a long transcript override it.

    Explicit context-size evidence remains sufficient; otherwise the provider classifier's
    named verdict wins, and the legacy status/length fallback is reserved for unclassified
    errors only.
    """
    explicit_reason = str(getattr(error, "failure_reason", "") or "").strip()
    if explicit_reason:
        return explicit_reason in _CONTEXT_PRESSURE_FAILURE_REASONS

    err = str(error or "").lower()
    try:
        from agent.error_classifier import classify_api_error

        classified = classify_api_error(
            error, num_messages=history_len, approx_tokens=approx_tokens, context_length=context_length,
        )
        if classified.should_compress:
            return True
        reason = classified.reason.value
    except Exception:
        logger.debug("Failed to classify gateway provider exception", exc_info=True)
        reason = ""
    if reason and reason not in {"format_error", "unknown"}:
        return False

    if any(p in err for p in _CONTEXT_OVERFLOW_ERROR_PHRASES):
        return True

    if reason:
        return False

    status_code = getattr(error, "status_code", None)
    return status_code in {400, 500} and history_len > 50
