"""Helpers for retry-path messaging and backoff selection.

Extracted from the monolithic ``run_conversation`` body in
``conversation_loop.py`` so that the backoff-parameter selection, the
terminal error-message construction, and the terminal return-dict shape
are independently testable without spinning up a full ``AIAgent``.

These functions encode the *production* decision logic — the conversation
loop calls them, and tests import them directly.
"""

from __future__ import annotations

from typing import Any

from agent.error_classifier import FailoverReason
from agent.retry_utils import jittered_backoff

# ── Constants ────────────────────────────────────────────────────────────

# FailoverReasons that represent a transient provider outage (server
# restart, network hiccup, provider overload/500/502/timeout).  These
# typically last 2-3 minutes, so the retry loop uses an extended backoff
# schedule that spans the outage window instead of giving up at ~14s.
# See issue #33693.
TRANSIENT_OUTAGE_REASONS: frozenset[FailoverReason] = frozenset({
    FailoverReason.overloaded,
    FailoverReason.server_error,
    FailoverReason.timeout,
})

# Extended backoff for transient outages: ~5s + ~10s + ~20s (+ ~40s +
# ~80s if the user configures more retries), which covers the 2-3 minute
# window of a real provider outage.
TRANSIENT_BACKOFF_PARAMS = {"base_delay": 5.0, "max_delay": 120.0}

# Default backoff for non-transient errors.
DEFAULT_BACKOFF_PARAMS = {"base_delay": 2.0, "max_delay": 60.0}


# ── Backoff selection ────────────────────────────────────────────────────


def is_transient_outage(reason: FailoverReason) -> bool:
    """Return *True* when *reason* represents a transient provider outage.

    This is the single source of truth for the ``_is_transient_outage``
    check at ``conversation_loop.py:2944``.  The conversation loop and
    the tests both call this function.
    """
    return reason in TRANSIENT_OUTAGE_REASONS


def select_backoff_params(
    reason: FailoverReason,
    *,
    retry_after: float | None = None,
) -> dict[str, float]:
    """Return the backoff keyword parameters for the given *reason*.

    When a ``retry_after`` header value is present the caller uses it
    directly (the conversation loop handles that before calling this
    function), so this helper only returns the *jittered_backoff*
    parameters — either the extended transient-outage schedule or the
    default schedule.

    Mirrors the decision at ``conversation_loop.py:3845-3853``.
    """
    if is_transient_outage(reason):
        return dict(TRANSIENT_BACKOFF_PARAMS)
    return dict(DEFAULT_BACKOFF_PARAMS)


def compute_backoff_wait(
    retry_count: int,
    reason: FailoverReason,
    *,
    retry_after: float | None = None,
) -> float:
    """Compute the backoff wait time for the given retry attempt.

    This is the pure scheduling decision — it does **not** include the
    adaptive rate-limit override (``adaptive_rate_limit_backoff``) which
    the conversation loop applies separately for rate-limited requests.

    *retry_count* is 1-based, matching the loop's logged attempt number.
    """
    if retry_after:
        return retry_after
    params = select_backoff_params(reason, retry_after=retry_after)
    return jittered_backoff(retry_count, **params)


# ── Terminal error message construction ──────────────────────────────────


def build_terminal_error_message(
    reason: FailoverReason,
    *,
    final_summary: str,
    max_retries: int,
    billing_guidance: str = "",
    is_thinking_timeout: bool = False,
    is_stream_drop: bool = False,
    provider: str = "",
    model: str = "",
) -> str:
    """Build the user-facing terminal error message after all retries fail.

    Mirrors the construction at ``conversation_loop.py:3773-3811``.

    Parameters
    ----------
    reason
        The classified ``FailoverReason`` from ``classify_api_error``.
    final_summary
        The ``agent._summarize_api_error(api_error)`` string.
    max_retries
        The configured maximum retry count.
    billing_guidance
        Optional billing/entitlement guidance appended to billing errors.
    is_thinking_timeout
        When *True*, thinking-timeout guidance is appended (overrides
        the generic stream-drop guidance).
    is_stream_drop
        When *True* (and not a thinking timeout), stream-drop guidance
        is appended.
    provider, model
        Provider and model names, used for thinking-timeout guidance.
    """
    if reason == FailoverReason.billing:
        msg = f"Billing or credits exhausted: {final_summary}"
        if billing_guidance:
            msg += f"\n\n{billing_guidance}"
    elif reason in TRANSIENT_OUTAGE_REASONS:
        # Transient outage — the conversation was saved, so the user can
        # resume once the provider recovers.  Don't show this for
        # permanent failures (billing/auth/policy) where /resume would
        # just hit the same wall.  See issue #33693.
        msg = (
            f"Provider temporarily unavailable after {max_retries} retries: {final_summary}\n\n"
            f"Your conversation has been saved. Use /resume to continue when the provider is back online."
        )
    else:
        msg = f"API call failed after {max_retries} retries: {final_summary}"

    if is_thinking_timeout:
        from agent.thinking_timeout_guidance import (
            build_thinking_timeout_guidance,
        )
        msg += build_thinking_timeout_guidance(provider=provider, model=model)
    elif is_stream_drop:
        msg += (
            "\n\nThe provider's stream connection keeps "
            "dropping — this often happens when generating "
            "very large tool call responses (e.g. write_file "
            "with long content). Try asking me to use "
            "execute_code with Python's open() for large "
            "files, or to write in smaller sections."
        )

    return msg


def build_terminal_return_dict(
    reason: FailoverReason,
    *,
    final_summary: str,
    max_retries: int,
    messages: list | None = None,
    api_call_count: int = 0,
    billing_guidance: str = "",
    is_thinking_timeout: bool = False,
    is_stream_drop: bool = False,
    provider: str = "",
    model: str = "",
) -> dict[str, Any]:
    """Build the terminal return dict after all retries are exhausted.

    Mirrors the return shape at ``conversation_loop.py:3812-3825``.

    The ``failure_reason`` field surfaces the classified reason so
    callers (notably the kanban worker path in ``cli.py``) can
    distinguish a transient throttle from a real failure and choose a
    different exit code.
    """
    final_response = build_terminal_error_message(
        reason,
        final_summary=final_summary,
        max_retries=max_retries,
        billing_guidance=billing_guidance,
        is_thinking_timeout=is_thinking_timeout,
        is_stream_drop=is_stream_drop,
        provider=provider,
        model=model,
    )
    return {
        "final_response": final_response,
        "messages": messages if messages is not None else [],
        "api_calls": api_call_count,
        "completed": False,
        "failed": True,
        "error": final_summary,
        "failure_reason": reason.value,
    }