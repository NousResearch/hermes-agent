"""Localized retry and rejection status copy, with canonical English carried to presentation sinks."""

from __future__ import annotations

from typing import Any, Optional

from agent.error_classifier import FailoverReason
from agent.i18n import tl


# Non-retryable reasons with their own status sentence (catalog key suffix); every other reason gets
# the generic one (naming the HTTP status on the fallback line, the provider on the terminal one).
_NONRETRYABLE_STATUS_KEYS = {
    FailoverReason.content_policy_blocked: "content_policy_blocked",
    FailoverReason.upstream_blocked: "upstream_blocked",
    FailoverReason.ssl_cert_verification: "ssl_cert_verification",
    # Only reached after the one-shot image shrink ran (recover_after_classification sets the flag first).
    FailoverReason.image_too_large: "image_too_large",
}


def nonretryable_fallback_status(reason: Any, status_code: Optional[int]) -> str:
    """``⚠️ … — trying fallback...`` announcing a fallback after a non-retryable 4xx."""
    suffix = _NONRETRYABLE_STATUS_KEYS.get(reason)
    if suffix:
        return tl(f"display.status.turn.nonretryable_fallback_{suffix}")
    return tl("display.status.turn.nonretryable_fallback", status_code=status_code)


def _nonretryable_failed_status(reason: Any, provider_label: str, summary: str) -> str:
    """Terminal ``❌ …: <summary>`` line for a non-retryable 4xx."""
    suffix = _NONRETRYABLE_STATUS_KEYS.get(reason)
    if suffix:
        return tl(f"display.status.turn.nonretryable_failed_{suffix}", summary=summary)
    return tl("display.status.turn.nonretryable_failed", provider=provider_label, summary=summary)


_ZAI_POLICY_NOTES = {
    "zai_coding_overload_long": " (Z.AI Coding overload adaptive long backoff)",
    "zai_coding_overload_short": " (Z.AI Coding overload short retry)",
}


def _adaptive_wait_status(
    *, overloaded: bool, reset: str, policy: Optional[str], wait_time: float, attempt: int, max_retries: int,
    free_busy: bool = False,
) -> str:
    """``⏱️ Rate limited/Provider overloaded. … Waiting …`` — one catalog sentence per shape."""
    kw = {"wait": f"{wait_time:.1f}", "attempt": attempt, "max": max_retries}
    reason = "free_busy" if free_busy else "overloaded" if overloaded else "rate_limited"
    if not policy:
        if reset:
            return tl(f"display.status.turn.{reason}_wait_reset", reset=reset, **kw)
        return tl(f"display.status.turn.{reason}_wait", **kw)
    if overloaded and not reset:
        return tl(f"display.status.turn.{policy}_wait", **kw)
    # A Z.AI policy on a rate-limit verdict or with a reset window: no catalog sentence for that mix.
    _wait_reason = "Provider overloaded" if overloaded else "Rate limited"
    return (
        f"⏱️ {_wait_reason}.{f' Resets in {reset}.' if reset else ''} Waiting {wait_time:.1f}s "
        f"(attempt {attempt}/{max_retries}){_ZAI_POLICY_NOTES.get(policy, '')}..."
    )
