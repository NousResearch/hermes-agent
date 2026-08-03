"""User-facing summaries for manual compression commands."""

from __future__ import annotations

from typing import Any, Sequence

from agent.redact import redact_sensitive_text


def summarize_manual_compression(
    before_messages: Sequence[dict[str, Any]],
    after_messages: Sequence[dict[str, Any]],
    before_tokens: int,
    after_tokens: int,
    *,
    compression_state: Any = None,
) -> dict[str, Any]:
    """Return consistent user-facing feedback for manual compression."""
    before_count = len(before_messages)
    after_count = len(after_messages)
    noop = list(after_messages) == list(before_messages)
    aborted = (
        compression_state is not None
        and getattr(compression_state, "_last_compress_aborted", False) is True
    )
    fallback_used = (
        compression_state is not None
        and getattr(compression_state, "_last_summary_fallback_used", False) is True
    )
    failure_reason = (
        getattr(compression_state, "_last_summary_error", None)
        if compression_state is not None
        else None
    )
    if not isinstance(failure_reason, str) or not failure_reason.strip():
        failure_reason = None

    if aborted:
        headline = f"Compression aborted: {before_count} messages preserved"
    elif fallback_used:
        headline = (
            f"Compressed with fallback: {before_count} → {after_count} messages"
        )
    elif noop:
        headline = f"No changes from compression: {before_count} messages"
    else:
        headline = f"Compressed: {before_count} → {after_count} messages"

    if noop and after_tokens == before_tokens:
        token_line = f"Approx request size: ~{before_tokens:,} tokens (unchanged)"
    else:
        token_line = (
            f"Approx request size: ~{before_tokens:,} → "
            f"~{after_tokens:,} tokens"
        )

    note = None
    if aborted:
        note = "Summary generation failed; no messages were removed."
    elif fallback_used:
        dropped_count = getattr(
            compression_state, "_last_summary_dropped_count", None
        )
        if not isinstance(dropped_count, int) or isinstance(dropped_count, bool):
            dropped_count = max(before_count - after_count, 0)
        note = (
            "Summary generation failed; Hermes used limited fallback context "
            f"and removed {dropped_count} message(s)."
        )
    elif not noop and after_count < before_count and after_tokens > before_tokens:
        note = (
            "Note: fewer messages can still raise this estimate when "
            "compression rewrites the transcript into denser summaries."
        )

    if failure_reason and (aborted or fallback_used):
        # This text crosses a user-facing UI boundary.  Never let a disabled
        # global redaction preference expose credentials embedded in provider
        # exception text.
        safe_reason = redact_sensitive_text(failure_reason.strip(), force=True)
        note = f"{note} Reason: {safe_reason}"

    return {
        "noop": noop,
        "aborted": aborted,
        "fallback_used": fallback_used,
        "headline": headline,
        "token_line": token_line,
        "note": note,
    }
