"""User-facing summaries for manual compression commands."""

from __future__ import annotations

from typing import Any, Sequence

from agent.redact import redact_sensitive_text


def describe_compression_lock_skip(lock_signal: Any) -> str:
    """User-facing text for a manual /compress skipped by the compression lock.

    ``lock_signal`` is a holder string when another compressor CONFIRMED holds
    the lock, else ``True``/``None``. A failed acquire is NOT proof another
    compression is running (``try_acquire_compression_lock`` swallows
    ``sqlite3.Error``), so the two cases are worded differently.
    """
    if isinstance(lock_signal, str) and lock_signal.strip():
        return (
            f"⏳ Compression already in progress for this session "
            f"(holder: {lock_signal}). Please wait for it to finish."
        )
    return (
        "⏳ Compression skipped: could not acquire this session's compression lock. Another compression may "
        "still be running, or the lock check failed — try again shortly."
    )


def summarize_manual_compression(
    before_messages: Sequence[dict[str, Any]],
    after_messages: Sequence[dict[str, Any]],
    before_tokens: int,
    after_tokens: int,
    *,
    compression_state: Any = None,
) -> dict[str, Any]:
    """Consistent user-facing feedback (headline, token line, optional note) for manual compression."""
    before_count = len(before_messages)
    after_count = len(after_messages)
    noop = list(after_messages) == list(before_messages)

    def flag(name: str) -> bool:
        return getattr(compression_state, name, False) is True

    aborted = flag("_last_compress_aborted")
    refused_would_grow = flag("_last_compress_refused_would_grow")
    fallback_used = flag("_last_summary_fallback_used")
    failure_reason = getattr(compression_state, "_last_summary_error", None)
    if not isinstance(failure_reason, str) or not failure_reason.strip():
        failure_reason = None

    note = None
    if refused_would_grow:
        headline = f"Compression refused (summary would grow the conversation): {before_count} messages preserved"
        note = "The generated summary was larger than what it would replace; no messages were removed."
    elif aborted:
        headline = f"Compression aborted: {before_count} messages preserved"
        note = "Summary generation failed; no messages were removed."
    elif fallback_used:
        headline = f"Compressed with fallback: {before_count} → {after_count} messages"
        dropped_count = getattr(compression_state, "_last_summary_dropped_count", None)
        if not isinstance(dropped_count, int) or isinstance(dropped_count, bool):
            dropped_count = max(before_count - after_count, 0)
        note = (
            "Summary generation failed; Hermes used limited fallback context "
            f"and removed {dropped_count} message(s)."
        )
    elif noop:
        headline = f"No changes from compression: {before_count} messages"
    else:
        headline = f"Compressed: {before_count} → {after_count} messages"
        if after_count < before_count and after_tokens > before_tokens:
            note = (
                "Note: fewer messages can still raise this estimate when "
                "compression rewrites the transcript into denser summaries."
            )

    if (noop and after_tokens == before_tokens) or refused_would_grow:
        token_line = f"Approx request size: ~{before_tokens:,} tokens (unchanged)"
    else:
        token_line = f"Approx request size: ~{before_tokens:,} → ~{after_tokens:,} tokens"

    if failure_reason and (aborted or fallback_used):
        # Crosses a user-facing UI boundary: never let a disabled global redaction
        # preference expose credentials embedded in provider exception text.
        note = f"{note} Reason: {redact_sensitive_text(failure_reason.strip(), force=True)}"

    return {
        "noop": noop,
        "aborted": aborted,
        "refused_would_grow": refused_would_grow,
        "fallback_used": fallback_used,
        "headline": headline,
        "token_line": token_line,
        "note": note,
    }
