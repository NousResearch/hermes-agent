"""User-facing summaries for manual compression commands."""

from __future__ import annotations

from typing import Any, Sequence


def summarize_manual_compression(
    before_messages: Sequence[dict[str, Any]],
    after_messages: Sequence[dict[str, Any]],
    before_tokens: int,
    after_tokens: int,
) -> dict[str, Any]:
    """Return consistent user-facing feedback for manual compression."""
    before_count = len(before_messages)
    after_count = len(after_messages)
    noop = list(after_messages) == list(before_messages)

    if noop:
        headline = f"No changes from compression: {before_count} messages"
        if after_tokens == before_tokens:
            token_line = (
                f"Rough transcript estimate: ~{before_tokens:,} tokens (unchanged)"
            )
        else:
            token_line = (
                f"Rough transcript estimate: ~{before_tokens:,} → "
                f"~{after_tokens:,} tokens"
            )
    else:
        headline = f"Compressed: {before_count} → {after_count} messages"
        token_line = (
            f"Rough transcript estimate: ~{before_tokens:,} → "
            f"~{after_tokens:,} tokens"
        )

    note = None
    if not noop and after_count < before_count and after_tokens > before_tokens:
        note = (
            "Note: fewer messages can still raise this rough transcript estimate "
            "when compression rewrites the transcript into denser summaries."
        )

    return {
        "noop": noop,
        "headline": headline,
        "token_line": token_line,
        "note": note,
    }
