"""Shared formatting helpers for Kanban user notifications."""

from __future__ import annotations


ACTIONABLE_TEXT_LIMIT = 1000
_MIDDLE_TRUNCATION_MARKER = " … [middle truncated] … "


def bound_actionable_text(value: object, limit: int = ACTIONABLE_TEXT_LIMIT) -> str:
    """Bound prompt-like text while preserving both context and trailing action.

    Block reasons are user-action prompts, so their trailing approval scope or
    reply marker must survive.  Keeping the formatted reason below a generous
    fixed limit also avoids turning one malformed reason into a multi-message
    platform flood.  Diagnostic errors and completed summaries intentionally
    remain shorter previews at their call sites; they are not response prompts.
    """
    text = str(value)
    if len(text) <= limit:
        return text
    if limit <= 0:
        return ""
    if limit <= len(_MIDDLE_TRUNCATION_MARKER):
        return text[-limit:]
    remaining = limit - len(_MIDDLE_TRUNCATION_MARKER)
    head_length = remaining // 2
    tail_length = remaining - head_length
    return (
        text[:head_length]
        + _MIDDLE_TRUNCATION_MARKER
        + text[-tail_length:]
    )
