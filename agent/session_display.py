"""Helpers for deriving user-visible session text from stored messages."""

import re


_LEADING_WORKSPACE_CONTEXT_RE = re.compile(
    r"^\s*<workspace_context\b(?:\"[^\"]*\"|'[^']*'|[^'\">])*/\s*>\s*"
)


def strip_leading_workspace_context(content: str | None) -> str:
    """Remove injected leading workspace metadata, preserving user content.

    Only self-closing ``workspace_context`` elements at the start of the
    message are removed. Later mentions and non-self-closing user-authored
    elements are deliberately left alone.
    """
    text = content or ""
    while True:
        stripped = _LEADING_WORKSPACE_CONTEXT_RE.sub("", text, count=1)
        if stripped == text:
            return text
        text = stripped


def session_preview(content: str | None, limit: int = 60) -> str:
    """Build a compact session-list preview from a raw user message."""
    raw = " ".join(strip_leading_workspace_context(content).split()).strip()
    if not raw:
        return ""
    return raw[:limit] + ("..." if len(raw) > limit else "")
