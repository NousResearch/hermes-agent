"""Formatting helpers for gateway tool-progress presentation.

These helpers are intentionally presentation-only: they choose what a user sees
in transient/editable progress bubbles, not what gets stored in conversation
history.
"""

from __future__ import annotations

import re


_SHELL_PROLOGUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Common first line in model-generated bash snippets.  It is useful for the
    # shell, but as a one-line progress preview it hides the actual work.
    re.compile(r"^set\s+[-+][A-Za-z]*(?:\s+[-+][A-Za-z]*)*(?:\s+pipefail)?\s*$"),
    re.compile(r"^set\s+[-+]o\s+\S+\s*$"),
)


def _is_blank_or_comment(line: str) -> bool:
    stripped = line.strip()
    return not stripped or stripped.startswith("#")


def is_shell_prologue_line(line: str) -> bool:
    """Return True for boilerplate shell setup lines to skip in short previews."""

    stripped = line.strip()
    if _is_blank_or_comment(stripped):
        return True

    # A trailing separator continues the compound command on the next line;
    # remove only that separator so same-line work remains visible.
    candidate = re.sub(r"(?:&&|;)\s*$", "", stripped).rstrip()
    return any(pattern.match(candidate) for pattern in _SHELL_PROLOGUE_PATTERNS)


def terminal_command_preview_line(command: str, *, cap: int = 40) -> str:
    """Return a concise, meaningful one-line preview for a terminal command.

    The gateway's default/all tool-progress mode shows only one line for
    multi-line terminal commands.  Models often start those snippets with
    ``set -euo pipefail``, so naively taking line 1 leaves Discord/Slack users
    seeing only shell boilerplate.  Prefer the first non-empty, non-comment,
    non-``set`` prologue line and append `` ...`` when the command continues.
    """

    raw = (command or "").rstrip()
    if not raw:
        return ""

    lines = raw.splitlines()
    selected_index = 0
    selected = ""

    for index, line in enumerate(lines):
        if is_shell_prologue_line(line):
            continue
        selected_index = index
        selected = line.strip()
        break

    if not selected:
        for index, line in enumerate(lines):
            stripped = line.strip()
            if stripped:
                selected_index = index
                selected = stripped
                break

    if not selected:
        return ""

    has_more = any(
        not _is_blank_or_comment(line) for line in lines[selected_index + 1 :]
    )
    suffix = " ..." if has_more else ""
    effective_cap = cap if cap and cap > 0 else 40

    # Keep room for the continuation marker.  If the command itself is too long,
    # use the normal ellipsis and omit the separate continuation marker; the
    # truncation already communicates that more text exists.
    if len(selected) + len(suffix) > effective_cap:
        if effective_cap <= 3:
            return selected[:effective_cap]
        return selected[: effective_cap - 3] + "..."
    return selected + suffix


__all__ = ["is_shell_prologue_line", "terminal_command_preview_line"]
