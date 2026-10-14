"""Tool-progress line shaping for the per-turn runner (``gateway/run_turn_runner.py``)."""

from __future__ import annotations


def _suffix_repeat_counter(msg: str, count: int) -> str:
    """Append the ``(xN)`` repeat counter to a progress line. When the line is a fenced block the
    counter goes on its OWN line: trailing text invalidates a closing fence for anchored parsers
    (Telegram MarkdownV2, CommonMark), which then render the backticks escaped (#138148)."""
    sep = "\n" if msg.rstrip().endswith("```") else " "
    return f"{msg}{sep}(×{count})"
