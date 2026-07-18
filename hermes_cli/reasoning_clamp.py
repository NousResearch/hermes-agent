"""Shared reasoning-display clamp logic for the interactive CLI.

The ``/reasoning`` clamp limits how much model thinking is shown: the first
``REASONING_CLAMP_LINES`` lines, then a "... (N more lines)" footer. This
module is the single source of truth for the threshold, the footer text, and
the line-counting, so the streaming reasoning path (``_stream_reasoning_delta``)
and the non-streaming recap box (``_render_assistant_response``) behave
identically and stay in sync. Controlled by the ``display.reasoning_full``
flag (``/reasoning full`` / ``/reasoning clamp``).

Kept dependency-free and pure so it is trivially unit-testable without
importing the heavyweight ``cli`` module.
"""

from __future__ import annotations

# Max reasoning lines shown before the clamp kicks in. Hardcoded (not a config
# key) on purpose: the non-streaming path has used 10 for a long time and a
# configurable threshold adds surface for little value. See issue #53529.
REASONING_CLAMP_LINES = 10


def clamp_notice(hidden_lines: int) -> str:
    """Footer shown when reasoning was clamped. Identical across both paths."""
    return f"... ({hidden_lines} more lines - /reasoning full to show)"


def clamp_lines(
    lines: list[str], show_full: bool, threshold: int = REASONING_CLAMP_LINES,
) -> tuple[list[str], int]:
    """Split a full reasoning block into (visible_lines, hidden_count).

    When ``show_full`` is set, or the block is within ``threshold`` lines,
    everything is visible and ``hidden_count`` is 0. Used by the non-streaming
    recap box, which has the whole block up front.
    """
    if show_full or len(lines) <= threshold:
        return lines, 0
    return lines[:threshold], len(lines) - threshold


class StreamingReasoningClamp:
    """Incremental counterpart to :func:`clamp_lines` for streamed reasoning.

    The streaming path receives reasoning one line at a time and cannot see
    the whole block up front. Call :meth:`should_show` once per line as it is
    about to be printed: it returns ``True`` while under the threshold (or when
    ``show_full`` is set) and otherwise returns ``False`` and counts the line
    as hidden. After the box closes, :attr:`hidden` holds the suppressed count
    for :func:`clamp_notice`.
    """

    def __init__(self, show_full: bool, threshold: int = REASONING_CLAMP_LINES):
        self.show_full = show_full
        self.threshold = threshold
        self.emitted = 0
        self.hidden = 0

    def should_show(self) -> bool:
        self.emitted += 1
        if self.show_full or self.emitted <= self.threshold:
            return True
        self.hidden += 1
        return False
