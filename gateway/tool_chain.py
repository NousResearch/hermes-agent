"""Collapse consecutive tool-call progress lines into a one-line summary.

Issue #15514: A2A-style turns can fire long heterogeneous tool chains
(web_search → web_extract → web_search → …) that bury the final answer
under one progress line per call.  When the run of *consecutive* tool
lines in the accumulated progress bubble reaches a configurable threshold
(``display.tool_chain_threshold``), the tracker rewrites those lines in
place into a single live-updating summary, e.g.::

    🔍 web_search ×3, 📄 web_extract ×3

The tracker owns no delivery logic: it mutates the progress consumer's
``list[str]`` line buffer in place and the consumer keeps editing the
platform bubble as before.  Users who want full per-call transparency can
either set the threshold to 0 (disabled) or switch the session to
``/verbose`` tool progress, which never collapses.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

# Default chain length at which collapsing kicks in.  Chains of 1-2 calls
# are common and cheap to read; 3+ is where the transcript starts to drown.
DEFAULT_TOOL_CHAIN_THRESHOLD = 3


def format_tool_chain_summary(counts: List[Tuple[str, str, int]]) -> str:
    """Render ``[(emoji, tool_name, count), ...]`` as a compact summary line.

    Tools appear in first-seen order; a count of 1 omits the multiplier.
    """
    parts = []
    for emoji, name, count in counts:
        label = f"{emoji} {name}".strip()
        parts.append(f"{label} ×{count}" if count > 1 else label)
    return ", ".join(parts)


class ToolChainCollapseTracker:
    """Tracks a run of consecutive tool-call lines and collapses it in place.

    Usage by the progress consumer:

    * After appending a tool line to the buffer, call
      ``add(emoji, tool_name, lines)``.  Once the chain reaches the
      threshold, the chain's lines are replaced by a single summary line
      that keeps updating as more calls arrive.
    * Call ``reset()`` whenever anything other than a tool line lands in
      the buffer (thinking text, hints, content-bubble reset, overflow
      roll) — the chain is broken and the next tool line starts fresh.

    The tracker relies on one invariant: once a chain starts at buffer
    index ``i``, every line appended from ``i`` onward belongs to the
    chain until ``reset()`` is called.  The consumer guarantees this by
    resetting on any non-tool append.
    """

    def __init__(self, threshold: int = 0):
        try:
            self.threshold = int(threshold or 0)
        except (TypeError, ValueError):
            self.threshold = 0
        # [[emoji, tool_name, count], ...] in first-seen order.
        self._counts: List[list] = []
        self._start_index: Optional[int] = None
        self.summary_index: Optional[int] = None

    @property
    def enabled(self) -> bool:
        # A threshold below 2 would collapse even trivial 1-call "chains".
        return self.threshold >= 2

    @property
    def active(self) -> bool:
        """True once a chain has been collapsed into a summary line."""
        return self.summary_index is not None

    def reset(self) -> None:
        self._counts.clear()
        self._start_index = None
        self.summary_index = None

    def total_calls(self) -> int:
        return sum(entry[2] for entry in self._counts)

    def _record(self, emoji: str, tool_name: str) -> None:
        for entry in self._counts:
            if entry[0] == emoji and entry[1] == tool_name:
                entry[2] += 1
                return
        self._counts.append([emoji, tool_name, 1])

    def _summary_text(self) -> str:
        return format_tool_chain_summary(
            [(e, n, c) for e, n, c in self._counts]
        )

    def add(self, emoji: str, tool_name: str, lines: List[str]) -> None:
        """Record one tool call whose line was just appended to ``lines``.

        May collapse the chain's lines into a single summary line (in
        place) once the threshold is reached; afterwards the summary line
        is rewritten in place on every subsequent call.
        """
        if not self.enabled:
            return
        if self.summary_index is not None:
            self._record(emoji, tool_name)
            if self.summary_index < len(lines):
                lines[self.summary_index] = self._summary_text()
            return
        if self._start_index is None:
            self._start_index = len(lines) - 1
        self._record(emoji, tool_name)
        if self.total_calls() >= self.threshold:
            lines[self._start_index:] = [self._summary_text()]
            self.summary_index = self._start_index
