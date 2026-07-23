# agent/progress_tracker.py
"""Cross-turn progress detection for subagents.

Distinct from tool_guardrails (which detects intra-turn loops): this tracks
whether the agent is producing meaningful output across turns. An agent that
calls 50 different tools without ever producing text output or creating a file
is "not converging" even though each individual turn looks unique.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Tools that count as "progress" even without text output
_FILE_MUTATION_TOOLS = frozenset({
    "write_file", "patch", "todo", "skill_manage",
    "terminal",  # shell commands create files, run build scripts, etc.
})

_WARN_AFTER = 15  # iterations without progress before warning
_HALT_AFTER = 25  # iterations without progress before hard stop


class ProgressTracker:
    """Track whether an agent is making meaningful progress across turns."""

    def __init__(
        self,
        *,
        warn_after: int = _WARN_AFTER,
        halt_after: int = _HALT_AFTER,
        enabled: bool = True,
    ):
        self.warn_after = warn_after
        self.halt_after = halt_after
        self.enabled = enabled
        self._iterations_since_progress = 0
        self._last_progress_type: str | None = None

    def record_text_response(self) -> None:
        """Call when the model produces a text response (not just tool calls)."""
        self._iterations_since_progress = 0
        self._last_progress_type = "text"

    def record_file_mutation(self, tool_name: str) -> None:
        """Call when a file-creating tool succeeds."""
        if tool_name in _FILE_MUTATION_TOOLS:
            self._iterations_since_progress = 0
            self._last_progress_type = f"mutation:{tool_name}"

    def record_iteration(self) -> None:
        """Call at the end of each tool-calling iteration."""
        if not self.enabled:
            return
        self._iterations_since_progress += 1

    def check(self) -> str | None:
        """Check if the agent is stalled. Returns warning message or None."""
        if not self.enabled:
            return None
        n = self._iterations_since_progress
        if n >= self.halt_after:
            return (
                f"[PROGRESS TRACKER: {n} iterations with no text output or file "
                f"changes. You MUST provide a final response NOW. Summarize your "
                f"findings and stop making tool calls.]"
            )
        if n >= self.warn_after:
            return (
                f"[PROGRESS TRACKER: {n} iterations since last meaningful output. "
                f"You are not converging. Produce text output or file changes soon "
                f"or you will be forced to stop.]"
            )
        return None

    def reset(self) -> None:
        """Reset state for a new conversation."""
        self._iterations_since_progress = 0
        self._last_progress_type = None
