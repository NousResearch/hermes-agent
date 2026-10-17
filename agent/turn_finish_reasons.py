"""Canonical turn-boundary semantics for persisted assistant finish reasons."""

from __future__ import annotations

from typing import Any


# Only reasons that end the whole user turn belong here. Recovery and guard
# reasons such as ``length``, ``incomplete``, ``verification_required``,
# ``verify_hook_continue``, and ``kanban_terminal_required`` continue the same
# turn and must not close display lineage. Unknown extension reasons are treated
# conservatively as continuing; this affects synthetic replay deduplication only.
TERMINAL_TURN_FINISH_REASONS = frozenset(
    {
        "stop",
        "content_filter",
        "error",
        "cancelled",
        "failed",
    }
)


def finish_reason_ends_turn(reason: Any) -> bool:
    """Whether a persisted assistant finish reason closes the user turn."""
    return isinstance(reason, str) and reason in TERMINAL_TURN_FINISH_REASONS
