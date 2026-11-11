"""Which user-role rows are runtime scaffolding rather than something the person typed."""

from __future__ import annotations

from typing import Any

_SYNTHETIC_USER_PREFIXES = (
    "[System: Your previous response was truncated", "[System: The previous response was cut off",
    "[System: Your previous tool call", "[Your active task list was preserved across context compression]",
    "[IMPORTANT: Background process ",
)


def _message_text(message: Any) -> str:
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text") or part.get("content") or "") for part in content if isinstance(part, dict)
        )
    return ""


_SYNTHETIC_USER_FLAGS = (
    "_todo_snapshot_synthetic", "_empty_recovery_synthetic", "_verification_stop_synthetic", "_pre_verify_synthetic",
    "_dropped_toolcall_nudge", "_dangling_promise_synthetic",
)


def has_scaffolding_flag(message: dict) -> bool:
    """A runtime user-role row stamped by its producer (nudges, task-list snapshots)."""
    return any(message.get(f) for f in _SYNTHETIC_USER_FLAGS)


def is_user_role_scaffolding(message: dict) -> bool:
    """A runtime row in the user role, by flag or opening text (nudges, task-list snapshots, single background-process completion and watch-match notices)."""
    return has_scaffolding_flag(message) or _message_text(message).strip().startswith(_SYNTHETIC_USER_PREFIXES)
