"""Controlled display projection for persisted synthetic conversation turns."""

from typing import Any


def project_history_message_content(message: dict) -> Any:
    """Return visible content without exposing model-only goal scaffolding."""
    display_kind = message.get("display_kind")
    if display_kind == "goal_resume":
        return "/goal resume"
    if display_kind == "goal_continue":
        return "Continuing standing goal…"
    return message.get("content")
