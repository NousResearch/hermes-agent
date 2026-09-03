"""Pure helpers for building user-visible transcript projections."""

from __future__ import annotations

from typing import Any, Mapping


def is_interim_assistant_message(message: Mapping[str, Any]) -> bool:
    """Return whether *message* is assistant narration attached to tool calls.

    Hermes persists text emitted alongside ``tool_calls`` so an enabled client
    can restore the complete narrated timeline. When the user disables
    ``display.interim_assistant_messages`` only the display projection should
    hide that text; the durable/model transcript and tool-call metadata remain
    untouched.
    """

    if message.get("role") != "assistant" or not message.get("tool_calls"):
        return False
    content = message.get("content")
    if isinstance(content, str):
        return bool(content.strip())
    return bool(content)


def project_interim_assistant_for_display(
    message: Mapping[str, Any], *, enabled: bool
) -> dict[str, Any]:
    """Return a display copy that hides disabled interim assistant narration."""

    projected = dict(message)
    if not enabled and is_interim_assistant_message(message):
        projected["display_kind"] = "hidden"
    return projected
