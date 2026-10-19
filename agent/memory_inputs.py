"""Human input for memory providers, separate from the model's cached transcript."""

from __future__ import annotations

from typing import Any

from agent.message_content import flatten_message_text
from agent.skill_commands import extract_user_instruction_from_skill_message


# The existing room-origin marker, authored by Desktop's
# apps/desktop/src/plugins/hermes-bots/group-round-prompt.ts::GROUP_PROMPT_HEADER_PREFIX.
# A room prompt contains peer messages and participation rules, not human-authored speech.
_GROUP_PROMPT_HEADER_PREFIX = '[Group chat: "'


def memory_user_text(content: Any) -> str | None:
    """Strip skill scaffolding and omit generated group-room turns from provider input."""
    if flatten_message_text(content).lstrip().startswith(_GROUP_PROMPT_HEADER_PREFIX):
        return None
    return extract_user_instruction_from_skill_message(content)


def memory_messages(messages: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """Omit generated room exchanges from a provider snapshot without changing live history.

    A per-group session also accepts genuine CLI/direct input. Resume memory ingestion at its next
    human turn, keeping room replies, tool results and continuation nudges with their excluded prompt.
    """
    if messages is None:
        return None
    # Lazy: the compressor imports the memory manager during turn setup.
    from agent.context_compressor import _synthetic_user_row, user_originated_turn_view

    result = []
    room_exchange = False
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            live = user_originated_turn_view(message)
            text = flatten_message_text((live if live is not None else message).get("content"))
            if text.lstrip().startswith(_GROUP_PROMPT_HEADER_PREFIX):
                room_exchange = True
            elif room_exchange and (
                message.get("display_kind") == "skill_invocation" or (live is not None and not _synthetic_user_row(text))
            ):
                room_exchange = False
        if not room_exchange:
            result.append(message)
    return result
