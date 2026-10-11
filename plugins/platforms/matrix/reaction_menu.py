"""Reaction menus use the adapter's existing choice-picker controls."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from tools.reaction_menu_model import ReactionMenu
from gateway.platforms.base import SendResult

if TYPE_CHECKING:
    from plugins.platforms.matrix.adapter import MatrixAdapter

MAX_PENDING_MENUS = 64
MENU_TIMEOUT_SECONDS = 300.0
EXPIRED_NOTICE = "This menu has expired. Ask for a new menu if you still want to choose."


def withdraw_menu(adapter: MatrixAdapter, prompt) -> None:
    """Remove the seeded reactions so an inactive menu no longer offers choices."""
    prompt.resolved = True
    for reaction_event_id in prompt.bot_reaction_events.values():
        adapter._schedule_reaction_redaction(prompt.chat_id, reaction_event_id, "menu no longer active")


async def expire_menu(adapter: MatrixAdapter, prompt) -> None:
    adapter._choice_picker_prompts_by_event.pop(prompt.message_id, None)
    withdraw_menu(adapter, prompt)
    await adapter._send_invalid_reaction_feedback(
        prompt.chat_id, prompt.message_id, EXPIRED_NOTICE, metadata=prompt.notice_metadata,
    )


async def send_reaction_menu(
    adapter: MatrixAdapter, menu: ReactionMenu, session_key: str, on_selected, metadata: dict,
) -> SendResult:
    if not adapter._client or not metadata.get("requester_user_id"):
        return SendResult(success=False, error="A connected Matrix requester is required")
    async with adapter._reaction_menu_send_lock:
        return await _send_menu_picker(adapter, menu, session_key, on_selected, metadata)


async def _send_menu_picker(adapter: MatrixAdapter, menu: ReactionMenu, session_key: str, on_selected, metadata: dict) -> SendResult:
    registry = adapter._choice_picker_prompts_by_event
    for event_id, prompt in list(registry.items()):
        if not prompt.is_menu:
            continue
        if adapter._matrix_prompt_expired(prompt) or prompt.session_key == session_key:
            registry.pop(event_id)
            withdraw_menu(adapter, prompt)
    if sum(prompt.is_menu for prompt in registry.values()) >= MAX_PENDING_MENUS:
        return SendResult(success=False, error="Too many pending Matrix menus")

    lines = [menu.prompt, "", *(f"{option.emoji} {option.label}" for option in menu.options), "", "React to choose within five minutes."]
    choices = {option.emoji: (menu, option) for option in menu.options}
    try:
        return await adapter._send_picker(
            metadata["chat_id"], lines, choices, session_key, on_selected, metadata, registry, "reaction menu", is_menu=True)
    except asyncio.CancelledError:
        for event_id, prompt in list(registry.items()):
            if prompt.is_menu and prompt.on_selected == on_selected:
                registry.pop(event_id)
        raise
