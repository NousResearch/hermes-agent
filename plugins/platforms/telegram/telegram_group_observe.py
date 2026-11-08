"""Telegram observe mode: align triggered group turns with the shared observed session."""

from __future__ import annotations

import dataclasses

from gateway.platforms.event import MessageEvent, MessageType


def apply_group_observe_attribution(adapter, event: MessageEvent) -> MessageEvent:
    """Route a triggered turn in an observed group onto the chat's shared session.

    Text turns get the sender-less shared source plus a ``[name|id]`` tag; the trusted sender
    stays on ``event.user_id`` for the live-turn actor. Commands keep ``user_id`` so slash access
    checks see the sender, but are flagged ``shared_session`` so they key the same session.
    """
    if not adapter._telegram_observe_unmentioned_group_messages():
        return event
    raw_message = getattr(event, "raw_message", None)
    if not raw_message or not adapter._is_group_chat(raw_message):
        return event
    allowed = adapter._telegram_observe_allowed_chats()
    if not allowed or adapter._chat_id_str(raw_message) not in allowed:
        return event
    observe_prompt = adapter._telegram_group_observe_channel_prompt()
    channel_prompt = f"{event.channel_prompt}\n\n{observe_prompt}" if event.channel_prompt else observe_prompt
    if event.message_type == MessageType.COMMAND:
        return dataclasses.replace(
            event, source=dataclasses.replace(event.source, shared_session=True), channel_prompt=channel_prompt)
    return dataclasses.replace(
        event, text=adapter._telegram_group_observe_attributed_text(event),
        source=adapter._telegram_group_observe_shared_source(event.source), channel_prompt=channel_prompt)
