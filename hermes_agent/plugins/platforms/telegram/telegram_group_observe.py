"""Telegram observe mode: align triggered group turns with the shared observed session."""

from __future__ import annotations

import dataclasses

from hermes_agent.gateway.platforms.event import MessageEvent, MessageType


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


# video_analyze is in the opt-in ``video`` toolset, so its hint keeps a fallback the default bundle has.
_OBSERVED_MEDIA_TOOL = {"image": "vision_analyze with this path as image_url",
                        "video": "video_analyze with this path as video_url when that tool is available, "
                                 "otherwise your file tools"}


def observed_media_note(cached) -> str:
    """Observed context is text-only (no image tokens on every addressed turn), so a later turn
    never carries the file inline: the note says so and how to open the cached file on demand."""
    how = _OBSERVED_MEDIA_TOOL.get(cached.kind, "your file tools on this path")
    return f"{cached.context_note()} (not attached inline; to look at it use {how})"


def shared_sender_suffix(event: MessageEvent) -> str:
    """Batch-key suffix naming the sender: two members addressing the bot inside one batch window
    are two requests with two actors, never one merged turn under the first sender. Observed-group
    commands keep ``source.user_id`` yet share the group session, so the sender-less check alone
    let their long-command chunks merge across members."""
    actor = getattr(event, "user_id", None) or event.source.user_id
    return f":from:{actor}" if actor else ""
