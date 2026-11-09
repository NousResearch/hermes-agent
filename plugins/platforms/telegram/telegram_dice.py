"""Inbound Telegram dice (🎲 🎯 🏀 ⚽ 🎳 🎰): an animated emoji with a server-rolled value.

A dice message carries no text or media, so without its own handler PTB matched no filter
and the throw vanished. Same gate/observe path as a location pin.
"""

from __future__ import annotations

from gateway.platforms.event import MessageType


async def handle_dice_message(adapter, update, context) -> None:
    msg = adapter._effective_update_message(update)
    dice = getattr(msg, "dice", None) if msg else None
    if dice is None:
        return
    if not adapter._is_user_authorized_from_message(msg):
        adapter._log_blocked_user(msg)
        return
    if not adapter._gate_or_observe(msg, update, MessageType.TEXT):
        return
    event = adapter._build_message_event(msg, MessageType.TEXT, update_id=update.update_id)
    event.text = f"[The user sent a Telegram dice: {dice.emoji} rolled {dice.value}.]"
    await adapter.handle_message(adapter._apply_telegram_group_observe_attribution(event))
