"""Per-message Telegram addressing facts, outside the cached session prompt."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from telegram import Message
    from plugins.platforms.telegram.adapter import TelegramAdapter


def group_addressing_prompt(
    adapter: "TelegramAdapter", message: "Message", channel_prompt: str | None,
) -> str | None:
    if not adapter._is_group_chat(message) or not getattr(adapter, "_bot", None):
        return channel_prompt
    username = adapter._current_bot_username()
    if not username:
        return channel_prompt
    # Use the same entity-aware check as admission, not the cleaned text or the
    # fact that a turn was dispatched (replies/wake words/open groups also pass).
    mentioned = "yes" if adapter._message_mentions_bot(message) else "no"
    addressing = (
        "Telegram addressing context (current message only):\n"
        f"- Your Telegram bot username: @{username}\n"
        f"- Current message explicitly mentions you: {mentioned}\n"
        "Mentions of other bots do not by themselves ask you to relay the message."
    )
    return f"{channel_prompt}\n\n{addressing}" if channel_prompt else addressing
