"""Which inbound Telegram updates core treats as messages to the bot."""

from typing import Any, Optional


def effective_update_message(update: Any) -> Optional[Any]:
    """Message-like payload for normal messages and channel posts (``update.channel_post``).

    Business updates (``business_message`` / ``edited_business_message``) are the owner's own chats
    with third parties, relayed because the bot is connected as a Telegram Business chatbot. They are
    never a DM to the bot: answering one as the owner's command sends the reply to the contact. Core
    drops them; a Business plugin claims them in an earlier handler group.
    """
    message = getattr(update, "effective_message", None) or getattr(update, "message", None)
    if message is not None and any(
            message is getattr(update, field, None) for field in ("business_message", "edited_business_message")):
        return None
    return message
