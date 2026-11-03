"""Telegram message deletion rights: whether the bot may delete a user's message in a chat (a secret
answer to a vault prompt is deleted on receipt, ``gateway/run_inbound_secret.py``)."""

import logging
from typing import Optional

from plugins.platforms.telegram.telegram_ids import normalize_telegram_chat_id

logger = logging.getLogger("plugins.platforms.telegram.adapter")


class TelegramMessageDeletionMixin:
    """The deletion-rights probe for ``TelegramAdapter`` (``delete_message`` itself stays on the adapter,
    behind the chat's flood window)."""

    async def can_delete_user_messages(self, chat_id: str, chat_type: Optional[str] = None) -> bool:
        """Bot API: a bot deletes incoming messages in a private chat; in a group or supergroup only as
        the creator or an administrator holding ``can_delete_messages``."""
        from telegram.error import TelegramError

        from plugins.platforms.telegram.adapter import _redact_telegram_error_text
        if not self._bot:
            return False
        if (chat_type or "").lower() in {"dm", "private"}:
            return True
        try:
            member = await self._bot.get_chat_member(chat_id=normalize_telegram_chat_id(chat_id), user_id=self._bot.id)
        except TelegramError as e:
            logger.warning("[%s] Could not read the bot's rights in chat %s: %s", self.name, chat_id, _redact_telegram_error_text(e))
            return False
        status = str(member.status).lower()
        # ``can_delete_messages`` exists on administrators only.
        return status == "creator" or (status == "administrator" and bool(getattr(member, "can_delete_messages", False)))
