"""Telegram message deletion: the bot's own messages, and a user's where the bot holds the right
(a secret answer to a vault prompt is deleted on receipt, ``gateway/run_inbound_secret.py``)."""

import logging
from typing import Optional

from plugins.platforms.telegram.telegram_ids import normalize_telegram_chat_id

logger = logging.getLogger("plugins.platforms.telegram.adapter")


class TelegramMessageDeletionMixin:
    """``delete_message`` and the deletion-rights probe for ``TelegramAdapter``."""

    async def delete_message(self, chat_id: str, message_id: str) -> bool:
        """Delete a message: one the bot posted (Bot API allows it within 48h), or a user's where
        ``can_delete_user_messages`` holds; failures are non-fatal and reported as False.

        Used by the stream consumer's fresh-final cleanup path (ported from openclaw/openclaw#72038) to
        remove long-lived preview messages after sending the completed reply as a fresh message, and by
        the secret-answer path, which checks the result.
        """
        from plugins.platforms.telegram.adapter import _redact_telegram_error_text
        if not self._bot:
            return False
        try:
            await self._bot.delete_message(chat_id=normalize_telegram_chat_id(chat_id), message_id=int(message_id))
            return True
        except Exception as e:  # health: allow BLE001 -- moved from adapter.py: a failed delete is non-fatal by contract, and its text is token-redacted, so no traceback
            logger.debug("[%s] Failed to delete Telegram message %s: %s", self.name, message_id, _redact_telegram_error_text(e))
            return False

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
