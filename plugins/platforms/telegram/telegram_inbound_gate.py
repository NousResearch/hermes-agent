"""Inbound gates the Telegram handlers run after the sender is authorized: the group trigger / observe
gate, blocked-sender logging, and the optional refusal of forwarded messages
(``platforms.telegram.extra.ignore_forwarded_messages``)."""

import logging
from collections.abc import Mapping

from gateway.platforms.event import MessageType

logger = logging.getLogger("plugins.platforms.telegram.adapter")

FORWARDED_NOTICE = (
    "Forwarded messages are ignored on this bot. If you want me to act on it, say so in your own words.")
# Pre-Bot-API-7.0 forward fields; PTB 21+ no longer models them and keeps them in ``api_kwargs``.
_LEGACY_FORWARD_FIELDS = ("forward_from", "forward_from_chat", "forward_from_message_id",
                          "forward_sender_name", "forward_signature", "forward_date")


def is_forwarded_message(msg) -> bool:
    """True for a Bot API 7.0+ ``forward_origin``, the legacy ``forward_*`` fields, or a channel post that
    Telegram auto-forwarded into its linked discussion group. The origin is matched by its Bot API type name
    (``MessageOriginUser`` / ``HiddenUser`` / ``Chat`` / ``Channel``) rather than truthiness, so a mock message
    whose attributes all exist never reads as a forward."""
    origin = getattr(msg, "forward_origin", None)
    if origin is not None and type(origin).__name__.startswith("MessageOrigin"):
        return True
    if getattr(msg, "is_automatic_forward", None) is True:
        return True
    api_kwargs = getattr(msg, "api_kwargs", None)  # PTB hands out a read-only mappingproxy
    if not isinstance(api_kwargs, Mapping):
        return False
    return any(api_kwargs.get(field) is not None for field in _LEGACY_FORWARD_FIELDS)


class TelegramInboundGateMixin:
    """Post-authorization inbound gates for ``TelegramAdapter``."""

    def _log_blocked_user(self, msg, *, level=logging.WARNING, what: str = "unauthorized user") -> None:
        logger.log(
            level, "[Telegram] Blocked %s %s in chat %s", what, getattr(getattr(msg, "from_user", None), "id", None),
            getattr(getattr(msg, "chat", None), "id", None))

    def _gate_or_observe(self, msg, update, msg_type: MessageType) -> bool:
        """Group trigger gate; observes unmentioned chatter when configured. True = proceed."""
        if self._should_process_message(msg):
            return True
        if self._should_observe_unmentioned_group_message(msg):
            self._observe_unmentioned_group_message(msg, msg_type, update_id=update.update_id)
        return False

    async def _drop_forwarded(self, msg) -> bool:
        """With ``ignore_forwarded_messages`` on, drop a forward before any event is built or a group transcript
        observed; True when dropped. Call after the authorization check so only allowed senders get the notice,
        which goes to private chats only and once per album."""
        if not is_forwarded_message(msg) or not self._coerce_bool_extra("ignore_forwarded_messages", False):
            return False
        chat = getattr(msg, "chat", None)
        logger.info("[Telegram] Ignored a forwarded message in chat %s", getattr(chat, "id", None))
        album = getattr(msg, "media_group_id", None)
        if getattr(chat, "type", None) != "private" or (album and album == getattr(self, "_forward_notice_album", None)):
            return True
        self._forward_notice_album = album
        try:
            await msg.reply_text(FORWARDED_NOTICE)
        except Exception:
            logger.warning("[Telegram] Could not send the forwarded-message notice", exc_info=True)
        return True
