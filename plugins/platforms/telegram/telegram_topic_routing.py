"""Telegram chat/topic identity and topic-creation bookkeeping (#136567).

Telegram normally stamps a newly created private-chat topic's opening message with the new
``message_thread_id``, but it sometimes delivers that opener with no thread id at all. The
``forum_topic_created`` service message for the same event always carries the new topic id, so it is
the only reliable source for it. Recording it here lets the opener adopt it instead of arriving
lobby-shaped, where gateway topic recovery would pin it to the most recently bound topic.
"""

import logging
import time
from typing import Any, Optional

logger = logging.getLogger("plugins.platforms.telegram.adapter")


class TelegramTopicRoutingMixin:
    """Chat/topic identity helpers and topic-creation bookkeeping for ``TelegramAdapter``."""

    # How long a recorded topic creation may claim a following thread-less message.
    _NEW_TOPIC_OPENER_TTL_SECONDS = 120.0

    @staticmethod
    def _chat_type_str(chat) -> str:
        """PTB enum or plain-string ``chat.type`` → bare lowercase name (``supergroup``)."""
        return str(getattr(chat, "type", "")).split(".")[-1].lower() if chat else ""

    @staticmethod
    def _chat_id_str(message) -> str:
        return str(getattr(getattr(message, "chat", None), "id", ""))

    @classmethod
    def _topic_id_or_general(cls, thread_id) -> str:
        return str(thread_id) if thread_id is not None else cls._GENERAL_TOPIC_THREAD_ID

    def _sender_id_str(self, message) -> str:
        """Sender identity as a string; falls back to the chat id (equal in a private chat)."""
        sender = getattr(message, "from_user", None)
        sender_id = getattr(sender, "id", None)
        return str(sender_id) if sender_id is not None else self._chat_id_str(message)

    def _cache_dm_topic_from_message(self, chat_id: str, thread_id: str, topic_name: str) -> None:
        """Cache a thread_id -> topic_name mapping discovered from an incoming message."""
        cache_key = f"{chat_id}:{topic_name}"
        if cache_key not in self._dm_topics:
            self._dm_topics[cache_key] = int(thread_id)
            logger.info("[%s] Cached DM topic from message: %s -> thread_id=%s", self.name, cache_key, thread_id)

    def _pending_created_topics_store(self) -> dict:
        """``(chat_id, sender_id) -> (thread_id, creation message id, monotonic stamp)``."""
        pending = getattr(self, "_pending_created_topics", None)
        if not isinstance(pending, dict):
            pending = {}
            self._pending_created_topics = pending
        return pending

    async def _handle_forum_topic_created(self, update: Any, context: Any) -> None:
        """Record the topic a ``forum_topic_created`` service message just created (#136567).

        The creation carries the new ``message_thread_id`` and no text, so no other handler matches it.
        Routing signal only: it is never handed to the agent.
        """
        message = getattr(update, "effective_message", None)
        created = getattr(message, "forum_topic_created", None) if message is not None else None
        if created is None:
            return
        # Private chats only: in a forum group a thread-less message is General, never an opener.
        if self._chat_type_str(getattr(message, "chat", None)) != "private":
            return
        thread_id = getattr(message, "message_thread_id", None)
        message_id = getattr(message, "message_id", None)
        if thread_id is None or not isinstance(message_id, int):
            return
        chat_id = self._chat_id_str(message)
        self._pending_created_topics_store()[(chat_id, self._sender_id_str(message))] = (
            str(thread_id), message_id, time.monotonic())
        topic_name = getattr(created, "name", None)
        if topic_name and isinstance(getattr(self, "_dm_topics", None), dict):
            # The creation is the one message that reveals the new topic's name.
            self._cache_dm_topic_from_message(chat_id, str(thread_id), str(topic_name))
        logger.debug("[%s] Recorded new Telegram topic %s (chat=%s)", self.name, thread_id, chat_id)

    def _adopt_created_topic_thread(self, message: Any, chat_type: str, thread_id: Optional[str]) -> Optional[str]:
        """Return the just-created topic's thread id for that topic's thread-less opening message (#136567).

        One-shot and adjacency-bounded: only the message immediately following the creation
        (``message_id == creation + 1``) from the same sender, within ``_NEW_TOPIC_OPENER_TTL_SECONDS``,
        adopts it. Everything else keeps its incoming thread id, so a later lobby message is unaffected.
        """
        if thread_id is not None or chat_type != "dm":
            return thread_id
        pending = getattr(self, "_pending_created_topics", None)
        if not isinstance(pending, dict) or not pending:
            return thread_id
        key = (self._chat_id_str(message), self._sender_id_str(message))
        record = pending.get(key)
        if record is None:
            return thread_id
        created_thread_id, created_message_id, stamped_at = record
        if (time.monotonic() - stamped_at) > self._NEW_TOPIC_OPENER_TTL_SECONDS:
            pending.pop(key, None)
            return thread_id
        if getattr(message, "message_id", None) != created_message_id + 1:
            return thread_id
        pending.pop(key, None)
        logger.info(
            "[%s] New Telegram topic %s adopted by its opening message (chat=%s)",
            self.name, created_thread_id, key[0])
        return created_thread_id
