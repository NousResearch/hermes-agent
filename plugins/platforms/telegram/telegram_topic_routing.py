"""Telegram chat/topic identity and topic-creation bookkeeping (#136567).

Telegram normally stamps a newly created private-chat topic's opening message with the new
``message_thread_id``, but it sometimes delivers that opener with no thread id at all. The
``Message`` carrying ``forum_topic_created`` supplies the new topic id, not ``ForumTopicCreated``
itself. Recording authorized, non-bot private-chat creations in a bounded, short-lived store lets
an adjacent thread-less opener adopt it instead of arriving lobby-shaped, where gateway topic
recovery would pin it to the most recently bound topic. The first later message from the same
chat and sender consumes the record, even when already threaded, so each creation is one-shot.
"""

import logging
import time
from typing import Any, Optional

logger = logging.getLogger("plugins.platforms.telegram.adapter")


class TelegramTopicRoutingMixin:
    """Chat/topic identity helpers and topic-creation bookkeeping for ``TelegramAdapter``."""

    # How long a recorded topic creation may claim a following thread-less message.
    _NEW_TOPIC_OPENER_TTL_SECONDS = 120.0
    _NEW_TOPIC_OPENER_MAX_PENDING = 256

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
        """``(chat_id, sender_id) -> (thread_id, creation message id, monotonic stamp)``.

        Prune expired records before returning the store for insertion. The caller then evicts
        the oldest stamps as needed to keep new records within ``_NEW_TOPIC_OPENER_MAX_PENDING``.
        A later message from the same chat and sender consumes its record even if already threaded,
        rather than leaving healthy openers' records around until TTL expiry (#136567).
        """
        pending = getattr(self, "_pending_created_topics", None)
        if not isinstance(pending, dict):
            pending = {}
            self._pending_created_topics = pending
            return pending
        now = time.monotonic()
        expired = [
            key for key, record in pending.items()
            if (now - record[2]) > self._NEW_TOPIC_OPENER_TTL_SECONDS
        ]
        for key in expired:
            pending.pop(key, None)
        return pending

    async def _handle_forum_topic_created(self, update: Any, context: Any) -> None:
        """Record an authorized, non-bot private-chat topic creation (#136567).

        The service ``Message`` supplies ``message_thread_id``; ``ForumTopicCreated`` does not.
        Apply the existing intake authorization gate before remembering routing state or caching
        a name, so rejected users cannot populate either. Routing signal only: never an agent turn.
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
        sender = getattr(message, "from_user", None)
        if sender is None or getattr(sender, "is_bot", False) is True:
            return
        if not self._is_user_authorized_from_message(message):
            return
        chat_id = self._chat_id_str(message)
        pending = self._pending_created_topics_store()
        while len(pending) >= self._NEW_TOPIC_OPENER_MAX_PENDING:
            oldest = min(pending, key=lambda key: pending[key][2])
            pending.pop(oldest)
        pending[(chat_id, self._sender_id_str(message))] = (
            str(thread_id), message_id, time.monotonic())
        topic_name = getattr(created, "name", None)
        if topic_name and isinstance(getattr(self, "_dm_topics", None), dict):
            # The creation is the one message that reveals the new topic's name.
            self._cache_dm_topic_from_message(chat_id, str(thread_id), str(topic_name))
        logger.debug("[%s] Recorded new Telegram topic %s (chat=%s)", self.name, thread_id, chat_id)

    def _adopt_created_topic_thread(self, message: Any, chat_type: str, thread_id: Optional[str]) -> Optional[str]:
        """Return the just-created topic's thread id for its thread-less non-bot opener (#136567).

        The first later integer message id from the same DM chat and sender consumes the record,
        including already-threaded, non-adjacent or expired cases: one-shot, not left until TTL.
        Adoption requires adjacency (``message_id == creation + 1``), a fresh record and no thread
        in either the incoming routing value or the raw ``Message``. Other messages keep their thread.
        """
        if chat_type != "dm":
            return thread_id
        pending = getattr(self, "_pending_created_topics", None)
        if not isinstance(pending, dict) or not pending:
            return thread_id
        key = (self._chat_id_str(message), self._sender_id_str(message))
        record = pending.get(key)
        if record is None:
            return thread_id
        created_thread_id, created_message_id, stamped_at = record
        message_id = getattr(message, "message_id", None)
        if not isinstance(message_id, int) or message_id <= created_message_id:
            return thread_id
        pending.pop(key)
        if (time.monotonic() - stamped_at) > self._NEW_TOPIC_OPENER_TTL_SECONDS:
            return thread_id
        if thread_id is not None or getattr(message, "message_thread_id", None) is not None:
            return thread_id
        if message_id != created_message_id + 1 or getattr(getattr(message, "from_user", None), "is_bot", False) is True:
            return thread_id
        logger.info(
            "[%s] New Telegram topic %s adopted by its opening message (chat=%s)",
            self.name, created_thread_id, key[0])
        return created_thread_id
