"""Route a client-created Telegram DM topic's thread-less opener into that topic (#136567).

Telegram for Android (13.0.x) delivers the opening message of a topic created from the "All
messages" view with no ``message_thread_id``. The ``Message`` carrying ``forum_topic_created`` is then
the only update that names the new topic (``ForumTopicCreated`` itself has only name/icon fields).
Recording authorized, non-bot private-chat creations lets the thread-less messages that follow adopt
the new topic instead of arriving lobby-shaped, where gateway topic recovery would pin them to the
most recently bound topic.

Adoption follows time and per-chat order, never message-id adjacency: private-chat message ids come
from one sequence shared by every private chat of the bot account (bot replies included), so a send
elsewhere between the creation and the opener breaks ``creation + 1``. Per-chat FIFO update
processing guarantees nothing from this chat lands in between.
"""

import logging
import time
from typing import Any, Optional

logger = logging.getLogger("plugins.platforms.telegram.adapter")


class TelegramTopicRoutingMixin:
    """Topic-creation bookkeeping for ``TelegramAdapter`` (relies on its ``_chat_type_str``/``_chat_id_str``)."""

    # A thread-less message from the creator adopts the new topic while it was SENT within this many
    # seconds of the creation or of the previous adopted message (the client sends the opener
    # ~0.2-1 s after the creation; a split long text or an album follows within the same second).
    # Measured on Telegram's ``Message.date`` stamps, never on handler time: per-chat FIFO holds a
    # follow-up behind the previous handler (a photo download, STT), and it must not age out there.
    _NEW_TOPIC_FOLLOW_SECONDS = 5.0
    _NEW_TOPIC_OPENER_MAX_PENDING = 256
    # Memory bound only (local clock), far above any handler backlog; adoption never reads it.
    _NEW_TOPIC_RECORD_TTL_SECONDS = 600.0

    @staticmethod
    def _creator_key(message) -> str:
        return str(getattr(getattr(message, "from_user", None), "id", ""))

    @staticmethod
    def _sent_at(message) -> Optional[float]:
        """Telegram's send stamp (whole seconds); an edit keeps its message's original ``date``."""
        date = getattr(message, "date", None)
        return date.timestamp() if hasattr(date, "timestamp") else None

    async def _handle_forum_topic_created(self, update: Any, context: Any) -> None:
        """Record an authorized, non-bot private-chat topic creation (#136567).

        Routing signal only, never an agent turn. The user-chosen name is NOT cached in ``_dm_topics``:
        entries there mark operator-declared topics, which session-title auto-rename skips.
        """
        message = getattr(update, "effective_message", None)
        if message is None or getattr(message, "forum_topic_created", None) is None:
            return
        # Private chats only: in a forum group a thread-less message is General, never an opener.
        if self._chat_type_str(getattr(message, "chat", None)) != "private":
            return
        thread_id = getattr(message, "message_thread_id", None)
        sender = getattr(message, "from_user", None)
        sent_at = self._sent_at(message)
        if thread_id is None or sent_at is None or sender is None or getattr(sender, "is_bot", False) is True:
            return
        if not self._is_user_authorized_from_message(message):
            return
        pending = self.__dict__.setdefault("_pending_created_topics", {})
        now = time.monotonic()
        for key in [k for k, rec in pending.items() if now - rec[2] > self._NEW_TOPIC_RECORD_TTL_SECONDS]:
            del pending[key]
        while len(pending) >= self._NEW_TOPIC_OPENER_MAX_PENDING:
            del pending[min(pending, key=lambda k: pending[k][2])]
        chat_id = self._chat_id_str(message)
        pending[(chat_id, self._creator_key(message))] = (str(thread_id), sent_at, now)
        logger.debug("[%s] Recorded new Telegram topic %s (chat=%s)", self.name, thread_id, chat_id)

    def _adopt_created_topic_thread(self, message: Any, chat_type: str, thread_id: Optional[str]) -> Optional[str]:
        """Thread id of the topic this sender just created, for its thread-less follow-up messages.

        A threaded message (the topic now routes normally) or a send gap longer than
        ``_NEW_TOPIC_FOLLOW_SECONDS`` ends the follow; each adoption extends it, so every chunk of a
        split opener lands in the same topic.
        """
        # .get: adapters built without __init__ (object.__new__ in tests) have no store yet.
        pending = self.__dict__.get("_pending_created_topics")
        if chat_type != "dm" or not pending:
            return thread_id
        key = (self._chat_id_str(message), self._creator_key(message))
        record = pending.get(key)
        sent_at = self._sent_at(message)
        # An edit of a message sent before the creation stays where that message lived and leaves
        # the record for the real opener.
        if record is None or sent_at is None or sent_at < record[1]:
            return thread_id
        del pending[key]
        if thread_id is not None or getattr(message, "message_thread_id", None) is not None:
            return thread_id
        created_thread_id, last_sent_at, _recorded = record
        if sent_at - last_sent_at > self._NEW_TOPIC_FOLLOW_SECONDS:
            return thread_id
        pending[key] = (created_thread_id, sent_at, time.monotonic())
        logger.info("[%s] Thread-less message %s adopted by new Telegram topic %s (chat=%s)",
                    self.name, getattr(message, "message_id", "?"), created_thread_id, key[0])
        return created_thread_id
