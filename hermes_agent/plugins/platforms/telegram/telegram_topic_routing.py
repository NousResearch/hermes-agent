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
    """Topic-creation bookkeeping and topic gates for ``TelegramAdapter`` (relies on its ``_chat_type_str``/``_chat_id_str``)."""

    # A thread-less message from the creator adopts the new topic while it was SENT within this many
    # seconds of the creation or of the previous adopted message (the client sends the opener
    # ~0.2-1 s after the creation; a split long text or an album follows within the same second).
    # Measured on Telegram's ``Message.date`` stamps, never on handler time: per-chat FIFO holds a
    # follow-up behind the previous handler (a photo download, STT), and it must not age out there.
    _NEW_TOPIC_FOLLOW_SECONDS = 5.0
    _NEW_TOPIC_OPENER_MAX_PENDING = 256
    # Memory bound only (local clock), far above any handler backlog; adoption never reads it.
    _NEW_TOPIC_RECORD_TTL_SECONDS = 600.0
    # Per-message decisions, held from update admission until that message's handler runs.
    _NEW_TOPIC_DECISIONS_CAP = 4096

    @staticmethod
    def _creator_key(message) -> str:
        return str(getattr(getattr(message, "from_user", None), "id", ""))

    @staticmethod
    def _sent_at(message) -> Optional[float]:
        """Telegram's send stamp (whole seconds); an edit keeps its message's original ``date``."""
        date = getattr(message, "date", None)
        return date.timestamp() if hasattr(date, "timestamp") else None

    async def _handle_forum_topic_created(self, update: Any, context: Any) -> None:
        """Routing signal only, never an agent turn (#136567). The user-chosen name is NOT cached in
        ``_dm_topics``: entries there mark operator-declared topics, which session-title auto-rename skips."""
        message = getattr(update, "effective_message", None)
        if message is not None and getattr(message, "forum_topic_created", None) is not None:
            self._admit_topic_follow(message)

    def _adopt_created_topic_thread(self, message: Any, chat_type: str, thread_id: Optional[str]) -> Optional[str]:
        """Thread id of the topic this sender just created, for its thread-less follow-up messages."""
        if chat_type != "dm":
            return thread_id
        adopted = self._admit_topic_follow(message)
        return thread_id if thread_id is not None or adopted is None else adopted

    def _admit_topic_follow(self, message: Any) -> Optional[str]:
        """The one ordered decision for client-created DM topics: the created topic's id for a thread-less
        follower, else None. Update admission calls it in arrival order, before any handler waits on a
        per-topic lane; the handlers reuse that per-message result. Deciding again at handler time would
        read a record a later update already changed (a threaded message ends the follow while a slow
        root update still holds the creation's own handler)."""
        message_id = getattr(message, "message_id", None)
        if self._chat_type_str(getattr(message, "chat", None)) != "private" or message_id is None:
            return None
        decided = self.__dict__.setdefault("_topic_follow_decisions", {})
        ident = (self._chat_id_str(message), message_id)  # an edit keeps its message's decision
        if ident not in decided:
            while len(decided) >= self._NEW_TOPIC_DECISIONS_CAP:
                del decided[next(iter(decided))]
            decided[ident] = self._decide_topic_follow(message)
        return decided[ident]

    def _decide_topic_follow(self, message: Any) -> Optional[str]:
        """Record an authorized, non-bot creation; a threaded message or a send gap longer than
        ``_NEW_TOPIC_FOLLOW_SECONDS`` ends the follow; each adoption extends it, so every chunk of a
        split opener lands in the same topic."""
        sender, sent_at = getattr(message, "from_user", None), self._sent_at(message)
        thread_id = getattr(message, "message_thread_id", None)
        if sender is None or sent_at is None:
            return None
        pending = self.__dict__.setdefault("_pending_created_topics", {})
        key, now = (self._chat_id_str(message), self._creator_key(message)), time.monotonic()
        if getattr(message, "forum_topic_created", None) is not None:
            if thread_id is None or getattr(sender, "is_bot", False) is True or not self._is_user_authorized_from_message(message):
                return None
            for stale in [k for k, rec in pending.items() if now - rec[2] > self._NEW_TOPIC_RECORD_TTL_SECONDS]:
                del pending[stale]
            while len(pending) >= self._NEW_TOPIC_OPENER_MAX_PENDING:
                del pending[min(pending, key=lambda k: pending[k][2])]
            pending[key] = (str(thread_id), sent_at, now)
            logger.debug("[%s] Recorded new Telegram topic %s (chat=%s)", self.name, thread_id, key[0])
            return None
        record = pending.get(key)
        # An edit of a message sent before the creation stays where that message lived and leaves
        # the record for the real opener.
        if record is None or sent_at < record[1]:
            return None
        del pending[key]
        if thread_id is not None or sent_at - record[1] > self._NEW_TOPIC_FOLLOW_SECONDS:
            return None
        pending[key] = (record[0], sent_at, now)
        logger.info("[%s] Thread-less message %s adopted by new Telegram topic %s (chat=%s)",
                    self.name, getattr(message, "message_id", "?"), record[0], key[0])
        return record[0]

    def _topic_gates_pass(self, thread_id, *, warn_non_numeric: bool, message: Any = None) -> Optional[bool]:
        """``allowed_topics`` / ``ignored_threads`` gates; False = blocked, None = undecided."""
        allowed_topics = self._telegram_allowed_topics()
        if allowed_topics and self._topic_id_or_general(thread_id) not in allowed_topics:
            self._log_topic_gate_drop(message, thread_id, "allowed_topics", sorted(allowed_topics))
            return False
        if thread_id is not None:
            try:
                if int(thread_id) in self._telegram_ignored_threads():
                    return False
            except (TypeError, ValueError):
                if not warn_non_numeric:
                    return False
                logger.warning("[%s] Ignoring non-numeric Telegram message_thread_id: %r", self.name, thread_id)
        return None

    def _log_topic_gate_drop(self, message: Any, thread_id: Any, gate: str, allowed: list) -> None:
        """One INFO line per (chat, topic) the first time ``allowed_topics`` drops it: the drop is
        otherwise silent, and a topic the operator meant to allow looks like a dead gateway (#52635).
        A missing ``message_thread_id`` is the General topic ``1``."""
        logged = self.__dict__.setdefault("_topic_gate_drops_logged", set())
        key = (self._chat_id_str(message), self._topic_id_or_general(thread_id))
        if key in logged:
            return
        if len(logged) >= self._NEW_TOPIC_OPENER_MAX_PENDING:
            logged.clear()
        logged.add(key)
        logger.info("[%s] Ignoring messages in chat %s topic %s: not in %s %s (a message without "
                    "message_thread_id counts as the General topic 1)", self.name, key[0], key[1], gate, allowed)

    def _is_ignored_root_dm(self, message: Any, *, is_command: bool) -> bool:
        """``extra.ignore_root_dm``: a non-command root-DM message (no topic) from a chat with ``dm_topics``
        configured is dropped, so the root DM is a command lobby; other chats are unaffected. A thread-less
        opener of a just-created topic is not root chatter: it adopts that topic (``_admit_topic_follow``),
        nor is a client-split continuation of an admitted near-limit command (``_continues_lobby_split``)."""
        lobby = None if is_command else self._root_lobby_key(message)
        return (lobby is not None and self._admit_topic_follow(message) is None
                and not self._continues_lobby_split(message, lobby))

    def _root_lobby_key(self, message: Any) -> Optional[tuple]:
        """(chat, sender) when ``extra.ignore_root_dm`` makes ``message``'s root DM a command lobby, else None."""
        if (not self.config.extra.get("ignore_root_dm", False) or self._effective_message_thread_id(message) is not None
                or self._chat_id_str(message) not in self._dm_topic_chat_ids):
            return None
        return self._chat_id_str(message), self._creator_key(message)

    def _continues_lobby_split(self, message: Any, lobby: tuple) -> bool:
        """A text chunk while this sender's admitted lobby command (``_handle_command`` tags it
        ``_lobby_split_owner``) is still buffered with a near-limit last chunk: Telegram split a >4096-char
        command, and dropping the TEXT remainder would run the command truncated. Root chatter after a
        short final chunk, or after the batch flushed, is still dropped."""
        return bool(getattr(message, "text", None)) and any(
            getattr(event, "_lobby_split_owner", None) == lobby
            and getattr(event, "_last_chunk_len", 0) >= self._SPLIT_THRESHOLD
            for event in self._pending_text_batches.values())

    def _text_batch_key(self, event: Any) -> str:
        """Session-scoped batching key; topic recovery first so DM-topic batches coalesce on the recovered lane."""
        self._apply_topic_recovery(event)
        return super()._text_batch_key(event)

    def _apply_topic_recovery(self, event: Any) -> None:
        """Topic recovery plus the recovered topic's own prompt inputs (#131232)."""
        from plugins.platforms.telegram.telegram_topic_transfer import apply_topic_recovery_with_inputs
        apply_topic_recovery_with_inputs(self, event, super()._apply_topic_recovery)

    def configured_channels(self) -> list[dict[str, Any]]:
        """Channel-directory targets from ``extra.dm_topics`` / ``extra.group_topics`` that carry a thread_id,
        so configured topics stay sendable after their sessions are pruned (#52629). A topic name used in
        more than one chat is qualified with its chat id so a by-name send cannot pick the wrong chat."""
        group_topics = self.config.extra.get("group_topics") or []
        if isinstance(group_topics, dict):
            group_topics = [{"chat_id": cid, "topics": topics} for cid, topics in group_topics.items()]
        rows = [(entry, "dm") for entry in self._dm_topics_config or ()] + [(entry, "group") for entry in group_topics]
        found = []
        for entry, kind in rows:
            chat_id = entry.get("chat_id") if isinstance(entry, dict) else None
            for topic in (entry.get("topics") or []) if chat_id is not None else ():
                if isinstance(topic, dict) and topic.get("thread_id"):  # group-topic ``name`` is optional
                    tid = str(topic["thread_id"])
                    found.append((str(chat_id), tid, str(topic.get("name") or f"{chat_id} / topic {tid}"), kind))

        def _match_key(name: str) -> str:  # resolve_channel_name's normalisation: a "#x" label resolves as "x"
            return name.lstrip("#").strip().lower()
        chats_per_name: dict[str, set] = {}
        for chat_id, _tid, name, _kind in found:
            chats_per_name.setdefault(_match_key(name), set()).add(chat_id)
        return [{"id": f"{chat_id}:{tid}", "thread_id": tid, "type": kind,
                 "name": name if len(chats_per_name[_match_key(name)]) == 1 else f"{chat_id} / {name}"}
                for chat_id, tid, name, kind in found]

    def _merge_saved_dm_topic_ids(self) -> None:
        """Fill thread_ids saved since boot into this adapter's ``dm_topics``: a rebuilt adapter's boot-time
        snapshot predates them (#132522). Read from where ``_persist_dm_topic_thread_id`` writes and merged,
        never swapped in: the gateway loader also accepts ``dm_topics`` in a top-level ``telegram:`` block
        or under ``gateway.platforms``, which that path does not hold."""
        try:
            from hermes_agent.hermes_cli.config import load_config_readonly
            saved = load_config_readonly().get("platforms", {}).get("telegram", {}).get("extra", {}).get("dm_topics") or []
        except Exception as e:
            logger.debug("[%s] Could not read saved DM topic thread_ids: %s", self.name, e, exc_info=True)
            saved = []
        ids = {(str(chat.get("chat_id")), topic.get("name")): topic["thread_id"]
               for chat in saved if isinstance(chat, dict)
               for topic in chat.get("topics") or () if isinstance(topic, dict) and topic.get("thread_id")}
        for chat in self._dm_topics_config or ():
            for topic in chat.get("topics") or ():
                if saved_id := ids.get((str(chat.get("chat_id")), topic.get("name"))):
                    topic["thread_id"] = saved_id
