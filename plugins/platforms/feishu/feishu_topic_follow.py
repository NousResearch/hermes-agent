"""Feishu topic follow: answer a group @mention inside a new topic, then keep answering in it.

Mixed into ``FeishuAdapter``. Two opt-in switches under ``platforms.feishu.extra`` (each
overridable on a ``group_rules`` entry):

- ``thread_reply``: a top-level group message that triggers the bot gets its reply in a new
  topic. A short in-thread acknowledgement opens the topic before the turn runs, so the first
  turn already belongs to the topic's session instead of the group's.
- ``thread_follow``: inside a topic, follow-ups need no @mention. A message that @mentions
  someone else (a colleague, another bot) is left to them. Who answers depends on the mode:

  - ``last_active`` (``true`` means this): the bot triggered last in the topic — the newest
    message either written by a bot or @mentioning one decides. A bot @mentioned mid-topic takes
    the topic over at once, even before it has replied, and a bot message @mentioning another bot
    hands the topic to it.
  - ``owner``: the bot that replied in the topic first keeps it; a bot @mentioned once in another
    bot's topic answers that message only.

Either way the decision is read from the topic itself, so it survives gateway restarts without a
state file, and every bot in the chat reaches the same answer from the same history.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Optional

from agent.i18n import t

logger = logging.getLogger(__name__)

# Messages scanned for the first bot reply; a topic with more human messages before any bot reply
# is treated as unowned.
_OWNER_SCAN_PAGE_SIZE = 50
_OWNER_CACHE_MAX = 4096
# Newest messages scanned for the last trigger; older un-@ chatter than this leaves the topic idle.
_LAST_ACTIVE_SCAN_PAGE_SIZE = 20

FOLLOW_LAST_ACTIVE = "last_active"
FOLLOW_OWNER = "owner"
_FOLLOW_MODES = {FOLLOW_LAST_ACTIVE, FOLLOW_OWNER}


def _is_mention_all(mention: Any) -> bool:
    return getattr(mention, "key", None) == "@_all" or getattr(mention, "id", None) == "all"


class FeishuTopicFollowMixin:
    """Topic opening and no-@ follow-ups for ``FeishuAdapter`` (needs its client/send helpers)."""

    def _init_topic_follow_state(self) -> None:
        # thread_id -> owned by this bot. Only settled answers are cached: a topic no bot has
        # answered yet is asked again, since this bot may still become its owner.
        self._topic_owner_cache: dict[str, bool] = {}

    def _topic_option(self, chat_id: str, key: str) -> bool:
        from plugins.platforms.feishu.adapter import _to_boolean

        extra = getattr(getattr(self, "config", None), "extra", None) or {}
        rules = extra.get("group_rules")
        rule = rules.get(chat_id) if isinstance(rules, dict) and chat_id else None
        if isinstance(rule, dict) and key in rule:
            return _to_boolean(rule[key])
        return _to_boolean(extra.get(key, False))

    def _topic_follow_mode(self, chat_id: str) -> Optional[str]:
        """``last_active`` / ``owner`` when ``thread_follow`` is on for the chat, else ``None``."""
        extra = getattr(getattr(self, "config", None), "extra", None) or {}
        rules = extra.get("group_rules")
        rule = rules.get(chat_id) if isinstance(rules, dict) and chat_id else None
        value = rule["thread_follow"] if isinstance(rule, dict) and "thread_follow" in rule else extra.get("thread_follow")
        mode = str(value).strip().lower() if isinstance(value, str) else None
        if mode in _FOLLOW_MODES:
            return mode
        return FOLLOW_LAST_ACTIVE if self._topic_option(chat_id, "thread_follow") else None

    def _remember_topic_owner(self, thread_id: str, owned: bool) -> None:
        cache = self._topic_owner_cache
        if thread_id not in cache and len(cache) >= _OWNER_CACHE_MAX:
            cache.pop(next(iter(cache)))
        cache[thread_id] = owned

    async def _open_reply_topic(self, chat_id: str, message_id: str) -> Optional[str]:
        """Open a topic on a group message with an in-thread acknowledgement; return its id.

        ``None`` (switch off, or Feishu refused) keeps the ordinary in-group reply.
        """
        if not self._topic_option(chat_id, "thread_reply") or not self._client:
            return None
        payload = json.dumps({"text": t("platform.feishu.topic.opened")}, ensure_ascii=False)
        body = self._build_reply_message_body(
            content=payload, msg_type="text", reply_in_thread=True, uuid_value=str(uuid.uuid4()),
        )
        request = self._build_reply_message_request(message_id, body)
        try:
            response = await self._run_blocking(self._client.im.v1.message.reply, request)
        except Exception as exc:
            logger.warning("[Feishu] Could not open a topic on %s: %s", message_id, exc, exc_info=True)
            return None
        thread_id = self._extract_response_field(response, "thread_id")
        if not thread_id:
            logger.warning(
                "[Feishu] Could not open a topic on %s: [%s] %s", message_id,
                getattr(response, "code", "?"), getattr(response, "msg", "no thread_id returned"),
            )
            return None
        self._remember_topic_owner(thread_id, True)
        return thread_id

    async def _topic_follow_admits(self, sender: Any, message: Any) -> bool:
        """Admit a topic message that was rejected only for lacking an @mention of this bot."""
        from plugins.platforms.feishu.adapter import _is_bot_sender

        chat_id = getattr(message, "chat_id", "") or ""
        thread_id = getattr(message, "thread_id", None)
        mode = self._topic_follow_mode(chat_id)
        if not thread_id or getattr(message, "chat_type", "p2p") == "p2p" or _is_bot_sender(sender) or not mode:
            return False
        # The mention gate and the group policy share one reject reason: re-check the policy.
        if not self._allow_group_message(getattr(sender, "sender_id", None), chat_id):
            return False
        # Any @mention left here is of someone else: an @ of this bot (or @all) was admitted.
        if getattr(message, "mentions", None):
            return False
        if mode == FOLLOW_OWNER:
            return await self._topic_owned_by_self(thread_id)
        return await self._topic_last_triggered_self(thread_id, getattr(message, "message_id", "") or "")

    async def _topic_last_triggered_self(self, thread_id: str, current_message_id: str) -> bool:
        """Whether this bot is the one triggered last in the topic (newest message first).

        Never cached: every bot @mention or reply can move the topic to another bot.
        """
        items = await self._list_topic_messages(thread_id, "ByCreateTimeDesc", _LAST_ACTIVE_SCAN_PAGE_SIZE)
        for item in items or []:
            if getattr(item, "message_id", None) == current_message_id:
                continue
            decided = self._trigger_decision(item)
            if decided is not None:
                return decided
        return False

    def _trigger_decision(self, item: Any) -> Optional[bool]:
        """True/False when ``item`` settles who was triggered last; ``None`` to look further back.

        - @mentions this bot (from a person or a peer bot handing over) -> this bot.
        - written by a bot -> that bot, unless it @mentions others (a handover to them).
        - a person @mentioning others -> them; this bot cannot tell a colleague from a bot,
          so the topic goes quiet until a bot answers or is @mentioned again.
        - a person without @mentions (or with only @all) -> look further back.
        """
        mentions = [m for m in getattr(item, "mentions", None) or [] if not _is_mention_all(m)]
        if any(self._history_mention_is_self(m) for m in mentions):
            return True
        sender = getattr(item, "sender", None)
        if getattr(sender, "sender_type", "") == "app":
            # GET/list report sender.id = app_id for bot messages.
            return not mentions and str(getattr(sender, "id", "") or "") == self._app_id
        return False if mentions else None

    def _history_mention_is_self(self, mention: Any) -> bool:
        # list/GET mentions carry ``id`` as a plain app-scoped open_id string; events nest it.
        mention_id = getattr(mention, "id", None)
        open_id = mention_id if isinstance(mention_id, str) else getattr(mention_id, "open_id", None)
        return self._bot_identity().matches(
            open_id=(open_id or "").strip(), user_id="", name=(getattr(mention, "name", None) or "").strip(),
        )

    async def _topic_owned_by_self(self, thread_id: str) -> bool:
        cached = self._topic_owner_cache.get(thread_id)
        if cached is not None:
            return cached
        owner = await self._first_bot_sender_in_topic(thread_id)
        if owner is None:
            return False
        owned = owner == self._app_id
        self._remember_topic_owner(thread_id, owned)
        return owned

    async def _first_bot_sender_in_topic(self, thread_id: str) -> Optional[str]:
        """App id of the first bot-authored message in a topic; ``None`` if none or unreadable."""
        for item in await self._list_topic_messages(thread_id, "ByCreateTimeAsc", _OWNER_SCAN_PAGE_SIZE) or []:
            sender = getattr(item, "sender", None)
            # GET/list report sender.id = app_id for bot messages; peer bots differ only there.
            if getattr(sender, "sender_type", "") == "app":
                return str(getattr(sender, "id", "") or "")
        return None

    async def _list_topic_messages(self, thread_id: str, sort_type: str, page_size: int) -> Optional[list]:
        """One page of a topic's messages; ``None`` if it cannot be read."""
        if not self._client:
            return None
        from lark_oapi.api.im.v1 import ListMessageRequest

        request = (
            ListMessageRequest.builder().container_id_type("thread").container_id(thread_id)
            .sort_type(sort_type).page_size(page_size).build()
        )
        try:
            response = await self._run_blocking(self._client.im.v1.message.list, request)
        except Exception as exc:
            logger.warning("[Feishu] Could not read topic %s: %s", thread_id, exc, exc_info=True)
            return None
        if not self._response_succeeded(response):
            logger.warning(
                "[Feishu] Could not read topic %s: [%s] %s", thread_id,
                getattr(response, "code", "?"), getattr(response, "msg", ""),
            )
            return None
        return list(getattr(getattr(response, "data", None), "items", None) or [])

    async def _fetch_last_message_in_thread(self, thread_id: str) -> Optional[str]:
        """Fetch the last message_id in a thread for reply-based routing."""
        if not self._client or not thread_id:
            return None
        try:
            from lark_oapi.api.im.v1 import ListMessageRequest
            request = ListMessageRequest.builder().container_id_type("thread").container_id(thread_id).page_size(1).build()
            response = await self._run_blocking(self._client.im.v1.message.list, request)
            if self._response_succeeded(response):
                items = getattr(getattr(response, "data", None), "items", None)
                if items and len(items) > 0:
                    return getattr(items[0], "message_id", None)
        except Exception as exc:
            logger.debug("[Feishu] Failed to fetch last message in thread %s: %s", thread_id, exc)
        return None
