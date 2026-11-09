"""Telegram private-chat topic send ladder: what to do after Bot API answers "message thread not found".

Bot private-chat topics are addressed by ``message_thread_id`` (Bot API: "for forum supergroups and private
chats of bots with forum topic mode enabled"). When Telegram rejects that id (and the same id as
``direct_messages_topic_id``, see below) the topic is gone: the user
deleted it, or forum mode was turned off (#95517). Every retry removes exactly the constraint that failed
(#136808): the dead route goes, and so does the reply anchor, which points into the dead topic and would
otherwise steer the retry back there. Pruning the dead topic's binding (the chat's last one disables topic
mode, ``hermes_state_telegram.delete_telegram_topic_binding``) stops inbound recovery steering into it.
"""

from __future__ import annotations

from typing import Any


def private_chat(chat_id: Any) -> bool:
    try:
        return int(str(chat_id).strip()) > 0
    except (TypeError, ValueError):
        return False


def topic_gone(chat_id: Any, error: Exception) -> bool:
    """Telegram's wordings for a private topic that no longer takes messages (origin/main's media retry matched the
    same set); elsewhere only "thread not found". A parse, permission or flood error says nothing about the topic."""
    markers = ("thread not found", "topic_deleted", "topic not found", "direct_messages_topic") if private_chat(chat_id) else ("thread not found",)
    return any(m in str(error).lower() for m in markers)


def dead_topic_may_reroute(chat_id: Any, metadata: Any) -> bool:
    """A dead private topic may be retried by native id, then the chat root, unless the send targets a named topic
    (``telegram_dm_topic_created_for_send``): that fails closed for text and media alike so DeliveryRouter recreates it."""
    return private_chat(chat_id) and not (metadata or {}).get("telegram_dm_topic_created_for_send")


def direct_topic_retry_kwargs(chat_id: Any, thread_id: Any, metadata: Any, tried: dict[str, Any]) -> dict[str, Any] | None:
    """The same topic by ``direct_messages_topic_id``, tried once before the topic is declared gone: some
    native DM-topic chats reject ``message_thread_id`` for an id they accept there (#137568)."""
    if tried.get("direct_messages_topic_id") is not None or not private_chat(chat_id):
        return None
    meta = metadata or {}
    topic = meta.get("direct_messages_topic_id") or meta.get("telegram_direct_messages_topic_id") or tried.get("message_thread_id") or thread_id
    if topic is None or not str(topic).isdigit() or str(topic) == "1":  # "1": General (TelegramAdapter._GENERAL_TOPIC_THREAD_ID)
        return None
    return {"message_thread_id": None, "direct_messages_topic_id": int(topic)}


def dead_topic_retry_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """A control send's kwargs without the topic route, and without the anchor in a private chat."""
    retry = {k: v for k, v in kwargs.items() if k not in ("message_thread_id", "direct_messages_topic_id")}
    if private_chat(kwargs.get("chat_id")):
        retry.pop("reply_to_message_id", None)
    return retry
