"""Telegram HTML widgets (Bot API 10.4) for plugins.

python-telegram-bot 22.8 models Bot API 10.0: ``Update.ALL_TYPES`` has no ``widget_query``, an update
carrying one keeps it in ``Update.api_kwargs``, and ``sendWidget`` / ``editMessageWidgetData`` /
``answerWidgetQuery`` exist only as raw ``Bot.do_api_request`` calls (the way the adapter already sends
``sendRichMessage``). This module keeps that knowledge in one place: the update types the adapter asks
Telegram for, the ``widget_query`` platform event a tap becomes, and the three Bot API calls behind the
``ctx.platform_actions`` widget verbs. A PTB that models the type works unchanged: its own list is not
extended twice and a native ``widget_query`` object is read through ``to_dict()``.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any, Optional

UPDATE_TYPE = "widget_query"
EVENT_TYPE = "widget_query"
# Ids and the 1-64 byte query data as text; the bound the other Telegram events put on ids.
_TEXT_LIMIT = 128
# Widget messages whose topic one adapter remembers; the oldest is forgotten first.
_THREAD_MEMORY_LIMIT = 4096


def allowed_updates(update_cls: Any) -> list[str]:
    """``Update.ALL_TYPES`` plus ``widget_query`` as a new list; a PTB that lists the type itself is left alone."""
    kinds = list(getattr(update_cls, "ALL_TYPES", None) or ())
    if UPDATE_TYPE not in kinds:
        kinds.append(UPDATE_TYPE)
    return kinds


def widget_query_of(update: Any) -> Optional[dict[str, Any]]:
    """The update's ``widget_query`` as a plain dict, or ``None`` for any other update."""
    extra = getattr(update, "api_kwargs", None)
    query = extra.get(UPDATE_TYPE) if isinstance(extra, Mapping) else None
    if query is None:
        to_dict = getattr(getattr(update, UPDATE_TYPE, None), "to_dict", None)
        query = to_dict() if callable(to_dict) else None
    return query if isinstance(query, dict) else None


def _text(value: Any) -> Optional[str]:
    """An id or the query data as bounded text; bools and non-scalars are never ids."""
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return None
    text = str(value).strip()
    return text[:_TEXT_LIMIT] if text else None


def identities(query: Mapping[str, Any]) -> dict[str, Any]:
    """Who tapped and where: the sender and the chat of the widget message. An inline-message widget
    carries neither ``chat`` nor ``message_id``, so the gateway has no chat to authorize it in."""
    sender = query.get("from")
    chat = query.get("chat")
    sender = sender if isinstance(sender, Mapping) else {}
    chat = chat if isinstance(chat, Mapping) else {}
    user_name = sender.get("username") or sender.get("first_name")
    return {
        "user_id": _text(sender.get("id")),
        "user_name": user_name if isinstance(user_name, str) and user_name.strip() else None,
        "chat_id": _text(chat.get("id")),
        "chat_type": chat.get("type"),
        "is_forum": chat.get("is_forum") is True,
        "message_id": _text(query.get("message_id")),
    }


class WidgetThreads:
    """The topic each widget message was sent into, as Telegram reported it in ``sendWidget``'s reply.

    A ``widget_query`` carries no ``message_thread_id``, while the gateway keys a topic of a private chat (or of a
    forum) as its own session; this is how a tap names the topic again. The memory is per adapter and in process,
    bounded to the most recent widgets: a tap on a widget sent before a restart, or by another process, has no
    known topic."""

    def __init__(self, limit: int = _THREAD_MEMORY_LIMIT) -> None:
        self._limit = limit
        self._threads: OrderedDict[tuple[str, str], str] = OrderedDict()

    def remember(self, sent: Any) -> None:
        """Record the topic of a sent widget message; a reply outside a topic or of an unexpected shape is
        ignored (the rule ``message_edited`` uses: ``message_thread_id`` only on a topic message)."""
        if not isinstance(sent, Mapping) or sent.get("is_topic_message") is not True:
            return
        chat = sent.get("chat")
        chat_id = _text(chat.get("id")) if isinstance(chat, Mapping) else None
        message_id = _text(sent.get("message_id"))
        thread_id = _text(sent.get("message_thread_id"))
        if not chat_id or not message_id or not thread_id:
            return
        self._threads[(chat_id, message_id)] = thread_id
        self._threads.move_to_end((chat_id, message_id))
        while len(self._threads) > self._limit:
            self._threads.popitem(last=False)

    def thread_of(self, query: Mapping[str, Any]) -> Optional[str]:
        """The topic of the widget message a query came from, or ``None`` when it is not known."""
        ids = identities(query)
        return self._threads.get((ids["chat_id"], ids["message_id"]))


def _query_type(value: Any) -> Optional[dict[str, Any]]:
    """The query's ``WidgetQueryType`` as a JSON copy: ``type`` is ``data``, ``sign_data`` or ``sent_transaction``,
    with the TON wallet fields Telegram sends for them, uncut (a signature is useless shortened); ``None`` for
    any other shape."""
    if not isinstance(value, Mapping) or not isinstance(value.get("type"), str):
        return None
    try:
        return json.loads(json.dumps(dict(value)))
    except (TypeError, ValueError):
        return None


def normalize_event(query: Mapping[str, Any], thread_id: Optional[str] = None) -> Optional[dict[str, Any]]:
    """``widget_query`` → the ``widget_query`` platform event: ``query_id``, ``chat_id``, ``message_id`` and
    the ``data`` the widget sent, all text, plus ``thread_id``, the widget's topic when the adapter knows it
    (see ``WidgetThreads``), the query's ``type`` object and the ``widget_id`` of a shared widget; ``None``
    when an identity is missing or malformed."""
    ids = identities(query)
    query_id = _text(query.get("id"))
    data = query.get("data")
    if not query_id or not ids["chat_id"] or not ids["message_id"] or not isinstance(data, str):
        return None
    return {
        "platform": "telegram",
        "event_type": EVENT_TYPE,
        "payload": {
            "query_id": query_id, "chat_id": ids["chat_id"], "message_id": ids["message_id"],
            "data": data[:_TEXT_LIMIT], "thread_id": thread_id,
            "type": _query_type(query.get("type")), "widget_id": _text(query.get("widget_id"))},
    }


def _given(**params: Any) -> dict[str, Any]:
    """Only the parameters the caller set: an omitted data packet leaves Telegram's unchanged, an empty
    string removes it."""
    return {name: value for name, value in params.items() if value is not None}


async def send_widget(
        bot: Any, chat_id: Any, html: str, *, chat_data: Optional[str] = None,
        public_data: Optional[str] = None, caption: Optional[str] = None, allow_queries: bool = True,
        disable_notification: bool = False, thread_id: Optional[str] = None, width: Optional[int] = None,
        height: Optional[int] = None) -> Any:
    """``sendWidget``; returns the raw message Telegram sends back. ``width`` / ``height`` are logical pixels
    (the app clamps the width to 200 up to the bubble's width and the height to 50-600); an omitted width fills
    the bubble and an omitted height follows the content."""
    params: dict[str, Any] = {
        "chat_id": chat_id, "html": html, "allow_queries": allow_queries,
        "disable_notification": disable_notification}
    params.update(_given(
        chat_data=chat_data, public_data=public_data, caption=caption, width=width, height=height,
        message_thread_id=None if thread_id is None else int(thread_id)))
    return await bot.do_api_request("sendWidget", api_kwargs=params)


async def edit_widget_data(
        bot: Any, chat_id: Any, message_id: int, *, chat_data: Optional[str] = None,
        public_data: Optional[str] = None) -> bool:
    """``editMessageWidgetData``: True when the data changed, False when Telegram already had the same
    data ("message is not modified" is Telegram's no-op, not a failure). Other errors propagate."""
    params = {"chat_id": chat_id, "message_id": message_id, **_given(chat_data=chat_data, public_data=public_data)}
    try:
        await bot.do_api_request("editMessageWidgetData", api_kwargs=params)
    except Exception as exc:
        if "not modified" in str(exc).lower():
            return False
        raise
    return True


async def answer_widget_query(bot: Any, query_id: str, data: Optional[str] = None) -> None:
    """``answerWidgetQuery``: Telegram waits 15 seconds for the answer, then shows the widget a timeout."""
    await bot.do_api_request("answerWidgetQuery", api_kwargs={"widget_query_id": query_id, **_given(data=data)})
