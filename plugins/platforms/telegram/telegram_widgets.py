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

from collections.abc import Mapping
from typing import Any, Optional

UPDATE_TYPE = "widget_query"
EVENT_TYPE = "widget_query"
# Ids and the 1-64 byte query data as text; the bound the other Telegram events put on ids.
_TEXT_LIMIT = 128


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


def normalize_event(query: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    """``widget_query`` → the ``widget_query`` platform event: ``query_id``, ``chat_id``, ``message_id`` and
    the ``data`` the widget sent, all text; ``None`` when an identity is missing or malformed."""
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
            "data": data[:_TEXT_LIMIT]},
    }


def _given(**params: Any) -> dict[str, Any]:
    """Only the parameters the caller set: an omitted data packet leaves Telegram's unchanged, an empty
    string removes it."""
    return {name: value for name, value in params.items() if value is not None}


async def send_widget(
        bot: Any, chat_id: Any, html: str, *, chat_data: Optional[str] = None,
        public_data: Optional[str] = None, caption: Optional[str] = None, allow_queries: bool = True,
        disable_notification: bool = False, thread_id: Optional[str] = None) -> Any:
    """``sendWidget``; returns the raw message Telegram sends back."""
    params: dict[str, Any] = {
        "chat_id": chat_id, "html": html, "allow_queries": allow_queries,
        "disable_notification": disable_notification}
    params.update(_given(
        chat_data=chat_data, public_data=public_data, caption=caption,
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
