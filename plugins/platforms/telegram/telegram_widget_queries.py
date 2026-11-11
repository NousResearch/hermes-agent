"""Widget requests are bounded user input, never gateway commands or payment confirmations."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
from plugins.platforms.telegram.telegram_widget_store import WidgetStore


def widget_store(adapter) -> WidgetStore:
    # Captured at adapter construction under its profile scope, not the current global home.
    token = str(getattr(adapter.config, "token", "") or "")
    bot_id = token.split(":", 1)[0]
    if not bot_id.isdecimal():
        raise ValueError("Widget state requires an identified Telegram bot")
    return WidgetStore(adapter._update_receipt_dir, bot_id)


def _query(update):
    for field in ("widget_query", "shared_widget_query"):
        value = getattr(update, field, None)
        if value is None:
            value = (getattr(update, "api_kwargs", None) or {}).get(field)
        if value is not None:
            if not isinstance(value, dict):
                value = value.to_dict()
            return field, value
    return None, None


async def handle_widget_query(adapter, update, context=None) -> None:
    field, query = _query(update)
    if not isinstance(query, dict):
        return
    query_id = query.get("id")
    if not isinstance(query_id, str) or not query_id or len(query_id) > 256:
        return
    adapter._accept_update()
    try:
        actor = query.get("from") or {}
        user_id = actor.get("id")
        data = query.get("data")
        if type(user_id) is not int or user_id <= 0 or not isinstance(data, str) or not 1 <= len(data.encode("utf-8")) <= 64:
            raise ValueError("Invalid widget request")
        store = widget_store(adapter)
        chat = query.get("chat")
        if field == "widget_query":
            if not isinstance(chat, dict) or chat.get("type") != "private" or type(query.get("message_id")) is not int:
                raise ValueError("Inline and group widgets are not available on Telegram production")
            record = await asyncio.to_thread(store.message, str(chat.get("id")), str(query["message_id"]))
        else:
            widget_id = query.get("widget_id")
            if not isinstance(widget_id, str) or not widget_id:
                raise ValueError("Missing shared widget identity")
            record = await asyncio.to_thread(store.shared, widget_id)
        if not record or str(user_id) != record.get("user_id") or not record.get("allow_queries"):
            raise ValueError("This widget is not available to this user")
        if query.get("widget_id") and str(query["widget_id"]) != record.get("widget_id"):
            raise ValueError("Widget identity mismatch")
        if not adapter._is_callback_user_authorized(
                str(user_id), chat_id=record["chat_id"], chat_type="private", thread_id=record.get("thread_id")):
            raise ValueError("Not authorized")
        session_store = getattr(adapter, "_session_store", None)
        current = await asyncio.to_thread(session_store.peek_session_id, record["session_key"]) if session_store else None
        if current != record["session_id"]:
            raise ValueError("This widget's conversation has ended; create a new widget")
        # Refresh never starts an LLM turn: Telegram cannot distinguish timers from taps.
        if data == "refresh":
            reply = json.dumps({"public_data": record.get("public_data"), "chat_data": record.get("chat_data")}, ensure_ascii=False)
            if len(reply.encode()) > 1024:
                reply = '{"ok":true,"state":"available via TelegramWidget.chatData/publicData"}'
            await adapter.answer_widget_query(query_id, reply)
            return
        actions = record.get("actions") or {}
        intent = actions.get(data)
        if not isinstance(intent, str) or not intent:
            raise ValueError("Unknown widget action")
        # Receipts are owned by TelegramApplication. A short per-widget cooldown also prevents
        # a visible widget's background request loop from continually launching model turns.
        now = time.monotonic()
        recent = getattr(adapter, "_widget_action_times", None)
        if recent is None:
            adapter._widget_action_times = recent = {}
        key = (record["chat_id"], record["message_id"], data)
        if now - recent.get(key, float("-inf")) < 5:
            await adapter.answer_widget_query(query_id, '{"ok":false,"error":"Please wait before repeating this action"}')
            return
        from gateway.platforms.helpers import bounded_put
        bounded_put(recent, key, now, 4096)
        kind = query.get("type") or {"type": "data"}
        if not isinstance(kind, dict) or kind.get("type") not in {"data", "sign_data", "sent_transaction"}:
            raise ValueError("Unsupported widget query type")
        # A wallet notification is evidence submitted by the sender, not proof of settlement
        # or a verified signature. No balance, entitlement, or authority changes here.
        wallet_note = ""
        if kind.get("type") in {"sign_data", "sent_transaction"}:
            wallet_note = "\nWallet event (signature/settlement NOT verified): " + json.dumps(kind, ensure_ascii=False)[:4096]
        source = SessionSource(
            platform=Platform.TELEGRAM, chat_id=record["chat_id"], chat_type="dm",
            user_id=str(user_id), user_name=str(actor.get("first_name") or ""),
            thread_id=record.get("thread_id"), message_id=record["message_id"], profile=record.get("profile"))
        event = MessageEvent(
            text=f"Telegram widget action: {intent}{wallet_note}", message_type=MessageType.TEXT,
            source=source, user_id=str(user_id), user_name=source.user_name,
            message_id=f"widget:{query_id}", reply_anchor_override=record["message_id"],
            platform_update_id=getattr(update, "update_id", None), allow_gateway_control=False,
            timestamp=datetime.now(timezone.utc),
            metadata={"telegram_widget_message_id": record["message_id"], "telegram_widget_action": data})
        # Ack before model/tool work, inside Telegram's 15-second answer window.
        if not await adapter.answer_widget_query(query_id, '{"ok":true,"status":"accepted"}'):
            return
        if session_store.peek_session_id(record["session_key"]) != record["session_id"]:
            return
        await adapter.handle_message(event)
    except (ValueError, TypeError, KeyError) as exc:
        await adapter.answer_widget_query(query_id, json.dumps({"ok": False, "error": str(exc)[:300]}))
