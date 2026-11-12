"""Session-owned Telegram UI operations. Routing/identity are supplied by the gateway, not the model."""

from __future__ import annotations

import asyncio
import json

from plugins.platforms.telegram.telegram_widget_queries import widget_store
from plugins.platforms.telegram.telegram_widget_store import WidgetStore


def _result(result):
    raw = result.raw_response if isinstance(result.raw_response, dict) else {}
    return {"success": result.success, "message_id": result.message_id, "error": result.error,
            "retryable": result.retryable, **({"ambiguous": True} if raw.get("ambiguous") else {})}


def _actions(value):
    if value is None:
        return {}
    if not isinstance(value, dict) or len(value) > 32:
        raise ValueError("actions must be an object with at most 32 request-data keys")
    for key, intent in value.items():
        if not isinstance(key, str) or not 1 <= len(key.encode()) <= 64 or key == "refresh":
            raise ValueError("Action keys must be 1-64 UTF-8 bytes; refresh is reserved for cached state")
        if not isinstance(intent, str) or not intent.strip() or len(intent) > 1024:
            raise ValueError("Each action needs a nonempty intent of at most 1024 characters")
    return dict(value)


async def execute_ui_action(adapter, owner: dict, args: dict) -> dict:
    """Called on the owning gateway loop after the live turn's identity has been checked."""
    if not isinstance(args, dict):
        raise ValueError("Telegram UI arguments must be an object")
    allowed = {"action", "widget", "actions", "message_id", "widget_id", "changes", "public_data",
               "cache_time", "content", "quote_rich_texts", "quote_position", "name"}
    if set(args) - allowed:
        raise ValueError("Unknown Telegram UI arguments; destinations and credentials cannot be selected")
    action = args.get("action")
    chat_id = owner["chat_id"]
    metadata = {"chat_type": owner["chat_type"], "thread_id": owner.get("thread_id"),
                "hermes_profile": owner.get("profile")}
    if action == "send_rich_reply":
        if str(args.get("message_id") or "") not in owner.get("quote_message_ids", set()):
            raise ValueError("Rich replies can quote only the current message or its replied-to message")
        from .telegram_rich_quotes import build_rich_reply_parameters, send_rich_reply
        params = build_rich_reply_parameters(args.get("message_id"), args.get("quote_rich_texts"), args.get("quote_position"))
        return _result(await send_rich_reply(adapter, chat_id, args.get("content"), params, metadata))
    if action == "topic_link":
        thread = owner.get("thread_id")
        if owner["chat_type"] not in {"private", "dm"} or not thread or str(thread) == "1":
            raise ValueError("A private-chat topic is required")
        bot = await adapter._bot.get_me()
        username = getattr(bot, "username", None)
        if not username:
            raise ValueError("This bot has no public username")
        return {"success": True, "url": f"https://t.me/{username}/{int(thread)}",
                "visibility": "Opens only for the user whose private chat contains this topic"}
    if action in {"chat_info", "gift_info", "share_theme_backdrops"}:
        methods = {"chat_info": ("getChat", {"chat_id": int(chat_id)}),
                   "share_theme_backdrops": ("getChatShareThemeBackdrops", {})}
        if action == "gift_info":
            name = args.get("name")
            if not isinstance(name, str) or not name or len(name) > 256 or not name.isascii() or not all(c.isalnum() or c == "-" for c in name):
                raise ValueError("name must be the unique Telegram gift slug")
            method, payload = "getUniqueGift", {"name": name}
        else:
            method, payload = methods[action]
        from .adapter import _await_with_thread_deadline
        info = await _await_with_thread_deadline(adapter._bot.do_api_request(method, api_kwargs=payload),
                                               timeout=15, label="telegram-ui-info", dump_on_blocked_loop=False)
        if not isinstance(info, dict):
            raise ValueError("Telegram returned an invalid information response")
        if action == "chat_info":
            info = {key: info[key] for key in ("id", "type", "title", "username", "first_name", "last_name",
                    "is_forum", "is_direct_messages", "ton_wallet_card", "share_theme", "permissions") if key in info}
        return {"success": True, "data": info}
    store = widget_store(adapter)
    if action in {"send_widget", "send_shared_widget"}:
        if not isinstance(args.get("widget"), dict):
            raise ValueError("widget must be an object")
        spec = dict(args["widget"])
        if action == "send_widget":
            actions = _actions(args.get("actions"))
            if actions and spec.get("allow_queries") is False:
                raise ValueError("actions require allow_queries")
            if actions:
                spec["allow_queries"] = True
            result = await adapter.send_widget(chat_id, spec, metadata)
            inherited = {}
        else:
            shared = await asyncio.to_thread(store.shared, str(spec.get("widget_id", "")))
            inherited = WidgetStore.require_owner(shared, owner)
            actions = inherited.get("actions") or {}
            result = await adapter.send_shared_widget(chat_id, spec, metadata)
        output = _result(result)
        if not result.success:
            return output
        raw = result.raw_response if isinstance(result.raw_response, dict) else {}
        message = raw.get("message") or raw
        widget = message.get("widget") or raw.get("widget") or {}
        if not isinstance(widget, dict):
            widget = widget.to_dict()
        record = {**owner, "message_id": str(result.message_id),
                  "widget_id": str(widget.get("id") or spec.get("widget_id") or ""),
                  "actions": actions, "allow_queries": spec.get("allow_queries", inherited.get("allow_queries", False)),
                  "public_data": widget.get("public_data", spec.get("public_data", inherited.get("public_data"))),
                  "chat_data": widget.get("chat_data", spec.get("chat_data")), "caption": spec.get("caption", "")}
        try:
            await asyncio.to_thread(store.save_message, record)
        except Exception:
            # The send already landed. Never ask the agent to retry and duplicate it.
            return {**output, "success": False, "retryable": False, "delivered": True,
                    "error": "Widget delivered, but ownership could not be saved; interactions are disabled"}
        return {**output, "widget_id": record["widget_id"] or None}
    if action == "edit_widget":
        record = await asyncio.to_thread(store.message, chat_id, str(args.get("message_id", "")))
        record = WidgetStore.require_owner(record, owner)
        changes = args.get("changes")
        result = await adapter.edit_widget_data(chat_id, record["message_id"], changes)
        if result.success:
            await asyncio.to_thread(store.save_message, {**record, **changes})
        return _result(result)
    if action == "edit_shared_widget":
        widget_id = str(args.get("widget_id", ""))
        record = await asyncio.to_thread(store.shared, widget_id)
        record = WidgetStore.require_owner(record, owner)
        result = await adapter.edit_shared_widget_data(widget_id, args.get("public_data"), args.get("cache_time"))
        if result.success:
            await asyncio.to_thread(store.save_message, {**record, "public_data": args["public_data"]})
        return _result(result)
    if action == "widget_state":
        record = await asyncio.to_thread(store.message, chat_id, str(args.get("message_id", "")))
        record = WidgetStore.require_owner(record, owner)
        return {"success": True, "widget": {key: record.get(key) for key in (
            "message_id", "widget_id", "public_data", "chat_data", "caption", "actions")}}
    raise ValueError("Unknown Telegram UI action")
