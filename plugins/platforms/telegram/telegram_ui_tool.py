"""Agent UI tool bound to one live Telegram turn; no caller-selected destination or token."""

from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from contextvars import ContextVar

_BOUND = ContextVar("telegram_ui_turn", default=None)


@contextmanager
def bind_telegram_ui(runner, source, session_id, session_key, generation):
    from gateway.config import Platform
    binding = None
    if source.platform == Platform.TELEGRAM and session_key and generation is not None:
        binding = {"runner": runner, "source": source, "session_id": session_id,
                   "session_key": session_key, "generation": generation, "live": True}
    token = _BOUND.set(binding)
    try:
        yield
    finally:
        if binding is not None:
            binding["live"] = False
        _BOUND.reset(token)


async def telegram_ui(args, **kwargs):
    binding = _BOUND.get()
    if not binding:
        return json.dumps({"success": False, "error": "Telegram UI requires an active local Telegram conversation"})
    runner, source = binding["runner"], binding["source"]
    from tools.send_message_tool import _dispatch_on_gateway_loop

    async def dispatch():
        from gateway.session_context import get_session_env
        from .telegram_ui_actions import execute_ui_action
        if not binding["live"] or not runner._is_session_run_current(binding["session_key"], binding["generation"]):
            raise ValueError("The originating turn has ended")
        actor = get_session_env("HERMES_SESSION_ACTOR_ID")
        # A merged multi-user steer has no sole actor and must not inherit the first user's UI authority.
        if not actor or actor != str(source.user_id or ""):
            raise ValueError("Telegram UI needs one authenticated requester")
        adapter = runner._delivery_adapter_for(source)
        if adapter is None or not adapter._bot:
            raise ValueError("Telegram adapter is unavailable")
        if not adapter._is_callback_user_authorized(
                actor, chat_id=source.chat_id, chat_type=source.chat_type, thread_id=source.thread_id):
            raise ValueError("Requester is no longer authorized")
        from gateway.session_context import _SESSION_TURN_EVENT
        event = _SESSION_TURN_EVENT.get()
        quote_ids = {str(value) for value in (
            source.message_id, getattr(event, "message_id", None), getattr(event, "reply_to_message_id", None)) if value}
        owner = {"chat_id": str(source.chat_id), "chat_type": source.chat_type,
                 "quote_message_ids": sorted(quote_ids),
                 "thread_id": source.thread_id, "user_id": actor, "profile": source.profile,
                 "session_key": binding["session_key"], "session_id": binding["session_id"]}
        lock = getattr(adapter, "_telegram_ui_lock", None)
        if lock is None:
            adapter._telegram_ui_lock = lock = asyncio.Lock()
        async with lock:
            if not binding["live"] or not runner._is_session_run_current(binding["session_key"], binding["generation"]):
                raise ValueError("The originating turn has ended")
            return await execute_ui_action(adapter, owner, args)

    try:
        result = await _dispatch_on_gateway_loop(runner, dispatch, "Telegram UI dispatch failed")
        return json.dumps(result, ensure_ascii=False)
    except (ValueError, TypeError, KeyError) as exc:
        return json.dumps({"success": False, "error": str(exc), "retryable": False})
    except Exception as exc:
        from .adapter import _redact_telegram_error_text
        return json.dumps({"success": False, "error": _redact_telegram_error_text(exc), "retryable": False})


_DESCRIPTION = (
    "Create and update Telegram HTML widgets, quote rich messages, inspect Telegram metadata, or get a private topic link. "
    "Acts ONLY in this conversation as this bot; cannot choose another destination. Widgets require a private chat. "
    "Use a caption for unsupported clients. HTML is immutable, UTF-8 <=32768 bytes and zlib9 <=4096 bytes, "
    "inline CSS/JS only, TelegramWidget sandbox has no network or keyboard. State strings <=1024 UTF-8 bytes. "
    "TelegramWidget.request(action_key) dispatches the matching actions intent to Hermes; reserve refresh for "
    "cached state without starting a model turn. Do not put secrets into HTML/public_data. "
    "Wallet events do not prove settlement/signature validity. No payment or destructive administration operations.")

SCHEMA = {"name": "telegram_ui", "description": _DESCRIPTION, "parameters": {
    "type": "object", "additionalProperties": False, "required": ["action"], "properties": {
        "action": {"type": "string", "enum": ["send_widget", "send_shared_widget", "edit_widget", "edit_shared_widget",
                    "widget_state", "send_rich_reply", "topic_link", "chat_info", "gift_info", "share_theme_backdrops"]},
        "widget": {"type": "object", "description": "Send spec: html (new widget) or widget_id (reuse), caption, optional width/height, public_data/chat_data strings, allow_queries/is_shared booleans.",
                   "properties": {"html": {"type": "string"}, "widget_id": {"type": "string"}, "caption": {"type": "string"},
                       "width": {"type": "integer"}, "height": {"type": "integer"}, "public_data": {"type": "string"},
                       "chat_data": {"type": "string"}, "allow_queries": {"type": "boolean"}, "is_shared": {"type": "boolean"},
                       "show_caption_above_media": {"type": "boolean"}}, "additionalProperties": False},
        "actions": {"type": "object", "additionalProperties": {"type": "string"},
                    "description": "Map request data (1-64 UTF-8 bytes, except refresh) to the intent Hermes handles, e.g. {\"details\":\"Explain the current report\"}. At most32."},
        "message_id": {"type": "string", "description": "Widget to inspect/edit, or message being quoted, within this chat."},
        "widget_id": {"type": "string", "description": "Shared widget returned by a send in this conversation."},
        "changes": {"type": "object", "properties": {"chat_data": {"type": "string"}, "public_data": {"type": "string"}},
                    "additionalProperties": False, "description": "Edit state: omitted stays unchanged; empty string clears."},
        "public_data": {"type": "string"}, "cache_time": {"type": "integer", "minimum": 5, "maximum": 3600},
        "content": {"type": "string", "description": "Markdown body for send_rich_reply."},
        "quote_rich_texts": {"type": "array", "description": "Exact selected RichText nodes, original formatting retained; 1-32 blocks, total <=1024 characters.", "items": {}},
        "quote_position": {"type": "integer", "minimum": 0},
        "name": {"type": "string", "description": "Unique gift slug for gift_info."},
    }}}


def register_tools(ctx):
    from .tools import register_tools as _register
    _register(ctx)
