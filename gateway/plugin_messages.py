"""Where ``PluginContext.post_message`` lands: the conversation the calling plugin code works for.

The route is never chosen by the plugin. It is the session whose key is bound in the calling context
(a tool or hook in its turn, its slash command, a hook fired for it), resolved here on the gateway loop:

* a messaging chat posts through the adapter that answers it (``adapter.send``) and edits through
  ``adapter.edit_message``; an adapter without edit support, or a refused edit, gets a fresh send and
  the handle follows the new message;
* a local CLI/TUI/Desktop session has no transport of its own (``LocalSessionAdapter.send`` is a no-op),
  so the text goes to its viewers as a keyed ``notification.show``; an edit re-publishes the same key,
  which Desktop and the Ink TUI replace in place (the classic CLI queues each as a line for its next
  turn end).
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from gateway.config import Platform

logger = logging.getLogger(__name__)


@dataclass
class PostTarget:
    """A resolved conversation: a messaging ``adapter`` + ``chat_id``, or a local session's ``live``."""
    platform: str
    chat_id: str
    adapter: Any = None
    metadata: Optional[dict] = None
    live: Any = None
    notice_key: str = ""


def _err(code: str, detail: str = "") -> dict:
    return {"ok": False, "error": code, **({"detail": detail} if detail else {})}


def _fit(adapter: Any, chat_id: str, text: str) -> str:
    """Clamp *text* to the adapter's per-chat message limit (ellipsis), never split it."""
    try:
        limit = int(adapter.max_message_length_for_chat(chat_id))
        measure = adapter.message_len_fn_for_chat(chat_id)
    except (AttributeError, TypeError, ValueError):  # an adapter not built on BasePlatformAdapter
        limit, measure = 4096, len
    if limit <= 0 or measure(text) <= limit:
        return text
    cut = text
    while cut and measure(cut + "…") > limit:
        cut = cut[: -max(1, measure(cut + "…") - limit)]
    return cut + "…"


def _supports_edit(adapter: Any) -> bool:
    from gateway.platforms.base import BasePlatformAdapter
    if getattr(adapter, "SUPPORTS_MESSAGE_EDITING", True) is False:
        return False
    return getattr(type(adapter), "edit_message", None) is not BasePlatformAdapter.edit_message


# Every source field the gateway binds for a call (``GatewayRunner._set_session_env``), so a rebuilt
# source keys exactly like the live one (Slack ``scope_id``, ``user_id_alt`` group keys, ...).
_ROUTE_VARS = {field: "HERMES_SESSION_" + field.upper() for field in (
    "platform", "chat_id", "chat_type", "chat_name", "thread_id", "user_id", "user_id_alt", "user_name",
    "scope_id", "parent_chat_id", "message_id", "profile")}


def bound_route() -> dict:
    """The messaging route the gateway bound in THIS context (read on the caller's thread: the
    gateway loop runs the send in its own context)."""
    from gateway.session_context import get_session_env
    return {field: get_session_env(var, "") for field, var in _ROUTE_VARS.items()}


def _bound_source(runner, route: dict, *, plugin_home, plugin_id: str):
    """A source for a messaging chat whose session has no store entry yet (a plugin command that is
    the chat's first message), rebuilt from the route the gateway bound for this call and given the
    receiving bot's provenance by the same canonicalizer a plugin ``origin`` goes through."""
    if not route.get("platform") or not route.get("chat_id") or route["platform"] == Platform.LOCAL.value:
        return None
    return runner._plugin_origin_source({k: v for k, v in route.items() if v}, plugin_home=plugin_home,
                                        plugin_id=plugin_id)


def resolve(runner, *, session_key: str, route: dict, plugin_home, plugin_id: str):
    """``(PostTarget, None)`` for the session *session_key* names, or ``(None, error dict)``."""
    entry = runner.session_store.lookup_by_session_key(session_key)
    if entry is not None and entry.origin is not None:
        from gateway.session_identity import replace_source
        source = replace_source(runner._restored_source(entry))
    else:
        source = _bound_source(runner, route, plugin_home=plugin_home, plugin_id=plugin_id)
        # The bound route must be the chat this key names (both are bound by the gateway together).
        if source is None or runner._session_key_for_source(source) != session_key:
            return None, _err("no_conversation", f"session {session_key!r} is not known to this gateway")
    if source.platform == Platform.LOCAL:
        live = runner._plugin_local_session(source, session_key, plugin_home=plugin_home, plugin_id=plugin_id)
        if live is None:
            return None, _err("no_conversation", "the local session is not live in this plugin's profile")
        key = f"plugin-message:{plugin_id}:{uuid.uuid4().hex[:12]}"
        return PostTarget(platform="local", chat_id=source.chat_id, live=live, notice_key=key), None
    if not runner._plugin_messaging_session_owned(session_key, plugin_home=plugin_home, plugin_id=plugin_id):
        return None, _err("no_conversation", "the session belongs to another profile")
    adapter = runner._delivery_adapter_for(source)
    if adapter is None:
        return None, _err("adapter_unavailable", f"no connected {source.platform.value} adapter answers this chat")
    try:
        metadata = runner._thread_metadata_for_source(source)
    except Exception:  # thread routing is optional: post to the chat itself
        logger.debug("plugin_message: no thread metadata for %s", source.platform.value, exc_info=True)
        metadata = None
    return PostTarget(platform=source.platform.value, chat_id=str(source.chat_id), adapter=adapter,
                      metadata=metadata), None


def _publish_local(target: PostTarget, text: str) -> dict:
    delivered = target.live.event_stream.publish(target.chat_id, {
        "text": text, "level": "info", "kind": "sticky", "key": target.notice_key, "id": target.notice_key,
    }, event_type="notification.show")
    return {"ok": True, "message_id": target.notice_key, "delivered": bool(delivered)}


async def _send(target: PostTarget, text: str) -> dict:
    # Never the turn's answer: a relay stream open for the reply must not be sealed with it.
    metadata = {**(target.metadata or {}), "_interim_send": True}
    result = await target.adapter.send(target.chat_id, _fit(target.adapter, target.chat_id, text),
                                       metadata=metadata)
    if not getattr(result, "success", False):
        return _err("send_failed", str(getattr(result, "error", "") or "")[:512])
    return {"ok": True, "message_id": str(result.message_id) if result.message_id is not None else None}


async def post(target: PostTarget, text: str, *, plugin_id: str) -> dict:
    """First message of a handle."""
    try:
        result = _publish_local(target, text) if target.live is not None else await _send(target, text)
    except Exception as exc:  # adapter boundary: the plugin gets a result, never the raise
        logger.warning("plugin_message plugin=%s op=post send raised", plugin_id, exc_info=True)
        result = _err("send_failed", str(exc)[:512])
    logger.info("plugin_message plugin=%s op=post platform=%s chat=%s ok=%s message_id=%s%s", plugin_id,
                target.platform, target.chat_id, result.get("ok"), result.get("message_id"),
                "" if result.get("ok") else f" error={result.get('error')}")
    return result


async def edit(target: PostTarget, message_id: Optional[str], text: str, *, plugin_id: str) -> dict:
    """Edit *message_id* in place; a surface that cannot edit gets a fresh message (``fallback``)."""
    result: dict
    try:
        if target.live is not None:
            result = {**_publish_local(target, text), "edited": True}
        elif message_id and _supports_edit(target.adapter):
            try:
                sent = await target.adapter.edit_message(target.chat_id, message_id,
                                                         _fit(target.adapter, target.chat_id, text))
                edited, edit_error = bool(getattr(sent, "success", False)), str(getattr(sent, "error", "") or "")
            except Exception as exc:  # a raised edit falls back like a refused one
                logger.debug("plugin_message plugin=%s edit_message raised", plugin_id, exc_info=True)
                sent, edited, edit_error = None, False, str(exc) or type(exc).__name__
            if edited:
                result = {"ok": True, "message_id": str(sent.message_id or message_id), "edited": True}
            else:
                result = {**await _send(target, text), "edited": False, "fallback": True,
                          "edit_error": edit_error[:256]}
        else:
            result = {**await _send(target, text), "edited": False, "fallback": True}
    except Exception as exc:  # adapter boundary, as in post()
        logger.warning("plugin_message plugin=%s op=edit raised", plugin_id, exc_info=True)
        result = _err("edit_failed", str(exc)[:512])
    logger.info("plugin_message plugin=%s op=edit platform=%s chat=%s ok=%s edited=%s message_id=%s%s",
                plugin_id, target.platform, target.chat_id, result.get("ok"), result.get("edited", False),
                result.get("message_id"), "" if result.get("ok") else f" error={result.get('error')}")
    return result
