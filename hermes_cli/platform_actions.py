"""Capability-gated platform action facade for plugins (#64176, action half).

Every verb returns a structured result dict — ``{"ok": True, ...}`` on success, ``{"ok": False,
"error": <code>, "detail": <str>}`` on failure — and never raises into hook dispatch.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

ACTIONS_CONTRACT_VERSION = 1

CAPABILITY_ID = "gateway.platform_actions"


def _err(code: str, detail: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {"ok": False, "error": code}
    if detail:
        result["detail"] = detail
    return result


def _ok(**fields: Any) -> dict[str, Any]:
    return {"ok": True, **fields}


# -- per-platform verb implementations (adapter, *args) -> result ------------


async def _telegram_add_reaction(adapter, chat_id, message_id, emoji):
    if await adapter._set_reaction(chat_id, message_id, emoji):
        return _ok(action="add_reaction")
    return _err("action_failed", "telegram set_message_reaction failed")


async def _discord_add_reaction(adapter: Any, chat_id: str, message_id: str, emoji: str) -> dict[str, Any]:
    client = getattr(adapter, "_client", None)
    if client is None:
        return _err("adapter_disconnected", "discord client unavailable")
    try:
        channel_id = int(str(chat_id))
        msg_id = int(str(message_id))
    except (TypeError, ValueError):
        return _err("invalid_argument", "discord ids must be numeric")
    channel = client.get_channel(channel_id)
    if channel is None:
        channel = await client.fetch_channel(channel_id)
    message = await channel.fetch_message(msg_id)
    await message.add_reaction(emoji)
    return _ok(action="add_reaction")


async def _telegram_set_thread_title(adapter, chat_id, thread_id, title):
    await adapter.rename_dm_topic(chat_id, int(thread_id), title)
    return _ok(action="set_thread_title")


async def _discord_set_thread_title(adapter, chat_id, thread_id, title):
    if await adapter.rename_thread(thread_id, title):
        return _ok(action="set_thread_title")
    return _err("action_failed", "discord thread rename failed")


def _optional_text(**values: Any) -> Optional[dict[str, Any]]:
    """``invalid_argument`` for an optional parameter that is given but is not text; ``None`` when all are fine."""
    for name, value in values.items():
        if value is not None and not isinstance(value, str):
            return _err("invalid_argument", f"{name} must be a string when given")
    return None


def _optional_size(**values: Any) -> Optional[dict[str, Any]]:
    """``invalid_argument`` for a widget dimension that is given but is not a positive integer of pixels."""
    for name, value in values.items():
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
            return _err("invalid_argument", f"{name} must be a positive integer when given")
    return None


# -- Telegram HTML widgets (Bot API 10.4): the adapter does the Bot API calls -------------------------


async def _telegram_send_widget(
        adapter, chat_id, html, chat_data, public_data, caption, allow_queries, disable_notification, thread_id,
        width, height):
    error = (_optional_text(chat_data=chat_data, public_data=public_data, caption=caption, thread_id=thread_id)
             or _optional_size(width=width, height=height))
    if error is not None:
        return error
    sent = await adapter.send_widget(
        chat_id, html, chat_data=chat_data, public_data=public_data, caption=caption,
        allow_queries=bool(allow_queries), disable_notification=bool(disable_notification), thread_id=thread_id,
        width=width, height=height)
    message_id = sent.get("message_id") if isinstance(sent, dict) else None
    if message_id is None:
        return _err("action_failed", "telegram sendWidget returned no message id")
    return _ok(action="send_widget", message_id=str(message_id))


async def _telegram_edit_widget_data(adapter, chat_id, message_id, chat_data, public_data):
    error = _optional_text(chat_data=chat_data, public_data=public_data)
    if error is not None:
        return error
    if chat_data is None and public_data is None:
        return _err("invalid_argument", "chat_data or public_data is required")
    modified = await adapter.edit_widget_data(chat_id, message_id, chat_data=chat_data, public_data=public_data)
    return _ok(action="edit_widget_data", modified=bool(modified))


async def _telegram_answer_widget_query(adapter, query_id, data):
    error = _optional_text(data=data)
    if error is not None:
        return error
    await adapter.answer_widget_query(query_id, data)
    return _ok(action="answer_widget_query")


_VERBS = {
    "add_reaction": {"telegram": _telegram_add_reaction, "discord": _discord_add_reaction},
    "set_thread_title": {"telegram": _telegram_set_thread_title, "discord": _discord_set_thread_title},
    # HTML widgets are a Telegram surface (Bot API 10.4); other platforms get unsupported_platform_action.
    "send_widget": {"telegram": _telegram_send_widget},
    "edit_widget_data": {"telegram": _telegram_edit_widget_data},
    "answer_widget_query": {"telegram": _telegram_answer_widget_query},
}


class PlatformActions:
    """Per-plugin facade over the live gateway adapter registry.

    Instances are cheap and hold only the owning plugin id; the gateway runner and adapters are
    resolved at call time so a facade created before the gateway starts (plugin ``register()`` runs
    first) still works once adapters connect.
    """

    def __init__(self, plugin_id: str):
        self._plugin_id = plugin_id

    # -- shared plumbing ----------------------------------------------------

    def _capability_granted(self) -> bool:
        try:
            from hermes_cli.plugin_capabilities import plugin_capability_granted

            return plugin_capability_granted(self._plugin_id, CAPABILITY_ID)
        except Exception:
            # Ground rule: failure to read consent state = not granted.
            logger.debug("platform_actions capability check failed for %s", self._plugin_id, exc_info=True)
            return False

    def _resolve_adapter(self, platform: str):
        """Return ``(adapter, error_dict)``; exactly one is non-None."""
        try:
            from gateway.run import _gateway_runner_ref

            runner = _gateway_runner_ref()
        except Exception:
            runner = None
        if runner is None:
            return None, _err("gateway_unavailable", "no gateway runner is active in this process")
        try:
            from gateway.config import Platform

            platform_enum = Platform(str(platform).strip().lower())
        except Exception:
            return None, _err("unknown_platform", f"unknown platform {platform!r}")
        # Multiplex/Team-Gateway: a secondary profile's adapters live in runner._profile_adapters,
        # not runner.adapters. Every adapter-resolution path goes through the same profile-aware,
        # fail-closed lookup so a plugin scoped to one profile can never act through another
        # profile's bot identity. The bare default-profile lookup is only for a runner predating
        # _authorization_adapter (defensive, not expected).
        resolve_fn = getattr(runner, "_authorization_adapter", None)
        if callable(resolve_fn):
            try:
                from hermes_cli.profiles import get_active_profile_name

                profile_name = get_active_profile_name()
            except Exception:
                # Fail closed: an unresolvable profile must not degrade to the default profile's bot.
                logger.debug(
                    "platform_actions: profile resolution failed for %s",
                    self._plugin_id, exc_info=True,
                )
                return None, _err(
                    "adapter_not_registered",
                    f"no {platform_enum.value} adapter is registered "
                    "(active profile could not be resolved)",
                )
            adapter = resolve_fn(platform_enum, profile_name)
        else:
            adapter = getattr(runner, "adapters", {}).get(platform_enum)
        if adapter is None:
            return None, _err("adapter_not_registered", f"no {platform_enum.value} adapter is registered")
        try:
            connected = bool(adapter.is_connected)
        except Exception:
            connected = False
        if not connected:
            return None, _err("adapter_disconnected", f"the {platform_enum.value} adapter is not connected")
        return adapter, None

    def _gate(self, platform: str, **required: Any):
        """Run the shared gate chain. Returns ``(adapter, error_dict)``."""
        if not self._capability_granted():
            return None, _err(
                "capability_not_granted",
                f"plugin {self._plugin_id!r} lacks the {CAPABILITY_ID!r} "
                "capability (grant via consent flow or "
                f"plugins.entries.{self._plugin_id}.allow_platform_actions)",
            )
        for name, value in required.items():
            if not isinstance(value, str) or not value.strip():
                return None, _err("invalid_argument", f"{name} must be a non-empty string")
        return self._resolve_adapter(platform)

    async def _run(self, verb: str, platform: str, *args: str, **required: Any) -> dict[str, Any]:
        """Gate, dispatch *verb* to the adapter's platform implementation, audit, return."""
        adapter, error = self._gate(platform, **required)
        if error is None and adapter is not None:
            try:
                impl = _VERBS[verb].get(getattr(adapter.platform, "value", None))
                if impl is None:
                    result = _err("unsupported_platform_action", f"{verb} is not implemented for {platform}")
                else:
                    result = await impl(adapter, *args)
            except Exception as exc:
                result = _err("action_failed", str(exc)[:512])
        else:
            result = error or _err("gateway_unavailable")
        self._audit(verb, platform, result)
        return result

    # -- v1 verbs -----------------------------------------------------------

    async def add_reaction(self, platform: str, chat_id: str, message_id: str, emoji: str) -> dict[str, Any]:
        """Add/set an emoji reaction on a platform message."""
        return await self._run(
            "add_reaction", platform, chat_id, message_id, emoji,
            chat_id=chat_id, message_id=message_id, emoji=emoji,
        )

    async def set_thread_title(self, platform: str, chat_id: str, thread_id: str, title: str) -> dict[str, Any]:
        """Rename a thread / forum topic."""
        return await self._run(
            "set_thread_title", platform, chat_id, thread_id, title,
            chat_id=chat_id, thread_id=thread_id, title=title,
        )

    # -- Telegram HTML widgets (Bot API 10.4) ------------------------------------

    async def send_widget(
            self, platform: str, chat_id: str, html: str, *, chat_data: Optional[str] = None,
            public_data: Optional[str] = None, caption: Optional[str] = None, allow_queries: bool = True,
            disable_notification: bool = False, thread_id: Optional[str] = None, width: Optional[int] = None,
            height: Optional[int] = None) -> dict[str, Any]:
        """Send a message with an HTML widget; ``message_id`` in the result addresses later data edits.

        The HTML is fixed once sent; ``chat_data`` and ``public_data`` (up to 1024 bytes each) are what
        ``edit_widget_data`` rewrites. With ``allow_queries`` a tap in the widget reaches the plugin as a
        ``widget_query`` platform event. ``width`` / ``height`` fix the widget's size in logical pixels
        (Telegram clamps the width to 200 up to the bubble's width and the height to 50-600); omitted, the
        widget fills the bubble's width and follows its content's height. Telegram currently delivers widgets
        to private chats only."""
        return await self._run(
            "send_widget", platform, chat_id, html, chat_data, public_data, caption, allow_queries,
            disable_notification, thread_id, width, height, chat_id=chat_id, html=html,
        )

    async def edit_widget_data(
            self, platform: str, chat_id: str, message_id: str, *, chat_data: Optional[str] = None,
            public_data: Optional[str] = None) -> dict[str, Any]:
        """Rewrite a widget message's data without touching its HTML: an omitted packet stays, an empty
        string removes it. ``modified`` is False when Telegram already had the same data."""
        return await self._run(
            "edit_widget_data", platform, chat_id, message_id, chat_data, public_data,
            chat_id=chat_id, message_id=message_id,
        )

    async def answer_widget_query(
            self, platform: str, query_id: str, *, data: Optional[str] = None) -> dict[str, Any]:
        """Answer a ``widget_query`` platform event (its ``payload["query_id"]``) within Telegram's 15 seconds;
        ``data`` (up to 1024 bytes) reaches only the widget that asked."""
        return await self._run("answer_widget_query", platform, query_id, data, query_id=query_id)

    def _audit(self, verb: str, platform: str, result: dict[str, Any]) -> None:
        """Every platform action is logged (the #64176 'all actions logged' rule)."""
        logger.info(
            "platform_action plugin=%s verb=%s platform=%s ok=%s%s",
            self._plugin_id,
            verb,
            platform,
            result.get("ok"),
            "" if result.get("ok") else f" error={result.get('error')}",
        )
