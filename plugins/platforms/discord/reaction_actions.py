"""Plugin-owned Discord reaction action hooks.

This module keeps the optional ``on_discord_reaction_add`` surface out of the
large Discord adapter file. The adapter still owns Discord SDK setup and core
authorization helpers; this module owns the plugin hook contract and the small
set of best-effort side effects callbacks may request.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional


logger = logging.getLogger(__name__)


def reaction_action_hooks_subscribed() -> bool:
    try:
        from hermes_cli.lifecycle import has_hook

        return has_hook("on_discord_reaction_add")
    except Exception:
        return False


def raw_reaction_authorized(
    adapter: Any,
    payload: Any,
    channel: Any,
    user_id: str,
    *,
    discord_module: Any,
) -> bool:
    """Apply Discord channel/user gates before plugin hook dispatch."""
    is_dm = isinstance(channel, discord_module.DMChannel)
    channel_keys: Optional[set[str]] = None
    guild = getattr(channel, "guild", None)

    if not is_dm:
        parent_channel_id = adapter._get_parent_channel_id(channel)
        channel_keys = adapter._discord_channel_keys_from_channel(channel, parent_channel_id)

        allowed = adapter._get_allowed_channels()
        if allowed and "*" not in allowed and not (channel_keys & allowed):
            logger.debug(
                "[%s] Ignoring reaction in non-allowed channel: %s",
                adapter.name,
                channel_keys,
            )
            return False

        ignored = adapter._get_ignored_channels()
        if "*" in ignored or (channel_keys & ignored):
            logger.debug(
                "[%s] Ignoring reaction in ignored channel: %s",
                adapter.name,
                channel_keys,
            )
            return False

        if guild is None:
            guild_id = getattr(payload, "guild_id", None)
            if guild_id is not None and adapter._client is not None:
                get_guild = getattr(adapter._client, "get_guild", None)
                if callable(get_guild):
                    try:
                        guild = get_guild(int(guild_id))
                    except Exception:
                        guild = None

    author = getattr(payload, "member", None)
    return adapter._is_allowed_user(
        user_id,
        author=author,
        guild=guild,
        is_dm=is_dm,
        channel_ids=channel_keys if not is_dm else None,
    )


async def dispatch_raw_reaction_add(
    adapter: Any,
    payload: Any,
    *,
    discord_module: Any,
) -> None:
    """Dispatch user-added Discord reactions to subscribed plugins."""
    if not adapter._client or not reaction_action_hooks_subscribed():
        return
    user_id = str(getattr(payload, "user_id", "") or "")
    bot_user = getattr(adapter._client, "user", None)
    if bot_user is not None and user_id == str(getattr(bot_user, "id", "")):
        return

    try:
        channel_id = str(getattr(payload, "channel_id", "") or "")
        message_id = str(getattr(payload, "message_id", "") or "")
        guild_id_raw = getattr(payload, "guild_id", None)
        guild_id = str(guild_id_raw) if guild_id_raw is not None else None
        emoji = str(getattr(payload, "emoji", "") or "")
        if not channel_id or not message_id or not user_id or not emoji:
            return

        channel = adapter._client.get_channel(int(channel_id))
        if channel is None:
            channel = await adapter._client.fetch_channel(int(channel_id))
        if channel is None or not hasattr(channel, "fetch_message"):
            return
        if not raw_reaction_authorized(
            adapter,
            payload,
            channel,
            user_id,
            discord_module=discord_module,
        ):
            return
        message = await channel.fetch_message(int(message_id))
        if message is None:
            return

        author = getattr(message, "author", None)
        if author is not None and not getattr(author, "bot", False):
            return

        results = await asyncio.to_thread(
            invoke_discord_reaction_hook,
            emoji=emoji,
            user_id=user_id,
            channel_id=channel_id,
            message_id=message_id,
            guild_id=guild_id,
            message_content=getattr(message, "content", ""),
            message_author_id=str(getattr(author, "id", "") or "") if author is not None else None,
        )
        for result in results:
            await apply_reaction_hook_result(adapter, result, channel, message, payload)
    except Exception:
        logger.debug("[%s] Discord reaction action hook failed", adapter.name, exc_info=True)


def invoke_discord_reaction_hook(**payload: Any) -> list[Any]:
    from hermes_cli.lifecycle import invoke_hook

    return invoke_hook("on_discord_reaction_add", **payload)


async def apply_reaction_hook_result(
    adapter: Any,
    result: Any,
    channel: Any,
    message: Any,
    payload: Any,
) -> None:
    """Apply optional best-effort Discord side effects returned by plugins."""
    if isinstance(result, list):
        for item in result:
            await apply_reaction_hook_result(adapter, item, channel, message, payload)
        return
    if not isinstance(result, dict):
        return
    actions = result.get("actions")
    if isinstance(actions, list):
        for item in actions:
            await apply_reaction_hook_result(adapter, item, channel, message, payload)
        return

    action = str(result.get("action", "")).strip().lower()
    if action == "send_message":
        text = result.get("content", result.get("message", ""))
        if isinstance(text, str) and text.strip() and hasattr(channel, "send"):
            await channel.send(content=text)
    elif action in {"add_reaction", "add_bot_reaction"}:
        emoji = result.get("emoji")
        if isinstance(emoji, str) and emoji.strip():
            await adapter._add_reaction(message, emoji)
    elif action == "remove_bot_reaction":
        emoji = result.get("emoji")
        if isinstance(emoji, str) and emoji.strip():
            await adapter._remove_reaction(message, emoji)
    elif action == "remove_user_reaction" and hasattr(message, "remove_reaction"):
        emoji = result.get("emoji") or str(getattr(payload, "emoji", "") or "")
        user = await resolve_reaction_payload_user(adapter, payload)
        if isinstance(emoji, str) and emoji.strip() and user is not None:
            try:
                await message.remove_reaction(emoji, user)
            except Exception:
                logger.debug("[%s] remove_user_reaction failed", adapter.name, exc_info=True)


async def resolve_reaction_payload_user(adapter: Any, payload: Any) -> Any:
    member = getattr(payload, "member", None)
    if member is not None:
        return member
    user_id = getattr(payload, "user_id", None)
    if user_id is None or not adapter._client:
        return None
    get_user = getattr(adapter._client, "get_user", None)
    if callable(get_user):
        user = get_user(int(user_id))
        if user is not None:
            return user
    fetch_user = getattr(adapter._client, "fetch_user", None)
    if callable(fetch_user):
        try:
            return await fetch_user(int(user_id))
        except Exception:
            return None
    return None
