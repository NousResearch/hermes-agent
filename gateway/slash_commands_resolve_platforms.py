"""Per-platform ``resolve_access_ref`` implementations for ``/access`` (mixin).

The core command (``gateway/slash_commands_access.py``) calls
``adapter.resolve_access_ref(ref, scope=…, event=…)`` when present.  This module
supplies the resolver for the SDK-backed platforms — Telegram (python-telegram-bot
``get_chat``), Discord (guild roster), Slack (``users.list``/``conversations.list``) —
so ``/access allow user @name`` / ``/access allow group #channel`` work natively there
too.  WAHA keeps its own resolver in the adapter (it needs the WAHA HTTP API); any
platform not covered here just gets the generic core fallback.

Contract (mirrors the WAHA implementation): return an :class:`AccessResolution`; a
reference matching several identities returns ``candidates`` (the command asks the
sender, never guesses); ``None`` means "not my shape" so the generic fallback tries.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class PlatformAccessResolversMixin:
    """``resolve_access_ref`` for Telegram / Discord / Slack adapters."""

    async def resolve_access_ref(self, ref: str, *, scope: str, event=None) -> Optional["AccessResolution"]:  # noqa: F821
        from gateway.slash_commands_access import AccessResolution

        text = str(ref or "").strip()
        if not text:
            return None
        platform = getattr(getattr(self, "name", ""), "lower", lambda: "")()
        try:
            if platform == "telegram":
                return await self._access_resolve_telegram(text, scope=scope, event=event)
            if platform == "discord":
                return await self._access_resolve_discord(text, scope=scope, event=event)
            if platform == "slack":
                return await self._access_resolve_slack(text, scope=scope, event=event)
        except Exception:
            logger.warning("[access] %s resolver failed for %r", platform, text, exc_info=True)
            return AccessResolution()
        return None  # not this mixin's platform

    # ------------------------------------------------------------------ Telegram

    async def _access_resolve_telegram(self, text: str, *, scope: str, event=None) -> "AccessResolution":
        """Telegram ids are numeric.  ``@username``/titles resolve via ``Bot.get_chat``,
        which only succeeds for chats the Bot API can see (interacted with the bot, or a
        public group/channel) — that is a Telegram platform constraint, not an error:
        on failure return ``None`` so the generic fallback tries (and the command replies
        with the honest "give the numeric id" guidance) instead of fabricating a match."""
        from gateway.slash_commands_access import AccessResolution

        if text.lstrip("-").isdigit():
            return AccessResolution(canonical=text)
        if not text.startswith("@"):
            # Bare title: the Bot API cannot search by title, but the bot has SEEN the
            # chats it participates in — resolve against that learned directory
            # (private groups have no @username, so this is the only name path).
            cache = getattr(self, "_seen_chats", None) or {}
            needle = text.lower()
            exact = cache.get(needle)
            if exact:
                return AccessResolution(canonical=exact, label=text)
            substring = [(cid, name) for name, cid in cache.items() if needle in name]
            if len(substring) == 1:
                return AccessResolution(canonical=substring[0][0], label=substring[0][1])
            if substring:
                return AccessResolution(candidates=tuple(substring[:8]))
            return None
        if self._bot is None:
            return None
        try:
            chat = await self._bot.get_chat(text)
        except Exception:
            return None
        chat_id = str(chat.id)
        label = getattr(chat, "title", None) or getattr(chat, "full_name", None) \
            or (f"@{chat.username}" if getattr(chat, "username", None) else chat_id)
        return AccessResolution(canonical=chat_id, label=label)

    # ------------------------------------------------------------------ Discord

    async def _access_resolve_discord(self, text: str, *, scope: str, event=None) -> "AccessResolution":
        """Discord ids are snowflakes; ``@name`` resolves against every guild's member
        roster (same matching as the startup username resolution), ``#name`` and plain
        names against guild channels."""
        from gateway.slash_commands_access import AccessResolution

        client = self._client
        if client is None:
            return AccessResolution(canonical=text) if text.isdigit() else None
        if text.startswith("<@") or text.startswith("<#"):
            inner = text[2:-1].lstrip("!")
            if inner.isdigit():
                return AccessResolution(canonical=inner)
            return None
        if text.isdigit():
            return AccessResolution(canonical=text)
        needle = text.lstrip("@#").lower()
        matches: list = []
        if scope == "group":
            def _iter_channels():
                for guild in client.guilds:
                    yield from getattr(guild, "channels", []) or []
                    # Active threads are not in guild.channels — scan them too so
                    # "/access allow group <thread name>" resolves like channel names.
                    yield from getattr(guild, "threads", []) or []

            for channel in _iter_channels():
                if (channel.name or "").lower() == needle:
                    return AccessResolution(canonical=str(channel.id), label=channel.name)
            substring = [(str(channel.id), channel.name) for channel in _iter_channels()
                         if needle in (channel.name or "").lower()]
            if len(substring) == 1:
                return AccessResolution(canonical=substring[0][0], label=substring[0][1])
            if substring:
                return AccessResolution(candidates=tuple(substring[:8]))
            return AccessResolution()
        for guild in client.guilds:
            try:
                members = guild.members
                if len(members) < (guild.member_count or 0):
                    members = [m async for m in guild.fetch_members(limit=None)]
            except Exception:
                continue
            for member in members or []:
                names = {(member.name or "").lower(), (member.display_name or "").lower(),
                         (member.global_name or "").lower()}
                if needle in names:
                    return AccessResolution(canonical=str(member.id), label=member.display_name or member.name)
                if needle in (member.name or "").lower() or needle in (member.display_name or "").lower():
                    matches.append((str(member.id), member.display_name or member.name))
        if len(matches) == 1:
            return AccessResolution(canonical=matches[0][0], label=matches[0][1])
        if matches:
            return AccessResolution(candidates=tuple(matches[:8]))
        return AccessResolution()

    # ------------------------------------------------------------------ Slack

    async def _access_resolve_slack(self, text: str, *, scope: str, event=None) -> "AccessResolution":
        """Slack ids are ``U…``/``C…``; ``@name`` → ``users.list``, ``#channel`` and
        plain names → ``conversations.list``.  ``<@U…>`` wrappers strip to the id."""
        from gateway.slash_commands_access import AccessResolution

        team_id = None
        if event is not None and isinstance(getattr(event, "metadata", None), dict):
            team_id = event.metadata.get("slack_team_id")
        client = self._client_for("", {"slack_team_id": team_id} if team_id else None)
        if client is None:
            return AccessResolution(canonical=text) if text[:1] in ("U", "C", "W", "G") else None
        if text.startswith("<@") and text.endswith(">"):
            return AccessResolution(canonical=text[2:-1].lstrip("!"))
        if text.startswith("<#") and text.endswith(">"):
            return AccessResolution(canonical=text[2:-1])
        needle = text.lstrip("@#").lower()
        if scope == "group":
            try:
                cursor = None
                channels = []
                while True:
                    kwargs = {"types": "public_channel,private_channel", "limit": 200}
                    if cursor:
                        kwargs["cursor"] = cursor
                    result = await asyncio_to_thread(client.conversations_list, **kwargs)
                    channels.extend(result.get("channels") or [])
                    cursor = (result.get("response_metadata") or {}).get("next_cursor")
                    if not cursor:
                        break
            except Exception:
                return AccessResolution(canonical=text) if text[:1] in ("C", "G") else None
            exact = [c for c in channels if (c.get("name") or "").lower() == needle]
            if len(exact) == 1:
                return AccessResolution(canonical=exact[0]["id"], label=exact[0]["name"])
            substring = [(c["id"], c["name"]) for c in channels if needle in (c.get("name") or "").lower()]
            if len(substring) == 1:
                return AccessResolution(canonical=substring[0][0], label=substring[0][1])
            if substring:
                return AccessResolution(candidates=tuple(substring[:8]))
            return AccessResolution(canonical=text) if text[:1] in ("C", "G") else None
        if text[:1] in ("U", "W"):
            return AccessResolution(canonical=text)
        try:
            cursor = None
            users = []
            while True:
                kwargs = {"limit": 200}
                if cursor:
                    kwargs["cursor"] = cursor
                result = await asyncio_to_thread(client.users_list, **kwargs)
                users.extend(result.get("members") or [])
                cursor = (result.get("response_metadata") or {}).get("next_cursor")
                if not cursor:
                    break
        except Exception:
            return None
        by_name = [u for u in users if (u.get("name") or "").lower() == needle]
        by_real = [u for u in users if ((u.get("profile") or {}).get("real_name") or "").lower() == needle]
        combined = by_name + [u for u in by_real if u not in by_name]
        if len(combined) == 1:
            u = combined[0]
            return AccessResolution(canonical=u["id"], label=(u.get("profile") or {}).get("real_name") or u.get("name"))
        if combined:
            return AccessResolution(candidates=tuple((u["id"], u.get("real_name") or u.get("name") or u["id"]) for u in combined[:8]))
        return AccessResolution()


async def asyncio_to_thread(fn, *args, **kwargs):
    """Local alias so the mixin doesn't drag asyncio imports to module scope."""
    import asyncio
    return await asyncio.to_thread(fn, *args, **kwargs)
