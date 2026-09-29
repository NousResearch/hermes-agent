"""Discord desk connection: validate a bot token, user id, and channel, then post text.

The desktop form is the only caller. Posting uses the saved home channel, never a
channel id supplied with the message, so the form cannot be aimed at an arbitrary room.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any, Callable

DISCORD_MESSAGE_LIMIT = 2000
_API = "https://discord.com/api/v10"
_SNOWFLAKE = re.compile(r"\d{15,22}")
_MENTION = re.compile(r"<(?:@!?|#)(\d{15,22})>")
_TEXT_CHANNEL_TYPES = frozenset({0, 5})


class DiscordConnectError(ValueError):
    """A form value Discord will not accept. The message is safe to show."""


def clean_snowflake(raw: str) -> str:
    """A Discord user or channel id, including a pasted ``<@id>`` or ``<#id>``."""
    text = (raw or "").strip()
    mention = _MENTION.fullmatch(text)
    if mention:
        text = mention.group(1)
    if not _SNOWFLAKE.fullmatch(text):
        raise DiscordConnectError(
            "Use the numeric ID. In Discord, turn on Developer Mode, then right-click and Copy ID."
        )
    return text


def clean_token(raw: str) -> str:
    """The bot token from the Bot page. An application id is only digits."""
    token = (raw or "").strip().strip('"').strip("'")
    if len(token) < 20 or token.isdigit() or token.count(".") < 2:
        raise DiscordConnectError(
            "Paste the bot token from the Discord Bot page, not the application ID."
        )
    return token


def clean_message(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        raise DiscordConnectError("Write a message first.")
    if len(text) > DISCORD_MESSAGE_LIMIT:
        raise DiscordConnectError("Discord accepts 2000 characters. Shorten the message.")
    return text


def resolve_token(submitted: str, saved: str) -> str:
    """A new token replaces the saved one. A blank field keeps the saved token."""
    if (submitted or "").strip():
        return clean_token(submitted)
    saved_token = (saved or "").strip()
    if not saved_token:
        raise DiscordConnectError("Paste the bot token from the Discord Bot page.")
    return saved_token


def outcome_error(raw: str) -> str | None:
    """``None`` when a send_message result succeeded. Otherwise a short error."""
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return "Discord did not accept the message."
    if isinstance(payload, dict) and payload.get("success") is True:
        return None
    if isinstance(payload, dict):
        err = payload.get("error") or payload.get("message") or "Discord did not accept the message."
        return str(err)[:300]
    return "Discord did not accept the message."


def text_channels_from_guilds(
    guilds: list[dict[str, Any]], channels_by_guild: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    """Guild text channels the bot can see, in Discord's own order."""
    listed: list[dict[str, Any]] = []
    for guild in guilds:
        guild_id = str(guild.get("id") or "")
        name = str(guild.get("name") or "Server")
        rows = []
        for channel in channels_by_guild.get(guild_id, []):
            if channel.get("type") not in _TEXT_CHANNEL_TYPES:
                continue
            channel_id = str(channel.get("id") or "")
            if not _SNOWFLAKE.fullmatch(channel_id):
                continue
            rows.append((int(channel.get("position") or 0), str(channel.get("name") or channel_id), channel_id))
        rows.sort()
        if rows:
            listed.append({
                "guild": name,
                "channels": [{"id": channel_id, "name": channel_name} for _, channel_name, channel_id in rows],
            })
    if not listed:
        raise DiscordConnectError(
            "The bot is not in a channel it can see. Invite it to the server and allow View Channel, then try again."
        )
    return listed


def list_text_channels(token: str, opener: Callable[..., Any] | None = None) -> list[dict[str, Any]]:
    """Live channel list for a bot token. ``opener`` is urllib's ``urlopen`` signature."""
    clean_token(token)
    open_url = opener or urllib.request.urlopen
    guilds = _discord_get(token, "/users/@me/guilds", open_url)
    if not isinstance(guilds, list) or not guilds:
        raise DiscordConnectError(
            "This bot is not in a server yet. Invite it, then load channels again."
        )
    by_guild: dict[str, list[dict[str, Any]]] = {}
    for guild in guilds:
        if not isinstance(guild, dict):
            continue
        guild_id = str(guild.get("id") or "")
        payload = _discord_get(token, f"/guilds/{guild_id}/channels", open_url)
        by_guild[guild_id] = payload if isinstance(payload, list) else []
    return text_channels_from_guilds([g for g in guilds if isinstance(g, dict)], by_guild)


def _discord_get(token: str, path: str, opener: Callable[..., Any]) -> Any:
    request = urllib.request.Request(
        _API + path,
        headers={"Authorization": "Bot " + token, "User-Agent": "HermesDesktop (discord-connect)"},
    )
    try:
        with opener(request, timeout=20) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            raise DiscordConnectError(
                "Discord rejected that bot token. Paste the token from the Bot page."
            ) from None
        raise DiscordConnectError(
            "Discord could not list channels. Invite the bot into the server, then try again."
        ) from None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        raise DiscordConnectError("Discord could not be reached. Try again.") from None
