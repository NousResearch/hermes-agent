"""
Channel directory -- cached map of reachable channels/contacts per platform.

Built on gateway startup, refreshed periodically (every 5 min), and saved to
~/.hermes/channel_directory.json.  The send_message tool reads this file for
action="list" and for resolving human-friendly channel names to numeric IDs.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DIRECTORY_PATH = Path.home() / ".hermes" / "channel_directory.json"


# ---------------------------------------------------------------------------
# Build / refresh
# ---------------------------------------------------------------------------


def build_channel_directory(adapters: Dict[Any, Any]) -> Dict[str, Any]:
    """
    Build a channel directory from connected platform adapters and session data.

    Returns the directory dict and writes it to DIRECTORY_PATH.
    """
    from gateway.config import Platform

    platforms: Dict[str, List[Dict[str, str]]] = {}

    for platform, adapter in adapters.items():
        try:
            if platform == Platform.DISCORD:
                platforms["discord"] = _build_discord(adapter)
            elif platform == Platform.SLACK:
                platforms["slack"] = _build_slack(adapter)
        except Exception as e:
            logger.warning("Channel directory: failed to build %s: %s", platform.value, e)

    # Telegram & WhatsApp can't enumerate chats -- pull from session history
    for plat_name in ("telegram", "whatsapp"):
        if plat_name not in platforms:
            platforms[plat_name] = _build_from_sessions(plat_name)

    directory = {
        "updated_at": datetime.now().isoformat(),
        "platforms": platforms,
    }

    try:
        DIRECTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DIRECTORY_PATH, "w") as f:
            json.dump(directory, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning("Channel directory: failed to write: %s", e)

    return directory


def _build_discord(adapter) -> List[Dict[str, str]]:
    """Enumerate all text channels the Discord bot can see."""
    channels = []
    client = getattr(adapter, "_client", None)
    if not client:
        return channels

    try:
        import discord as _discord
    except ImportError:
        return channels

    for guild in client.guilds:
        for ch in guild.text_channels:
            channels.append(
                {
                    "id": str(ch.id),
                    "name": ch.name,
                    "guild": guild.name,
                    "type": "channel",
                }
            )
        # Also include DM-capable users we've interacted with is not
        # feasible via guild enumeration; those come from sessions.

    # Merge any DMs from session history
    channels.extend(_build_from_sessions("discord"))
    return channels


def _build_slack(adapter) -> List[Dict[str, str]]:
    """List Slack channels the bot has joined."""
    channels = []
    # Slack adapter may expose a web client
    client = getattr(adapter, "_app", None) or getattr(adapter, "_client", None)
    if not client:
        return _build_from_sessions("slack")

    try:
        import asyncio

        from tools.send_message_tool import _send_slack  # noqa: F401
        # Use the Slack Web API directly if available
    except Exception:
        pass

    # Fallback to session data
    return _build_from_sessions("slack")


def _build_from_sessions(platform_name: str) -> List[Dict[str, str]]:
    """Pull known channels/contacts from sessions.json origin data."""
    sessions_path = Path.home() / ".hermes" / "sessions" / "sessions.json"
    if not sessions_path.exists():
        return []

    entries = []
    try:
        with open(sessions_path) as f:
            data = json.load(f)

        seen_ids = set()
        for _key, session in data.items():
            origin = session.get("origin") or {}
            if origin.get("platform") != platform_name:
                continue
            chat_id = origin.get("chat_id")
            if not chat_id or chat_id in seen_ids:
                continue
            seen_ids.add(chat_id)
            entries.append(
                {
                    "id": str(chat_id),
                    "name": origin.get("chat_name") or origin.get("user_name") or str(chat_id),
                    "type": session.get("chat_type", "dm"),
                }
            )
    except Exception as e:
        logger.debug("Channel directory: failed to read sessions for %s: %s", platform_name, e)

    return entries


# ---------------------------------------------------------------------------
# Read / resolve
# ---------------------------------------------------------------------------


def load_directory() -> Dict[str, Any]:
    """Load the cached channel directory from disk."""
    if not DIRECTORY_PATH.exists():
        return {"updated_at": None, "platforms": {}}
    try:
        with open(DIRECTORY_PATH) as f:
            return json.load(f)
    except Exception:
        return {"updated_at": None, "platforms": {}}


def resolve_channel_name(platform_name: str, name: str) -> Optional[str]:
    """
    Resolve a human-friendly channel name to a numeric ID.

    Matching strategy (case-insensitive, first match wins):
    - Discord: "bot-home", "#bot-home", "GuildName/bot-home"
    - Telegram: display name or group name
    - Slack: "engineering", "#engineering"
    """
    directory = load_directory()
    channels = directory.get("platforms", {}).get(platform_name, [])
    if not channels:
        return None

    query = name.lstrip("#").lower()

    # 1. Exact name match
    for ch in channels:
        if ch["name"].lower() == query:
            return ch["id"]

    # 2. Guild-qualified match for Discord ("GuildName/channel")
    if "/" in query:
        guild_part, ch_part = query.rsplit("/", 1)
        for ch in channels:
            guild = ch.get("guild", "").lower()
            if guild == guild_part and ch["name"].lower() == ch_part:
                return ch["id"]

    # 3. Partial prefix match (only if unambiguous)
    matches = [ch for ch in channels if ch["name"].lower().startswith(query)]
    if len(matches) == 1:
        return matches[0]["id"]

    return None


def format_directory_for_display() -> str:
    """Format the channel directory as a human-readable list for the model."""
    directory = load_directory()
    platforms = directory.get("platforms", {})

    if not any(platforms.values()):
        return "No messaging platforms connected or no channels discovered yet."

    lines = ["Available messaging targets:\n"]

    for plat_name, channels in sorted(platforms.items()):
        if not channels:
            continue

        # Group Discord channels by guild
        if plat_name == "discord":
            guilds: Dict[str, List] = {}
            dms: List = []
            for ch in channels:
                guild = ch.get("guild")
                if guild:
                    guilds.setdefault(guild, []).append(ch)
                else:
                    dms.append(ch)

            for guild_name, guild_channels in sorted(guilds.items()):
                lines.append(f"Discord ({guild_name}):")
                for ch in sorted(guild_channels, key=lambda c: c["name"]):
                    lines.append(f"  discord:#{ch['name']}")
            if dms:
                lines.append("Discord (DMs):")
                for ch in dms:
                    lines.append(f"  discord:{ch['name']}")
            lines.append("")
        else:
            lines.append(f"{plat_name.title()}:")
            for ch in channels:
                type_label = f" ({ch['type']})" if ch.get("type") else ""
                lines.append(f"  {plat_name}:{ch['name']}{type_label}")
            lines.append("")

    lines.append('Use these as the "target" parameter when sending.')
    lines.append('Bare platform name (e.g. "telegram") sends to home channel.')

    return "\n".join(lines)
