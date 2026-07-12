"""
Channel directory -- cached map of reachable channels/contacts per platform.

Built on gateway startup, refreshed periodically (every 5 min), and saved to
~/.hermes/channel_directory.json.  The send_message tool reads this file for
action="list" and for resolving human-friendly channel names to numeric IDs.
"""

import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from hermes_cli.config import get_hermes_home
from utils import atomic_json_write

logger = logging.getLogger(__name__)

DIRECTORY_PATH = get_hermes_home() / "channel_directory.json"
# User-maintained friendly-name overlay. The directory is fully regenerated
# from live adapters + session data on a timer, so hand-edits to
# channel_directory.json don't survive. Aliases declared here are re-applied
# on every build AND every load, giving durable human-friendly names (and
# letting you pre-name a chat before it has produced any traffic).
# Format:
# {"<platform>": {"<chat_id>": "<friendly name>", ...}, ...}
# or:
# {"<platform>": {"<chat_id>": {"name": "<friendly name>", "aliases": [...]}, ...}, ...}
CHANNEL_ALIASES_PATH = get_hermes_home() / "channel_aliases.json"


def _load_channel_aliases() -> Dict[str, Dict[str, Any]]:
    if not CHANNEL_ALIASES_PATH.exists():
        return {}
    try:
        with open(CHANNEL_ALIASES_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _alias_entry_parts(raw: Any) -> tuple[Optional[str], list[str]]:
    """Return ``(friendly_name, aliases)`` for a channel_aliases entry."""
    if isinstance(raw, str):
        friendly = raw.strip()
        return (friendly or None), []
    if not isinstance(raw, dict):
        return None, []

    friendly = raw.get("name")
    if isinstance(friendly, str):
        friendly = friendly.strip() or None
    else:
        friendly = None

    aliases_raw = raw.get("aliases", [])
    if isinstance(aliases_raw, str):
        aliases_raw = [aliases_raw]
    aliases: list[str] = []
    seen: set[str] = set()
    if isinstance(aliases_raw, list):
        for alias in aliases_raw:
            if not isinstance(alias, str):
                continue
            alias = alias.strip()
            if not alias:
                continue
            normalized = _normalize_channel_query(alias)
            if normalized in seen:
                continue
            seen.add(normalized)
            aliases.append(alias)
    return friendly, aliases


def _merge_entry_aliases(entry: Dict[str, Any], aliases: list[str]) -> None:
    if not aliases:
        return
    existing_raw = entry.get("aliases", [])
    existing = existing_raw if isinstance(existing_raw, list) else []
    merged: list[str] = []
    seen: set[str] = set()
    for alias in [*existing, *aliases]:
        if not isinstance(alias, str):
            continue
        alias = alias.strip()
        if not alias:
            continue
        normalized = _normalize_channel_query(alias)
        if normalized in seen:
            continue
        seen.add(normalized)
        merged.append(alias)
    if merged:
        entry["aliases"] = merged


def _apply_channel_aliases(platforms: Dict[str, Any]) -> None:
    """Overlay friendly names onto directory entries by chat_id.

    Renames matching entries in place; injects a placeholder entry for an
    aliased id that hasn't been discovered yet (so a freshly-created group is
    addressable by name before its first message). Mutates *platforms*.
    """
    aliases = _load_channel_aliases()
    for plat_name, id_map in aliases.items():
        if not isinstance(id_map, dict):
            continue
        entries = platforms.setdefault(plat_name, [])
        if not isinstance(entries, list):
            continue
        for chat_id, raw_alias_entry in id_map.items():
            friendly, aliases = _alias_entry_parts(raw_alias_entry)
            if not friendly and not aliases:
                continue
            chat_id = str(chat_id)
            matched = False
            for e in entries:
                if isinstance(e, dict) and e.get("id") == chat_id:
                    if friendly:
                        e["name"] = friendly
                    _merge_entry_aliases(e, aliases)
                    matched = True
            if not matched:
                entries.append({
                    "id": chat_id,
                    "name": friendly or chat_id,
                    "type": "group" if str(chat_id).endswith("@g.us") else "dm",
                    "thread_id": None,
                    "aliases": aliases,
                })


def _normalize_channel_query(value: str) -> str:
    normalized = value.lstrip("#").strip().casefold()
    normalized = re.sub(r"\s*/\s*", "/", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _channel_aliases(channel: Dict[str, Any]) -> list[str]:
    aliases = channel.get("aliases", [])
    if isinstance(aliases, str):
        aliases = [aliases]
    if not isinstance(aliases, list):
        return []
    return [alias for alias in aliases if isinstance(alias, str) and alias.strip()]


def _channel_target_name(platform_name: str, channel: Dict[str, Any]) -> str:
    """Return the human-facing target label shown to users for a channel entry."""
    name = channel["name"]
    if platform_name == "discord" and channel.get("guild"):
        return f"#{name}"
    if platform_name != "discord" and channel.get("type"):
        return f"{name} ({channel['type']})"
    return name


def _session_entry_id(origin: Dict[str, Any]) -> Optional[str]:
    chat_id = origin.get("chat_id")
    if not chat_id:
        return None
    thread_id = origin.get("thread_id")
    if thread_id:
        # Discord thread sessions use the thread itself as ``chat_id`` while
        # preserving the parent lane as ``parent_chat_id``.  send_message
        # targets need the parent lane plus the thread id so lane validation and
        # delivery both address the same public thread.
        parent_chat_id = origin.get("parent_chat_id")
        if parent_chat_id:
            return f"{parent_chat_id}:{thread_id}"
        return f"{chat_id}:{thread_id}"
    return str(chat_id)


def _session_entry_name(origin: Dict[str, Any]) -> str:
    base_name = origin.get("chat_name") or origin.get("user_name") or str(origin.get("chat_id"))
    thread_id = origin.get("thread_id")
    if not thread_id:
        return base_name

    topic_label = origin.get("chat_topic") or f"topic {thread_id}"
    return f"{base_name} / {topic_label}"


# ---------------------------------------------------------------------------
# Build / refresh
# ---------------------------------------------------------------------------

async def build_channel_directory(adapters: Dict[Any, Any]) -> Dict[str, Any]:
    """
    Build a channel directory from connected platform adapters and session data.

    Returns the directory dict and writes it to DIRECTORY_PATH.
    """
    from gateway.config import Platform

    platforms: Dict[str, List[Dict[str, str]]] = {}

    for platform, adapter in adapters.items():
        try:
            if platform == Platform.DISCORD:
                platforms["discord"] = await asyncio.to_thread(_build_discord, adapter)
            elif platform == Platform.SLACK:
                platforms["slack"] = await _build_slack(adapter)
        except Exception as e:
            logger.warning("Channel directory: failed to build %s: %s", platform.value, e)

    # Platforms that don't support direct channel enumeration get session-based
    # discovery automatically, but only for platforms connected in THIS gateway
    # process. Historical session origins for disabled/decommissioned platforms
    # must not be resurrected into the active send-target directory (stale
    # targets make send_message route to platforms that can no longer deliver).
    _SKIP_SESSION_DISCOVERY = frozenset({"local", "api_server", "webhook"})
    adapter_platform_names = {getattr(p, "value", str(p)) for p in adapters}
    for plat in Platform:
        plat_name = plat.value
        if (
            plat_name in _SKIP_SESSION_DISCOVERY
            or plat_name in platforms
            or plat_name not in adapter_platform_names
        ):
            continue
        platforms[plat_name] = await asyncio.to_thread(_build_from_sessions, plat_name)

    # Include plugin-registered platforms (dynamic enum members aren't in
    # Platform.__members__, so the loop above misses them). Same
    # connected-only rule: don't expose stale session targets for plugins
    # that are not loaded.
    try:
        from gateway.platform_registry import platform_registry
        for entry in platform_registry.plugin_entries():
            if (
                entry.name not in _SKIP_SESSION_DISCOVERY
                and entry.name not in platforms
                and entry.name in adapter_platform_names
            ):
                platforms[entry.name] = await asyncio.to_thread(_build_from_sessions, entry.name)
    except Exception:
        pass

    # Overlay user-maintained friendly names before persisting.
    _apply_channel_aliases(platforms)

    directory = {
        "updated_at": datetime.now().isoformat(),
        "platforms": platforms,
    }

    try:
        atomic_json_write(DIRECTORY_PATH, directory)
    except Exception as e:
        logger.warning("Channel directory: failed to write: %s", e)

    return directory


_DISCORD_PUBLIC_DIRECTORY_TYPE_VALUES = frozenset({0, 5, 10, 11, 15, 16})
_DISCORD_PUBLIC_DIRECTORY_TYPE_NAMES = frozenset({
    "text",
    "news",
    "news_thread",
    "announcement_thread",
    "public_thread",
    "forum",
    "media",
})


def _discord_public_directory_policy_required() -> bool:
    """Return the frozen public-only policy, failing closed on import errors."""

    try:
        from gateway.canonical_writer_boundary import (
            writer_boundary_policy_required,
        )

        return bool(writer_boundary_policy_required())
    except Exception:
        return True


def _discord_cached_public_directory_target(channel: Any) -> bool:
    """Prove a cached Discord object is a public text-capable guild surface.

    This proof is intentionally mechanical: the object must have an allowed
    Discord channel type and exact effective ``view_channel=True`` permission
    for its guild's ``@everyone``/default role. Missing or malformed cache data
    fails closed.
    """

    if channel is None:
        return False
    guild = getattr(channel, "guild", None)
    default_role = getattr(guild, "default_role", None)
    permissions_for = getattr(channel, "permissions_for", None)
    if guild is None or default_role is None or not callable(permissions_for):
        return False

    channel_type = getattr(channel, "type", None)
    type_value = getattr(channel_type, "value", channel_type)
    type_name = str(getattr(channel_type, "name", "") or "").strip().casefold()
    value_is_public = (
        isinstance(type_value, int)
        and not isinstance(type_value, bool)
        and type_value in _DISCORD_PUBLIC_DIRECTORY_TYPE_VALUES
    )
    if not value_is_public and type_name not in _DISCORD_PUBLIC_DIRECTORY_TYPE_NAMES:
        return False

    try:
        permissions = permissions_for(default_role)
    except Exception:
        return False
    return getattr(permissions, "view_channel", None) is True


def _discord_session_target_id(entry: Dict[str, Any]) -> Optional[int]:
    """Return the exact cached target id for a Discord session entry."""

    raw_target = entry.get("thread_id") or entry.get("id")
    if raw_target is None:
        return None
    raw_target = str(raw_target).strip()
    if not raw_target:
        return None
    if ":" in raw_target:
        raw_target = raw_target.rsplit(":", 1)[-1]
    try:
        return int(raw_target)
    except (TypeError, ValueError):
        return None


def _discord_verified_session_entry(
    client: Any,
    entry: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Return a live-cache-proven public Discord session target or ``None``."""

    get_channel = getattr(client, "get_channel", None)
    target_id = _discord_session_target_id(entry)
    if target_id is None or not callable(get_channel):
        return None
    try:
        target = get_channel(target_id)
    except Exception:
        return None
    if not _discord_cached_public_directory_target(target):
        return None

    channel_type = getattr(target, "type", None)
    type_value = getattr(channel_type, "value", channel_type)
    type_name = str(getattr(channel_type, "name", "") or "").strip().casefold()
    numeric_type = (
        type_value
        if isinstance(type_value, int) and not isinstance(type_value, bool)
        else None
    )
    if numeric_type in {10, 11} or type_name in {
        "news_thread",
        "announcement_thread",
        "public_thread",
    }:
        public_type = "thread"
    elif numeric_type in {15, 16} or type_name in {"forum", "media"}:
        public_type = "forum"
    else:
        public_type = "channel"

    verified = dict(entry)
    verified["type"] = public_type
    guild = getattr(target, "guild", None)
    guild_id = getattr(guild, "id", None)
    if not guild_id:
        return None
    verified["guild_id"] = str(guild_id)
    guild_name = getattr(guild, "name", None)
    if guild_name:
        verified["guild"] = str(guild_name)
    if public_type == "thread":
        parent_id = getattr(target, "parent_id", None)
        if not parent_id:
            return None
        verified["parent_channel_id"] = str(parent_id)
        verified["target_type"] = "public_guild_thread"
    elif public_type == "forum":
        verified["target_type"] = "public_guild_forum"
    else:
        verified["target_type"] = "public_guild_channel"
    return verified


def _build_discord(adapter) -> List[Dict[str, str]]:
    """Enumerate Discord send targets, public-only under writer policy."""
    channels = []
    client = getattr(adapter, "_client", None)
    if not client:
        return channels

    try:
        import discord as _discord  # noqa: F401 — SDK presence check
    except ImportError:
        return channels

    public_only = _discord_public_directory_policy_required()

    for guild in getattr(client, "guilds", None) or []:
        for ch in getattr(guild, "text_channels", None) or []:
            if public_only and not _discord_cached_public_directory_target(ch):
                continue
            guild_id = str(getattr(guild, "id", "") or "")
            if public_only and not guild_id:
                continue
            channels.append({
                "id": str(ch.id),
                "name": ch.name,
                "guild": guild.name,
                "guild_id": guild_id,
                "type": "channel",
                "target_type": "public_guild_channel",
            })
        # Forum channels (type 15) — creating a message auto-spawns a thread post.
        forums = getattr(guild, "forum_channels", None) or []
        for ch in forums:
            if public_only and not _discord_cached_public_directory_target(ch):
                continue
            guild_id = str(getattr(guild, "id", "") or "")
            if public_only and not guild_id:
                continue
            channels.append({
                "id": str(ch.id),
                "name": ch.name,
                "guild": guild.name,
                "guild_id": guild_id,
                "type": "forum",
                "target_type": "public_guild_forum",
            })
        # Also include DM-capable users we've interacted with is not
        # feasible via guild enumeration; those come from sessions.

    session_entries = _build_from_sessions("discord")
    if public_only:
        for entry in session_entries:
            if not isinstance(entry, dict):
                continue
            verified = _discord_verified_session_entry(client, entry)
            if verified is not None:
                channels.append(verified)
    else:
        # Preserve generic Hermes behavior when the privileged writer policy is
        # disabled, including legacy session-discovered Discord targets.
        channels.extend(session_entries)
    return channels


async def _build_slack(adapter) -> List[Dict[str, Any]]:
    """List Slack channels the bot has joined across all workspaces.

    Uses ``users.conversations`` against each workspace's web client. Pulls
    public + private channels the bot is a member of, then merges in DMs
    discovered from session history (IMs aren't useful to enumerate
    proactively).
    """
    team_clients = getattr(adapter, "_team_clients", None) or {}
    if not team_clients:
        return await asyncio.to_thread(_build_from_sessions, "slack")

    channels: List[Dict[str, Any]] = []
    seen_ids: set = set()

    for team_id, client in team_clients.items():
        try:
            cursor: Optional[str] = None
            for _page in range(20):  # safety cap on pagination
                response = await client.users_conversations(
                    types="public_channel,private_channel",
                    exclude_archived=True,
                    limit=200,
                    cursor=cursor,
                )
                if not response.get("ok"):
                    logger.warning(
                        "Channel directory: users.conversations not ok for team %s: %s",
                        team_id,
                        response.get("error", "unknown"),
                    )
                    break
                for ch in response.get("channels", []):
                    cid = ch.get("id")
                    name = ch.get("name")
                    if not cid or not name or cid in seen_ids:
                        continue
                    seen_ids.add(cid)
                    channels.append({
                        "id": cid,
                        "name": name,
                        "type": "private" if ch.get("is_private") else "channel",
                    })
                cursor = (response.get("response_metadata") or {}).get("next_cursor")
                if not cursor:
                    break
        except Exception as e:
            logger.warning(
                "Channel directory: failed to list Slack channels for team %s: %s",
                team_id, e,
            )
            continue

    # Merge in DM/group entries discovered from session history.
    for entry in await asyncio.to_thread(_build_from_sessions, "slack"):
        if entry.get("id") not in seen_ids:
            channels.append(entry)
            seen_ids.add(entry.get("id"))

    return channels


def _build_from_sessions(platform_name: str) -> List[Dict[str, str]]:
    """Pull known channels/contacts from gateway session origin data.

    state.db is the primary source (#9006): gateway session rows persist
    origin_json.  Falls back to sessions.json for pre-migration databases.
    """
    entries = _build_from_sessions_db(platform_name)
    if entries:
        return entries
    return _build_from_sessions_json(platform_name)


def _build_from_sessions_db(platform_name: str) -> List[Dict[str, str]]:
    """Pull channels/contacts from state.db gateway session rows."""
    entries: List[Dict[str, str]] = []
    try:
        from hermes_state import SessionDB
        db = SessionDB()
        try:
            lister = getattr(db, "list_gateway_sessions", None)
            if not callable(lister):
                return []
            rows = lister(platform=platform_name, active_only=False)
        finally:
            db.close()

        seen_ids = set()
        for row in rows:
            origin: Dict[str, Any] = {}
            if row.get("origin_json"):
                try:
                    parsed = json.loads(row["origin_json"])
                    if isinstance(parsed, dict):
                        origin = parsed
                except (TypeError, ValueError):
                    pass
            if not origin:
                origin = {
                    "chat_id": row.get("chat_id"),
                    "thread_id": row.get("thread_id"),
                    "chat_name": row.get("display_name"),
                }
            entry_id = _session_entry_id(origin)
            if not entry_id or entry_id in seen_ids:
                continue
            seen_ids.add(entry_id)
            entries.append({
                "id": entry_id,
                "name": _session_entry_name(origin),
                "type": row.get("chat_type") or "dm",
                "thread_id": origin.get("thread_id"),
            })
    except Exception as e:
        logger.debug(
            "Channel directory: state.db session read failed for %s: %s",
            platform_name, e,
        )
    return entries


def _build_from_sessions_json(platform_name: str) -> List[Dict[str, str]]:
    """Legacy fallback: pull channels/contacts from sessions.json origin data."""
    sessions_path = get_hermes_home() / "sessions" / "sessions.json"
    if not sessions_path.exists():
        return []

    entries = []
    try:
        with open(sessions_path, encoding="utf-8") as f:
            data = json.load(f)

        seen_ids = set()
        for _key, session in data.items():
            # Skip documentation/metadata sentinels (keys starting with "_",
            # e.g. the gateway's "_README" note) — not session entries.
            if str(_key).startswith("_") or not isinstance(session, dict):
                continue
            origin = session.get("origin") or {}
            if origin.get("platform") != platform_name:
                continue
            entry_id = _session_entry_id(origin)
            if not entry_id or entry_id in seen_ids:
                continue
            seen_ids.add(entry_id)
            entries.append({
                "id": entry_id,
                "name": _session_entry_name(origin),
                "type": session.get("chat_type", "dm"),
                "thread_id": origin.get("thread_id"),
            })
    except Exception as e:
        logger.debug("Channel directory: failed to read sessions for %s: %s", platform_name, e)

    return entries


# ---------------------------------------------------------------------------
# Read / resolve
# ---------------------------------------------------------------------------

def load_directory() -> Dict[str, Any]:
    """Load the cached channel directory from disk."""
    if not DIRECTORY_PATH.exists():
        base = {"updated_at": None, "platforms": {}}
        _apply_channel_aliases(base["platforms"])
        return base
    try:
        with open(DIRECTORY_PATH, encoding="utf-8") as f:
            data = json.load(f)
        # Re-apply aliases on read so friendly names take effect immediately,
        # even between timed rebuilds and for brand-new alias entries.
        _apply_channel_aliases(data.setdefault("platforms", {}))
        return data
    except Exception:
        base = {"updated_at": None, "platforms": {}}
        _apply_channel_aliases(base["platforms"])
        return base


def lookup_channel_type(platform_name: str, chat_id: str) -> Optional[str]:
    """Return the channel ``type`` string (e.g. ``"channel"``, ``"forum"``) for *chat_id*, or *None* if unknown."""
    directory = load_directory()
    for ch in directory.get("platforms", {}).get(platform_name, []):
        if ch.get("id") == chat_id or ch.get("thread_id") == chat_id:
            return ch.get("type")
    return None


_DISCORD_NON_PUBLIC_TYPES = frozenset({
    "dm", "group", "group_dm", "private", "private_channel", "private_thread",
})


def is_discord_public_target(chat_id: str) -> bool:
    """Return True only for a directory-confirmed guild channel/thread.

    Unknown targets fail closed: a numeric Discord ID alone does not prove
    whether it is a public guild surface or a DM.
    """
    channel_type = str(lookup_channel_type("discord", str(chat_id or "")) or "").strip().lower()
    return bool(channel_type) and channel_type not in _DISCORD_NON_PUBLIC_TYPES


def lookup_discord_public_target(chat_id: str) -> Optional[Dict[str, str]]:
    """Return exact non-secret guild metadata for a directory-proven target.

    This cached tuple is only request construction evidence.  The privileged
    edge independently re-reads Discord and proves the current channel type,
    parent, guild, public visibility, and bot permissions before dispatch.
    """

    target_id = str(chat_id or "").strip()
    if not target_id:
        return None
    directory = load_directory()
    matches = []
    for channel in directory.get("platforms", {}).get("discord", []):
        if not isinstance(channel, dict):
            continue
        channel_id = str(channel.get("thread_id") or channel.get("id") or "").strip()
        if channel_id != target_id:
            continue
        channel_type = str(channel.get("type") or "").strip().lower()
        guild_id = str(channel.get("guild_id") or "").strip()
        target_type = str(channel.get("target_type") or "").strip()
        if (
            not guild_id
            or channel_type in _DISCORD_NON_PUBLIC_TYPES
            or target_type not in {
                "public_guild_channel",
                "public_guild_thread",
                "public_guild_forum",
            }
        ):
            continue
        result = {
            "target_type": target_type,
            "guild_id": guild_id,
            "channel_id": target_id,
        }
        if target_type == "public_guild_thread":
            parent_id = str(channel.get("parent_channel_id") or "").strip()
            if not parent_id or parent_id == target_id:
                continue
            result["parent_channel_id"] = parent_id
        matches.append(result)
    unique = {json.dumps(item, sort_keys=True): item for item in matches}
    return next(iter(unique.values())) if len(unique) == 1 else None


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
    if platform_name == "discord":
        channels = [
            channel for channel in channels
            if str(channel.get("type") or "").strip().lower() not in _DISCORD_NON_PUBLIC_TYPES
            and str(channel.get("type") or "").strip()
        ]
    if not channels:
        return None

    # 0. Exact ID match — case-sensitive, no normalization. Lets callers pass
    # raw platform IDs (e.g. Slack "C0B0QV5434G") even when the format guard
    # in _parse_target_ref hasn't recognized them as explicit.
    raw = name.strip()
    stale_discord_thread_match: str | None = None
    for ch in channels:
        if ch.get("id") == raw:
            return ch["id"]
        if platform_name == "discord" and ch.get("thread_id") == raw:
            target_id = str(ch.get("id") or "")
            if target_id == f"{raw}:{raw}":
                # Older directory builds recorded Discord threads as
                # thread_id:thread_id because session chat_id is the thread
                # itself.  Keep looking so a durable alias or newer
                # parent_channel:thread entry can win.
                stale_discord_thread_match = target_id
                continue
            return ch["id"]

    query = _normalize_channel_query(name)

    # 1. Exact name match, including the display labels shown by send_message(action="list")
    for ch in channels:
        if _normalize_channel_query(ch["name"]) == query:
            return ch["id"]
        if _normalize_channel_query(_channel_target_name(platform_name, ch)) == query:
            return ch["id"]
        for alias in _channel_aliases(ch):
            if _normalize_channel_query(alias) == query:
                return ch["id"]

    # 2. Guild-qualified match for Discord ("GuildName/channel")
    if "/" in query:
        guild_part, ch_part = query.rsplit("/", 1)
        for ch in channels:
            guild = ch.get("guild", "").strip().lower()
            if guild == guild_part and _normalize_channel_query(ch["name"]) == ch_part:
                return ch["id"]

    # 3. Partial prefix match (only if unambiguous)
    matches = [ch for ch in channels if _normalize_channel_query(ch["name"]).startswith(query)]
    if len(matches) == 1:
        return matches[0]["id"]

    if stale_discord_thread_match:
        # A bare Discord thread ID is a valid REST send target and is less
        # misleading than the stale self-parent composite.
        return raw

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
                    suffix = _format_alias_suffix(ch)
                    lines.append(f"  discord:{_channel_target_name(plat_name, ch)}{suffix}")
            lines.append("")
        else:
            lines.append(f"{plat_name.title()}:")
            for ch in channels:
                suffix = _format_alias_suffix(ch)
                lines.append(f"  {plat_name}:{_channel_target_name(plat_name, ch)}{suffix}")
            lines.append("")

    lines.append('Use these as the "target" parameter when sending.')
    lines.append('Bare platform name (e.g. "telegram") sends to home channel.')

    return "\n".join(lines)


def _format_alias_suffix(channel: Dict[str, Any]) -> str:
    aliases = _channel_aliases(channel)
    if not aliases:
        return ""
    visible = ", ".join(aliases[:8])
    if len(aliases) > 8:
        visible = f"{visible}, ..."
    return f" (aliases: {visible})"
