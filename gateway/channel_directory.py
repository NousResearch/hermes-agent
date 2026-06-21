"""
Channel directory -- cached map of reachable channels/contacts per platform.

Built on gateway startup, refreshed periodically (every 5 min), and saved to
~/.hermes/channel_directory.json.  The send_message tool reads this file for
action="list" and for resolving human-friendly channel names to numeric IDs.
"""

import json
import logging
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
# Format: {"<platform>": {"<chat_id>": "<friendly name>", ...}, ...}
CHANNEL_ALIASES_PATH = get_hermes_home() / "channel_aliases.json"


def _load_channel_aliases() -> Dict[str, Dict[str, str]]:
    if not CHANNEL_ALIASES_PATH.exists():
        return {}
    try:
        with open(CHANNEL_ALIASES_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


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
        for chat_id, friendly in id_map.items():
            if not isinstance(friendly, str) or not friendly.strip():
                continue
            chat_id = str(chat_id)
            friendly = friendly.strip()
            matched = False
            for e in entries:
                if isinstance(e, dict) and e.get("id") == chat_id:
                    e["name"] = friendly
                    matched = True
            if not matched:
                entries.append({
                    "id": chat_id,
                    "name": friendly,
                    "type": "group" if str(chat_id).endswith("@g.us") else "dm",
                    "thread_id": None,
                })


def _normalize_channel_query(value: str) -> str:
    return value.lstrip("#").strip().lower()


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
                platforms["discord"] = _build_discord(adapter)
            elif platform == Platform.SLACK:
                platforms["slack"] = await _build_slack(adapter)
        except Exception as e:
            logger.warning("Channel directory: failed to build %s: %s", platform.value, e)

    # Platforms that don't support direct channel enumeration get session-based
    # discovery automatically.  Skip infrastructure entries that aren't messaging
    # platforms — everything else falls through to _build_from_sessions().
    _SKIP_SESSION_DISCOVERY = frozenset({"local", "api_server", "webhook"})
    for plat in Platform:
        plat_name = plat.value
        if plat_name in _SKIP_SESSION_DISCOVERY or plat_name in platforms:
            continue
        platforms[plat_name] = _build_from_sessions(plat_name)

    # Include plugin-registered platforms (dynamic enum members aren't in
    # Platform.__members__, so the loop above misses them).
    try:
        from gateway.platform_registry import platform_registry
        for entry in platform_registry.plugin_entries():
            if entry.name not in _SKIP_SESSION_DISCOVERY and entry.name not in platforms:
                platforms[entry.name] = _build_from_sessions(entry.name)
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


def _build_discord(adapter) -> List[Dict[str, str]]:
    """Enumerate Discord channels plus currently active threads.

    Discord guild enumeration gives us authoritative text/forum channels and
    active thread cache.  Session history is still useful for DMs/group DMs, but
    old thread sessions should not be surfaced as send targets forever: archived
    or stale threads make ``send_message(action='list')`` noisy and can resolve
    to destinations that are no longer active.
    """
    channels = []
    seen_ids = set()
    parent_names: Dict[str, str] = {}
    client = getattr(adapter, "_client", None)
    if not client:
        return channels

    try:
        import discord as _discord  # noqa: F401 — SDK presence check
    except ImportError:
        return channels

    def add(entry: Dict[str, str]) -> None:
        entry_id = entry.get("id")
        if not entry_id or entry_id in seen_ids:
            return
        channels.append(entry)
        seen_ids.add(entry_id)

    for guild in client.guilds:
        for ch in guild.text_channels:
            channel_id = str(ch.id)
            parent_names[channel_id] = ch.name
            add({
                "id": channel_id,
                "name": ch.name,
                "guild": guild.name,
                "type": "channel",
            })
        # Forum channels (type 15) — creating a message auto-spawns a thread post.
        forums = getattr(guild, "forum_channels", None) or []
        for ch in forums:
            channel_id = str(ch.id)
            parent_names[channel_id] = ch.name
            add({
                "id": channel_id,
                "name": ch.name,
                "guild": guild.name,
                "type": "forum",
            })

        # Active Discord threads are addressable by thread id, but the tool
        # target format uses parent_id:thread_id so callers keep topic context.
        threads = getattr(guild, "threads", None) or []
        for thread in threads:
            thread_id = str(getattr(thread, "id", "") or "")
            parent_id = getattr(thread, "parent_id", None)
            parent = getattr(thread, "parent", None)
            if parent_id is None and parent is not None:
                parent_id = getattr(parent, "id", None)
            if not thread_id or parent_id is None:
                continue
            parent_id = str(parent_id)
            parent_name = None
            if parent is not None:
                parent_name = getattr(parent, "name", None)
            parent_name = parent_name or parent_names.get(parent_id)
            thread_name = getattr(thread, "name", None) or thread_id
            display_name = f"{parent_name} / {thread_name}" if parent_name else thread_name
            add({
                "id": f"{parent_id}:{thread_id}",
                "thread_id": thread_id,
                "parent_id": parent_id,
                "name": display_name,
                "guild": guild.name,
                "type": "thread",
            })

    # Guild APIs cannot enumerate DMs/group DMs.  Preserve only those
    # session-derived direct contacts; skip stale session threads and duplicate
    # guild channels that are already covered by authoritative enumeration.
    for entry in _build_from_sessions("discord"):
        if entry.get("thread_id"):
            continue
        if entry.get("type") not in {"dm", "group"}:
            continue
        add(entry)
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
        return _build_from_sessions("slack")

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
    for entry in _build_from_sessions("slack"):
        if entry.get("id") not in seen_ids:
            channels.append(entry)
            seen_ids.add(entry.get("id"))

    return channels


def _build_from_sessions(platform_name: str) -> List[Dict[str, str]]:
    """Pull known channels/contacts from sessions.json origin data."""
    sessions_path = get_hermes_home() / "sessions" / "sessions.json"
    if not sessions_path.exists():
        return []

    entries = []
    try:
        with open(sessions_path, encoding="utf-8") as f:
            data = json.load(f)

        seen_ids = set()
        for _key, session in data.items():
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
        if ch.get("id") == chat_id:
            return ch.get("type")
    return None


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

    # 0. Exact ID match — case-sensitive, no normalization. Lets callers pass
    # raw platform IDs (e.g. Slack "C0B0QV5434G") even when the format guard
    # in _parse_target_ref hasn't recognized them as explicit.
    raw = name.strip()
    for ch in channels:
        if ch.get("id") == raw:
            return ch["id"]

    query = _normalize_channel_query(name)

    # 1. Exact name match, including the display labels shown by send_message(action="list")
    for ch in channels:
        if _normalize_channel_query(ch["name"]) == query:
            return ch["id"]
        if _normalize_channel_query(_channel_target_name(platform_name, ch)) == query:
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
                    lines.append(f"  discord:{_channel_target_name(plat_name, ch)}")
            if dms:
                lines.append("Discord (DMs):")
                for ch in dms:
                    lines.append(f"  discord:{_channel_target_name(plat_name, ch)}")
            lines.append("")
        else:
            lines.append(f"{plat_name.title()}:")
            for ch in channels:
                lines.append(f"  {plat_name}:{_channel_target_name(plat_name, ch)}")
            lines.append("")

    lines.append('Use these as the "target" parameter when sending.')
    lines.append('Bare platform name (e.g. "telegram") sends to home channel.')

    return "\n".join(lines)
