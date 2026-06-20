"""
Channel directory -- cached map of reachable channels/contacts per platform.

Built on gateway startup, refreshed periodically (every 5 min), and saved to
~/.hermes/channel_directory.json.  The send_message tool reads this file for
action="list" and for resolving human-friendly channel names to numeric IDs.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

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
    """Enumerate all text channels and forum channels the Discord bot can see."""
    channels = []
    client = getattr(adapter, "_client", None)
    if not client:
        return channels

    try:
        import discord as _discord  # noqa: F401 — SDK presence check
    except ImportError:
        return channels

    for guild in client.guilds:
        for ch in guild.text_channels:
            channels.append({
                "id": str(ch.id),
                "name": ch.name,
                "guild": guild.name,
                "type": "channel",
            })
        # Forum channels (type 15) — creating a message auto-spawns a thread post.
        forums = getattr(guild, "forum_channels", None) or []
        for ch in forums:
            channels.append({
                "id": str(ch.id),
                "name": ch.name,
                "guild": guild.name,
                "type": "forum",
            })
        # Also include DM-capable users we've interacted with is not
        # feasible via guild enumeration; those come from sessions.

    # Merge any DMs from session history
    channels.extend(_build_from_sessions("discord"))
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
                    # Track the workspace (team_id) and the human-friendly
                    # workspace name on every entry. Without this, the channel
                    # name → ID resolver can't tell two workspaces apart, so
                    # "send to #engineering" silently picks whichever
                    # workspace's engineering channel the loop happened to
                    # visit first — a cross-channel Slack misroute. The
                    # workspace name is best-effort: users.conversations
                    # doesn't return it, so we fall back to the team_id and
                    # let callers (display, resolve_channel_name_strict)
                    # surface it.
                    channels.append({
                        "id": cid,
                        "name": name,
                        "type": "private" if ch.get("is_private") else "channel",
                        "team_id": team_id,
                        "team_name": ch.get("team_name") or team_id,
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
        if ch.get("id") == chat_id:
            return ch.get("type")
    return None


def lookup_channel_team(platform_name: str, chat_id: str) -> Optional[str]:
    """Return the workspace ``team_id`` (Slack) for *chat_id*, or *None* if unknown.

    Used by ``send_message`` to select the right bot token when the caller
    passes an explicit channel ID like ``slack:C0X``. Without this, an
    explicit ID would still post via the primary token, even when the
    channel lives in a different workspace — a cross-channel misroute on
    the explicit-ID path.
    """
    directory = load_directory()
    for ch in directory.get("platforms", {}).get(platform_name, []):
        if ch.get("id") == chat_id:
            team_id = ch.get("team_id")
            return team_id if team_id else None
    return None


def resolve_channel_name(platform_name: str, name: str) -> Optional[str]:
    """
    Resolve a human-friendly channel name to a numeric ID.

    Matching strategy (case-insensitive, first match wins UNLESS the same
    name exists in multiple workspaces, in which case the resolver refuses
    with ``None`` rather than silently misrouting):

    - Discord: "bot-home", "#bot-home", "GuildName/bot-home"
    - Telegram: display name or group name
    - Slack: "engineering", "#engineering"

    Multi-workspace ambiguity check
    -------------------------------
    For Slack, channels from different workspaces often share the same
    human-friendly name. Picking the first match would silently send the
    message to whichever workspace happened to be enumerated first — a
    cross-channel Slack misroute. The fix is to refuse the resolution and
    return ``None`` so the caller surfaces a disambiguation error. The
    strict sibling :func:`resolve_channel_name_strict` returns the
    ``(chat_id, team_id)`` pair so the caller can route to the correct
    workspace client.
    """
    directory = load_directory()
    channels = directory.get("platforms", {}).get(platform_name, [])
    if not channels:
        return None

    # 0. Exact ID match — case-sensitive, no normalization. Lets callers pass
    # raw platform IDs (e.g. Slack "C0B0QV5434G") even when the format guard
    # in _parse_target_ref hasn't recognized them as explicit. An exact ID
    # is unambiguous by construction (Slack CIDs are globally unique across
    # workspaces), so no workspace-disambiguation check is needed here.
    raw = name.strip()
    for ch in channels:
        if ch.get("id") == raw:
            return ch["id"]

    query = _normalize_channel_query(name)

    # 1. Exact name match, including the display labels shown by send_message(action="list")
    exact_matches = [
        ch for ch in channels
        if _normalize_channel_query(ch["name"]) == query
        or _normalize_channel_query(_channel_target_name(platform_name, ch)) == query
    ]
    if exact_matches:
        # Reject cross-workspace ambiguity for Slack: two workspaces with
        # the same channel name = no way to pick safely. Callers fall back
        # to the strict resolver or surface a disambiguation error.
        if _is_cross_workspace_ambiguous(exact_matches, platform_name):
            return None
        return exact_matches[0]["id"]

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
    if len(matches) > 1 and _is_cross_workspace_ambiguous(matches, platform_name):
        # Multiple workspaces match by prefix — refuse to pick.
        return None

    return None


def resolve_channel_name_strict(
    platform_name: str, name: str,
) -> Optional[Tuple[str, str]]:
    """
    Resolve a channel name to ``(chat_id, team_id)`` — workspace-aware.

    Same matching strategy as :func:`resolve_channel_name` but returns the
    workspace identifier alongside the channel ID. The ``team_id`` is
    forwarded to :class:`SlackAdapter` so its :meth:`_get_client` selects
    the correct workspace's WebClient instead of falling back to the
    primary bot token (which can silently post to the wrong workspace).

    Returns ``None`` for both unknown names and cross-workspace ambiguities.
    For an exact ID match, ``team_id`` is the empty string — callers
    resolve the workspace via the channel ID or via a session-origin
    lookup as a fallback.
    """
    directory = load_directory()
    channels = directory.get("platforms", {}).get(platform_name, [])
    if not channels:
        return None

    # 0. Exact ID match. The ID uniquely identifies the channel; the
    # ``team_id`` is whatever we recorded (empty string for legacy entries
    # that pre-date the workspace-tracking fix). Callers can resolve the
    # workspace through the adapter's ``_channel_team`` map.
    raw = name.strip()
    for ch in channels:
        if ch.get("id") == raw:
            return ch["id"], ch.get("team_id", "") or ""

    query = _normalize_channel_query(name)

    # 1. Exact name match.
    exact_matches = [
        ch for ch in channels
        if _normalize_channel_query(ch["name"]) == query
        or _normalize_channel_query(_channel_target_name(platform_name, ch)) == query
    ]
    if exact_matches:
        if _is_cross_workspace_ambiguous(exact_matches, platform_name):
            return None
        match = exact_matches[0]
        return match["id"], match.get("team_id", "") or ""

    # 2. Guild-qualified match for Discord ("GuildName/channel")
    if "/" in query:
        guild_part, ch_part = query.rsplit("/", 1)
        for ch in channels:
            guild = ch.get("guild", "").strip().lower()
            if guild == guild_part and _normalize_channel_query(ch["name"]) == ch_part:
                return ch["id"], ch.get("team_id", "") or ""

    # 3. Partial prefix match (only if unambiguous)
    matches = [ch for ch in channels if _normalize_channel_query(ch["name"]).startswith(query)]
    if len(matches) == 1:
        return matches[0]["id"], matches[0].get("team_id", "") or ""
    if len(matches) > 1 and _is_cross_workspace_ambiguous(matches, platform_name):
        return None

    return None


def _is_cross_workspace_ambiguous(
    matches: List[Dict[str, Any]], platform_name: str,
) -> bool:
    """Return True if ``matches`` span more than one workspace.

    Used by the channel-name resolvers to refuse silent misroutes when the
    same human-friendly name exists in multiple Slack workspaces (or,
    defensively, across multiple Discord guilds if entries ever carry a
    team_id). The cross-workspace case is platform-specific:

    - Slack: a bot in two workspaces with the same channel name (e.g. two
      orgs both with ``#engineering``) — first-match-wins would silently
      send to the wrong workspace.
    - Other platforms: ``team_id`` is not populated, so a single match is
      never cross-workspace ambiguous regardless of how many entries
      exist.
    """
    if platform_name != "slack":
        return False
    team_ids = {ch.get("team_id", "") for ch in matches}
    # Filter out empty strings so legacy entries (no team_id) don't falsely
    # signal ambiguity. If every match is untagged, the original
    # first-match-wins behavior is preserved.
    tagged = {tid for tid in team_ids if tid}
    return len(tagged) > 1


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
        elif plat_name == "slack":
            # Group Slack channels by workspace (team). A bot may be installed
            # in multiple workspaces and several can share the same channel
            # name (e.g. two orgs each with ``#engineering``); without the
            # workspace label the LLM has no way to pick the right one and
            # the message silently lands in the wrong workspace.
            #
            # Channels without a team_id (legacy single-workspace cache or
            # session-derived DMs) fall into a separate "Untagged" bucket so
            # they are still visible rather than hidden.
            workspaces: Dict[str, List] = {}
            untagged: List = []
            for ch in channels:
                team_id = ch.get("team_id", "")
                if team_id:
                    workspaces.setdefault(team_id, []).append(ch)
                else:
                    untagged.append(ch)

            for team_id, ws_channels in sorted(workspaces.items()):
                team_label = ws_channels[0].get("team_name") or team_id
                lines.append(f"Slack ({team_label}):")
                for ch in sorted(ws_channels, key=lambda c: c["name"]):
                    lines.append(f"  slack:{_channel_target_name(plat_name, ch)}")
            if untagged:
                lines.append("Slack (Untagged — legacy entries):")
                for ch in sorted(untagged, key=lambda c: c["name"]):
                    lines.append(f"  slack:{_channel_target_name(plat_name, ch)}")
            lines.append("")
        else:
            lines.append(f"{plat_name.title()}:")
            for ch in channels:
                lines.append(f"  {plat_name}:{_channel_target_name(plat_name, ch)}")
            lines.append("")

    lines.append('Use these as the "target" parameter when sending.')
    lines.append('Bare platform name (e.g. "telegram") sends to home channel.')

    return "\n".join(lines)
