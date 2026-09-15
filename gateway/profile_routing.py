"""Profile-based routing: route guilds/channels/threads to different profiles.

Matching priority, most specific first (``gateway.profile_routes`` in config.yaml):
platform + chat_id + thread_id (14) → platform + chat_id (6) → platform + guild_id (2)
→ default profile. For Discord threads/forum posts ``parent_chat_id`` carries the
direct parent, so a channel route also matches any thread/post under it.

A discriminator may be declared as an exact id, a channel/guild NAME, or a regex pattern
anchored at the name start (implicit ``^``); names and patterns resolve AFTER the exact-id
compare misses, from the gateway's cached channel directory (#109676).

A route applies only to messages received by the bot of its ``bot_profile`` (default: the
default profile's shared bot). Telegram DM ``chat_id == user_id`` for EVERY bot, so without
this a ``chat_id`` route meant for the shared bot would re-home the same user's DM with a
dedicated secondary bot into another profile (#104933).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Baileys and Cloud share phone/JID/LID identity rules; other platforms compare exactly.
_WHATSAPP_IDENTITY_PLATFORMS = {"whatsapp", "whatsapp_cloud"}
_WHATSAPP_NON_USER_SUFFIXES = ("@g.us", "@broadcast", "@newsletter")


def _is_whatsapp_non_user_chat(chat_id: Optional[str]) -> bool:
    """True for group / broadcast / newsletter JIDs — not a sender identity."""
    return bool(chat_id) and str(chat_id).strip().lower().endswith(_WHATSAPP_NON_USER_SUFFIXES)


def _whatsapp_user_chat_ids_match(platform: str, left: Optional[str], right: Optional[str]) -> bool:
    """True when two WhatsApp *user* chat_ids refer to the same person.

    Uses ``expand_whatsapp_aliases`` (same helper as session keys and adapter allowlists) so a bare
    number, JID and LID collapse to one identity. Group/broadcast JIDs are chats, not senders;
    non-WhatsApp platforms → False.
    """
    if (
        (platform or "").strip().lower() not in _WHATSAPP_IDENTITY_PLATFORMS
        or not left or not right
        or _is_whatsapp_non_user_chat(left) or _is_whatsapp_non_user_chat(right)
    ):
        return False
    from gateway.whatsapp_identity import expand_whatsapp_aliases

    left_aliases = expand_whatsapp_aliases(str(left))
    return bool(left_aliases and left_aliases & expand_whatsapp_aliases(str(right)))


class ProfileRouteRejected(RuntimeError):
    """An explicit route matched a profile this gateway does not serve."""


# A discriminator holding any of these is a regex PATTERN over channel/guild names, anchored at the
# name start (implicit ``^``); every other value stays a literal id/name, so existing configs are
# untouched (#109676).
_PATTERN_METACHARS = frozenset("^$+?[](){}|\\")


def _is_pattern(value: str) -> bool:
    """True when a route discriminator holds regex metacharacters (opt-in pattern matching)."""
    return any(char in _PATTERN_METACHARS for char in value)


def _compile_pattern(value: str):
    """Compiled pattern for *value*; None when it does not compile (such a value never matches)."""
    try:
        return re.compile(value)
    except re.error:
        return None


def _matches_name_or_pattern(value: str, names: Iterable[str]) -> bool:
    """True when a route discriminator *value* equals a resolved NAME, or (pattern-shaped) matches one."""
    names = list(names)
    if value in names:
        return True
    pattern = _compile_pattern(value) if _is_pattern(value) else None
    return pattern is not None and any(pattern.match(name) for name in names)


def _directory_scope(platform: str, ids: Iterable[Optional[str]]) -> Tuple[List[str], List[str]]:
    """``(channel_names, guild_names)`` for inbound *ids*, from the gateway's cached channel directory.

    Lets a route declare a channel/guild NAME instead of an opaque id (#109676). One directory read
    per call; unknown ids, an absent directory, or a DM resolve to empty lists — never an exception
    into the routing path, so an unresolvable name simply does not match.
    """
    try:
        from gateway.channel_directory import lookup_channel_entries
        platform_name = str(getattr(platform, "value", platform))
        entries = list(lookup_channel_entries(platform_name, ids).values())
    except Exception:
        logger.debug("Profile route name resolution unavailable for platform %s", platform, exc_info=True)
        return [], []
    return (
        [str(e["name"]) for e in entries if e.get("name")],
        [str(e["guild"]) for e in entries if e.get("guild")],
    )


def _discriminator_holds(value: str, platform: str, ids: Iterable[Optional[str]], *, guild: bool = False) -> bool:
    """NAME/pattern match for a route discriminator, after the caller's exact-id compare missed.

    *ids* are the inbound ids whose scope names the discriminator may match: the channel name for
    ``chat_id``/``thread_id``, the enclosing guild name for ``guild_id``.
    """
    names, guild_names = _directory_scope(platform, ids)
    return _matches_name_or_pattern(value, guild_names if guild else names)


@dataclass(frozen=True)
class ProfileRoute:
    """A single routing rule that maps a platform scope to a profile."""

    name: str
    platform: str
    profile: str
    guild_id: Optional[str] = None
    chat_id: Optional[str] = None
    thread_id: Optional[str] = None
    enabled: bool = True
    bot_profile: Optional[str] = None  # None = the default profile's bot

    @property
    def specificity(self) -> int:
        """Higher value = more specific match."""
        return 2 * bool(self.guild_id) + 4 * bool(self.chat_id) + 8 * bool(self.thread_id)

    def matches(
        self, platform: str, guild_id: Optional[str] = None, chat_id: Optional[str] = None,
        thread_id: Optional[str] = None, parent_chat_id: Optional[str] = None,
        adapter_profile: Optional[str] = None,
    ) -> bool:
        """True if every discriminator the route declares holds (AND).

        ``chat_id`` matches the channel directly or as the parent of a thread/forum post; WhatsApp
        ``chat_id`` also matches across number/JID/LID after the exact check (groups/broadcasts stay exact-only).
        ``adapter_profile`` is the profile owning the receiving bot (``None`` = default); it must equal
        the route's ``bot_profile``.

        A discriminator declared as a channel/guild NAME or regex pattern resolves from the cached
        channel directory only AFTER its exact-id compare misses (#109676) — id routes read nothing —
        and a pattern is anchored at the name start.
        """
        if not self.enabled or self.platform != platform:
            return False
        if _bot_profile_key(self.bot_profile) != _bot_profile_key(adapter_profile):
            return False
        if self.thread_id and self.thread_id != thread_id:
            if not _discriminator_holds(self.thread_id, platform, (thread_id,)):
                return False
        if self.chat_id and self.chat_id not in (chat_id, parent_chat_id):
            if not (
                _whatsapp_user_chat_ids_match(platform, self.chat_id, chat_id)
                or _whatsapp_user_chat_ids_match(platform, self.chat_id, parent_chat_id)
                or _discriminator_holds(self.chat_id, platform, (chat_id, parent_chat_id))
            ):
                return False
        if self.guild_id and self.guild_id != guild_id:
            if not _discriminator_holds(self.guild_id, platform, (chat_id, parent_chat_id), guild=True):
                return False
        return True


def _bot_profile_key(name: Optional[str]) -> Optional[str]:
    """``None`` for the default profile, else the profile name (mirrors ``set_owner_profile``)."""
    name = (name or "").strip()
    return None if not name or name == "default" else name


def _coerce_route_id(value: Any) -> Optional[str]:
    """Normalize a route discriminator to str for strict equality matching.

    PyYAML loads unquoted numeric IDs as ``int`` while ``SessionSource`` fields are ``str``. Only
    ``int`` (not ``bool``) is coerced; floats stringify to something (``"123.0"``) that can never
    match, so they get a load-time warning instead.

    ``bool`` is an ``int`` subclass but never a valid id; floats and other types stringify to something
    (``"123.0"``) that can never equal an inbound id — recreating the silent no-match this exists to fix —
    so they are passed through with a load-time warning instead of being silently "fixed" (#86470).
    """
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    logger.warning(
        "Profile route discriminator %r (type %s) can never match an inbound "
        "id — quote it in config.yaml (e.g. chat_id: \"%s\").",
        value, type(value).__name__, value,
    )
    return str(value)


def _warn_unmatchable_patterns(route: ProfileRoute) -> None:
    """Load-time warning for a pattern-shaped discriminator that cannot compile — it never matches.

    Match-time failures stay silent (a typo must not log per message); a literal value of the same
    text still matches normally, so only a genuine broken pattern warns (#109676).
    """
    for field, value in (("guild_id", route.guild_id), ("chat_id", route.chat_id), ("thread_id", route.thread_id)):
        if value and _is_pattern(value) and _compile_pattern(value) is None:
            logger.warning(
                "Profile route %s: %s %r is not a valid regex; it can never match", route.name, field, value
            )


def parse_profile_routes(raw: Optional[List[Dict[str, Any]]]) -> List[ProfileRoute]:
    """Parse profile_routes from config.yaml, sorted most-specific-first."""
    if not raw:
        return []
    routes: List[ProfileRoute] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name", "")
        platform = entry.get("platform", "")
        profile = entry.get("profile", "")
        if not platform or not profile:
            logger.warning("Skipping profile route %s: missing platform or profile", name)
            continue
        # Validate profile name to prevent path traversal (lazy import: cycle).
        try:
            from hermes_cli.profiles import normalize_profile_name, validate_profile_name

            profile = normalize_profile_name(profile)
            validate_profile_name(profile)
        except (ValueError, ImportError):
            logger.warning("Skipping profile route %s: invalid profile name %r", name, profile)
            continue
        routes.append(ProfileRoute(
            name=name, platform=platform, profile=profile,
            guild_id=_coerce_route_id(entry.get("guild_id")),
            chat_id=_coerce_route_id(entry.get("chat_id")),
            thread_id=_coerce_route_id(entry.get("thread_id")),
            enabled=entry.get("enabled", True),
            bot_profile=_bot_profile_key(entry.get("bot_profile")),
        ))
        _warn_unmatchable_patterns(routes[-1])
    routes.sort(key=lambda r: r.specificity, reverse=True)
    logger.debug("Loaded %d profile routes (most-specific-first)", len(routes))
    return routes


def match_profile_route(
    routes: List[ProfileRoute], platform: str, guild_id: Optional[str] = None, chat_id: Optional[str] = None,
    thread_id: Optional[str] = None, parent_chat_id: Optional[str] = None,
    adapter_profile: Optional[str] = None,
) -> Optional[ProfileRoute]:
    """Return the first (most specific) matching route, or None."""
    for route in routes:
        if route.matches(platform, guild_id=guild_id, chat_id=chat_id, thread_id=thread_id,
                         parent_chat_id=parent_chat_id, adapter_profile=adapter_profile):
            return route
    return None
