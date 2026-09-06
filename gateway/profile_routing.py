"""Profile-based routing: route guilds/channels/threads to different profiles.

Matching priority, most specific first (``gateway.profile_routes`` in config.yaml):
platform + chat_id + thread_id (14) → platform + chat_id (6) → platform + guild_id (2)
→ default profile. For Discord threads/forum posts ``parent_chat_id`` carries the
direct parent, so a channel route also matches any thread/post under it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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

    @property
    def specificity(self) -> int:
        """Higher value = more specific match."""
        return 2 * bool(self.guild_id) + 4 * bool(self.chat_id) + 8 * bool(self.thread_id)

    def matches(
        self, platform: str, guild_id: Optional[str] = None, chat_id: Optional[str] = None,
        thread_id: Optional[str] = None, parent_chat_id: Optional[str] = None,
    ) -> bool:
        """True if every discriminator the route declares holds (AND).

        ``chat_id`` matches the channel directly or as the parent of a thread/forum post; WhatsApp
        ``chat_id`` also matches across number/JID/LID after the exact check (groups/broadcasts stay exact-only).
        """
        if not self.enabled or self.platform != platform:
            return False
        if self.thread_id and self.thread_id != thread_id:
            return False
        if (
            self.chat_id
            and self.chat_id not in (chat_id, parent_chat_id)
            and not _whatsapp_user_chat_ids_match(platform, self.chat_id, chat_id)
            and not _whatsapp_user_chat_ids_match(platform, self.chat_id, parent_chat_id)
        ):
            return False
        return not (self.guild_id and self.guild_id != guild_id)


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
        ))
    routes.sort(key=lambda r: r.specificity, reverse=True)
    logger.debug("Loaded %d profile routes (most-specific-first)", len(routes))
    return routes


def match_profile_route(
    routes: List[ProfileRoute], platform: str, guild_id: Optional[str] = None, chat_id: Optional[str] = None,
    thread_id: Optional[str] = None, parent_chat_id: Optional[str] = None,
) -> Optional[ProfileRoute]:
    """Return the first (most specific) matching route, or None."""
    for route in routes:
        if route.matches(platform, guild_id=guild_id, chat_id=chat_id, thread_id=thread_id, parent_chat_id=parent_chat_id):
            return route
    return None
