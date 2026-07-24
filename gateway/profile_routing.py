"""Profile-based routing for the gateway with hierarchical matching.

Allows a single Hermes instance to route specific Discord guilds/channels/threads
to different profiles — each with their own model, tools, memory, and persona.

Matching priority (most specific first):
  1. platform + chat_id + thread_id (exact thread)  — specificity 14
  2. platform + chat_id (channel route)             — specificity 6
  3. platform + guild_id (guild/server route)       — specificity 2
  4. No match                                       → default profile

Parent-chain matching:
For Discord threads and forum posts, ``parent_chat_id`` carries the
direct parent (the channel for a thread, the forum channel for a post).
Routes keyed on a channel match both direct messages and messages in
any thread/post whose parent is that channel.

Configuration (config.yaml):

    gateway:
      profile_routes:
        - name: server-default
          platform: discord
          guild_id: "YOUR_GUILD_ID"
          profile: server-profile

        - name: special-channel
          platform: discord
          guild_id: "YOUR_GUILD_ID"
          chat_id: "YOUR_CHANNEL_ID"
          profile: channel-profile

        - name: thread-route
          platform: discord
          chat_id: "YOUR_CHANNEL_ID"
          thread_id: "YOUR_THREAD_ID"
          profile: thread-profile

        - name: owner-whatsapp
          platform: whatsapp
          chat_id: "15551234567"   # phone, JID, or LID — all equivalent
          profile: owner
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import logging

logger = logging.getLogger(__name__)

# Baileys and Cloud share phone/JID/LID identity rules. Other platforms keep
# exact string compare so Telegram numeric ids and Discord snowflakes stay
# unchanged.
_WHATSAPP_IDENTITY_PLATFORMS = {"whatsapp", "whatsapp_cloud"}
_WHATSAPP_NON_USER_SUFFIXES = ("@g.us", "@broadcast", "@newsletter")


def _is_whatsapp_non_user_chat(chat_id: Optional[str]) -> bool:
    """True for group / broadcast / newsletter JIDs — not a sender identity."""
    if not chat_id:
        return False
    cid = str(chat_id).strip().lower()
    return any(cid.endswith(suffix) for suffix in _WHATSAPP_NON_USER_SUFFIXES)


def _whatsapp_user_chat_ids_match(platform: str, left: Optional[str], right: Optional[str]) -> bool:
    """True when two WhatsApp *user* chat_ids refer to the same person.

    Reuses :func:`gateway.whatsapp_identity.expand_whatsapp_aliases` so a
    bare phone number, a ``@s.whatsapp.net`` JID, and a ``@lid`` LID collapse
    to one identity — the same helper session keys and adapter allowlists
    already use. Group/broadcast JIDs are excluded: those are chats, not
    senders. Returns False for non-WhatsApp platforms (exact match only).
    """
    if (platform or "").strip().lower() not in _WHATSAPP_IDENTITY_PLATFORMS:
        return False
    if not left or not right:
        return False
    if _is_whatsapp_non_user_chat(left) or _is_whatsapp_non_user_chat(right):
        return False
    from gateway.whatsapp_identity import expand_whatsapp_aliases

    left_aliases = expand_whatsapp_aliases(str(left))
    if not left_aliases:
        return False
    return bool(left_aliases & expand_whatsapp_aliases(str(right)))


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
        s = 0
        if self.guild_id:
            s += 2
        if self.chat_id:
            s += 4
        if self.thread_id:
            s += 8
        return s

    def matches(
        self,
        platform: str,
        guild_id: Optional[str] = None,
        chat_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        parent_chat_id: Optional[str] = None,
    ) -> bool:
        """Return True if this route matches the given source fields.

        All configured discriminators are matched conjunctively (AND): every
        discriminator that the route declares must hold. ``chat_id`` supports
        hierarchical matching for Discord forums/threads:
        - Direct channel match: chat_id == route.chat_id
        - Thread in channel: parent_chat_id == route.chat_id
        A route declaring both ``guild_id`` and ``chat_id`` requires both to
        match (a chat match alone does not satisfy a guild constraint).

        WhatsApp / WhatsApp Cloud ``chat_id`` also matches across user-identity
        forms (bare number, JID, LID) after the exact-string check. Exact
        matches always win first, so existing configs keep working. Groups
        (``@g.us``) and broadcasts stay exact-only.
        """
        if not self.enabled:
            return False
        if self.platform != platform:
            return False
        if self.thread_id and self.thread_id != thread_id:
            return False
        if self.chat_id and self.chat_id != chat_id and self.chat_id != parent_chat_id:
            if not (
                _whatsapp_user_chat_ids_match(platform, self.chat_id, chat_id)
                or _whatsapp_user_chat_ids_match(platform, self.chat_id, parent_chat_id)
            ):
                return False
        if self.guild_id and self.guild_id != guild_id:
            return False
        return True


def _coerce_route_id(value: Any) -> Optional[str]:
    """Normalize a route discriminator to str for strict equality matching.

    PyYAML loads unquoted numeric IDs (Discord snowflakes, Telegram negative
    chat ids) as ``int``. Inbound ``SessionSource`` fields are always ``str``
    via ``build_source``, so leaving ints here makes ``matches()`` fail silently.
    """
    if value is None:
        return None
    return str(value)


def parse_profile_routes(raw: Optional[List[Dict[str, Any]]]) -> List[ProfileRoute]:
    """Parse profile_routes from config.yaml into ProfileRoute objects.

    Returns routes sorted by specificity (most specific first).
    """
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
            logger.warning(
                "Skipping profile route %s: missing platform or profile",
                name,
            )
            continue
        # Validate profile name to prevent path traversal. Lazy import avoids a
        # circular dependency at module load time.
        try:
            from hermes_cli.profiles import (
                normalize_profile_name,
                validate_profile_name,
            )
            profile = normalize_profile_name(profile)
            validate_profile_name(profile)
        except (ValueError, ImportError):
            logger.warning("Skipping profile route %s: invalid profile name %r", name, profile)
            continue
        routes.append(
            ProfileRoute(
                name=name,
                platform=platform,
                profile=profile,
                guild_id=_coerce_route_id(entry.get("guild_id")),
                chat_id=_coerce_route_id(entry.get("chat_id")),
                thread_id=_coerce_route_id(entry.get("thread_id")),
                enabled=entry.get("enabled", True),
            )
        )
    # Sort: most specific first so the first match wins.
    routes.sort(key=lambda r: r.specificity, reverse=True)
    logger.debug("Loaded %d profile routes (most-specific-first)", len(routes))
    return routes


def match_profile_route(
    routes: List[ProfileRoute],
    platform: str,
    guild_id: Optional[str] = None,
    chat_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    parent_chat_id: Optional[str] = None,
) -> Optional[ProfileRoute]:
    """Return the best-matching route, or None for no match."""
    for route in routes:
        if route.matches(platform, guild_id=guild_id, chat_id=chat_id, thread_id=thread_id, parent_chat_id=parent_chat_id):
            return route
    return None
