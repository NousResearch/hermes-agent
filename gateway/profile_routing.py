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

        - name: admin-dm
          platform: whatsapp
          chat_id: "15551234567@s.whatsapp.net"
          profile: ops

WhatsApp chat ids are matched through the bridge's phone/LID alias mapping,
so the route above also matches that person when WhatsApp delivers them as
``<lid>@lid`` — one route per human, not one per id form. Group JIDs
(``@g.us``) are matched as exact strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import logging

logger = logging.getLogger(__name__)


class ProfileRouteRejected(RuntimeError):
    """An explicit route matched a profile this gateway does not serve."""


# Platforms that can surface the same person under more than one chat id.
# WhatsApp delivers a DM as either the phone JID (``<msisdn>@s.whatsapp.net``)
# or the LID (``<lid>@lid``) for the same human, so an operator's msisdn-keyed
# route silently misses the LID form and the message lands on the default
# profile instead. Authorization already collapses both forms through
# ``gateway.whatsapp_identity``; routing has to read the same mapping or the
# two gates disagree about who a sender is.
_ALIASED_PLATFORMS = frozenset({"whatsapp"})

# Only *user* identities alias. A group JID (``@g.us``) and a broadcast list
# live in a different id space, and normalizing them to their bare numeric core
# would let a group id collide with a LID that happens to share digits — so
# they stay exact-string comparisons.
_WHATSAPP_USER_DOMAINS = frozenset({"s.whatsapp.net", "lid", "c.us"})


def _is_whatsapp_user_id(value: str) -> bool:
    """True for a WhatsApp id that identifies a person rather than a group."""
    if not value:
        return False
    if "@" not in value:
        # A bare phone number, which is how operators usually write a route.
        return True
    return value.rsplit("@", 1)[1].lower() in _WHATSAPP_USER_DOMAINS


def whatsapp_chat_id_aliases(chat_id: Optional[str]) -> frozenset:
    """Every normalized id the given WhatsApp chat id may also appear as.

    Empty for a non-aliasing id (group/broadcast) or when the mapping cannot
    be read. Only the *inbound* id needs expanding: the returned set holds
    normalized identifiers, so a route is matched by normalizing its declared
    chat id and testing membership — which keeps this to one mapping walk per
    message instead of one per route.
    """
    value = str(chat_id or "")
    if not _is_whatsapp_user_id(value):
        return frozenset()
    try:
        from gateway.whatsapp_identity import expand_whatsapp_aliases

        return frozenset(expand_whatsapp_aliases(value))
    except Exception:
        # Routing must survive a broken mapping, but silently falling back to
        # exact-string matching is what makes a mis-routed sender so hard to
        # diagnose. An unreadable mapping *file* is already reported once per
        # file by whatsapp_identity; what reaches here is the unexpected.
        logger.warning(
            "WhatsApp alias expansion failed for chat_id %r; profile routes "
            "written in the other id form (phone vs LID) will not match",
            value,
            exc_info=True,
        )
        return frozenset()


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
        chat_id_aliases: Optional[frozenset] = None,
    ) -> bool:
        """Return True if this route matches the given source fields.

        All configured discriminators are matched conjunctively (AND): every
        discriminator that the route declares must hold. ``chat_id`` supports
        hierarchical matching for Discord forums/threads:
        - Direct channel match: chat_id == route.chat_id
        - Thread in channel: parent_chat_id == route.chat_id
        A route declaring both ``guild_id`` and ``chat_id`` requires both to
        match (a chat match alone does not satisfy a guild constraint).

        On WhatsApp an exact-string miss falls back to alias matching, so a
        route keyed on a phone number still matches the same person arriving
        under their LID. ``chat_id_aliases`` lets a caller matching many
        routes resolve that set once; when omitted it is resolved here.
        """
        if not self.enabled:
            return False
        if self.platform != platform:
            return False
        if self.thread_id and self.thread_id != thread_id:
            return False
        if self.chat_id and self.chat_id != chat_id and self.chat_id != parent_chat_id:
            if not self._chat_id_matches_alias(platform, chat_id, chat_id_aliases):
                return False
        if self.guild_id and self.guild_id != guild_id:
            return False
        return True

    def _chat_id_matches_alias(
        self,
        platform: str,
        chat_id: Optional[str],
        chat_id_aliases: Optional[frozenset],
    ) -> bool:
        """True when the inbound chat id is a known alias of this route's.

        ``parent_chat_id`` is not consulted: it carries a Discord thread's
        parent channel, and WhatsApp — the only aliasing platform — has no
        parent-chat concept.
        """
        if platform not in _ALIASED_PLATFORMS:
            return False
        if not _is_whatsapp_user_id(str(self.chat_id or "")):
            return False
        if chat_id_aliases is None:
            chat_id_aliases = whatsapp_chat_id_aliases(chat_id)
        if not chat_id_aliases:
            return False

        from gateway.whatsapp_identity import normalize_whatsapp_identifier

        normalized = normalize_whatsapp_identifier(str(self.chat_id))
        return bool(normalized) and normalized in chat_id_aliases


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
                guild_id=entry.get("guild_id"),
                chat_id=entry.get("chat_id"),
                thread_id=entry.get("thread_id"),
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
    # Resolved once for the whole route list: the mapping walk touches the
    # bridge's session directory, and this runs on the inbound path for every
    # routed message. Non-aliasing platforms never pay for it.
    chat_id_aliases = (
        whatsapp_chat_id_aliases(chat_id)
        if platform in _ALIASED_PLATFORMS
        else frozenset()
    )
    for route in routes:
        if route.matches(
            platform,
            guild_id=guild_id,
            chat_id=chat_id,
            thread_id=thread_id,
            parent_chat_id=parent_chat_id,
            chat_id_aliases=chat_id_aliases,
        ):
            return route
    return None
