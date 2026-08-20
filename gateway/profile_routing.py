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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import logging

logger = logging.getLogger(__name__)


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
        """
        if not self.enabled:
            return False
        if self.platform != platform:
            return False
        if self.thread_id and self.thread_id != thread_id:
            return False
        if self.chat_id and self.chat_id != chat_id and self.chat_id != parent_chat_id:
            return False
        if self.guild_id and self.guild_id != guild_id:
            return False
        return True


def _coerce_route_id(key: str, value: Any, name: str) -> Optional[str]:
    """Coerce a route discriminator to ``str``, warning on non-string input.

    YAML parses unquoted numeric IDs (e.g. Discord snowflakes) as ``int``,
    but the ``SessionSource`` fields they are matched against are always
    ``str`` — an ``int`` route would silently never match and messages
    would fall through to the default profile. Normalize here so the
    comparison is type-consistent, and surface the misconfiguration.

    Only ``int`` is coerced (the legitimate YAML-numeric-ID case). Other
    non-string types (float, bool, ...) are almost always mis-parsed YAML:
    stringifying them would produce a value that can never match a string
    discriminator (``123.0`` vs ``"123"``) — recreating the silent-no-match
    this helper exists to fix. They are still returned as strings (never
    None: a None discriminator makes the route unconstrained and would
    match everything), but the warning says outright that the value can
    never match.
    """
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):  # bool is an int subclass — check first
        logger.warning(
            "Profile route %r: %s=%r is a boolean and can never match — "
            "quote the value in config.yaml (e.g. `%s: \"%s\"`).",
            name, key, value, key, value,
        )
        return str(value)
    if isinstance(value, int):
        logger.warning(
            "Profile route %r: %s=%r is not a string — coercing to %r. "
            "Quote the value in config.yaml (e.g. `%s: \"%s\"`) to be explicit.",
            name, key, value, str(value), key, value,
        )
        return str(value)
    logger.warning(
        "Profile route %r: %s=%r (type %s) can never match a string route "
        "discriminator — quote it in config.yaml (e.g. `%s: \"%s\"`).",
        name, key, value, type(value).__name__, key, value,
    )
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
                guild_id=_coerce_route_id("guild_id", entry.get("guild_id"), name),
                chat_id=_coerce_route_id("chat_id", entry.get("chat_id"), name),
                thread_id=_coerce_route_id("thread_id", entry.get("thread_id"), name),
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
