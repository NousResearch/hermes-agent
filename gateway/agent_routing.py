"""Inbound-message -> agent_id resolver.

The gateway calls ``resolve_agent_id(source, routes, default)`` once per
inbound message before ``build_session_key``.  It walks the declarative
routes list (first match wins) and returns the matched ``agent`` value,
or ``default`` if nothing matches.

A separate plugin hook (``select_agent``) can override the route result;
that wiring lives in the adapter layer (see
``BasePlatformAdapter._attach_agent_id``).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .session import SessionSource

logger = logging.getLogger(__name__)


# Keys we know how to match against on the source.  Adding a new key is
# safe — it simply gets ignored on sources that lack the attribute.
_SUPPORTED_KEYS = (
    "platform",
    "chat_id",
    "thread_id",
    "user_id",
    "user_id_alt",
    "chat_type",
    "guild_id",
    "parent_chat_id",
    "topic_id",
)


def _source_value(source: SessionSource, key: str) -> Optional[str]:
    """Read *key* from *source*, normalised to a string for comparison."""
    if key == "platform":
        return source.platform.value if source.platform is not None else None
    val = getattr(source, key, None)
    if val is None:
        return None
    return str(val)


def _route_matches(match: Dict[str, Any], source: SessionSource) -> bool:
    """Return True iff every key in *match* equals the corresponding
    attribute on *source*.  Unknown match keys cause the route to be
    skipped (logged at DEBUG) — silently ignoring would let typos bind
    every message."""
    if not match:
        return False  # An empty match block matches nothing — guard rail.
    for key, expected in match.items():
        if key not in _SUPPORTED_KEYS:
            logger.debug(
                "agent_routing: ignoring route with unsupported match key %r", key,
            )
            return False
        actual = _source_value(source, key)
        if actual is None:
            return False
        if str(expected) != actual:
            return False
    return True


def resolve_agent_id(
    source: SessionSource,
    routes: List[Dict[str, Any]],
    default: Optional[str] = None,
) -> Optional[str]:
    """Walk *routes* in declared order; first match wins.

    Each route is a dict ``{"match": {...}, "agent": "<id>"}``.  Returns
    the matched agent id, or *default* if no route matched.  Returns
    ``None`` when nothing matched and *default* is ``None`` — the caller
    (adapter) can then fall back to a separate plugin hook or its own
    default.
    """
    for route in routes or []:
        if not isinstance(route, dict):
            continue
        match = route.get("match")
        agent = route.get("agent")
        if not isinstance(agent, str) or not agent.strip():
            continue
        if not isinstance(match, dict):
            continue
        if _route_matches(match, source):
            return agent.strip()
    return default
