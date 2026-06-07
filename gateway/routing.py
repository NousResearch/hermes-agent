"""Platform-agnostic profile routing table (Tier-2).

Resolves an inbound SessionSource to a target profile name, or None (host).
Exact-wins: thread/channel > guild > platform > default. A thread with no
thread-specific route inherits its parent channel's route.
"""

from __future__ import annotations

from gateway.session import SessionSource


def _route_score(route: dict, source: SessionSource) -> int | None:
    """Specificity score if *route* matches *source*, else None (higher = more specific)."""
    score = 0
    chat = str(source.chat_id or "")
    parent = str(source.parent_chat_id or "")

    if "platform" in route:
        if str(route["platform"]) != source.platform.value:
            return None
        score += 1

    # channel_id is a Discord-ergonomic alias for chat_id; a thread inherits its
    # parent channel's route, so match either the chat or its parent.
    route_chat = route.get("chat_id", route.get("channel_id"))
    if route_chat is not None:
        if str(route_chat) not in (chat, parent):
            return None
        score += 4

    if "thread_id" in route:
        if str(route["thread_id"]) != str(source.thread_id or ""):
            return None
        score += 8

    if "guild_id" in route:
        if str(route["guild_id"]) != str(source.guild_id or ""):
            return None
        score += 2

    if "user_id" in route:
        if str(route["user_id"]) != str(source.user_id_alt or source.user_id or ""):
            return None
        score += 2

    return score


def resolve_profile_route(profile_routing: dict | None, source: SessionSource) -> str | None:
    if not profile_routing:
        return None
    best_score, best_profile = -1, None
    for route in profile_routing.get("routes") or []:
        if not isinstance(route, dict):
            continue
        score = _route_score(route, source)
        if score is not None and score > best_score:
            best_score, best_profile = score, route.get("profile")
    return best_profile or profile_routing.get("default")
