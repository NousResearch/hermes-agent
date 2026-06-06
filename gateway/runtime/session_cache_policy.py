"""Session-scoped AIAgent cache policy helpers for the Hermes gateway.

This module contains the behavior-only decisions for the gateway's per-session
agent cache.  It deliberately does not perform resource cleanup, start threads,
or mutate gateway runner state beyond the caller-owned cache mutations described
by returned eviction plans.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

AGENT_CACHE_MAX_SIZE_DEFAULT = 128
AGENT_CACHE_IDLE_TTL_SECS_DEFAULT = 3600.0


def is_lru_cache_cap_enforceable(cache: Any) -> bool:
    """Return whether ``cache`` supports the LRU operations used by the gateway.

    The real gateway uses ``OrderedDict``.  Some legacy tests replace it with a
    plain ``dict``; preserving the old behavior means treating those fixtures as
    non-enforceable no-ops for cap eviction.
    """

    return cache is not None and hasattr(cache, "move_to_end")


def cached_agent_from_entry(entry: Any) -> Any:
    """Return the cached agent object from a gateway cache entry, if present."""

    if isinstance(entry, tuple) and entry:
        return entry[0]
    return None


def active_agent_ids(
    running_agents: Mapping[Any, Any] | None,
    *,
    pending_sentinel: Any = None,
) -> set[int]:
    """Return ``id(agent)`` values for agents that are actively mid-turn."""

    if not running_agents:
        return set()
    return {
        id(agent)
        for agent in running_agents.values()
        if agent is not None and agent is not pending_sentinel
    }


def plan_lru_cache_evictions(
    cache: MutableMapping[Any, Any] | None,
    *,
    running_agents: Mapping[Any, Any] | None = None,
    max_size: int = AGENT_CACHE_MAX_SIZE_DEFAULT,
    pending_sentinel: Any = None,
) -> tuple[list[tuple[Any, Any]], int]:
    """Plan LRU cache-cap evictions without doing cleanup side effects.

    Returns ``(evict_plan, remaining_over_cap)`` where ``evict_plan`` contains
    ``(key, agent)`` pairs.  It preserves the gateway's safety rule: only entries
    in the LRU excess window are candidates, and active mid-turn agents are
    skipped without evicting newer substitute entries.
    """

    if cache is None or not is_lru_cache_cap_enforceable(cache):
        return [], 0

    excess = max(0, len(cache) - int(max_size))
    evict_plan: list[tuple[Any, Any]] = []
    if excess <= 0:
        return evict_plan, 0

    running_ids = active_agent_ids(running_agents, pending_sentinel=pending_sentinel)
    ordered_keys = list(cache.keys())
    for key in ordered_keys[:excess]:
        agent = cached_agent_from_entry(cache.get(key))
        if agent is not None and id(agent) in running_ids:
            continue
        evict_plan.append((key, agent))

    remaining_over_cap = len(cache) - len(evict_plan) - int(max_size)
    return evict_plan, max(0, remaining_over_cap)


def apply_cache_eviction_plan(
    cache: MutableMapping[Any, Any] | None,
    evict_plan: list[tuple[Any, Any]],
) -> None:
    """Remove planned keys from ``cache``; cleanup remains the caller's job."""

    if cache is None:
        return
    for key, _agent in evict_plan:
        cache.pop(key, None)


def plan_idle_cache_evictions(
    cache: Mapping[Any, Any] | None,
    *,
    running_agents: Mapping[Any, Any] | None = None,
    idle_ttl_secs: float = AGENT_CACHE_IDLE_TTL_SECS_DEFAULT,
    now: float,
    pending_sentinel: Any = None,
) -> list[tuple[Any, Any]]:
    """Plan idle-TTL cache evictions without mutating state or doing cleanup."""

    if cache is None:
        return []

    running_ids = active_agent_ids(running_agents, pending_sentinel=pending_sentinel)
    to_evict: list[tuple[Any, Any]] = []
    for key, entry in list(cache.items()):
        agent = cached_agent_from_entry(entry)
        if agent is None:
            continue
        if id(agent) in running_ids:
            continue
        last_activity = getattr(agent, "_last_activity_ts", None)
        if last_activity is None:
            continue
        if (now - last_activity) > float(idle_ttl_secs):
            to_evict.append((key, agent))
    return to_evict
