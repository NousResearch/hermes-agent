"""Agent cache management for the gateway.

Extracted from gateway/run.py to reduce monolith size.  The CacheManager
owns the OrderedDict cache, its lock, and the eviction/sweep logic.
GatewayRunner instantiates one in __init__ and delegates cache management
calls to it.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)

# --- Cache tuning constants -----------------------------------------------
# Bounds the per-session AIAgent cache to prevent unbounded growth in
# long-lived gateways (each AIAgent holds LLM clients, tool schemas,
# memory providers, etc.).  LRU order + idle TTL eviction are enforced
# from enforce_cap() and the session expiry watcher.
_AGENT_CACHE_MAX_SIZE = 128
_AGENT_CACHE_IDLE_TTL_SECS = 3600.0  # evict agents idle for >1h


class CacheManager:
    """Manages the per-session AIAgent cache with LRU eviction and idle sweeps.

    Parameters
    ----------
    gateway : Any
        The GatewayRunner instance.  Used to access _running_agents and to
        call _cleanup_agent_resources via the cleanup_fn callback.
    max_size : int
        Maximum number of cached agents before LRU eviction kicks in.
    idle_ttl : float
        Seconds of inactivity after which an idle cached agent is evicted.
    cleanup_fn : callable
        Called with an agent instance to perform full resource teardown
        (memory provider shutdown, tool resource close).
    """

    def __init__(
        self,
        gateway: Any,
        max_size: int = _AGENT_CACHE_MAX_SIZE,
        idle_ttl: float = _AGENT_CACHE_IDLE_TTL_SECS,
        cleanup_fn: Optional[Callable[[Any], None]] = None,
    ) -> None:
        self.gateway = gateway
        self.max_size = max_size
        self.idle_ttl = idle_ttl
        self.cleanup_fn = cleanup_fn
        self.cache: "OrderedDict[str, tuple]" = OrderedDict()
        self.lock = threading.Lock()

    # -- Public API ---------------------------------------------------------

    def get(self, session_key: str) -> Optional[Any]:
        """Return the cached AIAgent for *session_key*, or None."""
        with self.lock:
            entry = self.cache.get(session_key)
            if entry is None:
                return None
            # Touch LRU order.
            self.cache.move_to_end(session_key)
            agent = entry[0] if isinstance(entry, tuple) else entry
            return agent

    def put(self, session_key: str, agent: Any, metadata: Any = None) -> None:
        """Store *agent* in the cache and enforce capacity limits."""
        with self.lock:
            self.cache[session_key] = (agent, metadata)
            self.cache.move_to_end(session_key)
            self._enforce_cap_locked()

    def evict(self, session_key: str) -> None:
        """Remove a cached agent for *session_key*."""
        with self.lock:
            self.cache.pop(session_key, None)

    def reset_turn_state(self, agent: Any, interrupt_depth: int = 0) -> None:
        """Reset per-turn state on a cached agent before a new turn starts.

        Must be called after the agent is retrieved from the cache and
        before its first API call of the new turn (#9051).
        """
        if interrupt_depth == 0:
            agent._last_activity_ts = time.time()
            agent._last_activity_desc = "starting new turn (cached)"
        agent._api_call_count = 0

    def soft_release(self, agent: Any) -> None:
        """Soft cleanup for cache-evicted agents — preserves session tool state.

        Distinct from full teardown because a cache-evicted session may
        resume at any time — its terminal sandbox, browser daemon, and
        tracked bg processes must outlive the Python AIAgent instance so
        the next agent built for the same task_id inherits them.
        """
        if agent is None:
            return
        try:
            if hasattr(agent, "release_clients"):
                agent.release_clients()
            elif self.cleanup_fn is not None:
                # Older agent instance — fall back to the legacy
                # full-close path.
                self.cleanup_fn(agent)
        except Exception:
            pass

    def sweep_idle(self) -> int:
        """Evict cached agents whose AIAgent has been idle > idle_ttl.

        Returns the number of entries evicted.  Resource cleanup is
        scheduled on daemon threads.

        Agents currently in gateway._running_agents are SKIPPED — tearing
        down an active turn's clients mid-flight would crash the request.
        """
        now = time.time()
        to_evict: List[tuple] = []
        running_ids = self._running_agent_ids()

        with self.lock:
            for key, entry in list(self.cache.items()):
                agent = entry[0] if isinstance(entry, tuple) and entry else None
                if agent is None:
                    continue
                if id(agent) in running_ids:
                    continue  # mid-turn — don't tear it down
                last_activity = getattr(agent, "_last_activity_ts", None)
                if last_activity is None:
                    continue
                if (now - last_activity) > self.idle_ttl:
                    to_evict.append((key, agent))
            for key, _ in to_evict:
                self.cache.pop(key, None)

        for key, agent in to_evict:
            logger.info(
                "Agent cache idle-TTL evict: session=%s (idle=%.0fs)",
                key, now - getattr(agent, "_last_activity_ts", now),
            )
            threading.Thread(
                target=self.soft_release,
                args=(agent,),
                daemon=True,
                name=f"agent-cache-idle-{key[:24]}",
            ).start()

        return len(to_evict)

    # -- Internals ----------------------------------------------------------

    def _running_agent_ids(self) -> set:
        """Return a set of id() for agents currently mid-turn."""
        running = getattr(self.gateway, "_running_agents", {})
        pending = getattr(self.gateway, "_AGENT_PENDING_SENTINEL", None)
        return {
            id(a)
            for a in running.values()
            if a is not None and a is not pending
        }

    def _enforce_cap_locked(self) -> None:
        """Evict oldest cached agents when cache exceeds max_size.

        Must be called with self.lock held.  Resource cleanup is scheduled
        on a daemon thread so the caller doesn't block on slow teardown
        while holding the cache lock.

        Agents currently in _running_agents are SKIPPED — their clients,
        terminal sandboxes, background processes, and child subagents
        are all in active use by the running turn.  If every candidate in
        the LRU order is active, we simply leave the cache over the cap;
        it will be re-checked on the next insert.
        """
        if not hasattr(self.cache, "move_to_end"):
            return  # plain dict lacks the arg

        running_ids = self._running_agent_ids()
        excess = max(0, len(self.cache) - self.max_size)
        evict_plan: List[tuple] = []
        if excess > 0:
            ordered_keys = list(self.cache.keys())
            for key in ordered_keys[:excess]:
                entry = self.cache.get(key)
                agent = entry[0] if isinstance(entry, tuple) and entry else None
                if agent is not None and id(agent) in running_ids:
                    continue  # active mid-turn; don't evict, don't substitute
                evict_plan.append((key, agent))

        for key, _ in evict_plan:
            self.cache.pop(key, None)

        remaining_over_cap = len(self.cache) - self.max_size
        if remaining_over_cap > 0:
            logger.warning(
                "Agent cache over cap (%d > %d); %d excess slot(s) held by "
                "mid-turn agents — will re-check on next insert.",
                len(self.cache), self.max_size, remaining_over_cap,
            )

        for key, agent in evict_plan:
            logger.info(
                "Agent cache at cap; evicting LRU session=%s (cache_size=%d)",
                key, len(self.cache),
            )
            if agent is not None:
                threading.Thread(
                    target=self.soft_release,
                    args=(agent,),
                    daemon=True,
                    name=f"agent-cache-evict-{key[:24]}",
                ).start()


# Re-export constants for backward compatibility.
__all__ = [
    "CacheManager",
    "_AGENT_CACHE_MAX_SIZE",
    "_AGENT_CACHE_IDLE_TTL_SECS",
]
