"""Memory-provider continuity across api_server requests (#120116).

The api_server adapter builds a fresh ``AIAgent`` per request (per-request callbacks, model
route, ephemeral prompt), unlike the messaging platforms, whose cached agent — and with it the
memory provider — lives for the whole session. External providers deliver recall as the
PREVIOUS turn's background prefetch held on the provider instance, so a provider that is
re-initialised per request never has anything to inject, and each init re-runs the provider's
startup (for an embedded daemon: a restart that also kills the retain still in flight).

This registry keeps one initialised ``MemoryManager`` per (profile home, session id): a request
checks the session's manager out before building its agent (``AIAgent(memory_manager=...)``
skips provider init) and checks it back in when the turn ends. Check-out is exclusive, so two
concurrent requests on one session never share a manager; the loser's fresh manager is shut down
when it checks in behind the winner. Idle entries and LRU overflow are shut down under the owning
profile's scope, like the gateway agent cache's eviction.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from contextlib import nullcontext, suppress
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ApiServerMemorySessions:
    """Session-keyed ``MemoryManager`` registry with exclusive check-out/check-in."""

    def __init__(self, *, max_size: Optional[int] = None, idle_ttl_secs: Optional[float] = None) -> None:
        self._entries: "OrderedDict[Tuple[str, str], Tuple[Any, Optional[Path], float]]" = OrderedDict()
        self._lock = threading.Lock()
        self._max_size = max_size
        self._idle_ttl_secs = idle_ttl_secs

    # -- bounds (same knobs as the gateway agent cache, resolved lazily) -------------------------

    def _bounds(self) -> Tuple[int, float]:
        if self._max_size is None or self._idle_ttl_secs is None:
            from gateway.run import _AGENT_CACHE_IDLE_TTL_SECS, _AGENT_CACHE_MAX_SIZE, _load_gateway_config
            from gateway.agent_cache_pressure import resolve_agent_cache_bounds
            configured = None
            with suppress(Exception):
                configured = resolve_agent_cache_bounds(_load_gateway_config())
            if self._max_size is None:
                self._max_size = getattr(configured, "max_size", None) or _AGENT_CACHE_MAX_SIZE
            if self._idle_ttl_secs is None:
                self._idle_ttl_secs = getattr(configured, "idle_ttl_secs", None) or _AGENT_CACHE_IDLE_TTL_SECS
        return self._max_size, self._idle_ttl_secs

    @staticmethod
    def _owner_home() -> Tuple[str, Optional[Path]]:
        """(registry key, profile home to re-enter on eviction) for the CURRENT scope. Callers run
        inside ``_profile_scope`` (or a single-profile gateway), so the ambient home is the owner's."""
        from hermes_constants import get_hermes_home, hermes_home_key
        home = Path(get_hermes_home())
        return hermes_home_key(home), home

    # -- check-out / check-in ----------------------------------------------------------------

    def checkout(self, session_id: Optional[str]) -> Optional[Any]:
        """The manager a previous request on ``session_id`` checked in, or None (build a new one)."""
        if not session_id:
            return None
        home_key, _home = self._owner_home()
        with self._lock:
            entry = self._entries.pop((home_key, session_id), None)
        return entry[0] if entry else None

    def checkin(self, agent: Any) -> None:
        """Park ``agent``'s manager under the session the turn ended on (``agent.session_id`` carries a
        mid-turn compression rotation) and shut down whatever this displaces or has gone idle."""
        manager = getattr(agent, "_memory_manager", None)
        session_id = str(getattr(agent, "session_id", "") or "")
        if manager is None or not session_id:
            return
        home_key, home = self._owner_home()
        max_size, idle_ttl = self._bounds()
        now = time.monotonic()
        doomed: List[Tuple[Any, Optional[Path]]] = []
        with self._lock:
            displaced = self._entries.pop((home_key, session_id), None)
            if displaced is not None and displaced[0] is not manager:
                doomed.append((displaced[0], displaced[1]))
            self._entries[(home_key, session_id)] = (manager, home, now)
            for key, (mgr, owner, last_used) in list(self._entries.items()):
                if mgr is manager:
                    continue
                if now - last_used > idle_ttl or len(self._entries) > max_size:
                    del self._entries[key]
                    doomed.append((mgr, owner))
        for mgr, owner in doomed:
            self._shutdown_async(mgr, owner)

    def close_all(self) -> None:
        """Adapter shutdown: drain and shut down every parked manager (inline: the process is ending)."""
        with self._lock:
            entries = list(self._entries.values())
            self._entries.clear()
        for mgr, owner, _ in entries:
            self._shutdown(mgr, owner)

    # -- teardown -------------------------------------------------------------------------------

    def _shutdown_async(self, manager: Any, owner: Optional[Path]) -> None:
        """Eviction runs inside a request's own turn: never make that reply wait on a provider drain."""
        from agent.memory_provider import spawn_context_thread
        spawn_context_thread(self._shutdown, args=(manager, owner), name="api-server-memory-evict").start()

    @staticmethod
    def _shutdown(manager: Any, owner: Optional[Path]) -> None:
        """Bounded drain then provider shutdown, under the OWNING profile's scope: eviction runs inside
        whichever request happened to trigger it, and a provider reads its home/credentials at call time."""
        scope: Any = nullcontext()
        with suppress(Exception):
            from agent.secret_scope import is_multiplex_active
            if owner is not None and is_multiplex_active():
                from gateway.run import _profile_runtime_scope
                scope = _profile_runtime_scope(owner)
        try:
            with scope:
                with suppress(Exception):
                    manager.flush_pending(timeout=10)
                manager.shutdown_all()
        except Exception:
            logger.debug("api_server memory manager shutdown failed", exc_info=True)

    # -- introspection (tests) --------------------------------------------------------------------

    def parked(self) -> Dict[Tuple[str, str], Any]:
        with self._lock:
            return {key: entry[0] for key, entry in self._entries.items()}
