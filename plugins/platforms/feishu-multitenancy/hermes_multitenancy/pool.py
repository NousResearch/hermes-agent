"""LRU RuntimePool: bounded in-memory cache of ProfileRuntime instances.

Configurable parameters:
  - max_loaded_runtimes: cache cap (default 50)
  - idle_evict_seconds: drop entries idle longer than this (default 300)
  - cold_start_concurrency: limit concurrent cold-starts (default 8)
  - inflight_timeout_seconds: cancel a dispatch hung > this (default 600)
  - inflight_no_evict: never evict a runtime currently dispatching

Eviction strategy: lazy. Every ``acquire()`` call sweeps idle entries first.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from .runtime import ProfileRuntime

logger = logging.getLogger(__name__)

DEFAULT_MAX_LOADED = 50
DEFAULT_IDLE_EVICT = 300.0
DEFAULT_COLD_START_CONCURRENCY = 8
DEFAULT_INFLIGHT_TIMEOUT = 600.0


@dataclass
class _PoolEntry:
    profile_name: str
    runtime: ProfileRuntime
    last_used: float = field(default_factory=time.time)
    in_flight: int = 0


# A factory takes (profile_name, profile_home) and returns a fresh runtime.
RuntimeFactory = Callable[[str, Path], ProfileRuntime]


def _default_factory(profile_name: str, profile_home: Path) -> ProfileRuntime:
    return ProfileRuntime(profile_home=profile_home)


class RuntimePool:
    """Bounded LRU cache of profile runtimes with cold-start throttling.

    Use as::

        pool = RuntimePool()
        async with pool.dispatch("alice", profile_home, event) as response:
            ...
    """

    def __init__(
        self,
        *,
        max_loaded_runtimes: int = DEFAULT_MAX_LOADED,
        idle_evict_seconds: float = DEFAULT_IDLE_EVICT,
        cold_start_concurrency: int = DEFAULT_COLD_START_CONCURRENCY,
        inflight_timeout_seconds: float = DEFAULT_INFLIGHT_TIMEOUT,
        runtime_factory: RuntimeFactory = _default_factory,
        time_fn: Callable[[], float] = time.time,
    ) -> None:
        self.max_loaded_runtimes = max_loaded_runtimes
        self.idle_evict_seconds = idle_evict_seconds
        self.inflight_timeout_seconds = inflight_timeout_seconds
        self._factory = runtime_factory
        self._now = time_fn
        self._entries: "OrderedDict[str, _PoolEntry]" = OrderedDict()
        self._cold_start_sem = asyncio.Semaphore(cold_start_concurrency)
        # Lock-free LRU manipulation under asyncio cooperative scheduling.

    # -- queries -----------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def loaded_profiles(self) -> list[str]:
        return list(self._entries.keys())

    def in_flight_count(self, profile_name: str) -> int:
        entry = self._entries.get(profile_name)
        return entry.in_flight if entry else 0

    # -- core API ----------------------------------------------------------

    async def dispatch(
        self,
        profile_name: str,
        profile_home: Path,
        event: Any,
    ) -> str:
        """Acquire (or cold-start) the runtime, run dispatch, release.

        Honors ``inflight_timeout_seconds`` via ``asyncio.wait_for``.
        """
        entry = await self._acquire(profile_name, profile_home)
        try:
            return await asyncio.wait_for(
                entry.runtime.dispatch(event),
                timeout=self.inflight_timeout_seconds,
            )
        finally:
            entry.in_flight -= 1
            entry.last_used = self._now()

    # -- helpers -----------------------------------------------------------

    async def _acquire(self, profile_name: str, profile_home: Path) -> _PoolEntry:
        # Lazy idle sweep on every acquire.
        self._evict_idle_inplace()

        existing = self._entries.get(profile_name)
        if existing is not None:
            self._entries.move_to_end(profile_name)
            existing.in_flight += 1
            existing.last_used = self._now()
            return existing

        # Cold-start path — throttled by semaphore.
        async with self._cold_start_sem:
            # Re-check after acquiring sem: another task may have created it.
            existing = self._entries.get(profile_name)
            if existing is not None:
                self._entries.move_to_end(profile_name)
                existing.in_flight += 1
                existing.last_used = self._now()
                return existing

            self._evict_to_capacity()
            runtime = self._factory(profile_name, profile_home)
            entry = _PoolEntry(profile_name=profile_name, runtime=runtime, last_used=self._now())
            entry.in_flight = 1
            self._entries[profile_name] = entry
            return entry

    def _evict_idle_inplace(self) -> None:
        """Drop entries idle > idle_evict_seconds and not in flight."""
        cutoff = self._now() - self.idle_evict_seconds
        for name in list(self._entries.keys()):
            entry = self._entries[name]
            if entry.in_flight > 0:
                continue
            if entry.last_used < cutoff:
                del self._entries[name]
                logger.debug("multitenancy: pool evicted idle %s", name)

    def _evict_to_capacity(self) -> None:
        """Drop oldest non-in-flight entries until len < max_loaded_runtimes.

        If every loaded entry is in-flight, we exceed capacity rather than
        blocking the gateway dispatch path.
        """
        while len(self._entries) >= self.max_loaded_runtimes:
            for name in list(self._entries.keys()):
                entry = self._entries[name]
                if entry.in_flight == 0:
                    del self._entries[name]
                    logger.debug("multitenancy: pool LRU-evicted %s", name)
                    break
            else:
                # All entries in-flight — log + bail
                logger.warning(
                    "multitenancy: pool over capacity (loaded=%d, max=%d), all in-flight",
                    len(self._entries),
                    self.max_loaded_runtimes,
                )
                return

    def evict_idle(self) -> int:
        """Public wrapper — returns number of entries evicted."""
        before = len(self._entries)
        self._evict_idle_inplace()
        return before - len(self._entries)
