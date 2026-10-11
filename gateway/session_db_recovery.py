"""Recoverable per-path SessionDB handle caches for the gateway."""

from __future__ import annotations

import contextlib
import threading
import time
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

_INITIAL_RETRY_DELAY_SECONDS = 1.0
_MAX_RETRY_DELAY_SECONDS = 60.0


@dataclass
class _Unavailable:
    failures: int = 0
    next_retry_at: float = 0.0
    in_flight: bool = False


class _HealthSource:
    """Weak-keyable identity for one cache's health entries."""


_health_lock = threading.Lock()
_health_states: weakref.WeakKeyDictionary[_HealthSource, dict[Path, str]]
_health_states = weakref.WeakKeyDictionary()


def _publish_health(source: _HealthSource, path: Path, state: str) -> None:
    """Publish one privacy-safe aggregate (no paths, no errors) across all live caches."""
    with _health_lock:
        _health_states.setdefault(source, {})[path] = state
        all_states = {value for item in _health_states.values() for value in item.values()}
        aggregate = next((s for s in ("retrying", "unavailable") if s in all_states), "ok")
    try:
        from gateway.status import publish_runtime_status
        publish_runtime_status(session_store={"status": aggregate})
    except Exception:
        pass  # Runtime health is diagnostic only; persistence must not depend on it.


def _registry_tore_down(handle: Any) -> bool:
    """True once the process-wide registry force-closed *handle* (``close_all`` / ``close_all_under``
    clear ``_shared_registry_owned`` at teardown); an ``AsyncSessionDB`` (the only handle type with
    a ``_db``) is judged by the SessionDB it wraps. Meaningful only for a handle that came from the
    registry: a test double without the flag reads as live."""
    inner = getattr(handle, "_db", handle)
    return getattr(inner, "_shared_registry_owned", True) is False


class RecoverableHandleCache:
    """Cache handles by path while allowing failed opens to heal in-process.

    Opens run OUTSIDE ``lock`` (single-flight per path via ``in_flight``); ``close_all`` bumps
    ``_generation`` so a later-completing open is rejected instead of resurrecting the cache.

    ``registry_backed`` declares that every handle ``opener`` returns came from
    ``hermes_state_registry`` (directly or wrapped). The registry may force-close such a handle at
    any time (profile unserve/delete), so a torn-down handle is never published and a cached one is
    evicted on its next read. The owner declares the provenance instead of the cache sampling it from
    the handle: a teardown that lands between the open and the publication has already cleared the
    flag such a sample would read, and the dead handle would be filed as a never-shared one.
    """

    def __init__(
        self, *, handles: dict[Path, Any] | None = None, lock: threading.Lock | None = None,
        clock: Callable[[], float] = time.monotonic,
        initial_retry_delay: float = _INITIAL_RETRY_DELAY_SECONDS,
        max_retry_delay: float = _MAX_RETRY_DELAY_SECONDS,
        registry_backed: bool = False,
    ) -> None:
        self.handles = handles if handles is not None else {}
        self.lock = lock if lock is not None else threading.Lock()
        self._clock = clock
        self._initial_retry_delay = max(0.0, float(initial_retry_delay))
        self._max_retry_delay = max(self._initial_retry_delay, float(max_retry_delay))
        self._unavailable: dict[Path, _Unavailable] = {}
        self._registry_backed = registry_backed
        self._health_source = _HealthSource()
        self._generation = 0
        self._close_rejected: Callable[[Any], None] | None = None

    def _is_stale(self, path: Path, unavailable: _Unavailable, generation: int) -> bool:
        """Caller holds ``lock``: True when close_all ran or the slot was replaced mid-open."""
        return generation != self._generation or self._unavailable.get(path) is not unavailable

    def get(
        self, path: Path, opener: Callable[[], Any], *, raise_on_error: bool = False,
        on_recovered: Callable[[], None] | None = None,
        non_cacheable: Callable[[Exception], bool] | None = None,
    ) -> Any:
        """Return a cached handle or make one bounded, single-flight open attempt; None while
        a retry is in flight or backing off.  ``non_cacheable`` exceptions (e.g. a live-system
        guard) are re-raised without recording a failure so the next call retries at once."""
        path = Path(path)
        with self.lock:
            if path in self.handles:
                handle = self.handles[path]
                if not self._registry_backed or not _registry_tore_down(handle):
                    return handle
                # The registry force-closed this generation. Serving it would let its self-heal reopen
                # a writer the registry cannot see (a second writer beside the next ``acquire``), or keep
                # raising StateDbReplacedError after a delete + recreate: reopen through the registry.
                del self.handles[path]
            unavailable = self._unavailable.setdefault(path, _Unavailable())
            if unavailable.in_flight or self._clock() < unavailable.next_retry_at:
                return None
            unavailable.in_flight = True
            was_unavailable = unavailable.failures > 0
            generation = self._generation
        if was_unavailable:
            _publish_health(self._health_source, path, "retrying")

        try:
            handle = opener()
        except Exception as exc:
            uncacheable = non_cacheable is not None and non_cacheable(exc)
            with self.lock:
                stale = self._is_stale(path, unavailable, generation)
                if uncacheable:
                    if not stale:
                        self._unavailable.pop(path, None)
                    raise
                if not stale:
                    unavailable.failures += 1
                    backoff = self._initial_retry_delay * (2 ** min(unavailable.failures - 1, 30))
                    unavailable.next_retry_at = self._clock() + min(backoff, self._max_retry_delay)
                    unavailable.in_flight = False
            if not stale:
                _publish_health(self._health_source, path, "unavailable")
            if raise_on_error:
                raise
            return None

        with self.lock:
            stale = self._is_stale(path, unavailable, generation)
            # The registry can tear the generation down between the open and this publication (a
            # profile unserve racing that profile's first use). The handle is already closed, so
            # caching it as a never-shared one would serve it on every later read; don't publish it.
            # The slot stays in its failure state (a pending recovery is still owed to
            # ``on_recovered``) with the in-flight flag and backoff released, so the next call
            # reopens at once. This narrows the window, it cannot close it: teardown does not take
            # ``lock``, so it can still land after this check or after the handle was returned.
            retired = not stale and self._registry_backed and _registry_tore_down(handle)
            if not stale:
                if retired:
                    unavailable.in_flight = False
                    unavailable.next_retry_at = 0.0
                else:
                    self.handles[path] = handle
                    self._unavailable.pop(path, None)
            close_rejected = self._close_rejected if stale else None
        if stale:
            if close_rejected is not None:
                with contextlib.suppress(Exception):
                    close_rejected(handle)
            return None
        if retired:
            return None
        _publish_health(self._health_source, path, "ok")
        if was_unavailable and on_recovered is not None:
            on_recovered()
        return handle

    def close_all(self, close: Callable[[Any], None]) -> None:
        """Drain cached handles under the lock and close them outside it."""
        with self.lock:
            self._generation += 1
            self._close_rejected = close
            handles = list(self.handles.values())
            paths = set(self.handles) | set(self._unavailable)
            self.handles.clear()
            self._unavailable.clear()
        for handle in handles:
            with contextlib.suppress(Exception):
                close(handle)
        with _health_lock:
            states = _health_states.get(self._health_source, {})
            for path in paths:
                states.pop(path, None)
