"""Recoverable per-path SessionDB handle caches for the gateway."""

from __future__ import annotations

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
    pass


_health_lock = threading.Lock()
_health_states: weakref.WeakKeyDictionary[_HealthSource, dict[Path, str]] = (
    weakref.WeakKeyDictionary()
)


def _publish_health(source: _HealthSource, path: Path, state: str) -> None:
    """Publish one privacy-safe aggregate for every live gateway DB cache."""
    with _health_lock:
        states = _health_states.setdefault(source, {})
        states[path] = state
        all_states = [value for item in _health_states.values() for value in item.values()]
        if "retrying" in all_states:
            aggregate = "retrying"
        elif "unavailable" in all_states:
            aggregate = "unavailable"
        else:
            aggregate = "ok"

    try:
        from gateway.status import write_runtime_status

        write_runtime_status(session_store={"status": aggregate})
    except Exception:
        # Runtime health is diagnostic only; persistence must not depend on it.
        pass


class RecoverableHandleCache:
    """Cache handles by path while allowing failed opens to heal in-process."""

    def __init__(
        self,
        *,
        handles: dict[Path, Any] | None = None,
        lock: threading.Lock | None = None,
        clock: Callable[[], float] = time.monotonic,
        initial_retry_delay: float = _INITIAL_RETRY_DELAY_SECONDS,
        max_retry_delay: float = _MAX_RETRY_DELAY_SECONDS,
    ) -> None:
        self.handles = handles if handles is not None else {}
        self.lock = lock if lock is not None else threading.Lock()
        self._clock = clock
        self._initial_retry_delay = max(0.0, float(initial_retry_delay))
        self._max_retry_delay = max(
            self._initial_retry_delay, float(max_retry_delay)
        )
        self._unavailable: dict[Path, _Unavailable] = {}
        self._health_source = _HealthSource()

    def get(
        self,
        path: Path,
        opener: Callable[[], Any],
        *,
        raise_on_error: bool = False,
        on_recovered: Callable[[], None] | None = None,
        non_cacheable: Callable[[Exception], bool] | None = None,
    ) -> Any:
        """Return a cached handle or make one bounded, single-flight open attempt."""
        path = Path(path)
        with self.lock:
            if path in self.handles:
                return self.handles[path]

            unavailable = self._unavailable.setdefault(path, _Unavailable())
            now = self._clock()
            if unavailable.in_flight or now < unavailable.next_retry_at:
                return None
            unavailable.in_flight = True
            was_unavailable = unavailable.failures > 0

        if was_unavailable:
            _publish_health(self._health_source, path, "retrying")

        try:
            handle = opener()
        except Exception as exc:
            if non_cacheable is not None and non_cacheable(exc):
                with self.lock:
                    self._unavailable.pop(path, None)
                raise
            with self.lock:
                unavailable = self._unavailable[path]
                unavailable.failures += 1
                delay = min(
                    self._initial_retry_delay
                    * (2 ** min(unavailable.failures - 1, 30)),
                    self._max_retry_delay,
                )
                unavailable.next_retry_at = self._clock() + delay
                unavailable.in_flight = False
            _publish_health(self._health_source, path, "unavailable")
            if raise_on_error:
                raise
            return None

        with self.lock:
            self.handles[path] = handle
            self._unavailable.pop(path, None)
        _publish_health(self._health_source, path, "ok")
        if was_unavailable and on_recovered is not None:
            on_recovered()
        return handle

    def close_all(self, close: Callable[[Any], None]) -> None:
        """Drain cached handles under the lock and close them outside it."""
        with self.lock:
            handles = list(self.handles.values())
            paths = set(self.handles) | set(self._unavailable)
            self.handles.clear()
            self._unavailable.clear()
        for handle in handles:
            try:
                close(handle)
            except Exception:
                pass
        with _health_lock:
            states = _health_states.get(self._health_source)
            if states is not None:
                for path in paths:
                    states.pop(path, None)

    def status_for(self, path: Path) -> str:
        """Return a sanitized state for tests and internal diagnostics."""
        with self.lock:
            if Path(path) in self.handles:
                return "ok"
            unavailable = self._unavailable.get(Path(path))
            if unavailable is None:
                return "unknown"
            return "retrying" if unavailable.in_flight else "unavailable"
