"""Shared async gate for heavyweight SessionDB read bursts.

The dashboard REST app and its JSON-RPC WebSocket sidecar run on the same
uvicorn event loop. Heavy session-list scans must therefore queue on the loop
before they occupy executor threads; otherwise a burst of blocked worker calls
can starve unrelated WebSocket operations.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import weakref
from contextlib import asynccontextmanager
from typing import Any

_LOG = logging.getLogger(__name__)
_DEFAULT_MAX_CONCURRENCY = 2
_QUEUE_WAIT_TIMEOUT_S = 1.0
_QUEUE_WAIT_LOG_THRESHOLD_S = 0.001

_SEMAPHORES: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, tuple[int, asyncio.Semaphore]
] = weakref.WeakKeyDictionary()
_SEMAPHORES_LOCK = threading.Lock()
_STATS_LOCK = threading.Lock()
_STATS: dict[str, float | int] = {
    "acquired_count": 0,
    "queued_count": 0,
    "shed_count": 0,
    "queue_wait_seconds_total": 0.0,
}


class SessionDBHeavyReadBusy(RuntimeError):
    """Raised when the heavy-read queue stays saturated past its wait bound."""

    def __init__(
        self, *, queue_wait: float, retry_after: float = _QUEUE_WAIT_TIMEOUT_S
    ):
        super().__init__("backend busy; retry shortly")
        self.queue_wait = queue_wait
        self.retry_after = max(0.1, retry_after)

    def to_payload(self) -> dict[str, Any]:
        return {
            "error": "backend_busy",
            "message": str(self),
            "retryable": True,
            "retry_after": self.retry_after,
        }


def _coerce_positive_int(value: Any, default: int) -> int:
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return default
    return coerced if coerced >= 1 else default


def _configured_max_concurrency() -> int:
    """Read dashboard.heavy_read_max_concurrency lazily from config.yaml."""
    default = _DEFAULT_MAX_CONCURRENCY
    try:
        from hermes_cli.config import DEFAULT_CONFIG, load_config

        default_dashboard = DEFAULT_CONFIG.get("dashboard")
        if isinstance(default_dashboard, dict):
            default = _coerce_positive_int(
                default_dashboard.get("heavy_read_max_concurrency"),
                _DEFAULT_MAX_CONCURRENCY,
            )
        cfg = load_config() or {}
        dashboard = cfg.get("dashboard") if isinstance(cfg, dict) else {}
        if not isinstance(dashboard, dict):
            dashboard = {}
        return _coerce_positive_int(
            dashboard.get("heavy_read_max_concurrency"),
            default,
        )
    except Exception:
        return default


def session_db_heavy_read_semaphore() -> asyncio.Semaphore:
    """Return the loop-local heavy-read semaphore, sized from live config."""
    loop = asyncio.get_running_loop()
    limit = _configured_max_concurrency()
    with _SEMAPHORES_LOCK:
        existing = _SEMAPHORES.get(loop)
        if existing is not None:
            existing_limit, existing_semaphore = existing
            if existing_limit == limit:
                return existing_semaphore
            # Apply config changes without reimport once the old limiter has
            # drained. Replacing while permits are checked out would let old and
            # new semaphores admit work simultaneously and temporarily exceed
            # both bounds.
            if getattr(existing_semaphore, "_value", 0) < existing_limit:
                return existing_semaphore
        semaphore = asyncio.Semaphore(limit)
        _SEMAPHORES[loop] = (limit, semaphore)
        return semaphore


def _record_stats(
    *, acquired: int = 0, queued: int = 0, shed: int = 0, queue_wait: float = 0.0
) -> None:
    with _STATS_LOCK:
        _STATS["acquired_count"] = int(_STATS["acquired_count"]) + acquired
        _STATS["queued_count"] = int(_STATS["queued_count"]) + queued
        _STATS["shed_count"] = int(_STATS["shed_count"]) + shed
        _STATS["queue_wait_seconds_total"] = (
            float(_STATS["queue_wait_seconds_total"]) + queue_wait
        )


def session_db_heavy_read_stats() -> dict[str, float | int]:
    with _STATS_LOCK:
        stats = dict(_STATS)
    stats["max_concurrency"] = _configured_max_concurrency()
    stats["queue_timeout_seconds"] = _QUEUE_WAIT_TIMEOUT_S
    return stats


@asynccontextmanager
async def session_db_heavy_read_slot(surface: str, operation: str):
    """Queue for a heavyweight SessionDB read without occupying a worker thread."""
    loop = asyncio.get_running_loop()
    semaphore = session_db_heavy_read_semaphore()
    queued_at_entry = semaphore.locked()
    start = loop.time()
    try:
        await asyncio.wait_for(semaphore.acquire(), timeout=_QUEUE_WAIT_TIMEOUT_S)
    except asyncio.TimeoutError as exc:
        queue_wait = loop.time() - start
        _record_stats(queued=1, shed=1, queue_wait=queue_wait)
        _LOG.warning(
            "session_db_heavy_read queue_wait=%.3fs timeout=%.3fs surface=%s operation=%s outcome=shed",
            queue_wait,
            _QUEUE_WAIT_TIMEOUT_S,
            surface,
            operation,
        )
        raise SessionDBHeavyReadBusy(
            queue_wait=queue_wait,
            retry_after=_QUEUE_WAIT_TIMEOUT_S,
        ) from exc

    queue_wait = loop.time() - start
    was_queued = queued_at_entry or queue_wait >= _QUEUE_WAIT_LOG_THRESHOLD_S
    _record_stats(
        acquired=1,
        queued=1 if was_queued else 0,
        queue_wait=queue_wait if was_queued else 0.0,
    )
    if was_queued:
        _LOG.info(
            "session_db_heavy_read queue_wait=%.3fs surface=%s operation=%s outcome=acquired",
            queue_wait,
            surface,
            operation,
        )
    try:
        yield
    finally:
        semaphore.release()


def reset_session_db_heavy_read_gate_for_tests() -> None:
    with _SEMAPHORES_LOCK:
        _SEMAPHORES.clear()
    with _STATS_LOCK:
        _STATS.update(
            {
                "acquired_count": 0,
                "queued_count": 0,
                "shed_count": 0,
                "queue_wait_seconds_total": 0.0,
            }
        )
