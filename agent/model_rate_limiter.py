"""Per-model rate limiter using a token-bucket algorithm.

Proactively spaces API requests to avoid 429 errors.  Each
(provider, model) pair gets its own limiter instance so that
different models on the same provider are rate-limited
independently.

Usage::

    limiter = ModelRateLimiterGlobal.get_or_create("nvidia", "minimax-m2.7")
    async with limiter.acquire(40):  # 40 RPM
        response = client.chat.completions.create(...)

Or synchronously (e.g. inside a worker thread)::

    limiter.acquire_sync(40)  # blocks until a token is available
    response = client.chat.completions.create(...)
"""

from __future__ import annotations

import asyncio
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, Tuple


class ModelRateLimiter:
    """Token-bucket rate limiter for a single (provider, model) pair.

    Bucket capacity is 1 token (no burst).  Each token replenishes after
    ``1 / rate`` seconds where ``rate`` is requests-per-second.
    """

    def __init__(self, provider: str, model: str) -> None:
        self.provider = provider
        self.model = model
        self._lock = threading.Lock()
        self._last_request_time: float = 0.0
        self._requests_made: int = 0
        self._initialized = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def acquire_sync(self, rate_limit_rpm: int) -> None:
        """Block until a token is available.  No-op when *rate_limit_rpm* is 0."""
        if rate_limit_rpm <= 0:
            return
        rate_per_sec = rate_limit_rpm / 60.0
        interval = 1.0 / rate_per_sec
        with self._lock:
            now = time.monotonic()
            if not self._initialized:
                self._initialized = True
                self._last_request_time = now
                self._requests_made += 1
                return
            elapsed = now - self._last_request_time
            wait = max(0.0, interval - elapsed)
            self._last_request_time = now + wait
            self._requests_made += 1
        if wait > 0:
            time.sleep(wait)

    @asynccontextmanager
    async def acquire(self, rate_limit_rpm: int):
        """Async context manager that waits for a token before yielding."""
        if rate_limit_rpm <= 0:
            yield
            return
        rate_per_sec = rate_limit_rpm / 60.0
        interval = 1.0 / rate_per_sec
        with self._lock:
            now = time.monotonic()
            if not self._initialized:
                self._initialized = True
                self._last_request_time = now
                self._requests_made += 1
                yield
                return
            elapsed = now - self._last_request_time
            wait = max(0.0, interval - elapsed)
            self._last_request_time = now + wait
            self._requests_made += 1
        if wait > 0:
            await asyncio.sleep(wait)
        yield

    def get_status(self) -> Dict[str, Any]:
        """Return current limiter stats."""
        with self._lock:
            return {
                "provider": self.provider,
                "model": self.model,
                "requests_made": self._requests_made,
                "last_request_time": self._last_request_time,
            }


class ModelRateLimiterGlobal:
    """Global singleton registry of ``ModelRateLimiter`` instances.

    Keyed by ``(provider, model)`` tuple.
    """

    _registry: Dict[Tuple[str, str], ModelRateLimiter] = {}
    _lock = threading.Lock()

    @classmethod
    def get_or_create(cls, provider: str, model: str) -> ModelRateLimiter:
        """Return the limiter for *(provider, model)*, creating if needed."""
        key = (provider or "", model or "")
        with cls._lock:
            if key not in cls._registry:
                cls._registry[key] = ModelRateLimiter(provider, model)
            return cls._registry[key]

    @classmethod
    def set_limit(cls, provider: str, model: str, rpm: int) -> ModelRateLimiter:
        """Convenience: get-or-create and store a configured limit."""
        return cls.get_or_create(provider, model)

    @classmethod
    def get_status(cls) -> Dict[str, Any]:
        """Return status for all registered limiters."""
        with cls._lock:
            return {
                f"{k[0]}:{k[1]}": v.get_status()
                for k, v in cls._registry.items()
            }
