"""Token-bucket rate limiter for MCP tool calls (in-process)."""

from __future__ import annotations

import threading
import time
from typing import Dict


class TokenBucket:
    """Simple token bucket: `capacity` tokens refill at `refill_per_sec`."""

    def __init__(self, *, capacity: float, refill_per_sec: float) -> None:
        self._capacity = capacity
        self._refill = refill_per_sec
        self._tokens = float(capacity)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, cost: float = 1.0) -> bool:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            self._last = now
            self._tokens = min(self._capacity, self._tokens + elapsed * self._refill)
            if self._tokens >= cost:
                self._tokens -= cost
                return True
            return False


class ToolRateLimiter:
    """Per-tool-name buckets (defaults are conservative for public RPC)."""

    def __init__(self) -> None:
        self._buckets: Dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def _bucket(self, tool: str) -> TokenBucket:
        with self._lock:
            if tool not in self._buckets:
                # send paths are tighter than read-only
                cap, rps = (8.0, 2.0) if "send" in tool else (30.0, 10.0)
                self._buckets[tool] = TokenBucket(capacity=cap, refill_per_sec=rps)
            return self._buckets[tool]

    def allow(self, tool: str) -> bool:
        return self._bucket(tool).acquire(1.0)
