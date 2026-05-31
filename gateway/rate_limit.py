"""Per-profile token-bucket rate limiting (Tier-2, #9514).

The shared bot token draws on one global API budget, so a chatty routed profile
must not starve the others.  The front checks a per-profile bucket before
dispatching a turn — the single chokepoint where the ask belongs (design §7).
"""

from __future__ import annotations

import time
from typing import Callable


class TokenBucket:
    def __init__(self, capacity: float, refill_per_sec: float, *, clock: Callable[[], float] = time.monotonic):
        self.capacity = capacity
        self.refill_per_sec = refill_per_sec
        self._clock = clock
        self._tokens = float(capacity)
        self._last = clock()

    def allow(self, cost: float = 1.0) -> bool:
        now = self._clock()
        self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.refill_per_sec)
        self._last = now
        if self._tokens >= cost:
            self._tokens -= cost
            return True
        return False


class ProfileRateLimiter:
    def __init__(self, capacity: float = 20, refill_per_sec: float = 1.0, *, clock: Callable[[], float] = time.monotonic):
        self.capacity = capacity
        self.refill_per_sec = refill_per_sec
        self._clock = clock
        self._buckets: dict[str, TokenBucket] = {}

    def allow(self, profile: str, cost: float = 1.0) -> bool:
        bucket = self._buckets.get(profile)
        if bucket is None:
            bucket = self._buckets[profile] = TokenBucket(self.capacity, self.refill_per_sec, clock=self._clock)
        return bucket.allow(cost)
