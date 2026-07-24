"""In-memory TTL cache for inbound message deduplication.

Both the WebSocket and the polling fallback can deliver the same message,
and ``message:updated`` re-fires when transcription completes. We dedupe
on ``message_id`` with a short TTL so the same id doesn't dispatch twice
inside the window but old ids don't grow the map unboundedly.
"""

from __future__ import annotations

import time
from typing import Dict

from .constants import DEFAULT_SEEN_TTL_S


class SeenCache:
    def __init__(self, ttl_s: float = DEFAULT_SEEN_TTL_S):
        self._ttl_s = ttl_s
        self._seen: Dict[str, float] = {}

    def is_seen(self, message_id: str) -> bool:
        exp = self._seen.get(message_id)
        if exp is None:
            return False
        if time.time() > exp:
            self._seen.pop(message_id, None)
            return False
        return True

    def mark(self, message_id: str) -> None:
        self._seen[message_id] = time.time() + self._ttl_s
        if len(self._seen) % 100 == 0:
            self._sweep()

    def _sweep(self) -> None:
        now = time.time()
        for mid, exp in list(self._seen.items()):
            if now > exp:
                self._seen.pop(mid, None)
