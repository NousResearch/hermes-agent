"""Process-wide, content-free observations from the existing loop watchdog."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProbeSnapshot:
    misses: int
    response_seconds: float


_lock = threading.Lock()
_latest: ProbeSnapshot | None = None


def reset_probe() -> None:
    global _latest
    with _lock:
        _latest = None


def probe_snapshot() -> ProbeSnapshot | None:
    with _lock:
        return _latest


def record_probe(*, misses: int, response_seconds: float) -> None:
    global _latest
    sample = ProbeSnapshot(misses, response_seconds)
    with _lock:
        previous, _latest = _latest, sample
    if misses:
        logger.warning('Gateway event loop liveness probe timed out: consecutive_misses=%d response_seconds=%.3f',
                       misses, response_seconds)
    elif previous is not None and previous.misses:
        logger.info('Gateway event loop liveness recovered after %d missed probe(s)', previous.misses)
