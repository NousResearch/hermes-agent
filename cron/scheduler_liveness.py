"""Liveness side of a running cron job: the inactivity watchdog loop and the execution-ledger
progress stamps.

Both read the same signal (the agent's idle time) and act on opposite edges of it. The
watchdog interrupts the run once idle time crosses the configured limit. The stamper refreshes
``progress_at`` while idle time stays short, so the live-owner stale-claim sweep (#115692)
measures silence since the last stamp rather than run length: a busy multi-hour job keeps
stamping and is never reclaimed under a healthy owner, while a wedged worker reports growing
idle time, stops stamping, and is still released once the derived bound passes.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

from agent.session_activity import AwakeIdleMeter
from cron.executions import touch_execution_progress

logger = logging.getLogger(__name__)


def _inactivity_watchdog_loop(
    *, get_idle_seconds: Callable[[], float], limit_s: float, poll_s: float, stop: threading.Event,
    future_done: Callable[[], bool], meter: Optional[AwakeIdleMeter] = None,
) -> bool:
    """Poll idle time until limit (-> True), stop, or the future completes (-> False). Uses
    ``threading.Event.wait``, not asyncio, so a blocked event loop cannot disable the watchdog.

    Driven by ``threading.Event.wait`` (a kernel timeout), not asyncio, so a blocked event-loop /
    ``run_job`` thread cannot disable this watchdog the way ``asyncio.sleep`` / ``wait_for`` would (family A
    of #94285 — the 4118s-idle-on-a-600s-limit cron hang). Returns True when *limit_s* of inactivity was
    observed.
    """
    # A sleeping host freezes the job with it, so time asleep never counts as inactivity.
    if meter is None:
        meter = AwakeIdleMeter()
    while not stop.wait(poll_s):
        if future_done():
            return False
        try:
            idle = float(get_idle_seconds() or 0.0)
        except Exception:
            idle = 0.0
        if meter.measure(idle) >= limit_s:
            return True
    return False


class ExecutionProgressStamper:
    def __init__(
        self, execution_id: str, job_name: str, *, idle_seconds: Callable[[], float],
        every_seconds: float,
    ):
        self._execution_id = execution_id
        self._job_name = job_name
        self._idle_seconds = idle_seconds
        self._every = every_seconds
        self._last = time.monotonic()

    def tick(self) -> None:
        if not self._execution_id:
            return
        now = time.monotonic()
        if now - self._last < self._every:
            return
        # Idle past the cadence means the agent itself stopped: no stamp, so the ledger goes
        # quiet exactly when the watchdog would.
        if self._idle_seconds() >= self._every:
            return
        self._last = now
        try:
            touch_execution_progress(self._execution_id)
        except Exception:
            logger.debug("Job '%s': execution progress stamp failed", self._job_name, exc_info=True)
