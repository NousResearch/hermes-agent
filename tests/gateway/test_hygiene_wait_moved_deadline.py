"""The gateway's hygiene wait follows the compression fence when prompt reading moves its deadline.

A local llama-server reading a long conversation reports progress, and each report of newly read
tokens pushes the fence deadline back. The gateway waits in short slices and must judge each slice
against the deadline as it stands after the slice, not as it stood before: an extension that lands
in the last slice before the old deadline would otherwise cancel a healthy summary.
"""

import asyncio
import time
from types import SimpleNamespace

from gateway.run import GatewayRunner


class _ReadingFence:
    """A fence whose deadline moves as the old one passes, the way a late progress chunk moves it."""

    def __init__(self, ceiling: float):
        self.wait_started = time.monotonic()
        self.deadline_monotonic = self.wait_started + ceiling
        self._moved = False

    @property
    def is_cancelled(self) -> bool:
        if not self._moved and time.monotonic() >= self.deadline_monotonic:
            self.deadline_monotonic += 5.0
            self._moved = True
        return time.monotonic() >= self.deadline_monotonic

    def seconds_since_progress(self) -> float:
        return 0.0


def test_hygiene_wait_reads_the_moved_deadline_after_each_slice():
    async def drive():
        fence = _ReadingFence(ceiling=2.0)
        future = asyncio.get_running_loop().create_future()
        asyncio.get_running_loop().call_later(4.0, future.set_result, ("compressed", None))
        attempt = GatewayRunner._HygieneAttempt(
            agent=None, meta=None, commit_fence=fence, future=future, wait_started=fence.wait_started)
        hs = SimpleNamespace(timeout_seconds=30.0, total_ceiling_seconds=2.0, max_turn_hold_seconds=30.0)
        return await GatewayRunner._hmwa_hygiene_wait_for_summary(
            None, attempt, hs, SimpleNamespace(session_id="sess-1"))

    assert asyncio.run(drive()) == "compressed"
