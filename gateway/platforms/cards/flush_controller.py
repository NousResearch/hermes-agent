"""Throttled flush controller for Feishu streaming card updates.

Avoids rate-limiting by batching LLM token events into time-windowed
PATCH calls. Pure scheduling primitive — no business logic.
"""

import asyncio
import logging
import time
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger(__name__)

# Default throttle window in seconds (300 ms)
DEFAULT_THROTTLE_MS: int = 300

# After a long gap (>2 s), batch briefly before first flush
LONG_GAP_THRESHOLD_MS: int = 2000
BATCH_AFTER_GAP_MS: int = 80


# ---------------------------------------------------------------------------
# FlushController
# ---------------------------------------------------------------------------

class FlushController:
    """Async throttled flush controller.

    Accepts streaming update signals and schedules batched flushes at a
    configurable minimum interval, preventing Feishu API rate-limiting.

    The actual flush work is provided as a coroutine callback so this
    class remains business-logic-free.

    Usage::

        async def do_patch():
            await feishu_client.patch_card(...)

        fc = FlushController(do_flush=do_patch, throttle_ms=300)
        fc.set_ready(True)

        # On each LLM token / tool event:
        await fc.schedule_flush()

        # When streaming completes:
        await fc.force_flush_now()
        fc.complete()

    """

    def __init__(
        self,
        do_flush: Callable[[], Coroutine[Any, Any, None]],
        throttle_ms: int = DEFAULT_THROTTLE_MS,
    ) -> None:
        """Initialise the controller.

        Args:
            do_flush: Async callable that performs the actual card PATCH.
            throttle_ms: Minimum milliseconds between flushes.
        """
        self._do_flush = do_flush
        self._throttle_ms = throttle_ms

        # Mutex state
        self._flush_in_progress: bool = False
        self._flush_resolvers: list[asyncio.Future] = []
        self._needs_reflush: bool = False

        # Timer state
        self._pending_task: Optional[asyncio.Task] = None
        self._last_update_time: float = 0.0

        # Lifecycle state
        self._is_completed: bool = False
        self._is_ready: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_ready(self, ready: bool) -> None:
        """Gate the controller. Flushes are no-ops until ready=True.

        Args:
            ready: True once the card message has been created and is
                   patchable.
        """
        self._is_ready = ready
        if ready:
            # Initialise timestamp so the first throttledUpdate sees a
            # small elapsed value (matching original JS behaviour).
            self._last_update_time = time.monotonic()

    def complete(self) -> None:
        """Mark the stream as completed — no more flushes after current one."""
        self._is_completed = True

    def cancel_pending(self) -> None:
        """Cancel any pending deferred flush task."""
        if self._pending_task and not self._pending_task.done():
            self._pending_task.cancel()
            self._pending_task = None

    async def wait_for_flush(self) -> None:
        """Wait until any in-progress flush finishes."""
        if not self._flush_in_progress:
            return
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        self._flush_resolvers.append(fut)
        await asyncio.shield(fut)

    async def schedule_flush(self, throttle_ms: Optional[int] = None) -> None:
        """Throttled entry point called on each streaming event.

        Args:
            throttle_ms: Override the default throttle window for this call.
        """
        if not self._is_ready:
            return

        ms = throttle_ms if throttle_ms is not None else self._throttle_ms
        now = time.monotonic()
        elapsed_ms = (now - self._last_update_time) * 1000.0

        if elapsed_ms >= ms:
            self.cancel_pending()
            if elapsed_ms > LONG_GAP_THRESHOLD_MS:
                # After a long gap, batch briefly so the first visible
                # update contains meaningful text rather than 1-2 chars.
                self._last_update_time = now
                self._pending_task = asyncio.ensure_future(
                    self._delayed_flush(BATCH_AFTER_GAP_MS / 1000.0)
                )
            else:
                await self._flush()
        elif self._pending_task is None or self._pending_task.done():
            # Inside throttle window — schedule a deferred flush.
            delay_s = (ms - elapsed_ms) / 1000.0
            self._pending_task = asyncio.ensure_future(
                self._delayed_flush(delay_s)
            )

    async def force_flush_now(self) -> None:
        """Cancel pending timer and flush immediately.

        Intended to be called when the stream ends to guarantee a final
        up-to-date patch is sent.
        """
        self.cancel_pending()
        await self._flush()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _delayed_flush(self, delay_s: float) -> None:
        """Wait delay_s seconds then flush."""
        try:
            await asyncio.sleep(delay_s)
        except asyncio.CancelledError:
            return
        self._pending_task = None
        await self._flush()

    async def _flush(self) -> None:
        """Execute the flush callback with mutex guard and reflush support.

        If a flush is already in progress, marks _needs_reflush so a
        follow-up flush fires immediately after the current one completes.
        """
        if not self._is_ready or self._is_completed:
            return
        if self._flush_in_progress:
            self._needs_reflush = True
            return

        self._flush_in_progress = True
        self._needs_reflush = False
        # Stamp before the API call to block concurrent callers.
        self._last_update_time = time.monotonic()

        try:
            await self._do_flush()
            self._last_update_time = time.monotonic()
        except Exception:
            logger.exception("FlushController: flush callback raised")
        finally:
            self._flush_in_progress = False
            resolvers = self._flush_resolvers
            self._flush_resolvers = []
            for fut in resolvers:
                if not fut.done():
                    fut.set_result(None)

            # Events arrived while the API call was in flight.
            if self._needs_reflush and not self._is_completed and self._pending_task is None:
                self._needs_reflush = False
                self._pending_task = asyncio.ensure_future(
                    self._delayed_flush(0.0)
                )
