"""Generic throttled scheduler — FlushController."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

_logger = logging.getLogger("hermes_lark_streaming")


CARDKIT_MS = 0.100  # refresh interval for the CardKit streaming API
LONG_GAP_MS = 2.000  # gaps beyond this are treated as long idles
BATCH_AFTER_GAP_MS = 0.300  # after a long idle, wait this long before flushing


class FlushController:
    """Generic throttled scheduler with a lock and delayed flushes.

    Contains no Feishu logic; it only decides when the callback runs.
    """

    def __init__(self, throttle_ms: float = CARDKIT_MS, *, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self._throttle_ms = throttle_ms
        self._flush_in_progress = False
        self._needs_reflush = False
        self._pending_timer: asyncio.TimerHandle | None = None
        self._last_update_time = 0.0
        self._completed = False
        self._card_message_ready = False
        self._flush_resolvers: list[asyncio.Future[None]] = []
        self._loop = loop if loop is not None else asyncio.get_running_loop()

    @property
    def throttle_ms(self) -> float:
        return self._throttle_ms

    def schedule_update(self, do_flush: Callable[[], Awaitable[None]]) -> None:
        """Request a throttled card refresh.

        do_flush: async callable performing the actual API call.
        """
        if self._completed or not self._card_message_ready:
            return
        now = time.monotonic()
        elapsed = now - self._last_update_time

        if elapsed >= self._throttle_ms:
            # beyond the throttle window
            if elapsed > LONG_GAP_MS:
                # long idle → delay a small batch so content is more complete
                if self._pending_timer is None:
                    self._schedule(delay=BATCH_AFTER_GAP_MS, do_flush=do_flush)
            else:
                # flush now
                self._do_flush_task(do_flush)
        else:
            # still inside the throttle window → defer to the window edge
            if self._pending_timer is None:
                delay = self._throttle_ms - elapsed
                self._schedule(delay=delay, do_flush=do_flush)

    async def flush_now(self, do_flush: Callable[[], Awaitable[None]]) -> None:
        """Run one flush immediately and wait for it."""
        if self._completed or not self._card_message_ready:
            return
        self._cancel_timer()
        await self._do_flush(do_flush)

    async def wait_for_flush(self) -> None:
        """Wait for an in-flight flush to finish.

        If already mark_completed (the complete path calls mark_completed before
        wait_for_flush), return at once — the resolvers are cleared and waiting
        would hang forever (observed: a card stuck streaming with a bouncing
        ellipsis and no error logged).
        """
        if self._completed:
            return
        if not self._flush_in_progress:
            return
        future: asyncio.Future[None] = self._loop.create_future()
        self._flush_resolvers.append(future)
        await future

    def mark_completed(self) -> None:
        """Mark complete; no further updates are accepted."""
        self._completed = True
        self._cancel_timer()
        for r in self._flush_resolvers:
            if not r.done():
                r.set_result(None)
        self._flush_resolvers.clear()

    def reset_for_reactivate(self) -> None:
        """Reactivate: undo mark_completed and accept updates again.

        Used by cross-turn merging (background turns reuse a COMPLETED card): the
        terminal card accepts further segment deltas again, appended via
        batch_update (streaming_mode is already off — no typewriter, but content
        still updates).
        """
        self._completed = False
        self._flush_in_progress = False
        self._needs_reflush = False
        self._cancel_timer()
        self._last_update_time = time.monotonic()

    def set_throttle(self, ms: float) -> None:
        self._throttle_ms = ms

    def set_card_message_ready(self, ready: bool) -> None:
        """Mark the card message as ready; initialize timestamps."""
        self._card_message_ready = ready
        if ready:
            self._last_update_time = time.monotonic()

    def _schedule(self, delay: float, do_flush: Callable[[], Awaitable[None]]) -> None:
        self._cancel_timer()
        self._pending_timer = self._loop.call_later(
            delay,
            self._do_flush_task,
            do_flush,
        )

    def _do_flush_task(self, do_flush: Callable[[], Awaitable[None]]) -> None:
        self._pending_timer = None
        self._loop.call_soon(asyncio.create_task, self._do_flush(do_flush))

    async def _do_flush(self, do_flush: Callable[[], Awaitable[None]]) -> None:
        if self._completed or self._flush_in_progress:
            self._needs_reflush = True
            return

        self._flush_in_progress = True
        self._needs_reflush = False
        try:
            await do_flush()
        except Exception:
            _logger.debug("flush error suppressed", exc_info=True)
        finally:
            self._flush_in_progress = False
            self._last_update_time = time.monotonic()
            # wake waiters
            resolvers = self._flush_resolvers
            self._flush_resolvers = []
            for r in resolvers:
                if not r.done():
                    r.set_result(None)

        # new data arrived during the flush → refresh again immediately
        if self._needs_reflush and not self._completed:
            self._needs_reflush = False
            self._loop.call_soon(asyncio.create_task, self._do_flush(do_flush))

    def _cancel_timer(self) -> None:
        if self._pending_timer is not None:
            self._pending_timer.cancel()
            self._pending_timer = None
