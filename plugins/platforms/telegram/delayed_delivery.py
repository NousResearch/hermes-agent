"""Teardown of Telegram's buffered inbound deliveries."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from plugins.platforms.telegram.adapter import TelegramAdapter

logger = logging.getLogger("plugins.platforms.telegram.adapter")


class TelegramDelayedDeliveryMixin:
    async def _cancel_pending_delivery_tasks(self: TelegramAdapter) -> None:
        """Cancel every delayed-delivery task family before disconnect completes (media-group, photo-batch, text-batch flushes plus
        polling recovery all sit behind ``asyncio.sleep()`` and would dispatch ``handle_message`` into a torn-down session)."""
        current_task = asyncio.current_task()
        pending_tasks = self._collect_live_tasks(
            [
                *self._media_group_tasks.values(), *self._pending_photo_batch_tasks.values(), *self._pending_text_batch_tasks.values(),
                getattr(self, "_polling_error_task", None), getattr(self, "_polling_progress_verifier_task", None),
                # Hold-queue redispatch must be cancellable+awaitable on teardown too.
                getattr(self, "_held_inbound_redispatch_task", None),
           ],
            current_task)
        awaitable_tasks = [t for t in pending_tasks if asyncio.isfuture(t) or asyncio.iscoroutine(t)]
        # Hold-queue redispatch must be cancellable+awaitable on teardown so it cannot dispatch
        # handle_message into a torn-down session (same lifecycle rule teknium called out on #72037 for
        # shielded flush dispatch).
        for task in pending_tasks:
            task.cancel()
        if awaitable_tasks:
            await asyncio.gather(*awaitable_tasks, return_exceptions=True)
        # Salvage buffered inbound events before clearing maps — unless permanent fatal, where no
        # reconnect can drain and hold would re-orphan them.
        if self._is_permanent_fatal():
            n_pending = len(self._pending_text_batches) + len(self._pending_photo_batches) + len(self._media_group_events)
            if n_pending:
                logger.warning("[Telegram] Non-retryable fatal teardown; discarding %d pending inbound batch(es)", n_pending)
        else:
            for events, where in (
                (self._pending_text_batches, "text-batch-teardown"), (self._pending_photo_batches, "photo-batch-teardown"),
                (self._media_group_events, "media-group-teardown")):
                for event in list(events.values()):
                    self._hold_inbound_event(event, where=where)
        for d in (
            self._media_group_tasks, self._media_group_events, self._pending_photo_batch_tasks,
            self._pending_photo_batches, self._pending_text_batch_tasks, self._pending_text_batches):
            d.clear()
        self._clear_task_attrs_except(
            current_task, "_polling_error_task", "_polling_progress_verifier_task", "_held_inbound_redispatch_task")

