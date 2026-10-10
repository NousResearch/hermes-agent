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
        """Cancel delayed deliveries and retain pending events for reconnect."""
        current_task = asyncio.current_task()
        boundary = self._text_boundary_buffer()
        pending_tasks = self._collect_live_tasks(
            [
                *boundary.tasks.values(),
                *self._media_group_tasks.values(),
                *self._pending_photo_batch_tasks.values(),
                *self._pending_text_batch_tasks.values(),
                getattr(self, "_polling_error_task", None),
                getattr(self, "_polling_progress_verifier_task", None),
                # Hold-queue redispatch must be cancellable+awaitable on teardown too.
                getattr(self, "_held_inbound_redispatch_task", None),
            ],
            current_task,
        )
        awaitable_tasks = [
            t for t in pending_tasks if asyncio.isfuture(t) or asyncio.iscoroutine(t)
        ]
        for task in pending_tasks:
            task.cancel()
        if awaitable_tasks:
            await asyncio.gather(*awaitable_tasks, return_exceptions=True)
        # A permanent failure cannot reconnect to drain the retained events.
        if self._is_permanent_fatal():
            n_pending = (
                len(boundary.pending)
                + len(self._pending_text_batches)
                + len(self._pending_photo_batches)
                + len(self._media_group_events)
            )
            if n_pending:
                logger.warning(
                    "[Telegram] Non-retryable fatal teardown; discarding %d pending inbound batch(es)",
                    n_pending,
                )
        else:
            for events, where in (
                (boundary.pending, "text-boundary-teardown"),
                (self._pending_text_batches, "text-batch-teardown"),
                (self._pending_photo_batches, "photo-batch-teardown"),
                (self._media_group_events, "media-group-teardown"),
            ):
                for event in list(events.values()):
                    self._hold_inbound_event(event, where=where)
        for d in (
            boundary.pending,
            boundary.tasks,
            self._media_group_tasks,
            self._media_group_events,
            self._pending_photo_batch_tasks,
            self._pending_photo_batches,
            self._pending_text_batch_tasks,
            self._pending_text_batches,
        ):
            d.clear()
        self._clear_task_attrs_except(
            current_task,
            "_polling_error_task",
            "_polling_progress_verifier_task",
            "_held_inbound_redispatch_task",
        )
