"""Telegram text batching and tracked delayed dispatch."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from gateway.platforms.event import MessageEvent
from gateway.platforms.base_pending import _can_join_pending_event, ingress_order

if TYPE_CHECKING:
    from plugins.platforms.telegram.adapter import TelegramAdapter

logger = logging.getLogger("plugins.platforms.telegram.adapter")


class TelegramTextBatchingMixin:
    def _text_batch_key(self: TelegramAdapter, event: MessageEvent) -> str:
        """Session-scoped batching key; topic recovery first so DM-topic batches coalesce on the recovered lane."""
        self._apply_topic_recovery(event)
        return super()._text_batch_key(event)


    def _enqueue_text_event(self: TelegramAdapter, event: MessageEvent) -> None:
        """Buffer a text chunk, or hold it while delayed delivery must be dropped."""
        ingress_order(event)
        if self._should_drop_delayed_delivery():
            self._hold_inbound_event(event, where="text-enqueue")
            return
        key = self._text_batch_key(event)
        existing = self._pending_text_batches.get(key)
        if existing is not None and (
            not self._text_batch_context_compatible(existing, event)
            or not _can_join_pending_event(existing, event)
        ):
            prior_task = self._pending_text_batch_tasks.pop(key, None)
            if prior_task and not prior_task.done():
                prior_task.cancel()
            self._pending_text_batches.pop(key, None)
            logger.info(
                "[Telegram] Flushing text batch %s before incompatible reply context",
                key,
            )
            self._hold_inbound_event(existing, where="text-reply-context-boundary")
        super()._enqueue_text_event(event)
        self._accept_update()


    async def _flush_buffered(self: TelegramAdapter, pending: dict, tasks: dict, key: str, delay: float, where: str, log_fn=None) -> None:
        """Shared delayed-flush body: sleep, pop, hold if teardown started, else dispatch. A cancel after
        the pop but before durable dispatch re-holds the event (never lose it)."""
        current_task = asyncio.current_task()
        event = None
        try:
            await asyncio.sleep(delay)
            # Superseded flush (a newer chunk re-armed the timer while our sleep was already done):
            # CancelledError only lands at the next await, so check synchronously before the pop.
            owner = tasks.get(key)
            if owner is not None and owner is not current_task:
                return
            event = pending.pop(key, None)
            if not event:
                return
            if self._should_drop_delayed_delivery():
                self._hold_inbound_event(event, where=f"{where}-flush")
                event = None
                return
            if log_fn is not None:
                log_fn(event)
            await self.handle_message(event)
            event = None
        except asyncio.CancelledError:
            if event is not None:
                self._hold_inbound_event(event, where=f"{where}-flush-cancelled")
            raise
        finally:
            if tasks.get(key) is current_task:
                tasks.pop(key, None)


    async def _flush_text_batch(self: TelegramAdapter, key: str, *, delay: float | None = None) -> None:
        """Telegram keeps its own flush body: a cancel after the pop must HOLD the event and re-raise
        (PTB already acked the update; the hold queue redispatches after reconnect) rather than shield
        the dispatch — teardown must be able to stop a flush from reaching a torn-down session."""
        await self._flush_buffered(
            self._pending_text_batches, self._pending_text_batch_tasks, key,
            self._text_batch_delay_for(self._pending_text_batches.get(key)) if delay is None else delay, "text",
            lambda ev: logger.info("[Telegram] Flushing text batch %s (%d chars)", key, len(ev.text or "")))
