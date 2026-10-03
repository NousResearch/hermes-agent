"""Telegram text batching and tracked delayed dispatch."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from gateway.platforms.event import MessageEvent

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
        if self._should_drop_delayed_delivery():
            self._hold_inbound_event(event, where="text-enqueue")
            return
        super()._enqueue_text_event(event)
        self._accept_update()

    def _text_batch_boundary(self: TelegramAdapter, key: str, existing: MessageEvent | None,
                             event: MessageEvent) -> MessageEvent | None:
        """Flush ``existing`` first when ``event`` carries an incompatible reply context."""
        if existing is None or self._text_batch_context_compatible(existing, event):
            return existing
        prior_task = self._pending_text_batch_tasks.pop(key, None)
        if prior_task and not prior_task.done():
            prior_task.cancel()
        self._pending_text_batches.pop(key, None)
        logger.info(
            "[Telegram] Flushing text batch %s before incompatible reply context",
            key,
        )
        # One-shot flush through the normal body (holds on disconnect/cancel, no hold-queue hop);
        # it is armed before the new chunk's flush, whose delay is never shorter.
        task = asyncio.create_task(self._flush_buffered({key: existing}, {}, key, 0, "text-boundary"))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return None


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


    async def _flush_text_batch(self: TelegramAdapter, key: str) -> None:
        """Telegram keeps its own flush body: a cancel after the pop must HOLD the event and re-raise
        (PTB already acked the update; the hold queue redispatches after reconnect) rather than shield
        the dispatch — teardown must be able to stop a flush from reaching a torn-down session."""
        await self._flush_buffered(
            self._pending_text_batches, self._pending_text_batch_tasks, key,
            self._text_batch_delay_for(self._pending_text_batches.get(key)), "text",
            lambda ev: logger.info("[Telegram] Flushing text batch %s (%d chars)", key, len(ev.text or "")))

    @staticmethod
    def _text_batch_has_reply_context(event: MessageEvent) -> bool:
        return any(value not in (None, "", False) for value in event.reply_context())

    def _text_batch_context_compatible(
        self: TelegramAdapter,
        existing: MessageEvent,
        incoming: MessageEvent,
    ) -> bool:
        """Return whether coalescing preserves the reply/quote semantics.

        Telegram may attach reply metadata only to the first near-limit chunk
        of a client-split long message. A following metadata-free chunk can
        inherit that first chunk's reply context.
        """
        if existing.reply_context() == incoming.reply_context():
            return True

        existing_last_len = getattr(
            existing,
            "_last_chunk_len",
            len(existing.text or ""),
        )
        return (
            existing_last_len >= self._SPLIT_THRESHOLD
            and not self._text_batch_has_reply_context(incoming)
        )
