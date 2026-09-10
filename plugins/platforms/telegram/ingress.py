"""Bounded pre-dispatch assembly of Telegram comments and forwarded media.

The normal adapter buffers still own text splits, albums and voice bursts. This
coordinator joins their *ready* events before the base adapter claims a turn;
it never reads or writes the base adapter's busy/FIFO queue.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from gateway.platforms.base import MessageType, _invalidate_pending_stt_cache

if TYPE_CHECKING:
    from gateway.platforms.base import MessageEvent

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngressToken:
    key: str
    epoch: int
    received: float


@dataclass
class StartupBatch:
    token: IngressToken
    admission_end: float
    drain_end: float
    events: list = field(default_factory=list)
    later: list = field(default_factory=list)
    task: asyncio.Task | None = None
    sealed: bool = False


class TelegramIngress:
    ADMISSION_SECONDS = 0.8
    DRAIN_SECONDS = 5.0

    def __init__(self, adapter):
        self.adapter = adapter
        self.ADMISSION_SECONDS = min(self.ADMISSION_SECONDS, getattr(adapter, "_media_batch_delay_seconds", 0.8))
        self.epochs: dict[str, int] = {}
        self.downloads: dict[int, IngressToken] = {}
        self.batches: dict[str, StartupBatch] = {}
        self.tasks: set[asyncio.Task] = set()

    def stamp(self, event, update_id=None):
        token = getattr(event, "_telegram_ingress_token", None)
        if token is None:
            token = self.downloads.get(update_id)
            if token is None:
                key = self.adapter._text_batch_key(event)
                token = IngressToken(key, self.epochs.setdefault(key, 0), time.monotonic())
            event._telegram_ingress_token = token
        return token

    def current(self, event, key=None):
        token = getattr(event, "_telegram_ingress_token", None)
        return token is None or (
            self.epochs.get(token.key, 0) == token.epoch
            and (key is None or key == token.key)
        )

    def reserve(self, update_id, event):
        token = self.stamp(event)
        self.downloads.setdefault(update_id, token)
        return self.downloads[update_id]

    def finish(self, update_id, token):
        if self.downloads.get(update_id) is token:
            self.downloads.pop(update_id, None)

    def _buffers(self):
        a = self.adapter
        for buffer_name, tasks_name in (
            ("_pending_text_batches", "_pending_text_batch_tasks"),
            ("_pending_photo_batches", "_pending_photo_batch_tasks"),
            ("_media_group_events", "_media_group_tasks"),
            ("_pending_voice_batches", "_pending_voice_batch_tasks"),
        ):
            buffer = getattr(a, buffer_name, {})
            tasks = getattr(a, tasks_name, {})
            for key, value in list(buffer.items()):
                events = value if isinstance(value, list) else [value]
                yield buffer, tasks, key, events

    @staticmethod
    def inline_forward(event):
        origin = event.forward_origin
        if not origin:
            return
        parts = ["Forwarded message"]
        for label, field_name in (("From", "sender_name"), ("Chat", "chat_name"),
                                  ("Author", "author_signature"), ("Date", "date")):
            if origin.get(field_name):
                parts.append(f"{label}: {origin[field_name]}")
        event.text = "[" + " | ".join(parts) + "]\n" + (event.text or "")
        event.forward_origin = None
        event._telegram_forwarded = True

    @staticmethod
    def _is_media(event):
        return bool(event.media_urls) or bool(
            event.forward_origin or getattr(event, "_telegram_forwarded", False)
        )

    def _matches(self, event, batch):
        token = self.stamp(event)
        return (token.key == batch.token.key and token.epoch == batch.token.epoch
                and token.received <= batch.admission_end
                and event.message_type != MessageType.COMMAND
                and not event.internal)

    def _pending(self, batch):
        if any(t.key == batch.token.key and t.epoch == batch.token.epoch
               and t.received <= batch.admission_end for t in self.downloads.values()):
            return True
        return any(any(self._matches(e, batch) for e in events)
                   for _, _, _, events in self._buffers())

    def _media_pending(self, token, event):
        if any(t.key == token.key and t.epoch == token.epoch
               and update_id != event.platform_update_id for update_id, t in self.downloads.items()):
            return True
        return any(self.stamp(e).key == token.key and self.current(e) and self._is_media(e)
                   for _, _, _, events in self._buffers() for e in events)

    def _take_buffers(self, batch):
        for buffer, tasks, key, events in self._buffers():
            # Buffer helpers only merge events within the same session. Never
            # partially consume a buffer with a command or a post-admission item.
            if events and all(self._matches(e, batch) for e in events):
                buffer.pop(key, None)
                task = tasks.pop(key, None)
                if task is not None and task is not asyncio.current_task():
                    task.cancel()
                batch.events.extend(events)

    async def offer(self, event):
        token = self.stamp(event)
        if not self.current(event):
            return
        self.inline_forward(event)
        batch = self.batches.get(token.key)
        if batch is not None and event.message_type != MessageType.COMMAND and not event.internal:
            (batch.events if not batch.sealed and self._matches(event, batch) else batch.later).append(event)
            return
        # A running turn owns the regular busy queue, not this coordinator.
        busy = token.key in getattr(self.adapter, "_active_sessions", {})
        if (busy or event.internal or event.message_type == MessageType.COMMAND
                or not (self._is_media(event) and getattr(event, "_telegram_forwarded", False)
                        or self._media_pending(token, event))):
            await self.adapter.handle_message(event)
            return
        now = time.monotonic()
        batch = StartupBatch(token, now + self.ADMISSION_SECONDS, now + self.DRAIN_SECONDS, [event])
        self.batches[token.key] = batch
        task = asyncio.create_task(self._collect(batch))
        batch.task = task
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    @staticmethod
    def merge(events):
        # Numeric Telegram message IDs preserve user order despite downloads
        # finishing out of order. The input list is stable for synthetic events.
        def order(event):
            mid = event.message_id or event.source.message_id
            return int(mid) if str(mid or "").isdigit() else 0
        events = sorted(events, key=order)
        result = events[0]
        TelegramIngress.inline_forward(result)
        # Copy lists: callers/tests may retain the individual physical event.
        result.media_urls = list(result.media_urls)
        result.media_types = list(result.media_types)
        for event in events[1:]:
            TelegramIngress.inline_forward(event)
            if event.text:
                result.text = f"{result.text}\n\n{event.text}" if result.text else event.text
            result.media_urls.extend(event.media_urls)
            result.media_types.extend(event.media_types)
            if event.media_urls and result.message_type == MessageType.TEXT:
                result.message_type = event.message_type
        _invalidate_pending_stt_cache(result)
        return result

    async def _collect(self, batch):
        key = batch.token.key
        try:
            while self.epochs.get(key, 0) == batch.token.epoch:
                self._take_buffers(batch)
                now = time.monotonic()
                if now >= batch.drain_end or (now >= batch.admission_end and not self._pending(batch)):
                    break
                await asyncio.sleep(0.02)
            if self.epochs.get(key, 0) != batch.token.epoch:
                return
            self._take_buffers(batch)
            batch.sealed = True
            item_count = len(batch.events)
            event = self.merge(batch.events)
            batch.events = [event]
            # Keep ownership until the final base seam returns. Input arriving
            # during async topic recovery remains ordered behind this turn.
            await self.adapter.handle_message(event)
            logger.info("[Telegram] Coalesced startup session=%s items=%d media=%d",
                        key, item_count, len(event.media_urls))
            if self.batches.get(key) is batch:
                self.batches.pop(key)
            for later in batch.later:
                if self.current(later):
                    await self.adapter.handle_message(later)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[Telegram] Startup ingress dispatch failed session=%s", key)
            # Preserve ready payloads for the adapter's existing recovery path.
            for event in batch.events + batch.later:
                if self.current(event):
                    self.adapter._hold_inbound_event(event, where="startup-dispatch-failed")
        finally:
            if self.batches.get(key) is batch:
                self.batches.pop(key)

    def invalidate(self, key):
        self.epochs[key] = self.epochs.get(key, 0) + 1
        batch = self.batches.pop(key, None)
        if batch and batch.task and batch.task is not asyncio.current_task():
            batch.task.cancel()
        for buffer, tasks, buffer_key, events in self._buffers():
            if any(self.stamp(e).key == key for e in events):
                buffer.pop(buffer_key, None)
                task = tasks.pop(buffer_key, None)
                if task and task is not asyncio.current_task():
                    task.cancel()
        # Keep in-flight reservations as tombstones until callback completion;
        # a scheduled pre-boundary callback must not acquire a fresh epoch.

    async def close(self):
        # Retryable reconnect preserves *ready* input, but never resurrects
        # downloads or callbacks that belonged to the old connection epoch.
        if not self.adapter._is_permanent_fatal():
            for batch in list(self.batches.values()):
                for event in batch.events + batch.later:
                    if self.current(event):
                        self.adapter._hold_inbound_event(event, where="startup-teardown", schedule=False)
        ready = [e for e in getattr(self.adapter, "_held_inbound_events", []) if self.current(e)]
        keys = set(self.epochs) | set(self.batches) | {t.key for t in self.downloads.values()}
        keys.update(self.stamp(e).key for _, _, _, events in self._buffers() for e in events)
        for key in keys:
            self.invalidate(key)
        reconnect_epochs = dict(self.epochs)
        tasks = list(self.tasks)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for event in ready:
            old = getattr(event, "_telegram_ingress_token", None)
            if old is not None:
                event._telegram_ingress_token = IngressToken(old.key, reconnect_epochs[old.key], old.received)
