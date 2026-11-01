"""Finite admission for Yuanbao's raw debounce and per-group dispatch stages."""

from __future__ import annotations

import asyncio
import collections
import logging
from typing import Any, Callable

from gateway.platforms.event import MessageEvent, MessageType

logger = logging.getLogger(__name__)

INBOUND_MAX_RETAINED_FRAMES = 64
INBOUND_MAX_RETAINED_BYTES = 4 * 1024 * 1024
DEBOUNCE_MAX_FRAMES_PER_BATCH = 8
DEBOUNCE_MAX_BYTES_PER_BATCH = 512 * 1024
INBOUND_PIPELINE_WORKERS = 4

GROUP_QUEUE_MAX_MESSAGES = 32
GROUP_QUEUE_MAX_BYTES = 1024 * 1024
GROUP_GLOBAL_MAX_MESSAGES = 256
GROUP_GLOBAL_MAX_BYTES = 8 * 1024 * 1024
GROUP_MAX_ACTIVE_QUEUES = 64
GROUP_OVERLOAD_NOTICE_INTERVAL_S = 30.0
GROUP_OVERLOAD_NOTICE_KEYS = 256
GROUP_OVERLOAD_NOTICE = (
    "Hermes is handling as much from this chat as it can right now. This message was not queued; "
    "please resend it after the current turn finishes. Owner controls such as /stop are still available."
)


class InboundAdmission:
    """Bound raw work across queueing, fixed-age debounce, and pipeline execution."""

    def __init__(self, adapter, key_fn: Callable[[bytes], str], execution_key_fn: Callable[[bytes], str],
                 context_factory: Callable[[list[bytes]], Any], debounce_window: Callable[[], float]) -> None:
        self._adapter = adapter
        self._key_fn = key_fn
        self._execution_key_fn = execution_key_fn
        self._context_factory = context_factory
        self._debounce_window = debounce_window
        self.accepting = True
        self.capacity_changed = asyncio.Condition()
        self.retained_frames = 0
        self.retained_bytes = 0
        self.raw_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=INBOUND_MAX_RETAINED_FRAMES)
        self.ready: asyncio.Queue[tuple[str, list[bytes]]] = asyncio.Queue(maxsize=INBOUND_MAX_RETAINED_FRAMES)
        self.ingest_task: asyncio.Task | None = None
        self.pipeline_tasks: set[asyncio.Task] = set()
        self.active_keys: set[str] = set()
        self.pending: dict[str, collections.deque[list[bytes]]] = {}
        self.buffer: dict[str, list[bytes]] = {}
        self.buffer_bytes: dict[str, int] = {}
        self.timers: dict[str, asyncio.TimerHandle] = {}

    def reopen(self) -> None:
        self.accepting = True

    def _start_workers(self) -> None:
        if self.ingest_task is None or self.ingest_task.done():
            self.ingest_task = asyncio.create_task(self._ingest_loop(), name="yuanbao-inbound-ingest")
        self.pipeline_tasks = {task for task in self.pipeline_tasks if not task.done()}
        while len(self.pipeline_tasks) < INBOUND_PIPELINE_WORKERS:
            task = asyncio.create_task(self._pipeline_loop(), name=f"yuanbao-inbound-pipeline-{len(self.pipeline_tasks) + 1}")
            self.pipeline_tasks.add(task)
            task.add_done_callback(self.pipeline_tasks.discard)

    async def admit(self, raw_data: bytes) -> bool:
        """Wait for finite capacity; the single WS reader is the only production waiter."""
        size = len(raw_data)
        if size > INBOUND_MAX_RETAINED_BYTES:
            logger.warning("[%s] Inbound frame exceeds Yuanbao byte budget (%d > %d); withholding ACK",
                           self._adapter.name, size, INBOUND_MAX_RETAINED_BYTES)
            return False
        self._start_workers()
        async with self.capacity_changed:
            await self.capacity_changed.wait_for(lambda: not self.accepting or (
                self.retained_frames < INBOUND_MAX_RETAINED_FRAMES
                and self.retained_bytes + size <= INBOUND_MAX_RETAINED_BYTES))
            if not self.accepting:
                return False
            self.retained_frames += 1
            self.retained_bytes += size
        # This cannot be full after reserving one of the same global frame slots.
        self.raw_queue.put_nowait(raw_data)
        return True

    async def _ingest_loop(self) -> None:
        while True:
            raw_data = await self.raw_queue.get()
            try:
                self._buffer(raw_data)
            finally:
                self.raw_queue.task_done()

    def _buffer(self, raw_data: bytes) -> None:
        key = self._key_fn(raw_data)
        frames = self.buffer.get(key)
        size = self.buffer_bytes.get(key, 0)
        if frames and (len(frames) >= DEBOUNCE_MAX_FRAMES_PER_BATCH
                       or size + len(raw_data) > DEBOUNCE_MAX_BYTES_PER_BATCH):
            self.flush(key)
            frames, size = None, 0
        if frames is None:
            frames = self.buffer[key] = []
            self.timers[key] = asyncio.get_running_loop().call_later(self._debounce_window(), self.flush, key)
        frames.append(raw_data)
        self.buffer_bytes[key] = size + len(raw_data)
        logger.debug("[%s] Debounce: buffered frame for key=%s, count=%d bytes=%d",
                     self._adapter.name, key, len(frames), self.buffer_bytes[key])
        if (len(frames) >= DEBOUNCE_MAX_FRAMES_PER_BATCH
                or self.buffer_bytes[key] >= DEBOUNCE_MAX_BYTES_PER_BATCH):
            self.flush(key)

    def flush(self, key: str) -> None:
        """Move one batch to fixed workers; batches for the same sender remain FIFO."""
        timer = self.timers.pop(key, None)
        if timer is not None:
            timer.cancel()
        frames = self.buffer.pop(key, [])
        self.buffer_bytes.pop(key, None)
        if not frames:
            return
        logger.info("[%s] Debounce flush: key=%s, aggregated %d frames", self._adapter.name, key, len(frames))
        execution_key = self._execution_key_fn(frames[0])
        if execution_key in self.active_keys:
            self.pending.setdefault(execution_key, collections.deque()).append(frames)
        else:
            self.active_keys.add(execution_key)
            self.ready.put_nowait((execution_key, frames))

    async def _release(self, frame_count: int, frame_bytes: int) -> None:
        async with self.capacity_changed:
            self.retained_frames = max(0, self.retained_frames - frame_count)
            self.retained_bytes = max(0, self.retained_bytes - frame_bytes)
            self.capacity_changed.notify_all()

    async def _pipeline_loop(self) -> None:
        while True:
            key, frames = await self.ready.get()
            frame_count = len(frames)
            frame_bytes = sum(map(len, frames))
            try:
                await self._adapter._inbound_pipeline.execute(self._context_factory(frames))
            except Exception:
                logger.exception("[%s] Inbound pipeline failed for key=%s", self._adapter.name, key)
            finally:
                await self._release(frame_count, frame_bytes)
                pending = self.pending.get(key)
                if self.accepting and pending:
                    self.ready.put_nowait((key, pending.popleft()))
                    if not pending:
                        self.pending.pop(key, None)
                else:
                    self.pending.pop(key, None)
                    self.active_keys.discard(key)
                self.ready.task_done()

    async def close(self) -> None:
        self.accepting = False
        async with self.capacity_changed:
            self.capacity_changed.notify_all()
        for timer in self.timers.values():
            timer.cancel()
        self.timers.clear()
        tasks = [task for task in [self.ingest_task, *self.pipeline_tasks] if task]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.ingest_task = None
        self.pipeline_tasks.clear()
        self.buffer.clear()
        self.buffer_bytes.clear()
        self.active_keys.clear()
        self.pending.clear()
        self._drain_queue(self.raw_queue)
        self._drain_queue(self.ready)
        async with self.capacity_changed:
            self.retained_frames = self.retained_bytes = 0
            self.capacity_changed.notify_all()

    @staticmethod
    def _drain_queue(queue: asyncio.Queue) -> None:
        while not queue.empty():
            queue.get_nowait()
            queue.task_done()


class GroupDispatchState:
    """Bound post-authorization group work and rate-limit explicit overflow notices."""

    def __init__(self) -> None:
        self.queues: dict[str, asyncio.Queue] = {}
        self.queued_bytes: dict[str, int] = {}
        self.notices: collections.OrderedDict[str, float] = collections.OrderedDict()
        self.retained_messages = 0
        self.retained_bytes = 0
        self.accepted = 0
        self.rejected = 0

    async def admit(self, adapter, session_key: str, event: MessageEvent, payload_bytes: int,
                    chat_id: str) -> tuple[asyncio.Queue | None, bool]:
        queue = self.queues.get(session_key)
        queued_bytes = self.queued_bytes.get(session_key, 0)
        reason = None
        if queue is not None and queue.full():
            reason = "message capacity"
        elif payload_bytes > GROUP_QUEUE_MAX_BYTES or queued_bytes + payload_bytes > GROUP_QUEUE_MAX_BYTES:
            reason = "payload capacity"
        elif queue is None and len(self.queues) >= GROUP_MAX_ACTIVE_QUEUES:
            reason = "active-group capacity"
        elif self.retained_messages >= GROUP_GLOBAL_MAX_MESSAGES:
            reason = "global message capacity"
        elif self.retained_bytes + payload_bytes > GROUP_GLOBAL_MAX_BYTES:
            reason = "global payload capacity"
        if reason is not None:
            await self._reject(adapter, session_key, chat_id, reason)
            return None, False
        is_new = queue is None
        if queue is None:
            queue = self.queues[session_key] = asyncio.Queue(maxsize=GROUP_QUEUE_MAX_MESSAGES)
        queue.put_nowait((event, payload_bytes))
        self.queued_bytes[session_key] = queued_bytes + payload_bytes
        self.retained_messages += 1
        self.retained_bytes += payload_bytes
        self.accepted += 1
        logger.info("[%s] Group message enqueued (qsize=%d bytes=%d) for %s", adapter.name,
                    queue.qsize(), self.queued_bytes[session_key], (session_key or "")[:50])
        return queue, is_new

    async def _reject(self, adapter, session_key: str, chat_id: str, reason: str) -> None:
        self.rejected += 1
        now = asyncio.get_running_loop().time()
        last = self.notices.get(session_key)
        if last is not None and now - last < GROUP_OVERLOAD_NOTICE_INTERVAL_S:
            return
        self.notices[session_key] = now
        self.notices.move_to_end(session_key)
        while len(self.notices) > GROUP_OVERLOAD_NOTICE_KEYS:
            self.notices.popitem(last=False)
        queue = self.queues.get(session_key)
        logger.warning("[%s] Group inbound rejected: reason=%s session=%s queued=%d bytes=%d rejected_total=%d",
                       adapter.name, reason, (session_key or "")[:50], queue.qsize() if queue else 0,
                       self.queued_bytes.get(session_key, 0), self.rejected)
        try:
            await adapter.send(chat_id, GROUP_OVERLOAD_NOTICE)
        except Exception:
            logger.warning("[%s] Failed to send group-overload notice", adapter.name, exc_info=True)

    def dequeued(self, session_key: str, payload_bytes: int) -> None:
        self.queued_bytes[session_key] = max(0, self.queued_bytes.get(session_key, 0) - payload_bytes)

    def completed(self, payload_bytes: int) -> None:
        self.retained_messages = max(0, self.retained_messages - 1)
        self.retained_bytes = max(0, self.retained_bytes - payload_bytes)

    def cleanup(self, session_key: str, queue: asyncio.Queue) -> None:
        if self.queues.get(session_key) is queue:
            self.queues.pop(session_key, None)
            self.queued_bytes.pop(session_key, None)

    def clear(self) -> None:
        self.queues.clear()
        self.queued_bytes.clear()
        self.notices.clear()
        self.retained_messages = 0
        self.retained_bytes = 0


class BoundedDispatchMixin:
    """Dispatch implementation kept outside the already-oversized adapter module."""

    @staticmethod
    def _event_from_context(ctx) -> MessageEvent:
        if any(mt.startswith(("application/", "text/")) for mt in ctx.media_types):
            msg_type = MessageType.DOCUMENT
        else:
            msg_type = ctx.msg_type if isinstance(ctx.msg_type, MessageType) else MessageType.TEXT
        return MessageEvent(
            text=ctx.raw_text, message_type=msg_type, source=ctx.source, message_id=ctx.msg_id or None,
            raw_message=ctx.push, media_urls=list(ctx.media_urls), media_types=list(ctx.media_types),
            reply_to_message_id=ctx.reply_to_message_id, reply_to_text=ctx.reply_to_text,
            channel_prompt=ctx.channel_prompt,
        )

    @staticmethod
    def _retained_payload_bytes(ctx) -> int:
        raw_bytes = sum(len(frame) for frame in ctx.raw_frames if isinstance(frame, (bytes, bytearray)))
        text_bytes = sum(len(str(value or "").encode("utf-8")) for value in (
            ctx.raw_text, ctx.reply_to_text, ctx.channel_prompt, *ctx.media_urls, *ctx.media_types))
        return max(1, raw_bytes, text_bytes)

    @staticmethod
    async def _dispatch_event(adapter, session_key: str, event: MessageEvent) -> None:
        if session_key and event.message_id:
            adapter._processing_msg_ids[session_key] = event.message_id
            adapter._processing_msg_texts[session_key] = event.text or ""
        if event.message_id and event.text:
            cache = adapter._msg_content_cache
            cache[event.message_id] = event.text
            for key in list(cache)[:max(0, len(cache) - 200)]:
                del cache[key]
        await adapter.handle_message(event)

    async def handle_bounded(self, ctx, next_fn) -> None:
        adapter = ctx.adapter
        session_key = adapter._source_session_key(ctx.source)
        event = self._event_from_context(ctx)
        payload_bytes = self._retained_payload_bytes(ctx)
        ctx.raw_frames.clear()
        if ctx.chat_type == "group" and not ctx.owner_command:
            queue, is_new = await adapter._group_dispatch.admit(
                adapter, session_key, event, payload_bytes, ctx.chat_id)
            if queue is not None and is_new:
                self._track_inbound(adapter, self._consume_group_queue(adapter, session_key),
                                    f"yuanbao-group-consumer-{(session_key or '')[:30]}")
        elif ctx.chat_type == "group":
            # Only OwnerCommandMiddleware can set this flag after checking bot_owner_id.
            await self._dispatch_event(adapter, session_key, event)
        else:
            self._track_inbound(adapter, self._dispatch_event(adapter, session_key, event),
                                f"yuanbao-inbound-{ctx.msg_id or 'unknown'}")
        await next_fn()

    @staticmethod
    def _track_inbound(adapter, coro, name: str) -> None:
        task = asyncio.create_task(coro, name=name)
        adapter._inbound_tasks.add(task)
        task.add_done_callback(adapter._inbound_tasks.discard)

    @staticmethod
    async def _consume_group_queue(adapter, session_key: str) -> None:
        queue = adapter._group_dispatch.queues.get(session_key)
        if not queue:
            return
        try:
            while True:
                try:
                    event, payload_bytes = await asyncio.wait_for(queue.get(), timeout=2.0)
                except TimeoutError:
                    if queue.empty():
                        break
                    continue
                adapter._group_dispatch.dequeued(session_key, payload_bytes)
                try:
                    await BoundedDispatchMixin._dispatch_event(adapter, session_key, event)
                    while session_key in adapter._active_sessions:
                        await asyncio.sleep(0.1)
                except Exception:
                    logger.exception("[%s] Group queue consumer error", adapter.name)
                finally:
                    adapter._group_dispatch.completed(payload_bytes)
        finally:
            adapter._group_dispatch.cleanup(session_key, queue)
