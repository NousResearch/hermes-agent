"""Text batching for gateway adapters."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Optional

from gateway.platforms.base_pending import (
    _PendingDispatchReservation, can_join_pending_event, ingress_order,
    merge_recorded, pending_dispatch_scope, release_pending_dispatch_record,
    reserve_pending_dispatch,
)
from gateway.platforms.base_pending_merge import _append_batched_text
from gateway.platforms.event import MessageEvent

if TYPE_CHECKING:
    from gateway.platforms.base import BasePlatformAdapter

logger = logging.getLogger("gateway.platforms.base")


class BaseTextBatchingMixin:
    def _text_batch_dependencies(self: BasePlatformAdapter) -> dict[str, asyncio.Task]:
        from gateway.platforms.base import _lazy_attr

        return _lazy_attr(self, "_pending_text_batch_dependencies", dict)

    def _text_batch_key(self: BasePlatformAdapter, event: MessageEvent) -> str:
        """Session-scoped key for text batching (subclasses may override)."""
        return self._event_session_key(event)

    def _enqueue_text_event(self: BasePlatformAdapter, event: MessageEvent) -> None:
        """Buffer a text event (merging into a pending one) and restart the flush timer."""
        if self._drop_unresolved(event):
            return
        ingress_order(event)
        key = self._text_batch_key(event)
        existing = self._text_batch_boundary(
            key, self._pending_text_batches.get(key), event
        )
        if existing is None:
            existing = self._pending_text_batches[key] = event
        else:
            merge_recorded(existing, event, _append_batched_text)
        existing._last_chunk_len = len(event.text or "")  # type: ignore[attr-defined]
        prior_task = self._pending_text_batch_tasks.get(key)
        if prior_task and not prior_task.done():
            prior_task.cancel()
        self._pending_text_batch_tasks[key] = asyncio.create_task(
            self._flush_text_batch(key)
        )

    def _text_batch_boundary(
        self: BasePlatformAdapter,
        key: str,
        existing: Optional[MessageEvent],
        event: MessageEvent,
    ) -> Optional[MessageEvent]:
        """Dispatch the completed batch when its reply target conflicts."""
        if existing is None or can_join_pending_event(existing, event):
            return existing

        previous = self._pending_text_batch_tasks.pop(key, None)
        if previous is not None and not previous.done():
            previous.cancel()
        self._pop_text_batch(key)
        boundary_key = f"{key}:boundary:{id(existing)}"
        self._pending_text_batches[boundary_key] = existing
        dependencies = BaseTextBatchingMixin._text_batch_dependencies(self)
        predecessor = dependencies.pop(key, None)
        if predecessor is not None:
            dependencies[boundary_key] = predecessor
        task = asyncio.create_task(BaseTextBatchingMixin._flush_text_batch(self, boundary_key, delay=0))
        self._pending_text_batch_tasks[boundary_key] = task
        dependencies[key] = task
        return None

    async def _dispatch_completed_text_batch(
        self: BasePlatformAdapter, event: MessageEvent
    ) -> None:
        """Dispatch a completed batch with the shared cancellation policy."""
        try:
            await asyncio.shield(self._dispatch_text_batch(event))
        except asyncio.CancelledError:
            pass

    def _text_batch_delay_for(
        self: BasePlatformAdapter, pending: Optional[MessageEvent]
    ) -> float:
        """Quiet period before ``pending`` is dispatched; near-split chunks wait longer."""
        last_len = getattr(pending, "_last_chunk_len", 0) if pending is not None else 0
        return (
            self._text_batch_split_delay_seconds
            if last_len >= self._SPLIT_THRESHOLD
            else self._text_batch_delay_seconds
        )

    def _pop_text_batch(self: BasePlatformAdapter, key: str) -> Optional[MessageEvent]:
        """Remove and return the pending batch for ``key`` (adapters with side tables override)."""
        return self._pending_text_batches.pop(key, None)

    async def _dispatch_text_batch(
        self: BasePlatformAdapter, event: MessageEvent
    ) -> None:
        """Hand a flushed batch to the pipeline (adapters with per-chat guards override)."""
        await self.handle_message(event)

    async def _flush_text_batch_now(self: BasePlatformAdapter, key: str) -> None:
        """Dispatch the pending batch for ``key`` immediately (no quiet period)."""
        event = self._pop_text_batch(key)
        if event is not None:
            await self._dispatch_text_batch(event)


    async def _dispatch_owned_text_batch(
        self: BasePlatformAdapter, event: MessageEvent, session_key: str,
        reservation: _PendingDispatchReservation,
    ) -> None:
        try:
            with pending_dispatch_scope(self, session_key, event):
                await self._dispatch_text_batch(event)
        finally:
            release_pending_dispatch_record(self, session_key, reservation)

    async def _flush_text_batch(self: BasePlatformAdapter, key: str, *, delay: float | None = None) -> None:
        """Wait for the quiet period, then dispatch the batch for ``key``.

        Two races share this body. (1) ``_enqueue_text_event`` cancels the prior flush task
        on each new chunk; when ``Task.cancel()`` lands after ``sleep()`` already completed,
        CancelledError is delivered at the *next* await — after a superseded task would have
        popped the event, so the successor finds nothing and the message is lost. The identity
        check therefore runs synchronously between the sleep and the pop. (2) A cancel that
        lands while the dispatch is in flight would abort the agent turn (#12444), so the
        dispatch is shielded and the outer CancelledError swallowed."""
        current_task = asyncio.current_task()
        try:
            await asyncio.sleep(self._text_batch_delay_for(self._pending_text_batches.get(key)) if delay is None else delay)
            dependencies = BaseTextBatchingMixin._text_batch_dependencies(self)
            predecessor = dependencies.get(key)
            if predecessor is not None:
                await asyncio.gather(asyncio.shield(predecessor), return_exceptions=True)
            owner = self._pending_text_batch_tasks.get(key)
            if owner is not None and owner is not current_task:
                return
            event = self._pop_text_batch(key)
            if event is None:
                return
            logger.info("[%s] Flushing text batch %s (%d chars)", self.name, key, len(event.text or ""))
            session_key = self._event_session_key(event)
            reservation = reserve_pending_dispatch(self, session_key, event, accepted=False)
            dispatch = asyncio.create_task(BaseTextBatchingMixin._dispatch_owned_text_batch(
                self, event, session_key, reservation))
            reservation.task = dispatch
            from gateway.platforms.base import _lazy_attr

            background = _lazy_attr(self, "_background_tasks", set)
            background.add(dispatch)
            dispatch.add_done_callback(background.discard)
            await asyncio.shield(dispatch)
        except asyncio.CancelledError:
            pass
        finally:
            if self._pending_text_batch_tasks.get(key) is current_task:
                self._pending_text_batch_tasks.pop(key, None)
                BaseTextBatchingMixin._text_batch_dependencies(self).pop(key, None)
