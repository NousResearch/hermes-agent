"""Tracked photo and album batching for the Telegram adapter."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from gateway.platforms.event import MessageEvent
from gateway.platforms.base_pending import can_join_pending_event, merge_recorded, ingress_order
from gateway.platforms.base_pending_merge import _absorb_pending_media

if TYPE_CHECKING:
    from telegram import Message
    from plugins.platforms.telegram.adapter import TelegramAdapter

logger = logging.getLogger("plugins.platforms.telegram.adapter")


class TelegramMediaBatchingMixin:
    def _photo_batch_key(self: TelegramAdapter, event: MessageEvent, msg: Message) -> str:
        """Return a batching key for Telegram photos/albums."""
        session_key = self._event_session_key(event)
        media_group_id = getattr(msg, "media_group_id", None)
        return f"{session_key}:album:{media_group_id}" if media_group_id else f"{session_key}:photo-burst"

    async def _flush_photo_batch(self: TelegramAdapter, batch_key: str) -> None:
        """Send a buffered photo burst/album as a single MessageEvent."""
        await self._flush_buffered(
            self._pending_photo_batches, self._pending_photo_batch_tasks, batch_key, self._media_batch_delay_seconds, "photo",
            lambda ev: logger.info("[Telegram] Flushing photo batch %s with %d image(s)", batch_key, len(ev.media_urls)))

    def _enqueue_photo_event(self: TelegramAdapter, batch_key: str, event: MessageEvent) -> None:
        """Merge photo events into a pending batch and schedule flush."""
        ingress_order(event)
        if self._should_drop_delayed_delivery():
            self._hold_inbound_event(event, where="photo-enqueue")
            return
        self._merge_into_pending(self._pending_photo_batches, batch_key, event)
        self._accept_update()
        prior_task = self._pending_photo_batch_tasks.get(batch_key)
        if prior_task and not prior_task.done():
            prior_task.cancel()
        self._pending_photo_batch_tasks[batch_key] = asyncio.create_task(self._flush_photo_batch(batch_key))

    async def _route_photo_event(self: TelegramAdapter, msg, event: MessageEvent) -> None:
        """Album items debounce on media_group_id; singles go through the photo burst batcher."""
        if self._drop_unresolved(event):  # identity FIRST: the batch lane is derived from it
            return
        media_group_id = getattr(msg, "media_group_id", None)
        if media_group_id:
            await self._queue_media_group_event(str(media_group_id), event)
        else:
            self._enqueue_photo_event(self._photo_batch_key(event, msg), event)

    async def _queue_media_group_event(self: TelegramAdapter, media_group_id: str, event: MessageEvent) -> None:
        """Debounce album items (shared media_group_id) into one MessageEvent so the second image isn't
        treated as a new message interrupting the first."""
        ingress_order(event)
        if self._should_drop_delayed_delivery():
            self._hold_inbound_event(event, where="media-group-enqueue")
            return
        self._merge_into_pending(self._media_group_events, media_group_id, event)
        self._accept_update()
        prior_task = self._media_group_tasks.get(media_group_id)
        if prior_task:
            prior_task.cancel()
        self._media_group_tasks[media_group_id] = asyncio.create_task(self._flush_media_group_event(media_group_id))

    async def _flush_media_group_event(self: TelegramAdapter, media_group_id: str) -> None:
        await self._flush_buffered(
            self._media_group_events, self._media_group_tasks, media_group_id, self.MEDIA_GROUP_WAIT_SECONDS, "media-group")

    def _merge_into_pending(self: TelegramAdapter, pending: dict, key: str, event: MessageEvent) -> None:
        """Merge ``event`` into ``pending[key]`` (media + caption) or seed it."""
        existing = pending.get(key)
        if existing is None:
            pending[key] = event
            return
        if not can_join_pending_event(existing, event):
            pending[key] = event
            self._hold_inbound_event(existing, where="media-reply-context-boundary")
            return
        merge_recorded(existing, event, _absorb_pending_media)
