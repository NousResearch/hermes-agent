"""Telegram forward-with-comment: merge a pending same-sender text batch into media."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from gateway.platforms.event import MessageEvent

logger = logging.getLogger("plugins.platforms.telegram.adapter")


class TelegramSiblingMediaMixin:
    """Claim, wait, and absorb a short text comment into the following media event."""

    # Forward-with-comment is two updates: a short TEXT comment, then media.
    # The text handler returns before its flush sleep, so media from the same
    # poll can claim the pending comment during the normal text delay. A lone
    # short text flushes at that delay. If a same-sender download has claimed
    # this batch, the flush waits up to this long for the comment to be merged
    # into the media event.
    _TEXT_MEDIA_INFLIGHT_CAP_S = 20.0

    def _sibling_media_key(self, event: MessageEvent) -> str:
        """Session+sender key so one user's in-flight media does not hold another's text."""
        return f"{self._text_batch_key(event)}:{self._inbound_sender_id(event)}"

    def _begin_media_inflight(self, key: str) -> int:
        generation = getattr(self, "_media_inflight_generation", 0)
        inflight = getattr(self, "_media_inflight", None)
        if inflight is None:
            self._media_inflight = {}
            inflight = self._media_inflight
        inflight[key] = inflight.get(key, 0) + 1
        return generation

    def _end_media_inflight(self, key: str, generation: Optional[int] = None) -> None:
        if generation is not None and generation != getattr(self, "_media_inflight_generation", 0):
            return
        inflight = getattr(self, "_media_inflight", None)
        if not inflight:
            return
        remaining = inflight.get(key, 0) - 1
        if remaining <= 0:
            inflight.pop(key, None)
            self._signal_sibling_wait(key)
        else:
            inflight[key] = remaining

    def _absorb_pending_text_into_media_event(self, event: MessageEvent) -> bool:
        """Merge a pending text-batch comment into a sibling media event.

        Telegram forward-with-comment is two updates: the user's comment
        (TEXT) and the forwarded post (media + caption). Absorb the comment
        so one turn sees both. Returns True if a pending batch was consumed.
        Same-sender only — a shared topic must not steal another user's text.
        An addressed comment keeps its reply expectation and, when the media
        event has none, the comment's reply quote.

        Lookup uses the admission-time batch key. Group-observe attribution
        may rewrite ``event.source`` after we snapshotted; the pin is the
        conversation the comment was already queued on.
        """
        snapshot = getattr(event, "_sibling_pending", None)
        key = getattr(event, "_sibling_pending_key", None)
        if snapshot is None or not key:
            return False
        pending = self._pending_text_batches.get(key)
        if pending is not snapshot:
            return False
        # Re-check the pending object (it can grow / mix senders while media
        # downloads). Do not re-check sender vs ``event``: group-observe
        # attribution may have rewritten event.source after we snapshotted.
        if not self._pending_text_is_sibling_comment(pending):
            return False
        pending = self._pending_text_batches.pop(key, None)
        task = self._pending_text_batch_tasks.pop(key, None)
        if task is not None and not task.done():
            task.cancel()
        comment = (pending.text or "").strip()
        caption = (event.text or "").strip()
        if comment and not caption:
            event.text = comment
        elif comment and caption:
            # Comment is the user's instruction; caption is the forwarded post.
            event.text = self._merge_caption(comment, caption)
        event.absorb_reply_expected(pending)
        if pending.reply_to_message_id and not event.reply_to_message_id:
            event.reply_to_message_id = pending.reply_to_message_id
            event.reply_to_text = pending.reply_to_text
            event.reply_to_author_id = pending.reply_to_author_id
            event.reply_to_author_name = pending.reply_to_author_name
            event.reply_to_is_own_message = pending.reply_to_is_own_message
        logger.info(
            "[Telegram] Merged pending text (%d chars) into media event %s",
            len(comment),
            key,
        )
        return True

    def _mark_sibling_absorb_eligibility(self, event: MessageEvent) -> None:
        """Snapshot the pending text batch this media event may consume.

        Only the comment already buffered at media admission is eligible
        (text-first, media-second). A later text is a different object and
        starts its own turn.
        """
        pending_key = self._text_batch_key(event)
        pending = self._pending_text_batches.get(pending_key)
        sender = self._inbound_sender_id(event)
        if (
            self._pending_text_is_sibling_comment(pending)
            and sender
            and sender == self._inbound_sender_id(pending)
        ):
            event._sibling_pending = pending  # type: ignore[attr-defined]
            event._sibling_pending_key = pending_key  # type: ignore[attr-defined]
            pending._sibling_media_claimed = True  # type: ignore[attr-defined]

    @staticmethod
    def _copy_sibling_pending(src: MessageEvent, dest: MessageEvent) -> MessageEvent:
        """Keep the admission snapshot if attribution replaced the event object."""
        dest._sibling_pending = getattr(src, "_sibling_pending", None)  # type: ignore[attr-defined]
        dest._sibling_pending_key = getattr(src, "_sibling_pending_key", None)  # type: ignore[attr-defined]
        return dest

    async def _dispatch_media_event(self, event: MessageEvent) -> None:
        """Absorb a sibling comment (if any) then dispatch. Media-path only."""
        self._absorb_pending_text_into_media_event(event)
        if self._should_drop_delayed_delivery():
            self._hold_inbound_event(event, where="media-dispatch")
            return
        try:
            await self.handle_message(event)
        except asyncio.CancelledError:
            self._hold_inbound_event(event, where="media-dispatch-cancelled")
            raise

    async def _wait_for_sibling_media(self, key: str) -> None:
        """Wait for the download that claimed this text batch, up to the cap.

        If this batch was not claimed, return and let the normal text delay flush it.
        The download signals when it finishes. The cap is the only timer.
        """
        inflight_cap = float(getattr(self, "_TEXT_MEDIA_INFLIGHT_CAP_S", 0) or 0)
        pending = self._pending_text_batches.get(key)
        if (
            inflight_cap <= 0
            or not self._pending_text_is_sibling_comment(pending)
            or not getattr(pending, "_sibling_media_claimed", False)
        ):
            return
        sibling_key = self._sibling_media_key(pending)
        deadline = time.monotonic() + inflight_cap
        while True:
            if self._sibling_wait_done(key, sibling_key):
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            event = self._sibling_wait_event(sibling_key)
            try:
                await asyncio.wait_for(event.wait(), timeout=remaining)
            except TimeoutError:
                return
            finally:
                self._drop_sibling_wait(sibling_key, event)

    def _sibling_wait_done(self, key: str, sibling_key: str) -> bool:
        if self._pending_text_batches.get(key) is None:
            return True
        inflight = getattr(self, "_media_inflight", None) or {}
        return inflight.get(sibling_key, 0) <= 0

    def _sibling_wait_event(self, sibling_key: str) -> asyncio.Event:
        events = self.__dict__.setdefault("_sibling_media_done", {})
        event = events.get(sibling_key)
        if event is None:
            event = asyncio.Event()
            events[sibling_key] = event
        return event

    def _drop_sibling_wait(self, sibling_key: str, event: asyncio.Event) -> None:
        """Forget this wait's event. A newer wait for the same key keeps its own."""
        events = getattr(self, "_sibling_media_done", None)
        if events and events.get(sibling_key) is event:
            events.pop(sibling_key, None)

    def _signal_sibling_wait(self, sibling_key: str) -> None:
        events = getattr(self, "_sibling_media_done", None)
        event = (events or {}).pop(sibling_key, None)
        if event is not None:
            event.set()

    def _release_sibling_media_waits(self) -> None:
        events = getattr(self, "_sibling_media_done", None)
        if not events:
            return
        for event in list(events.values()):
            event.set()
        events.clear()

    def _pending_text_is_sibling_comment(self, pending: Optional[MessageEvent]) -> bool:
        """True for a short text-only comment that sibling media may absorb."""
        if pending is None:
            return False
        if pending.is_command() or (pending.media_urls or []):
            return False
        if len(pending.text or "") > self._TEXT_BATCH_SHORT_LEN:
            return False
        if getattr(pending, "_sibling_media_mixed", False):
            return False
        return True

    def _enqueue_text_event(self, event: MessageEvent) -> None:
        """Buffer a text chunk, or hold it while delayed delivery must be dropped."""
        if self._should_drop_delayed_delivery():
            self._hold_inbound_event(event, where="text-enqueue")
            return
        key = self._text_batch_key(event)
        existing = self._pending_text_batches.get(key)
        if existing is not None and self._inbound_sender_id(existing) != self._inbound_sender_id(event):
            # Base still concatenates same-session text; do not let sibling media
            # steal another sender's words from that mixed buffer.
            existing._sibling_media_mixed = True  # type: ignore[attr-defined]
        super()._enqueue_text_event(event)
        self._accept_update()

    async def _flush_buffered(
        self, pending: dict, tasks: dict, key: str, delay: float, where: str, log_fn=None,
    ) -> None:
        """Shared delayed-flush body: sleep, optional text hold, pop, re-hold if teardown started, else dispatch.
        A cancel after the pop but before durable dispatch re-holds the event (never lose it).
        """
        current_task = asyncio.current_task()
        committed = self.__dict__.setdefault("_committed_flush_tasks", {})
        event = None
        try:
            await asyncio.sleep(delay)
            # Superseded flush (a newer chunk re-armed the timer while our sleep was already done):
            # CancelledError only lands at the next await, so check synchronously before the pop.
            owner = tasks.get(key)
            if owner is not None and owner is not current_task:
                return
            if where == "text":
                await self._wait_for_sibling_media(key)
                owner = tasks.get(key)
                if owner is not None and owner is not current_task:
                    return
            event = pending.pop(key, None)
            if not event:
                return
            # Committed: leave the per-key slot so a newer chunk re-arms without cancelling this
            # dispatch mid-flight (e.g. during a root-DM createForumTopic, whose ID would be lost).
            tasks.pop(key, None)
            prior = committed.get((where, key))
            committed[(where, key)] = current_task
            if prior is not None and not prior.done():
                await asyncio.wait({prior})  # dispatch in order, after the earlier batch settles
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
            if committed.get((where, key)) is current_task:
                committed.pop((where, key), None)
            if tasks.get(key) is current_task:
                tasks.pop(key, None)

    def _text_batch_delay_for(self, pending: Optional[MessageEvent]) -> float:
        """Adaptive delay: near-split-point last chunk → long delay (continuation almost certain);
        short/medium totals → capped fast delays; else configured cap (all min()'d with the operator cap)."""
        last_len = getattr(pending, "_last_chunk_len", 0) if pending else 0
        total_len = len(getattr(pending, "text", "") or "") if pending else 0
        if last_len >= self._SPLIT_THRESHOLD:
            return self._text_batch_split_delay_seconds
        if total_len <= self._TEXT_BATCH_FAST_LEN:
            return min(self._text_batch_delay_seconds, self._TEXT_BATCH_FAST_DELAY_S)
        if total_len <= self._TEXT_BATCH_SHORT_LEN:
            return min(self._text_batch_delay_seconds, self._TEXT_BATCH_SHORT_DELAY_S)
        return self._text_batch_delay_seconds

    async def _flush_text_batch(self, key: str) -> None:
        """Telegram keeps its own flush body: a cancel after the pop must HOLD the event and re-raise
        (PTB already acked the update; the hold queue redispatches after reconnect) rather than shield
        the dispatch — teardown must be able to stop a flush from reaching a torn-down session."""
        await self._flush_buffered(
            self._pending_text_batches, self._pending_text_batch_tasks, key,
            self._text_batch_delay_for(self._pending_text_batches.get(key)), "text",
            lambda ev: logger.info("[Telegram] Flushing text batch %s (%d chars)", key, len(ev.text or "")),
        )
