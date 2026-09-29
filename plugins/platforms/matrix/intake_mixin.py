"""Matrix gateway admission and deferred text-batch receipts."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent, MessageType
from plugins.platforms.matrix.location import format_location_content
from plugins.platforms.matrix.relations import MatrixRelation
from plugins.platforms.matrix.reply_context import MatrixEventContext, MatrixEventContextCache
from plugins.platforms.matrix.sync_transport import DurableSyncStore, SyncDispatch
from plugins.platforms.matrix.voice_mention import ParkedVoices


class MatrixIntakeMixin(BasePlatformAdapter):
    _client: Any
    _require_mention: bool
    _event_context_cache: MatrixEventContextCache
    _parked_voices: ParkedVoices
    _text_batch_intakes: dict[int, list[tuple[str, asyncio.Future[bool]]]]
    _buffered_intakes: dict[str, asyncio.Future[bool]]
    _strip_mention: Callable[[str], str]
    _content_mentions_bot: Callable[[str, dict], bool]
    _handle_media_message: Callable[..., Awaitable[bool | None]]
    _background_read_receipt: Callable[[str, str], None]
    _build_inbound_event: Callable[..., Awaitable[MessageEvent | None]]

    async def handle_message(self, event: MessageEvent) -> None:
        if getattr(event, "_hermes_startup_restore_replay", False) is True:
            await super().handle_message(event)
            return
        store = getattr(self._client, "sync_store", None)
        event_id = str(event.message_id or "")
        receipts = self._text_batch_intakes.get(id(event), [])
        event_ids = tuple(receipt_id for receipt_id, _receipt in receipts) or (event_id,)
        if not isinstance(store, DurableSyncStore) or not event_id.startswith("$"):
            await super().handle_message(event)
            return
        if not store.reserve_intake(event_id):
            event._gateway_accepted = True
            return
        consumed = self._message_handler is not None
        try:
            await super().handle_message(event)
            if consumed:
                if receipts:
                    await store.accept_intakes(event_ids)
                else:
                    await store.accept_intake(event_id)
        finally:
            store.release_intake(event_id)


    async def _admit(self, event: MessageEvent) -> bool:
        """Pass *event* to the gateway and return whether the gateway consumed it.

        The gateway consumes an event when it starts or queues a turn, and also when it handles
        the event inline or drops it on purpose: a command or redirect while the session is
        busy, a clarify answer, an unauthorised sender or an unresolved profile route. A sync
        retry would run any of those again, so only an event that found no message handler is
        left unacknowledged.
        """
        consumed = getattr(self, "_message_handler", None) is not None
        await self.handle_message(event)
        return consumed


    async def _handle_text_message(
        self, room_id: str, sender: str, event_id: str, event_ts: float, source_content: dict,
        relates_to: dict, *, reply_parent: MatrixEventContext | None = None) -> asyncio.Future[bool] | bool | None:
        from plugins.platforms.matrix.adapter import _normalize_matrix_bang_command, _split_reply_fallback, logger

        body = source_content.get("body", "") or ""
        location_text = None
        if source_content.get("msgtype") == "m.location":
            if not isinstance(body, str):
                body = ""
            location_body = body
            if relates_to.get("m.in_reply_to"):
                _, location_body = _split_reply_fallback(body)
            if self._require_mention and self._content_mentions_bot(body, source_content):
                location_body = self._strip_mention(location_body)
            location_text = format_location_content({**source_content, "body": location_body})
            if location_text is None:
                logger.debug("Matrix: ignoring invalid location %s in %s", event_id, room_id)
                return
        if not body and location_text is None:
            return
        reply_target = MatrixRelation.from_content(relates_to).reply_target
        reply_parent = reply_parent or (self._event_context_cache.retain(room_id, reply_target) if reply_target else None)
        # Dict lookup first: the mention regexes only run when a voice is parked or being gated
        # (both only happen under require_mention).
        if (location_text is None and self._parked_voices.pending(room_id, sender)
                and not self._strip_mention(body).strip() and self._content_mentions_bot(body, source_content)):
            limit = self._parked_voices.mark()  # never claim a voice sent after this mention
            await self._parked_voices.settle(room_id, sender)  # same-/sync-batch voice still gating
            parked = self._parked_voices.claim(room_id, sender, before=limit)
            if parked:  # answer the voice this bare mention was typed for, not an empty text
                voice_id, voice_ts, voice_content, voice_relates = parked
                await self._handle_media_message(
                    room_id, sender, voice_id, voice_ts, voice_content, voice_relates, "m.audio",
                    mention_claimed=True)
                self._background_read_receipt(room_id, event_id)  # the claim receipted the voice
                return
        msg_event = await self._build_inbound_event(
            room_id, sender, event_id, _normalize_matrix_bang_command(body), source_content, relates_to,
            reply_parent=reply_parent, event_ts=event_ts)
        if msg_event is None:
            return
        if location_text is not None:
            msg_event.text = location_text
            msg_event.message_type = MessageType.TEXT
        self._event_context_cache.store(room_id, event_id, MatrixEventContext(sender, msg_event.text))
        return await self._admit_text_event(msg_event)

    async def _admit_text_event(self, event: MessageEvent) -> asyncio.Future[bool] | bool:
        if event.message_type == MessageType.TEXT and self._text_batch_delay_seconds > 0:
            self._enqueue_text_event(event)
            pending = self._pending_text_batches.get(self._text_batch_key(event))
            if pending is None:
                return True
            event_id = str(event.message_id or "")
            receipt: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
            self._text_batch_intakes.setdefault(id(pending), []).append((event_id, receipt))
            self._buffered_intakes[event_id] = receipt
            receipt.add_done_callback(lambda _: self._buffered_intakes.pop(event_id, None))
            # The flush task reports a dispatch error; the receipt only marks the batch as failed.
            receipt.add_done_callback(lambda done: done.cancelled() or done.exception())
            return receipt
        return await self._admit(event)


    async def _dispatch_text_batch(self, event: MessageEvent) -> None:
        dispatch = getattr(self._client, "hermes_sync", None)
        task = asyncio.current_task()
        if isinstance(dispatch, SyncDispatch) and task is not None:
            dispatch.own_background_task(task)
        receipts = self._text_batch_intakes.get(id(event), [])
        try:
            consumed = await self._admit(event)
        except BaseException as exc:
            for _event_id, receipt in receipts:
                if not receipt.done():
                    if isinstance(exc, asyncio.CancelledError):
                        receipt.cancel()
                    else:
                        receipt.set_exception(exc)
            raise
        else:
            for _event_id, receipt in receipts:
                if not receipt.done():
                    receipt.set_result(consumed)
        finally:
            self._text_batch_intakes.pop(id(event), None)
