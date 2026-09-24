"""Single-message Telegram delivery, deliberately without split/recreate/fallback.

This adapter-owned entry point uses the shared chat lock/rate budget, but not send(),
whose reconnect forwarding and continuation semantics cannot issue a single receipt.
"""
from __future__ import annotations

import asyncio
import logging
import math
from typing import Any
from uuid import uuid4

from gateway.live_todo import DeliveryOutcome, DeliveryStatus
from plugins.platforms.telegram.telegram_ids import normalize_telegram_chat_id
from .transport_admission import OperationAdmission, operation_admission


class LiveTodoTransportMixin:
    live_todo_transport = 1

    def _init_live_todo_transport(self):
        self._live_todo_epoch = uuid4().hex
        self._live_todo_client = None
        self._live_todo_sources = set()

    def _fence_live_todo_transport(self):
        sources = tuple(getattr(self, "_live_todo_sources", ()))
        for source in sources:
            source.close()
        self._live_todo_epoch = uuid4().hex
        return sources

    async def _drain_live_todo_transport(self, sources):
        if sources:
            results = await asyncio.gather(*(source.finish() for source in sources), return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logging.getLogger(__name__).error("Live todo drain failed: %s", type(result).__name__)

    async def deliver_live_todo(self, source, text):
        rejected = DeliveryStatus.REJECTED
        if source.adapter is not self or not source.admitted():
            return DeliveryOutcome(rejected, reason="stale binding")
        if not isinstance(text, str) or not text.strip() or len(text.encode("utf-16-le")) // 2 > 4096:
            return DeliveryOutcome(rejected, reason="invalid single-message payload")
        chat_id = source.binding.chat_id
        async with self._chat_send_lock(chat_id):
            cooldown = self._send_flood_cooldown_remaining(chat_id)
            if cooldown is not None:
                return DeliveryOutcome(DeliveryStatus.SKIPPED, reason="cooldown", retry_after=cooldown)
            delay = self._chat_outbound_slot_remaining(chat_id)
            if delay > 0:
                await asyncio.sleep(delay)
            # Early rejection saves SDK work. The exact operation also travels through
            # SDK, fallback, pool, connect and stream waits to the synchronous write fence.
            with source.registration.lock, source.lock:
                if not source.admitted() or getattr(self, "_send_path_degraded", False):
                    return DeliveryOutcome(rejected, reason="stale or disconnected binding")
                kwargs: dict[str, Any] = dict(chat_id=normalize_telegram_chat_id(chat_id), text=text,
                              parse_mode=None, disable_web_page_preview=True)
                markup = getattr(source, "reply_markup", None)
                if markup is not None:
                    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
                    kwargs["reply_markup"] = InlineKeyboardMarkup([
                        [InlineKeyboardButton(**button) for button in row] for row in markup]) if markup else {"inline_keyboard": []}
                if source.message_id is None:
                    if source.binding.thread_id is not None:
                        kwargs["message_thread_id"] = int(source.binding.thread_id)
                    kwargs["disable_notification"] = True
                    operation = self._bot.send_message
                else:
                    kwargs["message_id"] = int(source.message_id)
                    operation = self._bot.edit_message_text
                self._hold_chat_outbound_slot(chat_id)
                source.inflight = True
            admission = OperationAdmission(source)
            token = operation_admission.set(admission)
            try:
                # SDK timeouts bound the actual I/O. A deadline/cancel is ambiguous, not retryable.
                async with asyncio.timeout(20.0):
                    result = await operation(**kwargs)
                message_id = getattr(result, "message_id", None)
                if message_id is None or (source.message_id is not None and str(message_id) != source.message_id):
                    outcome = DeliveryOutcome(DeliveryStatus.UNKNOWN, reason="missing/mismatched message receipt")
                else:
                    outcome = DeliveryOutcome(DeliveryStatus.DELIVERED, str(message_id))
            except asyncio.CancelledError:
                uncertain = admission.dispatched or not admission.transport_entered
                source.unknown = uncertain
                source.last_outcome = DeliveryOutcome(
                    DeliveryStatus.UNKNOWN if uncertain else rejected, reason="cancelled operation")
                raise
            except Exception as exc:
                from telegram.error import BadRequest, Forbidden, RetryAfter
                if admission.transport_entered and not admission.dispatched:
                    source.unknown = False  # exact transport proves no request bytes were enqueued
                    outcome = DeliveryOutcome(rejected, reason="operation failed before request write")
                elif isinstance(exc, RetryAfter):
                    delay = exc.retry_after
                    delay = delay.total_seconds() if hasattr(delay, "total_seconds") else float(delay)
                    if math.isfinite(delay) and delay >= 0:
                        self._record_send_flood_cooldown(chat_id, delay)
                        delay = max(delay, self._send_flood_cooldown_remaining(chat_id) or 0)
                    outcome = DeliveryOutcome(rejected, reason="rate limited",
                                              retry_after=delay if math.isfinite(delay) else None)
                elif isinstance(exc, (BadRequest, Forbidden)):
                    # Even "not modified" isn't a verified new delivery. Never create on edit failure.
                    deleted = (getattr(source, "durable_card", False) and source.message_id is not None
                               and isinstance(exc, BadRequest)
                               and "message to edit not found" in str(exc).lower())
                    unchanged = (getattr(source, "durable_card", False) and source.message_id is not None
                                 and isinstance(exc, BadRequest)
                                 and "message is not modified" in str(exc).lower())
                    outcome = DeliveryOutcome(
                        rejected,
                        reason=("known message deleted" if deleted else
                                "known message unchanged" if unchanged else
                                "Telegram refused operation"),
                    )
                else:
                    outcome = DeliveryOutcome(DeliveryStatus.UNKNOWN, reason="unconfirmed dispatch")
            finally:
                operation_admission.reset(token)
                source.inflight = False
            # Host records exact settlement even after revocation; it never revives the writer.
            source.last_outcome = outcome
            if outcome.status == DeliveryStatus.UNKNOWN:
                source.unknown = True
            elif outcome.status == DeliveryStatus.DELIVERED:
                source.message_id = outcome.message_id
            return outcome
