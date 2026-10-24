"""Deliver each completed reply before running its queued follow-up."""

from __future__ import annotations

from contextlib import suppress
import inspect
import logging
from typing import Any

from gateway.response_filters import silence_allowed
from gateway.turn_context import TurnContext

logger = logging.getLogger("gateway.run")


class GatewayQueuedDeliveryMixin:
    async def _run_agent_deliver_first_response(
        self,
        turn_ctx: TurnContext,
        adapter: Any,
        response: Any,
        result: Any,
        stream_task: Any,
    ) -> None:
        """Deliver the first response before a queued follow-up runs, unless streaming already did."""
        from gateway.run_turn import _unexpected_silence_reply

        if turn_ctx.mute_notification_reply:
            return
        session_key = turn_ctx.session_key
        _sc = turn_ctx.stream_consumer_holder[0]
        if _sc and stream_task:
            try:
                await self._await_stream_task(stream_task)
            except Exception as e:
                logger.debug("Stream consumer wait before queued message failed: %s", e, exc_info=True)
        # Delivery uses the finalized task result (empty/failure normalization), not raw ``result``.
        _delivery_result = response if isinstance(response, dict) else (result or {})
        first_response = _delivery_result.get("final_response", "")
        footer = self._hmwa_runtime_footer_line(
            _delivery_result,
            turn_ctx.source,
            getattr(turn_ctx, "turn_seconds", None),
        )
        _already_streamed = self._run_agent_stream_confirmed_final_delivery(
            _sc,
            first_response,
            previewed=bool(_delivery_result.get("response_previewed")),
        )
        # Same silence predicate as the normal path, else this branch leaks the literal marker.
        if self._is_intentional_silence(_delivery_result, first_response):
            if silence_allowed(
                turn_ctx.persist_user_display_kind, turn_ctx.reply_expected
            ):
                logger.info(
                    "Queued follow-up for session %s: suppressing intentional silence marker before continuing.",
                    session_key or "?",
                )
                first_response = ""
            else:
                logger.warning(
                    "Queued follow-up for session %s: replacing a human-turn silence marker.",
                    session_key or "?",
                )
                first_response = _unexpected_silence_reply()
                _already_streamed = False
        # Failed turns deliver their text but never their attachments (completed-turn parity).
        _deliver_media = not _delivery_result.get("failed")
        if first_response:
            logger.info(
                "Queued follow-up for session %s: final text delivery confirmed; delivering explicit media before continuing."
                if _already_streamed
                else "Queued follow-up for session %s: final stream delivery not confirmed; sending first response before continuing.",
                session_key or "?",
            )
            # A streamed body cannot be edited to add the footer (prefix stability), so it rides
            # this delivery call as a trailing send; otherwise include it in the one text message.
            delivery_text = (
                first_response
                if not footer or _already_streamed
                else f"{first_response}\n\n{footer}"
            )
            try:
                _text_delivered = await self._deliver_queued_first_response(
                    delivery_text,
                    source=turn_ctx.source,
                    adapter=adapter,
                    metadata=turn_ctx._status_thread_metadata,
                    event_message_id=turn_ctx.event_message_id,
                    text_already_delivered=_already_streamed,
                    deliver_media=_deliver_media,
                    stream_consumer=_sc,
                    # The text send records a delivery-ledger obligation under this key, keyed on
                    # the raw inbound id (the anchor above is only the reply target).
                    session_key=session_key,
                    inbound_message_id=turn_ctx.inbound_message_id,
                )
                # Both worker results describe this delivery. Mark them before any merge copies
                # either one, and before a trailing footer send can fail independently of the body.
                for delivered_result in (_delivery_result, result):
                    if _text_delivered and isinstance(delivered_result, dict):
                        delivered_result["already_sent"] = True
                        delivered_result["media_already_delivered"] = _deliver_media
                _footer_delivered = False
                if footer and not _already_streamed:
                    _footer_delivered = _text_delivered
                elif footer and _already_streamed and _text_delivered:
                    _footer_sent = await adapter.send(
                        turn_ctx.source.chat_id,
                        footer,
                        metadata=turn_ctx._status_thread_metadata,
                    )
                    _footer_delivered = bool(getattr(_footer_sent, "success", False))
                if _footer_delivered:
                    for delivered_result in (_delivery_result, result):
                        if isinstance(delivered_result, dict):
                            delivered_result["runtime_footer_already_delivered"] = True
            except Exception as e:
                logger.warning(
                    "Failed to send first response before queued message: %s",
                    e,
                    exc_info=True,
                )
        # Release deferred bg-review notifications: pop (no double-fire in base.py's finally) and call.
        _bg_cb = self._pop_post_delivery_callback(
            adapter, session_key, turn_ctx.run_generation
        )
        if callable(_bg_cb):
            with suppress(Exception):
                _bg_result = _bg_cb()
                if inspect.isawaitable(_bg_result):
                    await _bg_result
