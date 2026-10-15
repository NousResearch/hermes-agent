"""Queued first-response delivery extracted from the run_turn facade.

The queued (/queue) lane sends the completed first turn's reply before recursion into the
follow-up. Keeping it beside the queue drain in ``run_turn`` pushed that facade past its file
ratchet; the topical sibling keeps production behavior and method resolution unchanged.
"""

from __future__ import annotations

from typing import Any

from contextlib import suppress
import inspect

from gateway.response_filters import silence_allowed
from gateway.turn_context import TurnContext

# Log-record parity with the origin module.
import logging

logger = logging.getLogger("gateway.run")


def _unexpected_silence_reply():
    from gateway.run_turn import _unexpected_silence_reply as _impl

    return _impl()


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
            except Exception as e:
                logger.warning(
                    "Failed to send first response before queued message: %s",
                    e,
                    exc_info=True,
                )
            else:
                # One source of truth for "this turn's final already reached the chat": the normal
                # completion path (`_hmwa_deliver_turn_response`) consults ``already_sent`` on the
                # result the queued lane hands back. Every early `return result` after this point
                # (follow-up text refused, stale goal continuation) otherwise re-sends the text the
                # fallback just delivered — the #81052 duplicate. A REFUSED send reports False, and
                # the completion send stays the fallback so the user is not left with nothing.
                if _text_delivered and isinstance(result, dict):
                    result["already_sent"] = True
                    # The queued lane already uploaded this response's MEDIA: attachments; without
                    # this the completion path's already_sent rescan uploads every file twice.
                    result["media_already_delivered"] = _deliver_media
                # Receipt is keyed to THIS response, not to the chain: a real follow-up result
                # owns its own footer. Refused/aborted continuations can return this same dict;
                # object identity (not merely equal text) is what proves it is the same delivery.
                if _footer_delivered:
                    turn_ctx.queued_footer_delivery = {"result": _delivery_result}
                    if isinstance(result, dict):
                        result["runtime_footer_already_delivered"] = True
        # Release deferred bg-review notifications: pop (no double-fire in base.py's finally) and call.
        _bg_cb = self._pop_post_delivery_callback(
            adapter, session_key, turn_ctx.run_generation
        )
        if callable(_bg_cb):
            with suppress(Exception):
                _bg_result = _bg_cb()
                if inspect.isawaitable(_bg_result):
                    await _bg_result
