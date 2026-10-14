"""Stream creation and final payload selection for a gateway turn."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

from gateway.media_repair import repair_explicit_computer_use_media_paths
from gateway.turn_context import TurnContext

if TYPE_CHECKING:
    from gateway.run import GatewayRunner

logger = logging.getLogger("gateway.run")


class TurnStreamMixin:
    _ctx: TurnContext
    _runner: GatewayRunner
    _send_status_text: Callable[[str, Any, str], None]

    def _setup_stream_consumer(self, platform_key):
        ctx = self._ctx
        if ctx.mute_notification_reply:
            return None, None, None, False
        stream_consumer = None
        # The streaming-TTS consumer is created on the outer loop thread before run_sync launches;
        # run_sync only reads it via the holder for delta-callback wiring.
        stts = ctx.streaming_tts_consumer_holder[0]
        scfg = getattr(getattr(self._runner, 'config', None), 'streaming', None)
        if scfg is None:
            from gateway.config import StreamingConfig
            scfg = StreamingConfig()
        # display.platforms.<plat>.streaming may disable streaming per platform; None = follow global.
        plat_streaming = ctx.resolve_display_setting(ctx.user_config, platform_key, "streaming")
        want_stream_deltas = not ctx.scheduled_heartbeat and scfg.enabled_for(plat_streaming)
        want_interim_messages = bool(ctx.interim_assistant_messages_enabled) and not ctx.scheduled_heartbeat
        if want_stream_deltas or want_interim_messages:
            try:
                from gateway.stream_consumer import GatewayStreamConsumer
                adapter = self._runner._delivery_adapter_for(ctx.source)
                if adapter:
                    supports_incremental_stream = (
                        getattr(adapter, "SUPPORTS_MESSAGE_EDITING", True)
                        or bool(getattr(adapter, "SUPPORTS_NATIVE_STREAMING", False))
                    )
                    consumer_stream_deltas = want_stream_deltas and supports_incremental_stream
                    consumer_cfg, pause_typing_before_finalize = self._runner._build_stream_consumer_config(
                        ctx.source, scfg, adapter,
                        # A complete commentary message needs no edit cursor.  Keeping it on the
                        # consumer records what reached non-editable platforms, so an interim
                        # callback carrying the final answer participates in final-send dedup.
                        on_missing_cursor="fallback" if want_interim_messages else "raise",
                    )
                    stream_consumer = GatewayStreamConsumer(
                        adapter=adapter, chat_id=ctx.source.chat_id, config=consumer_cfg,
                        metadata=ctx._status_thread_metadata,
                        on_new_message=(
                            (lambda: ctx.progress_queue.put(("__reset__",))) if ctx.progress_queue is not None else None
                        ),
                        on_before_finalize=pause_typing_before_finalize,
                        initial_reply_to_id=ctx.event_message_id, run_still_current=ctx._run_still_current,
                    )
                    ctx.stream_consumer_holder[0] = stream_consumer
                    # #105341: a consumer created only for interim commentary (text streaming off)
                    # is never fed the final reply's deltas — mark it so the duplicate-risk
                    # diagnostic in ``_run_agent_mark_streamed_delivery`` stays silent.
                    stream_consumer.stream_deltas_enabled = consumer_stream_deltas
            except Exception as err:
                logger.debug("Could not set up stream consumer: %s", err)
        # Deltas tee to the stream consumer (when text streaming is on) and to streaming TTS.
        delta_sinks = [
            sc for sc in (
                stream_consumer if stream_consumer and stream_consumer.stream_deltas_enabled else None,
                stts,
            ) if sc is not None
        ]
        stream_delta_cb = None
        if delta_sinks:
            def stream_delta_cb(text: Optional[str]) -> None:
                if ctx._run_still_current():
                    for sink in delta_sinks:
                        sink.on_delta(text)

        def interim_assistant_cb(text: str, *, already_streamed: bool = False) -> None:
            if not ctx._run_still_current():
                return
            if stts is not None:
                # Flush accepted deltas; completed commentary is a separate speech segment.
                stts.on_delta(None)
                if not already_streamed:
                    stts.on_delta(text)
                    stts.on_delta(None)
            if stream_consumer is not None:
                stream_consumer.on_segment_break() if already_streamed else stream_consumer.on_commentary(text)
            elif not already_streamed and ctx._status_adapter and str(text or "").strip():
                self._send_status_text(text, ctx._status_thread_metadata, "interim_assistant_callback scheduling error")

        return stream_consumer, stream_delta_cb, interim_assistant_cb, want_interim_messages

    def _finish_stream_consumer(self, result, agent_history, stream_consumer):
        ctx = self._ctx
        # Canonicalize a model-emitted computer-use screenshot path at the common result boundary so
        # the streaming finalizer and the non-streaming delivery path see the same response.
        if isinstance(result, dict) and isinstance(result.get("final_response"), str):
            result["final_response"] = repair_explicit_computer_use_media_paths(
                result["final_response"], result.get("messages", []), history_offset=len(agent_history),
            )
        ctx.result_holder[0] = result
        if stream_consumer is None:
            return
        # Pass final_response as the authoritative finalize payload: it includes post-stream
        # augmentation (verifier footer, explainer) the accumulator never saw. Adopt ONLY a genuinely
        # completed final: interrupt paths return {interrupted: True, completed: False} with a
        # DIAGNOSTIC final_response — adopting it would seal the partial answer over with the
        # diagnostic AND suppress the gateway's own error delivery.
        _final_for_stream = None
        if (
            isinstance(result, dict) and not result.get("failed") and not result.get("interrupted")
            and result.get("completed") is not False
        ):
            fr = result.get("final_response")
            if isinstance(fr, str) and fr.strip() and fr != "(empty)":
                _final_for_stream = fr
        if _final_for_stream is None:
            stream_consumer.finish()
            return
        # Duck-type safe: test doubles / older consumers may expose a zero-arg finish().
        try:
            stream_consumer.finish(_final_for_stream)
        except TypeError:
            stream_consumer.finish()
