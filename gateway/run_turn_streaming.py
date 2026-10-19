"""Stream-consumer construction for agent turns, shared by the in-process path (via TurnRunner) and
the proxy path, including the adapter streaming-capability gate; inherited by GatewayTurnMixin.
``gateway.run`` internals are imported lazily inside method bodies (import cycle), so
``patch("gateway.run.X")`` keeps intercepting them at call time.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

from gateway.config import Platform
from gateway.platforms.base_edit_targets import adapter_refuses_token_streaming, adapter_supports_streaming_edits

if TYPE_CHECKING:  # string annotations only; never imported at runtime (cycle)
    from gateway.session import SessionSource

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")


class GatewayTurnStreamingMixin:
    """Builds the per-turn ``GatewayStreamConsumer`` configuration (see module docstring)."""

    def _build_stream_consumer_config(
        self, source: SessionSource, scfg: Any, adapter: Any, *, on_missing_cursor: str,
    ) -> tuple[Any, Optional[Callable[[], None]]]:
        """Build the shared ``StreamConsumerConfig`` and optional Telegram pause-typing closure.
        For adapters without streaming-safe edits ``on_missing_cursor="fallback"`` streams with an empty
        cursor; ``"raise"`` raises ``RuntimeError`` so the caller skips streaming entirely."""
        from gateway.stream_consumer import StreamConsumerConfig
        _pause_typing_before_finalize = None
        if source.platform == Platform.TELEGRAM and hasattr(adapter, "pause_typing_for_chat"):
            def _pause_typing_before_finalize(_adapter=adapter, _chat_id=source.chat_id) -> None:
                _adapter.pause_typing_for_chat(_chat_id)
        # Non-editing platforms (QQ, WeChat) skip streaming — the partial first message could never
        # be updated — unless they have a native-streaming transport (WeCom msgtype "stream"). Signal
        # can edit but opts out of streaming edits (every edit is visible), so it counts as non-editing.
        _adapter_supports_edit = adapter_supports_streaming_edits(adapter)
        _adapter_supports_native_stream = bool(getattr(adapter, "SUPPORTS_NATIVE_STREAMING", False))
        if not _adapter_supports_edit and not _adapter_supports_native_stream and on_missing_cursor == "raise":
            raise RuntimeError("skip streaming for non-editable platform")
        _effective_cursor = scfg.cursor if _adapter_supports_edit else ""
        # Some Matrix clients render the cursor as tofu: stream text, no cursor.
        if source.platform == Platform.MATRIX:
            _effective_cursor = ""
        # Fresh-final applies to Telegram only (others edit in place cheaply).
        # Fresh-final applies to Telegram only — other platforms either edit in place cheaply (Discord,
        # Slack) or don't have the timestamp-on-edit / edit-timestamp-stays-stale problem. (Ported from
        # openclaw/openclaw#72038.)
        _fresh_final_secs = (
            float(getattr(scfg, "fresh_final_after_seconds", 0.0) or 0.0)
            if source.platform == Platform.TELEGRAM else 0.0
        )
        _consumer_cfg = StreamConsumerConfig(
            edit_interval=scfg.edit_interval, buffer_threshold=scfg.buffer_threshold,
            cursor=_effective_cursor,
            fresh_final_after_seconds=_fresh_final_secs, transport=scfg.transport or "edit",
            chat_type=getattr(source, "chat_type", "") or "",
        )
        return _consumer_cfg, _pause_typing_before_finalize

    def _proxy_stream_consumer(self, source: SessionSource, event_message_id, _thread_metadata, _run_still_current):
        """Platform stream consumer for the proxy path when streaming is enabled, else ``None``."""
        from gateway.run import _load_gateway_config, _platform_config_key
        _scfg = getattr(getattr(self, "config", None), "streaming", None)
        # #60671 — streaming TTS consumer is created on the outer event-loop thread before run_sync
        # launches.  run_sync only reads it via ``streaming_tts_consumer_holder[0]`` for delta callback
        # wiring.
        if _scfg is None:
            from gateway.config import StreamingConfig
            _scfg = StreamingConfig()
        # Global master switch first: skips the config.yaml re-read on the default (off) path.
        if not _scfg.globally_enabled:
            return None
        from gateway.display_config import resolve_display_setting
        _plat_streaming = resolve_display_setting(_load_gateway_config(), _platform_config_key(source.platform), "streaming")
        if not _scfg.enabled_for(_plat_streaming):
            return None
        try:
            from gateway.stream_consumer import GatewayStreamConsumer
            _adapter = self._delivery_adapter_for(source)
            # Proxy SSE always streams tokens: an adapter that opted out of streaming edits (Signal)
            # gets only the final reply, as on the in-process path.
            if not _adapter or adapter_refuses_token_streaming(_adapter):
                return None
            _consumer_cfg, _pause_typing_before_finalize = self._build_stream_consumer_config(
                source, _scfg, _adapter, on_missing_cursor="fallback",
            )
            return GatewayStreamConsumer(
                adapter=_adapter, chat_id=source.chat_id, config=_consumer_cfg,
                metadata=_thread_metadata, on_before_finalize=_pause_typing_before_finalize,
                initial_reply_to_id=event_message_id, run_still_current=_run_still_current,
            )
        except Exception as _sc_err:  # health: allow BLE001 -- moved unchanged from run_turn.py (main has the same handler): a consumer that cannot be built only loses the preview; the final reply still sends
            logger.debug("Proxy: could not set up stream consumer: %s", _sc_err)
            return None
