"""Transport capabilities and per-turn streaming configuration."""

from __future__ import annotations

from typing import Any, Callable, Optional
from gateway.config import Platform
from gateway.session import SessionSource


class GatewayStreamConfigMixin:
    def _build_stream_consumer_config(
        self, source: SessionSource, scfg: Any, adapter: Any, *, on_missing_cursor: str,
    ) -> tuple[Any, Optional[Callable[[], None]]]:
        """Build the shared ``StreamConsumerConfig`` and optional Telegram pause-typing closure.
        For non-editing adapters ``on_missing_cursor="fallback"`` streams with an empty cursor;
        ``"raise"`` raises ``RuntimeError`` so the caller skips streaming entirely."""
        from gateway.stream_consumer import StreamConsumerConfig
        _pause_typing_before_finalize = None
        if source.platform == Platform.TELEGRAM and hasattr(adapter, "pause_typing_for_chat"):
            def _pause_typing_before_finalize(_adapter=adapter, _chat_id=source.chat_id) -> None:
                _adapter.pause_typing_for_chat(_chat_id)
        # Append-only blocks and native streams do not require message editing.
        _adapter_supports_edit = getattr(adapter, "SUPPORTS_MESSAGE_EDITING", True)
        _adapter_supports_native_stream = bool(getattr(adapter, "SUPPORTS_NATIVE_STREAMING", False))
        _block_streaming = getattr(adapter, "SUPPORTS_BLOCK_STREAMING", False) is True
        if not (_adapter_supports_edit or _adapter_supports_native_stream or _block_streaming) and on_missing_cursor == "raise":
            raise RuntimeError("skip streaming for non-editable platform")
        _effective_cursor = scfg.cursor if _adapter_supports_edit else ""
        # Some Matrix clients render the cursor as tofu: stream text, no cursor.
        if source.platform == Platform.MATRIX:
            _effective_cursor = ""
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

    def _run_still_current_fn(self, session_key: Optional[str], run_generation: Optional[int]) -> Callable[[], bool]:
        """Predicate: does this run's generation still own ``session_key``? (always True when untracked)."""
        def _run_still_current() -> bool:
            if run_generation is None or not session_key:
                return True
            return self._is_session_run_current(session_key, run_generation)
        return _run_still_current

    @staticmethod
    def _block_stream_delivery_state(response, consumer, *, seal=False):
        prefix_reader = getattr(type(consumer), "acknowledged_block_prefix", None)
        if not callable(prefix_reader) or not isinstance(response, dict):
            return
        response["_streamed_block_prefix"] = prefix_reader(consumer)
        if seal and consumer.final_content_delivered and consumer.delivered_final_matches(response.get("final_response", "")) is True:
            response["already_sent"] = True

    @staticmethod
    def _block_stream_delivery_tail(response, result):
        prefix = (result or {}).get("_streamed_block_prefix", "")
        if not prefix:
            return response
        from gateway.platforms.weixin_streaming import unsent_block_tail
        return unsent_block_tail(response, prefix)
