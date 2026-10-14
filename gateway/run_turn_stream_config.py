"""Per-platform streaming configuration shared by normal and proxy turns."""
from __future__ import annotations

import os
from typing import Any, Callable, Optional
from gateway.config import Platform
from gateway.session import SessionSource


def build_stream_consumer_config(
    source: SessionSource, scfg: Any, adapter: Any, *, on_missing_cursor: str,
) -> tuple[Any, Optional[Callable[[], None]]]:
    """Build the shared ``StreamConsumerConfig`` and optional Telegram pause-typing closure.
    For non-editing adapters ``on_missing_cursor="fallback"`` streams with an empty cursor;
    ``"raise"`` raises ``RuntimeError`` so the caller skips streaming entirely."""
    from gateway.stream_consumer import StreamConsumerConfig
    _pause_typing_before_finalize = None
    if source.platform == Platform.TELEGRAM and hasattr(adapter, "pause_typing_for_chat"):
        def _pause_typing_before_finalize(_adapter=adapter, _chat_id=source.chat_id) -> None:
            _adapter.pause_typing_for_chat(_chat_id)
    # Non-editing platforms (QQ, WeChat) skip streaming — the partial first message could never
    # be updated — unless they have a native-streaming transport (WeCom msgtype "stream").
    _adapter_supports_edit = getattr(adapter, "SUPPORTS_MESSAGE_EDITING", True)
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
    # Preserve the established text↔tool-progress chronology unless progress is quiet.
    # The gate asks for the mode the display path actually runs with (off|log is the
    # quiet pair — the same one tool_progress_enabled uses), so the
    # HERMES_TOOL_PROGRESS_MODE bridge can't leave bubbles on while this mode is set.
    # Telegram-only: other adapters keep their current segment-boundary semantics.
    from gateway.run import _load_gateway_config, _platform_config_key
    from gateway.display_config import resolve_display_setting, resolve_tool_progress
    _user_config = _load_gateway_config()
    _platform_key = _platform_config_key(source.platform)
    _single_message_per_turn = (
        source.platform == Platform.TELEGRAM
        and bool(resolve_display_setting(
            _user_config, _platform_key, "streaming_single_message", False,
        ))
        and resolve_tool_progress(
            _user_config, _platform_key, os.getenv("HERMES_TOOL_PROGRESS_MODE"),
        )[0] in ("off", "log")
    )
    # Transient-overlay sub-switches: tool-start lines (default on — quiet progress
    # would otherwise show nothing during tool runs) and thinking snippets (opt-in).
    _single_message_activity = _single_message_per_turn and bool(resolve_display_setting(
        _user_config, _platform_key, "streaming_single_message_activity", True,
    ))
    _single_message_thinking = _single_message_per_turn and bool(resolve_display_setting(
        _user_config, _platform_key, "streaming_single_message_thinking", False,
    ))
    _single_message_4096_split = _single_message_per_turn and bool(resolve_display_setting(
        _user_config, _platform_key, "streaming_single_message_4096_split", False,
    ))
    # Completion message effect (Telegram DM): emoji + minimum turn age (0 = every turn).
    _message_effects_on = bool(resolve_display_setting(
        _user_config, _platform_key, "message_effects", False,
    ))
    _message_effect = str(resolve_display_setting(
        _user_config, _platform_key, "message_effect", "🎉") or "") if _message_effects_on else ""
    try:
        _message_effect_min = float(resolve_display_setting(
            _user_config, _platform_key, "message_effect_min_seconds", 60) or 0)
    except (TypeError, ValueError):
        _message_effect_min = 60.0
    _consumer_cfg = StreamConsumerConfig(
        edit_interval=scfg.edit_interval, buffer_threshold=scfg.buffer_threshold,
        cursor=_effective_cursor,
        fresh_final_after_seconds=_fresh_final_secs, transport=scfg.transport or "edit",
        chat_type=getattr(source, "chat_type", "") or "",
        single_message_per_turn=_single_message_per_turn,
        single_message_activity=_single_message_activity,
        single_message_thinking=_single_message_thinking,
        single_message_4096_split=_single_message_4096_split,
        message_effect=_message_effect,
        message_effect_min_seconds=_message_effect_min,
    )
    return _consumer_cfg, _pause_typing_before_finalize
