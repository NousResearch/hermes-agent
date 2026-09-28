"""Regression for #126581: bare silence markers must suppress delivery for failed/interrupted
**machinery** turns (path 2) and must NOT be published as partial preview at a segment break (path 1).
Human turns that fail keep the existing guard — see test_exact_silence_tokens_are_intentional_silence for the success-path baseline.
"""
from gateway.response_filters import (
    INTERNAL_NOTIFICATION_DISPLAY_KIND,
    is_intentional_silence_agent_result,
)


_USER_AGENT_RESULT = {"failed": True, "display_kind": "user"}
_MACHINERY_AGENT_RESULT = {"failed": True, "display_kind": INTERNAL_NOTIFICATION_DISPLAY_KIND}


def test_user_failed_turn_is_NOT_intentional_silence():
    """Human turn that fails/interrupts must NOT be silenced — preserve the prior guard
    so a failed reply still surfaces to the user (the #120051 case)."""
    assert not is_intentional_silence_agent_result(_USER_AGENT_RESULT, "NO_REPLY")
    assert not is_intentional_silence_agent_result(_USER_AGENT_RESULT, "[SILENT]")


def test_machinery_failed_turn_IS_intentional_silence():
    """Machinery turn (heartbeat/cron/auto) that fails/interrupts → marker is silenced,
    closing #126581 path 2 (was: 154 warnings, bare marker delivered to chat)."""
    assert is_intentional_silence_agent_result(_MACHINERY_AGENT_RESULT, "NO_REPLY")
    assert is_intentional_silence_agent_result(_MACHINERY_AGENT_RESULT, "[SILENT]")


def test_successful_turn_path_unchanged():
    """The success path keeps its existing semantics — both user and machinery turns."""
    user_ok = {"failed": False, "display_kind": "user"}
    mach_ok = {"failed": False, "display_kind": INTERNAL_NOTIFICATION_DISPLAY_KIND}
    assert is_intentional_silence_agent_result(user_ok, "NO_REPLY")
    assert is_intentional_silence_agent_result(mach_ok, "NO_REPLY")


def test_non_final_silence_marker_preview_is_suppressed():
    """Path 1 of #126581: a segment-break tick that already accumulated the bare
    silence marker must NOT push it as a visible preview before got_done can suppress."""
    import asyncio
    from unittest.mock import MagicMock, AsyncMock
    from gateway.stream_consumer import GatewayStreamConsumer
    from gateway.response_filters import LIVE_GATEWAY_SILENT_MARKERS

    cfg = MagicMock()
    cfg.cursor = ""
    cfg.buffer_threshold = 80
    cfg.buffer_only = False
    consumer = GatewayStreamConsumer.__new__(GatewayStreamConsumer)
    consumer.cfg = cfg
    consumer._accumulated = next(iter(LIVE_GATEWAY_SILENT_MARKERS))
    consumer._stream_ledger = ""
    consumer._last_sent_text = ""
    consumer._already_sent = False
    consumer._message_id = None
    consumer._tool_progress_active = False
    consumer._native_stream_opened = False
    consumer._use_native_streaming = False
    consumer._use_draft_streaming = False
    consumer._native_last_pushed_len = 0
    consumer._turn_split_delivery = False
    consumer._clear_turn_final_flags = MagicMock()
    consumer.chat_id = "test"
    consumer._tool_progress_lines = []
    consumer._sent_segments = []
    consumer._stale_preview_ids = lambda: []
    consumer._last_edit_time = 0
    consumer._flood_strikes = 0
    consumer._current_edit_interval = 0.1
    consumer._send_or_edit = AsyncMock()

    tick = MagicMock()
    tick.is_interim = False
    tick.got_done = False
    tick.got_segment_break = True
    tick.draft_final_fresh_send = False

    asyncio.run(consumer._push_update(tick))
    assert consumer._accumulated == "", "segment buffer should be cleared"
    consumer._send_or_edit.assert_not_awaited()
