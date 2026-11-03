"""Bounded native tool-progress drafts keep actionable updates and pure final text."""

from unittest.mock import patch

import pytest

from gateway.config import Platform, StreamingConfig
from gateway.run_turn import GatewayTurnMixin
from gateway.session import SessionSource
from gateway.stream_consumer import GatewayStreamConsumer
from tests.gateway.test_stream_consumer_tool_progress import _make_native_streaming_adapter


@pytest.mark.parametrize(
    ("display", "cap"),
    [
        ({}, 8),
        ({"tool_progress_max_lines": "4"}, 4),
        ({"tool_progress_max_lines": 6, "platforms": {"wecom": {"tool_progress_max_lines": "4"}}}, 4),
        ({"tool_progress_max_lines": 0}, 0),
        ({"tool_progress_max_lines": "not-a-number"}, 0),
    ],
)
@pytest.mark.asyncio
async def test_gateway_config_reaches_native_progress_frames(display, cap):
    adapter = _make_native_streaming_adapter()
    source = SessionSource(platform=Platform.WECOM, chat_id="chat-1", chat_type="dm")
    with patch("gateway.run._load_gateway_config", return_value={"display": display}):
        config, _ = GatewayTurnMixin()._build_stream_consumer_config(
            source, StreamingConfig(), adapter, on_missing_cursor="raise",
        )
    assert config.max_tool_progress_lines == cap
    consumer = GatewayStreamConsumer(adapter, source.chat_id, config)
    consumer._use_native_streaming = True
    updates = [
        "Searching one", "terminal: git status (×3)", "Reading config",
        "⚠️ Rate limit warning", "Searching two", "Searching three",
    ]
    for update in updates:
        consumer.on_tool_progress(update)
    consumer._drain_queue()
    await consumer._native_push(consumer._compose_frame_content(), finalize=False, is_turn_final=False)
    frame = adapter.frames[-1]["text"]
    if cap == 0:
        assert frame.splitlines() == updates
    elif cap == 4:
        assert frame.splitlines() == [
            "⏳ Working · 6 tool updates", "… 2 earlier updates omitted",
            updates[1], updates[3], updates[4], updates[5],
        ]
    else:
        assert frame.splitlines() == ["⏳ Working · 6 tool updates", *updates]

    # Real delta delivery clears the draft; a later overlay never enters the final.
    consumer.on_delta("The answer is 42.")
    consumer._drain_queue()
    assert consumer._compose_frame_content() == "The answer is 42."
    consumer.on_tool_progress("terminal: verify")
    consumer.finish()
    await consumer.run()
    assert adapter.frames[-1]["finalize"] is True
    assert adapter.frames[-1]["text"] == "The answer is 42."
    consumer._reset_segment_state()
    assert consumer._compose_frame_content() == ""


@pytest.mark.asyncio
async def test_protected_progress_can_exceed_window():
    from gateway.stream_consumer import StreamConsumerConfig

    adapter = _make_native_streaming_adapter()
    consumer = GatewayStreamConsumer(adapter, "chat-1", StreamConsumerConfig(max_tool_progress_lines=1))
    consumer._use_native_streaming = True
    for update in ("old", "❌ failed", "terminal (×3)", "⚠️ warning", "recent"):
        consumer.on_tool_progress(update)
    consumer._drain_queue()
    await consumer._native_push(consumer._compose_frame_content(), finalize=False, is_turn_final=False)
    assert adapter.frames[-1]["text"].splitlines() == [
        "⏳ Working · 5 tool updates", "… 2 earlier updates omitted",
        "❌ failed", "terminal (×3)", "⚠️ warning",
    ]
