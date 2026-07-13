"""Regression tests for reasoning display after a streamed gateway reply."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.run import (
    GatewayRunner,
    _build_gateway_reasoning_block,
    _gateway_reasoning_delivery_mode,
    _is_successful_streamed_delivery,
)


def test_reasoning_block_uses_platform_style():
    config = {"display": {"show_reasoning": True}}

    assert _build_gateway_reasoning_block(
        config,
        Platform.TELEGRAM,
        "first\nsecond",
    ) == "💭 **Reasoning:**\n```\nfirst\nsecond\n```"
    assert _build_gateway_reasoning_block(
        config,
        Platform.DISCORD,
        "first\nsecond",
    ) == "-# 💭 Reasoning\n-# first\n-# second"


def test_mattermost_requires_platform_reasoning_opt_in():
    global_only = {"display": {"show_reasoning": True}}
    explicit = {
        "display": {
            "show_reasoning": False,
            "platforms": {"mattermost": {"show_reasoning": True}},
        }
    }

    assert _build_gateway_reasoning_block(
        global_only,
        Platform.MATTERMOST,
        "private scratch text",
    ) == ""
    assert _build_gateway_reasoning_block(
        explicit,
        Platform.MATTERMOST,
        "opted in",
    ) == "💭 **Reasoning:**\n```\nopted in\n```"


@pytest.mark.parametrize("config", [None, [], "bad"])
def test_reasoning_renderer_fails_closed_for_unavailable_config(config):
    assert _build_gateway_reasoning_block(
        config,
        Platform.TELEGRAM,
        "private scratch text",
        default=True,
    ) == ""


@pytest.mark.parametrize("last_reasoning", [None, {"text": "secret"}, b"secret"])
def test_reasoning_renderer_rejects_non_text_payloads(last_reasoning):
    assert _build_gateway_reasoning_block(
        {"display": {"show_reasoning": True}},
        Platform.TELEGRAM,
        last_reasoning,
    ) == ""


def test_reasoning_block_preserves_line_limit():
    block = _build_gateway_reasoning_block(
        {"display": {"show_reasoning": True}},
        Platform.TELEGRAM,
        "\n".join(f"line {number}" for number in range(18)),
    )

    assert "line 14" in block
    assert "line 15" not in block
    assert "_... (3 more lines)_" in block


@pytest.mark.parametrize(
    ("response", "already_sent", "failed", "silence", "expected"),
    [
        ("body", False, False, False, "prepend"),
        ("body", True, False, False, "trailing"),
        ("", True, False, False, "trailing"),
        ("error", True, True, False, "prepend"),
        ("", True, True, False, "none"),
        ("body", True, False, True, "none"),
    ],
)
def test_reasoning_delivery_mode_is_exactly_one_path(
    response,
    already_sent,
    failed,
    silence,
    expected,
):
    streamed_success = _is_successful_streamed_delivery(already_sent, failed)
    assert _gateway_reasoning_delivery_mode(
        response,
        streamed_success=streamed_success,
        intentional_silence=silence,
    ) == expected


@pytest.mark.parametrize(
    ("already_sent", "failed", "expected"),
    [(1, 0, True), ("sent", "", True), (True, 1, False), (None, False, False)],
)
def test_streamed_success_preserves_truthy_flag_compatibility(
    already_sent,
    failed,
    expected,
):
    assert _is_successful_streamed_delivery(already_sent, failed) is expected


@pytest.mark.asyncio
async def test_streamed_reasoning_uses_source_adapter_and_thread_metadata():
    source = SimpleNamespace(chat_id="chat-1", thread_id="topic-1")
    event = SimpleNamespace(message_id="message-1")
    adapter = SimpleNamespace(send=AsyncMock())
    runner = object.__new__(GatewayRunner)
    runner._adapter_for_source = lambda actual_source: adapter
    runner._reply_anchor_for_event = lambda actual_event: actual_event.message_id
    runner._thread_metadata_for_source = lambda actual_source, anchor: {
        "thread_id": actual_source.thread_id,
        "reply_to_message_id": anchor,
    }

    await runner._send_streamed_reasoning(source, event, "reasoning block")

    adapter.send.assert_awaited_once_with(
        "chat-1",
        "reasoning block",
        metadata={
            "thread_id": "topic-1",
            "reply_to_message_id": "message-1",
        },
    )


@pytest.mark.asyncio
async def test_streamed_extras_deliver_media_before_reasoning():
    calls = []
    source = SimpleNamespace(chat_id="chat-1")
    event = SimpleNamespace(message_id="message-1")
    runner = object.__new__(GatewayRunner)
    runner._adapter_for_source = lambda _source: object()
    runner._deliver_media_from_response = AsyncMock(
        side_effect=lambda *_args: calls.append("media")
    )
    runner._send_streamed_reasoning = AsyncMock(
        side_effect=lambda *_args: calls.append("reasoning")
    )

    await runner._deliver_streamed_response_extras(
        "streamed body with MEDIA tag",
        event,
        source,
        "reasoning block",
    )

    assert calls == ["media", "reasoning"]


@pytest.mark.asyncio
async def test_media_failure_does_not_suppress_streamed_reasoning():
    source = SimpleNamespace(chat_id="chat-1")
    event = SimpleNamespace(message_id="message-1")
    runner = object.__new__(GatewayRunner)
    runner._adapter_for_source = lambda _source: object()
    runner._deliver_media_from_response = AsyncMock(
        side_effect=RuntimeError("upload failed")
    )
    runner._send_streamed_reasoning = AsyncMock()

    await runner._deliver_streamed_response_extras(
        "streamed body with MEDIA tag",
        event,
        source,
        "reasoning block",
    )

    runner._send_streamed_reasoning.assert_awaited_once_with(
        source,
        event,
        "reasoning block",
    )
