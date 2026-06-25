"""Regression tests for the gateway /ingest command."""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_source(platform: Platform = Platform.SLACK) -> SessionSource:
    return SessionSource(
        platform=platform,
        user_id="U123",
        user_name="tester",
        chat_id="C123",
        chat_type="group",
    )


@pytest.mark.asyncio
async def test_ingest_command_forwards_structured_prompt_to_normal_handler():
    gw = GatewayRunner.__new__(GatewayRunner)
    gw.adapters = cast(Any, {Platform.SLACK: SimpleNamespace(typed_command_prefix="!")})

    captured = {}

    async def fake_handle_message(event):
        captured["event"] = event
        return "ok"

    gw._handle_message = AsyncMock(side_effect=fake_handle_message)

    source = _make_source()
    result = await gw._handle_ingest_command(
        MessageEvent(
            text="/ingest <#C0B7QVCLQF9> 채널의 6월 24일 업무만 요약해서 저장해줘",
            message_type=MessageType.COMMAND,
            source=source,
            message_id="m1",
            raw_message=MagicMock(),
            channel_prompt="channel prompt",
        )
    )

    assert result == "ok"
    forwarded = captured["event"]
    assert forwarded.message_type == MessageType.TEXT
    assert forwarded.source == source
    assert forwarded.message_id == "m1"
    assert forwarded.channel_prompt == "channel prompt"
    assert "Slack channel history ingest request." in forwarded.text
    assert "slack_history_read" in forwarded.text
    assert "Hermes session recall" in forwarded.text
    assert "<#C0B7QVCLQF9> 채널의 6월 24일 업무만 요약해서 저장해줘" in forwarded.text


@pytest.mark.asyncio
async def test_ingest_command_without_args_returns_slack_bang_usage():
    gw = GatewayRunner.__new__(GatewayRunner)
    gw.adapters = cast(Any, {Platform.SLACK: SimpleNamespace(typed_command_prefix="!")})

    result = await gw._handle_ingest_command(
        MessageEvent(
            text="/ingest",
            message_type=MessageType.COMMAND,
            source=_make_source(),
            message_id="m2",
        )
    )

    assert "!ingest" in result
    assert "Slack history request" in result
    assert "LLM-Wiki" in result
