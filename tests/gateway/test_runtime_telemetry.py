"""Tests for gateway runtime telemetry helpers in GatewayRunner."""

from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_event(text: str, message_type: MessageType = MessageType.TEXT) -> MessageEvent:
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="chan-1",
        chat_type="group",
        user_id="user-1",
        user_name="tester",
    )
    return MessageEvent(text=text, message_type=message_type, source=source)


class TestTaskClassification:
    def test_command_classification(self):
        event = _make_event("/reset")
        assert GatewayRunner._classify_task_type(event, event.text) == "command"

    def test_media_classifications(self):
        assert GatewayRunner._classify_task_type(_make_event("pic", MessageType.PHOTO), "pic") == "vision"
        assert GatewayRunner._classify_task_type(_make_event("clip", MessageType.VIDEO), "clip") == "vision"
        assert GatewayRunner._classify_task_type(_make_event("voice", MessageType.VOICE), "voice") == "audio"
        assert GatewayRunner._classify_task_type(_make_event("audio", MessageType.AUDIO), "audio") == "audio"
        assert GatewayRunner._classify_task_type(_make_event("doc", MessageType.DOCUMENT), "doc") == "document"

    def test_keyword_fallback_classifications(self):
        text = "I got a traceback from pytest, can you debug this exception?"
        assert GatewayRunner._classify_task_type(_make_event(text), text) == "code"

        text = "Please summarize this and explain key points"
        assert GatewayRunner._classify_task_type(_make_event(text), text) == "analysis"

        text = "just chatting"
        assert GatewayRunner._classify_task_type(_make_event(text), text) == "chat"


class TestTokenCostClass:
    @pytest.mark.parametrize(
        "total_tokens, expected",
        [
            (0, "tiny"),
            (2000, "tiny"),
            (2001, "small"),
            (8000, "small"),
            (8001, "medium"),
            (24000, "medium"),
            (24001, "large"),
        ],
    )
    def test_token_cost_class_boundaries(self, total_tokens, expected):
        assert GatewayRunner._token_cost_class(total_tokens) == expected


class TestEmitMessageTelemetry:
    @pytest.mark.asyncio
    async def test_emit_message_telemetry_success(self):
        runner = object.__new__(GatewayRunner)
        runner.hooks = AsyncMock()

        payload = {"task_type": "analysis", "latency_ms": 123}
        await runner._emit_message_telemetry(payload)

        runner.hooks.emit.assert_awaited_once_with("telemetry:message", payload)

    @pytest.mark.asyncio
    async def test_emit_message_telemetry_swallow_errors(self, caplog):
        runner = object.__new__(GatewayRunner)
        runner.hooks = AsyncMock()
        runner.hooks.emit.side_effect = RuntimeError("hook exploded")

        with caplog.at_level("DEBUG", logger="gateway.run"):
            await runner._emit_message_telemetry({"task_type": "chat"})

        assert "Telemetry hook emission failed" in caplog.text
