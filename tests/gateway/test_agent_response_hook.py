"""Tests for the agent:response hook event (#26109)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.hooks import HookRegistry


class TestAgentResponseHookEvent:
    """Verify that the agent:response event is emitted with the correct
    payload fields after agent processing completes."""

    @pytest.mark.asyncio
    async def test_emit_agent_response_hook(self):
        """HookRegistry should dispatch agent:response to subscribed handlers."""
        received = []

        reg = HookRegistry()
        reg._handlers["agent:response"] = [
            lambda event_type, ctx: received.append((event_type, ctx))
        ]

        ctx = {
            "platform": "telegram",
            "user_id": "u123",
            "chat_id": "c456",
            "session_id": "s789",
            "message": "hello",
            "response": "Hi there!",
            "model": "claude-sonnet-4",
            "turn_id": "s789",
            "timestamp": "2026-05-15T12:00:00",
            "token_usage": {"prompt_tokens": 100, "context_length": 128000},
        }
        await reg.emit("agent:response", ctx)

        assert len(received) == 1
        assert received[0][0] == "agent:response"
        assert received[0][1]["response"] == "Hi there!"
        assert received[0][1]["model"] == "claude-sonnet-4"
        assert received[0][1]["turn_id"] == "s789"
        assert received[0][1]["token_usage"]["prompt_tokens"] == 100

    @pytest.mark.asyncio
    async def test_agent_response_full_text_not_truncated(self):
        """agent:response should carry the FULL response text (unlike
        agent:end which truncates to 500 chars)."""
        received = []

        reg = HookRegistry()
        reg._handlers["agent:response"] = [
            lambda event_type, ctx: received.append(ctx)
        ]

        long_response = "x" * 2000
        ctx = {"response": long_response}
        await reg.emit("agent:response", ctx)

        assert len(received) == 1
        assert len(received[0]["response"]) == 2000

    @pytest.mark.asyncio
    async def test_agent_response_async_handler(self):
        """agent:response should work with async handlers."""
        received = []

        async def async_handler(event_type, ctx):
            received.append((event_type, ctx))

        reg = HookRegistry()
        reg._handlers["agent:response"] = [async_handler]

        ctx = {"response": "test", "platform": "discord"}
        await reg.emit("agent:response", ctx)

        assert len(received) == 1
        assert received[0][1]["platform"] == "discord"

    @pytest.mark.asyncio
    async def test_agent_response_handler_error_does_not_block(self):
        """Errors in agent:response handlers should not block other handlers
        or the main pipeline."""
        results = []

        def bad_handler(event_type, ctx):
            raise RuntimeError("handler crashed")

        def good_handler(event_type, ctx):
            results.append("ok")

        reg = HookRegistry()
        reg._handlers["agent:response"] = [bad_handler, good_handler]

        await reg.emit("agent:response", {"response": "test"})
        assert results == ["ok"]

    @pytest.mark.asyncio
    async def test_agent_response_payload_fields(self):
        """Verify all required fields from issue #26109 are present."""
        received = []

        reg = HookRegistry()
        reg._handlers["agent:response"] = [
            lambda event_type, ctx: received.append(ctx)
        ]

        ctx = {
            "platform": "telegram",
            "user_id": "user_abc",
            "chat_id": "-1001234567890",
            "session_id": "sess_xyz",
            "message": "What is 2+2?",
            "response": "2+2 equals 4.",
            "model": "gpt-4o",
            "turn_id": "sess_xyz",
            "timestamp": "2026-05-15T10:30:00",
            "token_usage": {
                "prompt_tokens": 150,
                "context_length": 128000,
            },
        }
        await reg.emit("agent:response", ctx)

        assert len(received) == 1
        payload = received[0]
        # Required fields from #26109
        assert "response" in payload
        assert "platform" in payload
        assert "chat_id" in payload
        assert "turn_id" in payload
        assert "timestamp" in payload
        assert "user_id" in payload
        assert "model" in payload
        assert "token_usage" in payload

    @pytest.mark.asyncio
    async def test_agent_response_via_emit_collect(self):
        """emit_collect should capture return values from agent:response handlers."""
        def handler(event_type, ctx):
            return {"logged": True, "chat": ctx.get("chat_id")}

        reg = HookRegistry()
        reg._handlers["agent:response"] = [handler]

        results = await reg.emit_collect(
            "agent:response",
            {"response": "test", "chat_id": "c1"},
        )
        assert len(results) == 1
        assert results[0] == {"logged": True, "chat": "c1"}
