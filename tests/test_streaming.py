"""Tests for streaming token output — accumulator shape, callback order, fallback."""

import queue
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

import pytest

from run_agent import AIAgent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_tool_defs(*names):
    return [
        {"type": "function", "function": {"name": n, "description": f"{n}", "parameters": {"type": "object", "properties": {}}}}
        for n in names
    ]


@pytest.fixture()
def agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        cb = MagicMock()
        a = AIAgent(
            api_key="test-key-1234567890",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            stream_delta_callback=cb,
        )
        a.client = MagicMock()
        a._stream_cb = cb
        return a


# ---------------------------------------------------------------------------
# Helpers — fake streaming chunks
# ---------------------------------------------------------------------------

def _chunk(content=None, tool_call_delta=None, finish_reason=None, usage=None, model=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_call_delta,
                            reasoning_content=None, reasoning=None)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    c = SimpleNamespace(choices=[choice])
    if usage is not None:
        c.usage = SimpleNamespace(**usage)
    if model:
        c.model = model
    return c


def _usage_chunk(**kw):
    c = SimpleNamespace(choices=[], usage=SimpleNamespace(**kw))
    return c


def _tc_delta(index, id=None, name=None, arguments=None, type=None):
    fn = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=id, type=type, function=fn)


# ---------------------------------------------------------------------------
# Tests: accumulator shape
# ---------------------------------------------------------------------------


class TestStreamingAccumulator:
    def test_text_only_response(self, agent):
        """Streaming text-only response produces correct synthetic shape."""
        chunks = [
            _chunk(content="Hello", model="test/m"),
            _chunk(content=" world"),
            _chunk(finish_reason="stop"),
            _usage_chunk(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        ]
        agent.client.chat.completions.create.return_value = iter(chunks)

        resp = agent._interruptible_streaming_api_call({"model": "test"})

        assert resp.choices[0].message.content == "Hello world"
        assert resp.choices[0].message.tool_calls is None
        assert resp.choices[0].finish_reason == "stop"
        assert resp.usage.prompt_tokens == 10
        assert resp.model == "test/m"

    def test_tool_call_response(self, agent):
        """Streaming tool-call response accumulates function name + arguments."""
        chunks = [
            _chunk(tool_call_delta=[_tc_delta(0, id="call_1", name="web_search", arguments='{"q', type="function")]),
            _chunk(tool_call_delta=[_tc_delta(0, arguments='uery": "hi"}')]),
            _chunk(finish_reason="tool_calls"),
        ]
        agent.client.chat.completions.create.return_value = iter(chunks)

        resp = agent._interruptible_streaming_api_call({"model": "test"})

        tc = resp.choices[0].message.tool_calls
        assert tc is not None
        assert len(tc) == 1
        assert tc[0].id == "call_1"
        assert tc[0].function.name == "web_search"
        assert tc[0].function.arguments == '{"query": "hi"}'
        assert resp.choices[0].finish_reason == "tool_calls"

    def test_mixed_content_and_tool_calls(self, agent):
        """Content + tool calls in same stream are both accumulated."""
        chunks = [
            _chunk(content="Let me check."),
            _chunk(tool_call_delta=[_tc_delta(0, id="c1", name="web_search", arguments="{}", type="function")]),
            _chunk(finish_reason="tool_calls"),
        ]
        agent.client.chat.completions.create.return_value = iter(chunks)

        resp = agent._interruptible_streaming_api_call({"model": "test"})

        assert resp.choices[0].message.content == "Let me check."
        assert len(resp.choices[0].message.tool_calls) == 1


class TestStreamingCallbacks:
    def test_deltas_fire_in_order(self, agent):
        """stream_delta_callback receives content deltas in order."""
        received = []
        agent.stream_delta_callback = lambda t: received.append(t)
        chunks = [_chunk(content="a"), _chunk(content="b"), _chunk(content="c"), _chunk(finish_reason="stop")]
        agent.client.chat.completions.create.return_value = iter(chunks)

        agent._interruptible_streaming_api_call({"model": "test"})

        assert received == ["a", "b", "c"]

    def test_on_first_delta_fires_once(self, agent):
        first = MagicMock()
        chunks = [_chunk(content="x"), _chunk(content="y"), _chunk(finish_reason="stop")]
        agent.client.chat.completions.create.return_value = iter(chunks)

        agent._interruptible_streaming_api_call({"model": "test"}, on_first_delta=first)

        first.assert_called_once()

    def test_tool_only_does_not_fire_callback(self, agent):
        """Tool-call-only stream does not invoke stream_delta_callback."""
        received = []
        agent.stream_delta_callback = lambda t: received.append(t)
        chunks = [
            _chunk(tool_call_delta=[_tc_delta(0, id="c1", name="t", arguments="{}", type="function")]),
            _chunk(finish_reason="tool_calls"),
        ]
        agent.client.chat.completions.create.return_value = iter(chunks)

        agent._interruptible_streaming_api_call({"model": "test"})

        assert received == []


class TestStreamingFallback:
    def test_stream_error_falls_back(self, agent):
        """When streaming fails with 'not support', falls back to non-streaming."""
        agent.client.chat.completions.create.side_effect = [
            Exception("streaming not supported by this provider"),
            SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content="ok", tool_calls=None, reasoning=None, reasoning_content=None, reasoning_details=None),
                    finish_reason="stop",
                )],
                usage=None,
                model="test/m",
            ),
        ]

        resp = agent._interruptible_streaming_api_call({"model": "test"})

        assert resp.choices[0].message.content == "ok"
        assert agent.client.chat.completions.create.call_count == 2

    def test_non_stream_error_raises(self, agent):
        """Non-stream-related errors propagate normally."""
        agent.client.chat.completions.create.side_effect = ValueError("bad request")

        with pytest.raises(ValueError, match="bad request"):
            agent._interruptible_streaming_api_call({"model": "test"})


# ---------------------------------------------------------------------------
# Tests: base.py already_sent contract
# ---------------------------------------------------------------------------

class TestAlreadySentContract:
    def _make_adapter(self, send_side_effect=None):
        from gateway.platforms.base import BasePlatformAdapter, SendResult
        from gateway.config import Platform, PlatformConfig

        class FakeAdapter(BasePlatformAdapter):
            async def connect(self): return True
            async def disconnect(self): pass
            async def get_chat_info(self, chat_id): return {"name": "test"}
            async def send(self, chat_id, content, reply_to=None, metadata=None):
                if send_side_effect is not None:
                    send_side_effect(content)
                return SendResult(success=True, message_id="1")

        cfg = PlatformConfig(enabled=True)
        adapter = FakeAdapter(cfg, Platform.TELEGRAM)
        adapter._running = True
        return adapter

    @pytest.mark.asyncio
    async def test_already_sent_skips_send(self):
        """Handler returning already_sent=True prevents base from calling send()."""
        from gateway.platforms.base import MessageEvent
        from gateway.config import Platform
        from gateway.session import SessionSource

        sent = []
        adapter = self._make_adapter(send_side_effect=lambda c: sent.append(c))

        async def handler(event):
            return {"content": "hello", "already_sent": True}
        adapter.set_message_handler(handler)

        event = MessageEvent(
            text="hi",
            source=SessionSource(platform=Platform.TELEGRAM, chat_id="1", user_id="u1"),
        )
        await adapter._process_message_background(event, "s1")

        assert sent == [], "send() should not be called when already_sent=True"

    @pytest.mark.asyncio
    async def test_string_response_sends_normally(self):
        """Handler returning a plain string triggers send() as before."""
        from gateway.platforms.base import MessageEvent
        from gateway.config import Platform
        from gateway.session import SessionSource

        sent = []
        adapter = self._make_adapter(send_side_effect=lambda c: sent.append(c))

        async def handler(event):
            return "hello"
        adapter.set_message_handler(handler)

        event = MessageEvent(
            text="hi",
            source=SessionSource(platform=Platform.TELEGRAM, chat_id="1", user_id="u1"),
        )
        await adapter._process_message_background(event, "s1")

        assert "hello" in sent
