"""Tests for streaming token delivery infrastructure.

Tests the unified streaming API call, delta callbacks, tool-call
suppression, provider fallback, and CLI streaming display.
"""
import json
import threading
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_stream_chunk(
    content=None, tool_calls=None, finish_reason=None,
    model=None, reasoning_content=None, usage=None,
):
    """Build a mock streaming chunk matching OpenAI's ChatCompletionChunk shape."""
    delta = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        reasoning_content=reasoning_content,
        reasoning=None,
    )
    choice = SimpleNamespace(
        index=0,
        delta=delta,
        finish_reason=finish_reason,
    )
    chunk = SimpleNamespace(
        choices=[choice],
        model=model,
        usage=usage,
    )
    return chunk


def _make_tool_call_delta(index=0, tc_id=None, name=None, arguments=None, extra_content=None, model_extra=None):
    """Build a mock tool call delta."""
    func = SimpleNamespace(name=name, arguments=arguments)
    delta = SimpleNamespace(index=index, id=tc_id, function=func)
    if extra_content is not None:
        delta.extra_content = extra_content
    if model_extra is not None:
        delta.model_extra = model_extra
    return delta


def _make_empty_chunk(model=None, usage=None):
    """Build a chunk with no choices (usage-only final chunk)."""
    return SimpleNamespace(choices=[], model=model, usage=usage)


# ── Test: Streaming Accumulator ──────────────────────────────────────────


class TestStreamingAccumulator:
    """Verify that _interruptible_streaming_api_call accumulates content
    and tool calls into a response matching the non-streaming shape."""

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_text_only_response(self, mock_close, mock_create):
        """Text-only stream produces correct response shape."""
        from run_agent import AIAgent

        chunks = [
            _make_stream_chunk(content="Hello"),
            _make_stream_chunk(content=" world"),
            _make_stream_chunk(content="!", finish_reason="stop", model="test-model"),
            _make_empty_chunk(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=3)),
        ]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(chunks)
        mock_create.return_value = mock_client

        agent = AIAgent(
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False

        response = agent._interruptible_streaming_api_call({})

        assert response.choices[0].message.content == "Hello world!"
        assert response.choices[0].message.tool_calls is None
        assert response.choices[0].finish_reason == "stop"
        assert response.usage is not None
        assert response.usage.completion_tokens == 3

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_tool_call_response(self, mock_close, mock_create):
        """Tool call stream accumulates ID, name, and arguments."""
        from run_agent import AIAgent

        chunks = [
            _make_stream_chunk(tool_calls=[
                _make_tool_call_delta(index=0, tc_id="call_123", name="terminal")
            ]),
            _make_stream_chunk(tool_calls=[
                _make_tool_call_delta(index=0, arguments='{"command":')
            ]),
            _make_stream_chunk(tool_calls=[
                _make_tool_call_delta(index=0, arguments=' "ls"}')
            ]),
            _make_stream_chunk(finish_reason="tool_calls"),
        ]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(chunks)
        mock_create.return_value = mock_client

        agent = AIAgent(
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False

        response = agent._interruptible_streaming_api_call({})

        tc = response.choices[0].message.tool_calls
        assert tc is not None
        assert len(tc) == 1
        assert tc[0].id == "call_123"
        assert tc[0].function.name == "terminal"
        assert tc[0].function.arguments == '{"command": "ls"}'

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_tool_call_extra_content_preserved(self, mock_close, mock_create):
        """Streamed tool calls preserve provider-specific extra_content metadata."""
        from run_agent import AIAgent

        chunks = [
            _make_stream_chunk(tool_calls=[
                _make_tool_call_delta(
                    index=0,
                    tc_id="call_gemini",
                    name="cronjob",
                    model_extra={
                        "extra_content": {
                            "google": {"thought_signature": "sig-123"}
                        }
                    },
                )
            ]),
            _make_stream_chunk(tool_calls=[
                _make_tool_call_delta(index=0, arguments='{"task": "deep index on ."}')
            ]),
            _make_stream_chunk(finish_reason="tool_calls"),
        ]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(chunks)
        mock_create.return_value = mock_client

        agent = AIAgent(
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False

        response = agent._interruptible_streaming_api_call({})

        tc = response.choices[0].message.tool_calls
        assert tc is not None
        assert tc[0].extra_content == {
            "google": {"thought_signature": "sig-123"}
        }

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_mixed_content_and_tool_calls(self, mock_close, mock_create):
        """Stream with both text and tool calls accumulates both."""
        from run_agent import AIAgent

        chunks = [
            _make_stream_chunk(content="Let me check"),
            _make_stream_chunk(tool_calls=[
                _make_tool_call_delta(index=0, tc_id="call_456", name="web_search")
            ]),
            _make_stream_chunk(tool_calls=[
                _make_tool_call_delta(index=0, arguments='{"query": "test"}')
            ]),
            _make_stream_chunk(finish_reason="tool_calls"),
        ]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(chunks)
        mock_create.return_value = mock_client

        agent = AIAgent(
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False

        response = agent._interruptible_streaming_api_call({})

        assert response.choices[0].message.content == "Let me check"
        assert len(response.choices[0].message.tool_calls) == 1


# ── Test: Streaming Callbacks ────────────────────────────────────────────


class TestStreamingCallbacks:
    """Verify that delta callbacks fire correctly."""

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_deltas_fire_in_order(self, mock_close, mock_create):
        """Callbacks receive text deltas in order."""
        from run_agent import AIAgent

        chunks = [
            _make_stream_chunk(content="a"),
            _make_stream_chunk(content="b"),
            _make_stream_chunk(content="c"),
            _make_stream_chunk(finish_reason="stop"),
        ]

        deltas = []

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(chunks)
        mock_create.return_value = mock_client

        agent = AIAgent(
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            stream_delta_callback=lambda t: deltas.append(t),
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False

        agent._interruptible_streaming_api_call({})

        assert deltas == ["a", "b", "c"]

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_on_first_delta_fires_once(self, mock_close, mock_create):
        """on_first_delta callback fires exactly once."""
        from run_agent import AIAgent

        chunks = [
            _make_stream_chunk(content="a"),
            _make_stream_chunk(content="b"),
            _make_stream_chunk(finish_reason="stop"),
        ]

        first_delta_calls = []

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(chunks)
        mock_create.return_value = mock_client

        agent = AIAgent(
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False

        agent._interruptible_streaming_api_call(
            {}, on_first_delta=lambda: first_delta_calls.append(True)
        )

        assert len(first_delta_calls) == 1

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_tool_only_does_not_fire_callback(self, mock_close, mock_create):
        """Tool-call-only stream does not fire the delta callback."""
        from run_agent import AIAgent

        chunks = [
            _make_stream_chunk(tool_calls=[
                _make_tool_call_delta(index=0, tc_id="call_789", name="terminal")
            ]),
            _make_stream_chunk(tool_calls=[
                _make_tool_call_delta(index=0, arguments='{"command": "ls"}')
            ]),
            _make_stream_chunk(finish_reason="tool_calls"),
        ]

        deltas = []

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(chunks)
        mock_create.return_value = mock_client

        agent = AIAgent(
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            stream_delta_callback=lambda t: deltas.append(t),
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False

        agent._interruptible_streaming_api_call({})

        assert deltas == []

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_text_suppressed_when_tool_calls_present(self, mock_close, mock_create):
        """Text deltas are suppressed when tool calls are also in the stream."""
        from run_agent import AIAgent

        chunks = [
            _make_stream_chunk(content="thinking..."),
            _make_stream_chunk(tool_calls=[
                _make_tool_call_delta(index=0, tc_id="call_abc", name="read_file")
            ]),
            _make_stream_chunk(content=" more text"),
            _make_stream_chunk(finish_reason="tool_calls"),
        ]

        deltas = []

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(chunks)
        mock_create.return_value = mock_client

        agent = AIAgent(
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            stream_delta_callback=lambda t: deltas.append(t),
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False

        response = agent._interruptible_streaming_api_call({})

        # Text before tool call IS fired (we don't know yet it will have tools)
        assert "thinking..." in deltas
        # Text after tool call is NOT fired
        assert " more text" not in deltas
        # But content is still accumulated in the response
        assert response.choices[0].message.content == "thinking... more text"


# ── Test: Streaming Fallback ────────────────────────────────────────────


class TestStreamingFallback:
    """Verify fallback to non-streaming on ANY streaming error."""

    @patch("run_agent.AIAgent._interruptible_api_call")
    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_stream_error_falls_back(self, mock_close, mock_create, mock_non_stream):
        """'not supported' error triggers fallback to non-streaming."""
        from run_agent import AIAgent

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception(
            "Streaming is not supported for this model"
        )
        mock_create.return_value = mock_client

        fallback_response = SimpleNamespace(
            id="fallback",
            model="test",
            choices=[SimpleNamespace(
                index=0,
                message=SimpleNamespace(
                    role="assistant",
                    content="fallback response",
                    tool_calls=None,
                    reasoning_content=None,
                ),
                finish_reason="stop",
            )],
            usage=None,
        )
        mock_non_stream.return_value = fallback_response

        agent = AIAgent(
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False

        response = agent._interruptible_streaming_api_call({})

        assert response.choices[0].message.content == "fallback response"
        mock_non_stream.assert_called_once()

    @patch("run_agent.AIAgent._interruptible_api_call")
    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_any_stream_error_falls_back(self, mock_close, mock_create, mock_non_stream):
        """ANY streaming error triggers fallback — not just specific messages."""
        from run_agent import AIAgent

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception(
            "Connection reset by peer"
        )
        mock_create.return_value = mock_client

        fallback_response = SimpleNamespace(
            id="fallback",
            model="test",
            choices=[SimpleNamespace(
                index=0,
                message=SimpleNamespace(
                    role="assistant",
                    content="fallback after connection error",
                    tool_calls=None,
                    reasoning_content=None,
                ),
                finish_reason="stop",
            )],
            usage=None,
        )
        mock_non_stream.return_value = fallback_response

        agent = AIAgent(
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False

        response = agent._interruptible_streaming_api_call({})

        assert response.choices[0].message.content == "fallback after connection error"
        mock_non_stream.assert_called_once()

    @patch("run_agent.AIAgent._interruptible_api_call")
    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_fallback_error_propagates(self, mock_close, mock_create, mock_non_stream):
        """When both streaming AND fallback fail, the fallback error propagates."""
        from run_agent import AIAgent

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("stream broke")
        mock_create.return_value = mock_client

        mock_non_stream.side_effect = Exception("Rate limit exceeded")

        agent = AIAgent(
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False

        with pytest.raises(Exception, match="Rate limit exceeded"):
            agent._interruptible_streaming_api_call({})

    @patch("run_agent.AIAgent._interruptible_api_call")
    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_exhausted_transient_stream_error_falls_back(self, mock_close, mock_create, mock_non_stream):
        """Transient stream errors retry first, then fall back after retries are exhausted."""
        from run_agent import AIAgent
        import httpx

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = httpx.ConnectError("socket closed")
        mock_create.return_value = mock_client

        fallback_response = SimpleNamespace(
            id="fallback",
            model="test",
            choices=[SimpleNamespace(
                index=0,
                message=SimpleNamespace(
                    role="assistant",
                    content="fallback after retries exhausted",
                    tool_calls=None,
                    reasoning_content=None,
                ),
                finish_reason="stop",
            )],
            usage=None,
        )
        mock_non_stream.return_value = fallback_response

        agent = AIAgent(
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False

        response = agent._interruptible_streaming_api_call({})

        assert response.choices[0].message.content == "fallback after retries exhausted"
        assert mock_client.chat.completions.create.call_count == 3
        mock_non_stream.assert_called_once()
        assert mock_close.call_count >= 1


# ── Test: Reasoning Streaming ────────────────────────────────────────────


class TestReasoningStreaming:
    """Verify reasoning content is accumulated and callback fires."""

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_reasoning_callback_fires(self, mock_close, mock_create):
        """Reasoning deltas fire the reasoning_callback."""
        from run_agent import AIAgent

        chunks = [
            _make_stream_chunk(reasoning_content="Let me think"),
            _make_stream_chunk(reasoning_content=" about this"),
            _make_stream_chunk(content="The answer is 42"),
            _make_stream_chunk(finish_reason="stop"),
        ]

        reasoning_deltas = []
        text_deltas = []

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(chunks)
        mock_create.return_value = mock_client

        agent = AIAgent(
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            stream_delta_callback=lambda t: text_deltas.append(t),
            reasoning_callback=lambda t: reasoning_deltas.append(t),
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False

        response = agent._interruptible_streaming_api_call({})

        assert reasoning_deltas == ["Let me think", " about this"]
        assert text_deltas == ["The answer is 42"]
        assert response.choices[0].message.reasoning_content == "Let me think about this"
        assert response.choices[0].message.content == "The answer is 42"


# ── Test: _has_stream_consumers ──────────────────────────────────────────


class TestHasStreamConsumers:
    """Verify _has_stream_consumers() detects registered callbacks."""

    def test_no_consumers(self):
        from run_agent import AIAgent
        agent = AIAgent(
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        assert agent._has_stream_consumers() is False

    def test_delta_callback_set(self):
        from run_agent import AIAgent
        agent = AIAgent(
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            stream_delta_callback=lambda t: None,
        )
        assert agent._has_stream_consumers() is True

    def test_stream_callback_set(self):
        from run_agent import AIAgent
        agent = AIAgent(
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent._stream_callback = lambda t: None
        assert agent._has_stream_consumers() is True


# ── Test: Codex stream fires callbacks ────────────────────────────────


class TestCodexStreamCallbacks:
    """Verify _run_codex_stream fires delta callbacks."""

    def test_codex_text_delta_fires_callback(self):
        from run_agent import AIAgent

        deltas = []

        agent = AIAgent(
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            stream_delta_callback=lambda t: deltas.append(t),
        )
        agent.api_mode = "codex_responses"
        agent._interrupt_requested = False

        # Mock the stream context manager
        mock_event_text = SimpleNamespace(
            type="response.output_text.delta",
            delta="Hello from Codex!",
        )
        mock_event_done = SimpleNamespace(
            type="response.completed",
            delta="",
        )

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.__iter__ = MagicMock(return_value=iter([mock_event_text, mock_event_done]))
        mock_stream.get_final_response.return_value = SimpleNamespace(
            output=[SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="Hello from Codex!")],
            )],
            status="completed",
        )

        mock_client = MagicMock()
        mock_client.responses.stream.return_value = mock_stream

        response = agent._run_codex_stream({}, client=mock_client)
        assert "Hello from Codex!" in deltas
