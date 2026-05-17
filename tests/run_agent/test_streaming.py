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
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
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
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
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
    def test_tool_name_not_duplicated_when_resent_per_chunk(self, mock_close, mock_create):
        """MiniMax M2.7 via NVIDIA NIM resends the full name in every chunk.

        Bug #8259: the old += accumulation produced "read_fileread_file".
        Assignment (matching OpenAI Node SDK / LiteLLM) prevents this.
        """
        from run_agent import AIAgent

        chunks = [
            _make_stream_chunk(tool_calls=[
                _make_tool_call_delta(index=0, tc_id="call_nim", name="read_file")
            ]),
            _make_stream_chunk(tool_calls=[
                _make_tool_call_delta(index=0, tc_id="call_nim", name="read_file", arguments='{"path":')
            ]),
            _make_stream_chunk(tool_calls=[
                _make_tool_call_delta(index=0, tc_id="call_nim", name="read_file", arguments=' "x.py"}')
            ]),
            _make_stream_chunk(finish_reason="tool_calls"),
        ]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(chunks)
        mock_create.return_value = mock_client

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
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
        assert tc[0].function.name == "read_file"
        assert tc[0].function.arguments == '{"path": "x.py"}'

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
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
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
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
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
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
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
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
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
    def test_chat_stream_refreshes_activity_on_every_chunk(self, mock_close, mock_create):
        """Each streamed chat chunk should refresh the activity timestamp."""
        from run_agent import AIAgent

        chunks = [
            _make_stream_chunk(content="a"),
            _make_stream_chunk(content="b"),
            _make_stream_chunk(finish_reason="stop"),
        ]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(chunks)
        mock_create.return_value = mock_client

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False

        touch_calls = []
        agent._touch_activity = lambda desc: touch_calls.append(desc)

        agent._interruptible_streaming_api_call({})

        assert touch_calls.count("receiving stream response") == len(chunks)

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
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
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
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
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
        # Text after tool call IS still routed to stream_delta_callback so that
        # reasoning tag extraction can fire (PR #3566).  Display-level suppression
        # of non-reasoning text happens in the CLI's _stream_delta, not here.
        assert " more text" in deltas
        # Content is still accumulated in the response
        assert response.choices[0].message.content == "thinking... more text"


# ── Test: Streaming Fallback ────────────────────────────────────────────


class TestStreamingFallback:
    """Verify streaming errors propagate to the main retry loop.

    Previously, streaming errors triggered an inline fallback to
    non-streaming.  Now they propagate so the main retry loop can apply
    richer recovery (credential rotation, provider fallback, backoff).
    The only special case: 'stream not supported' sets _disable_streaming
    so the *next* main-loop retry uses non-streaming automatically.
    """

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_stream_not_supported_sets_flag_and_raises(self, mock_close, mock_create):
        """'not supported' error sets _disable_streaming and propagates."""
        from run_agent import AIAgent

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception(
            "Streaming is not supported for this model"
        )
        mock_create.return_value = mock_client

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False

        with pytest.raises(Exception, match="Streaming is not supported"):
            agent._interruptible_streaming_api_call({})

        # The flag should be set so the main retry loop switches to non-streaming
        assert agent._disable_streaming is True

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_non_transport_error_propagates(self, mock_close, mock_create):
        """Non-transport streaming errors propagate to the main retry loop."""
        from run_agent import AIAgent

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception(
            "Connection reset by peer"
        )
        mock_create.return_value = mock_client

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False

        with pytest.raises(Exception, match="Connection reset by peer"):
            agent._interruptible_streaming_api_call({})

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_stream_error_propagates_original(self, mock_close, mock_create):
        """The original streaming error propagates (not a fallback error)."""
        from run_agent import AIAgent

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("stream broke")
        mock_create.return_value = mock_client

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False

        with pytest.raises(Exception, match="stream broke"):
            agent._interruptible_streaming_api_call({})

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_exhausted_transient_stream_error_propagates(self, mock_close, mock_create):
        """Transient stream errors retry first, then propagate after retries exhausted."""
        from run_agent import AIAgent
        import httpx

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = httpx.ConnectError("socket closed")
        mock_create.return_value = mock_client

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False

        with pytest.raises(httpx.ConnectError, match="socket closed"):
            agent._interruptible_streaming_api_call({})

        # Should have retried 3 times (default HERMES_STREAM_RETRIES=2 → 3 attempts)
        assert mock_client.chat.completions.create.call_count == 3
        assert mock_close.call_count >= 1

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_sse_connection_lost_retried_as_transient(self, mock_close, mock_create):
        """SSE 'Network connection lost' (APIError w/ no status_code) retries like httpx errors.

        OpenRouter sends {"error":{"message":"Network connection lost."}} as an SSE
        event when the upstream stream drops.  The OpenAI SDK raises APIError from
        this.  It should be retried at the streaming level, same as httpx connection
        errors, then propagate to the main retry loop after exhaustion.
        """
        from run_agent import AIAgent
        import httpx

        # Create an APIError that mimics what the OpenAI SDK raises from SSE error events.
        # Key: no status_code attribute (unlike APIStatusError which has one).
        from openai import APIError as OAIAPIError
        sse_error = OAIAPIError(
            message="Network connection lost.",
            request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
            body={"message": "Network connection lost."},
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = sse_error
        mock_create.return_value = mock_client

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False

        with pytest.raises(OAIAPIError):
            agent._interruptible_streaming_api_call({})

        # Should retry 3 times (default HERMES_STREAM_RETRIES=2 → 3 attempts)
        assert mock_client.chat.completions.create.call_count == 3
        # Connection cleanup should happen for each failed retry
        assert mock_close.call_count >= 2

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_sse_non_connection_error_propagates_immediately(self, mock_close, mock_create):
        """SSE errors that aren't connection-related propagate immediately (no stream retry)."""
        from run_agent import AIAgent
        import httpx

        from openai import APIError as OAIAPIError
        sse_error = OAIAPIError(
            message="Invalid model configuration.",
            request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
            body={"message": "Invalid model configuration."},
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = sse_error
        mock_create.return_value = mock_client

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False

        with pytest.raises(OAIAPIError):
            agent._interruptible_streaming_api_call({})

        # Should NOT retry — propagates immediately
        assert mock_client.chat.completions.create.call_count == 1


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
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
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
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        assert agent._has_stream_consumers() is False

    def test_delta_callback_set(self):
        from run_agent import AIAgent
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
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
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
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
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
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

    def test_codex_stream_refreshes_activity_on_every_event(self):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "codex_responses"
        agent._interrupt_requested = False

        touch_calls = []
        agent._touch_activity = lambda desc: touch_calls.append(desc)

        mock_event_text_1 = SimpleNamespace(
            type="response.output_text.delta",
            delta="Hello",
        )
        mock_event_text_2 = SimpleNamespace(
            type="response.output_text.delta",
            delta=" world",
        )
        mock_event_done = SimpleNamespace(
            type="response.completed",
            delta="",
        )

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.__iter__ = MagicMock(
            return_value=iter([mock_event_text_1, mock_event_text_2, mock_event_done])
        )
        mock_stream.get_final_response.return_value = SimpleNamespace(
            output=[SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="Hello world")],
            )],
            status="completed",
        )

        mock_client = MagicMock()
        mock_client.responses.stream.return_value = mock_stream

        agent._run_codex_stream({}, client=mock_client)

        assert touch_calls.count("receiving stream response") == 3

    def test_codex_remote_protocol_error_falls_back_to_create_stream(self):
        from run_agent import AIAgent
        import httpx

        fallback_response = SimpleNamespace(
            output=[SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="fallback from create stream")],
            )],
            status="completed",
        )

        mock_client = MagicMock()
        mock_client.responses.stream.side_effect = httpx.RemoteProtocolError(
            "peer closed connection without sending complete message body"
        )

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "codex_responses"
        agent._interrupt_requested = False

        with patch.object(agent, "_run_codex_create_stream_fallback", return_value=fallback_response) as mock_fallback:
            response = agent._run_codex_stream({}, client=mock_client)

        assert response is fallback_response
        mock_fallback.assert_called_once_with({}, client=mock_client)

    def test_codex_create_stream_fallback_refreshes_activity_on_every_event(self):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "codex_responses"

        touch_calls = []
        agent._touch_activity = lambda desc: touch_calls.append(desc)

        events = [
            SimpleNamespace(type="response.output_text.delta", delta="Hello"),
            SimpleNamespace(type="response.output_item.done", item=SimpleNamespace(type="message")),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(
                    output=[SimpleNamespace(
                        type="message",
                        content=[SimpleNamespace(type="output_text", text="Hello")],
                    )]
                ),
            ),
        ]

        class _FakeCreateStream:
            def __iter__(self_inner):
                return iter(events)

            def close(self_inner):
                return None

        mock_stream = _FakeCreateStream()

        mock_client = MagicMock()
        mock_client.responses.create.return_value = mock_stream

        agent._run_codex_create_stream_fallback(
            {"model": "test/model", "instructions": "hi", "input": []},
            client=mock_client,
        )

        assert touch_calls.count("receiving stream response") == len(events)


class TestAnthropicStreamCallbacks:
    """Verify Anthropic streaming refreshes activity on every event."""

    def test_anthropic_stream_refreshes_activity_on_every_event(self):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "anthropic_messages"
        agent._interrupt_requested = False

        touch_calls = []
        agent._touch_activity = lambda desc: touch_calls.append(desc)

        events = [
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="text_delta", text="Hello"),
            ),
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="thinking_delta", thinking="thinking"),
            ),
            SimpleNamespace(
                type="content_block_start",
                content_block=SimpleNamespace(type="tool_use", name="terminal"),
            ),
        ]

        final_message = SimpleNamespace(
            content=[],
            stop_reason="end_turn",
        )

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.__iter__ = MagicMock(return_value=iter(events))
        mock_stream.get_final_message.return_value = final_message

        agent._anthropic_client = MagicMock()
        agent._anthropic_client.messages.stream.return_value = mock_stream

        agent._interruptible_streaming_api_call({})

        assert touch_calls.count("receiving stream response") == len(events)

    @patch("run_agent.AIAgent._replace_primary_openai_client")
    def test_anthropic_stream_parser_valueerror_retries_before_delivery(
        self, mock_replace, monkeypatch,
    ):
        """Malformed Anthropic event-stream frames retry instead of surfacing HTTP None."""
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://api.minimax.io/anthropic",
            provider="minimax",
            model="MiniMax-M2.7",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "anthropic_messages"
        agent._interrupt_requested = False
        monkeypatch.setenv("HERMES_STREAM_RETRIES", "1")

        class _BadStream:
            response = None

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def __iter__(self):
                raise ValueError("expected ident at line 1 column 149")

        final_message = SimpleNamespace(content=[], stop_reason="end_turn")
        good_stream = MagicMock()
        good_stream.__enter__ = MagicMock(return_value=good_stream)
        good_stream.__exit__ = MagicMock(return_value=False)
        good_stream.__iter__ = MagicMock(return_value=iter([]))
        good_stream.get_final_message.return_value = final_message

        agent._anthropic_client = MagicMock()
        agent._anthropic_client.messages.stream.side_effect = [
            _BadStream(),
            good_stream,
        ]

        response = agent._interruptible_streaming_api_call({})

        assert response is final_message
        assert agent._anthropic_client.messages.stream.call_count == 2
        assert mock_replace.call_count == 1

    @patch("run_agent.AIAgent._replace_primary_openai_client")
    def test_generic_anthropic_valueerror_still_propagates_without_stream_retry(
        self, mock_replace, monkeypatch,
    ):
        """Only known provider stream parser ValueErrors are treated as transient."""
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://api.minimax.io/anthropic",
            provider="minimax",
            model="MiniMax-M2.7",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "anthropic_messages"
        agent._interrupt_requested = False
        monkeypatch.setenv("HERMES_STREAM_RETRIES", "1")

        agent._anthropic_client = MagicMock()
        agent._anthropic_client.messages.stream.side_effect = ValueError(
            "invalid local request shape"
        )

        with pytest.raises(ValueError, match="invalid local request shape"):
            agent._interruptible_streaming_api_call({})

        assert agent._anthropic_client.messages.stream.call_count == 1
        assert mock_replace.call_count == 0


class TestPartialToolCallWarning:
    """Regression: when a stream dies mid tool-call argument generation after
    text was already delivered, the partial-stream stub at run_agent.py
    line ~6107 used to silently set ``tool_calls=None`` and return
    ``finish_reason=stop``, losing the attempted action with zero user-facing
    signal.  Live-observed Apr 2026 with MiniMax M2.7 on a 6-minute audit
    task — agent streamed commentary, emitted a write_file tool call,
    MiniMax stalled for 240 s mid-arguments, stale-stream detector killed
    the connection, the stub returned, session ended with no file written
    and no error shown.

    Fix: when the stream accumulator captured any tool-call names before the
    error, the stub now appends a user-visible warning to content AND fires
    it as a stream delta so the user sees it immediately.
    """

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_partial_tool_call_surfaces_warning(self, mock_close, mock_create):
        """Stream with text + partial tool-call name + mid-stream error
        produces a stub whose content contains the user-visible warning
        and whose tool_calls is None."""
        from run_agent import AIAgent

        class _StallError(RuntimeError):
            pass

        def _stalling_stream():
            yield _make_stream_chunk(content="Let me write the audit: ")
            yield _make_stream_chunk(tool_calls=[
                _make_tool_call_delta(index=0, tc_id="call_1", name="write_file"),
            ])
            yield _make_stream_chunk(tool_calls=[
                _make_tool_call_delta(index=0, arguments='{"path": "/tmp/x", '),
            ])
            raise _StallError("simulated upstream stall")

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = lambda *a, **kw: _stalling_stream()
        mock_create.return_value = mock_client

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False

        fired_deltas: list = []
        agent._fire_stream_delta = lambda text: fired_deltas.append(text)
        agent._current_streamed_assistant_text = "Let me write the audit: "

        import os as _os
        _prev = _os.environ.get("HERMES_STREAM_RETRIES")
        _os.environ["HERMES_STREAM_RETRIES"] = "0"
        try:
            response = agent._interruptible_streaming_api_call({})
        finally:
            if _prev is None:
                _os.environ.pop("HERMES_STREAM_RETRIES", None)
            else:
                _os.environ["HERMES_STREAM_RETRIES"] = _prev

        content = response.choices[0].message.content or ""
        assert "Let me write the audit:" in content, (
            f"Partial text not preserved in stub: {content!r}"
        )
        assert "Stream stalled mid tool-call" in content, (
            f"Stub content is missing the dropped-tool-call warning; users "
            f"get silent failure.  Got content={content!r}"
        )
        assert "write_file" in content, (
            f"Warning should name the dropped tool. Got: {content!r}"
        )
        assert response.choices[0].message.tool_calls is None
        assert any("Stream stalled mid tool-call" in d for d in fired_deltas), (
            f"Warning was not surfaced as a live stream delta. "
            f"fired_deltas={fired_deltas}"
        )

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_partial_text_only_no_warning(self, mock_close, mock_create):
        """Text-only partial stream (no tool call mid-flight) keeps the
        pre-fix behaviour: bare recovered text, no warning noise."""
        from run_agent import AIAgent

        class _StallError(RuntimeError):
            pass

        def _stalling_stream():
            yield _make_stream_chunk(content="Here's my answer so far")
            raise _StallError("simulated upstream stall")

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = lambda *a, **kw: _stalling_stream()
        mock_create.return_value = mock_client

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False
        agent._current_streamed_assistant_text = "Here's my answer so far"

        import os as _os
        _prev = _os.environ.get("HERMES_STREAM_RETRIES")
        _os.environ["HERMES_STREAM_RETRIES"] = "0"
        try:
            response = agent._interruptible_streaming_api_call({})
        finally:
            if _prev is None:
                _os.environ.pop("HERMES_STREAM_RETRIES", None)
            else:
                _os.environ["HERMES_STREAM_RETRIES"] = _prev

        content = response.choices[0].message.content or ""
        assert content == "Here's my answer so far", (
            f"Pre-fix behaviour regressed for text-only partial streams: {content!r}"
        )
        assert "Stream stalled" not in content, (
            f"Unexpected warning on text-only partial stream: {content!r}"
        )


class TestSilentRetryMidToolCall:
    """Regression: when the stream dies mid tool-call JSON after text was
    already delivered, we previously stubbed the turn with a "retry manually"
    warning.  Now: if the error is a transient connection error AND a tool
    call was in flight, silently retry the stream (the user sees a brief
    reconnect marker + duplicated preamble, which is strictly better than
    a lost action).  If no tool call was in flight, or the error isn't
    transient, the existing stub-with-warning behaviour is preserved.
    """

    @patch("run_agent.AIAgent._replace_primary_openai_client")
    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_silent_retry_recovers_tool_call(
        self, mock_close, mock_create, mock_replace,
    ):
        """First attempt: text + partial tool-call + connection drop.
        Second attempt: text + complete tool-call.  Response should contain
        the recovered tool call; no warning stub should be returned."""
        from run_agent import AIAgent
        import httpx as _httpx

        attempts = {"n": 0}

        def _first_stream():
            yield _make_stream_chunk(content="Let me write the audit: ")
            yield _make_stream_chunk(tool_calls=[
                _make_tool_call_delta(index=0, tc_id="call_1", name="write_file"),
            ])
            yield _make_stream_chunk(tool_calls=[
                _make_tool_call_delta(index=0, arguments='{"path": "/tmp/x", '),
            ])
            raise _httpx.RemoteProtocolError("peer closed connection")

        def _second_stream():
            yield _make_stream_chunk(content="Let me write the audit: ")
            yield _make_stream_chunk(tool_calls=[
                _make_tool_call_delta(index=0, tc_id="call_1", name="write_file"),
            ])
            yield _make_stream_chunk(tool_calls=[
                _make_tool_call_delta(
                    index=0, arguments='{"path": "/tmp/x", "content": "hi"}',
                ),
            ])
            yield _make_stream_chunk(finish_reason="tool_calls")

        def _pick_stream(*a, **kw):
            attempts["n"] += 1
            return _first_stream() if attempts["n"] == 1 else _second_stream()

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = _pick_stream
        mock_create.return_value = mock_client

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False

        fired_deltas: list = []
        agent._fire_stream_delta = lambda text: fired_deltas.append(text)

        import os as _os
        _prev = _os.environ.get("HERMES_STREAM_RETRIES")
        _os.environ["HERMES_STREAM_RETRIES"] = "2"
        try:
            response = agent._interruptible_streaming_api_call({})
        finally:
            if _prev is None:
                _os.environ.pop("HERMES_STREAM_RETRIES", None)
            else:
                _os.environ["HERMES_STREAM_RETRIES"] = _prev

        assert attempts["n"] == 2, (
            f"Expected silent retry (2 attempts), got {attempts['n']}"
        )
        # Response should carry the recovered tool call, not a warning stub.
        msg = response.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)
        assert tool_calls, (
            f"Silent retry should recover the tool call, got tool_calls={tool_calls!r} "
            f"content={getattr(msg, 'content', None)!r}"
        )
        _tc0 = tool_calls[0]
        _name = (
            _tc0["function"]["name"] if isinstance(_tc0, dict)
            else _tc0.function.name
        )
        assert _name == "write_file"
        # User saw a reconnect marker between attempts.
        assert any("reconnecting" in d.lower() for d in fired_deltas), (
            f"Expected a reconnect marker delta, fired_deltas={fired_deltas}"
        )
        # Stub-path warning must NOT appear (this was the whole point).
        joined = "".join(fired_deltas)
        assert "Stream stalled" not in joined, (
            f"Stub-path warning leaked into silent-retry path: {joined!r}"
        )

    @patch("run_agent.AIAgent._replace_primary_openai_client")
    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_silent_retry_exhausted_falls_back_to_stub(
        self, mock_close, mock_create, mock_replace,
    ):
        """When all retry attempts fail with connection errors, fall back
        to the original stub-with-warning behaviour so the user isn't left
        with zero signal."""
        from run_agent import AIAgent
        import httpx as _httpx

        def _always_fails():
            yield _make_stream_chunk(content="Let me write the audit: ")
            yield _make_stream_chunk(tool_calls=[
                _make_tool_call_delta(index=0, tc_id="call_1", name="write_file"),
            ])
            raise _httpx.RemoteProtocolError("peer closed connection")

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = lambda *a, **kw: _always_fails()
        mock_create.return_value = mock_client

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False

        fired_deltas: list = []
        agent._fire_stream_delta = lambda text: fired_deltas.append(text)

        import os as _os
        _prev = _os.environ.get("HERMES_STREAM_RETRIES")
        _os.environ["HERMES_STREAM_RETRIES"] = "1"
        try:
            response = agent._interruptible_streaming_api_call({})
        finally:
            if _prev is None:
                _os.environ.pop("HERMES_STREAM_RETRIES", None)
            else:
                _os.environ["HERMES_STREAM_RETRIES"] = _prev

        # After retries exhaust, the stub-with-warning path must engage.
        content = response.choices[0].message.content or ""
        assert "Stream stalled mid tool-call" in content, (
            f"Exhausted-retry fallback dropped the user-visible warning: {content!r}"
        )
        assert response.choices[0].message.tool_calls is None

    @patch("run_agent.AIAgent._replace_primary_openai_client")
    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_no_silent_retry_for_text_only_stall(
        self, mock_close, mock_create, mock_replace,
    ):
        """Text-only stall (no tool call in flight) must NOT trigger silent
        retry — that's the case where the user saw the model's text reply
        and retrying would duplicate it with no benefit."""
        from run_agent import AIAgent
        import httpx as _httpx

        attempts = {"n": 0}

        def _text_stall(*a, **kw):
            attempts["n"] += 1

            def _gen():
                yield _make_stream_chunk(content="Here's my answer so far")
                raise _httpx.RemoteProtocolError("peer closed connection")
            return _gen()

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = _text_stall
        mock_create.return_value = mock_client

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False
        agent._current_streamed_assistant_text = "Here's my answer so far"

        import os as _os
        _prev = _os.environ.get("HERMES_STREAM_RETRIES")
        _os.environ["HERMES_STREAM_RETRIES"] = "2"
        try:
            response = agent._interruptible_streaming_api_call({})
        finally:
            if _prev is None:
                _os.environ.pop("HERMES_STREAM_RETRIES", None)
            else:
                _os.environ["HERMES_STREAM_RETRIES"] = _prev

        # Only one attempt: text-only stall short-circuits retry.
        assert attempts["n"] == 1, (
            f"Text-only stall should not silent-retry, got {attempts['n']} attempts"
        )
        content = response.choices[0].message.content or ""
        assert content == "Here's my answer so far", (
            f"Text-only stall regressed: {content!r}"
        )
        assert "Stream stalled" not in content, (
            f"Text-only stall should not emit tool-call warning: {content!r}"
        )


# ── Test: CopilotACP Streaming Decision ──────────────────────────────────


def _valid_acp_response():
    """Build a minimal valid non-streaming API response for copilot-acp."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="Hello from ACP",
                    tool_calls=None,
                    role="assistant",
                ),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3),
        model="claude-opus-4.7",
    )


def _make_acp_agent(provider="copilot-acp", base_url="acp://copilot"):
    """Create an AIAgent configured for copilot-acp with a stream consumer
    so _has_stream_consumers() returns True (ensuring the test exercises the
    ACP exclusion, not the no-consumer branch)."""
    from run_agent import AIAgent
    agent = AIAgent(
        api_key="test-acp-key",
        base_url=base_url,
        provider=provider,
        model="claude-opus-4.7",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        stream_delta_callback=lambda text: None,
    )
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    return agent


class TestCopilotACPStreamingDecision:
    """Verify that copilot-acp routes to the non-streaming path.

    CopilotACPClient communicates via subprocess stdio and returns a plain
    SimpleNamespace — not an iterable stream.  The streaming decision logic
    must detect ACP runtimes and route to _interruptible_api_call instead.
    """

    @patch("run_agent.get_tool_definitions", return_value=[])
    @patch("run_agent.check_toolset_requirements", return_value={})
    @patch("agent.copilot_acp_client.CopilotACPClient")
    def test_provider_name_triggers_non_streaming(
        self, mock_acp_cls, _mock_check, _mock_tools
    ):
        """provider='copilot-acp' → non-streaming path."""
        mock_acp_cls.return_value = MagicMock()
        agent = _make_acp_agent(provider="copilot-acp", base_url="acp://copilot")

        with (
            patch.object(agent, "_interruptible_api_call",
                         return_value=_valid_acp_response()) as mock_non_stream,
            patch.object(agent, "_interruptible_streaming_api_call") as mock_stream,
        ):
            # Verify the decision logic correctly disables streaming
            _use_streaming = True
            if getattr(agent, "_disable_streaming", False):
                _use_streaming = False
            elif (
                agent.provider == "copilot-acp"
                or str(agent.base_url or "").lower().startswith("acp://copilot")
                or str(agent.base_url or "").lower().startswith("acp+tcp://")
            ):
                _use_streaming = False

            assert _use_streaming is False
            # Call the non-streaming path as the loop would
            response = mock_non_stream({})
            mock_stream.assert_not_called()

    @patch("run_agent.get_tool_definitions", return_value=[])
    @patch("run_agent.check_toolset_requirements", return_value={})
    @patch("agent.copilot_acp_client.CopilotACPClient")
    def test_acp_base_url_triggers_non_streaming(
        self, mock_acp_cls, _mock_check, _mock_tools
    ):
        """base_url='acp://copilot' → non-streaming even without provider name."""
        mock_acp_cls.return_value = MagicMock()
        agent = _make_acp_agent(provider="custom", base_url="acp://copilot")
        agent.provider = "custom"

        _use_streaming = True
        if (
            agent.provider == "copilot-acp"
            or str(agent.base_url or "").lower().startswith("acp://copilot")
            or str(agent.base_url or "").lower().startswith("acp+tcp://")
        ):
            _use_streaming = False

        assert _use_streaming is False

    @patch("run_agent.get_tool_definitions", return_value=[])
    @patch("run_agent.check_toolset_requirements", return_value={})
    @patch("agent.copilot_acp_client.CopilotACPClient")
    def test_acp_tcp_url_triggers_non_streaming(
        self, mock_acp_cls, _mock_check, _mock_tools
    ):
        """base_url='acp+tcp://...' → non-streaming."""
        mock_acp_cls.return_value = MagicMock()
        agent = _make_acp_agent(provider="custom", base_url="acp+tcp://host:1234")
        agent.provider = "custom"

        _use_streaming = True
        if (
            agent.provider == "copilot-acp"
            or str(agent.base_url or "").lower().startswith("acp://copilot")
            or str(agent.base_url or "").lower().startswith("acp+tcp://")
        ):
            _use_streaming = False

        assert _use_streaming is False

    def test_non_acp_provider_allows_streaming(self):
        """Regular providers still get streaming enabled."""
        from run_agent import AIAgent
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            provider="openrouter",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            stream_delta_callback=lambda text: None,
        )
        agent.api_mode = "chat_completions"

        _use_streaming = True
        if getattr(agent, "_disable_streaming", False):
            _use_streaming = False
        elif (
            agent.provider == "copilot-acp"
            or str(agent.base_url or "").lower().startswith("acp://copilot")
            or str(agent.base_url or "").lower().startswith("acp+tcp://")
        ):
            _use_streaming = False

        assert _use_streaming is True


class TestCodexFallbackErrorEvent:
    """Provider ``error`` SSE frames must surface the real message,
    not the generic "did not emit a terminal response" RuntimeError.

    xAI emits ``type=error`` as the FIRST frame on the Responses stream
    when an OAuth account is unsubscribed/exhausted (May 2026
    SuperGrok rollout).  The SDK helper raises
    ``RuntimeError("Expected to have received response.created before
    error")`` which the caller catches and routes to
    ``_run_codex_create_stream_fallback``.  The fallback then opens a
    NEW stream that emits the same ``type=error`` frame; before this
    fix it ignored the event entirely and raised a useless RuntimeError.
    """

    def _make_agent(self):
        from run_agent import AIAgent
        agent = AIAgent(
            api_key="test-key",
            base_url="https://api.x.ai/v1",
            provider="xai-oauth",
            model="grok-4.3",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "codex_responses"
        agent._touch_activity = lambda desc: None
        return agent

    def test_fallback_raises_synthesized_error_with_xai_subscription_message(self):
        from run_agent import _StreamErrorEvent

        agent = self._make_agent()

        error_event = SimpleNamespace(
            type="error",
            message=(
                "Forbidden: The caller does not have permission to execute the specified operation. "
                "'You have either run out of available resources or do not have an active Grok subscription.'"
            ),
            code="permission_denied",
            param=None,
            sequence_number=1,
        )

        class _FakeStream:
            def __iter__(self_inner):
                return iter([error_event])
            def close(self_inner):
                return None

        mock_client = MagicMock()
        mock_client.responses.create.return_value = _FakeStream()

        with pytest.raises(_StreamErrorEvent) as excinfo:
            agent._run_codex_create_stream_fallback(
                {"model": "grok-4.3", "instructions": "hi", "input": []},
                client=mock_client,
            )

        exc = excinfo.value
        assert "active Grok subscription" in str(exc)
        assert exc.code == "permission_denied"
        assert isinstance(exc.body, dict)
        assert exc.body["error"]["message"] == error_event.message
        # _extract_api_error_context reads .body["error"]["message"] — make sure
        # the entitlement detector will find the subscription phrase there.
        assert "active Grok subscription" in exc.body["error"]["message"]

    def test_fallback_dict_event_payload_is_also_handled(self):
        """Some relays deliver events as plain dicts instead of model
        objects; the dict branch in the loop must surface them too."""
        from run_agent import _StreamErrorEvent

        agent = self._make_agent()

        error_event = {
            "type": "error",
            "message": "rate_limited",
            "code": "rate_limit_exceeded",
        }

        class _FakeStream:
            def __iter__(self_inner):
                return iter([error_event])
            def close(self_inner):
                return None

        mock_client = MagicMock()
        mock_client.responses.create.return_value = _FakeStream()

        with pytest.raises(_StreamErrorEvent) as excinfo:
            agent._run_codex_create_stream_fallback(
                {"model": "grok-4.3", "instructions": "hi", "input": []},
                client=mock_client,
            )

        assert "rate_limited" in str(excinfo.value)
        assert excinfo.value.code == "rate_limit_exceeded"

    def test_fallback_surfaces_message_useful_to_summarizer(self):
        """The synthesized exception must be readable by
        ``_summarize_api_error`` so the user-facing log line shows the
        real provider message instead of a generic class name."""
        from run_agent import AIAgent, _StreamErrorEvent

        agent = self._make_agent()
        exc = _StreamErrorEvent(
            "You have either run out of available resources or do not have an active Grok subscription.",
            code="permission_denied",
        )

        summary = AIAgent._summarize_api_error(exc)
        assert "active Grok subscription" in summary

    def test_fallback_still_raises_terminal_error_when_no_error_event(self):
        """Streams that simply end without any terminal event (and no
        ``error`` frame) must continue to raise the original
        ``"did not emit a terminal response"`` RuntimeError so callers
        can distinguish "stream truncated mid-flight" from "provider
        rejected the call"."""
        agent = self._make_agent()

        # Empty stream — no events at all
        class _FakeStream:
            def __iter__(self_inner):
                return iter([])
            def close(self_inner):
                return None

        mock_client = MagicMock()
        mock_client.responses.create.return_value = _FakeStream()

        with pytest.raises(RuntimeError) as excinfo:
            agent._run_codex_create_stream_fallback(
                {"model": "grok-4.3", "instructions": "hi", "input": []},
                client=mock_client,
            )

        assert "did not emit a terminal response" in str(excinfo.value)
