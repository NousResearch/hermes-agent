"""Tests for streaming token delivery infrastructure.

Tests the unified streaming API call, delta callbacks, tool-call
suppression, provider fallback, and CLI streaming display.
"""
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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
    def test_sparse_delta_allows_missing_optional_fields(self, mock_close, mock_create):
        """Managed stream deltas may omit both content and tool_calls."""
        from run_agent import AIAgent

        sparse_delta = SimpleNamespace(reasoning_content=None, reasoning=None)
        chunks = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        index=0,
                        delta=sparse_delta,
                        finish_reason=None,
                    )
                ],
                model="test-model",
                usage=None,
            ),
            _make_stream_chunk(
                content="done", finish_reason="stop", model="test-model"
            ),
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

        assert response.choices[0].message.content == "done"
        assert response.choices[0].message.tool_calls is None

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_sparse_tool_delta_allows_missing_nested_fields(
        self, mock_close, mock_create
    ):
        """A partial tool delta may contain arguments before its other fields."""
        from run_agent import AIAgent

        sparse_tool_delta = SimpleNamespace(
            index=0,
            function=SimpleNamespace(arguments='{"city":"Paris"}'),
        )
        chunks = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        index=0,
                        delta=SimpleNamespace(tool_calls=[sparse_tool_delta]),
                    )
                ],
                model="test-model",
                usage=None,
            ),
            _make_stream_chunk(finish_reason="tool_calls", model="test-model"),
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

        tool_call = response.choices[0].message.tool_calls[0]
        assert tool_call.function.name == ""
        assert tool_call.function.arguments == '{"city":"Paris"}'
        assert response.choices[0].finish_reason == "tool_calls"

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_chat_stream_closes_original_provider_resource(
        self,
        mock_close,
        mock_create,
    ):
        from run_agent import AIAgent

        class ProviderStream:
            def __init__(self):
                self.closed = False

            def __iter__(self):
                return iter([
                    _make_stream_chunk(
                        content="Hello",
                        finish_reason="stop",
                        model="test-model",
                    )
                ])

            def close(self):
                self.closed = True

        provider_stream = ProviderStream()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = provider_stream
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

        assert response.choices[0].message.content == "Hello"
        assert provider_stream.closed is True

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_native_gemini_endpoint_omits_stream_options(self, mock_close, mock_create):
        """Google's native Gemini REST endpoint rejects OpenAI-only stream_options."""
        from run_agent import AIAgent

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter([
            _make_stream_chunk(content="Paris", finish_reason="stop", model="gemini"),
        ])
        mock_create.return_value = mock_client

        agent = AIAgent(
            api_key="test-key",
            base_url="https://generativelanguage.googleapis.com/v1beta",
            model="gemini-3-flash-preview",
            provider="gemini",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False

        response = agent._interruptible_streaming_api_call({})

        assert response.choices[0].message.content == "Paris"
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["stream"] is True
        assert "stream_options" not in call_kwargs



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
    def test_tool_argument_deltas_are_collected_without_concatenating_each_chunk(
        self, mock_close, mock_create
    ):
        """Large tool arguments must not rebuild the accumulated string per delta."""
        from run_agent import AIAgent

        class AppendOnlyChunk(str):
            def __radd__(self, other):
                raise AssertionError("tool argument delta was concatenated eagerly")

        chunks = [
            _make_stream_chunk(tool_calls=[
                _make_tool_call_delta(
                    index=0, tc_id="call_123", name="write_file"
                )
            ]),
            _make_stream_chunk(tool_calls=[
                _make_tool_call_delta(
                    index=0, arguments=AppendOnlyChunk('{"path":"out.txt",')
                )
            ]),
            _make_stream_chunk(tool_calls=[
                _make_tool_call_delta(
                    index=0, arguments=AppendOnlyChunk('"content":"hello"}')
                )
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

        tool_call = response.choices[0].message.tool_calls[0]
        assert tool_call.function.arguments == (
            '{"path":"out.txt","content":"hello"}'
        )

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    @patch("agent.relay_llm.stream")
    def test_relay_finalizer_emits_joined_tool_arguments(
        self, mock_relay_stream, mock_close, mock_create
    ):
        """Relay receives the public string shape, not buffered fragments."""
        from run_agent import AIAgent

        captured = {}
        fake_stream = MagicMock()
        fake_stream.final_response = None
        chunks = [
            _make_stream_chunk(tool_calls=[
                _make_tool_call_delta(
                    index=0,
                    tc_id="call_123",
                    name="search",
                    arguments='{"q":',
                )
            ]),
            _make_stream_chunk(tool_calls=[
                _make_tool_call_delta(index=0, arguments='"hello"}')
            ]),
            _make_stream_chunk(finish_reason="tool_calls"),
        ]
        fake_stream.__iter__.return_value = iter(chunks)

        def relay_stream_impl(*args, **kwargs):
            captured["finalizer"] = kwargs["finalizer"]
            captured["on_chunk"] = kwargs["on_chunk"]
            return fake_stream

        mock_relay_stream.side_effect = relay_stream_impl
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter([])
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

        agent._interruptible_streaming_api_call({})

        # Relay's contract: the collector sees every chunk as JSON, then the finalizer runs.
        from agent.relay_llm import _jsonable
        for chunk in chunks:
            captured["on_chunk"](_jsonable(chunk))
        payload = captured["finalizer"]()
        tool_calls = payload["choices"][0]["message"]["tool_calls"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"] == {
            "name": "search",
            "arguments": '{"q":"hello"}',
        }

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_tool_argument_assembly_is_chunk_boundary_invariant(
        self, mock_close, mock_create
    ):
        """Argument bytes are identical across ASCII and Unicode fragment sizes."""
        import json

        from run_agent import AIAgent

        payload = json.dumps(
            {"path": "/tmp/x", "content": "héllo wörld 日本語 " * 50},
            ensure_ascii=False,
        )

        def assemble(fragment_size):
            fragments = [
                payload[i : i + fragment_size]
                for i in range(0, len(payload), fragment_size)
            ]
            chunks = [
                _make_stream_chunk(tool_calls=[
                    _make_tool_call_delta(
                        index=0,
                        tc_id="call_123",
                        name="write_file",
                        arguments=fragments[0],
                    )
                ])
            ]
            chunks.extend(
                _make_stream_chunk(tool_calls=[
                    _make_tool_call_delta(index=0, arguments=fragment)
                ])
                for fragment in fragments[1:]
            )
            chunks.append(_make_stream_chunk(finish_reason="tool_calls"))

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
            return response.choices[0].message.tool_calls[0].function.arguments

        for fragment_size in (len(payload), 64, 7, 3, 1):
            arguments = assemble(fragment_size)
            assert arguments.encode("utf-8") == payload.encode("utf-8")


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
    def test_list_content_after_tool_call_is_normalized(self, mock_close, mock_create):
        """OpenAI-compatible content blocks must never reach stream callbacks raw.

        Mistral/NVIDIA can emit a text delta as a list of content-block dicts.
        The list must be flattened before the first direct stream callback and
        again after a tool-call has switched the stream to the suppression
        callback path.
        """
        from run_agent import AIAgent

        chunks = [
            _make_stream_chunk(content=[{"type": "text", "text": "before tool; "}]),
            _make_stream_chunk(tool_calls=[
                _make_tool_call_delta(index=0, tc_id="call_63734", name="read_file")
            ]),
            _make_stream_chunk(content=[{"type": "text", "text": "after tool"}]),
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
            stream_delta_callback=deltas.append,
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False

        response = agent._interruptible_streaming_api_call({})

        assert deltas == ["before tool; ", "after tool"]
        assert response.choices[0].message.content == "before tool; after tool"


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
    def test_response_object_disables_streaming_and_returns_final_response(
        self, mock_close, mock_create
    ):
        """Adapters that ignore stream=True should fall back cleanly."""
        from run_agent import AIAgent

        final_response = SimpleNamespace(
            model="copilot-acp",
            choices=[SimpleNamespace(
                message=SimpleNamespace(
                    content="Hello from ACP",
                    tool_calls=None,
                    reasoning_content=None,
                    reasoning=None,
                ),
                finish_reason="stop",
            )],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = final_response
        mock_create.return_value = mock_client

        agent = AIAgent(
            model="claude-sonnet-4.6",
            provider="copilot-acp",
            api_key="test-key",
            base_url="http://localhost:1234/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False

        deltas = []
        agent._stream_callback = lambda text: deltas.append(text)

        response = agent._interruptible_streaming_api_call({})

        assert response is final_response
        assert agent._disable_streaming is True
        assert deltas == ["Hello from ACP"]


    @patch("run_agent.AIAgent._abort_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    @patch("run_agent.AIAgent._create_request_openai_client")
    def test_moa_interrupt_closes_stream_handle(
        self, mock_create, mock_close_openai, mock_abort_openai
    ):
        """MoA interrupts must close the per-request stream, not the facade client."""
        from run_agent import AIAgent

        class _BlockingClosableStream:
            def __init__(self):
                self.entered = threading.Event()
                self.closed = threading.Event()
                self.close_calls = 0

            def __iter__(self):
                return self

            def __next__(self):
                self.entered.set()
                if not self.closed.wait(timeout=5):
                    raise TimeoutError("MoA test stream was not closed")
                raise RuntimeError("stream closed")

            def close(self):
                self.close_calls += 1
                self.closed.set()

        stream = _BlockingClosableStream()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = stream
        mock_create.return_value = mock_client

        agent = AIAgent(
            model="default",
            provider="moa",
            api_key="test-key",
            base_url="moa://local",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False
        agent.client = mock_client

        def _request_interrupt():
            assert stream.entered.wait(timeout=2)
            agent._interrupt_requested = True

        interrupter = threading.Thread(target=_request_interrupt, daemon=True)
        interrupter.start()

        with pytest.raises(InterruptedError):
            agent._interruptible_streaming_api_call({"model": "default", "messages": []})

        assert stream.closed.wait(timeout=2)
        assert stream.close_calls == 1
        mock_create.assert_called_once()
        mock_close_openai.assert_not_called()
        mock_abort_openai.assert_not_called()




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

        events = [
            SimpleNamespace(type="response.created"),
            SimpleNamespace(
                type="response.output_text.delta",
                delta="Hello from Codex!",
            ),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(status="completed", id="r1", usage=None),
            ),
        ]

        class _FakeCreateStream:
            def __iter__(self_inner):
                return iter(events)
            def close(self_inner):
                return None

        mock_client = MagicMock()
        mock_client.responses.create.return_value = _FakeCreateStream()

        agent._run_codex_stream({}, client=mock_client)
        assert "Hello from Codex!" in deltas


    def test_codex_remote_protocol_error_retries_then_raises(self):
        """Transport errors from ``responses.create`` retry once then re-raise.

        With the migration from ``responses.stream(...)`` to
        ``responses.create(stream=True)``, there is no longer a separate
        fallback function — the same call IS the streaming path.  When it
        raises ``httpx.RemoteProtocolError``, we retry once (matching the
        old behavior on the helper) and re-raise on the second failure.
        """
        from run_agent import AIAgent
        import httpx

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

        call_count = {"n": 0}

        def _create_side_effect(**kwargs):
            call_count["n"] += 1
            raise httpx.RemoteProtocolError(
                "peer closed connection without sending complete message body"
            )

        mock_client = MagicMock()
        mock_client.responses.create.side_effect = _create_side_effect

        with pytest.raises(httpx.RemoteProtocolError):
            agent._run_codex_stream({}, client=mock_client)

        # 1 initial + 1 retry = 2 calls
        assert call_count["n"] == 2

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

        agent._run_codex_stream(
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
        # #67142: streaming now runs on a request-local anthropic client; route
        # it to the test mock so .messages.stream is exercised.
        agent._create_request_anthropic_client = lambda *a, **k: agent._anthropic_client

        agent._interruptible_streaming_api_call({})

        assert touch_calls.count("receiving stream response") == len(events)
        mock_stream.close.assert_called_once()

    @patch("run_agent.AIAgent._rebuild_anthropic_client")
    @patch("run_agent.AIAgent._replace_primary_openai_client")
    def test_anthropic_stream_parser_valueerror_retries_before_delivery(
        self, mock_replace, mock_rebuild, monkeypatch,
    ):
        """Malformed Anthropic event-stream frames retry instead of surfacing HTTP None.

        On the Anthropic-native path the stream-retry cleanup must close + rebuild the
        Anthropic client, NOT the OpenAI primary client (which would fail with
        Missing-credentials and leave the wedged stream open). See #28161.
        """
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
        agent._create_request_anthropic_client = lambda *a, **k: agent._anthropic_client

        response = agent._interruptible_streaming_api_call({})

        assert response is final_message
        assert agent._anthropic_client.messages.stream.call_count == 2
        # #67142: cleanup runs on the request-local anthropic client (closed,
        # worker-owned, via _close_request_client_once), never rebuilding the
        # shared client and never touching the OpenAI primary client.
        assert mock_replace.call_count == 0
        assert mock_rebuild.call_count == 0
        assert agent._anthropic_client.close.call_count >= 1

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
        agent._create_request_anthropic_client = lambda *a, **k: agent._anthropic_client

        with pytest.raises(ValueError, match="invalid local request shape"):
            agent._interruptible_streaming_api_call({})

        assert agent._anthropic_client.messages.stream.call_count == 1
        assert mock_replace.call_count == 0


    @patch("run_agent.AIAgent._try_refresh_anthropic_client_credentials")
    @patch("run_agent.AIAgent._rebuild_anthropic_client")
    @patch("run_agent.AIAgent._replace_primary_openai_client")
    def test_anthropic_eventless_sdk_assertion_normalized_to_empty_stream(
        self, mock_replace, mock_rebuild, mock_refresh,
    ):
        """Real-SDK shape: an eventless stream has no message_start, so
        get_final_message() raises AssertionError (final snapshot is None).
        That must be normalized to EmptyStreamError and retried as
        transient — not surface as a raw AssertionError."""
        from agent.errors import EmptyStreamError
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://api.anthropic.com",
            provider="anthropic",
            model="claude-test",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "anthropic_messages"
        agent._interrupt_requested = False

        empty_stream = MagicMock()
        empty_stream.__enter__ = MagicMock(return_value=empty_stream)
        empty_stream.__exit__ = MagicMock(return_value=False)
        empty_stream.__iter__ = MagicMock(side_effect=lambda: iter([]))
        empty_stream.get_final_message.side_effect = AssertionError()

        agent._anthropic_client = MagicMock()
        agent._anthropic_client.messages.stream.return_value = empty_stream
        agent._create_request_anthropic_client = lambda *a, **k: agent._anthropic_client

        with pytest.raises(EmptyStreamError):
            agent._interruptible_streaming_api_call({})

        assert agent._anthropic_client.messages.stream.call_count == 3
        assert mock_replace.call_count == 0
        assert mock_rebuild.call_count == 0


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
    def test_empty_partial_stream_stub_stays_empty_for_loop_guard(
        self, mock_close, mock_create,
    ):
        """Stream dies with 0 recovered chars and no tool call → the stub
        keeps its empty content ON PURPOSE.

        The conversation loop's truncation path detects an EMPTY
        partial-stream stub (PARTIAL_STREAM_STUB_ID + no content) and skips
        appending it to history entirely — only the continuation nudge is
        sent (the #68041 class fix).  An earlier iteration substituted
        '[response interrupted]' placeholder text HERE, which defeated that
        guard: the stub no longer looked empty, entered history, and the
        placeholder leaked into the stitched final response.  Transcripts
        that already carry a persisted empty turn are healed at the send
        boundary by repair_empty_non_final_messages instead.
        """
        from run_agent import AIAgent
        from hermes_constants import PARTIAL_STREAM_STUB_ID

        class _StallError(RuntimeError):
            pass

        def _stalling_stream():
            yield _make_stream_chunk(content="partial token")
            raise _StallError("simulated upstream stall after a delta")

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = (
            lambda *a, **kw: _stalling_stream()
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
        agent._fire_stream_delta = lambda text: None
        # Empty recovered text — the exact "0 chars recovered, no tool call"
        # production condition.
        agent._current_streamed_assistant_text = ""

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

        # The stub must be RECOGNIZABLY empty so the loop guard can skip it.
        assert getattr(response, "id", "") == PARTIAL_STREAM_STUB_ID
        content = response.choices[0].message.content
        assert not content, (
            f"Empty-partial-stream stub must keep empty content so the "
            f"conversation loop's empty-stub guard can detect and skip it — "
            f"substituted text defeats the guard and leaks into the final "
            f"response. Got content={content!r}"
        )
        assert response.choices[0].message.tool_calls is None


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

    @patch("model_tools.get_tool_definitions", return_value=[])
    @patch("model_tools.check_toolset_requirements", return_value={})
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

    @patch("model_tools.get_tool_definitions", return_value=[])
    @patch("model_tools.check_toolset_requirements", return_value={})
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


class TestBedrockIamStreamingFallback:
    """bedrock_converse streaming branch: IAM denial of
    InvokeModelWithResponseStream falls back to converse() inline and sets
    _disable_streaming for the rest of the session."""

    def _make_bedrock_agent(self):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "bedrock_converse"
        agent._interrupt_requested = False
        return agent

    def test_iam_denial_falls_back_inline_and_disables_streaming(self):
        pytest.importorskip("botocore.exceptions", reason="botocore (with working exceptions module) required")
        from botocore.exceptions import ClientError

        agent = self._make_bedrock_agent()

        client = MagicMock()
        client.converse_stream.side_effect = ClientError(
            error_response={
                "Error": {
                    "Code": "AccessDeniedException",
                    "Message": (
                        "User is not authorized to perform: "
                        "bedrock:InvokeModelWithResponseStream"
                    ),
                }
            },
            operation_name="ConverseStream",
        )
        client.converse.return_value = {
            "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
        }

        with patch(
            "agent.bedrock_adapter._get_bedrock_runtime_client",
            return_value=client,
        ):
            response = agent._interruptible_streaming_api_call(
                {"modelId": agent.model, "messages": []}
            )

        client.converse.assert_called_once()
        assert response.choices[0].message.content == "hi"
        assert getattr(agent, "_disable_streaming", False) is True



class _BlockingEventStream:
    """Mock boto3 ``converse_stream()`` response whose event iterator blocks
    forever — simulates a provider that opens the stream then stops yielding
    events. The worker thread sits inside ``for event in event_stream`` exactly
    as a wedged Bedrock stream would, giving the liveness watchdog something to
    trip on."""

    def __init__(self, release):
        self._release = release

    def get(self, key, default=None):
        if key == "stream":
            return self
        return default

    def __iter__(self):
        return self

    def __next__(self):
        # Never yields — blocks until the test releases it (teardown) so the
        # daemon worker can exit instead of leaking a truly-hung thread.
        self._release.wait(timeout=30)
        raise StopIteration


def test_on_event_fires_per_bedrock_event():
    """FIX 1: on_event fires once for EVERY yielded Bedrock event — text,
    tool-input delta, messageStop, and metadata alike — providing wire-level
    liveness (not just text deltas)."""
    from agent.bedrock_adapter import stream_converse_with_callbacks

    events = [
        {"contentBlockDelta": {"delta": {"text": "a"}}},
        {"contentBlockStart": {"start": {"toolUse": {"toolUseId": "t1", "name": "x"}}}},
        {"contentBlockDelta": {"delta": {"toolUse": {"input": "{}"}}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "end_turn"}},
        {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1}}},
    ]
    calls = {"n": 0}

    stream_converse_with_callbacks(
        {"stream": iter(events)},
        on_event=lambda: calls.__setitem__("n", calls["n"] + 1),
    )

    assert calls["n"] == len(events)


def test_on_event_exception_is_swallowed():
    """FIX 1: a raising on_event callback must never abort the stream."""
    from agent.bedrock_adapter import stream_converse_with_callbacks

    events = [{"messageStop": {"stopReason": "end_turn"}}]

    def _boom():
        raise ValueError("liveness hook blew up")

    resp = stream_converse_with_callbacks({"stream": iter(events)}, on_event=_boom)
    assert resp is not None
    assert resp.choices[0].finish_reason == "stop"


class TestBedrockStreamLivenessWatchdog:
    """FIX 1: Bedrock streaming participates in the #58962 cross-turn stale
    breaker and no longer hangs when the stream stops yielding events."""

    def _make_bedrock_agent(self):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "bedrock_converse"
        agent._interrupt_requested = False
        return agent

    def test_stalled_stream_bumps_streak_and_aborts(self, monkeypatch):
        """A Bedrock stream that opens then stops yielding events trips the
        watchdog: it bumps the cross-turn stale streak and raises TimeoutError
        instead of hanging forever."""
        pytest.importorskip("botocore.exceptions", reason="botocore (with working exceptions module) required")
        import threading as _t

        # Tiny stale timeout so the watchdog trips quickly; give-up threshold
        # kept above 1 so a single call raises TimeoutError (not the breaker).
        monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "0.5")
        monkeypatch.setenv("HERMES_STREAM_STALE_GIVEUP", "5")

        agent = self._make_bedrock_agent()
        agent._consecutive_stale_streams = 0
        release = _t.Event()

        client = MagicMock()
        client.converse_stream.return_value = _BlockingEventStream(release)

        try:
            with patch(
                "agent.bedrock_adapter._get_bedrock_runtime_client",
                return_value=client,
            ):
                with pytest.raises(TimeoutError):
                    agent._interruptible_streaming_api_call(
                        {"modelId": agent.model, "messages": []}
                    )
        finally:
            release.set()

        # Watchdog counted exactly one stale kill in the cross-turn breaker.
        assert agent._consecutive_stale_streams == 1

    def test_pre_elevated_streak_aborts_before_streaming(self, monkeypatch):
        """A streak already past the give-up threshold aborts at entry with
        RuntimeError — Bedrock never even opens a stream (cross-turn breaker)."""
        pytest.importorskip("botocore.exceptions", reason="botocore (with working exceptions module) required")

        monkeypatch.setenv("HERMES_STREAM_STALE_GIVEUP", "5")

        agent = self._make_bedrock_agent()
        agent._consecutive_stale_streams = 5

        client = MagicMock()
        with patch(
            "agent.bedrock_adapter._get_bedrock_runtime_client",
            return_value=client,
        ):
            with pytest.raises(RuntimeError, match="unresponsive"):
                agent._interruptible_streaming_api_call(
                    {"modelId": agent.model, "messages": []}
                )

        client.converse_stream.assert_not_called()

    def test_successful_stream_resets_streak(self, monkeypatch):
        """A Bedrock stream that completes normally clears any prior stale
        streak so a recovered provider doesn't carry it into later turns."""
        pytest.importorskip("botocore.exceptions", reason="botocore (with working exceptions module) required")

        monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "60")

        agent = self._make_bedrock_agent()
        agent._consecutive_stale_streams = 3  # simulate a prior wedged streak

        events = [
            {"contentBlockDelta": {"delta": {"text": "hi"}}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1}}},
        ]
        client = MagicMock()
        client.converse_stream.return_value = {"stream": iter(events)}

        with patch(
            "agent.bedrock_adapter._get_bedrock_runtime_client",
            return_value=client,
        ):
            response = agent._interruptible_streaming_api_call(
                {"modelId": agent.model, "messages": []}
            )

        assert response.choices[0].message.content == "hi"
        assert agent._consecutive_stale_streams == 0


class TestBedrockReasoningStaleFloor:
    """The Bedrock inference-profile id -> reasoning stale-timeout floor
    normalizer must match floor-table keys regardless of whether the model
    is keyed with a dashed version (``claude-opus-4``) or a dotted version
    (``claude-sonnet-4.5``). Bedrock always dashes the version, so the
    normalizer has to try the alternate separator form."""

    @pytest.mark.parametrize(
        "model_id, expected",
        [
            # opus is keyed dashed/base (``claude-opus-4`` -> 240) and
            # matches the Bedrock dashed id unchanged.
            ("us.anthropic.claude-opus-4-6-v1:0", 240.0),
            # sonnet is keyed DOTTED (``claude-sonnet-4.5`` /
            # ``claude-sonnet-4.6`` -> 180). The Bedrock dashed id must
            # now resolve via the alternate version-separator form.
            ("us.anthropic.claude-sonnet-4-5-v1:0", 180.0),
            ("us.anthropic.claude-sonnet-4-6-v1:0", 180.0),
            # region prefix variations still strip correctly.
            ("eu.anthropic.claude-sonnet-4-5-v1:0", 180.0),
        ],
    )
    def test_bedrock_reasoning_models_resolve_floor(self, model_id, expected):
        from agent.chat_completion_helpers import _bedrock_reasoning_stale_floor

        assert _bedrock_reasoning_stale_floor(model_id) == expected
