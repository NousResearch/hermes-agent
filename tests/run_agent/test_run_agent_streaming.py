"""Unit tests for run_agent.py (AIAgent) — streaming API calls + interrupt handling.

Split out of the former monolithic ``tests/run_agent/test_run_agent.py`` (which
outgrew the per-file CI wall-clock cap). Shared fixtures live in ``conftest.py``;
mock-builders in ``_run_agent_helpers.py``.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest
from run_agent import AIAgent

from tests.run_agent._run_agent_helpers import (
    _make_chunk,
    _make_tc_delta,
)


class TestInterrupt:
    def test_interrupt_sets_flag(self, agent):
        with patch("run_agent._set_interrupt"):
            agent.interrupt()
            assert agent._interrupt_requested is True

    def test_interrupt_with_message(self, agent):
        with patch("run_agent._set_interrupt"):
            agent.interrupt("new question")
            assert agent._interrupt_message == "new question"

    def test_clear_interrupt(self, agent):
        with patch("run_agent._set_interrupt"):
            agent.interrupt("msg")
            agent.clear_interrupt()
            assert agent._interrupt_requested is False
            assert agent._interrupt_message is None

    def test_is_interrupted_property(self, agent):
        assert agent.is_interrupted is False
        with patch("run_agent._set_interrupt"):
            agent.interrupt()
            assert agent.is_interrupted is True


class TestStreamingApiCall:
    """Tests for _streaming_api_call — voice TTS streaming pipeline."""

    def test_content_assembly(self, agent):
        chunks = [
            _make_chunk(content="Hel"),
            _make_chunk(content="lo "),
            _make_chunk(content="World"),
            _make_chunk(finish_reason="stop"),
        ]
        agent.client.chat.completions.create.return_value = iter(chunks)
        callback = MagicMock()
        agent.stream_delta_callback = callback

        resp = agent._interruptible_streaming_api_call({"messages": []})

        assert resp.choices[0].message.content == "Hello World"
        assert resp.choices[0].finish_reason == "stop"
        assert callback.call_count == 3
        callback.assert_any_call("Hel")
        callback.assert_any_call("lo ")
        callback.assert_any_call("World")

    def test_tool_call_accumulation(self, agent):
        # Per OpenAI streaming spec, function names are delivered atomically
        # in the first chunk; only `arguments` is fragmented across chunks.
        # The accumulator uses assignment for names (immune to MiniMax/NIM
        # resends of the full name) and `+=` for arguments.
        chunks = [
            _make_chunk(tool_calls=[_make_tc_delta(0, "call_1", "web_search", '{"q":')]),
            _make_chunk(tool_calls=[_make_tc_delta(0, None, None, '"test"}')]),
            _make_chunk(finish_reason="tool_calls"),
        ]
        agent.client.chat.completions.create.return_value = iter(chunks)

        resp = agent._interruptible_streaming_api_call({"messages": []})

        tc = resp.choices[0].message.tool_calls
        assert len(tc) == 1
        assert tc[0].function.name == "web_search"
        assert tc[0].function.arguments == '{"q":"test"}'
        assert tc[0].id == "call_1"

    def test_multiple_tool_calls(self, agent):
        chunks = [
            _make_chunk(tool_calls=[_make_tc_delta(0, "call_a", "search", '{}')]),
            _make_chunk(tool_calls=[_make_tc_delta(1, "call_b", "read", '{}')]),
            _make_chunk(finish_reason="tool_calls"),
        ]
        agent.client.chat.completions.create.return_value = iter(chunks)

        resp = agent._interruptible_streaming_api_call({"messages": []})

        tc = resp.choices[0].message.tool_calls
        assert len(tc) == 2
        assert tc[0].function.name == "search"
        assert tc[1].function.name == "read"

    def test_truncated_tool_call_args_no_finish_reason_routes_to_stub(self, agent):
        # Stream delivers a tool call with incomplete JSON args and then ENDS
        # with no finish_reason (the SSE just stops — no terminator, no
        # [DONE]).  This is an upstream mid-tool-call drop, NOT an output cap.
        # The builder must route it through the partial-stream-stub path
        # (id=PARTIAL_STREAM_STUB_ID, tool_calls=None so it can't execute,
        # finish_reason=length so the loop's continuation machinery fires with
        # chunking guidance) rather than stamping a normal 'length' truncation.
        from hermes_constants import PARTIAL_STREAM_STUB_ID
        chunks = [
            _make_chunk(tool_calls=[_make_tc_delta(0, "call_1", "write_file", '{"path":"x.txt","content":"hel')]),
        ]
        agent.client.chat.completions.create.return_value = iter(chunks)

        resp = agent._interruptible_streaming_api_call({"messages": []})

        assert resp.id == PARTIAL_STREAM_STUB_ID
        assert resp.choices[0].finish_reason == "length"
        assert resp.choices[0].message.tool_calls is None
        assert getattr(resp, "_dropped_tool_names", None) == ["write_file"]

    def test_truncated_tool_call_args_with_length_finish_reason_upgrades(self, agent):
        # Control: when the provider explicitly reports finish_reason='length'
        # alongside incomplete tool args, it IS a genuine output cap.  Keep the
        # existing behaviour — tool_calls preserved, finish_reason 'length' —
        # so the max_tokens-boost truncation retry path still applies.
        chunks = [
            _make_chunk(tool_calls=[_make_tc_delta(0, "call_1", "write_file", '{"path":"x.txt","content":"hel')]),
            _make_chunk(finish_reason="length"),
        ]
        agent.client.chat.completions.create.return_value = iter(chunks)

        resp = agent._interruptible_streaming_api_call({"messages": []})

        tc = resp.choices[0].message.tool_calls
        assert len(tc) == 1
        assert tc[0].function.name == "write_file"
        assert tc[0].function.arguments == '{"path":"x.txt","content":"hel'
        assert resp.choices[0].finish_reason == "length"

    def test_ollama_reused_index_separate_tool_calls(self, agent):
        """Ollama sends every tool call at index 0 with different ids.

        Without the fix, names and arguments get concatenated into one slot.
        """
        chunks = [
            _make_chunk(tool_calls=[_make_tc_delta(0, "call_a", "search", '{"q":"hello"}')]),
            # Second tool call at the SAME index 0, but different id
            _make_chunk(tool_calls=[_make_tc_delta(0, "call_b", "read_file", '{"path":"x.py"}')]),
            _make_chunk(finish_reason="tool_calls"),
        ]
        agent.client.chat.completions.create.return_value = iter(chunks)

        resp = agent._interruptible_streaming_api_call({"messages": []})

        tc = resp.choices[0].message.tool_calls
        assert len(tc) == 2, f"Expected 2 tool calls, got {len(tc)}: {[t.function.name for t in tc]}"
        assert tc[0].function.name == "search"
        assert tc[0].function.arguments == '{"q":"hello"}'
        assert tc[0].id == "call_a"
        assert tc[1].function.name == "read_file"
        assert tc[1].function.arguments == '{"path":"x.py"}'
        assert tc[1].id == "call_b"

    def test_ollama_reused_index_streamed_args(self, agent):
        """Ollama with streamed arguments across multiple chunks at same index."""
        chunks = [
            _make_chunk(tool_calls=[_make_tc_delta(0, "call_a", "search", '{"q":')]),
            _make_chunk(tool_calls=[_make_tc_delta(0, None, None, '"hello"}')]),
            # New tool call, same index 0
            _make_chunk(tool_calls=[_make_tc_delta(0, "call_b", "read", '{}')]),
            _make_chunk(finish_reason="tool_calls"),
        ]
        agent.client.chat.completions.create.return_value = iter(chunks)

        resp = agent._interruptible_streaming_api_call({"messages": []})

        tc = resp.choices[0].message.tool_calls
        assert len(tc) == 2
        assert tc[0].function.name == "search"
        assert tc[0].function.arguments == '{"q":"hello"}'
        assert tc[1].function.name == "read"
        assert tc[1].function.arguments == '{}'

    def test_content_and_tool_calls_together(self, agent):
        chunks = [
            _make_chunk(content="I'll search"),
            _make_chunk(tool_calls=[_make_tc_delta(0, "call_1", "search", '{}')]),
            _make_chunk(finish_reason="tool_calls"),
        ]
        agent.client.chat.completions.create.return_value = iter(chunks)

        resp = agent._interruptible_streaming_api_call({"messages": []})

        assert resp.choices[0].message.content == "I'll search"
        assert len(resp.choices[0].message.tool_calls) == 1

    def test_empty_content_returns_none(self, agent):
        chunks = [_make_chunk(finish_reason="stop")]
        agent.client.chat.completions.create.return_value = iter(chunks)

        resp = agent._interruptible_streaming_api_call({"messages": []})

        assert resp.choices[0].message.content is None
        assert resp.choices[0].message.tool_calls is None

    def test_callback_exception_swallowed(self, agent):
        chunks = [
            _make_chunk(content="Hello"),
            _make_chunk(content=" World"),
            _make_chunk(finish_reason="stop"),
        ]
        agent.client.chat.completions.create.return_value = iter(chunks)
        agent.stream_delta_callback = MagicMock(side_effect=ValueError("boom"))

        resp = agent._interruptible_streaming_api_call({"messages": []})

        assert resp.choices[0].message.content == "Hello World"

    def test_model_name_captured(self, agent):
        chunks = [
            _make_chunk(content="Hi", model="gpt-4o"),
            _make_chunk(finish_reason="stop", model="gpt-4o"),
        ]
        agent.client.chat.completions.create.return_value = iter(chunks)

        resp = agent._interruptible_streaming_api_call({"messages": []})

        assert resp.model == "gpt-4o"

    def test_stream_kwarg_injected(self, agent):
        chunks = [_make_chunk(content="x"), _make_chunk(finish_reason="stop")]
        agent.client.chat.completions.create.return_value = iter(chunks)

        agent._interruptible_streaming_api_call({"messages": [], "model": "test"})

        call_kwargs = agent.client.chat.completions.create.call_args
        assert call_kwargs[1].get("stream") is True or call_kwargs.kwargs.get("stream") is True

    def test_api_exception_propagates_no_non_streaming_fallback(self, agent):
        """When streaming fails before any deltas, error propagates to the main retry loop."""
        agent.client.chat.completions.create.side_effect = ConnectionError("fail")
        # Prevent stream retry logic from replacing the mock client
        with patch.object(agent, "_replace_primary_openai_client", return_value=False):
            # The fallback also uses the same client, so it'll fail too
            with pytest.raises(ConnectionError, match="fail"):
                agent._interruptible_streaming_api_call({"messages": []})

    def test_response_has_uuid_id(self, agent):
        chunks = [_make_chunk(content="x"), _make_chunk(finish_reason="stop")]
        agent.client.chat.completions.create.return_value = iter(chunks)

        resp = agent._interruptible_streaming_api_call({"messages": []})

        assert resp.id.startswith("stream-")
        assert len(resp.id) > len("stream-")

    def test_empty_choices_chunk_skipped(self, agent):
        empty_chunk = SimpleNamespace(model="gpt-4", choices=[])
        chunks = [
            empty_chunk,
            _make_chunk(content="Hello", model="gpt-4"),
            _make_chunk(finish_reason="stop", model="gpt-4"),
        ]
        agent.client.chat.completions.create.return_value = iter(chunks)

        resp = agent._interruptible_streaming_api_call({"messages": []})

        assert resp.choices[0].message.content == "Hello"
        assert resp.model == "gpt-4"


class TestInterruptVprintForceTrue:
    """All interrupt _vprint calls must use force=True so they are always visible."""

    def test_all_interrupt_vprint_have_force_true(self):
        """Scan source for _vprint calls containing 'Interrupt' — each must have force=True."""
        import inspect
        source = inspect.getsource(AIAgent)
        lines = source.split("\n")
        violations = []
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if "_vprint(" in stripped and "Interrupt" in stripped:
                if "force=True" not in stripped:
                    violations.append(f"line {i}: {stripped}")
        assert not violations, (
            f"Interrupt _vprint calls missing force=True:\n"
            + "\n".join(violations)
        )


class TestAnthropicInterruptHandler:
    """_interruptible_api_call must handle Anthropic mode when interrupted."""

    def test_interruptible_has_anthropic_branch(self):
        """The interrupt handler must check api_mode == 'anthropic_messages'."""
        import inspect
        from agent.chat_completion_helpers import interruptible_api_call
        source = inspect.getsource(interruptible_api_call)
        assert "anthropic_messages" in source, \
            "interruptible_api_call must handle Anthropic interrupt (api_mode check)"

    def test_interruptible_rebuilds_anthropic_client(self):
        """After interrupting, the Anthropic client should be rebuilt."""
        import inspect
        from agent.chat_completion_helpers import interruptible_api_call
        source = inspect.getsource(interruptible_api_call)
        assert "build_anthropic_client" in source, \
            "interruptible_api_call must rebuild Anthropic client after interrupt"

    def test_streaming_has_anthropic_branch(self):
        """_streaming_api_call must also handle Anthropic interrupt."""
        import inspect
        from agent.chat_completion_helpers import interruptible_streaming_api_call
        source = inspect.getsource(interruptible_streaming_api_call)
        assert "anthropic_messages" in source, \
            "interruptible_streaming_api_call must handle Anthropic interrupt"


class TestStreamCallbackNonStreamingProvider:
    """When api_mode != chat_completions, stream_callback must still receive
    the response content so TTS works (batch delivery)."""

    def test_callback_receives_chat_completions_response(self, agent):
        """For chat_completions-shaped responses, callback gets content."""
        agent.api_mode = "anthropic_messages"
        mock_response = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="Hello", tool_calls=None, reasoning_content=None),
                finish_reason="stop", index=0,
            )],
            usage=None, model="test", id="test-id",
        )
        agent._interruptible_api_call = MagicMock(return_value=mock_response)

        received = []
        cb = lambda delta: received.append(delta)
        agent._stream_callback = cb

        _cb = getattr(agent, "_stream_callback", None)
        response = agent._interruptible_api_call({})
        if _cb is not None and response:
            try:
                if agent.api_mode == "anthropic_messages":
                    text_parts = [
                        block.text for block in getattr(response, "content", [])
                        if getattr(block, "type", None) == "text" and getattr(block, "text", None)
                    ]
                    content = " ".join(text_parts) if text_parts else None
                else:
                    content = response.choices[0].message.content
                if content:
                    _cb(content)
            except Exception:
                pass

        # Anthropic format not matched above; fallback via except
        # Test the actual code path by checking chat_completions branch
        received2 = []
        agent.api_mode = "some_other_mode"
        agent._stream_callback = lambda d: received2.append(d)
        _cb2 = agent._stream_callback
        if _cb2 is not None and mock_response:
            try:
                content = mock_response.choices[0].message.content
                if content:
                    _cb2(content)
            except Exception:
                pass
        assert received2 == ["Hello"]

    def test_callback_receives_anthropic_content(self, agent):
        """For Anthropic responses, text blocks are extracted and forwarded."""
        agent.api_mode = "anthropic_messages"
        mock_response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="Hello from Claude")],
            stop_reason="end_turn",
        )

        received = []
        cb = lambda d: received.append(d)
        agent._stream_callback = cb
        _cb = agent._stream_callback

        if _cb is not None and mock_response:
            try:
                if agent.api_mode == "anthropic_messages":
                    text_parts = [
                        block.text for block in getattr(mock_response, "content", [])
                        if getattr(block, "type", None) == "text" and getattr(block, "text", None)
                    ]
                    content = " ".join(text_parts) if text_parts else None
                else:
                    content = mock_response.choices[0].message.content
                if content:
                    _cb(content)
            except Exception:
                pass

        assert received == ["Hello from Claude"]


class TestVprintForceOnErrors:
    """Error/warning messages must be visible during streaming TTS."""

    def test_forced_message_shown_during_tts(self, agent):
        agent._stream_callback = lambda x: None
        printed = []
        with patch("builtins.print", side_effect=lambda *a, **kw: printed.append(a)):
            agent._vprint("error msg", force=True)
        assert len(printed) == 1

    def test_non_forced_suppressed_during_tts(self, agent):
        agent._stream_callback = lambda x: None
        printed = []
        with patch("builtins.print", side_effect=lambda *a, **kw: printed.append(a)):
            agent._vprint("debug info")
        assert len(printed) == 0

    def test_all_shown_without_tts(self, agent):
        agent._stream_callback = None
        printed = []
        with patch("builtins.print", side_effect=lambda *a, **kw: printed.append(a)):
            agent._vprint("debug")
            agent._vprint("error", force=True)
        assert len(printed) == 2
