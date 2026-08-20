"""The Anthropic streaming path must not accept a tool_use block whose stream
died before its input arrived (#80498 sibling).

The chat_completions accumulator flags zero-byte tool-call arguments on a
clean no-finish_reason stream end and routes them through the
partial-stream-stub retry path. The Anthropic path had the same gap in a
different shape: a clean SSE close after ``content_block_start(tool_use)``
but before any ``input_json_delta``/``message_delta`` yields an SDK
final-message snapshot whose content is NON-empty (the tool_use block is
there, ``input={}``) and whose ``stop_reason`` is None — which sailed past
both empty-stream guards and executed the tool with empty input, no retry.

The fix raises EmptyStreamError for a tool_use-bearing message with no
stop_reason, riding the same bounded stream-retry the eventless case uses.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hermes_constants import PARTIAL_STREAM_STUB_ID


def _make_anthropic_agent(**kwargs):
    from run_agent import AIAgent

    defaults = dict(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="claude-opus-4-7",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    defaults.update(kwargs)
    agent = AIAgent(**defaults)
    agent.api_mode = "anthropic_messages"
    agent._anthropic_client = MagicMock()
    agent._anthropic_api_key = "test-anthropic-key"
    agent._create_request_anthropic_client = lambda *a, **k: agent._anthropic_client
    return agent


def _stream_cm(final_message, events=()):
    cm = MagicMock()
    stream = MagicMock()
    stream.__iter__ = MagicMock(return_value=iter(list(events)))
    stream.get_final_message = MagicMock(return_value=final_message)
    cm.__enter__ = MagicMock(return_value=stream)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


def _tool_use_block(name="write_file", input_obj=None):
    return SimpleNamespace(
        type="tool_use",
        id="toolu_x",
        name=name,
        input=input_obj if input_obj is not None else {},
    )


def _tool_use_start_event(name="write_file"):
    return SimpleNamespace(
        type="content_block_start",
        content_block=SimpleNamespace(type="tool_use", name=name),
    )


def _text_delta_event(text="I'll write that now."):
    return SimpleNamespace(
        type="content_block_delta",
        delta=SimpleNamespace(type="text_delta", text=text),
    )


class TestAnthropicMidToolCallStreamDrop:
    def test_tool_use_without_stop_reason_raises_empty_stream(self):
        """The #80498-sibling shape: tool_use block present, stop_reason None."""
        from agent.chat_completion_helpers import EmptyStreamError

        dropped = MagicMock()
        dropped.content = [_tool_use_block()]
        dropped.stop_reason = None
        dropped.usage = SimpleNamespace(input_tokens=10, output_tokens=2)

        agent = _make_anthropic_agent()
        agent._anthropic_client.messages.stream = MagicMock(
            return_value=_stream_cm(dropped, events=[_tool_use_start_event()])
        )

        with pytest.raises(EmptyStreamError, match="tool_use"):
            agent._interruptible_streaming_api_call({"model": "claude-opus-4-7"})

    def test_completed_tool_use_with_stop_reason_passes(self):
        """A legitimate tool_use completion (stop_reason set) is untouched."""
        done = MagicMock()
        done.content = [_tool_use_block(input_obj={"path": "a.txt"})]
        done.stop_reason = "tool_use"
        done.usage = SimpleNamespace(input_tokens=10, output_tokens=5)

        agent = _make_anthropic_agent()
        agent._anthropic_client.messages.stream = MagicMock(
            return_value=_stream_cm(done, events=[_tool_use_start_event()])
        )

        response = agent._interruptible_streaming_api_call(
            {"model": "claude-opus-4-7"}
        )
        assert response is done

    def test_text_only_message_without_stop_reason_passes(self):
        """No tool_use block -> the new gate stays out of the way (text-only
        no-stop_reason handling keeps its pre-existing behavior)."""
        text_only = MagicMock()
        text_only.content = [SimpleNamespace(type="text", text="partial answer")]
        text_only.stop_reason = None
        text_only.usage = SimpleNamespace(input_tokens=10, output_tokens=5)

        agent = _make_anthropic_agent()
        agent._anthropic_client.messages.stream = MagicMock(
            return_value=_stream_cm(text_only)
        )

        response = agent._interruptible_streaming_api_call(
            {"model": "claude-opus-4-7"}
        )
        assert response is text_only

    def test_preamble_then_dropped_tool_retries_in_place(self, monkeypatch):
        """A tool drop remains retryable after Anthropic streamed preamble text."""
        monkeypatch.setenv("HERMES_STREAM_RETRIES", "1")

        dropped = MagicMock()
        dropped.content = [_tool_use_block()]
        dropped.stop_reason = None
        dropped.usage = SimpleNamespace(input_tokens=10, output_tokens=2)

        done = MagicMock()
        done.content = [_tool_use_block(input_obj={"path": "a.txt"})]
        done.stop_reason = "tool_use"
        done.usage = SimpleNamespace(input_tokens=10, output_tokens=5)

        agent = _make_anthropic_agent()
        agent.stream_delta_callback = MagicMock()
        agent._anthropic_client.messages.stream = MagicMock(
            side_effect=[
                _stream_cm(
                    dropped,
                    events=[_text_delta_event(), _tool_use_start_event()],
                ),
                _stream_cm(done, events=[_tool_use_start_event()]),
            ]
        )

        response = agent._interruptible_streaming_api_call(
            {"model": "claude-opus-4-7"}
        )

        assert response is done
        assert agent._anthropic_client.messages.stream.call_count == 2

    def test_exhausted_retry_stub_names_dropped_anthropic_tool(self, monkeypatch):
        """Retry exhaustion preserves the dropped tool name in the warning stub."""
        monkeypatch.setenv("HERMES_STREAM_RETRIES", "1")

        dropped = MagicMock()
        dropped.content = [_tool_use_block()]
        dropped.stop_reason = None
        dropped.usage = SimpleNamespace(input_tokens=10, output_tokens=2)

        agent = _make_anthropic_agent()
        agent.stream_delta_callback = MagicMock()
        agent._anthropic_client.messages.stream = MagicMock(
            side_effect=[
                _stream_cm(
                    dropped,
                    events=[_text_delta_event(), _tool_use_start_event()],
                ),
                _stream_cm(
                    dropped,
                    events=[_text_delta_event(), _tool_use_start_event()],
                ),
            ]
        )

        response = agent._interruptible_streaming_api_call(
            {"model": "claude-opus-4-7"}
        )

        assert agent._anthropic_client.messages.stream.call_count == 2
        assert response.id == PARTIAL_STREAM_STUB_ID
        assert response._dropped_tool_names == ["write_file"]
        assert (
            "Stream stalled mid tool-call (write_file)"
            in response.choices[0].message.content
        )

    def test_retry_state_does_not_promote_later_text_only_drop(self, monkeypatch):
        """A prior tool drop must not make a later text-only drop retryable."""
        monkeypatch.setenv("HERMES_STREAM_RETRIES", "2")

        dropped_tool = MagicMock()
        dropped_tool.content = [_tool_use_block()]
        dropped_tool.stop_reason = None
        dropped_tool.usage = SimpleNamespace(input_tokens=10, output_tokens=2)

        dropped_text = MagicMock()
        dropped_text.content = []
        dropped_text.stop_reason = None
        dropped_text.usage = SimpleNamespace(input_tokens=10, output_tokens=2)

        unexpected_retry = MagicMock()
        unexpected_retry.content = [_tool_use_block(input_obj={"path": "a.txt"})]
        unexpected_retry.stop_reason = "tool_use"
        unexpected_retry.usage = SimpleNamespace(input_tokens=10, output_tokens=5)

        agent = _make_anthropic_agent()
        agent.stream_delta_callback = MagicMock()
        agent._anthropic_client.messages.stream = MagicMock(
            side_effect=[
                _stream_cm(dropped_tool, events=[_tool_use_start_event()]),
                _stream_cm(dropped_text, events=[_text_delta_event()]),
                _stream_cm(unexpected_retry, events=[_tool_use_start_event()]),
            ]
        )

        response = agent._interruptible_streaming_api_call(
            {"model": "claude-opus-4-7"}
        )

        assert agent._anthropic_client.messages.stream.call_count == 2
        assert response.id == PARTIAL_STREAM_STUB_ID
        assert response._dropped_tool_names is None
