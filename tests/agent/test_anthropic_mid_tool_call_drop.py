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
    stream.__iter__ = MagicMock(side_effect=lambda *_a: iter(list(events)))  # fresh per retry
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


_MESSAGE_STOP = SimpleNamespace(type="message_stop")


def _tool_use_start_event(name="write_file"):
    return SimpleNamespace(
        type="content_block_start",
        content_block=SimpleNamespace(type="tool_use", name=name),
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
            return_value=_stream_cm(dropped, events=[_tool_use_start_event(), _MESSAGE_STOP])
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
            return_value=_stream_cm(done, events=[_tool_use_start_event(), _MESSAGE_STOP])
        )

        response = agent._interruptible_streaming_api_call(
            {"model": "claude-opus-4-7"}
        )
        assert response is done

    def test_text_only_message_without_message_stop_raises_empty_stream(self):
        """No tool_use block and no message_stop -> the stream was cut short, so
        the message_stop gate (#121320) raises instead of returning the partial."""
        from agent.chat_completion_helpers import EmptyStreamError

        text_only = MagicMock()
        text_only.content = [SimpleNamespace(type="text", text="partial answer")]
        text_only.stop_reason = None
        text_only.usage = SimpleNamespace(input_tokens=10, output_tokens=5)
        text_delta = SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(type="text_delta", text="partial answer"),
        )

        agent = _make_anthropic_agent()
        agent._anthropic_client.messages.stream = MagicMock(
            return_value=_stream_cm(text_only, events=[text_delta])
        )

        with pytest.raises(EmptyStreamError, match="message_stop"):
            agent._interruptible_streaming_api_call({"model": "claude-opus-4-7"})
