"""Tests for _estimate_chunk_bytes — the cheap per-chunk stream size proxy.

Replaced ``len(repr(chunk))`` in the streaming hot loop. Contract:
- proportional to delta payload length (content, reasoning, tool args);
- never raises on unknown shapes (returns the framing floor);
- monotonic accumulation stays positive.
"""

from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import (
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)

from agent.chat_completion_helpers import _estimate_chunk_bytes


def _chunk(delta):
    return ChatCompletionChunk(
        id="chatcmpl-1",
        created=1730000000,
        model="m",
        object="chat.completion.chunk",
        choices=[Choice(index=0, delta=delta, finish_reason=None)],
    )


def test_content_chunk_scales_with_payload():
    small = _estimate_chunk_bytes(_chunk(ChoiceDelta(content="hi")))
    large = _estimate_chunk_bytes(_chunk(ChoiceDelta(content="x" * 500)))
    assert large - small == 498
    assert small > 0


def test_reasoning_content_counted():
    base = _estimate_chunk_bytes(_chunk(ChoiceDelta()))

    class Delta:
        content = None
        reasoning_content = "thinking..."
        reasoning = None
        tool_calls = None

    class Choice0:
        delta = Delta()

    class Chunk:
        choices = [Choice0()]

    assert _estimate_chunk_bytes(Chunk()) == base + len("thinking...")


def test_tool_call_arguments_counted():
    delta = ChoiceDelta(
        tool_calls=[
            ChoiceDeltaToolCall(
                index=0,
                function=ChoiceDeltaToolCallFunction(
                    arguments='{"path": "/tmp/file"}', name="read_file"
                ),
            )
        ]
    )
    est = _estimate_chunk_bytes(_chunk(delta))
    assert est >= 40 + len('{"path": "/tmp/file"}') + len("read_file")


def test_unknown_shape_returns_floor_never_raises():
    class Weird:
        pass

    assert _estimate_chunk_bytes(Weird()) == 40
    assert _estimate_chunk_bytes(None) == 40
    assert _estimate_chunk_bytes(object()) == 40


def test_anthropic_style_event_text_counted():
    class Delta:
        text = "anthropic text delta"
        partial_json = None

    class Event:
        choices = None
        delta = Delta()

    assert _estimate_chunk_bytes(Event()) == 40 + len("anthropic text delta")
