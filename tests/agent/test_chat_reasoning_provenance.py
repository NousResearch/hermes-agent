"""Reasoning wire provenance survives both chat streaming assembly paths."""

from types import SimpleNamespace

import pytest

from agent.chat_completion_helpers import _StreamingCall
from agent.chat_completion_helpers_relay import RelayChatAccumulator


@pytest.mark.parametrize(
    ("source", "reasoning", "reasoning_content"),
    [
        ("reasoning", "parser answer", None),
        ("reasoning_content", None, "private thought"),
        ("mixed", None, "private thought"),
    ],
)
def test_chat_stream_assembly_preserves_reasoning_source(source, reasoning, reasoning_content):
    call = object.__new__(_StreamingCall)

    response = call._finish_chat_stream(
        SimpleNamespace(response=None),
        "assistant",
        [],
        ["parser answer" if source == "reasoning" else "private thought"],
        {},
        "stop",
        "test-model",
        None,
        flush_pending=lambda: None,
        reasoning_source=source,
    )

    message = response.choices[0].message
    assert message.reasoning == reasoning
    assert message.reasoning_content == reasoning_content


@pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
def test_relay_chat_accumulator_preserves_reasoning_source(field):
    accumulator = RelayChatAccumulator()
    accumulator.observe({
        "choices": [{
            "delta": {field: "reasoning text"},
            "finish_reason": "stop",
        }],
    })

    message = accumulator.finalize()["choices"][0]["message"]

    assert message[field] == "reasoning text"
    other = "reasoning_content" if field == "reasoning" else "reasoning"
    assert message[other] is None


def test_relay_chat_accumulator_treats_mixed_reasoning_fields_as_private():
    accumulator = RelayChatAccumulator()
    accumulator.observe({"choices": [{"delta": {"reasoning": "first"}}]})
    accumulator.observe({"choices": [{"delta": {"reasoning_content": "second"}}]})

    message = accumulator.finalize()["choices"][0]["message"]

    assert message["reasoning"] is None
    assert message["reasoning_content"] == "firstsecond"
