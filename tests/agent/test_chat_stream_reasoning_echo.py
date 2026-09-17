"""Reasoning-echo rejection in _finish_chat_stream.

Regression for #109664: a provider that returns the same text in both
``reasoning_content`` and visible ``content`` with finish_reason="stop" and
no tool calls must not persist as a valid final answer. The duplicated
reasoning is internal thought, not an answer.
"""

from types import SimpleNamespace

import pytest

from agent.chat_completion_helpers import _StreamingCall
from agent.errors import EmptyStreamError


def _finish(content, reasoning, finish_reason="stop", tool_calls_acc=None):
    call = object.__new__(_StreamingCall)
    stream = SimpleNamespace(response=None)
    return call._finish_chat_stream(
        stream, "assistant", [content] if content else [],
        [reasoning] if reasoning else [], tool_calls_acc or {},
        finish_reason, "test-model", SimpleNamespace(),
        flush_pending=lambda: None,
    )


def test_identical_content_and_reasoning_is_rejected():
    with pytest.raises(EmptyStreamError):
        _finish("private reasoning with no closing boundary",
                "private reasoning with no closing boundary")


def test_distinct_content_and_reasoning_passes_through():
    final = _finish("Here is your answer", "let me think about this")
    message = final.choices[0].message
    assert message.content == "Here is your answer"
    assert message.reasoning_content == "let me think about this"
    assert final.choices[0].finish_reason == "stop"
