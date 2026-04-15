"""F-L1 step 1: direct tests of agent.message_sanitizer.

Catches future regressions that bypass the AIAgent class-level shims —
e.g. if someone calls ``from agent.message_sanitizer import sanitize_api_messages``
directly (new call sites will). The class-level shims are separately
covered by tests/test_agent_guardrails.py.
"""
from __future__ import annotations

import pytest

from agent.message_sanitizer import (
    cap_delegate_task_calls,
    deduplicate_tool_calls,
    get_tool_call_id,
    sanitize_api_messages,
)


class _FakeToolCall:
    class _Fn:
        def __init__(self, name, arguments):
            self.name = name
            self.arguments = arguments

    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.function = self._Fn(name, arguments)


def test_get_tool_call_id_from_dict():
    assert get_tool_call_id({"id": "call-1"}) == "call-1"
    assert get_tool_call_id({}) == ""
    assert get_tool_call_id({"id": None}) == ""


def test_get_tool_call_id_from_object():
    tc = _FakeToolCall("call-2", "x", "{}")
    assert get_tool_call_id(tc) == "call-2"


def test_sanitize_drops_orphan_tool_results():
    msgs = [
        {"role": "assistant", "tool_calls": [{"id": "a"}]},
        {"role": "tool", "tool_call_id": "a", "content": "ok"},
        {"role": "tool", "tool_call_id": "ORPHAN", "content": "should drop"},
    ]
    out = sanitize_api_messages(msgs)
    ids = [m.get("tool_call_id") for m in out if m.get("role") == "tool"]
    assert ids == ["a"]


def test_sanitize_patches_missing_tool_results():
    msgs = [
        {"role": "assistant", "tool_calls": [{"id": "a"}, {"id": "b"}]},
        {"role": "tool", "tool_call_id": "a", "content": "ok"},
        # b's result is missing — must be stubbed
    ]
    out = sanitize_api_messages(msgs)
    ids = [m.get("tool_call_id") for m in out if m.get("role") == "tool"]
    assert set(ids) == {"a", "b"}
    stub = next(m for m in out if m.get("tool_call_id") == "b")
    assert "unavailable" in stub["content"].lower()


def test_cap_delegate_preserves_non_delegate_calls():
    from tools.delegate_tool import MAX_CONCURRENT_CHILDREN as CAP
    calls = [_FakeToolCall(f"d{i}", "delegate_task", "{}") for i in range(CAP + 3)]
    calls.append(_FakeToolCall("keep", "web_search", "{}"))
    out = cap_delegate_task_calls(calls)
    delegates = [c for c in out if c.function.name == "delegate_task"]
    assert len(delegates) == CAP
    assert any(c.function.name == "web_search" for c in out)


def test_cap_delegate_noop_when_under_cap():
    calls = [_FakeToolCall("x", "delegate_task", "{}")]
    assert cap_delegate_task_calls(calls) is calls


def test_deduplicate_keeps_first_occurrence():
    calls = [
        _FakeToolCall("1", "foo", '{"a": 1}'),
        _FakeToolCall("2", "foo", '{"a": 1}'),   # dup
        _FakeToolCall("3", "foo", '{"a": 2}'),   # different args
    ]
    out = deduplicate_tool_calls(calls)
    assert [c.id for c in out] == ["1", "3"]


def test_deduplicate_noop_when_clean():
    calls = [
        _FakeToolCall("1", "foo", "{}"),
        _FakeToolCall("2", "bar", "{}"),
    ]
    assert deduplicate_tool_calls(calls) is calls
