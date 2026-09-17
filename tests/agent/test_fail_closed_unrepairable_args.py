"""Polarity tests for the fail-closed unrepairable-args path (RCA 2026-09-17).

Sentinel args (__hermes_malformed_tool_arguments__) must be dropped from
execution with a re-issue error result; normal and empty-object args must
pass through untouched.
"""
import json
from types import SimpleNamespace

from agent.chat_completion_helpers import _StreamingCall  # noqa: F401  (locate helper)


def _assemble(tool_calls_acc, finish_reason):
    fn = _StreamingCall._assemble_tool_calls
    return fn(tool_calls_acc, finish_reason)


def _acc(arguments, name="memory", idx=0):
    return {idx: {
        "id": f"call_{idx}", "type": "function", "extra_content": None,
        "function": {"name": name, "arguments": arguments},
    }}


def test_unrepairable_args_get_sentinel_and_flag():
    malformed = '{"operations": [{"action": "add", "content": "x'  # unterminated
    calls, flagged = _assemble(_acc(malformed), "tool_calls")
    parsed = json.loads(calls[0].function.arguments)
    assert parsed.get("__hermes_malformed_tool_arguments__") is True, calls[0].function.arguments
    assert flagged is True


def test_valid_args_pass_through_untouched():
    good = json.dumps({"todos": [{"id": "1", "content": "x", "status": "pending"}]})
    calls, flagged = _assemble(_acc(good), "tool_calls")
    assert json.loads(calls[0].function.arguments) == {"todos": [{"id": "1", "content": "x", "status": "pending"}]}
    assert flagged is False


def test_repairable_args_still_repaired_without_sentinel():
    trailing_comma = '{"a": [1, 2,]}'  # repair pass 1 fixes this
    calls, flagged = _assemble(_acc(trailing_comma, name="terminal"), "tool_calls")
    parsed = json.loads(calls[0].function.arguments)
    assert parsed == {"a": [1, 2]}, parsed
    assert "__hermes_malformed_tool_arguments__" not in parsed
    assert flagged is False


def test_empty_args_with_finish_reason_not_flagged():
    calls, flagged = _assemble(_acc("", name="delegate_task"), "tool_calls")
    assert flagged is False
    assert calls[0].function.arguments == ""
