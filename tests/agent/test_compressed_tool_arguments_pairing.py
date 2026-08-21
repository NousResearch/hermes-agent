import json

from agent.tool_argument_integrity import (
    INCOMPLETE_TOOL_ARGUMENTS_KEY,
    neutralize_completed_incomplete_tool_calls,
)


def call(marker=False):
    args = {INCOMPLETE_TOOL_ARGUMENTS_KEY: {"version": 1}} if marker else {"path": "x"}
    return {"id": "same", "type": "function", "function": {"name": "read_file", "arguments": json.dumps(args)}}


def assistant(marker=False):
    return {"role": "assistant", "content": "", "tool_calls": [call(marker)]}


def test_prior_pair_with_reused_id_does_not_complete_later_marker():
    messages = [assistant(), {"role": "tool", "tool_call_id": "same", "content": "ok"}, {"role": "user", "content": "later"}, assistant(True)]
    assert neutralize_completed_incomplete_tool_calls(messages) == messages


def test_result_preceding_marker_does_not_complete_it():
    messages = [{"role": "tool", "tool_call_id": "same", "content": "orphan"}, assistant(True)]
    assert neutralize_completed_incomplete_tool_calls(messages) == messages


def test_duplicate_ids_in_one_batch_remain_fail_closed():
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [call(True), call()]},
        {"role": "tool", "tool_call_id": "same", "content": "ambiguous"},
    ]
    assert neutralize_completed_incomplete_tool_calls(messages) == messages


def test_duplicate_results_leave_marker_history_unchanged():
    messages = [
        assistant(True),
        {"role": "tool", "tool_call_id": "same", "content": "first"},
        {"role": "tool", "tool_call_id": "same", "content": "duplicate"},
    ]
    assert neutralize_completed_incomplete_tool_calls(messages) == messages


def test_preceding_orphan_makes_later_pair_ambiguous():
    messages = [
        {"role": "tool", "tool_call_id": "same", "content": "orphan"},
        assistant(True),
        {"role": "tool", "tool_call_id": "same", "content": "later"},
    ]
    assert neutralize_completed_incomplete_tool_calls(messages) == messages
