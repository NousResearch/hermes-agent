"""Ordered desktop operations are not redundant just because their arguments match."""

import json

import pytest

from agent.tool_call_batches import expand_local_tool_batches
from agent.transports.types import ToolCall
from run_agent import AIAgent


def _calls(actions, route):
    if route == "batch":
        return expand_local_tool_batches([ToolCall(
            id="batch", name="tool_call", arguments=json.dumps({"calls": [
                {"name": "computer_use", "arguments": action} for action in actions
            ]}),
        )])
    return [ToolCall(
        id=f"input_{index}", name="computer_use" if route == "direct" else "tool_call",
        arguments=json.dumps(action if route == "direct" else {
            "calls": [{"name": "computer_use", "arguments": action}],
        }),
    ) for index, action in enumerate(actions)]


@pytest.mark.parametrize("route", ["direct", "bridge", "batch"])
@pytest.mark.parametrize("actions", [
    # 66 * 65: both adjacent repeats and repeats separated by another input matter.
    [{"action": "click", "element": element} for element in (11, 11, 8, 11, 10, 19)],
    [{"action": "key", "keys": "tab"}] * 2,
    [{"action": "type", "text": "6"}] * 2,
    [{"action": "scroll", "direction": "down", "amount": 3}] * 2,
    [{"action": "capture"}, {"action": "click", "element": 11}, {"action": "capture"}],
])
def test_desktop_sequence_survives_deduplication_in_order(actions, route):
    calls = _calls(actions, route)
    original_arguments = [call.function.arguments for call in calls]
    original_ids = [call.id for call in calls]

    result = AIAgent._deduplicate_tool_calls(calls)

    assert result is calls
    assert [call.function.arguments for call in result] == original_arguments
    assert [call.id for call in result] == original_ids
    assert len(set(original_ids)) == len(actions)


@pytest.mark.parametrize("route", ["direct", "bridge"])
def test_desktop_exception_preserves_other_tool_deduplication(route):
    clicks = _calls([{"action": "click", "element": 11}] * 2, route)
    first = ToolCall(id="search_1", name="web_search", arguments='{"query":"test","limit":2}')
    duplicate = ToolCall(id="search_2", name="web_search", arguments='{"limit": 2, "query": "test"}')
    calls = [first, clicks[0], duplicate, clicks[1]]

    assert AIAgent._deduplicate_tool_calls(calls) == [first, *clicks]
