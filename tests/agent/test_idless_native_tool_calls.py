import pytest

from agent.message_sanitization import coalesce_tool_call_id, uniquify_tool_call_ids
from agent.transports.types import ToolCall
from openai.types.chat import ChatCompletionMessageToolCall


@pytest.mark.parametrize("provider_data", [None, {}, {"call_id": " "}])
def test_idless_normalized_calls_stay_distinct_across_turns(provider_data):
    first = ToolCall(
        None, "terminal", '{"command":"git status"}',
        dict(provider_data) if provider_data is not None else None,
    )
    uniquify_tool_call_ids([first])
    first_id = coalesce_tool_call_id(first)
    second = ToolCall(
        None, first.name, first.arguments,
        dict(provider_data) if provider_data is not None else None,
    )
    uniquify_tool_call_ids([second], taken={first_id})
    assert coalesce_tool_call_id(second) == first_id + "_d2"
    assert second.id == coalesce_tool_call_id(second)


def test_idless_openai_calls_stay_distinct_across_turns():
    first = ChatCompletionMessageToolCall(
        id="", type="function", function={"name": "terminal", "arguments": "{}"}
    )
    uniquify_tool_call_ids([first])
    first_id = coalesce_tool_call_id(first)
    second = ChatCompletionMessageToolCall(
        id="", type="function", function={"name": "terminal", "arguments": "{}"}
    )
    uniquify_tool_call_ids([second], taken={first_id})
    assert second.id == first_id + "_d2"
