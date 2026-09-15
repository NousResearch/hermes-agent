"""Provider errors can cut tool JSON off without a `length` finish reason."""

import copy
import json
from unittest.mock import patch

import pytest

from tests.agent.test_run_agent import _mock_response, _mock_tool_call
from tests.agent.test_run_agent import agent as agent


def _tool_response(call_id, arguments, *, finish_reason="tool_calls"):
    return _mock_response(
        content="", finish_reason=finish_reason,
        tool_calls=[_mock_tool_call(
            name="web_search", arguments=arguments, call_id=call_id,
        )],
    )


def _run(agent, responses):
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    requests = []
    replies = iter(responses)

    def request(**kwargs):
        requests.append(copy.deepcopy(kwargs["messages"]))
        return next(replies)

    agent.client.chat.completions.create.side_effect = request
    with (
        patch("model_tools.handle_function_call", return_value='{"ok":true}') as dispatch,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("Inspect the first item, then the second and third.")
    return result, dispatch, requests


@pytest.mark.parametrize("finish_reason", ["error", "tool_calls", "stop"])
def test_rejected_batch_retries_without_replaying_completed_tools(agent, finish_reason):
    rejected = _tool_response("cut", '{"query":"third', finish_reason=finish_reason)
    rejected.choices[0].message.tool_calls.insert(0, _mock_tool_call(
        name="web_search", arguments='{"query":"second"}', call_id="not-yet",
    ))
    recovered = _tool_response("third", '{"query":"third"}')
    recovered.choices[0].message.tool_calls.insert(0, _mock_tool_call(
        name="web_search", arguments='{"query":"second"}', call_id="second",
    ))
    result, dispatch, requests = _run(agent, [
        _tool_response("first", '{"query":"first"}'),
        rejected,
        recovered,
        _mock_response(content="All three inspected."),
    ])

    assert result["completed"] is True, result.get("error")
    assert result["final_response"] == "All three inspected."
    assert dispatch.call_count == 3
    assert sorted(call.args[1]["query"] for call in dispatch.call_args_list) == [
        "first", "second", "third",
    ]
    # Retry only the rejected model generation: keep the prior tool result and
    # never append the incomplete response (including its valid sibling call).
    assert requests[1] == requests[2]
    assert '"cut"' not in json.dumps(result["messages"])
    assert '"not-yet"' not in json.dumps(result["messages"])


def test_repeated_truncation_stops_without_dispatching_or_corrupting_history(agent):
    result, dispatch, requests = _run(agent, [
        _tool_response("first", '{"query":"first"}'),
        *[_tool_response(f"cut-{i}", '{"query":"second', finish_reason="error")
          for i in range(3)],
    ])

    assert result["completed"] is False
    assert result["partial"] is True
    assert "truncated tool arguments" in result["error"].lower()
    assert "output length limit" not in result["error"]
    assert len(requests) == 4
    assert requests[1] == requests[2] == requests[3]
    dispatch.assert_called_once()
    messages = result["messages"]
    assert messages[-1]["role"] == "assistant"
    assert any(message["role"] == "tool" for message in messages)
    assert "cut-" not in json.dumps(messages)
