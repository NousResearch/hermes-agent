"""Regression coverage for raw SSE responses from OpenAI-compatible auxiliary providers."""

import pytest

from agent.auxiliary_client import _recover_raw_sse_response, _validate_llm_response


def test_raw_sse_body_is_recovered_as_chat_completion():
    response = _validate_llm_response(
        'data: {"id":"req-1","model":"vision","choices":[{"delta":{"role":"assistant"}}]}\n'
        'data: {"choices":[{"delta":{"content":"first "}}]}\n'
        'data: {"choices":[{"delta":{"content":"second"},"finish_reason":"stop"}]}\n'
        'data: [DONE]\n',
        task="vision",
    )

    assert response.choices[0].message.content == "first second"
    assert response.choices[0].message.tool_calls is None
    assert response.choices[0].finish_reason == "stop"


@pytest.mark.parametrize(
    "body",
    [
        'data: {"error":{"message":"upstream failed"}}\n',
        "data: [DONE]\n",
        'data: {"choices":[{"delta":{"role":"assistant"}}]}\n',
    ],
)
def test_raw_sse_without_content_or_tools_remains_invalid(body):
    assert _recover_raw_sse_response(body) is None
    with pytest.raises(RuntimeError, match="invalid response"):
        _validate_llm_response(body, task="vision")


def test_raw_sse_tool_calls_are_reassembled():
    response = _validate_llm_response(
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-1","type":"function",'
        '"function":{"name":"lookup","arguments":"{\\"q\\":"}}]}}]}\n'
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"hermes\\"}"}}]},'
        '"finish_reason":"tool_calls"}]}\n',
        task="mcp",
    )

    choice = response.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message.content == ""
    assert choice.message.tool_calls[0].id == "call-1"
    assert choice.message.tool_calls[0].function.name == "lookup"
    assert choice.message.tool_calls[0].function.arguments == '{"q":"hermes"}'
