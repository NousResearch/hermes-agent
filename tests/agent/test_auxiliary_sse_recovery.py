"""Regression coverage for raw SSE responses from OpenAI-compatible auxiliary providers."""

from agent.auxiliary_client import _validate_llm_response


def test_raw_sse_body_is_recovered_as_chat_completion():
    response = _validate_llm_response(
        'data: {"id":"req-1","model":"vision","choices":[{"delta":{"role":"assistant"}}]}\n'
        'data: {"choices":[{"delta":{"content":"first "}}]}\n'
        'data: {"choices":[{"delta":{"content":"second"},"finish_reason":"stop"}]}\n'
        'data: [DONE]\n',
        task="vision",
    )

    assert response.choices[0].message.content == "first second"
    assert response.choices[0].finish_reason == "stop"
