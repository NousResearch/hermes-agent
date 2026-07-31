"""P0 regression for the Copilot ACP model-input boundary."""

from agent.copilot_acp_client import _format_messages_as_prompt


def test_prompt_formatter_bounds_tool_result_text():
    prompt = _format_messages_as_prompt(
        [
            {"role": "user", "content": "continue"},
            {
                "role": "tool",
                "tool_call_id": "call_large",
                "content": "x" * 20_000,
                "_tool_result_budget": {
                    "limit_tokens": 10_000,
                    "override_requested": False,
                },
            },
        ]
    )

    assert prompt.count("x") < 10_000
    assert "model-visible 10,000-unit limit" in prompt
    assert "_tool_result_budget" not in prompt
