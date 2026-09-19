"""Regression tests for custom OpenAI-compatible message-count limits."""

from types import SimpleNamespace

from agent.agent_runtime_helpers import cap_custom_provider_messages


def _custom_agent():
    return SimpleNamespace(provider="custom")


def test_custom_provider_wire_copy_is_capped_without_mutating_transcript():
    messages = [{"role": "system", "content": "instructions"}]
    for index in range(64):
        call_id = f"call-{index}"
        messages.extend(
            [
                {"role": "user", "content": f"request {index}"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": "inspect", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": call_id, "content": "done"},
            ]
        )
    original_count = len(messages)

    capped = cap_custom_provider_messages(_custom_agent(), messages)

    assert len(messages) == original_count
    assert len(capped) <= 128
    assert capped[0]["role"] == "system"
    assert capped[1]["role"] == "user"
    assert capped[-1]["content"] == "done"
    for index, message in enumerate(capped):
        if message.get("role") == "assistant" and message.get("tool_calls"):
            assert capped[index + 1]["role"] == "tool"


def test_non_custom_provider_is_not_capped():
    messages = [{"role": "user", "content": str(index)} for index in range(129)]

    result = cap_custom_provider_messages(
        SimpleNamespace(provider="openrouter"), messages
    )

    assert result is messages
    assert len(result) == 129


def test_build_api_kwargs_caps_final_transport_payload(monkeypatch):
    from agent import chat_completion_helpers

    monkeypatch.setattr(
        chat_completion_helpers,
        "_build_api_kwargs_for_mode",
        lambda agent, api_messages, tools_for_api=None: {"messages": api_messages},
    )
    agent = SimpleNamespace(provider="custom", base_url="", session_id=None)
    messages = [{"role": "user", "content": str(index)} for index in range(129)]

    kwargs = chat_completion_helpers.build_api_kwargs(agent, messages)
    assert len(kwargs["messages"]) == 128


def test_cap_keeps_hard_limit_when_instructions_fill_budget():
    messages = [{"role": "system", "content": "instruction"}] * 129
    messages.append({"role": "user", "content": "latest"})

    result = cap_custom_provider_messages(_custom_agent(), messages)

    assert len(result) == 128
    assert result[0]["role"] == "system"
    assert result[-1]["content"] == "latest"
