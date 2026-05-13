from run_agent import AIAgent


def _agent(*, provider="xiaomi", base_url="https://api.xiaomimimo.com/v1", reasoning_config=None):
    agent = object.__new__(AIAgent)
    agent.provider = provider
    agent.base_url = base_url
    agent.model = "mimo-v2.5"
    agent.reasoning_config = reasoning_config if reasoning_config is not None else {"enabled": False}
    return agent


def test_xiaomi_disabled_thinking_strips_reasoning_fields_from_api_copy():
    agent = _agent(reasoning_config={"enabled": False})
    source_msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "x", "arguments": "{}"}}],
        "reasoning_content": "stale chain from another provider",
        "reasoning_details": [{"type": "reasoning.text", "text": "stale"}],
    }
    api_msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": source_msg["tool_calls"],
        "reasoning_content": source_msg["reasoning_content"],
        "reasoning_details": source_msg["reasoning_details"],
    }

    agent._copy_reasoning_content_for_api(source_msg, api_msg)

    assert "reasoning_content" not in api_msg
    assert "reasoning_details" not in api_msg


def test_xiaomi_disabled_thinking_detects_xiaomimimo_base_url():
    agent = _agent(provider="custom", base_url="https://api.xiaomimimo.com/v1", reasoning_config={"enabled": False})
    source_msg = {"role": "assistant", "reasoning_content": "stale"}
    api_msg = {"role": "assistant", "reasoning_content": "stale"}

    agent._copy_reasoning_content_for_api(source_msg, api_msg)

    assert "reasoning_content" not in api_msg


def test_xiaomi_thinking_enabled_preserves_reasoning_content():
    agent = _agent(reasoning_config={"enabled": True})
    source_msg = {"role": "assistant", "reasoning_content": "valid mimo reasoning"}
    api_msg = {"role": "assistant"}

    agent._copy_reasoning_content_for_api(source_msg, api_msg)

    assert api_msg["reasoning_content"] == "valid mimo reasoning"


def test_non_xiaomi_disabled_thinking_preserves_existing_reasoning_content():
    agent = _agent(provider="openrouter", base_url="https://openrouter.ai/api/v1", reasoning_config={"enabled": False})
    source_msg = {"role": "assistant", "reasoning_content": "provider reasoning"}
    api_msg = {"role": "assistant"}

    agent._copy_reasoning_content_for_api(source_msg, api_msg)

    assert api_msg["reasoning_content"] == "provider reasoning"
