"""Regression: an in-flight /model note must not rename the fallback on retry.

The gateway prepends a user-turn note naming the requested model. When that
route fails, the system message forwarded to the retry must correct the note
without changing the original user message.
"""

from unittest.mock import patch

from agent.conversation_loop import _sync_failover_system_message
from tests.agent.test_primary_runtime_restore import _make_agent, _mock_resolve


def test_retry_identifies_fallback_after_failed_model_switch():
    requested = "requested-after-switch"
    fallback = "anthropic/claude-sonnet-4"
    original_prompt = f"You are Hermes.\nModel: {requested}\nProvider: custom"
    agent = _make_agent(fallback_model={"provider": "openrouter", "model": fallback})
    agent.model = requested
    agent._primary_runtime["model"] = requested
    agent._cached_system_prompt = original_prompt

    # A real gateway /model note is already in the request when the primary fails.
    user_turn = (
        "[Note: model was just switched from previous to requested-after-switch "
        "via custom. Adjust your self-identification accordingly.]\n"
        "Which model are you?"
    )
    api_messages = [
        {"role": "system", "content": original_prompt},
        {"role": "user", "content": user_turn},
    ]

    with patch(
        "agent.auxiliary_client.resolve_provider_client",
        return_value=(_mock_resolve(), None),
    ):
        assert agent._try_activate_fallback() is True

    active_prompt = _sync_failover_system_message(agent, api_messages, original_prompt)
    wire_content = api_messages[0]["content"]
    if isinstance(wire_content, list):
        wire_content = "".join(block.get("text", "") for block in wire_content)

    assert agent.model == fallback
    assert active_prompt == agent._cached_system_prompt
    assert f"Model: {fallback}" in wire_content
    assert f"You are actually {fallback} via openrouter" in wire_content
    assert f"fallback for {requested}" in wire_content
    assert api_messages[1]["content"] == user_turn
