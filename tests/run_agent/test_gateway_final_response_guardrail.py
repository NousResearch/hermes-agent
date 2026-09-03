"""Regression tests for gateway final-response fan-out guardrails."""

from typing import Any

from run_agent import AIAgent


def _agent(platform: Any = "discord"):
    agent = AIAgent.__new__(AIAgent)
    setattr(agent, "platform", platform)
    setattr(agent, "_gateway_session_key", "discord:channel:thread")
    setattr(agent, "session_id", "session-123")
    return agent


def test_gateway_final_response_guardrail_caps_response_and_persisted_message(monkeypatch):
    monkeypatch.setenv("HERMES_GATEWAY_MAX_FINAL_RESPONSE_CHARS", "200")
    agent = _agent()
    long_text = "x" * 500
    messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": long_text},
    ]

    bounded, truncated, original_chars, limit = AIAgent._apply_gateway_final_response_guardrail(
        agent,
        long_text,
        messages,
    )

    assert truncated is True
    assert original_chars == 500
    assert limit == 200
    assert len(bounded) <= 200
    assert "Response truncated by Hermes gateway" in bounded
    assert len(messages[-1]["content"]) <= 200
    assert "Response truncated by Hermes gateway" in messages[-1]["content"]
    assert messages[-1]["gateway_response_truncated"] is True
    assert messages[-1]["gateway_response_original_chars"] == 500
    assert messages[-1]["gateway_response_limit_chars"] == 200


def test_gateway_final_response_guardrail_preserves_tool_call_messages(monkeypatch):
    monkeypatch.setenv("HERMES_GATEWAY_MAX_FINAL_RESPONSE_CHARS", "200")
    agent = _agent()
    tool_call_message = {
        "role": "assistant",
        "content": "x" * 500,
        "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "noop", "arguments": "{}"}},
        ],
    }
    final_message = {"role": "assistant", "content": "y" * 500}
    messages = [tool_call_message, final_message]

    AIAgent._apply_gateway_final_response_guardrail(agent, "y" * 500, messages)

    assert tool_call_message["content"] == "x" * 500
    assert tool_call_message.get("gateway_response_truncated") is None
    assert len(final_message["content"]) <= 200


def test_gateway_final_response_guardrail_disabled_for_cli_without_env(monkeypatch):
    monkeypatch.delenv("HERMES_GATEWAY_MAX_FINAL_RESPONSE_CHARS", raising=False)
    monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)
    agent = _agent(platform=None)
    setattr(agent, "_gateway_session_key", None)
    long_text = "x" * 500
    messages = [{"role": "assistant", "content": long_text}]

    bounded, truncated, original_chars, limit = AIAgent._apply_gateway_final_response_guardrail(
        agent,
        long_text,
        messages,
    )

    assert bounded == long_text
    assert messages[-1]["content"] == long_text
    assert truncated is False
    assert original_chars == 500
    assert limit == 0
