"""Regression coverage for NVIDIA NIM thinking-model reasoning replay."""

from types import SimpleNamespace

import pytest

from agent.nim_reasoning import is_nim_thinking_model
from run_agent import AIAgent


def _agent(*, provider="nvidia", model="deepseek-ai/deepseek-r1", base_url="https://integrate.api.nvidia.com/v1", reasoning_config=None):
    agent = object.__new__(AIAgent)
    agent.provider = provider
    agent.model = model
    agent.base_url = base_url
    agent.reasoning_config = reasoning_config
    agent.verbose_logging = False
    agent.reasoning_callback = None
    agent.stream_delta_callback = None
    agent._stream_callback = None
    return agent


def _tool_message():
    return SimpleNamespace(
        content=None,
        reasoning=None,
        reasoning_content=None,
        reasoning_details=None,
        codex_reasoning_items=None,
        codex_message_items=None,
        tool_calls=[SimpleNamespace(
            id="call_1", call_id=None, response_item_id=None, type="function",
            function=SimpleNamespace(name="terminal", arguments="{}"),
        )],
    )


@pytest.mark.parametrize("model", [
    "deepseek-ai/deepseek-r1",
    "deepseek-ai/deepseek-v4-flash-0731",
    "qwen/qwen3-next-80b-a3b-thinking",
    "moonshotai/kimi-k2-thinking",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
])
def test_nim_thinking_model_families(model):
    assert is_nim_thinking_model(model)


@pytest.mark.parametrize("model", [
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "meta/llama-3.3-70b-instruct",
    "deepseek-ai/deepseek-coder-6.7b-instruct",
])
def test_nim_non_thinking_models(model):
    assert not is_nim_thinking_model(model)


def test_nim_tool_call_pins_reasoning_content():
    agent = _agent(reasoning_config={"enabled": True, "effort": "medium"})
    message = agent._build_assistant_message(_tool_message(), "tool_calls")
    assert message["reasoning_content"] == " "


def test_custom_provider_on_official_nim_host_is_supported():
    agent = _agent(provider="custom", reasoning_config={"enabled": True})
    assert agent._needs_nim_tool_reasoning()


def test_local_nim_qwen_tool_call_pins_reasoning_content():
    agent = _agent(
        model="qwen/qwen3-next-80b-a3b-thinking",
        base_url="http://localhost:8000/v1",
        reasoning_config={"enabled": True},
    )
    message = agent._build_assistant_message(_tool_message(), "tool_calls")
    assert agent._needs_nim_tool_reasoning()
    assert message["reasoning_content"] == " "


def test_local_nim_deepseek_respects_disabled_reasoning():
    agent = _agent(
        base_url="http://localhost:8000/v1",
        reasoning_config={"enabled": False},
    )
    message = agent._build_assistant_message(_tool_message(), "tool_calls")
    assert not agent._needs_nim_tool_reasoning()
    assert not agent._needs_deepseek_tool_reasoning()
    assert "reasoning_content" not in message


@pytest.mark.parametrize("reasoning_config", [None, {"enabled": False}, {"effort": "none"}])
def test_nim_echo_requires_enabled_reasoning(reasoning_config):
    agent = _agent(reasoning_config=reasoning_config)
    assert not agent._needs_nim_tool_reasoning()
    assert not agent._needs_thinking_reasoning_pad()
    assert "reasoning_content" not in agent._build_assistant_message(
        _tool_message(), "tool_calls"
    )


def test_nim_model_on_aggregator_does_not_enable_echo():
    agent = _agent(
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        reasoning_config={"enabled": True},
    )
    assert not agent._needs_nim_tool_reasoning()
