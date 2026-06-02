"""Tests for Bedrock Converse transport request construction."""

from types import SimpleNamespace

from agent.chat_completion_helpers import build_api_kwargs
from agent.transports.bedrock import BedrockTransport


_MESSAGES = [{"role": "user", "content": "please write a long report"}]


def _max_tokens_for(model: str, **params) -> int:
    kwargs = BedrockTransport().build_kwargs(
        model=model,
        messages=_MESSAGES,
        **params,
    )
    return kwargs["inferenceConfig"]["maxTokens"]


def test_bedrock_converse_defaults_sonnet_46_to_native_output_limit():
    assert _max_tokens_for("global.anthropic.claude-sonnet-4-6") == 64_000


def test_bedrock_converse_defaults_opus_48_to_native_output_limit():
    assert _max_tokens_for("anthropic.claude-opus-4-8") == 128_000


def test_bedrock_converse_preserves_explicit_max_tokens():
    assert _max_tokens_for(
        "global.anthropic.claude-sonnet-4-6",
        max_tokens=12_345,
    ) == 12_345


def test_bedrock_converse_keeps_existing_default_for_non_anthropic_models():
    assert _max_tokens_for("amazon.nova-pro-v1:0") == 4096


def test_bedrock_agent_build_api_kwargs_passes_unset_max_tokens_to_transport():
    agent = SimpleNamespace(
        api_mode="bedrock_converse",
        tools=[],
        model="global.anthropic.claude-sonnet-4-6",
        max_tokens=None,
        _bedrock_region="eu-west-1",
        _bedrock_guardrail_config=None,
    )
    agent._get_transport = BedrockTransport

    kwargs = build_api_kwargs(agent, _MESSAGES)

    assert kwargs["inferenceConfig"]["maxTokens"] == 64_000


def test_bedrock_agent_build_api_kwargs_keeps_non_anthropic_default_when_unset():
    agent = SimpleNamespace(
        api_mode="bedrock_converse",
        tools=[],
        model="amazon.nova-pro-v1:0",
        max_tokens=None,
        _bedrock_region="us-east-1",
        _bedrock_guardrail_config=None,
    )
    agent._get_transport = BedrockTransport

    kwargs = build_api_kwargs(agent, _MESSAGES)

    assert kwargs["inferenceConfig"]["maxTokens"] == 4096
