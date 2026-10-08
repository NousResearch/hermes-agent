"""Bedrock Converse takes part in the truncation-retry output ladder like every other wire: the
one-shot boosted cap reaches ``inferenceConfig.maxTokens``, the cap reader finds it there, and the
ladder knows the Claude ceiling so it never asks Bedrock for more than the model allows."""
from types import SimpleNamespace

from agent.chat_completion_helpers import _build_bedrock_kwargs
from agent.transports import get_transport
from agent.turn_truncation import _model_output_limit, boosted_output_cap
from run_agent import AIAgent

_MSGS = [{"role": "user", "content": "hi"}]


def _bedrock_agent(model="global.anthropic.claude-sonnet-5-5", max_tokens=None, ephemeral=None):
    import agent.transports.bedrock  # noqa: F401
    transport = get_transport("bedrock_converse")
    return SimpleNamespace(
        model=model, api_mode="bedrock_converse", max_tokens=max_tokens,
        _ephemeral_max_output_tokens=ephemeral, _get_transport=lambda: transport,
    )


def test_converse_request_consumes_the_one_shot_boosted_cap():
    agent = _bedrock_agent(ephemeral=16_384)
    assert _build_bedrock_kwargs(agent, _MSGS, None)["inferenceConfig"]["maxTokens"] == 16_384
    assert agent._ephemeral_max_output_tokens is None  # consumed exactly once
    # Back to the default: no configured cap means the model ceiling, not Bedrock's 4096.
    assert _build_bedrock_kwargs(agent, _MSGS, None)["inferenceConfig"]["maxTokens"] == 128_000


def test_configured_cap_still_wins_without_a_boost():
    agent = _bedrock_agent(max_tokens=32_768)
    assert _build_bedrock_kwargs(agent, _MSGS, None)["inferenceConfig"]["maxTokens"] == 32_768


def test_cap_reader_finds_converse_max_tokens():
    assert AIAgent._requested_output_cap_from_api_kwargs({"inferenceConfig": {"maxTokens": 4096}}) == 4096
    assert AIAgent._requested_output_cap_from_api_kwargs({"inferenceConfig": {}}) is None
    assert AIAgent._requested_output_cap_from_api_kwargs({"max_tokens": 42}) == 42


def test_ladder_knows_the_claude_ceiling_on_converse():
    assert _model_output_limit(_bedrock_agent()) == 128_000
    assert _model_output_limit(_bedrock_agent(model="amazon.nova-pro-v1:0")) is None
    # Already at the ceiling: a retry must not ask Bedrock for more than the model allows.
    assert boosted_output_cap(_bedrock_agent(), 128_000, 1) == 128_000
