"""Every non-startup Bedrock client rebuild (/model switch, fallback-to-Bedrock, fallback restore)
must land on the same wire startup uses: AnthropicBedrock SDK for Claude, boto3-direct for
Converse, with region and guardrail state derived from the active endpoint. Before the fix these
paths handed the ``aws-sdk`` sentinel to the generic Anthropic/OpenAI builders (401) and left
``_bedrock_region`` unset (requests to the wrong region; guardrails dropped)."""

from unittest.mock import MagicMock, patch

import pytest

from agent.bedrock_adapter import BedrockOpenAISigV4Auth
from agent.context_compressor import ContextCompressor
from run_agent import AIAgent

MANTLE = "https://bedrock-mantle.us-east-1.api.aws/openai/v1"
RUNTIME_EU = "https://bedrock-runtime.eu-west-1.amazonaws.com"


def _agent() -> AIAgent:
    agent = AIAgent.__new__(AIAgent)
    agent.model, agent.provider = "claude-opus-4.8", "anthropic"
    agent.base_url, agent.api_key, agent.api_mode = "https://api.anthropic.com", "sk-ant", "anthropic_messages"
    agent.client, agent._anthropic_client, agent._client_kwargs = None, MagicMock(), {}
    agent.quiet_mode, agent._config_context_length, agent._primary_runtime = True, None, {}
    agent.context_compressor = ContextCompressor(
        model=agent.model, threshold_percent=0.5, base_url=agent.base_url, api_key="sk-ant",
        provider="anthropic", quiet_mode=True, config_context_length=None,
    )
    return agent


@pytest.fixture
def aws_env(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAFAKEFAKEFAKEFAKE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fake-secret")
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)


@patch("agent.model_metadata.get_model_context_length", return_value=272_000)
def test_switch_to_bedrock_mantle_installs_sigv4_http_client(_ctx, aws_env):
    agent = _agent()

    agent.switch_model("openai.gpt-5.6-terra", "bedrock", api_key="aws-sdk", base_url=MANTLE, api_mode="codex_responses")

    auth = getattr(agent.client._client, "auth", None)
    assert isinstance(auth, BedrockOpenAISigV4Auth), f"switched client would send Bearer aws-sdk (auth={auth!r})"
    assert auth.region == "us-east-1"


@pytest.mark.parametrize(
    "model, api_mode, anthropic_client_type",
    [("us.anthropic.claude-opus-4-6-v1", "anthropic_messages", "AnthropicBedrock"),
     ("us.amazon.nova-pro-v1:0", "bedrock_converse", "NoneType")],
)
@patch("agent.model_metadata.get_model_context_length", return_value=200_000)
def test_switch_to_bedrock_runtime_wires_binds_region_and_sdk(_ctx, aws_env, model, api_mode, anthropic_client_type):
    agent = _agent()

    agent.switch_model(model, "bedrock", api_key="aws-sdk", base_url=RUNTIME_EU, api_mode=api_mode)

    assert agent.client is None, "no OpenAI client belongs on a bedrock-runtime wire"
    assert type(agent._anthropic_client).__name__ == anthropic_client_type
    assert agent._bedrock_region == "eu-west-1"
    assert hasattr(agent, "_bedrock_guardrail_config")


@pytest.mark.parametrize("api_mode", ["anthropic_messages", "bedrock_converse"])
def test_fallback_to_bedrock_binds_runtime_not_generic_client(aws_env, api_mode):
    from agent.client_lifecycle import _swap_fallback_clients
    agent = _agent()
    fb_client = MagicMock(base_url=RUNTIME_EU, api_key="aws-sdk")
    agent.model, agent.provider, agent.base_url, agent.api_mode = "us.amazon.nova-pro-v1:0", "bedrock", RUNTIME_EU, api_mode

    _swap_fallback_clients(agent, fb_client, "bedrock", agent.model, RUNTIME_EU, api_mode)

    assert agent.client is None and agent._client_kwargs == {}
    assert agent._bedrock_region == "eu-west-1"
    assert type(agent._anthropic_client).__name__ == ("AnthropicBedrock" if api_mode == "anthropic_messages" else "NoneType")
