"""A /model switch onto a Bedrock Mantle (OpenAI-wire) model must rebuild the client with SigV4
auth. ``aws-sdk`` is a Hermes sentinel for IAM-chain auth; agent_init installed the SigV4
http_client but switch_model rebuilt bare ``{api_key, base_url}`` kwargs, so the SDK sent
``Authorization: Bearer aws-sdk`` and Mantle answered 401."""

from unittest.mock import MagicMock, patch

from agent.bedrock_adapter import BedrockOpenAISigV4Auth
from agent.context_compressor import ContextCompressor
from run_agent import AIAgent

MANTLE = "https://bedrock-mantle.us-east-1.api.aws/openai/v1"


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


@patch("agent.model_metadata.get_model_context_length", return_value=272_000)
def test_switch_to_bedrock_mantle_installs_sigv4_http_client(_ctx, monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAFAKEFAKEFAKEFAKE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fake-secret")
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    agent = _agent()

    agent.switch_model("openai.gpt-5.6-terra", "bedrock", api_key="aws-sdk", base_url=MANTLE, api_mode="codex_responses")

    auth = getattr(agent.client._client, "auth", None)
    assert isinstance(auth, BedrockOpenAISigV4Auth), f"switched client would send Bearer aws-sdk (auth={auth!r})"
    assert auth.region == "us-east-1"
