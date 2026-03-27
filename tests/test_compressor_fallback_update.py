"""Tests that _try_activate_fallback updates the context compressor."""

from unittest.mock import MagicMock, patch

from run_agent import AIAgent
from agent.context_compressor import ContextCompressor


def _make_agent_with_compressor() -> AIAgent:
    """Build a minimal AIAgent with a context_compressor, skipping __init__."""
    agent = AIAgent.__new__(AIAgent)

    # Primary model settings
    agent.model = "primary-model"
    agent.provider = "openrouter"
    agent.base_url = "https://openrouter.ai/api/v1"
    agent.api_key = "sk-primary"
    agent.api_mode = "chat_completions"
    agent.client = MagicMock()
    agent.quiet_mode = True

    # Fallback config
    agent._fallback_activated = False
    agent._fallback_model = {
        "provider": "openai",
        "model": "gpt-4o",
    }

    # Context compressor with primary model values
    compressor = ContextCompressor(
        model="primary-model",
        threshold_percent=0.50,
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-primary",
        provider="openrouter",
        quiet_mode=True,
    )
    agent.context_compressor = compressor

    return agent


@patch("agent.auxiliary_client.resolve_provider_client")
@patch("agent.model_metadata.get_model_context_length", return_value=128_000)
def test_compressor_updated_on_fallback(mock_ctx_len, mock_resolve):
    """After fallback activation, the compressor must reflect the fallback model."""
    agent = _make_agent_with_compressor()

    assert agent.context_compressor.model == "primary-model"

    fb_client = MagicMock()
    fb_client.base_url = "https://api.openai.com/v1"
    fb_client.api_key = "sk-fallback"
    mock_resolve.return_value = (fb_client, None)

    agent._is_direct_openai_url = lambda url: "api.openai.com" in url
    agent._emit_status = lambda msg: None

    result = agent._try_activate_fallback()

    assert result is True
    assert agent._fallback_activated is True

    c = agent.context_compressor
    assert c.model == "gpt-4o"
    assert c.base_url == "https://api.openai.com/v1"
    assert c.api_key == "sk-fallback"
    assert c.provider == "openai"
    assert c.context_length == 128_000
    assert c.threshold_tokens == int(128_000 * c.threshold_percent)


@patch("agent.auxiliary_client.resolve_provider_client")
@patch("agent.model_metadata.get_model_context_length", return_value=128_000)
def test_compressor_not_present_does_not_crash(mock_ctx_len, mock_resolve):
    """If the agent has no compressor, fallback should still succeed."""
    agent = _make_agent_with_compressor()
    agent.context_compressor = None

    fb_client = MagicMock()
    fb_client.base_url = "https://api.openai.com/v1"
    fb_client.api_key = "sk-fallback"
    mock_resolve.return_value = (fb_client, None)

    agent._is_direct_openai_url = lambda url: "api.openai.com" in url
    agent._emit_status = lambda msg: None

    result = agent._try_activate_fallback()
    assert result is True
