"""Anthropic client rebuilds must preserve endpoint-scoped extra headers."""

from unittest.mock import patch

from run_agent import AIAgent


def test_rebuild_anthropic_client_reapplies_configured_extra_headers():
    base_url = "https://proxy.example.com/anthropic"
    config = {
        "custom_providers": [
            {
                "name": "anthropic-proxy",
                "base_url": base_url,
                "api_mode": "anthropic_messages",
                "extra_headers": {"User-Agent": "hermes-rebuild/1.0"},
            }
        ]
    }

    agent = AIAgent.__new__(AIAgent)
    agent.provider = "anthropic-proxy"
    agent.model = "claude-compatible"
    agent._anthropic_api_key = "sk-test"
    agent._anthropic_base_url = base_url
    agent._oauth_1m_beta_disabled = False

    with (
        patch("agent.anthropic_adapter._anthropic_sdk") as mock_sdk,
        patch("hermes_cli.config.load_config", return_value=config),
        patch("run_agent.get_provider_request_timeout", return_value=42.0),
    ):
        agent._rebuild_anthropic_client()

    kwargs = mock_sdk.Anthropic.call_args.kwargs
    assert kwargs["default_headers"]["User-Agent"] == "hermes-rebuild/1.0"
    assert "interleaved-thinking-2025-05-14" in kwargs["default_headers"]["anthropic-beta"]
    assert kwargs["timeout"].read == 42.0
