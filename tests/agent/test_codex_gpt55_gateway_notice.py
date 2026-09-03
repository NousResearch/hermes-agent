"""Codex GPT-5.x compression autoresolution gateway-notice behavior."""

from unittest.mock import patch

from run_agent import AIAgent


def test_codex_gpt55_autoraise_does_not_create_gateway_warning():
    """Gateway users must not receive an internal startup lifecycle notice."""
    cfg = {
        "compression": {
            "enabled": True,
            "threshold": 0.50,
            "codex_gpt55_autoraise": True,
        },
        "memory": {"memory_enabled": False, "user_profile_enabled": False},
        "tools": {},
    }

    with patch("hermes_cli.config.load_config", return_value=cfg):
        agent = AIAgent(
            model="gpt-5.5",
            provider="openai-codex",
            base_url="https://chatgpt.com/backend-api/codex",
            api_key="test-token",
            api_mode="codex_responses",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            enabled_toolsets=[],
            platform="telegram",
        )

    autoraise = agent._compression_threshold_autoraised
    assert autoraise["from"] == 0.50
    assert autoraise["to"] == 0.85
    assert agent.context_compressor.threshold_tokens == int(
        agent.context_compressor.context_length * 0.85
    )
    assert agent._compression_warning is None
