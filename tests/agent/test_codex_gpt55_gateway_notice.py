"""Codex gpt-5.5 autoraise notice behavior.

The threshold override is useful, but gateway replay of the informational notice
can spam Telegram when agents are repeatedly constructed. Gateway users should
not receive that lifecycle hint as a per-agent status message.
"""

from unittest.mock import patch

from run_agent import AIAgent


def test_codex_gpt55_autoraise_does_not_create_gateway_warning():
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

    assert getattr(agent, "_compression_threshold_autoraised") == {"from": 0.50, "to": 0.85}
    compressor = getattr(agent, "context_compressor")
    assert compressor.threshold_tokens == int(compressor.context_length * 0.85)
    assert getattr(agent, "_compression_warning") is None
