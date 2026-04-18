from unittest.mock import patch


def test_aiagent_uses_explicit_compression_threshold_tokens_from_config():
    cfg = {
        "model": {
            "default": "gpt5.4",
            "provider": "custom",
            "base_url": "http://localhost:4000/v1",
            "context_length": 200000,
        },
        "compression": {
            "enabled": True,
            "threshold": 0.50,
            "threshold_tokens": 185000,
            "target_ratio": 0.20,
            "protect_last_n": 20,
        },
        "agent": {},
    }

    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("agent.context_compressor.get_model_context_length", return_value=200_000),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            model="gpt5.4",
            api_key="test-key-1234567890",
            base_url="http://localhost:4000/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    assert agent.context_compressor.threshold_tokens == 185_000
    assert agent.context_compressor.tail_token_budget == 37_000
