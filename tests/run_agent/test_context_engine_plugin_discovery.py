"""Tests for external context engine plugin discovery fallback."""

from unittest.mock import patch

from run_agent import AIAgent


class StubPluginEngine:
    name = "lcm"
    threshold_tokens = 0
    context_length = 0
    compression_count = 0
    last_prompt_tokens = 0
    last_completion_tokens = 0
    last_total_tokens = 0

    def __init__(self):
        self.update_model_calls = []
        self.started = None

    def update_model(self, **kwargs):
        self.update_model_calls.append(kwargs)
        self.context_length = kwargs["context_length"]
        self.threshold_tokens = int(kwargs["context_length"] * 0.75)

    def get_tool_schemas(self):
        return [
            {
                "name": "lcm_grep",
                "description": "Search the external context engine",
                "parameters": {"type": "object", "properties": {}},
            }
        ]

    def on_session_start(self, session_id, **kwargs):
        self.started = (session_id, kwargs)

    def compress(self, messages, current_tokens=None, focus_topic=None):
        return list(messages)


def test_aiagent_discovers_external_plugin_engine_before_fallback():
    config = {
        "context": {"engine": "lcm"},
        "model": {"context_length": 131072},
        "compression": {"enabled": True},
    }
    engine = StubPluginEngine()

    with (
        patch("hermes_cli.config.load_config", return_value=config),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("plugins.context_engine.load_context_engine", return_value=None),
        patch("hermes_cli.plugins.discover_plugins") as mock_discover_plugins,
        patch("hermes_cli.plugins.get_plugin_context_engine", return_value=engine),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://llm.example.com/v1",
            provider="custom",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    mock_discover_plugins.assert_called_once()
    assert agent.context_compressor is engine
    assert engine.update_model_calls
    assert engine.context_length == 131072
    assert "lcm_grep" in agent.valid_tool_names
