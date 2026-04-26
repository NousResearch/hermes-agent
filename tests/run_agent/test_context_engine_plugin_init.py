"""Tests for external context-engine plugin initialization in AIAgent."""

from unittest.mock import MagicMock, patch

from run_agent import AIAgent


class ExternalStubEngine:
    name = "lcm"
    threshold_tokens = 0
    context_length = 0
    compression_count = 0
    last_prompt_tokens = 0
    last_completion_tokens = 0
    last_total_tokens = 0

    def __init__(self):
        self.update_model_calls = []

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

    def on_session_reset(self):
        pass

    def compress(self, messages, current_tokens=None):
        return list(messages)


class LegacyEngine:
    name = "legacy"
    threshold_tokens = 999999
    compression_count = 0
    last_prompt_tokens = 0
    last_completion_tokens = 0

    def __init__(self):
        self.seen_current_tokens = None

    def compress(self, messages, current_tokens=None):
        self.seen_current_tokens = current_tokens
        return list(messages)


class TestContextEnginePluginInit:
    def test_default_compressor_ignores_external_plugin_engine(self):
        config = {
            "context": {"engine": "compressor"},
            "model": {"context_length": 131072},
            "compression": {"enabled": True},
        }

        with (
            patch("hermes_cli.config.load_config", return_value=config),
            patch("run_agent.get_tool_definitions", return_value=[]),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
            patch("hermes_cli.plugins.discover_plugins") as mock_discover_plugins,
            patch("hermes_cli.plugins.get_plugin_context_engine"),
        ):
            agent = AIAgent(
                api_key="test-key-1234567890",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )

        mock_discover_plugins.assert_not_called()
        assert agent.context_compressor.name == "compressor"
        assert "lcm_grep" not in agent.valid_tool_names

    def test_missing_external_engine_falls_back_to_builtin_compressor(self):
        config = {
            "context": {"engine": "lcm"},
            "model": {"context_length": 131072},
            "compression": {"enabled": True},
        }

        with (
            patch("hermes_cli.config.load_config", return_value=config),
            patch("run_agent.get_tool_definitions", return_value=[]),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
            patch("plugins.context_engine.load_context_engine", return_value=None),
            patch("hermes_cli.plugins.discover_plugins") as mock_discover_plugins,
            patch("hermes_cli.plugins.get_plugin_context_engine", return_value=None),
        ):
            agent = AIAgent(
                api_key="test-key-1234567890",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )

        mock_discover_plugins.assert_called_once()
        assert agent.context_compressor.name == "compressor"
        assert agent._context_engine_tool_names == set()

    def test_aiagent_initializes_external_context_engine_plugin(self):
        config = {
            "context": {"engine": "lcm"},
            "model": {"context_length": 131072},
            "compression": {"enabled": True},
        }
        engine = ExternalStubEngine()

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
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )

        mock_discover_plugins.assert_called_once()
        assert agent.context_compressor is engine
        assert engine.update_model_calls
        assert engine.context_length == 131072
        assert engine.threshold_tokens == int(131072 * 0.75)
        assert "lcm_grep" in agent.valid_tool_names
        assert "lcm_grep" in agent._context_engine_tool_names

    def test_compress_context_supports_legacy_engine_without_focus_topic(self):
        config = {
            "model": {"context_length": 131072},
            "compression": {"enabled": True},
        }

        with (
            patch("hermes_cli.config.load_config", return_value=config),
            patch("run_agent.get_tool_definitions", return_value=[]),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
        ):
            agent = AIAgent(
                api_key="test-key-1234567890",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )

        agent.flush_memories = MagicMock()
        engine = LegacyEngine()
        agent.context_compressor = engine

        messages = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
        ]

        compressed, _ = agent._compress_context(
            messages,
            "",
            approx_tokens=1234,
            focus_topic="database schema",
        )

        assert compressed == messages
        assert engine.seen_current_tokens == 1234
