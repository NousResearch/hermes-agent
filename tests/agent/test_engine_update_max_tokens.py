"""Regression test for #115517 — external update_model() must forward max_tokens.

agent/agent_init.py called the context engine's update_model() without max_tokens
while ContextCompressor.update_model already accepts it; the ContextEngine ABC
rejected the kwarg, so plugin engines never saw the output reservation.
"""

import inspect
from unittest.mock import MagicMock, patch

from agent.context_engine import ContextEngine


class _StubEngine(ContextEngine):
    """Minimal concrete context engine for testing."""

    @property
    def name(self) -> str:
        return "stub"

    def update_from_response(self, usage):
        pass

    def should_compress(self, prompt_tokens=None):
        return False

    def compress(self, messages, current_tokens=None):
        return messages


def test_context_engine_update_model_accepts_max_tokens():
    """ABC signature carries max_tokens and the base impl records it."""
    assert "max_tokens" in inspect.signature(ContextEngine.update_model).parameters
    engine = _StubEngine()
    engine.update_model(model="m", context_length=1000, max_tokens=123)
    assert engine.max_tokens == 123


def test_plugin_engine_update_model_receives_max_tokens():
    """Init forwards agent.max_tokens to the plugin engine's update_model()."""
    engine = _StubEngine()
    engine.update_model = MagicMock()

    cfg = {"context": {"engine": "stub"}, "agent": {}}

    with (
        patch("hermes_cli.config.load_config", return_value=cfg), patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("plugins.context_engine.load_context_engine", return_value=engine),
        patch("agent.model_metadata.get_model_context_length", return_value=131_072),
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        from run_agent import AIAgent

        AIAgent(
            model="openrouter/auto",
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            max_tokens=4096,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    engine.update_model.assert_called_once()
    assert engine.update_model.call_args.kwargs.get("max_tokens") == 4096
