"""Tests that plugin context engines get update_model() called during init.

Regression test for #9071 — plugin engines were never initialized with
context_length, causing the CLI status bar to show 'ctx --'.
"""

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


def test_plugin_engine_gets_context_length_on_init():
    """Plugin context engine should have context_length set during AIAgent init."""
    engine = _StubEngine()
    assert engine.context_length == 0  # ABC default before fix

    cfg = {"context": {"engine": "stub"}, "agent": {}}

    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("plugins.context_engine.load_context_engine", return_value=engine),
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    assert agent.context_compressor is engine
    assert engine.context_length == 204_800
    assert engine.threshold_tokens == int(204_800 * engine.threshold_percent)


def test_plugin_engine_update_model_args():
    """Verify update_model() receives model, context_length, base_url, api_key, provider."""
    engine = _StubEngine()
    engine.update_model = MagicMock()

    cfg = {"context": {"engine": "stub"}, "agent": {}}

    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("plugins.context_engine.load_context_engine", return_value=engine),
        patch("agent.model_metadata.get_model_context_length", return_value=131_072),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            model="openrouter/auto",
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    engine.update_model.assert_called_once()
    kw = engine.update_model.call_args.kwargs
    assert kw["context_length"] == 131_072
    assert "model" in kw
    assert "provider" in kw
    # Should NOT pass api_mode — the ABC doesn't accept it
    assert "api_mode" not in kw
