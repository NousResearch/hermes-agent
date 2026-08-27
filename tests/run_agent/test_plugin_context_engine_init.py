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


class _ToolEngine(_StubEngine):
    def get_tool_schemas(self):
        return [
            {
                "name": "stub_recover",
                "description": "Recover context from the stub engine.",
                "parameters": {"type": "object", "properties": {}},
            }
        ]


def test_plugin_engine_gets_context_length_on_init():
    """Plugin context engine should have context_length set during AIAgent init."""
    engine = _StubEngine()
    assert engine.context_length == 0  # ABC default before fix

    cfg = {"context": {"engine": "stub"}, "agent": {}}

    with (
        patch("hermes_cli.config.load_config", return_value=cfg), patch("hermes_cli.config.load_config_readonly", return_value=cfg),
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


def test_active_context_engine_tools_survive_explicit_platform_toolsets():
    """LCM-style recovery tools must survive saved `hermes tools` lists."""
    engine = _ToolEngine()
    cfg = {
        "context": {"engine": "stub"},
        "platform_toolsets": {"cli": ["web", "terminal"]},
        "agent": {},
    }

    from hermes_cli.tools_config import _get_platform_tools

    enabled_toolsets = _get_platform_tools(cfg, "cli", include_default_mcp_servers=False)
    assert "context_engine" in enabled_toolsets

    with (
        patch("hermes_cli.config.load_config", return_value=cfg), patch("hermes_cli.config.load_config_readonly", return_value=cfg),
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
            enabled_toolsets=sorted(enabled_toolsets),
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    assert "stub_recover" in getattr(agent, "valid_tool_names", set())
    assert "stub_recover" in {
        tool.get("function", {}).get("name")
        for tool in getattr(agent, "tools", [])
    }


def test_plugin_engine_update_model_args():
    """Verify update_model() receives model, context_length, base_url, api_key, provider."""
    engine = _StubEngine()
    engine.update_model = MagicMock()

    cfg = {"context": {"engine": "stub"}, "agent": {}}

    with (
        patch("hermes_cli.config.load_config", return_value=cfg), patch("hermes_cli.config.load_config_readonly", return_value=cfg),
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
    assert "api_mode" in kw


def _codex_agent_kwargs():
    return dict(
        model="gpt-5.5",
        provider="openai-codex",
        api_key="test-key-1234567890",
        base_url="https://chatgpt.com/backend-api/codex",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )


def test_codex_gpt55_autoraise_notice_armed_for_plugin_engine():
    """Codex gpt-5.5 autoraise notice stays armed when an external engine is active.

    The host forwards the resolved compression threshold (config value or
    Codex gpt-5.x autoraise) to engines whose update_model accepts
    threshold_percent, so the notice is accurate for plugin engines instead of
    announcing a change that did not apply (#44439). Engines that do not
    accept the kwarg keep their own resolved policy.
    """
    engine = _StubEngine()
    cfg = {
        "context": {"engine": "stub"},
        "compression": {"enabled": True, "threshold": 0.75},
        "agent": {},
    }

    with (
        patch("hermes_cli.config.load_config", return_value=cfg), patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("plugins.context_engine.load_context_engine", return_value=engine),
        patch("agent.model_metadata.get_model_context_length", return_value=272_000),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(**_codex_agent_kwargs())

    assert agent.context_compressor is engine
    # The autoraise notice re-arms for plugin engines: the resolved host
    # threshold (0.75 config → 0.85 Codex gpt-5.5 autoraise) is forwarded to
    # engines that accept threshold_percent, so the announced change applies.
    assert agent._compression_threshold_autoraised == {
        "model": "gpt-5.5",
        "from": 0.75,
        "to": 0.85,
    }
    assert agent._compression_warning is not None
    assert "85%" in agent._compression_warning
    # The stub's update_model does not accept threshold_percent, so the engine
    # keeps its own resolved policy (config 0.75, not the autoraised 0.85).
    assert engine.threshold_percent == 0.75


def test_codex_gpt55_autoraise_still_applies_to_builtin_compressor():
    """Stock built-in compressor keeps the 50% → 85% Codex gpt-5.5 autoraise."""
    cfg = {
        "compression": {"enabled": True, "threshold": 0.50},
        "agent": {},
    }

    with (
        patch("hermes_cli.config.load_config", return_value=cfg), patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("agent.context_compressor.get_model_context_length", return_value=272_000),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(**_codex_agent_kwargs())

    assert agent._compression_threshold_autoraised == {"model": "gpt-5.5", "from": 0.50, "to": 0.85}
    assert agent.context_compressor.threshold_percent == 0.85
    # Gateway parity: the notice is stashed for replay on turn 1.
    assert agent._compression_warning and "85%" in agent._compression_warning


