"""Tests that plugin context engines get update_model() called during init.

Regression test for #9071 — plugin engines were never initialized with
context_length, causing the CLI status bar to show 'ctx --'.
"""

from unittest.mock import patch

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
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
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
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
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


def test_codex_gpt55_autoraise_suppressed_for_plugin_engine():
    """Codex gpt-5.5 autoraise must not fire when an external engine is active.

    Regression test for #44439 — the host compression threshold (including
    the 0.85 autoraise) never reaches a plugin context engine, so the notice
    announced a change that did not apply.
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
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(**_codex_agent_kwargs())

    assert agent.context_compressor is engine
    assert agent._compression_threshold_autoraised is None
    assert agent._compression_warning is None
    # The engine's own policy is untouched by the host threshold.
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
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(**_codex_agent_kwargs())

    assert agent._compression_threshold_autoraised == {"model": "gpt-5.5", "from": 0.50, "to": 0.85}
    assert agent.context_compressor.threshold_percent == 0.85
    # Gateway parity: the notice is stashed for replay on turn 1.
    assert agent._compression_warning


class _ConditionalRequiredEngine(_StubEngine):
    """Engine whose tool schema carries a top-level ``allOf`` conditional-required hint —
    the exact shape that 400s strict backends when it reaches the wire unsanitized."""

    def get_tool_schemas(self):
        return [
            {
                "name": "lcm_compile_evidence",
                "description": "Compile evidence.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "mode": {"type": "string", "enum": ["review", "proposal"]},
                        "proposal": {"type": "string"},
                    },
                    "required": ["mode"],
                    "allOf": [
                        {
                            "if": {"properties": {"mode": {"const": "proposal"}}},
                            "then": {"required": ["proposal"]},
                        }
                    ],
                },
            }
        ]


def test_context_engine_tool_top_level_combinator_sanitized():
    """Late-injected engine schemas must get the same top-level combinator strip as
    registry tools (#134383): model_tools' sanitize_tool_schemas pass has already run
    when these are appended, so without it one conditional-required allOf poisons the
    whole request — strict backends 400 with "input_schema does not support oneOf, allOf,
    or anyOf at the top level"."""
    import uuid

    engine = _ConditionalRequiredEngine()
    cfg = {"context": {"engine": "stub"}, "agent": {}}

    with (
        patch("hermes_cli.config.load_config", return_value=cfg), patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("plugins.context_engine.load_context_engine", return_value=engine),
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key=f"test-key-{uuid.uuid4().hex[:10]}",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    injected = [
        tool
        for tool in agent.tools
        if tool.get("function", {}).get("name") == "lcm_compile_evidence"
    ]
    assert injected, "engine tool never reached the tool surface"
    params = injected[0]["function"]["parameters"]
    for combinator in ("allOf", "anyOf", "oneOf"):
        assert combinator not in params, (
            f"top-level {combinator} survived the injection"
        )
    # Only the TOP level is stripped: the legitimate parts survive untouched.
    assert params["required"] == ["mode"]
    assert params["properties"]["mode"]["enum"] == ["review", "proposal"]
