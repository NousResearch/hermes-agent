"""Plugin registration on the exact branch head (H05, part 1).

Covers the three registration surfaces: repo-directory discovery, the
``register(ctx)`` entry point (including the repo-shipped collector), and the
host's engine-selection path in ``AIAgent.__init__`` with the *real* loader.
"""

from __future__ import annotations

import importlib
from unittest.mock import patch

from agent.context_engine import ContextEngine
from plugins.context_engine import _EngineCollector, discover_context_engines, load_context_engine
from plugins.context_engine.lcm.engine import LCMEngine

_LCM = importlib.import_module("plugins.context_engine.lcm")


class _FakeCtx:
    def __init__(self) -> None:
        self.engine = None
        self.commands: list = []
        self.skills: list = []

    def register_context_engine(self, engine):
        self.engine = engine

    def register_command(self, name, handler, description="", args_hint=""):
        self.commands.append((name, handler, args_hint))

    def register_skill(self, name, path, description="", frontmatter=None):
        self.skills.append((name, path))


def test_discovery_lists_the_lcm_engine_as_available():
    engines = {name: (desc, available) for name, desc, available in discover_context_engines()}
    assert "lcm" in engines, "the plugin directory is discovered by name"
    description, available = engines["lcm"]
    assert description, "plugin.yaml supplies the description shown by `hermes plugins`"
    assert available is True


def test_loader_returns_a_concrete_context_engine():
    engine = load_context_engine("lcm")
    assert engine is not None, "the repo-shipped engine loads without user installation"
    assert isinstance(engine, LCMEngine)
    assert isinstance(engine, ContextEngine)
    assert engine.name == "lcm"


def test_register_entry_point_wires_engine_command_and_skill():
    ctx = _FakeCtx()
    _LCM.register(ctx)

    assert isinstance(ctx.engine, LCMEngine)
    assert [name for name, _handler, _hint in ctx.commands] == ["lcm"]
    assert ctx.skills and ctx.skills[0][0] == "recall"
    assert ctx.skills[0][1].name == "SKILL.md"
    assert ctx.skills[0][1].exists()

    handler = ctx.commands[0][1]
    assert "LCM engine: lcm" in handler("status")


def test_register_survives_a_context_without_optional_methods():
    """The repo-shipped collector implements fewer methods than a full ctx."""

    class MinimalCtx:
        def __init__(self):
            self.engine = None

        def register_context_engine(self, engine):
            self.engine = engine

    ctx = MinimalCtx()
    _LCM.register(ctx)
    assert isinstance(ctx.engine, LCMEngine)


def test_repo_collector_records_the_engine_and_forwards_the_command():
    from hermes_cli.plugins import get_plugin_manager

    collector = _EngineCollector(engine_name="lcm")
    manager = get_plugin_manager()
    try:
        _LCM.register(collector)
        assert isinstance(collector.engine, LCMEngine)
        assert "lcm" in manager._plugin_commands
        assert manager._plugin_commands["lcm"]["plugin"] == "context-engine:lcm"
    finally:
        manager._plugin_commands.pop("lcm", None)


def test_repeated_registration_is_idempotent_and_quiet(caplog):
    """Discovery and engine loading both call register(). That must not conflict."""
    import logging

    from hermes_cli.plugins import get_plugin_manager

    manager = get_plugin_manager()
    try:
        _LCM.register(_EngineCollector(engine_name="lcm"))
        entry = manager._plugin_commands["lcm"]
        assert entry["plugin"] == "context-engine:lcm"
        handler = entry["handler"]

        with caplog.at_level(logging.WARNING, logger="plugins.context_engine"):
            _LCM.register(_EngineCollector(engine_name="lcm"))

        assert manager._plugin_commands["lcm"]["handler"] is handler, "not replaced"
        conflicts = [
            record
            for record in caplog.records
            if "already registered by a plugin" in record.getMessage()
        ]
        assert conflicts == [], "a repeat registration of our own command is not a conflict"
    finally:
        manager._plugin_commands.pop("lcm", None)


def test_host_selects_the_engine_from_config_and_exposes_its_tools():
    """End-to-end selection with the real loader. No engine patching.

    Proves the config-driven path on this head: ``context.engine: lcm`` makes
    the agent adopt the plugin engine, hand it the model's context length, and
    inject the engine's tools into the toolset.
    """
    cfg = {"context": {"engine": "lcm"}, "agent": {}}

    from hermes_cli.tools_config import _get_platform_tools

    enabled = _get_platform_tools(cfg, "cli", include_default_mcp_servers=False)
    assert "context_engine" in enabled

    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            enabled_toolsets=sorted(enabled),
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    assert isinstance(agent.context_compressor, LCMEngine)
    assert agent.context_compressor.name == "lcm"
    assert agent.context_compressor.context_length == 204_800
    assert agent.context_compressor.threshold_tokens > 0

    tool_names = {
        tool.get("function", {}).get("name")
        for tool in getattr(agent, "tools", [])
    }
    assert {"lcm_search", "lcm_expand", "lcm_page", "lcm_status"} <= tool_names
    assert {"lcm_search", "lcm_expand", "lcm_page", "lcm_status"} <= set(
        getattr(agent, "valid_tool_names", set())
    )
