"""User context-engine discovery regressions for #61839."""

from __future__ import annotations

import sys
import textwrap
from unittest.mock import patch

import yaml


_ENGINE_INIT = textwrap.dedent(
    '''\
    from pathlib import Path
    from agent.context_engine import ContextEngine

    _marker = Path(__file__).with_name("executions.txt")
    _marker.write_text(
        (_marker.read_text(encoding="utf-8") if _marker.exists() else "") + "import\\n",
        encoding="utf-8",
    )

    class TestEngine(ContextEngine):
        @property
        def name(self):
            return "context_chunking"

        def update_from_response(self, usage):
            pass

        def should_compress(self, prompt_tokens=None):
            return False

        def compress(self, messages, current_tokens=None, focus_topic=None):
            return messages

    def register(ctx):
        _marker.write_text(
            _marker.read_text(encoding="utf-8") + "register\\n",
            encoding="utf-8",
        )
        ctx.register_context_engine(TestEngine())
    '''
)


def _write_user_engine(hermes_home, *, enabled: bool):
    plugin_dir = hermes_home / "plugins" / "context_chunking"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "context_chunking",
                "version": "0.1.0",
                "description": "Test context engine plugin",
            }
        ),
        encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(_ENGINE_INIT, encoding="utf-8")
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "context": {"engine": "context_chunking"},
                "plugins": {
                    "enabled": ["context_chunking"] if enabled else [],
                    "disabled": [] if enabled else ["context_chunking"],
                },
            }
        ),
        encoding="utf-8",
    )
    return plugin_dir


def _install_stale_manager(monkeypatch):
    import hermes_cli.plugins as plugins_mod

    manager = plugins_mod.PluginManager()
    manager._discovered = True
    monkeypatch.setattr(plugins_mod, "_plugin_manager", manager)
    return manager


def test_first_agent_init_refreshes_stale_context_engine_discovery_once(
    tmp_path, monkeypatch, caplog
):
    """Agent init recovers when discovery finished before the plugin appeared."""
    hermes_home = tmp_path / "hermes"
    plugin_dir = _write_user_engine(hermes_home, enabled=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    manager = _install_stale_manager(monkeypatch)

    with (
        caplog.at_level("WARNING", logger="run_agent"),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch(
            "agent.model_metadata.get_model_context_length",
            return_value=128_000,
        ),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    assert manager._context_engine is not None
    assert agent.context_compressor.name == "context_chunking"
    assert not any(
        "context_chunking" in record.getMessage()
        and "not found" in record.getMessage()
        for record in caplog.records
    )
    assert (plugin_dir / "executions.txt").read_text(encoding="utf-8").splitlines() == [
        "import",
        "register",
    ]

    # A second lookup returns the registered instance without importing or
    # registering the plugin again.
    from hermes_cli.plugins import get_plugin_context_engine

    assert get_plugin_context_engine() is manager._context_engine
    assert (plugin_dir / "executions.txt").read_text(encoding="utf-8").splitlines() == [
        "import",
        "register",
    ]

    sys.modules.pop("hermes_plugins.context_chunking", None)


def test_stale_discovery_does_not_execute_disabled_context_engine(
    tmp_path, monkeypatch
):
    """The stale refresh remains subject to PluginManager's config gates."""
    hermes_home = tmp_path / "hermes"
    plugin_dir = _write_user_engine(hermes_home, enabled=False)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    manager = _install_stale_manager(monkeypatch)

    from hermes_cli.plugins import get_plugin_context_engine

    assert get_plugin_context_engine() is None
    assert manager._context_engine is None
    assert not (plugin_dir / "executions.txt").exists()
    assert manager._plugins["context_chunking"].error == "disabled via config"
