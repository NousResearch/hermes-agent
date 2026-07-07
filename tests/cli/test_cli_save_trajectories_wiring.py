"""Regression test: save_trajectories from config.yml must reach the agent.

Setting ``agent.save_trajectories: true`` in config.yml had no effect because
the CLI never passed the value to AIAgent. Two things must hold:

1. HermesCLI reads the value from CLI_CONFIG["agent"] and stores it on self.
2. CLIAgentSetupMixin._init_agent() passes it to AIAgent.
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch


def _make_cli(agent_cfg: dict = None, **kwargs):
    """Create a HermesCLI instance with a controlled agent config section."""
    _clean_config = {
        "model": {
            "default": "anthropic/claude-opus-4.6",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "auto",
        },
        "display": {"compact": False, "tool_progress": "all"},
        "agent": agent_cfg or {},
        "terminal": {"env_type": "local"},
    }
    clean_env = {"LLM_MODEL": "", "HERMES_MAX_ITERATIONS": ""}
    prompt_toolkit_stubs = {
        "prompt_toolkit": MagicMock(),
        "prompt_toolkit.history": MagicMock(),
        "prompt_toolkit.styles": MagicMock(),
        "prompt_toolkit.patch_stdout": MagicMock(),
        "prompt_toolkit.application": MagicMock(),
        "prompt_toolkit.layout": MagicMock(),
        "prompt_toolkit.layout.processors": MagicMock(),
        "prompt_toolkit.filters": MagicMock(),
        "prompt_toolkit.layout.dimension": MagicMock(),
        "prompt_toolkit.layout.menus": MagicMock(),
        "prompt_toolkit.widgets": MagicMock(),
        "prompt_toolkit.key_binding": MagicMock(),
        "prompt_toolkit.completion": MagicMock(),
        "prompt_toolkit.formatted_text": MagicMock(),
        "prompt_toolkit.auto_suggest": MagicMock(),
    }
    with patch.dict(sys.modules, prompt_toolkit_stubs), \
         patch.dict("os.environ", clean_env, clear=False):
        import cli as _cli_mod
        _cli_mod = importlib.reload(_cli_mod)
        with patch.object(_cli_mod, "get_tool_definitions", return_value=[]), \
             patch.dict(_cli_mod.__dict__, {"CLI_CONFIG": _clean_config}):
            return _cli_mod.HermesCLI(**kwargs)


class TestSaveTrajectoriesConfig:
    def test_true_when_set_in_config(self):
        """HermesCLI.__init__ must read save_trajectories from the agent config section."""
        cli = _make_cli(agent_cfg={"save_trajectories": True})
        assert cli.save_trajectories is True

    def test_false_by_default(self):
        """save_trajectories defaults to False when absent from config."""
        cli = _make_cli(agent_cfg={})
        assert cli.save_trajectories is False


class TestSaveTrajectoriesWiring:
    def test_init_agent_passes_save_trajectories_to_ai_agent(self):
        """_init_agent must forward save_trajectories=True from self to AIAgent.

        The bug: all three execution paths (CLI, gateway, TUI) constructed
        AIAgent without passing save_trajectories, so agent.save_trajectories
        stayed False regardless of config.yml.
        """
        cli_instance = _make_cli(agent_cfg={"save_trajectories": True})
        assert cli_instance.save_trajectories is True

        # Prevent _init_agent from opening a real SQLite DB.
        cli_instance._session_db = MagicMock()

        import cli as cli_mod

        mock_agent_cls = MagicMock()
        with patch.object(cli_mod, "AIAgent", mock_agent_cls), \
             patch.object(cli_mod, "_prepare_deferred_agent_startup"), \
             patch.object(cli_instance, "_install_tool_callbacks"), \
             patch.object(cli_instance, "_ensure_tirith_security"), \
             patch.object(cli_instance, "_ensure_runtime_credentials", return_value=True), \
             patch("hermes_cli.mcp_startup.wait_for_mcp_discovery"):
            cli_instance._init_agent()

        mock_agent_cls.assert_called_once()
        _, kwargs = mock_agent_cls.call_args
        assert kwargs.get("save_trajectories") is True
