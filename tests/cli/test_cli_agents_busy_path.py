"""Regression tests for classic-CLI mid-run /agents (alias /tasks) dispatch.

Background
----------
/agents (and its /tasks alias) is the in-flight introspection command —
users type it precisely because they want to monitor delegations *while*
the agent loop is running.  Without the inline-dispatch path the keystroke
flows through ``_pending_input`` to ``process_loop``, which is blocked
inside ``self.chat()`` for the duration of the run.  The command therefore
sits silently in the queue and never executes until the whole delegation
chain finishes — from the user's perspective "nothing happens" (#32477).

The fix mirrors the existing /steer pattern: detect the command on the UI
thread inside ``handle_enter`` and call ``process_command`` directly,
since ``_handle_agents_command`` is read-only and ``_cprint`` already
routes thread-unsafe output through ``run_in_terminal``.
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch


def _make_cli():
    """Create a HermesCLI instance with prompt_toolkit stubbed out."""
    _clean_config = {
        "model": {
            "default": "anthropic/claude-opus-4.6",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "auto",
        },
        "display": {"compact": False, "tool_progress": "all"},
        "agent": {},
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
    with patch.dict(sys.modules, prompt_toolkit_stubs), patch.dict(
        "os.environ", clean_env, clear=False
    ):
        import cli as _cli_mod

        _cli_mod = importlib.reload(_cli_mod)
        with patch.object(_cli_mod, "get_tool_definitions", return_value=[]), patch.dict(
            _cli_mod.__dict__, {"CLI_CONFIG": _clean_config}
        ):
            return _cli_mod.HermesCLI()


class TestAgentsInlineDetector:
    """_should_handle_agents_command_inline gates the busy-path fast dispatch."""

    def test_detects_agents_when_agent_running(self):
        cli = _make_cli()
        cli._agent_running = True
        assert cli._should_handle_agents_command_inline("/agents") is True

    def test_detects_tasks_alias_when_agent_running(self):
        """/tasks is an alias for /agents — both should dispatch inline."""
        cli = _make_cli()
        cli._agent_running = True
        assert cli._should_handle_agents_command_inline("/tasks") is True

    def test_ignores_agents_when_agent_idle(self):
        """Idle-path /agents should fall through to the normal process_loop
        dispatch — the queue isn't blocked, so the inline shortcut is
        unnecessary and would just duplicate work."""
        cli = _make_cli()
        cli._agent_running = False
        assert cli._should_handle_agents_command_inline("/agents") is False
        assert cli._should_handle_agents_command_inline("/tasks") is False

    def test_ignores_non_slash_input(self):
        cli = _make_cli()
        cli._agent_running = True
        assert cli._should_handle_agents_command_inline("agents") is False
        assert cli._should_handle_agents_command_inline("") is False

    def test_ignores_other_slash_commands(self):
        cli = _make_cli()
        cli._agent_running = True
        assert cli._should_handle_agents_command_inline("/queue hello") is False
        assert cli._should_handle_agents_command_inline("/stop") is False
        assert cli._should_handle_agents_command_inline("/help") is False
        assert cli._should_handle_agents_command_inline("/steer focus") is False

    def test_ignores_agents_with_attached_images(self):
        """Image payloads take the normal path; /agents takes no args."""
        cli = _make_cli()
        cli._agent_running = True
        assert cli._should_handle_agents_command_inline("/agents", has_images=True) is False


class TestAgentsBusyPathDispatch:
    """When the detector fires, /agents dispatch through process_command must
    reach _handle_agents_command directly rather than being queued."""

    def test_process_command_routes_to_agents_handler(self):
        """With _agent_running=True, /agents calls _handle_agents_command and
        does NOT enqueue onto _pending_input."""
        cli = _make_cli()
        cli._agent_running = True
        cli._pending_input = MagicMock()
        cli._handle_agents_command = MagicMock()

        cli.process_command("/agents")

        cli._handle_agents_command.assert_called_once()
        cli._pending_input.put.assert_not_called()

    def test_tasks_alias_routes_to_agents_handler(self):
        """/tasks resolves to canonical 'agents' and dispatches the same."""
        cli = _make_cli()
        cli._agent_running = True
        cli._pending_input = MagicMock()
        cli._handle_agents_command = MagicMock()

        cli.process_command("/tasks")

        cli._handle_agents_command.assert_called_once()
        cli._pending_input.put.assert_not_called()


if __name__ == "__main__":  # pragma: no cover
    import pytest

    pytest.main([__file__, "-v"])
