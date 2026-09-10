"""Regression tests for classic-CLI mid-run /bg and /btw dispatch.

Background
----------
``/bg`` (formerly ``/background``) exists to start independent work while
the current turn keeps running. Typed while the agent was busy it went into
``self._pending_input`` like ordinary input, and ``process_loop`` is blocked
inside ``self.chat()`` for the whole run, so the background task only started
once the foreground turn had finished. That is the one moment it was not
needed (#75221).

``/steer`` had the identical problem and was fixed by dispatching inline on
the UI thread; the command's own ``CommandDef`` already declares
``busy_policy="dispatch"``, which the gateway honours and the classic CLI
never consulted.

These tests exercise the detector without starting a prompt_toolkit app,
mirroring tests/cli/test_cli_steer_busy_path.py.
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


class TestBackgroundInlineDetector:
    def test_detects_background_when_agent_running(self):
        cli = _make_cli()
        cli._agent_running = True
        assert cli._should_handle_background_command_inline(
            "/bg inspect the test failures"
        ) is True

    def test_detects_both_commands(self):
        cli = _make_cli()
        cli._agent_running = True
        assert cli._should_handle_background_command_inline("/bg do work") is True
        assert cli._should_handle_background_command_inline("/btw do work") is True

    def test_background_alias_still_resolves_to_bg(self):
        """The retired /background spelling no longer resolves to a command."""
        cli = _make_cli()
        cli._agent_running = True
        assert cli._should_handle_background_command_inline(
            "/background do work"
        ) is False

    def test_ignores_background_when_agent_idle(self):
        """Idle input falls through to the normal process_loop dispatch."""
        cli = _make_cli()
        cli._agent_running = False
        assert cli._should_handle_background_command_inline("/bg do work") is False

    def test_ignores_non_slash_input(self):
        cli = _make_cli()
        cli._agent_running = True
        assert cli._should_handle_background_command_inline("bg without slash") is False
        assert cli._should_handle_background_command_inline("") is False

    def test_ignores_other_slash_commands(self):
        cli = _make_cli()
        cli._agent_running = True
        assert cli._should_handle_background_command_inline("/steer hello") is False
        assert cli._should_handle_background_command_inline("/queue hello") is False
        assert cli._should_handle_background_command_inline("/stop") is False

    def test_ignores_background_with_attached_images(self):
        """Image payloads take the normal path."""
        cli = _make_cli()
        cli._agent_running = True
        assert cli._should_handle_background_command_inline(
            "/bg look at this", has_images=True
        ) is False

    def test_case_and_whitespace_tolerant(self):
        cli = _make_cli()
        cli._agent_running = True
        assert cli._should_handle_background_command_inline("/BG do work") is True


class TestBackgroundBusyPolicyContract:
    """The registry already declares the intent this detector implements."""

    def test_bg_and_btw_declare_dispatch_while_busy(self):
        from hermes_cli.commands import resolve_command

        for name in ("bg", "btw"):
            cmd = resolve_command(name)
            assert cmd is not None
            assert cmd.name == name
            assert cmd.busy_policy == "dispatch"

    def test_background_name_is_retired(self):
        from hermes_cli.commands import resolve_command

        assert resolve_command("background") is None
