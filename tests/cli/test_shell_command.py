"""Tests for the /shell slash command in the CLI."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch
import importlib
import sys


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
        hermes_cli = importlib.import_module("cli")
        importlib.reload(hermes_cli)
        cls = hermes_cli.HermesCLI

    self_ = MagicMock()
    self_.config = _clean_config
    self_._console_print = MagicMock()
    
    # Bind the real process_command method
    self_.process_command = cls.process_command.__get__(self_, type(self_))
    return self_


def test_shell_command_executes_successfully():
    """/shell should execute a command and print output."""
    import subprocess
    cli = _make_cli()
    
    with patch.object(subprocess, 'run') as mock_run:
        mock_run.return_value = MagicMock(
            stdout="test output\n", 
            stderr="", 
            returncode=0
        )
        cli.process_command("/shell echo hello")
        
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        # subprocess.run is called as positional first arg (the command string)
        assert call_args[0][0] == "echo hello"
        assert call_args[1]["shell"] is True
        cli._console_print.assert_called()


def test_shell_command_timeout():
    """/shell should handle timeout gracefully."""
    cli = _make_cli()
    
    with patch.object(subprocess, 'run', side_effect=subprocess.TimeoutExpired(cmd="echo", timeout=30)):
        cli.process_command("/shell sleep 100")
        cli._console_print.assert_called()
        calls = cli._console_print.call_args_list
        assert any("timed out" in str(call).lower() for call in calls)


def test_shell_command_no_args():
    """/shell without arguments should show usage."""
    cli = _make_cli()
    cli.process_command("/shell")
    cli._console_print.assert_called()
    calls = cli._console_print.call_args_list
    assert any("Usage" in str(call) for call in calls)


def test_shell_command_empty_args():
    """/shell with only whitespace should show usage."""
    cli = _make_cli()
    cli.process_command("/shell   ")
    cli._console_print.assert_called()
    calls = cli._console_print.call_args_list
    assert any("Usage" in str(call) for call in calls)