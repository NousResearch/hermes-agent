"""Tests for stacked tool progress scrollback lines in the CLI TUI.

When tool_progress_mode is "all" or "new", _on_tool_progress should print
persistent lines to scrollback on tool.completed, restoring the stacked
tool history that was lost when the TUI switched to a single-line spinner.
"""

import os
import sys
import importlib
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Module-level reference to the cli module (set by _make_cli on first call)
_cli_mod = None


def _make_cli(tool_progress="all"):
    """Create a HermesCLI instance with minimal mocking."""
    global _cli_mod
    _clean_config = {
        "model": {
            "default": "anthropic/claude-opus-4.6",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "auto",
        },
        "display": {"compact": False, "tool_progress": tool_progress},
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
    with patch.dict(sys.modules, prompt_toolkit_stubs), \
         patch.dict("os.environ", clean_env, clear=False):
        import cli as mod
        mod = importlib.reload(mod)
        _cli_mod = mod
        with patch.object(mod, "get_tool_definitions", return_value=[]), \
             patch.dict(mod.__dict__, {"CLI_CONFIG": _clean_config}):
            return mod.HermesCLI()


class TestToolProgressScrollback:
    """Stacked scrollback lines for 'all' and 'new' modes."""

    def test_all_mode_prints_scrollback_on_completed(self):
        """In 'all' mode, tool.completed prints a stacked line."""
        cli = _make_cli(tool_progress="all")
        # Simulate tool.started
        cli._on_tool_progress("tool.started", "terminal", "git log", {"command": "git log"})
        # Simulate tool.completed
        with patch.object(_cli_mod, "_cprint") as mock_print:
            cli._on_tool_progress("tool.completed", "terminal", None, None, duration=1.5, is_error=False)

        mock_print.assert_called_once()
        line = mock_print.call_args[0][0]
        # Should contain tool info (the cute message format has "git log" for terminal)
        assert "git log" in line or "$" in line

    def test_all_mode_prints_every_call(self):
        """In 'all' mode, consecutive calls to the same tool each get a line."""
        cli = _make_cli(tool_progress="all")
        with patch.object(_cli_mod, "_cprint") as mock_print:
            # First call
            cli._on_tool_progress("tool.started", "read_file", "cli.py", {"path": "cli.py"})
            cli._on_tool_progress("tool.completed", "read_file", None, None, duration=0.1, is_error=False)
            # Second call (same tool)
            cli._on_tool_progress("tool.started", "read_file", "run_agent.py", {"path": "run_agent.py"})
            cli._on_tool_progress("tool.completed", "read_file", None, None, duration=0.2, is_error=False)

        assert mock_print.call_count == 2

    def test_new_mode_skips_consecutive_repeats(self):
        """In 'new' mode, consecutive calls to the same tool only print once."""
        cli = _make_cli(tool_progress="new")
        with patch.object(_cli_mod, "_cprint") as mock_print:
            cli._on_tool_progress("tool.started", "read_file", "cli.py", {"path": "cli.py"})
            cli._on_tool_progress("tool.completed", "read_file", None, None, duration=0.1, is_error=False)
            cli._on_tool_progress("tool.started", "read_file", "run_agent.py", {"path": "run_agent.py"})
            cli._on_tool_progress("tool.completed", "read_file", None, None, duration=0.2, is_error=False)

        assert mock_print.call_count == 1  # Only the first read_file

    def test_new_mode_prints_when_tool_changes(self):
        """In 'new' mode, a different tool name triggers a new line."""
        cli = _make_cli(tool_progress="new")
        with patch.object(_cli_mod, "_cprint") as mock_print:
            cli._on_tool_progress("tool.started", "read_file", "cli.py", {"path": "cli.py"})
            cli._on_tool_progress("tool.completed", "read_file", None, None, duration=0.1, is_error=False)
            cli._on_tool_progress("tool.started", "search_files", "pattern", {"pattern": "test"})
            cli._on_tool_progress("tool.completed", "search_files", None, None, duration=0.3, is_error=False)
            cli._on_tool_progress("tool.started", "read_file", "run_agent.py", {"path": "run_agent.py"})
            cli._on_tool_progress("tool.completed", "read_file", None, None, duration=0.2, is_error=False)

        # read_file, search_files, read_file (3rd prints because search_files broke the streak)
        assert mock_print.call_count == 3

    def test_off_mode_no_scrollback(self):
        """In 'off' mode, no stacked lines are printed."""
        cli = _make_cli(tool_progress="off")
        with patch.object(_cli_mod, "_cprint") as mock_print:
            cli._on_tool_progress("tool.started", "terminal", "ls", {"command": "ls"})
            cli._on_tool_progress("tool.completed", "terminal", None, None, duration=0.5, is_error=False)

        mock_print.assert_not_called()

    def test_error_suffix_on_failed_tool(self):
        """When is_error=True, the stacked line includes [error]."""
        cli = _make_cli(tool_progress="all")
        cli._on_tool_progress("tool.started", "terminal", "bad cmd", {"command": "bad cmd"})
        with patch.object(_cli_mod, "_cprint") as mock_print:
            cli._on_tool_progress("tool.completed", "terminal", None, None, duration=0.5, is_error=True)

        line = mock_print.call_args[0][0]
        assert "[error]" in line

    def test_spinner_still_updates_on_started(self):
        """tool.started still updates the spinner text for live display."""
        cli = _make_cli(tool_progress="all")
        cli._on_tool_progress("tool.started", "terminal", "git status", {"command": "git status"})
        assert "git status" in cli._spinner_text

    def test_spinner_timer_clears_on_completed(self):
        """tool.completed still clears the tool timer."""
        cli = _make_cli(tool_progress="all")
        cli._on_tool_progress("tool.started", "terminal", "git status", {"command": "git status"})
        assert cli._tool_start_time > 0
        with patch.object(_cli_mod, "_cprint"):
            cli._on_tool_progress("tool.completed", "terminal", None, None, duration=0.5, is_error=False)
        assert cli._tool_start_time == 0.0

    def test_concurrent_tools_produce_stacked_lines(self):
        """Multiple tool.started followed by multiple tool.completed all produce lines."""
        cli = _make_cli(tool_progress="all")
        with patch.object(_cli_mod, "_cprint") as mock_print:
            # All start first (concurrent pattern)
            cli._on_tool_progress("tool.started", "web_search", "query 1", {"query": "test 1"})
            cli._on_tool_progress("tool.started", "web_search", "query 2", {"query": "test 2"})
            # All complete
            cli._on_tool_progress("tool.completed", "web_search", None, None, duration=1.0, is_error=False)
            cli._on_tool_progress("tool.completed", "web_search", None, None, duration=1.5, is_error=False)

        assert mock_print.call_count == 2

    def test_verbose_mode_no_duplicate_scrollback(self):
        """In 'verbose' mode, scrollback lines are NOT printed (run_agent handles verbose output)."""
        cli = _make_cli(tool_progress="verbose")
        with patch.object(_cli_mod, "_cprint") as mock_print:
            cli._on_tool_progress("tool.started", "terminal", "ls", {"command": "ls"})
            cli._on_tool_progress("tool.completed", "terminal", None, None, duration=0.5, is_error=False)

        mock_print.assert_not_called()

    def test_pending_info_stores_on_started(self):
        """tool.started stores args for later use by tool.completed."""
        cli = _make_cli(tool_progress="all")
        cli._on_tool_progress("tool.started", "terminal", "ls", {"command": "ls"})
        assert "terminal" in cli._pending_tool_info
        assert len(cli._pending_tool_info["terminal"]) == 1
        assert cli._pending_tool_info["terminal"][0] == {"command": "ls"}

    def test_pending_info_consumed_on_completed(self):
        """tool.completed consumes stored args (FIFO for concurrent)."""
        cli = _make_cli(tool_progress="all")
        cli._on_tool_progress("tool.started", "terminal", "ls", {"command": "ls"})
        cli._on_tool_progress("tool.started", "terminal", "pwd", {"command": "pwd"})
        assert len(cli._pending_tool_info["terminal"]) == 2
        with patch.object(_cli_mod, "_cprint"):
            cli._on_tool_progress("tool.completed", "terminal", None, None, duration=0.1, is_error=False)
        # First entry consumed, second remains
        assert len(cli._pending_tool_info.get("terminal", [])) == 1
        assert cli._pending_tool_info["terminal"][0] == {"command": "pwd"}
