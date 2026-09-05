"""Tests for stacked tool progress scrollback lines in the CLI TUI.

When tool_progress_mode is "all" or "new", _on_tool_progress should print
persistent lines to scrollback on tool.completed, restoring the stacked
tool history that was lost when the TUI switched to a single-line spinner.
"""

import sys
import importlib
from unittest.mock import MagicMock, patch


# Module-level reference to the cli module (set by _make_cli on first call)
_cli_mod = None
_UNSET = object()


def _make_cli(tool_progress="all", verbose=_UNSET):
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
            if verbose is _UNSET:
                inst = mod.HermesCLI()
            else:
                inst = mod.HermesCLI(verbose=verbose)
    # patch.dict(sys.modules) above restores the pre-import state on exit, which
    # DROPS ``cli`` when this was the first import. The mixin handlers resolve
    # ``_cprint`` via a lazy ``from cli import ...``, so ``cli`` must stay
    # registered for ``patch.object(_cli_mod, "_cprint")`` to intercept.
    sys.modules["cli"] = mod
    return inst


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



    def test_off_mode_no_scrollback(self):
        """In 'off' mode, no stacked lines are printed."""
        cli = _make_cli(tool_progress="off")
        with patch.object(_cli_mod, "_cprint") as mock_print:
            cli._on_tool_progress("tool.started", "terminal", "ls", {"command": "ls"})
            cli._on_tool_progress("tool.completed", "terminal", None, None, duration=0.5, is_error=False)

        mock_print.assert_not_called()




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


    def test_verbose_mode_commits_every_call(self):
        """In 'verbose' mode, consecutive same-tool calls each commit a line.

        Mirrors 'all' (no consecutive-repeat suppression — that is 'new'-only),
        so a multi-step turn builds a full scrollable tool history.
        """
        cli = _make_cli(tool_progress="verbose")
        with patch.object(_cli_mod, "_cprint") as mock_print:
            cli._on_tool_progress("tool.started", "terminal", "echo one", {"command": "echo one"})
            cli._on_tool_progress("tool.completed", "terminal", None, None, duration=0.1, is_error=False)
            cli._on_tool_progress("tool.started", "terminal", "echo two", {"command": "echo two"})
            cli._on_tool_progress("tool.completed", "terminal", None, None, duration=0.1, is_error=False)

        assert mock_print.call_count == 2




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


class TestMoAReferenceBlocks:
    """moa.reference renders a labelled thinking-style block; moa.aggregating
    updates the spinner. Both are display-only and must commit regardless of
    tool_progress_mode (MoA is non-streaming)."""

    def test_reference_event_prints_labelled_block(self):
        cli = _make_cli(tool_progress="all")
        with patch.object(_cli_mod, "_cprint") as mock_print:
            cli._on_tool_progress(
                "moa.reference",
                "openrouter:openai/gpt-5.5",
                "Paris is the capital.",
                None,
                moa_index=1,
                moa_count=2,
            )
        printed = " ".join(str(c.args[0]) for c in mock_print.call_args_list)
        # Header names the source model + index/count; body carries the text.
        assert "openrouter:openai/gpt-5.5" in printed
        assert "Reference 1/2" in printed
        assert "Paris is the capital." in printed

    def test_reference_event_prints_even_when_progress_off(self):
        """Reference blocks are the MoA process view, not tool progress — they
        must show even with tool_progress: off."""
        cli = _make_cli(tool_progress="off")
        with patch.object(_cli_mod, "_cprint") as mock_print:
            cli._on_tool_progress(
                "moa.reference", "openrouter:anthropic/claude-opus-4.8", "Four.", None,
                moa_index=2, moa_count=2,
            )
        assert mock_print.called

    def test_aggregating_event_updates_spinner_only(self):
        cli = _make_cli(tool_progress="all")
        with patch.object(_cli_mod, "_cprint") as mock_print:
            cli._on_tool_progress("moa.aggregating", "openrouter:anthropic/claude-opus-4.8", None, None)
        assert "aggregating" in cli._spinner_text
        # aggregating is a spinner-only transition; no committed scrollback line.
        mock_print.assert_not_called()
