"""Verify _on_tool_gen_start does not print 'preparing' spam (#10478).

The spinner widget already shows tool activity, so the gen-start callback
should only close open stream/reasoning boxes — not print status lines.
"""

import os
import sys
import importlib
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_cli_mod = None


def _make_cli():
    """Create a HermesCLI instance with minimal mocking."""
    global _cli_mod
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
    with patch.dict(sys.modules, prompt_toolkit_stubs), \
         patch.dict("os.environ", clean_env, clear=False):
        import cli as mod
        mod = importlib.reload(mod)
        _cli_mod = mod
        with patch.object(mod, "get_tool_definitions", return_value=[]), \
             patch.dict(mod.__dict__, {"CLI_CONFIG": _clean_config}):
            return mod.HermesCLI()


class TestToolGenStartNoSpam:
    """_on_tool_gen_start must not print 'preparing' messages."""

    def test_no_print_on_single_tool(self):
        cli = _make_cli()
        with patch.object(_cli_mod, "_cprint") as mock_print:
            cli._on_tool_gen_start("terminal")
        # _cprint must not be called with any 'preparing' message
        for c in mock_print.call_args_list:
            assert "preparing" not in str(c), f"Unexpected preparing message: {c}"

    def test_no_print_on_parallel_tools(self):
        cli = _make_cli()
        with patch.object(_cli_mod, "_cprint") as mock_print:
            cli._on_tool_gen_start("terminal")
            cli._on_tool_gen_start("read_file")
            cli._on_tool_gen_start("terminal")
        for c in mock_print.call_args_list:
            assert "preparing" not in str(c), f"Unexpected preparing message: {c}"

    def test_stream_box_still_closed(self):
        """The stream/reasoning close logic must still work."""
        cli = _make_cli()
        cli._stream_box_opened = True
        cli._flush_stream = MagicMock()
        cli._close_reasoning_box = MagicMock()
        cli._on_tool_gen_start("terminal")
        cli._flush_stream.assert_called_once()
        cli._close_reasoning_box.assert_called_once()
        assert cli._stream_box_opened is False
