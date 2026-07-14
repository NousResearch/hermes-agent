"""Tests for the display.show_banner config option.

The option (default true) lets users skip the welcome banner at startup
and on /new for a minimal startup experience — without disabling the
diagnostic warnings that follow.
"""

from __future__ import annotations

import importlib
import os
import sys
from contextlib import contextmanager
from unittest.mock import MagicMock, patch


def _make_real_cli(show_banner_value, **kwargs):
    """Build a HermesCLI instance and also return its bound cli module.

    ``_make_real_cli`` from sibling test files only returns the instance, but
    after ``importlib.reload`` inside a ``patch.dict(sys.modules, ...)`` block
    the reloaded module is dropped from ``sys.modules`` on exit. A later
    ``import cli`` then yields a *different* module than the one cli_obj's
    class methods reference via __globals__. We need a handle on the right
    module to patch attributes that show_banner() looks up.
    """
    clean_config = {
        "model": {
            "default": "anthropic/claude-opus-4.6",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "auto",
        },
        "display": {
            "compact": False,
            "tool_progress": "all",
            "show_banner": show_banner_value,
        },
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
    }
    with (
        patch.dict(sys.modules, prompt_toolkit_stubs),
        patch.dict("os.environ", clean_env, clear=False),
    ):
        import cli as cli_mod

        cli_mod = importlib.reload(cli_mod)
        with (
            patch.object(cli_mod, "get_tool_definitions", return_value=[]),
            patch.dict(cli_mod.__dict__, {"CLI_CONFIG": clean_config}),
        ):
            return cli_mod.HermesCLI(**kwargs), cli_mod


@contextmanager
def _patch_banner_calls(cli_mod):
    """Patch the banner helpers + terminal size for the test's cli module.

    Uses patch.object on the bound cli_mod (not "cli.foo") because the cli_obj
    holds a reference to the reloaded module which gets dropped from
    sys.modules — a top-level "import cli" patch would target a different
    module than the one show_banner actually looks up.
    """
    with (
        patch.object(cli_mod, "build_welcome_banner") as mock_banner,
        patch.object(cli_mod, "_build_compact_banner") as mock_compact,
        patch.object(cli_mod, "get_tool_definitions", return_value=[]),
        patch.object(
            cli_mod.shutil,
            "get_terminal_size",
            return_value=os.terminal_size((120, 40)),
        ),
    ):
        yield mock_banner, mock_compact


def test_show_banner_renders_when_show_banner_true():
    cli_obj, cli_mod = _make_real_cli(show_banner_value=True, compact=False)
    cli_obj.console = MagicMock()

    with _patch_banner_calls(cli_mod) as (mock_banner, mock_compact):
        cli_obj.show_banner()

    assert mock_banner.call_count == 1
    assert mock_compact.call_count == 0


def test_show_banner_skipped_when_show_banner_false():
    """display.show_banner: false suppresses both full and compact banners."""
    cli_obj, cli_mod = _make_real_cli(show_banner_value=False, compact=False)
    cli_obj.console = MagicMock()

    with _patch_banner_calls(cli_mod) as (mock_banner, mock_compact), patch.object(
        cli_obj, "_show_tool_availability_warnings"
    ) as mock_warnings:
        cli_obj.show_banner()

    # Screen is still cleared, but no banner gets rendered.
    cli_obj.console.clear.assert_called_once()
    assert mock_banner.call_count == 0
    assert mock_compact.call_count == 0
    mock_warnings.assert_called_once()


def test_show_banner_skipped_in_compact_mode_when_disabled():
    """Even when compact=True, show_banner=False suppresses the compact banner."""
    cli_obj, cli_mod = _make_real_cli(show_banner_value=False, compact=True)
    cli_obj.console = MagicMock()

    with _patch_banner_calls(cli_mod) as (mock_banner, mock_compact):
        cli_obj.show_banner()

    assert mock_banner.call_count == 0
    assert mock_compact.call_count == 0


def test_show_banner_defaults_to_true_when_missing():
    """When display.show_banner is not configured, the banner is shown (back-compat)."""
    clean_config = {
        "model": {
            "default": "anthropic/claude-opus-4.6",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "auto",
        },
        # No show_banner key — exercise the default.
        "display": {"compact": False, "tool_progress": "all"},
        "agent": {},
        "terminal": {"env_type": "local"},
    }
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
    }
    with (
        patch.dict(sys.modules, prompt_toolkit_stubs),
        patch.dict(
            "os.environ", {"LLM_MODEL": "", "HERMES_MAX_ITERATIONS": ""}, clear=False
        ),
    ):
        import cli as cli_mod

        cli_mod = importlib.reload(cli_mod)
        with (
            patch.object(cli_mod, "get_tool_definitions", return_value=[]),
            patch.dict(cli_mod.__dict__, {"CLI_CONFIG": clean_config}),
        ):
            cli_obj = cli_mod.HermesCLI(compact=False)

    assert cli_obj.show_startup_banner is True


def test_clear_in_prompt_toolkit_skips_banner_when_disabled():
    """The prompt-toolkit /clear path honors display.show_banner."""
    cli_obj, cli_mod = _make_real_cli(show_banner_value=False, compact=False)
    cli_obj._pending_resume_sessions = None
    cli_obj._app = MagicMock()
    cli_obj._confirm_destructive_slash = MagicMock(return_value=True)
    cli_obj.new_session = MagicMock()

    with (
        patch.object(cli_mod, "ChatConsole") as mock_chat_console,
        patch.object(cli_mod, "build_welcome_banner") as mock_banner,
        patch.object(cli_mod, "_build_compact_banner") as mock_compact,
    ):
        assert cli_obj.process_command("/clear") is True

    cli_obj.new_session.assert_called_once_with(silent=True)
    cli_obj._app.output.erase_screen.assert_called_once()
    cli_obj._app.output.cursor_goto.assert_called_once_with(0, 0)
    cli_obj._app.output.flush.assert_called_once()
    mock_banner.assert_not_called()
    mock_compact.assert_not_called()
    mock_chat_console.return_value.print.assert_called()
