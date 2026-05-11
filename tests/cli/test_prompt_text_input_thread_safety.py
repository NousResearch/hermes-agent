"""Tests for ``HermesCLI._prompt_text_input`` thread-safe input dispatch.

Slash commands (``/clear``, ``/new``, ``/undo``, ``/reload-mcp``) are dispatched
from the ``process_loop`` daemon thread.  ``prompt_toolkit.run_in_terminal``
returns a coroutine that only the main-thread event loop can drive; calling it
from a daemon thread orphans the coroutine, ``_ask`` never runs, and user
keystrokes leak into the composer instead of the confirmation prompt
(see issue #23185).

The fix mirrors ``_run_curses_picker``: when off the main thread, fall back to
a direct ``input()`` call so the prompt actually renders and consumes
keystrokes.
"""

import threading
from unittest.mock import MagicMock, patch


def _make_cli():
    """Minimal HermesCLI shell exposing ``_prompt_text_input``."""
    import cli as cli_mod

    obj = object.__new__(cli_mod.HermesCLI)
    obj._app = MagicMock()
    obj._status_bar_visible = True
    return obj


class TestPromptTextInputThreadSafety:
    def test_main_thread_uses_run_in_terminal(self):
        """On the main thread with an active app, route through run_in_terminal."""
        cli = _make_cli()

        with patch("prompt_toolkit.application.run_in_terminal") as mock_rit, \
             patch("builtins.input", return_value="2"):
            result = cli._prompt_text_input("Choice: ")

        # run_in_terminal was invoked; the _ask closure passed to it would
        # call input() when driven by the event loop.  We assert dispatch path,
        # not the orphaned-coroutine result.
        assert mock_rit.called

    def test_background_thread_falls_back_to_direct_input(self):
        """On a daemon thread, skip run_in_terminal and call input() directly.

        This is the bug from issue #23185: process_loop dispatches slash
        commands on a daemon thread, so run_in_terminal's coroutine is
        orphaned.  The fallback must drive input() itself so user keystrokes
        don't leak into the agent buffer.
        """
        cli = _make_cli()
        captured = {}

        def fake_input(prompt):
            captured["prompt"] = prompt
            return "1"

        result_holder = {}

        def run_on_daemon():
            with patch("prompt_toolkit.application.run_in_terminal") as mock_rit, \
                 patch("builtins.input", side_effect=fake_input):
                result_holder["value"] = cli._prompt_text_input("Choice [1/2/3]: ")
                result_holder["rit_called"] = mock_rit.called

        t = threading.Thread(target=run_on_daemon, daemon=True)
        t.start()
        t.join(timeout=2.0)
        assert not t.is_alive(), "daemon thread hung — input() was not driven"

        # run_in_terminal was bypassed entirely on the background thread.
        assert result_holder["rit_called"] is False
        # input() was invoked with the prompt and its return value was captured.
        assert captured.get("prompt") == "Choice [1/2/3]: "
        assert result_holder["value"] == "1"

    def test_no_app_uses_direct_input(self):
        """Without an active prompt_toolkit app, always call input() directly."""
        cli = _make_cli()
        cli._app = None

        with patch("builtins.input", return_value="cancel") as mock_input:
            result = cli._prompt_text_input("Choice: ")

        assert mock_input.called
        assert result == "cancel"

    def test_run_in_terminal_exception_falls_back(self):
        """If run_in_terminal raises (WSL / Warp edge cases), fall back to input()."""
        cli = _make_cli()

        with patch(
            "prompt_toolkit.application.run_in_terminal",
            side_effect=RuntimeError("event loop dropped the coroutine"),
        ), patch("builtins.input", return_value="3") as mock_input:
            result = cli._prompt_text_input("Choice: ")

        assert mock_input.called
        assert result == "3"

    def test_eof_returns_none(self):
        """EOFError from input() yields None, not an unhandled exception."""
        cli = _make_cli()
        cli._app = None

        with patch("builtins.input", side_effect=EOFError()):
            result = cli._prompt_text_input("Choice: ")

        assert result is None
