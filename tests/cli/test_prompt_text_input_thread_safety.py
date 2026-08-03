"""Tests for ``HermesCLI._prompt_text_input`` thread-safe input dispatch.

Raw ``input()`` prompts can race with prompt_toolkit when called from the TUI.
The normal slash confirmations now use a prompt_toolkit-native modal, but
``_prompt_text_input`` remains as a fallback for non-interactive calls and edge
cases.
"""

import threading
from unittest.mock import MagicMock, patch


def _make_cli():
    """Minimal HermesCLI shell exposing prompt fallback helpers."""
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
            cli._prompt_text_input("Choice: ")

        # run_in_terminal was invoked; the _ask closure passed to it would
        # call input() when driven by the event loop.  We assert dispatch path,
        # not the orphaned-coroutine result.
        assert mock_rit.called

    def test_background_thread_cancels_instead_of_hanging(self):
        """On a daemon thread with an active app, cancel cleanly (return None).

        stdin is owned by the prompt_toolkit event loop / JSON-RPC pipe on the
        non-main (process_loop / slash-worker) thread, so a bare input() there
        would block until the worker's timeout (#23185 / billing auto-reload
        hang). The guard cancels to None instead of hanging — it must NOT call
        run_in_terminal (orphaned coroutine) and must NOT call input().
        """
        cli = _make_cli()

        result_holder = {}

        def run_on_daemon():
            with patch("prompt_toolkit.application.run_in_terminal") as mock_rit, \
                 patch("builtins.input", side_effect=AssertionError("input() must not be called off-main-thread")) as mock_input:
                result_holder["value"] = cli._prompt_text_input("Choice [1/2/3]: ")
                result_holder["rit_called"] = mock_rit.called
                result_holder["input_called"] = mock_input.called

        t = threading.Thread(target=run_on_daemon, daemon=True)
        t.start()
        t.join(timeout=2.0)
        assert not t.is_alive(), "daemon thread hung — guard did not cancel cleanly"

        # Cancelled cleanly: None returned, neither run_in_terminal nor input() called.
        assert result_holder["value"] is None
        assert result_holder["rit_called"] is False
        assert result_holder["input_called"] is False

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
