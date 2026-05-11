"""Tests for ``HermesCLI._prompt_text_input`` thread-safe input dispatch."""

import threading
from unittest.mock import MagicMock, patch


def _make_cli():
    """Minimal HermesCLI shell exposing ``_prompt_text_input``."""
    import cli as cli_mod

    obj = object.__new__(cli_mod.HermesCLI)
    obj._app = MagicMock()
    obj._app._is_running = True
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

    def test_background_thread_schedules_prompt_back_to_app_loop(self):
        """On a daemon thread, schedule the prompt on the app loop."""
        cli = _make_cli()
        captured = {}
        scheduled = {}

        class _FakeLoop:
            def call_soon_threadsafe(self, callback):
                scheduled["called"] = True
                callback()

        cli._app.loop = _FakeLoop()

        def fake_input(prompt):
            captured["prompt"] = prompt
            return "1"

        result_holder = {}

        def run_on_daemon():
            with patch("prompt_toolkit.application.run_in_terminal", side_effect=lambda fn: fn()) as mock_rit, \
                 patch("builtins.input", side_effect=fake_input):
                result_holder["value"] = cli._prompt_text_input("Choice [1/2/3]: ")
                result_holder["rit_called"] = mock_rit.called

        t = threading.Thread(target=run_on_daemon, daemon=True)
        t.start()
        t.join(timeout=2.0)
        assert not t.is_alive(), "daemon thread hung — prompt was not scheduled"
        assert scheduled.get("called") is True
        assert result_holder["rit_called"] is True
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
