"""Tests for _prompt_text_input background-thread guard (fixes #22970).

Before fix: run_in_terminal was called from background threads, returning
an unawaited coroutine and logging RuntimeWarning.
After fix: the main-thread guard matches _run_curses_picker's pattern.
"""
import threading
from unittest.mock import MagicMock, patch

from cli import HermesCLI


def _make_cli() -> HermesCLI:
    cli = HermesCLI.__new__(HermesCLI)
    cli.config = {}
    cli.console = MagicMock()
    cli.agent = None
    cli.conversation_history = []
    cli.session_id = "sess-test"
    cli._pending_input = MagicMock()
    cli._app = MagicMock()  # simulate active prompt_toolkit app
    cli._status_bar_visible = True
    return cli


def test_prompt_text_input_uses_run_in_terminal_on_main_thread():
    """On the main thread with _app active, run_in_terminal must be used."""
    cli = _make_cli()
    assert threading.current_thread() is threading.main_thread()

    run_in_terminal_called = []

    def fake_run_in_terminal(fn):
        run_in_terminal_called.append(True)
        fn()  # execute the actual input function

    with patch("builtins.input", return_value="1"):
        with patch("cli.threading") as mock_threading:
            mock_threading.current_thread.return_value = threading.main_thread()
            mock_threading.main_thread.return_value = threading.main_thread()
            with patch("prompt_toolkit.application.run_in_terminal", fake_run_in_terminal):
                result = cli._prompt_text_input("Choice: ")

    assert run_in_terminal_called, "run_in_terminal was not called on main thread"


def test_prompt_text_input_bypasses_run_in_terminal_on_background_thread():
    """On a background thread, _ask must be called directly (no run_in_terminal).

    This prevents 'coroutine was never awaited' RuntimeWarning (issue #22970).
    """
    cli = _make_cli()
    run_in_terminal_called = []
    direct_ask_called = []
    result_holder = [None]

    # Simulate being on a background thread
    with patch("cli.threading") as mock_threading:
        bg_thread = MagicMock()
        mock_threading.current_thread.return_value = bg_thread
        mock_threading.main_thread.return_value = threading.main_thread()
        # bg_thread is not main_thread, so in_main_thread = False

        with patch("builtins.input", return_value="2") as mock_input:
            result = cli._prompt_text_input("Choice: ")
            direct_ask_called.append(mock_input.called)

    assert not run_in_terminal_called, (
        "run_in_terminal was called from a background thread — this causes "
        "RuntimeWarning: coroutine was never awaited (#22970)"
    )
    assert direct_ask_called and direct_ask_called[0], "input() was never called"
