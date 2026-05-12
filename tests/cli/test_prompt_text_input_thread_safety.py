"""Tests for ``HermesCLI._prompt_text_input`` thread-safe input dispatch.

Slash commands (``/clear``, ``/new``, ``/undo``, ``/reload-mcp``) are dispatched
from the ``process_loop`` daemon thread.  The fix schedules ``run_in_terminal``
onto the app's event loop via ``loop.call_soon_threadsafe`` and blocks the
daemon thread with a ``threading.Event`` until the user answers.

Without this, ``input()`` races against prompt_toolkit's raw-mode stdin
ownership — keystrokes go to the PT handler on the main thread and the
confirmation prompt hangs forever (issue #23185).
"""

import threading
from unittest.mock import MagicMock, call, patch


def _make_cli(loop_running: bool = True):
    """Minimal HermesCLI shell exposing ``_prompt_text_input``.

    Sets up a fake prompt_toolkit ``_app`` with a mock event loop.
    ``loop_running=False`` exercises the fallback path for when the PT
    loop is not active (e.g. non-interactive / tests without a real loop).
    """
    import cli as cli_mod

    obj = object.__new__(cli_mod.HermesCLI)
    mock_loop = MagicMock()
    mock_loop.is_running.return_value = loop_running
    obj._app = MagicMock()
    obj._app.loop = mock_loop
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

    def test_background_thread_uses_call_soon_threadsafe(self):
        """On a daemon thread, schedule run_in_terminal via call_soon_threadsafe.

        This is the fix for issue #23185: process_loop dispatches slash commands
        on a daemon thread.  The old fallback (direct ``input()``) races against
        prompt_toolkit's raw-mode stdin and hangs — keystrokes never reach
        ``input()``.

        The fix posts ``run_in_terminal`` onto the running event loop via
        ``call_soon_threadsafe`` and blocks the daemon thread on a
        ``threading.Event`` until ``_ask`` finishes.
        """
        cli = _make_cli(loop_running=True)
        result_holder = {}

        # Simulate call_soon_threadsafe running _schedule synchronously so the
        # background thread's done.wait() unblocks during the test.
        def fake_call_soon_threadsafe(fn):
            fn()  # call _schedule immediately (simulates event loop dispatch)

        cli._app.loop.call_soon_threadsafe.side_effect = fake_call_soon_threadsafe

        def run_on_daemon():
            with patch("prompt_toolkit.application.run_in_terminal") as mock_rit, \
                 patch("builtins.input", return_value="1"):
                # run_in_terminal is called from inside _schedule, which runs
                # synchronously via fake_call_soon_threadsafe.
                mock_rit.side_effect = lambda fn: fn()  # drive _ask
                result_holder["value"] = cli._prompt_text_input("Choice [1/2/3]: ")
                result_holder["rit_called"] = mock_rit.called
                result_holder["css_called"] = cli._app.loop.call_soon_threadsafe.called

        t = threading.Thread(target=run_on_daemon, daemon=True)
        t.start()
        t.join(timeout=2.0)
        assert not t.is_alive(), "daemon thread hung — done.wait() blocked"

        # call_soon_threadsafe was used (not direct _ask() on the daemon thread).
        assert result_holder["css_called"] is True
        # run_in_terminal was called from inside _schedule (on the event loop).
        assert result_holder["rit_called"] is True
        # input() was driven by run_in_terminal's _ask closure.
        assert result_holder["value"] == "1"

    def test_background_thread_loop_not_running_falls_back_to_direct_input(self):
        """If the PT loop is not running (e.g. tests / non-interactive), call input() directly.

        When ``loop.is_running()`` returns False, call_soon_threadsafe would
        never fire, so we fall back to a direct ``_ask()`` call to avoid a
        permanent block on done.wait().
        """
        cli = _make_cli(loop_running=False)
        captured = {}

        def fake_input(prompt):
            captured["prompt"] = prompt
            return "2"

        result_holder = {}

        def run_on_daemon():
            with patch("prompt_toolkit.application.run_in_terminal") as mock_rit, \
                 patch("builtins.input", side_effect=fake_input):
                result_holder["value"] = cli._prompt_text_input("Choice [1/2/3]: ")
                result_holder["rit_called"] = mock_rit.called

        t = threading.Thread(target=run_on_daemon, daemon=True)
        t.start()
        t.join(timeout=2.0)
        assert not t.is_alive(), "daemon thread hung"

        # run_in_terminal was NOT called via call_soon_threadsafe (loop not running).
        assert result_holder["rit_called"] is False
        # input() was called directly as the fallback.
        assert captured.get("prompt") == "Choice [1/2/3]: "
        assert result_holder["value"] == "2"

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

    def test_background_thread_call_soon_threadsafe_exception_falls_back(self):
        """If call_soon_threadsafe raises, fall back to direct input()."""
        cli = _make_cli(loop_running=True)
        cli._app.loop.call_soon_threadsafe.side_effect = RuntimeError("loop closed")

        captured = {}
        result_holder = {}

        def fake_input(prompt):
            captured["prompt"] = prompt
            return "3"

        def run_on_daemon():
            with patch("builtins.input", side_effect=fake_input):
                result_holder["value"] = cli._prompt_text_input("Choice: ")

        t = threading.Thread(target=run_on_daemon, daemon=True)
        t.start()
        t.join(timeout=2.0)
        assert not t.is_alive(), "daemon thread hung after call_soon_threadsafe failure"

        assert captured.get("prompt") == "Choice: "
        assert result_holder["value"] == "3"

    def test_preamble_printed_before_input_prompt(self):
        """Preamble text is printed inside _ask before input() is called.

        Ensures that option lines always appear before the 'Choice:' cursor —
        the ordering bug where 'Choice [1/2/3]:' appeared before the option
        list because bare print() calls on the daemon thread raced against the
        run_in_terminal scheduling on the event loop.
        """
        cli = _make_cli()
        cli._app = None  # simple path — no PT app, no threading complexity

        call_order = []

        def fake_print(text, **kwargs):
            call_order.append(("print", text))

        def fake_input(prompt):
            call_order.append(("input", prompt))
            return "2"

        with patch("builtins.print", side_effect=fake_print), \
             patch("builtins.input", side_effect=fake_input):
            result = cli._prompt_text_input(
                "Choice [1/2/3]: ",
                preamble="  [1] Approve Once\n  [2] Always Approve\n  [3] Cancel\n\n"
            )

        assert result == "2"
        # print must come before input in the call order
        assert call_order[0][0] == "print", "preamble print must come before input()"
        assert call_order[-1][0] == "input", "input() must be last"
        # preamble content was printed
        assert any("Approve Once" in str(args) for op, args in call_order if op == "print")
