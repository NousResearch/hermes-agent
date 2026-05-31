"""Regression tests for #30768, #32383, and #33961.

``_prompt_text_input_modal`` answers destructive-slash confirmations through a
queue-based modal driven by prompt_toolkit key bindings.  When invoked from the
``process_loop`` daemon thread it sets the modal up on the app's event loop via
``call_soon_threadsafe``, so it is safe on every platform — including native
Windows (#33961), where the earlier ``sys.platform == "win32"`` → raw ``input()``
fallback deadlocked the daemon thread against prompt_toolkit's stdin ownership.

These tests verify:
1. Daemon-thread confirm uses the modal via the app loop on Linux AND native
   Windows (#33961) — never the raw stdin fallback, never a hang.
2. Main-thread confirm with a running app uses the modal.
3. The raw stdin fallback is kept ONLY for the safe cases: no running app, and
   (on win32, off-thread) a scheduling failure degrades to a clean cancel.
4. Empty choices returns None.
"""

import sys
import threading
import time
from unittest.mock import MagicMock, patch

import pytest


def _make_cli():
    """Minimal HermesCLI shell exposing the prompt/modal helpers."""
    import cli as cli_mod

    obj = object.__new__(cli_mod.HermesCLI)
    obj._app = MagicMock()
    obj._app.loop = MagicMock()
    obj._status_bar_visible = True
    obj._last_invalidate = 0.0
    obj._modal_input_snapshot = None
    obj._slash_confirm_state = None
    obj._slash_confirm_deadline = 0
    return obj


_SAMPLE_CHOICES = [
    ("once", "Approve Once", "proceed this time only"),
    ("always", "Always Approve", "proceed and silence this prompt permanently"),
    ("cancel", "Cancel", "keep current conversation"),
]


def _answer_modal_when_open(cli, response, stop=None):
    """Push ``response`` onto the modal's response_queue once it opens.

    Gives up after ~2s, or early when ``stop`` is set (the modal will never open,
    e.g. a scheduling failure) so degraded-path tests don't wait the full budget.
    """
    for _ in range(100):
        if stop is not None and stop.is_set():
            return
        state = cli._slash_confirm_state
        if state and "response_queue" in state:
            state["response_queue"].put(response)
            return
        time.sleep(0.02)


def _run_on_daemon(call, cli, *, platform, response, schedule=None):
    """Invoke ``call`` on a daemon thread — as the process_loop does — answering
    the modal with ``response`` once it opens.

    Returns ``{result, stdin_called, capture, restore}``.  ``schedule`` overrides
    the ``call_soon_threadsafe`` side effect (default: run the callback inline);
    pass a raiser to simulate a scheduling failure.  Fails if the worker hangs,
    which is the deadlock canary for #33961.
    """
    outcome = {"capture": [], "restore": [], "result": None, "stdin_called": False}
    done = threading.Event()

    def _worker():
        try:
            with patch.object(sys, "platform", platform), \
                 patch.object(cli._app.loop, "call_soon_threadsafe", side_effect=schedule or (lambda cb: cb())), \
                 patch.object(cli, "_prompt_text_input") as mock_stdin, \
                 patch.object(cli, "_invalidate"), \
                 patch.object(cli, "_capture_modal_input_snapshot", side_effect=lambda: outcome["capture"].append(1)), \
                 patch.object(cli, "_restore_modal_input_snapshot", side_effect=lambda: outcome["restore"].append(1)):
                outcome["result"] = call()
                outcome["stdin_called"] = mock_stdin.called
        finally:
            done.set()

    worker = threading.Thread(target=_worker, daemon=True)
    answerer = threading.Thread(target=_answer_modal_when_open, args=(cli, response, done), daemon=True)
    answerer.start()
    worker.start()
    worker.join(timeout=2.0)
    answerer.join(timeout=2.0)
    assert not worker.is_alive(), "daemon thread hung — modal deadlocked"
    return outcome


class TestModal:
    """Behaviour of _prompt_text_input_modal across platforms and threads."""

    @pytest.mark.parametrize("platform", ["linux", "win32"])
    def test_daemon_thread_uses_modal_via_app_loop(self, platform):
        """Off the process_loop daemon thread, the confirm uses the modal via
        call_soon_threadsafe on every platform — including native Windows, where
        the old win32 early-return deadlocked on raw input() (#33961)."""
        cli = _make_cli()
        outcome = _run_on_daemon(
            lambda: cli._prompt_text_input_modal(
                title="⚠️  /reset",
                detail="This starts a fresh session.",
                choices=_SAMPLE_CHOICES,
                timeout=5,
            ),
            cli,
            platform=platform,
            response="once",
        )
        assert outcome["stdin_called"] is False, "must use the modal, not raw input()"
        assert outcome["result"] == "once"
        assert outcome["capture"] == [1]
        assert outcome["restore"] == [1]
        assert cli._slash_confirm_state is None

    def test_main_thread_with_app_uses_modal(self):
        """On the main thread with a running app, the queue-based modal is used."""
        cli = _make_cli()
        with patch.object(sys, "platform", "darwin"), \
             patch.object(cli, "_capture_modal_input_snapshot"), \
             patch.object(cli, "_restore_modal_input_snapshot"), \
             patch.object(cli, "_invalidate"), \
             patch.object(cli, "_prompt_text_input") as mock_stdin:
            answerer = threading.Thread(target=_answer_modal_when_open, args=(cli, "once"), daemon=True)
            answerer.start()
            result = cli._prompt_text_input_modal(
                title="⚠️  /new",
                detail="This starts a fresh session.",
                choices=_SAMPLE_CHOICES,
                timeout=5,
            )
            answerer.join(timeout=2.0)

        mock_stdin.assert_not_called()
        assert result == "once"

    def test_no_app_falls_back_to_stdin(self):
        """Without a running app (oneshot / non-interactive), use the stdin prompt."""
        cli = _make_cli()
        cli._app = None

        with patch.object(cli, "_prompt_text_input", return_value="3") as mock_stdin:
            result = cli._prompt_text_input_modal(
                title="⚠️  /clear",
                detail="This clears the screen.",
                choices=_SAMPLE_CHOICES,
            )

        mock_stdin.assert_called_once_with("Choice [1/2/3]: ")
        assert result == "3"

    def test_windows_no_app_falls_back_to_stdin(self):
        """win32 without a running app keeps stdin — the only case where the raw
        prompt is safe on Windows, since no app owns the console to deadlock."""
        cli = _make_cli()
        cli._app = None

        with patch.object(sys, "platform", "win32"), \
             patch.object(cli, "_prompt_text_input", return_value="1") as mock_stdin:
            result = cli._prompt_text_input_modal(
                title="⚠️  /new — destroys conversation state",
                detail="This starts a fresh session.",
                choices=_SAMPLE_CHOICES,
            )

        mock_stdin.assert_called_once_with("Choice [1/2/3]: ")
        assert result == "1"

    def test_windows_scheduling_failure_clean_cancels(self):
        """win32 off the main thread: if marshaling onto the app loop fails, cancel
        cleanly (None) rather than fall to raw input() (which deadlocks on native
        Windows) or hang. Asserts the _stdin_fallback guard (#33961)."""
        cli = _make_cli()

        def _raise(_cb):
            raise RuntimeError("loop closed")

        outcome = _run_on_daemon(
            lambda: cli._prompt_text_input_modal(
                title="⚠️  /reset",
                detail="This starts a fresh session.",
                choices=_SAMPLE_CHOICES,
                timeout=5,
            ),
            cli,
            platform="win32",
            response="once",
            schedule=_raise,
        )
        assert outcome["stdin_called"] is False, "win32 off-thread must NOT call raw input()"
        assert outcome["result"] is None
        assert cli._slash_confirm_state is None

    @pytest.mark.parametrize(
        "platform, expect_stdin, expect_result",
        [("win32", False, None), ("linux", True, "1")],
    )
    def test_daemon_thread_no_app_loop_uses_fallback(self, platform, expect_stdin, expect_result):
        """Off the daemon thread with no resolvable app loop (``self._app.loop``
        is None / raises), the modal can never be scheduled, so the method short-
        circuits at the app_loop-is-None site (cli.py ~7260) — a distinct path
        from a call_soon_threadsafe failure. win32 clean-cancels (None) instead of
        deadlocking on raw input(); other platforms keep the stdin prompt."""
        cli = _make_cli()
        cli._app.loop = None  # forces app_loop is None, off the main thread

        outcome = {"result": None, "stdin_called": False}
        done = threading.Event()

        def _worker():
            try:
                with patch.object(sys, "platform", platform), \
                     patch.object(cli, "_prompt_text_input", return_value="1") as mock_stdin, \
                     patch.object(cli, "_invalidate"):
                    outcome["result"] = cli._prompt_text_input_modal(
                        title="⚠️  /reset",
                        detail="This starts a fresh session.",
                        choices=_SAMPLE_CHOICES,
                        timeout=5,
                    )
                    outcome["stdin_called"] = mock_stdin.called
            finally:
                done.set()

        worker = threading.Thread(target=_worker, daemon=True)
        worker.start()
        worker.join(timeout=2.0)
        assert not worker.is_alive(), "daemon thread hung — modal deadlocked"
        assert outcome["stdin_called"] is expect_stdin
        assert outcome["result"] == expect_result
        assert cli._slash_confirm_state is None

    def test_empty_choices_returns_none(self):
        """Empty choices returns None without prompting."""
        cli = _make_cli()

        with patch.object(cli, "_prompt_text_input") as mock_stdin:
            result = cli._prompt_text_input_modal(title="Test", detail="Test", choices=[])

        mock_stdin.assert_not_called()
        assert result is None


class TestConfirmDestructiveSlashWindows:
    """End-to-end _confirm_destructive_slash on the native-Windows daemon thread."""

    def _make_interactive_cli(self):
        cli = _make_cli()
        cli.model = "test-model"
        cli._agent_running = False
        cli._spinner_text = ""
        cli._should_exit = False
        cli._command_running = False
        cli.session_id = "test-session"
        cli._pending_tool_info = {}
        cli._tool_start_time = 0.0
        cli._last_scrollback_tool = ""
        return cli

    @pytest.mark.parametrize(
        "response, expected",
        [("once", "once"), ("cancel", None)],
    )
    def test_confirm_destructive_slash_uses_modal_on_windows(self, response, expected):
        """On native Windows, the bare /new confirm drives the modal (not stdin)
        and returns the chosen outcome — the bug #33961 froze this path."""
        cli = self._make_interactive_cli()
        with patch("cli.load_cli_config", return_value={"approvals": {"destructive_slash_confirm": True}}):
            outcome = _run_on_daemon(
                lambda: cli._confirm_destructive_slash(
                    "new",
                    "This starts a fresh session.\nThe current conversation history will be discarded.",
                ),
                cli,
                platform="win32",
                response=response,
            )

        assert outcome["stdin_called"] is False
        assert outcome["result"] == expected
