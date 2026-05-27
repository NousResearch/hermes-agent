"""Regression tests for issue #30768: /reset and /new freeze on Windows.

``_prompt_text_input_modal`` uses a queue-based modal that relies on
prompt_toolkit key bindings receiving keyboard events.  On Windows the
prompt_toolkit input channel can deadlock when the modal is entered from
the ``process_loop`` daemon thread.  The fix falls back to the simpler
``_prompt_text_input`` (stdin-based) prompt on Windows and non-main threads.

These tests verify:
1. Windows detection triggers the stdin fallback
2. Non-main thread detection triggers the stdin fallback
3. macOS/Linux main-thread path still uses the modal (no regression)
4. No-app path still uses the stdin fallback (existing behavior)
5. Empty choices returns None (existing behavior)
"""

import queue
import sys
import threading
import time
from unittest.mock import MagicMock, patch

import pytest


def _make_cli():
    """Minimal HermesCLI shell exposing prompt/modal helpers."""
    import cli as cli_mod

    obj = object.__new__(cli_mod.HermesCLI)
    obj._app = MagicMock()
    obj._status_bar_visible = True
    obj._last_invalidate = 0.0
    obj._modal_input_snapshot = None
    obj._slash_confirm_state = None
    obj._slash_confirm_deadline = 0
    return obj


# ---------------------------------------------------------------------------
# Sample choices used across tests
# ---------------------------------------------------------------------------
_SAMPLE_CHOICES = [
    ("once", "Approve Once", "proceed this time only"),
    ("always", "Always Approve", "proceed and silence this prompt permanently"),
    ("cancel", "Cancel", "keep current conversation"),
]


class TestModalWindowsFallback:
    """Windows dead-lock regression tests for _prompt_text_input_modal."""

    def test_windows_falls_back_to_stdin(self):
        """On Windows, _prompt_text_input_modal should use _prompt_text_input."""
        cli = _make_cli()

        with patch.object(sys, "platform", "win32"), \
             patch.object(cli, "_prompt_text_input", return_value="1") as mock_stdin:
            result = cli._prompt_text_input_modal(
                title="⚠️  /new — destroys conversation state",
                detail="This starts a fresh session.",
                choices=_SAMPLE_CHOICES,
            )

        # The stdin-based fallback was used, not the modal queue path.
        mock_stdin.assert_called_once_with("Choice [1/2/3]: ")
        assert result == "1"

    def test_non_main_thread_falls_back_to_stdin(self):
        """Off the main thread, _prompt_text_input_modal should use stdin fallback."""
        cli = _make_cli()
        result_holder = {}

        def run_on_daemon():
            # Patch platform to "linux" so the Windows check doesn't short-circuit.
            with patch.object(sys, "platform", "linux"), \
                 patch.object(cli, "_prompt_text_input", return_value="2") as mock_stdin:
                result_holder["result"] = cli._prompt_text_input_modal(
                    title="⚠️  /reset",
                    detail="This starts a fresh session.",
                    choices=_SAMPLE_CHOICES,
                )
                result_holder["stdin_called"] = mock_stdin.called

        t = threading.Thread(target=run_on_daemon, daemon=True)
        t.start()
        t.join(timeout=2.0)
        assert not t.is_alive(), "daemon thread hung — modal deadlocked"
        assert result_holder["stdin_called"] is True
        assert result_holder["result"] == "2"

    def test_main_thread_non_windows_uses_modal(self):
        """On macOS/Linux main thread, the queue-based modal is still used."""
        cli = _make_cli()

        # We need to simulate the modal receiving a response. We'll patch
        # the response_queue to immediately return a value.
        with patch.object(sys, "platform", "darwin"), \
             patch.object(cli, "_capture_modal_input_snapshot"), \
             patch.object(cli, "_restore_modal_input_snapshot"), \
             patch.object(cli, "_invalidate"):
            # Start the modal in a way that it will receive a response
            # immediately via the queue.
            original_queue = queue.Queue
            original_time = time.monotonic

            def _fake_modal_flow(*args, **kwargs):
                """Simulate the modal flow: set state, put response, return."""
                # We'll directly test that the modal path is entered by
                # checking that _slash_confirm_state was set.
                pass

            # Since we can't easily mock the internal queue, let's test
            # that the modal path is entered by checking that
            # _prompt_text_input was NOT called.
            with patch.object(cli, "_prompt_text_input") as mock_stdin:
                # Set up a response that will be put into the queue
                # after the modal starts waiting.
                def _submit_after_delay():
                    time.sleep(0.2)
                    state = cli._slash_confirm_state
                    if state and "response_queue" in state:
                        state["response_queue"].put("once")

                submitter = threading.Thread(target=_submit_after_delay, daemon=True)
                submitter.start()

                result = cli._prompt_text_input_modal(
                    title="⚠️  /new",
                    detail="This starts a fresh session.",
                    choices=_SAMPLE_CHOICES,
                    timeout=5,
                )

                submitter.join(timeout=2.0)

            # The stdin fallback should NOT have been called.
            mock_stdin.assert_not_called()
            # The result should be "once" from the simulated modal response.
            assert result == "once"

    def test_no_app_falls_back_to_stdin(self):
        """Without a prompt_toolkit app, always use stdin fallback."""
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

    def test_empty_choices_returns_none(self):
        """Empty choices list should return None without prompting."""
        cli = _make_cli()

        with patch.object(cli, "_prompt_text_input") as mock_stdin:
            result = cli._prompt_text_input_modal(
                title="Test",
                detail="Test",
                choices=[],
            )

        mock_stdin.assert_not_called()
        assert result is None

    def test_windows_fallback_does_not_set_modal_state(self):
        """Verify Windows fallback doesn't leave _slash_confirm_state set."""
        cli = _make_cli()

        with patch.object(sys, "platform", "win32"), \
             patch.object(cli, "_prompt_text_input", return_value="1"):
            cli._prompt_text_input_modal(
                title="⚠️  /reset",
                detail="This starts a fresh session.",
                choices=_SAMPLE_CHOICES,
            )

        assert cli._slash_confirm_state is None

    def test_non_main_thread_fallback_does_not_set_modal_state(self):
        """Verify daemon-thread fallback doesn't leave modal state set."""
        cli = _make_cli()
        errors = []

        def run_on_daemon():
            try:
                with patch.object(sys, "platform", "linux"), \
                     patch.object(cli, "_prompt_text_input", return_value="1"):
                    cli._prompt_text_input_modal(
                        title="⚠️  /new",
                        detail="This starts a fresh session.",
                        choices=_SAMPLE_CHOICES,
                    )
                if cli._slash_confirm_state is not None:
                    errors.append("_slash_confirm_state should be None")
            except Exception as exc:
                errors.append(str(exc))

        t = threading.Thread(target=run_on_daemon, daemon=True)
        t.start()
        t.join(timeout=2.0)
        assert not errors, f"unexpected errors: {errors}"
        assert cli._slash_confirm_state is None


class TestConfirmDestructiveSlashWindows:
    """Integration-level tests for _confirm_destructive_slash on Windows."""

    def test_confirm_destructive_slash_bypasses_modal_on_windows(self):
        """_confirm_destructive_slash should work on Windows via stdin fallback."""
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

        with patch.object(sys, "platform", "win32"), \
             patch.object(cli, "_prompt_text_input", return_value="1"), \
             patch("cli.load_cli_config", return_value={"approvals": {"destructive_slash_confirm": True}}):
            result = cli._confirm_destructive_slash(
                "new",
                "This starts a fresh session.\nThe current conversation history will be discarded.",
            )

        assert result == "once"

    def test_confirm_destructive_slash_cancelled_on_windows(self):
        """Cancellation via stdin fallback works on Windows."""
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

        with patch.object(sys, "platform", "win32"), \
             patch.object(cli, "_prompt_text_input", return_value="3"), \
             patch("cli.load_cli_config", return_value={"approvals": {"destructive_slash_confirm": True}}):
            result = cli._confirm_destructive_slash(
                "reset",
                "This starts a fresh session.\nThe current conversation history will be discarded.",
            )

        # Choice "3" normalizes to "cancel", which returns None.
        assert result is None
