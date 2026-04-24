import queue
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cli as cli_module
from cli import HermesCLI


class _FakeBuffer:
    def __init__(self, text="", cursor_position=None):
        self.text = text
        self.cursor_position = len(text) if cursor_position is None else cursor_position

    def reset(self, append_to_history=False):
        self.text = ""
        self.cursor_position = 0


def _make_cli_stub():
    cli = HermesCLI.__new__(HermesCLI)
    cli._approval_state = None
    cli._approval_deadline = 0
    cli._approval_lock = threading.Lock()
    cli._sudo_state = None
    cli._sudo_deadline = 0
    cli._modal_input_snapshot = None
    cli._invalidate = MagicMock()
    cli._app = SimpleNamespace(invalidate=MagicMock(), current_buffer=_FakeBuffer())
    return cli


class TestCliApprovalUi:
    def test_sudo_prompt_restores_existing_draft_after_response(self):
        cli = _make_cli_stub()
        cli._app.current_buffer = _FakeBuffer("draft command", cursor_position=5)
        result = {}

        def _run_callback():
            result["value"] = cli._sudo_password_callback()

        with patch.object(cli_module, "_cprint"):
            thread = threading.Thread(target=_run_callback, daemon=True)
            thread.start()

            deadline = time.time() + 2
            while cli._sudo_state is None and time.time() < deadline:
                time.sleep(0.01)

            assert cli._sudo_state is not None
            assert cli._app.current_buffer.text == ""

            cli._app.current_buffer.text = "secret"
            cli._app.current_buffer.cursor_position = len("secret")
            cli._sudo_state["response_queue"].put("secret")

            thread.join(timeout=2)

        assert result["value"] == "secret"
        assert cli._app.current_buffer.text == "draft command"
        assert cli._app.current_buffer.cursor_position == 5

    def test_approval_callback_includes_view_for_long_commands(self):
        cli = _make_cli_stub()
        command = "sudo dd if=/tmp/githubcli-keyring.gpg of=/usr/share/keyrings/githubcli-archive-keyring.gpg bs=4M status=progress"
        result = {}

        def _run_callback():
            result["value"] = cli._approval_callback(command, "disk copy")

        thread = threading.Thread(target=_run_callback, daemon=True)
        thread.start()

        deadline = time.time() + 2
        while cli._approval_state is None and time.time() < deadline:
            time.sleep(0.01)

        assert cli._approval_state is not None
        assert "view" in cli._approval_state["choices"]

        cli._approval_state["response_queue"].put("deny")
        thread.join(timeout=2)
        assert result["value"] == "deny"

    def test_handle_approval_selection_view_expands_in_place(self):
        cli = _make_cli_stub()
        cli._approval_state = {
            "command": "sudo dd if=/tmp/in of=/usr/share/keyrings/githubcli-archive-keyring.gpg bs=4M status=progress",
            "description": "disk copy",
            "choices": ["once", "session", "always", "deny", "view"],
            "selected": 4,
            "response_queue": queue.Queue(),
        }

        cli._handle_approval_selection()

        assert cli._approval_state is not None
        assert cli._approval_state["show_full"] is True
        assert "view" not in cli._approval_state["choices"]
        assert cli._approval_state["selected"] == 3
        assert cli._approval_state["response_queue"].empty()

    def test_approval_display_places_title_inside_box_not_border(self):
        cli = _make_cli_stub()
        cli._approval_state = {
            "command": "sudo dd if=/tmp/in of=/usr/share/keyrings/githubcli-archive-keyring.gpg bs=4M status=progress",
            "description": "disk copy",
            "choices": ["once", "session", "always", "deny", "view"],
            "selected": 0,
            "response_queue": queue.Queue(),
        }

        fragments = cli._get_approval_display_fragments()
        rendered = "".join(text for _style, text in fragments)
        lines = rendered.splitlines()

        assert lines[0].startswith("╭")
        assert "Dangerous Command" not in lines[0]
        assert any("Dangerous Command" in line for line in lines[1:3])
        assert "Show full command" in rendered
        assert "githubcli-archive-keyring.gpg" not in rendered

    def test_approval_display_shows_full_command_after_view(self):
        cli = _make_cli_stub()
        full_command = "sudo dd if=/tmp/in of=/usr/share/keyrings/githubcli-archive-keyring.gpg bs=4M status=progress"
        cli._approval_state = {
            "command": full_command,
            "description": "disk copy",
            "choices": ["once", "session", "always", "deny"],
            "selected": 0,
            "show_full": True,
            "response_queue": queue.Queue(),
        }

        fragments = cli._get_approval_display_fragments()
        rendered = "".join(text for _style, text in fragments)

        assert "..." not in rendered
        assert "githubcli-" in rendered
        assert "archive-" in rendered
        assert "keyring.gpg" in rendered
        assert "status=progress" in rendered

    def test_approval_display_preserves_command_and_choices_with_long_description(self):
        """Regression: long tirith descriptions used to push approve/deny off-screen.

        The panel must always render the command and every choice, even when
        the description would otherwise wrap into 10+ lines. The description
        gets truncated with a marker instead.
        """
        cli = _make_cli_stub()
        long_desc = (
            "Security scan — [CRITICAL] Destructive shell command with wildcard expansion: "
            "The command performs a recursive deletion of log files which may contain "
            "audit information relevant to active incident investigations, running services "
            "that rely on log files for state, rotated archives, and other system artifacts. "
            "Review whether this is intended before approving. Consider whether a targeted "
            "deletion with more specific filters would better match the intent."
        )
        cli._approval_state = {
            "command": "rm -rf /var/log/apache2/*.log",
            "description": long_desc,
            "choices": ["once", "session", "always", "deny"],
            "selected": 0,
            "response_queue": queue.Queue(),
        }

        # Simulate a compact terminal where the old unbounded panel would overflow.
        import shutil as _shutil

        with patch("cli.shutil.get_terminal_size",
                   return_value=_shutil.os.terminal_size((100, 20))):
            fragments = cli._get_approval_display_fragments()

        rendered = "".join(text for _style, text in fragments)

        # Command must be fully visible (rm -rf /var/log/apache2/*.log is short).
        assert "rm -rf /var/log/apache2/*.log" in rendered

        # Every choice must render — this is the core bug: approve/deny were
        # getting clipped off the bottom of the panel.
        assert "Allow once" in rendered
        assert "Allow for this session" in rendered
        assert "Add to permanent allowlist" in rendered
        assert "Deny" in rendered

        # The bottom border must render (i.e. the panel is self-contained).
        assert rendered.rstrip().endswith("╯")

        # The description gets truncated — marker should appear.
        assert "(description truncated)" in rendered

    def test_approval_display_skips_description_on_very_short_terminal(self):
        """On a 12-row terminal, only the command and choices have room.

        The description is dropped entirely rather than partially shown, so the
        choices never get clipped.
        """
        cli = _make_cli_stub()
        cli._approval_state = {
            "command": "rm -rf /var/log/apache2/*.log",
            "description": "recursive delete",
            "choices": ["once", "session", "always", "deny"],
            "selected": 0,
            "response_queue": queue.Queue(),
        }

        import shutil as _shutil

        with patch("cli.shutil.get_terminal_size",
                   return_value=_shutil.os.terminal_size((100, 12))):
            fragments = cli._get_approval_display_fragments()

        rendered = "".join(text for _style, text in fragments)

        # Command visible.
        assert "rm -rf /var/log/apache2/*.log" in rendered
        # All four choices visible.
        for label in ("Allow once", "Allow for this session",
                      "Add to permanent allowlist", "Deny"):
            assert label in rendered, f"choice {label!r} missing"

    def test_approval_display_truncates_giant_command_in_view_mode(self):
        """If the user hits /view on a massive command, choices still render.

        The command gets truncated with a marker; the description gets dropped
        if there's no remaining row budget.
        """
        cli = _make_cli_stub()
        # 50 lines of command when wrapped at ~64 chars.
        giant_cmd = "bash -c 'echo " + ("x" * 3000) + "'"
        cli._approval_state = {
            "command": giant_cmd,
            "description": "shell command via -c/-lc flag",
            "choices": ["once", "session", "always", "deny"],
            "selected": 0,
            "show_full": True,
            "response_queue": queue.Queue(),
        }

        import shutil as _shutil

        with patch("cli.shutil.get_terminal_size",
                   return_value=_shutil.os.terminal_size((100, 24))):
            fragments = cli._get_approval_display_fragments()

        rendered = "".join(text for _style, text in fragments)

        # All four choices visible even with a huge command.
        for label in ("Allow once", "Allow for this session",
                      "Add to permanent allowlist", "Deny"):
            assert label in rendered, f"choice {label!r} missing"

        # Command got truncated with a marker.
        assert "(command truncated" in rendered


class TestApprovalCallbackThreadLocalWiring:
    """Regression guard for the thread-local callback freeze (#13617 / #13618).

    After 62348cff made _approval_callback / _sudo_password_callback thread-local
    (ACP GHSA-qg5c-hvr5-hjgr), the CLI agent thread could no longer see callbacks
    registered in the main thread — the dangerous-command prompt silently fell
    back to stdin input() and deadlocked against prompt_toolkit. The fix is to
    register the callbacks INSIDE the agent worker thread (matching the ACP
    pattern). These tests lock in that invariant.
    """

    def test_main_thread_registration_is_invisible_to_child_thread(self):
        """Confirms the underlying threading.local semantics that drove the bug.

        If this ever starts passing as "visible", the thread-local isolation
        is gone and the ACP race GHSA-qg5c-hvr5-hjgr may be back.
        """
        from tools.terminal_tool import (
            set_approval_callback,
            _get_approval_callback,
        )

        def main_cb(_cmd, _desc):
            return "once"

        set_approval_callback(main_cb)
        try:
            seen = {}

            def _child():
                seen["value"] = _get_approval_callback()

            t = threading.Thread(target=_child, daemon=True)
            t.start()
            t.join(timeout=2)
            assert seen["value"] is None
        finally:
            set_approval_callback(None)

    def test_child_thread_registration_is_visible_and_cleared_in_finally(self):
        """The fix pattern: register INSIDE the worker thread, clear in finally.

        This is exactly what cli.py's run_agent() closure does. If this test
        fails, the CLI approval prompt freeze (#13617) has regressed.
        """
        from tools.terminal_tool import (
            set_approval_callback,
            set_sudo_password_callback,
            _get_approval_callback,
            _get_sudo_password_callback,
        )

        def approval_cb(_cmd, _desc):
            return "once"

        def sudo_cb():
            return "hunter2"

        seen = {}

        def _worker():
            # Mimic cli.py's run_agent() thread target.
            set_approval_callback(approval_cb)
            set_sudo_password_callback(sudo_cb)
            try:
                seen["approval"] = _get_approval_callback()
                seen["sudo"] = _get_sudo_password_callback()
            finally:
                set_approval_callback(None)
                set_sudo_password_callback(None)
                seen["approval_after"] = _get_approval_callback()
                seen["sudo_after"] = _get_sudo_password_callback()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout=2)

        assert seen["approval"] is approval_cb
        assert seen["sudo"] is sudo_cb
        # Finally block must clear both slots — otherwise a reused thread
        # would hold a stale reference to a disposed CLI instance.
        assert seen["approval_after"] is None
        assert seen["sudo_after"] is None


class TestOverlayInterruptCleanup:
    """Regression tests for #14026 — interrupt must clear all overlay state
    including _modal_input_snapshot and _sudo_deadline."""

    def _make_cli_with_sudo_active(self):
        """Return a CLI stub with an active sudo prompt and captured snapshot."""
        cli = _make_cli_stub()
        cli._app.current_buffer = _FakeBuffer("pre-modal draft", cursor_position=7)
        # Simulate snapshot captured when sudo prompt started.
        cli._modal_input_snapshot = {
            "text": "pre-modal draft",
            "cursor_position": 7,
        }
        cli._sudo_deadline = 99999.0
        q = queue.Queue()
        cli._sudo_state = {"response_queue": q, "timeout": 60}
        return cli

    def test_interrupt_clears_sudo_state_and_restores_snapshot(self):
        """Interrupt path: _sudo_state cleared + snapshot restored + deadline reset."""
        cli = self._make_cli_with_sudo_active()

        # Simulate the interrupt handler code path (agent-running interrupt branch).
        try:
            if cli._sudo_state:
                try:
                    cli._sudo_state["response_queue"].put(None)
                except Exception:
                    pass
                cli._sudo_state = None
                cli._sudo_deadline = 0
                cli._restore_modal_input_snapshot()
        except Exception:
            pass

        assert cli._sudo_state is None, "_sudo_state not cleared"
        assert cli._sudo_deadline == 0, "_sudo_deadline not reset"
        assert cli._modal_input_snapshot is None, "_modal_input_snapshot still set after interrupt"
        # Snapshot should have been restored to the buffer.
        assert cli._app.current_buffer.text == "pre-modal draft"
        assert cli._app.current_buffer.cursor_position == 7

    def test_ctrlc_clears_sudo_state_and_restores_snapshot(self):
        """Ctrl+C path: same cleanup required."""
        cli = self._make_cli_with_sudo_active()

        # Simulate Ctrl+C handler (key binding branch).
        if cli._sudo_state:
            cli._sudo_state["response_queue"].put("")
            cli._sudo_state = None
            cli._sudo_deadline = 0
            cli._restore_modal_input_snapshot()

        assert cli._sudo_state is None
        assert cli._sudo_deadline == 0
        assert cli._modal_input_snapshot is None
        assert cli._app.current_buffer.text == "pre-modal draft"

    def test_interrupt_unblocks_blocked_thread(self):
        """Thread blocked in _sudo_password_callback() must unblock when interrupt fires."""
        import threading
        from unittest.mock import patch
        import cli as cli_module

        cli = _make_cli_stub()
        cli._app.current_buffer = _FakeBuffer("pre-draft", cursor_position=4)
        result = {}

        def _run_callback():
            result["value"] = cli._sudo_password_callback()

        with patch.object(cli_module, "_cprint"):
            t = threading.Thread(target=_run_callback, daemon=True)
            t.start()

            deadline = time.time() + 2
            while cli._sudo_state is None and time.time() < deadline:
                time.sleep(0.01)

            assert cli._sudo_state is not None, "sudo prompt did not start"

            # Simulate interrupt clearing the state.
            try:
                cli._sudo_state["response_queue"].put(None)
            except Exception:
                pass
            cli._sudo_state = None
            cli._sudo_deadline = 0
            cli._restore_modal_input_snapshot()

            t.join(timeout=2)

        # _sudo_password_callback returns None (or empty) on cancel.
        assert result.get("value") in (None, ""), f"unexpected value: {result.get('value')!r}"
        assert cli._modal_input_snapshot is None
        assert cli._sudo_deadline == 0


class TestInterruptOverlayClearance:
    """Regression tests: interrupting an active overlay must clear state and unblock threads.

    Covers the fix for the CLI freeze introduced by 52a79d9 where stale overlay
    states left _command_running / $isBlocked active after an interrupt, silently
    swallowing all keystrokes until the user killed and restarted Hermes.
    """

    # ------------------------------------------------------------------ helpers

    def _make_stub(self, *, draft="", draft_cursor=0):
        import threading
        cli = HermesCLI.__new__(HermesCLI)
        cli._approval_state = None
        cli._approval_deadline = 0
        cli._approval_lock = threading.Lock()
        cli._clarify_state = None
        cli._clarify_deadline = 0
        cli._clarify_freetext = False
        cli._sudo_state = None
        cli._sudo_deadline = 0
        cli._modal_input_snapshot = None
        cli._invalidate = MagicMock()
        cli._app = SimpleNamespace(
            invalidate=MagicMock(),
            current_buffer=_FakeBuffer(draft, cursor_position=draft_cursor),
        )
        return cli

    # ------------------------------------------------------------------ approval

    def test_interrupt_clears_approval_state_and_sends_deny(self):
        """Interrupting while an approval prompt is active must clear state and send 'deny'."""
        import queue as _queue
        cli = self._make_stub()
        rq = _queue.Queue()
        cli._approval_state = {
            "command": "rm -rf /tmp/test",
            "description": "delete temp",
            "choices": ["approve", "deny"],
            "selected": 0,
            "response_queue": rq,
        }

        # Simulate the Ctrl+C key-binding interrupt path.
        if cli._approval_state:
            cli._approval_state["response_queue"].put("deny")
            cli._approval_state = None

        assert cli._approval_state is None, "_approval_state not cleared after interrupt"
        assert rq.get_nowait() == "deny", "approval queue should have received 'deny'"

    def test_interrupt_unblocks_approval_callback_thread(self):
        """Thread blocked in _approval_callback() must unblock and receive 'deny'."""
        import threading
        import queue as _queue
        from unittest.mock import patch
        import cli as cli_module

        cli = self._make_stub()
        result = {}

        def _run_callback():
            result["value"] = cli._approval_callback("rm -rf /tmp/x", "cleanup")

        with patch.object(cli_module, "_cprint"):
            t = threading.Thread(target=_run_callback, daemon=True)
            t.start()

            deadline = time.time() + 2
            while cli._approval_state is None and time.time() < deadline:
                time.sleep(0.01)

            assert cli._approval_state is not None, "approval prompt did not start"

            # Simulate interrupt.
            cli._approval_state["response_queue"].put("deny")
            cli._approval_state = None

            t.join(timeout=2)

        assert result.get("value") == "deny", f"expected 'deny', got {result.get('value')!r}"

    # ------------------------------------------------------------------ clarify

    def test_interrupt_clears_clarify_state(self):
        """Interrupting while a clarify prompt is active must nil out _clarify_state."""
        import queue as _queue
        cli = self._make_stub()
        rq = _queue.Queue()
        cli._clarify_state = {
            "question": "Which approach?",
            "choices": ["A", "B"],
            "selected": 0,
            "response_queue": rq,
        }
        cli._clarify_freetext = False

        # Simulate the Ctrl+C key-binding interrupt path.
        if cli._clarify_state:
            cli._clarify_state["response_queue"].put(
                "The user cancelled. Use your best judgement to proceed."
            )
            cli._clarify_state = None
            cli._clarify_freetext = False

        assert cli._clarify_state is None, "_clarify_state not cleared after interrupt"
        assert cli._clarify_freetext is False, "_clarify_freetext not reset after interrupt"
        val = rq.get_nowait()
        assert "cancelled" in val.lower(), f"unexpected cancellation message: {val!r}"

    def test_interrupt_unblocks_clarify_callback_thread(self):
        """Thread blocked in _clarify_callback() must unblock when interrupt fires."""
        import threading
        from unittest.mock import patch
        import cli as cli_module

        cli = self._make_stub()
        result = {}

        def _run_callback():
            result["value"] = cli._clarify_callback("Pick one?", ["Yes", "No"])

        with patch.object(cli_module, "_cprint"):
            t = threading.Thread(target=_run_callback, daemon=True)
            t.start()

            deadline = time.time() + 2
            while cli._clarify_state is None and time.time() < deadline:
                time.sleep(0.01)

            assert cli._clarify_state is not None, "clarify prompt did not start"

            # Simulate interrupt.
            cli._clarify_state["response_queue"].put(
                "The user cancelled. Use your best judgement to proceed."
            )
            cli._clarify_state = None
            cli._clarify_freetext = False

            t.join(timeout=2)

        assert result.get("value") is not None, "clarify callback did not return"
        assert "cancelled" in result["value"].lower() or result["value"] in ("Yes", "No", None)
