"""
Unit tests for session_orchestration/adapters/claude_code.py.

Coverage strategy
-----------------
Everything testable without a live tmux/claude session is covered here:

1. ``_shell_quote`` — pure function.
2. ``_parse_lifecycle`` — pure function; fed sample pane snapshots.
3. ``Capabilities`` declaration — asserts the declared values are correct.
4. ``drive()`` command sequence — asserts the exact tmux commands issued via
   a fake ``TmuxRunner``.
5. ``detect()`` pane-parse — feeds fake pane text through a stubbed capture.
6. ``resume()`` — asserts /clear + drive sequence when in PAUSED_HANDOFF;
   asserts no-op when not in PAUSED_HANDOFF.

LIVE-ONLY (not tested here):
- ``launch()`` — requires real tmux + claude binary.
- ``_handle_dialogs()`` — requires live pane output.
- ``_wait_for_prompt()`` timing with real sleeps.
"""

from __future__ import annotations

import re
import subprocess
from datetime import datetime, timezone
from typing import List
from unittest.mock import MagicMock, call, patch

import pytest

from session_orchestration.adapters.claude_code import (
    ACTIVITY_REGEX,
    HANDOFF_MARKER,
    PROMPT_MARKER,
    ClaudeCodeAdapter,
    TmuxRunner,
    _parse_lifecycle,
    _shell_quote,
)
from session_orchestration.types import Capabilities, SessionHandle, SessionLifecycle


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_handle(pane: str = "hermes-cc-abc:0.0") -> SessionHandle:
    return SessionHandle(
        session_id="abc12345-dead-beef-0000-000000000000",
        tmux_session="hermes-cc-abc12345",
        pane=pane,
        launch_ts=datetime.now(tz=timezone.utc),
    )


class FakeTmuxRunner:
    """Fake TmuxRunner that records calls and returns configurable output."""

    def __init__(self, capture_output: str = "") -> None:
        self.calls: list[list[str]] = []
        self._capture_output = capture_output

    def run(self, args: list[str], check: bool = True) -> str:
        self.calls.append(list(args))
        # Return configured output for capture-pane calls; empty for others.
        if args and args[0] == "capture-pane":
            return self._capture_output
        return ""

    def set_capture(self, text: str) -> None:
        self._capture_output = text


# ---------------------------------------------------------------------------
# _shell_quote
# ---------------------------------------------------------------------------


class TestShellQuote:
    def test_simple_path(self):
        assert _shell_quote("/home/user/project") == "'/home/user/project'"

    def test_path_with_spaces(self):
        assert _shell_quote("/my project/code") == "'/my project/code'"

    def test_path_with_single_quote(self):
        # "it's" → 'it'\''s'
        result = _shell_quote("/it's/a/path")
        assert result == "'/it'\\''s/a/path'"

    def test_empty_string(self):
        assert _shell_quote("") == "''"


# ---------------------------------------------------------------------------
# _parse_lifecycle
# ---------------------------------------------------------------------------


class TestParseLifecycle:
    def test_waiting_user_on_prompt_marker(self):
        """❯ in pane → WAITING_USER."""
        pane = "Some output\n❯ "
        assert _parse_lifecycle(pane) == SessionLifecycle.WAITING_USER

    def test_running_on_activity_marker(self):
        """● in pane (no ❯) → RUNNING."""
        pane = "● Reading file src/auth.py\n  processing..."
        assert _parse_lifecycle(pane) == SessionLifecycle.RUNNING

    def test_paused_handoff_on_handoff_marker(self):
        """HERMES_HANDOFF in pane → PAUSED_HANDOFF, even if ❯ also present."""
        pane = f"Task complete.\n{HANDOFF_MARKER}\n❯ "
        assert _parse_lifecycle(pane) == SessionLifecycle.PAUSED_HANDOFF

    def test_handoff_marker_takes_priority_over_prompt(self):
        """PAUSED_HANDOFF wins when both HANDOFF_MARKER and ❯ are present."""
        pane = f"❯ {HANDOFF_MARKER}"
        assert _parse_lifecycle(pane) == SessionLifecycle.PAUSED_HANDOFF

    def test_running_when_no_markers(self):
        """No markers → RUNNING (default; watcher tracks staleness)."""
        pane = "Claude is thinking...\nAnalyzing codebase..."
        assert _parse_lifecycle(pane) == SessionLifecycle.RUNNING

    def test_empty_pane(self):
        """Empty pane → RUNNING (startup or mid-render)."""
        assert _parse_lifecycle("") == SessionLifecycle.RUNNING

    def test_activity_regex_matches_bullet(self):
        """ACTIVITY_REGEX must match the ● character."""
        assert ACTIVITY_REGEX.search("● Executing bash command")

    def test_activity_regex_no_false_positive(self):
        """ACTIVITY_REGEX must NOT match plain text."""
        assert not ACTIVITY_REGEX.search("No active work")

    def test_prompt_marker_constant(self):
        """PROMPT_MARKER must be the ❯ character."""
        assert PROMPT_MARKER == "❯"

    def test_handoff_marker_in_middle_of_line(self):
        """HANDOFF_MARKER detected mid-line."""
        pane = f"output line\nChecking: {HANDOFF_MARKER} marker\nmore output"
        assert _parse_lifecycle(pane) == SessionLifecycle.PAUSED_HANDOFF

    def test_multiple_activity_markers(self):
        """Multiple ● markers still → RUNNING."""
        pane = "● Read\n● Edit\n● Bash"
        assert _parse_lifecycle(pane) == SessionLifecycle.RUNNING


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


class TestCapabilities:
    def setup_method(self):
        self.adapter = ClaudeCodeAdapter(tmux_runner=FakeTmuxRunner())

    def test_has_hooks_true(self):
        assert self.adapter.capabilities().has_hooks is True

    def test_supports_print_mode_true(self):
        assert self.adapter.capabilities().supports_print_mode is True

    def test_rpc_mode_false(self):
        assert self.adapter.capabilities().rpc_mode is False

    def test_json_mode_true(self):
        assert self.adapter.capabilities().json_mode is True

    def test_idle_indicator_regex_matches_activity(self):
        cap = self.adapter.capabilities()
        assert cap.idle_indicator_regex is not None
        assert cap.idle_indicator_regex.search("● Reading file")

    def test_idle_indicator_regex_does_not_match_prompt(self):
        cap = self.adapter.capabilities()
        assert cap.idle_indicator_regex is not None
        assert not cap.idle_indicator_regex.search("❯ ")

    def test_dialog_handlers_non_empty(self):
        cap = self.adapter.capabilities()
        assert len(cap.dialog_handlers) == 2

    def test_return_type_is_capabilities(self):
        assert isinstance(self.adapter.capabilities(), Capabilities)


# ---------------------------------------------------------------------------
# drive() — command sequence
# ---------------------------------------------------------------------------


class TestDrive:
    def setup_method(self):
        # Pane has ❯ so the readiness check passes immediately.
        self.fake_tmux = FakeTmuxRunner(capture_output="Last line\n❯ ")
        self.adapter = ClaudeCodeAdapter(tmux_runner=self.fake_tmux)
        self.handle = _make_handle()

    def _drive_with_stubbed_load(self, message: str) -> list[list[str]]:
        """Call drive() with _load_buffer stubbed out; return tmux call list."""
        with patch.object(self.adapter, "_load_buffer"):
            self.adapter.drive(self.handle, message)
        return self.fake_tmux.calls

    def test_drive_issues_paste_buffer(self):
        """drive() must call paste-buffer with the buffer name and pane target."""
        calls = self._drive_with_stubbed_load("Hello, Claude!")
        cmd_names = [c[0] for c in calls]
        assert "paste-buffer" in cmd_names

    def test_drive_paste_buffer_targets_correct_pane(self):
        """paste-buffer must target the session's pane."""
        calls = self._drive_with_stubbed_load("test message")
        paste_call = next(c for c in calls if c[0] == "paste-buffer")
        assert self.handle.pane in paste_call

    def test_drive_issues_enter_after_paste(self):
        """drive() must send Enter after pasting to submit the prompt."""
        calls = self._drive_with_stubbed_load("some prompt")
        # Last send-keys must carry Enter.
        send_key_calls = [c for c in calls if c[0] == "send-keys"]
        assert any("Enter" in c for c in send_key_calls)

    def test_drive_checks_readiness_first(self):
        """drive() must issue capture-pane before paste-buffer."""
        calls = self._drive_with_stubbed_load("prompt")
        cmd_names = [c[0] for c in calls]
        capture_idx = cmd_names.index("capture-pane")
        paste_idx = cmd_names.index("paste-buffer")
        assert capture_idx < paste_idx

    def test_drive_paste_buffer_before_enter(self):
        """paste-buffer must come before the final Enter send-keys."""
        calls = self._drive_with_stubbed_load("a multi\nline prompt")
        cmd_names = [c[0] for c in calls]
        paste_idx = cmd_names.index("paste-buffer")
        # There must be a send-keys after the paste.
        later_send_keys = [i for i, n in enumerate(cmd_names) if n == "send-keys" and i > paste_idx]
        assert later_send_keys, "No send-keys after paste-buffer"

    def test_drive_raises_timeout_when_no_prompt(self):
        """drive() raises TimeoutError if the pane never shows ❯."""
        no_prompt_tmux = FakeTmuxRunner(capture_output="● Busy...")
        adapter = ClaudeCodeAdapter(tmux_runner=no_prompt_tmux)
        handle = _make_handle()
        with patch("session_orchestration.adapters.claude_code._LAUNCH_READY_TIMEOUT", 0.1):
            with patch("session_orchestration.adapters.claude_code._POLL_INTERVAL", 0.05):
                with pytest.raises(TimeoutError):
                    adapter.drive(handle, "hi")

    def test_load_buffer_called_with_message(self):
        """_load_buffer must be called with the exact message text."""
        with patch.object(self.adapter, "_load_buffer") as mock_lb:
            self.adapter.drive(self.handle, "specific message content")
        mock_lb.assert_called_once()
        args = mock_lb.call_args[0]
        assert "specific message content" in args


# ---------------------------------------------------------------------------
# detect()
# ---------------------------------------------------------------------------


class TestDetect:
    def _adapter_with_pane(self, pane_text: str) -> ClaudeCodeAdapter:
        return ClaudeCodeAdapter(tmux_runner=FakeTmuxRunner(capture_output=pane_text))

    def test_detect_waiting_user(self):
        adapter = self._adapter_with_pane("last output\n❯ ")
        assert adapter.detect(_make_handle()) == SessionLifecycle.WAITING_USER

    def test_detect_running_via_activity(self):
        adapter = self._adapter_with_pane("● Editing src/main.py")
        assert adapter.detect(_make_handle()) == SessionLifecycle.RUNNING

    def test_detect_paused_handoff(self):
        adapter = self._adapter_with_pane(f"done.\n{HANDOFF_MARKER}\n❯ ")
        assert adapter.detect(_make_handle()) == SessionLifecycle.PAUSED_HANDOFF

    def test_detect_error_on_dead_pane(self):
        """When capture-pane fails (pane gone), detect() → ERROR."""
        class FailingRunner:
            def run(self, args, check=True):
                if args[0] == "capture-pane":
                    raise subprocess.CalledProcessError(1, "tmux")
                return ""

        adapter = ClaudeCodeAdapter(tmux_runner=FailingRunner())
        assert adapter.detect(_make_handle()) == SessionLifecycle.ERROR

    def test_detect_running_default(self):
        """No markers → RUNNING."""
        adapter = self._adapter_with_pane("Claude is processing the request...")
        assert adapter.detect(_make_handle()) == SessionLifecycle.RUNNING


# ---------------------------------------------------------------------------
# resume()
# ---------------------------------------------------------------------------


class TestResume:
    def test_resume_sends_clear_and_drives_when_handoff(self):
        """resume() must send /clear then drive() when in PAUSED_HANDOFF."""
        fake_tmux = FakeTmuxRunner(capture_output=f"{HANDOFF_MARKER}\n❯ ")
        adapter = ClaudeCodeAdapter(tmux_runner=fake_tmux)
        handle = _make_handle()

        with patch.object(adapter, "drive") as mock_drive:
            adapter.resume(handle, "continue from here")

        # /clear must have been sent via send-keys.
        clear_calls = [c for c in fake_tmux.calls if c[0] == "send-keys" and "/clear" in c]
        assert clear_calls, "No /clear send-keys found"

        # drive() must have been called with the new prompt.
        mock_drive.assert_called_once_with(handle, "continue from here")

    def test_resume_noop_when_not_handoff(self):
        """resume() must be a no-op when the session is NOT in PAUSED_HANDOFF."""
        fake_tmux = FakeTmuxRunner(capture_output="● Working...\n")
        adapter = ClaudeCodeAdapter(tmux_runner=fake_tmux)
        handle = _make_handle()

        with patch.object(adapter, "drive") as mock_drive:
            adapter.resume(handle, "irrelevant prompt")

        # drive() must NOT have been called.
        mock_drive.assert_not_called()
        # /clear must NOT have been sent.
        clear_calls = [c for c in fake_tmux.calls if c[0] == "send-keys" and "/clear" in c]
        assert not clear_calls

    def test_resume_noop_when_waiting_user_not_handoff(self):
        """resume() must be a no-op even if ❯ is visible (but no handoff marker)."""
        fake_tmux = FakeTmuxRunner(capture_output="output\n❯ ")
        adapter = ClaudeCodeAdapter(tmux_runner=fake_tmux)
        handle = _make_handle()

        with patch.object(adapter, "drive") as mock_drive:
            adapter.resume(handle, "prompt")

        mock_drive.assert_not_called()
