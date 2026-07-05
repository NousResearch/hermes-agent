"""Unit tests for session_orchestration/adapters/codex.py."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from session_orchestration.adapters.codex import (
    ACTIVITY_REGEX,
    HANDOFF_MARKER,
    PROMPT_FRAGMENT,
    CodexAdapter,
    build_interactive_argv,
    parse_pane_lifecycle,
)
from session_orchestration.types import SessionHandle, SessionLifecycle


def _make_handle(pane: str = "%1") -> SessionHandle:
    return SessionHandle(
        session_id="abc12345-dead-beef-0000-000000000000",
        tmux_session="hermes-codex-abc12345",
        pane=pane,
        launch_ts=datetime.now(tz=timezone.utc),
    )


class FakeTmuxRunner:
    def __init__(self, capture_output: str = "") -> None:
        self.calls: list[list[str]] = []
        self._capture_output = capture_output

    def run(self, args: list[str], check: bool = True) -> str:
        self.calls.append(list(args))
        if args and args[0] == "capture-pane":
            return self._capture_output
        if args and args[0] == "list-panes":
            return "%1"
        return ""

    def set_capture(self, text: str) -> None:
        self._capture_output = text


class TestBuildInteractiveArgv:
    def test_includes_workdir(self):
        assert build_interactive_argv(workdir="/repo") == ["-C", "/repo"]

    def test_includes_model_when_given(self):
        assert build_interactive_argv(workdir="/repo", model="gpt-5.5") == [
            "-C",
            "/repo",
            "--model",
            "gpt-5.5",
        ]


class TestParsePaneLifecycle:
    def test_waiting_user_on_composer(self):
        assert parse_pane_lifecycle(PROMPT_FRAGMENT) == SessionLifecycle.WAITING_USER

    def test_waiting_user_on_contextual_composer(self):
        assert parse_pane_lifecycle("› Explain this codebase") == (
            SessionLifecycle.WAITING_USER
        )

    def test_numbered_selector_is_not_a_composer_without_dialog_text(self):
        assert parse_pane_lifecycle("› 1. Yes, continue") == SessionLifecycle.RUNNING

    def test_waiting_user_on_trust_dialog(self):
        pane = "Do you trust the contents of this directory?\n› 1. Yes, continue"
        assert parse_pane_lifecycle(pane) == SessionLifecycle.WAITING_USER

    def test_waiting_user_on_update_dialog(self):
        pane = "Update available\n› 1. Update now\n  2. Skip"
        assert parse_pane_lifecycle(pane) == SessionLifecycle.WAITING_USER

    def test_paused_handoff_takes_priority(self):
        assert parse_pane_lifecycle(f"{HANDOFF_MARKER}\n{PROMPT_FRAGMENT}") == (
            SessionLifecycle.PAUSED_HANDOFF
        )

    def test_running_on_activity(self):
        assert parse_pane_lifecycle("⠋ thinking") == SessionLifecycle.RUNNING
        assert ACTIVITY_REGEX.search("Running command")


class TestCapabilities:
    def test_capabilities_match_probe_contract(self):
        caps = CodexAdapter(tmux_runner=FakeTmuxRunner()).capabilities()
        assert caps.supports_print_mode is True
        assert caps.has_hooks is False
        assert caps.rpc_mode is False
        assert caps.json_mode is False
        assert caps.idle_indicator_regex is ACTIVITY_REGEX
        assert len(caps.dialog_handlers) == 2


class TestDrive:
    def test_drive_pastes_and_submits(self):
        fake_tmux = FakeTmuxRunner(capture_output=f"ready\n{PROMPT_FRAGMENT}")
        adapter = CodexAdapter(tmux_runner=fake_tmux)
        handle = _make_handle()

        with patch.object(adapter, "_load_buffer") as load_buffer:
            adapter.drive(handle, "hello")

        load_buffer.assert_called_once()
        assert any(c[0] == "paste-buffer" and handle.pane in c for c in fake_tmux.calls)
        assert any(c[0] == "send-keys" and "Enter" in c for c in fake_tmux.calls)

    def test_drive_raises_timeout_when_not_ready(self):
        fake_tmux = FakeTmuxRunner(capture_output="⠋ thinking")
        adapter = CodexAdapter(tmux_runner=fake_tmux)

        with (
            patch("session_orchestration.adapters.codex._LAUNCH_READY_TIMEOUT", 0.01),
            patch("session_orchestration.adapters.codex.time.sleep"),
            pytest.raises(TimeoutError),
        ):
            adapter.drive(_make_handle(), "hello")


class TestLaunch:
    def test_launch_injects_marker_and_runs_codex(self):
        fake_tmux = FakeTmuxRunner(capture_output=PROMPT_FRAGMENT)
        adapter = CodexAdapter(tmux_runner=fake_tmux, binary="codex-test")

        with (
            patch.object(adapter, "_handle_dialogs"),
            patch.object(adapter, "_wait_for_ready"),
            patch("session_orchestration.adapters.codex.os.makedirs"),
        ):
            handle = adapter.launch("/repo", "start")

        new_session = next(c for c in fake_tmux.calls if c[0] == "new-session")
        assert "-e" in new_session
        assert f"HERMES_MARKER_FILE={handle.marker_file}" in new_session
        send_keys = [c for c in fake_tmux.calls if c[0] == "send-keys"]
        assert any("codex-test -C /repo" in c for c in send_keys)

    def test_terminate_swallows_missing_session(self):
        class ErrorOnKillRunner(FakeTmuxRunner):
            def run(self, args: list[str], check: bool = True) -> str:
                if args and args[0] == "kill-session":
                    raise subprocess.CalledProcessError(1, "tmux")
                return super().run(args, check=check)

        CodexAdapter(tmux_runner=ErrorOnKillRunner()).terminate(_make_handle())
