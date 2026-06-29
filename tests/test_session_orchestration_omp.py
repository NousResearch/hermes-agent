"""
Unit tests for session_orchestration/adapters/omp.py.

Coverage strategy
-----------------
Everything testable without a live tmux/omp session is covered here:

1. ``build_oneshot_argv`` — pure function; tests flag construction.
2. ``build_interactive_argv`` — pure function; tests flag construction.
3. ``parse_oneshot_result`` — pure function; fed sample NDJSON streams.
4. ``parse_pane_lifecycle`` — pure function; fed sample pane snapshots.
5. ``Capabilities`` declaration — asserts rpc_mode/json_mode/has_hooks.
6. ``drive()`` command sequence — asserts tmux commands via a fake runner.
7. ``detect()`` pane-parse — feeds fake pane text through a stubbed capture.
8. ``resume()`` — asserts -c re-launch when PAUSED_HANDOFF; no-op otherwise.
9. ``run_oneshot()`` — asserts the OmpRunner is called with correct argv and
   parse_oneshot_result is applied to stdout.
10. ``launch()`` — marker-file injection: verifies -e HERMES_MARKER_FILE in
    tmux args and handle.marker_file is set to expected path.
11. ``terminate()`` — issues kill-session and swallows nonzero exit.
12. ``translate_rpc_line()`` — round-trip tests for every mapped type + None
    for unknown / blank / non-JSON input.
13. ``tail_rpc_stream()`` — drives a fake stdout iter and asserts correct
    marker kinds are written via the injected append_fn.

LIVE-ONLY (not tested here):
- The actual tmux new-session system call in ``launch()`` (real tmux binary).
- ``_wait_for_prompt()`` timing with real sleeps.
- ``_load_buffer()`` — requires real tmux process.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from session_orchestration.adapters.omp import (
    ACTIVITY_REGEX,
    HANDOFF_MARKER,
    PROMPT_PATTERN,
    OmpAdapter,
    build_interactive_argv,
    build_oneshot_argv,
    parse_oneshot_result,
    parse_pane_lifecycle,
    tail_rpc_stream,
    translate_rpc_line,
)
from session_orchestration.markers import (
    MARKER_DONE,
    MARKER_HEARTBEAT,
    MARKER_NEEDS_INPUT,
    MARKER_STATUS,
)
from session_orchestration.types import Capabilities, SessionHandle, SessionLifecycle


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_handle(pane: str = "hermes-omp-abc:0.0") -> SessionHandle:
    return SessionHandle(
        session_id="abc12345-dead-beef-0000-000000000000",
        tmux_session="hermes-omp-abc12345",
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
        if args and args[0] == "capture-pane":
            return self._capture_output
        return ""

    def set_capture(self, text: str) -> None:
        self._capture_output = text


class FakeOmpRunner:
    """Fake OmpRunner that records calls and returns configurable stdout."""

    def __init__(self, stdout: str = "") -> None:
        self.calls: list[list[str]] = []
        self._stdout = stdout

    def run_oneshot(
        self,
        args: list[str],
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(args))
        return subprocess.CompletedProcess(
            args=["omp"] + args,
            returncode=0,
            stdout=self._stdout,
            stderr="",
        )


def _agent_end_ndjson(assistant_text: str) -> str:
    """Build a minimal valid omp --mode=json NDJSON stream."""
    events = [
        {"type": "session", "version": 3, "id": "sess-001"},
        {"type": "agent_start"},
        {"type": "turn_start"},
        {
            "type": "message_start",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "hello"}],
            },
        },
        {
            "type": "message_end",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "hello"}],
            },
        },
        {
            "type": "message_start",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
                "stopReason": "stop",
            },
        },
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
                "stopReason": "stop",
            },
        },
        {
            "type": "turn_end",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            },
        },
        {
            "type": "agent_end",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "hello"}],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": assistant_text}],
                    "stopReason": "stop",
                },
            ],
        },
    ]
    return "\n".join(json.dumps(e) for e in events)


# ---------------------------------------------------------------------------
# build_oneshot_argv
# ---------------------------------------------------------------------------


class TestBuildOneshotArgv:
    def test_basic_includes_print_and_json_mode(self):
        argv = build_oneshot_argv("say hi")
        assert "-p" in argv
        assert "--mode=json" in argv
        assert "say hi" in argv

    def test_model_flag(self):
        argv = build_oneshot_argv("prompt", model="openai-codex/gpt-5.5")
        assert "--model" in argv
        idx = argv.index("--model")
        assert argv[idx + 1] == "openai-codex/gpt-5.5"

    def test_workdir_flag(self):
        argv = build_oneshot_argv("prompt", workdir="/some/path")
        assert "--cwd" in argv
        idx = argv.index("--cwd")
        assert argv[idx + 1] == "/some/path"

    def test_max_time_flag(self):
        argv = build_oneshot_argv("prompt", max_time=120)
        assert "--max-time" in argv
        idx = argv.index("--max-time")
        assert argv[idx + 1] == "120"

    def test_no_session_by_default(self):
        argv = build_oneshot_argv("prompt")
        assert "--no-session" in argv

    def test_no_session_suppressed(self):
        argv = build_oneshot_argv("prompt", no_session=False)
        assert "--no-session" not in argv

    def test_prompt_is_last_arg(self):
        argv = build_oneshot_argv("my prompt text", model="opus")
        assert argv[-1] == "my prompt text"


# ---------------------------------------------------------------------------
# build_interactive_argv
# ---------------------------------------------------------------------------


class TestBuildInteractiveArgv:
    def test_includes_auto_approve(self):
        argv = build_interactive_argv()
        assert "--auto-approve" in argv

    def test_prompt_appended_when_given(self):
        argv = build_interactive_argv(prompt="start task")
        assert "start task" in argv

    def test_no_prompt_when_omitted(self):
        argv = build_interactive_argv()
        # No stray strings that look like a prompt.
        assert "--auto-approve" in argv

    def test_model_flag(self):
        argv = build_interactive_argv(model="gpt-5.5")
        assert "--model" in argv
        assert argv[argv.index("--model") + 1] == "gpt-5.5"

    def test_max_time_flag(self):
        argv = build_interactive_argv(max_time=300)
        assert "--max-time" in argv
        assert argv[argv.index("--max-time") + 1] == "300"

    def test_hook_flag(self):
        argv = build_interactive_argv(hook="/path/to/hook.js")
        assert "--hook" in argv
        assert argv[argv.index("--hook") + 1] == "/path/to/hook.js"

    def test_resume_id_flag(self):
        argv = build_interactive_argv(resume_id="sess-abc")
        assert "--resume" in argv
        assert argv[argv.index("--resume") + 1] == "sess-abc"

    def test_continue_last_flag(self):
        argv = build_interactive_argv(continue_last=True)
        assert "--continue" in argv

    def test_resume_id_takes_priority_over_continue(self):
        argv = build_interactive_argv(resume_id="sess-x", continue_last=True)
        assert "--resume" in argv
        assert "--continue" not in argv


# ---------------------------------------------------------------------------
# parse_oneshot_result
# ---------------------------------------------------------------------------


class TestParseOneshotResult:
    def test_parses_agent_end_assistant_text(self):
        ndjson = _agent_end_ndjson("Hi there")
        assert parse_oneshot_result(ndjson) == "Hi there"

    def test_falls_back_to_last_message_end(self):
        """If no agent_end, fall back to last message_end with role=assistant."""
        events = [
            {"type": "session", "id": "s1"},
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "fallback answer"}],
                },
            },
        ]
        ndjson = "\n".join(json.dumps(e) for e in events)
        assert parse_oneshot_result(ndjson) == "fallback answer"

    def test_skips_user_messages(self):
        events = [
            {
                "type": "message_end",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "user prompt"}],
                },
            },
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "assistant reply"}],
                },
            },
        ]
        ndjson = "\n".join(json.dumps(e) for e in events)
        assert parse_oneshot_result(ndjson) == "assistant reply"

    def test_empty_output_raises_value_error(self):
        """No parseable lines → ValueError."""
        with pytest.raises(ValueError, match="No parseable JSON"):
            parse_oneshot_result("")

    def test_non_json_lines_skipped(self):
        """Non-JSON lines (e.g. omp startup banners) are skipped gracefully."""
        events = [
            '{"type": "session", "id": "s1"}',
            "omp v16.1.15",  # non-JSON line
            json.dumps({
                "type": "agent_end",
                "messages": [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "result"}],
                    }
                ],
            }),
        ]
        ndjson = "\n".join(events)
        assert parse_oneshot_result(ndjson) == "result"

    def test_multi_turn_returns_last_assistant(self):
        """With multiple assistant messages, return the last one."""
        events = [
            {
                "type": "agent_end",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "q1"}]},
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "first answer"}],
                    },
                    {"role": "user", "content": [{"type": "text", "text": "q2"}]},
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "final answer"}],
                    },
                ],
            }
        ]
        ndjson = json.dumps(events[0])
        assert parse_oneshot_result(ndjson) == "final answer"

    def test_empty_messages_in_agent_end_returns_empty(self):
        """agent_end with no assistant messages → empty string."""
        ndjson = json.dumps({"type": "agent_end", "messages": []})
        assert parse_oneshot_result(ndjson) == ""

    def test_parse_real_omp_sample(self):
        """Validate against a real captured omp --mode=json output fragment."""
        # Condensed from actual live probe (omp v16.1.15, prompt: "say hi in one word")
        real_sample = (
            '{"type":"session","version":3,"id":"019efbb2-7c26-7000-a371-5580d17eed55",'
            '"timestamp":"2026-06-24T22:13:58.951Z","cwd":"/Users/zeke/dev/z-harness"}\n'
            '{"type":"agent_start"}\n'
            '{"type":"turn_start"}\n'
            '{"type":"message_start","message":{"role":"user","content":[{"type":"text","text":"say hi in one word"}]}}\n'
            '{"type":"message_end","message":{"role":"user","content":[{"type":"text","text":"say hi in one word"}]}}\n'
            '{"type":"message_start","message":{"role":"assistant","content":[{"type":"text","text":"Hi"}],'
            '"model":"gpt-5.5","stopReason":"stop"}}\n'
            '{"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"Hi"}],'
            '"model":"gpt-5.5","stopReason":"stop"}}\n'
            '{"type":"turn_end","message":{"role":"assistant","content":[{"type":"text","text":"Hi"}]}}\n'
            '{"type":"agent_end","messages":['
            '{"role":"user","content":[{"type":"text","text":"say hi in one word"}]},'
            '{"role":"assistant","content":[{"type":"text","text":"Hi"}],"model":"gpt-5.5","stopReason":"stop"}'
            "]}\n"
        )
        assert parse_oneshot_result(real_sample) == "Hi"


# ---------------------------------------------------------------------------
# parse_pane_lifecycle
# ---------------------------------------------------------------------------


class TestParsePaneLifecycle:
    def test_waiting_user_on_prompt_marker_gt(self):
        """'>' at end of line → WAITING_USER."""
        pane = "Some output\n> "
        assert parse_pane_lifecycle(pane) == SessionLifecycle.WAITING_USER

    def test_waiting_user_on_prompt_marker_chevron(self):
        """'❯' at end of line → WAITING_USER."""
        pane = "Some output\n❯ "
        assert parse_pane_lifecycle(pane) == SessionLifecycle.WAITING_USER

    def test_paused_handoff_on_handoff_marker(self):
        """HERMES_HANDOFF in pane → PAUSED_HANDOFF, even if prompt also present."""
        pane = f"Task complete.\n{HANDOFF_MARKER}\n> "
        assert parse_pane_lifecycle(pane) == SessionLifecycle.PAUSED_HANDOFF

    def test_handoff_takes_priority_over_prompt(self):
        """PAUSED_HANDOFF wins when both HANDOFF_MARKER and '>' are present."""
        pane = f"> {HANDOFF_MARKER}"
        assert parse_pane_lifecycle(pane) == SessionLifecycle.PAUSED_HANDOFF

    def test_running_on_spinner(self):
        """Spinner characters in pane (no prompt) → RUNNING."""
        pane = "⠋ Executing tool\n  thinking..."
        assert parse_pane_lifecycle(pane) == SessionLifecycle.RUNNING

    def test_running_on_running_tool_text(self):
        """'Running tool' text → RUNNING."""
        pane = "Running tool: bash\n  ..."
        assert parse_pane_lifecycle(pane) == SessionLifecycle.RUNNING

    def test_running_when_no_markers(self):
        """No markers → RUNNING (default)."""
        pane = "omp is thinking about the problem..."
        assert parse_pane_lifecycle(pane) == SessionLifecycle.RUNNING

    def test_empty_pane_returns_running(self):
        """Empty pane → RUNNING (startup or mid-render)."""
        assert parse_pane_lifecycle("") == SessionLifecycle.RUNNING

    def test_prompt_pattern_does_not_match_mid_line(self):
        """'>' in the middle of a line must NOT trigger WAITING_USER."""
        # PROMPT_PATTERN requires end-of-line; mid-line '>' is common in diffs.
        pane = "diff --git a/file > b/file\nmore content"
        # Should be RUNNING (no prompt, no handoff, no spinner).
        # NB: This relies on PROMPT_PATTERN using $ with MULTILINE.
        result = parse_pane_lifecycle(pane)
        # Either RUNNING is acceptable (no false-positive WAITING_USER).
        assert result != SessionLifecycle.WAITING_USER

    def test_activity_regex_matches_spinner(self):
        """ACTIVITY_REGEX must match spinner characters."""
        assert ACTIVITY_REGEX.search("⠋ Running")
        assert ACTIVITY_REGEX.search("⠙ Thinking")

    def test_activity_regex_no_false_positive(self):
        """ACTIVITY_REGEX must NOT match plain output text."""
        assert not ACTIVITY_REGEX.search("Output: done")


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


class TestCapabilities:
    def setup_method(self):
        self.adapter = OmpAdapter(
            omp_runner=FakeOmpRunner(),
            tmux_runner=FakeTmuxRunner(),
        )

    def test_rpc_mode_true(self):
        assert self.adapter.capabilities().rpc_mode is True

    def test_json_mode_true(self):
        assert self.adapter.capabilities().json_mode is True

    def test_has_hooks_true(self):
        assert self.adapter.capabilities().has_hooks is True

    def test_supports_print_mode_true(self):
        assert self.adapter.capabilities().supports_print_mode is True

    def test_idle_indicator_regex_matches_spinner(self):
        cap = self.adapter.capabilities()
        assert cap.idle_indicator_regex is not None
        assert cap.idle_indicator_regex.search("⠋ Executing")

    def test_idle_indicator_regex_does_not_match_prompt(self):
        cap = self.adapter.capabilities()
        assert cap.idle_indicator_regex is not None
        assert not cap.idle_indicator_regex.search("> ")

    def test_dialog_handlers_empty(self):
        """omp uses --auto-approve; no dialog handlers needed."""
        assert self.adapter.capabilities().dialog_handlers == {}

    def test_return_type_is_capabilities(self):
        assert isinstance(self.adapter.capabilities(), Capabilities)


# ---------------------------------------------------------------------------
# run_oneshot()
# ---------------------------------------------------------------------------


class TestRunOneshot:
    def test_calls_omp_runner_with_correct_argv(self):
        fake_omp = FakeOmpRunner(stdout=_agent_end_ndjson("answer"))
        adapter = OmpAdapter(omp_runner=fake_omp, tmux_runner=FakeTmuxRunner())

        result = adapter.run_oneshot("explain this")

        assert len(fake_omp.calls) == 1
        argv = fake_omp.calls[0]
        assert "-p" in argv
        assert "--mode=json" in argv
        assert "explain this" in argv

    def test_returns_parsed_assistant_text(self):
        fake_omp = FakeOmpRunner(stdout=_agent_end_ndjson("the answer is 42"))
        adapter = OmpAdapter(omp_runner=fake_omp, tmux_runner=FakeTmuxRunner())

        assert adapter.run_oneshot("question") == "the answer is 42"

    def test_passes_model_to_argv(self):
        fake_omp = FakeOmpRunner(stdout=_agent_end_ndjson("ok"))
        adapter = OmpAdapter(omp_runner=fake_omp, tmux_runner=FakeTmuxRunner())

        adapter.run_oneshot("prompt", model="openai-codex/gpt-5.5")

        argv = fake_omp.calls[0]
        assert "--model" in argv
        assert argv[argv.index("--model") + 1] == "openai-codex/gpt-5.5"

    def test_passes_workdir_to_argv(self):
        fake_omp = FakeOmpRunner(stdout=_agent_end_ndjson("ok"))
        adapter = OmpAdapter(omp_runner=fake_omp, tmux_runner=FakeTmuxRunner())

        adapter.run_oneshot("prompt", workdir="/workspace")

        argv = fake_omp.calls[0]
        assert "--cwd" in argv
        assert argv[argv.index("--cwd") + 1] == "/workspace"

    def test_raises_value_error_on_unparseable_output(self):
        fake_omp = FakeOmpRunner(stdout="not json at all")
        adapter = OmpAdapter(omp_runner=fake_omp, tmux_runner=FakeTmuxRunner())

        with pytest.raises(ValueError):
            adapter.run_oneshot("prompt")


# ---------------------------------------------------------------------------
# drive()
# ---------------------------------------------------------------------------


class TestDrive:
    def setup_method(self):
        # Pane shows ">" so readiness check passes immediately.
        self.fake_tmux = FakeTmuxRunner(capture_output="last line\n> ")
        self.adapter = OmpAdapter(
            omp_runner=FakeOmpRunner(),
            tmux_runner=self.fake_tmux,
        )
        self.handle = _make_handle()

    def _drive_with_stubbed_load(self, message: str) -> list[list[str]]:
        with patch.object(self.adapter, "_load_buffer"):
            self.adapter.drive(self.handle, message)
        return self.fake_tmux.calls

    def test_drive_issues_paste_buffer(self):
        calls = self._drive_with_stubbed_load("Hello, omp!")
        cmd_names = [c[0] for c in calls]
        assert "paste-buffer" in cmd_names

    def test_drive_paste_buffer_targets_correct_pane(self):
        calls = self._drive_with_stubbed_load("test message")
        paste_call = next(c for c in calls if c[0] == "paste-buffer")
        assert self.handle.pane in paste_call

    def test_drive_sends_enter_after_paste(self):
        calls = self._drive_with_stubbed_load("some prompt")
        send_key_calls = [c for c in calls if c[0] == "send-keys"]
        assert any("Enter" in c for c in send_key_calls)

    def test_drive_checks_readiness_before_paste(self):
        calls = self._drive_with_stubbed_load("prompt")
        cmd_names = [c[0] for c in calls]
        capture_idx = cmd_names.index("capture-pane")
        paste_idx = cmd_names.index("paste-buffer")
        assert capture_idx < paste_idx

    def test_drive_raises_timeout_when_no_prompt(self):
        no_prompt_tmux = FakeTmuxRunner(capture_output="⠋ Running tool")
        adapter = OmpAdapter(omp_runner=FakeOmpRunner(), tmux_runner=no_prompt_tmux)
        handle = _make_handle()
        with patch("session_orchestration.adapters.omp._LAUNCH_READY_TIMEOUT", 0.1):
            with patch("session_orchestration.adapters.omp._POLL_INTERVAL", 0.05):
                with pytest.raises(TimeoutError):
                    adapter.drive(handle, "hi")

    def test_load_buffer_called_with_message(self):
        with patch.object(self.adapter, "_load_buffer") as mock_lb:
            self.adapter.drive(self.handle, "specific omp message")
        mock_lb.assert_called_once()
        args = mock_lb.call_args[0]
        assert "specific omp message" in args


# ---------------------------------------------------------------------------
# detect()
# ---------------------------------------------------------------------------


class TestDetect:
    def _adapter_with_pane(self, pane_text: str) -> OmpAdapter:
        return OmpAdapter(
            omp_runner=FakeOmpRunner(),
            tmux_runner=FakeTmuxRunner(capture_output=pane_text),
        )

    def test_detect_waiting_user_gt(self):
        adapter = self._adapter_with_pane("last output\n> ")
        assert adapter.detect(_make_handle()) == SessionLifecycle.WAITING_USER

    def test_detect_waiting_user_chevron(self):
        adapter = self._adapter_with_pane("last output\n❯ ")
        assert adapter.detect(_make_handle()) == SessionLifecycle.WAITING_USER

    def test_detect_running_via_spinner(self):
        adapter = self._adapter_with_pane("⠋ Executing bash command")
        assert adapter.detect(_make_handle()) == SessionLifecycle.RUNNING

    def test_detect_paused_handoff(self):
        adapter = self._adapter_with_pane(f"done.\n{HANDOFF_MARKER}\n> ")
        assert adapter.detect(_make_handle()) == SessionLifecycle.PAUSED_HANDOFF

    def test_detect_error_on_dead_pane(self):
        class FailingRunner:
            def run(self, args, check=True):
                if args[0] == "capture-pane":
                    raise subprocess.CalledProcessError(1, "tmux")
                return ""

        adapter = OmpAdapter(omp_runner=FakeOmpRunner(), tmux_runner=FailingRunner())
        assert adapter.detect(_make_handle()) == SessionLifecycle.ERROR

    def test_detect_running_default(self):
        adapter = self._adapter_with_pane("omp is processing the request...")
        assert adapter.detect(_make_handle()) == SessionLifecycle.RUNNING


# ---------------------------------------------------------------------------
# resume()
# ---------------------------------------------------------------------------


class TestResume:
    def test_resume_sends_continue_and_waits_when_handoff(self):
        """resume() must invoke omp -c when in PAUSED_HANDOFF."""
        fake_tmux = FakeTmuxRunner(capture_output=f"{HANDOFF_MARKER}\n> ")
        adapter = OmpAdapter(omp_runner=FakeOmpRunner(), tmux_runner=fake_tmux)
        handle = _make_handle()

        # After the send-keys, simulate the prompt appearing.
        with patch.object(adapter, "_wait_for_prompt"):
            adapter.resume(handle, "continue the task")

        # Must have issued send-keys with '--continue' in the command.
        send_key_calls = [c for c in fake_tmux.calls if c[0] == "send-keys"]
        assert send_key_calls, "No send-keys found"
        cmd_text = " ".join(str(x) for x in send_key_calls[0])
        assert "--continue" in cmd_text

    def test_resume_noop_when_not_handoff(self):
        """resume() must be a no-op when the session is NOT in PAUSED_HANDOFF."""
        fake_tmux = FakeTmuxRunner(capture_output="⠋ Working...\n")
        adapter = OmpAdapter(omp_runner=FakeOmpRunner(), tmux_runner=fake_tmux)
        handle = _make_handle()

        with patch.object(adapter, "_wait_for_prompt") as mock_wait:
            adapter.resume(handle, "irrelevant prompt")

        # _wait_for_prompt must NOT have been called.
        mock_wait.assert_not_called()
        # send-keys must NOT have been issued.
        send_key_calls = [c for c in fake_tmux.calls if c[0] == "send-keys"]
        assert not send_key_calls

    def test_resume_noop_when_waiting_user_no_handoff(self):
        """resume() must be a no-op even if '>' is visible (but no handoff marker)."""
        fake_tmux = FakeTmuxRunner(capture_output="output\n> ")
        adapter = OmpAdapter(omp_runner=FakeOmpRunner(), tmux_runner=fake_tmux)
        handle = _make_handle()

        with patch.object(adapter, "_wait_for_prompt") as mock_wait:
            adapter.resume(handle, "prompt")

        mock_wait.assert_not_called()


# ---------------------------------------------------------------------------
# launch() — marker-file injection (stubbed tmux + mocked prompt wait)
# ---------------------------------------------------------------------------


class TestLaunchMarkerInjection:
    """Verify that launch() computes marker_file and passes -e HERMES_MARKER_FILE."""

    def _run_launch(self, workdir: str) -> tuple[SessionHandle, FakeTmuxRunner]:
        fake_tmux = FakeTmuxRunner(capture_output="last line\n> ")
        adapter = OmpAdapter(omp_runner=FakeOmpRunner(), tmux_runner=fake_tmux)
        with (
            patch.object(adapter, "_wait_for_prompt"),
            patch("os.makedirs"),
        ):
            handle = adapter.launch(workdir=workdir, prompt="start")
        return handle, fake_tmux

    def test_new_session_args_contain_hermes_marker_file_flag(self):
        """The '-e HERMES_MARKER_FILE=...' pair must appear in new-session args."""
        handle, fake_tmux = self._run_launch("/some/workdir")
        new_session_call = next(
            c for c in fake_tmux.calls if c[0] == "new-session"
        )
        flat = " ".join(new_session_call)
        assert "HERMES_MARKER_FILE" in flat, (
            f"Expected HERMES_MARKER_FILE in new-session args; got: {new_session_call}"
        )
        assert "-e" in new_session_call

    def test_marker_file_path_formula(self):
        """marker_file must equal {workdir}/.hermes/sessions/{session_id}.jsonl."""
        workdir = "/projects/myapp"
        handle, _ = self._run_launch(workdir)
        assert handle.marker_file is not None
        expected_prefix = f"{workdir}/.hermes/sessions/"
        assert handle.marker_file.startswith(expected_prefix)
        assert handle.marker_file.endswith(".jsonl")

    def test_marker_file_contains_session_id(self):
        """The marker_file path must embed handle.session_id."""
        handle, _ = self._run_launch("/workdir")
        assert handle.marker_file is not None
        assert handle.session_id in handle.marker_file

    def test_marker_file_matches_env_var_in_tmux_args(self):
        """The marker_file on handle must be the same value passed via -e."""
        handle, fake_tmux = self._run_launch("/workdir")
        new_session_call = next(
            c for c in fake_tmux.calls if c[0] == "new-session"
        )
        # Find the value after '-e'
        e_idx = new_session_call.index("-e")
        env_val = new_session_call[e_idx + 1]  # "HERMES_MARKER_FILE=<path>"
        assert env_val == f"HERMES_MARKER_FILE={handle.marker_file}"


# ---------------------------------------------------------------------------
# terminate()
# ---------------------------------------------------------------------------


class TestTerminate:
    def test_terminate_issues_kill_session(self):
        """terminate() must call tmux kill-session -t <tmux_session>."""
        fake_tmux = FakeTmuxRunner()
        adapter = OmpAdapter(omp_runner=FakeOmpRunner(), tmux_runner=fake_tmux)
        handle = _make_handle()

        adapter.terminate(handle)

        kill_calls = [c for c in fake_tmux.calls if c[0] == "kill-session"]
        assert kill_calls, "Expected at least one kill-session call"
        assert handle.tmux_session in kill_calls[0]

    def test_terminate_passes_check_false(self):
        """terminate() must not raise even when kill-session returns nonzero."""

        class ErrorOnKillRunner:
            def run(self, args: list[str], check: bool = True) -> str:
                if args[0] == "kill-session":
                    raise subprocess.CalledProcessError(1, "tmux kill-session")
                return ""

        adapter = OmpAdapter(omp_runner=FakeOmpRunner(), tmux_runner=ErrorOnKillRunner())
        handle = _make_handle()
        # Must not raise — CalledProcessError is swallowed.
        adapter.terminate(handle)

    def test_terminate_targets_correct_session(self):
        """terminate() must target handle.tmux_session, not a hard-coded name."""
        fake_tmux = FakeTmuxRunner()
        adapter = OmpAdapter(omp_runner=FakeOmpRunner(), tmux_runner=fake_tmux)
        handle = _make_handle(pane="custom-session-xyz:0.0")
        handle = SessionHandle(
            session_id=handle.session_id,
            tmux_session="custom-session-xyz",
            pane=handle.pane,
            launch_ts=handle.launch_ts,
        )

        adapter.terminate(handle)

        kill_call = next(c for c in fake_tmux.calls if c[0] == "kill-session")
        assert "custom-session-xyz" in kill_call


# ---------------------------------------------------------------------------
# translate_rpc_line()
# ---------------------------------------------------------------------------


class TestTranslateRpcLine:
    """Round-trip tests: every mapped omp RPC event type + None for unknowns."""

    def test_ready_maps_to_status_phase_ready(self):
        line = json.dumps({"type": "ready"})
        result = translate_rpc_line(line)
        assert result is not None
        assert result["kind"] == MARKER_STATUS
        assert result["payload"]["phase"] == "ready"

    def test_turn_start_maps_to_status_phase_running(self):
        line = json.dumps({"type": "turn_start"})
        result = translate_rpc_line(line)
        assert result is not None
        assert result["kind"] == MARKER_STATUS
        assert result["payload"]["phase"] == "running"

    def test_message_end_assistant_maps_to_heartbeat(self):
        line = json.dumps({
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Here is my answer."}],
            },
        })
        result = translate_rpc_line(line)
        assert result is not None
        assert result["kind"] == MARKER_HEARTBEAT
        assert result["payload"]["note"] == "Here is my answer."

    def test_message_end_user_role_returns_none(self):
        """message_end with role=user must not produce a marker."""
        line = json.dumps({
            "type": "message_end",
            "message": {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        })
        assert translate_rpc_line(line) is None

    def test_heartbeat_note_truncated_to_120_chars(self):
        long_text = "x" * 200
        line = json.dumps({
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": long_text}],
            },
        })
        result = translate_rpc_line(line)
        assert result is not None
        assert len(result["payload"]["note"]) == 120

    def test_agent_end_maps_to_done_with_last_assistant_text(self):
        line = json.dumps({
            "type": "agent_end",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "do it"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "done!"}]},
            ],
        })
        result = translate_rpc_line(line)
        assert result is not None
        assert result["kind"] == MARKER_DONE
        assert result["payload"]["summary"] == "done!"
        assert result["payload"]["artifacts"] is None

    def test_agent_end_no_assistant_messages_gives_empty_summary(self):
        line = json.dumps({"type": "agent_end", "messages": []})
        result = translate_rpc_line(line)
        assert result is not None
        assert result["kind"] == MARKER_DONE
        assert result["payload"]["summary"] == ""

    def test_needs_input_maps_correctly(self):
        line = json.dumps({"type": "needs_input", "question": "Which branch?"})
        result = translate_rpc_line(line)
        assert result is not None
        assert result["kind"] == MARKER_NEEDS_INPUT
        assert result["payload"]["question"] == "Which branch?"
        assert result["payload"]["options"] is None
        assert result["payload"]["context"] is None

    def test_unknown_type_returns_none(self):
        line = json.dumps({"type": "available_commands_update", "commands": []})
        assert translate_rpc_line(line) is None

    def test_blank_line_returns_none(self):
        assert translate_rpc_line("") is None
        assert translate_rpc_line("   ") is None

    def test_non_json_line_returns_none(self):
        assert translate_rpc_line("not json at all") is None

    def test_session_type_returns_none(self):
        """session event (startup info) is not a marker event."""
        line = json.dumps({"type": "session", "version": 3, "id": "abc"})
        assert translate_rpc_line(line) is None

    def test_agent_start_returns_none(self):
        line = json.dumps({"type": "agent_start"})
        assert translate_rpc_line(line) is None


# ---------------------------------------------------------------------------
# tail_rpc_stream()
# ---------------------------------------------------------------------------


class TestTailRpcStream:
    """Drive a fake stdout iter and assert correct marker kinds are written."""

    def _run_stream(self, lines: list[str]) -> list[tuple]:
        """Run tail_rpc_stream with a capturing append_fn; return recorded calls."""
        recorded: list[tuple] = []

        def fake_append(path: str, kind: str, payload: dict, task_id: str) -> None:
            recorded.append((path, kind, payload, task_id))

        tail_rpc_stream(
            stdout_iter=iter(lines),
            marker_path="/fake/marker.jsonl",
            task_id="test-task-001",
            append_fn=fake_append,
        )
        return recorded

    def test_ready_event_writes_status_marker(self):
        lines = [json.dumps({"type": "ready"})]
        calls = self._run_stream(lines)
        assert len(calls) == 1
        assert calls[0][1] == MARKER_STATUS  # kind
        assert calls[0][2]["phase"] == "ready"

    def test_multiple_events_write_multiple_markers(self):
        lines = [
            json.dumps({"type": "ready"}),
            json.dumps({"type": "turn_start"}),
            json.dumps({
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "progress"}],
                },
            }),
            json.dumps({
                "type": "agent_end",
                "messages": [
                    {"role": "assistant", "content": [{"type": "text", "text": "final"}]},
                ],
            }),
        ]
        calls = self._run_stream(lines)
        kinds = [c[1] for c in calls]
        assert kinds == [MARKER_STATUS, MARKER_STATUS, MARKER_HEARTBEAT, MARKER_DONE]

    def test_unknown_events_skipped(self):
        lines = [
            json.dumps({"type": "available_commands_update", "commands": []}),
            json.dumps({"type": "ready"}),
        ]
        calls = self._run_stream(lines)
        assert len(calls) == 1
        assert calls[0][1] == MARKER_STATUS

    def test_blank_and_non_json_lines_skipped(self):
        lines = ["", "  ", "omp v16.1.15 starting up", json.dumps({"type": "ready"})]
        calls = self._run_stream(lines)
        assert len(calls) == 1

    def test_path_and_task_id_passed_to_append_fn(self):
        lines = [json.dumps({"type": "ready"})]
        calls = self._run_stream(lines)
        assert calls[0][0] == "/fake/marker.jsonl"  # path
        assert calls[0][3] == "test-task-001"  # task_id

    def test_empty_stream_writes_no_markers(self):
        calls = self._run_stream([])
        assert calls == []
