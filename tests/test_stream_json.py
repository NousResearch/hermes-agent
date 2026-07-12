"""Tests for stream_json module — structured JSONL output for programmatic use."""

import io
import json
import signal
import sys

import pytest

from stream_json import StreamJsonEmitter


def _run_stream_json_chat(monkeypatch, capsys, run_conversation):
    """Exercise parser -> cmd_chat -> cli.main with a deterministic agent."""
    import cli
    import hermes_cli.main as cli_entry
    from hermes_cli._parser import build_top_level_parser

    class FakeAgent:
        model = "test-model"
        session_id = "session-123"

        def __init__(self):
            self.stream_delta_callback = None
            self.tool_gen_callback = None
            self.tool_progress_callback = None

        def run_conversation(self, **_kwargs):
            return run_conversation(self)

    class FakeCLI:
        def __init__(self, **_kwargs):
            self.session_id = "session-123"
            self.conversation_history = []
            self.provider = ""
            self.model = "test-model"
            self.agent = None
            self._active_agent_route_signature = None
            self.tool_progress_mode = None
            self.console = type("Console", (), {"print": staticmethod(print)})()

        def _claim_active_session(self, *_args, **_kwargs):
            return True

        def _ensure_runtime_credentials(self):
            return True

        def _resolve_turn_agent_config(self, _query):
            return {
                "signature": "test-route",
                "model": None,
                "runtime": None,
                "request_overrides": None,
            }

        def _init_agent(self, **_kwargs):
            self.agent = FakeAgent()
            return True

        def _show_security_advisories(self):
            pass

        def chat(self, _query, images=None):
            print("human output")

        def _print_exit_summary(self):
            pass

    monkeypatch.setattr(cli, "HermesCLI", FakeCLI)
    monkeypatch.setattr(cli, "_finalize_single_query", lambda _cli: None)
    monkeypatch.setattr(cli, "_emit_interrupted_session_end", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli.atexit, "register", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(signal, "signal", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli_entry, "_resolve_use_tui", lambda _args: False)
    monkeypatch.setattr(cli_entry, "_has_any_provider_configured", lambda: True)
    monkeypatch.setattr(cli_entry, "_termux_should_prefetch_update_check", lambda: False)
    monkeypatch.setattr(cli_entry, "_sync_bundled_skills_for_startup", lambda: None)
    monkeypatch.setattr(cli_entry, "_pin_kanban_board_env", lambda: None)

    parser, _, _ = build_top_level_parser()
    args = parser.parse_args(["chat", "-q", "hello", "--format", "stream-json"])
    with pytest.raises(SystemExit) as exc_info:
        cli_entry.cmd_chat(args)

    captured = capsys.readouterr()
    return exc_info.value.code, [json.loads(line) for line in captured.out.splitlines() if line]


class TestStreamJsonEmitter:
    """Unit tests for the StreamJsonEmitter class."""

    def _make_emitter(self, **kwargs):
        """Create an emitter that captures stdout output."""
        self._buf = io.StringIO()
        self._old_stdout = sys.stdout
        sys.stdout = self._buf
        return StreamJsonEmitter(**kwargs)

    def _teardown(self):
        sys.stdout = self._old_stdout

    def _lines(self):
        """Return all emitted lines as parsed JSON objects."""
        self._buf.flush()
        text = self._buf.getvalue()
        return [json.loads(line) for line in text.strip().split("\n") if line.strip()]

    def _raw_lines(self):
        """Return raw emitted lines (strings)."""
        self._buf.flush()
        text = self._buf.getvalue()
        return [line for line in text.strip().split("\n") if line.strip()]

    def test_init_emits_system_event(self):
        em = self._make_emitter(model="test-model", session_id="sess-123")
        lines = self._lines()
        self._teardown()
        assert len(lines) >= 1
        init = lines[0]
        assert init["type"] == "system"
        assert init["subtype"] == "init"
        assert init["model"] == "test-model"
        assert init["session_id"] == "sess-123"

    def test_text_delta_emits_text_event(self):
        em = self._make_emitter()
        # Skip the init event
        self._buf.truncate(0)
        self._buf.seek(0)
        em.on_text_delta("Hello, world!")
        lines = self._lines()
        self._teardown()
        assert len(lines) == 1
        assert lines[0]["type"] == "text"
        assert lines[0]["text"] == "Hello, world!"
        assert "timestamp" in lines[0]

    def test_text_delta_skips_empty(self):
        em = self._make_emitter()
        self._buf.truncate(0)
        self._buf.seek(0)
        em.on_text_delta("")
        em.on_text_delta("   ")
        lines = self._lines()
        self._teardown()
        assert len(lines) == 0

    def test_tool_gen_start_emits_tool_use_event(self):
        em = self._make_emitter()
        self._buf.truncate(0)
        self._buf.seek(0)
        em.on_tool_gen_start("read_file")
        lines = self._lines()
        self._teardown()
        assert len(lines) == 1
        assert lines[0]["type"] == "tool_use"
        assert lines[0]["name"] == "read_file"
        assert "timestamp" in lines[0]

    def test_tool_progress_complete_emits_result(self):
        em = self._make_emitter()
        self._buf.truncate(0)
        self._buf.seek(0)
        em.on_tool_progress(
            "complete",
            "read_file",
            {"path": "main.go"},
            "file contents here",
            duration=0.5,
            is_error=False,
        )
        lines = self._lines()
        self._teardown()
        assert len(lines) == 1
        assert lines[0]["type"] == "tool_result"
        assert lines[0]["name"] == "read_file"
        assert lines[0]["output"] == "file contents here"
        assert lines[0]["duration_ms"] == 500
        assert lines[0]["is_error"] is False

    def test_current_tool_progress_shape_emits_use_and_result(self):
        """Current Hermes progress callbacks use tool.started/tool.completed events."""
        em = self._make_emitter()
        self._buf.truncate(0)
        self._buf.seek(0)

        em.on_tool_progress(
            "tool.started",
            "read_file",
            "read: main.py",
            {"path": "main.py"},
        )
        em.on_tool_progress(
            "tool.completed",
            "read_file",
            None,
            None,
            duration=0.25,
            is_error=True,
            result="tool failed",
        )

        lines = self._lines()
        self._teardown()
        assert len(lines) == 2
        assert lines[0]["type"] == "tool_use"
        assert lines[0]["name"] == "read_file"
        assert lines[0]["input"] == {"path": "main.py"}
        assert lines[1]["type"] == "tool_result"
        assert lines[1]["output"] == "tool failed"
        assert lines[1]["duration_ms"] == 250
        assert lines[1]["is_error"] is True

    def test_tool_gen_then_started_does_not_duplicate_tool_use(self):
        """Streaming tool generation plus execution start should be one tool_use line."""
        em = self._make_emitter()
        self._buf.truncate(0)
        self._buf.seek(0)

        em.on_tool_gen_start("terminal")
        em.on_tool_progress(
            "tool.started",
            "terminal",
            "run: echo hi",
            {"command": "echo hi"},
        )

        lines = self._lines()
        self._teardown()
        assert len(lines) == 1
        assert lines[0]["type"] == "tool_use"
        assert lines[0]["name"] == "terminal"

    def test_tool_progress_start_does_not_emit(self):
        em = self._make_emitter()
        self._buf.truncate(0)
        self._buf.seek(0)
        em.on_tool_progress(
            "start",
            "read_file",
            {"path": "main.go"},
            None,
        )
        lines = self._lines()
        self._teardown()
        assert len(lines) == 0

    def test_emit_result_success(self):
        em = self._make_emitter()
        self._buf.truncate(0)
        self._buf.seek(0)

        # Capture stderr
        old_stderr = sys.stderr
        stderr_buf = io.StringIO()
        sys.stderr = stderr_buf

        result = {
            "final_response": "Task completed successfully.",
            "input_tokens": 5000,
            "output_tokens": 200,
            "total_tokens": 5200,
            "cache_read_tokens": 3000,
            "cache_write_tokens": 1000,
            "failed": False,
        }
        exit_code = em.emit_result(result, session_id="sess-abc")

        sys.stderr = old_stderr
        self._teardown()

        lines = self._lines()
        assert len(lines) == 1
        r = lines[0]
        assert r["type"] == "result"
        assert r["session_id"] == "sess-abc"
        assert r["exit_code"] == 0
        assert r["tokens"]["input"] == 5000
        assert r["tokens"]["output"] == 200
        assert r["tokens"]["total"] == 5200
        assert r["tokens"]["cache_read"] == 3000
        assert r["tokens"]["cache_write"] == 1000
        assert exit_code == 0

    def test_emit_result_failed(self):
        em = self._make_emitter()
        self._buf.truncate(0)
        self._buf.seek(0)

        old_stderr = sys.stderr
        stderr_buf = io.StringIO()
        sys.stderr = stderr_buf

        result = {"final_response": "", "failed": True}
        exit_code = em.emit_result(result, session_id="sess-fail")

        sys.stderr = old_stderr
        self._teardown()

        lines = self._lines()
        assert len(lines) == 1
        assert lines[0]["exit_code"] == 1
        assert exit_code == 1

    def test_emit_result_with_explicit_exit_code(self):
        em = self._make_emitter()
        self._buf.truncate(0)
        self._buf.seek(0)

        old_stderr = sys.stderr
        stderr_buf = io.StringIO()
        sys.stderr = stderr_buf

        result = {"final_response": "ok", "failed": False}
        exit_code = em.emit_result(result, session_id="sess-x", exit_code=2)

        sys.stderr = old_stderr
        self._teardown()

        lines = self._lines()
        assert lines[0]["exit_code"] == 2
        assert exit_code == 2

    def test_full_conversation_flow(self):
        """Simulate a complete conversation with tool use."""
        em = self._make_emitter(model="test-model", session_id="sess-flow")

        # Clear init event
        self._buf.truncate(0)
        self._buf.seek(0)

        em.on_text_delta("I'll read the file.")
        em.on_tool_gen_start("read_file")
        em.on_tool_progress(
            "complete",
            "read_file",
            {"path": "main.py"},
            "print('hello')",
            duration=0.1,
        )
        em.on_text_delta("The file prints hello.")

        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        exit_code = em.emit_result(
            {
                "final_response": "The file prints hello.",
                "input_tokens": 1000,
                "output_tokens": 50,
                "total_tokens": 1050,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "failed": False,
            },
            session_id="sess-flow",
        )
        sys.stderr = old_stderr
        self._teardown()

        lines = self._lines()
        # Expected: text, tool_use, tool_result, text, result
        assert len(lines) == 5
        assert lines[0]["type"] == "text"
        assert lines[1]["type"] == "tool_use"
        assert lines[1]["name"] == "read_file"
        assert lines[2]["type"] == "tool_result"
        assert lines[2]["name"] == "read_file"
        assert lines[3]["type"] == "text"
        assert lines[4]["type"] == "result"
        assert lines[4]["tokens"]["total"] == 1050
        assert exit_code == 0

    def test_output_is_valid_jsonl(self):
        """Every line must be valid JSON."""
        em = self._make_emitter(model="test")
        self._buf.truncate(0)
        self._buf.seek(0)

        em.on_text_delta("test text")
        em.on_tool_gen_start("bash")
        em.on_tool_progress("complete", "bash", {}, "output", duration=1.0)

        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        em.emit_result({"final_response": "done", "failed": False}, session_id="s1")
        sys.stderr = old_stderr

        raw = self._raw_lines()
        self._teardown()

        for line in raw:
            parsed = json.loads(line)
            assert "type" in parsed

    def test_tool_result_output_truncation(self):
        """Long tool outputs should be truncated to avoid bloating the stream."""
        em = self._make_emitter()
        self._buf.truncate(0)
        self._buf.seek(0)

        long_output = "x" * 10000
        em.on_tool_progress(
            "complete",
            "bash",
            {},
            long_output,
            duration=0.0,
        )
        lines = self._lines()
        self._teardown()
        assert len(lines[0]["output"]) <= 5010  # 5000 + "..." (3 bytes) + some margin

    def test_broken_pipe_does_not_crash(self):
        """Emitter should handle BrokenPipeError gracefully."""
        em = self._make_emitter()

        # Force a BrokenPipeError by replacing stdout
        class BrokenStdout:
            def write(self, s):
                raise BrokenPipeError()
            def flush(self):
                raise BrokenPipeError()

        old = sys.stdout
        sys.stdout = BrokenStdout()
        try:
            # Should not raise
            em.on_text_delta("test")
        finally:
            sys.stdout = old
            self._teardown()


def test_chat_stream_json_implies_quiet_and_emits_jsonl(monkeypatch, capsys):
    """The public chat command must not require callers to add ``-Q``."""
    def run_conversation(agent):
        agent.stream_delta_callback("hello")
        agent.tool_gen_callback("read_file")
        agent.tool_progress_callback(
            "tool.completed",
            "read_file",
            duration=0.01,
            result="contents",
        )
        return {"final_response": "hello", "failed": False}

    exit_code, events = _run_stream_json_chat(monkeypatch, capsys, run_conversation)

    assert exit_code == 0
    assert [event["type"] for event in events] == [
        "system",
        "text",
        "tool_use",
        "tool_result",
        "result",
    ]
    assert events[-1]["exit_code"] == 0


def test_chat_stream_json_interrupt_emits_terminal_result(monkeypatch, capsys):
    """Interrupts must still close a JSONL stream with the process exit code."""
    def run_conversation(_agent):
        raise KeyboardInterrupt

    exit_code, events = _run_stream_json_chat(monkeypatch, capsys, run_conversation)

    assert exit_code == 130
    assert events[-1]["type"] == "result"
    assert events[-1]["exit_code"] == 130


def test_chat_stream_json_requires_query(monkeypatch, capsys):
    """A stream format cannot fall through into an interactive session."""
    import hermes_cli.main as cli_entry
    from hermes_cli._parser import build_top_level_parser

    parser, _, _ = build_top_level_parser()
    args = parser.parse_args(["chat", "--format", "stream-json"])

    with pytest.raises(SystemExit) as exc_info:
        cli_entry.cmd_chat(args)

    assert exc_info.value.code == 2
    assert "--format stream-json requires -q/--query" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["chat", "-q", "hello", "--format", "stream-json", "--tui"],
        ["--tui", "chat", "-q", "hello", "--format", "stream-json"],
    ],
)
def test_chat_stream_json_rejects_explicit_tui(monkeypatch, capsys, argv):
    """The JSONL stream is incompatible with the interactive TUI transport."""
    import hermes_cli.main as cli_entry
    from hermes_cli._parser import build_top_level_parser

    parser, _, _ = build_top_level_parser()
    args = parser.parse_args(argv)
    monkeypatch.setattr(cli_entry, "_launch_tui", lambda *_args, **_kwargs: pytest.fail("TUI launched"))

    with pytest.raises(SystemExit) as exc_info:
        cli_entry.cmd_chat(args)

    assert exc_info.value.code == 2
    assert "--format stream-json cannot be used with --tui" in capsys.readouterr().err
