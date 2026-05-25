"""Tests for stream_json module — structured JSONL output for programmatic use."""

import json
import sys
import io
import pytest

from stream_json import StreamJsonEmitter


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
