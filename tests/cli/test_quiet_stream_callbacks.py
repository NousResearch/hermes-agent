"""Tests for the `chat -Q --stream` callback factory.

The `--stream` flag opts into live feedback in non-interactive quiet mode
(e.g. when the agent is running inside a remote sandbox and the user wants
to see progress instead of staring at a frozen terminal). The factory in
``cli._make_quiet_stream_callbacks`` produces the two callbacks that wire
into ``AIAgent.stream_delta_callback`` and ``AIAgent.tool_progress_callback``.

We test the factory directly — the surrounding ``main()`` call site is too
entangled to unit-test in isolation, but the contract that matters
(stdout = answer text, stderr = lifecycle markers, formatting stable) lives
in the factory.
"""

import pytest

from cli import _make_quiet_stream_callbacks


class TestStreamDeltaCallback:
    def test_writes_text_verbatim_to_stdout(self, capsys):
        delta_cb, _ = _make_quiet_stream_callbacks()

        delta_cb("Hello, ")
        delta_cb("world")
        delta_cb("!")

        # Concatenation is the load-bearing contract: wrappers that capture
        # stdout must reconstruct the full response by simple concatenation.
        out, err = capsys.readouterr()
        assert out == "Hello, world!"
        assert err == ""

    def test_empty_delta_is_safe(self, capsys):
        delta_cb, _ = _make_quiet_stream_callbacks()

        delta_cb("")  # streaming providers occasionally emit empty deltas

        out, _ = capsys.readouterr()
        assert out == ""


class TestToolProgressCallback:
    def test_started_event_emits_to_stderr(self, capsys):
        _, progress_cb = _make_quiet_stream_callbacks()

        progress_cb("tool.started", "terminal", "ls -la", {"cmd": "ls -la"})

        out, err = capsys.readouterr()
        assert err == "[tool] terminal started\n"
        assert out == ""  # never on stdout

    def test_completed_event_includes_duration(self, capsys):
        _, progress_cb = _make_quiet_stream_callbacks()

        progress_cb("tool.completed", "terminal", duration=1.234, is_error=False)

        _, err = capsys.readouterr()
        assert err == "[tool] terminal completed (1.2s)\n"

    def test_completed_event_marks_failure(self, capsys):
        _, progress_cb = _make_quiet_stream_callbacks()

        progress_cb("tool.completed", "web_search", duration=0.5, is_error=True)

        _, err = capsys.readouterr()
        assert err == "[tool] web_search failed (0.5s)\n"

    def test_completed_without_duration_omits_suffix(self, capsys):
        _, progress_cb = _make_quiet_stream_callbacks()

        progress_cb("tool.completed", "memory")  # no kwargs

        _, err = capsys.readouterr()
        assert err == "[tool] memory completed\n"

    @pytest.mark.parametrize(
        "event_type",
        ["_thinking", "reasoning.available", "subagent.started", "tool.progress"],
    )
    def test_unknown_or_internal_events_are_ignored(self, capsys, event_type):
        # Quiet-mode stderr stays compact; only lifecycle start/complete
        # events should surface. Anything else (subagent chatter, internal
        # thinking signals) must be dropped.
        _, progress_cb = _make_quiet_stream_callbacks()

        progress_cb(event_type, "anything", "preview", {"args": True})

        out, err = capsys.readouterr()
        assert out == ""
        assert err == ""

    def test_started_event_without_name_is_ignored(self, capsys):
        # Defensive: if the agent ever fires tool.started with name=None,
        # we shouldn't emit a malformed "[tool]  started" line.
        _, progress_cb = _make_quiet_stream_callbacks()

        progress_cb("tool.started", None)

        _, err = capsys.readouterr()
        assert err == ""


class TestStdoutStderrSeparation:
    """The headline contract: piped stdout must contain ONLY the answer."""

    def test_mixed_events_keep_streams_separate(self, capsys):
        delta_cb, progress_cb = _make_quiet_stream_callbacks()

        progress_cb("tool.started", "terminal")
        delta_cb("The answer is ")
        progress_cb("tool.completed", "terminal", duration=0.8)
        delta_cb("42.")

        out, err = capsys.readouterr()
        assert out == "The answer is 42."
        assert err == (
            "[tool] terminal started\n"
            "[tool] terminal completed (0.8s)\n"
        )
