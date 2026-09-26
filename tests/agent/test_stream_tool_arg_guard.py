"""Live guards for a streaming tool call that loops or stalls.

A model can degenerate into repeating the same text inside a tool call's
arguments until the output cap cuts it off mid-JSON. The truncated call is
refused and retried, but only after the whole runaway has streamed, which at
~150 tokens/s is several minutes. Chunks and argument text keep flowing the
whole time, so the chunk-based stale detector sees a healthy stream. The
repetition guard reconnects as soon as the loop is recognisable, and the
argument-stall watch covers the variant where chunks keep arriving but the
open tool call stops growing.
"""
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agent import chat_completion_helpers as helpers
from tests.agent.test_streaming import _make_stream_chunk, _make_tool_call_delta

LOOP = "The prompt list continues with the same line again and again. "


def _make_agent(**kwargs):
    from run_agent import AIAgent

    defaults = dict(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    defaults.update(kwargs)
    agent = AIAgent(**defaults)
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    return agent


class _Stream:
    response = SimpleNamespace(headers={})

    def __init__(self, chunks):
        self._chunks = chunks

    def __iter__(self):
        return iter(self._chunks)


def _write_file_call(arguments, tc_id="call_1"):
    return _make_tool_call_delta(index=0, tc_id=tc_id, name="write_file", arguments=arguments)


def _complete_retry(complete):
    return _Stream([
        _make_stream_chunk(tool_calls=[_write_file_call(complete, tc_id="call_2")]),
        _make_stream_chunk(finish_reason="tool_calls", model="test/model"),
    ])


class TestAccumulatorWatch:
    def _feed(self, acc, arguments, tc_id=None):
        acc.feed(_write_file_call(arguments, tc_id=tc_id))

    def test_flags_a_repetition_loop_past_the_size_threshold(self):
        acc = helpers._ToolCallAccumulator(watch_arguments=True)
        self._feed(acc, '{"path":"/tmp/a.md","content":"', tc_id="call_1")
        while acc.runaway is None and acc._argument_chars[0] < 100_000:
            self._feed(acc, LOOP * 10)
        assert acc.runaway is not None
        assert acc.runaway[0] == "write_file"
        assert acc.runaway[1] >= helpers._TOOL_ARG_RUNAWAY_MIN_CHARS
        # Buffered parts are untouched by the watch.
        assert acc.materialize()[0]["function"]["arguments"].startswith('{"path":"/tmp/a.md"')

    def test_ignores_large_varied_arguments(self):
        acc = helpers._ToolCallAccumulator(watch_arguments=True)
        self._feed(acc, '{"path":"/tmp/a.csv","content":"', tc_id="call_1")
        for i in range(1500):
            self._feed(acc, f"row {i}: value {i * 7919 % 104729}, label item-{i:05d}\\n")
        assert acc._argument_chars[0] > helpers._TOOL_ARG_RUNAWAY_MIN_CHARS
        assert acc.runaway is None

    def test_does_not_watch_unless_asked_or_after_the_finish(self):
        unwatched = helpers._ToolCallAccumulator()
        self._feed(unwatched, '{"a":"', tc_id="call_1")
        assert unwatched.last_growth_at is None

        watched = helpers._ToolCallAccumulator(watch_arguments=True)
        self._feed(watched, '{"a":"', tc_id="call_1")
        assert watched.last_growth_at is not None
        watched.close_watch()
        self._feed(watched, "more")
        assert watched.last_growth_at is None and watched.runaway is None


class TestStreamingGuards:
    @pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
    @patch("run_agent.AIAgent._replace_primary_openai_client")
    @patch("run_agent.AIAgent._abort_request_openai_client")
    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_reconnects_when_streamed_arguments_degenerate_into_repetition(
        self, mock_close, mock_create, mock_abort, mock_replace, monkeypatch, caplog
    ):
        monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "30")
        monkeypatch.setenv("HERMES_STREAM_RETRIES", "1")

        class RunawayArguments:
            response = SimpleNamespace(headers={})

            def __iter__(self):
                yield _make_stream_chunk(tool_calls=[_write_file_call('{"path":"/tmp/a.md","content":"')])
                # Growing arguments keep both the stale and the stall detector quiet.
                for _ in range(400):
                    time.sleep(0.005)
                    yield _make_stream_chunk(tool_calls=[_write_file_call(LOOP * 20, tc_id=None)])
                raise httpx.RemoteProtocolError("peer closed connection")

        complete = '{"path":"/tmp/a.md","content":"# Prompts"}'
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [RunawayArguments(), _complete_retry(complete)]
        mock_create.return_value = mock_client
        agent = _make_agent()

        with caplog.at_level(logging.WARNING):
            response = agent._interruptible_streaming_api_call({})

        assert mock_abort.called
        assert "arguments degenerated into repetition" in caplog.text
        assert response.choices[0].message.tool_calls[0].function.arguments == complete
        # A looping model is responsive: the unresponsive-provider breaker is untouched.
        assert helpers._stale_streak(agent) == 0

    @patch("run_agent.AIAgent._abort_request_openai_client")
    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_keeps_streaming_large_varied_arguments(
        self, mock_close, mock_create, mock_abort, monkeypatch, caplog
    ):
        monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "30")
        rows = [f"row {i}: value {i * 7919 % 104729}, label item-{i:05d}\\n" for i in range(1500)]

        class LargeVariedArguments:
            response = SimpleNamespace(headers={})

            def __iter__(self):
                yield _make_stream_chunk(tool_calls=[_write_file_call('{"path":"/tmp/a.csv","content":"')])
                for row in rows:
                    yield _make_stream_chunk(tool_calls=[_write_file_call(row, tc_id=None)])
                yield _make_stream_chunk(tool_calls=[_write_file_call('"}', tc_id=None)])
                yield _make_stream_chunk(finish_reason="tool_calls", model="test/model")

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [LargeVariedArguments()]
        mock_create.return_value = mock_client
        agent = _make_agent()

        with caplog.at_level(logging.WARNING):
            response = agent._interruptible_streaming_api_call({})

        assert not mock_abort.called
        assert "degenerated into repetition" not in caplog.text
        assert len(response.choices[0].message.tool_calls[0].function.arguments) > 32_000

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
    @patch("run_agent.AIAgent._replace_primary_openai_client")
    @patch("run_agent.AIAgent._abort_request_openai_client")
    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_reconnects_when_tool_arguments_stop_growing_while_chunks_keep_arriving(
        self, mock_close, mock_create, mock_abort, mock_replace, monkeypatch, caplog
    ):
        monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "30")
        monkeypatch.setenv("HERMES_TOOL_ARG_STALL_TIMEOUT", "0.2")
        monkeypatch.setenv("HERMES_STREAM_RETRIES", "1")

        class StalledArguments:
            response = SimpleNamespace(headers={})

            def __iter__(self):
                yield _make_stream_chunk(tool_calls=[_write_file_call('{"path":"/tmp/a.md","content":"# He')])
                # Chunks keep the stream alive, but the arguments never grow.
                for _ in range(40):
                    time.sleep(0.05)
                    yield _make_stream_chunk(content="")
                raise httpx.RemoteProtocolError("peer closed connection")

        complete = '{"path":"/tmp/a.md","content":"# Heading"}'
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [StalledArguments(), _complete_retry(complete)]
        mock_create.return_value = mock_client
        agent = _make_agent()

        with caplog.at_level(logging.WARNING):
            response = agent._interruptible_streaming_api_call({})

        assert mock_abort.called
        assert "Tool call arguments stalled" in caplog.text
        assert response.choices[0].message.tool_calls[0].function.arguments == complete

    @patch("run_agent.AIAgent._abort_request_openai_client")
    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_does_not_reconnect_while_tool_arguments_keep_growing(
        self, mock_close, mock_create, mock_abort, monkeypatch, caplog
    ):
        monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "30")
        monkeypatch.setenv("HERMES_TOOL_ARG_STALL_TIMEOUT", "0.2")

        class GrowingArguments:
            response = SimpleNamespace(headers={})

            def __iter__(self):
                yield _make_stream_chunk(tool_calls=[_write_file_call('{"path":"/tmp/a.md","content":"')])
                for _ in range(12):
                    time.sleep(0.05)
                    yield _make_stream_chunk(tool_calls=[_write_file_call("x", tc_id=None)])
                yield _make_stream_chunk(tool_calls=[_write_file_call('"}', tc_id=None)])
                yield _make_stream_chunk(finish_reason="tool_calls", model="test/model")

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [GrowingArguments()]
        mock_create.return_value = mock_client
        agent = _make_agent()

        with caplog.at_level(logging.WARNING):
            response = agent._interruptible_streaming_api_call({})

        assert not mock_abort.called
        assert "Tool call arguments stalled" not in caplog.text
        assert response.choices[0].message.tool_calls[0].function.arguments.endswith('"}')

    @patch("run_agent.AIAgent._abort_request_openai_client")
    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_does_not_watch_arguments_after_the_finish_reason_arrives(
        self, mock_close, mock_create, mock_abort, monkeypatch, caplog
    ):
        monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "30")
        monkeypatch.setenv("HERMES_TOOL_ARG_STALL_TIMEOUT", "0.2")

        class TrailingChunks:
            response = SimpleNamespace(headers={})

            def __iter__(self):
                yield _make_stream_chunk(tool_calls=[_write_file_call('{"path":"/tmp/a.md","content":"ok"}')])
                yield _make_stream_chunk(finish_reason="tool_calls", model="test/model")
                time.sleep(0.5)
                yield _make_stream_chunk(content="")

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [TrailingChunks()]
        mock_create.return_value = mock_client
        agent = _make_agent()

        with caplog.at_level(logging.WARNING):
            agent._interruptible_streaming_api_call({})

        assert not mock_abort.called
        assert "Tool call arguments stalled" not in caplog.text

    def test_argument_stall_watch_is_off_for_local_endpoints_unless_configured(self, monkeypatch):
        monkeypatch.delenv("HERMES_TOOL_ARG_STALL_TIMEOUT", raising=False)
        monkeypatch.delenv("HERMES_STREAM_STALE_TIMEOUT", raising=False)
        local = helpers._StreamingCall(_make_agent(base_url="http://127.0.0.1:11434/v1"), {"model": "m"}, None)
        local._resolve_stale_timeout()
        assert local._tool_arg_stall_timeout == float("inf")

        cloud = helpers._StreamingCall(_make_agent(), {"model": "m"}, None)
        cloud._resolve_stale_timeout()
        assert cloud._tool_arg_stall_timeout == 120.0

        monkeypatch.setenv("HERMES_TOOL_ARG_STALL_TIMEOUT", "45")
        local._resolve_stale_timeout()
        assert local._tool_arg_stall_timeout == 45.0


class _RunawayProvider(BaseHTTPRequestHandler):
    """Streams a looping ``write_file`` call for ``LOOP_SECONDS`` on the first
    streamed request, then a complete call."""

    protocol_version = "HTTP/1.1"
    LOOP_SECONDS = 20
    streamed = []

    def log_message(self, format, *args):
        pass

    def _sse(self, delta, finish=None):
        chunk = {"id": "chatcmpl-e2e", "object": "chat.completion.chunk", "created": 0, "model": "m",
                 "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())

    @staticmethod
    def _call(arguments, first=False):
        call = {"index": 0, "function": {"arguments": arguments}}
        if first:
            call.update(id="call_e2e", type="function")
            call["function"]["name"] = "write_file"
        return {"tool_calls": [call]}

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
        if not body.get("stream"):
            self.send_response(404)
            self.send_header("content-length", "0")
            self.end_headers()
            return
        self.streamed.append(time.time())
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        try:
            if len(self.streamed) == 1:
                self._sse(self._call('{"path":"/tmp/e2e.md","content":"', first=True))
                deadline = time.time() + self.LOOP_SECONDS
                while time.time() < deadline:
                    self._sse(self._call(LOOP))
                    self.wfile.flush()
                    time.sleep(0.005)
            else:
                self._sse(self._call('{"path":"/tmp/e2e.md","content":"# Prompts"}', first=True))
                self._sse({}, finish="tool_calls")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except OSError:
            pass


def test_real_socket_runaway_reconnects_within_seconds(monkeypatch):
    """End to end over a real socket: the guard's kill must actually end the
    looping attempt, not just mark it superseded while it streams on."""
    monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "30")
    _RunawayProvider.streamed = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RunawayProvider)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        agent = _make_agent(base_url=f"http://127.0.0.1:{server.server_address[1]}/v1", model="m")
        started = time.time()
        response = agent._interruptible_streaming_api_call(
            {"model": "m", "messages": [{"role": "user", "content": "write the file"}], "stream": True})
        elapsed = time.time() - started
    finally:
        server.shutdown()

    assert len(_RunawayProvider.streamed) == 2
    assert response.choices[0].message.tool_calls[0].function.arguments == (
        '{"path":"/tmp/e2e.md","content":"# Prompts"}')
    assert elapsed < _RunawayProvider.LOOP_SECONDS / 2
