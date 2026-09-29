"""A stream stuck in a repetition loop must not keep the turn alive for as long as it streams.

No output cap reaches an OpenAI-compatible endpoint by design (the server owns its generation
default), and the stale-stream detector fires only on silence, so a looping local model kept a
turn streaming indefinitely. Every repetition check ran after the stream ended, and the
callback-level attempts read only text a stream callback had accepted. These turns run with no
callback, the way cron, subagents and non-streaming gateway platforms call the model.

The loops are the ones reported: a Telegram gateway cycle (#125650), the fullwidth "！！！" echo
(#94224, the raw repeated characters of #125351) and a local model's status line (#78551).
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agent.repetition_guard import _STREAM_TAIL_CHARS, STOP_PATH_MIN_CHARS, RunawayStreamWatch

_TURN_BOUND_S = 30.0
# Reasoning whose plain field loops while the structured details beside it keep changing.
_LOOP_BEHIND_DETAILS = "reasoning-behind-details"

_CYCLE_125650 = (
    "De bot en de werkers blijven lopen. Ik zoek het punt waar de tekst binnenkomt. "
    "Daar stop ik de herhaling. De bot en de werkers blijven lopen. Ik zoek dat punt nu. "
)
_FULLWIDTH_ECHO_94224 = "！！！" * 12
_STATUS_LINE_78551 = "- **Texture** : `TEXTURE_LOCAL.md` (trace creative, emergence). Pas dans un Hermes stock.\n"

# Asked-for repetition stays below the stop path's scale floor (#121600's repeat-on-request case).
_ASKED_FOR_REPEAT = "Hello world, this is a sentence the user asked me to repeat many times.\n" * 50
# Distinct rows sharing a long suffix: repeated windows dominate, yet every line differs.
_DISTINCT_ROWS = "".join(
    f"| {i:05d} | status: pending review by the on-call engineer before the next release window |\n"
    for i in range(500)
)


class _FakeEndpoint:
    """Local endpoint streaming one reply per request, on the chat or the Anthropic wire."""

    def __init__(self, channel: str, text: str, *, endless: bool) -> None:
        self.channel, self.text, self.endless = channel, text, endless
        self.requests = 0
        self._stop = threading.Event()
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self):  # model metadata probes
                body = json.dumps({"object": "list", "data": [
                    {"id": "fixture-local", "object": "model", "context_length": 131072}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                endpoint.requests += 1
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                frames = endpoint._anthropic_frames() if self.path.endswith("/messages") else endpoint._chat_frames()
                try:
                    for frame in frames:
                        self.wfile.write(frame.encode())
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *args):
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self._server.server_port}/v1"

    def _pieces(self):
        if not self.endless:
            yield from (self.text[i:i + 200] for i in range(0, len(self.text), 200))
            return
        while not self._stop.is_set():
            yield self.text
            time.sleep(0.001)

    def _chat_frames(self):
        field = "reasoning_content" if self.channel == "reasoning" else "content"

        def chunk(delta, finish_reason=None):
            return "data: " + json.dumps({
                "id": "chatcmpl-fixture", "object": "chat.completion.chunk", "created": 0, "model": "fixture-local",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }) + "\n\n"

        for n, piece in enumerate(self._pieces()):
            if self.channel == _LOOP_BEHIND_DETAILS:
                yield chunk({"reasoning_content": piece, "reasoning_details": [
                    {"type": "reasoning.text", "text": f"Step {n}: weighing option {n * 7919}. ", "index": 0}]})
            else:
                yield chunk({field: piece})
        yield chunk({}, "stop")
        yield "data: [DONE]\n\n"

    def _anthropic_frames(self):
        thinking = self.channel == "reasoning"

        def event(kind, **payload):
            return f"event: {kind}\ndata: {json.dumps({'type': kind, **payload})}\n\n"

        yield event("message_start", message={
            "id": "msg_fixture", "type": "message", "role": "assistant", "model": "fixture-local", "content": [],
            "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 0}})
        yield event("content_block_start", index=0, content_block=(
            {"type": "thinking", "thinking": "", "signature": ""} if thinking else {"type": "text", "text": ""}))
        for piece in self._pieces():
            yield event("content_block_delta", index=0, delta=(
                {"type": "thinking_delta", "thinking": piece} if thinking else {"type": "text_delta", "text": piece}))
        yield event("content_block_stop", index=0)
        yield event("message_delta", delta={"stop_reason": "end_turn", "stop_sequence": None},
                    usage={"output_tokens": 1})
        yield event("message_stop")

    def close(self) -> None:
        self._stop.set()
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture()
def endpoint(monkeypatch):
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    started = []

    def start(channel: str, text: str, *, endless: bool) -> _FakeEndpoint:
        started.append(_FakeEndpoint(channel, text, endless=endless))
        return started[-1]

    yield start
    for server in started:
        server.close()


def _run_turn(server: _FakeEndpoint, api_mode: str) -> dict:
    from run_agent import AIAgent

    agent = AIAgent(
        model="fixture-local", provider="custom", base_url=server.url, api_key="fixture", api_mode=api_mode,
        quiet_mode=True, skip_memory=True, skip_context_files=True, enabled_toolsets=[], max_iterations=3,
    )
    outcome: dict = {}

    def run():
        try:
            outcome["result"] = agent.run_conversation("Summarize the backlog.")
        except BaseException as exc:  # surfaced below, on the test thread
            outcome["error"] = exc

    turn = threading.Thread(target=run, daemon=True)
    turn.start()
    turn.join(_TURN_BOUND_S)
    if turn.is_alive():
        agent.interrupt("test bound reached")
        turn.join(10)
        pytest.fail(f"turn still streaming after {_TURN_BOUND_S:.0f}s")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["result"]


@pytest.mark.parametrize(
    ("api_mode", "channel", "loop"),
    [
        ("chat_completions", "content", _CYCLE_125650),
        ("chat_completions", "reasoning", _FULLWIDTH_ECHO_94224),
        ("chat_completions", _LOOP_BEHIND_DETAILS, _CYCLE_125650),
        ("anthropic_messages", "content", _STATUS_LINE_78551),
        ("anthropic_messages", "reasoning", _CYCLE_125650),
    ],
    ids=["chat-content", "chat-reasoning", "chat-reasoning-behind-details", "anthropic-text", "anthropic-thinking"],
)
def test_looping_stream_is_cut_without_a_stream_callback(endpoint, api_mode, channel, loop):
    server = endpoint(channel, loop, endless=True)

    result = _run_turn(server, api_mode)

    assert (result["completed"], result["failure_reason"]) == (False, "truncated")
    assert "Repetition Detected" in result["final_response"]
    assert server.requests == 1  # cut once; never continued or retried into the same loop
    unit = loop.strip()[:40]
    assert not any(unit in json.dumps(m, ensure_ascii=False) for m in result["messages"])


@pytest.mark.parametrize("text", [_ASKED_FOR_REPEAT, _DISTINCT_ROWS], ids=["asked-for-repeat", "distinct-rows"])
def test_repetitive_but_legitimate_stream_is_delivered(endpoint, text):
    assert (len(text) < STOP_PATH_MIN_CHARS) == (text is _ASKED_FOR_REPEAT)
    server = endpoint("content", text, endless=False)

    result = _run_turn(server, "chat_completions")

    assert result["completed"] is True
    assert result["final_response"] == text.strip()


def test_watch_holds_a_bounded_tail_however_long_the_stream_runs():
    watch, chunk_chars, held_max = RunawayStreamWatch(), 0, 0
    for n in range(12_000):  # far past several doublings of the check threshold
        chunk = f"Row {n}: measured {n * 7919} units at site {n % 97}, carried into ledger {n * 31}.\n"
        assert watch.feed(chunk) is False
        chunk_chars = max(chunk_chars, len(chunk))
        held_max = max(held_max, sum(map(len, watch._parts)))

    assert watch._chars > 8 * _STREAM_TAIL_CHARS
    assert held_max <= 2 * _STREAM_TAIL_CHARS + chunk_chars
