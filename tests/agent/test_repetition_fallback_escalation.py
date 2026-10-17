"""A repetition loop must hand the turn to the fallback chain instead of ending it.

A degenerate loop is model-specific, not content-deterministic like a content filter: the
same prompt usually answers cleanly on a different backend. Before this, a looping primary
ended the turn with "Response Stopped — Repetition Detected" and the user had to re-send by
hand; the configured ``fallback_providers`` chain was never consulted, even though every
other unrecoverable stream failure (content filter, empty responses) escalates to it.

These tests pin the escalation itself: the loop is cut, the chain is consulted, and the
answer arrives from the fallback provider. They also pin the negative case — with an
exhausted chain the turn still ends with the existing user copy, so the fix cannot turn a
genuine dead end into a silent hang.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agent.repetition_guard import STOP_PATH_MIN_CHARS

_TURN_BOUND_S = 60.0

# A single line re-emitted forever: the shape the stream watch cuts.
_LOOP_LINE = (
    "De bot en de werkers blijven lopen. Ik zoek het punt waar de tekst binnenkomt. "
    "Daar stop ik de herhaling. De bot en de werkers blijven lopen. Ik zoek dat punt nu. "
)

# What the healthy fallback provider answers.
_FALLBACK_ANSWER = "Готово: вот связный ответ вместо зацикленного фрагмента."


class _TwoBackendEndpoint:
    """One endpoint: the first N requests loop, every later request answers cleanly.

    Modelling both providers on one socket keeps the fixture small; what matters is that the
    agent sees a looping stream first and a healthy completion after it switches.
    """

    def __init__(self, *, loop_requests: int) -> None:
        self.loop_requests = loop_requests
        self.requests = 0
        self.loop_hits = 0
        self._stop = threading.Event()
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self):  # model metadata probes
                body = json.dumps({"object": "list", "data": [
                    {"id": "fixture-local", "object": "model", "context_length": 131072},
                    {"id": "fixture-fallback", "object": "model", "context_length": 131072},
                ]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                endpoint.requests += 1
                looping = endpoint.loop_hits < endpoint.loop_requests
                if looping:
                    endpoint.loop_hits += 1
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                frames = endpoint._loop_frames() if looping else endpoint._answer_frames()
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

    def _chunk(self, delta, finish_reason=None):
        return "data: " + json.dumps({
            "id": "chatcmpl-fixture", "object": "chat.completion.chunk", "created": 0,
            "model": "fixture-local",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }) + "\n\n"

    def _loop_frames(self):
        """Endless repetition: the stream watch must cut this before the turn bound."""
        yield self._chunk({"role": "assistant", "content": ""})
        emitted = 0
        while not self._stop.is_set():
            if emitted >= 100_000:  # a watch that has not cut by now hangs the test
                self._stop.wait()
                return
            yield self._chunk({"content": _LOOP_LINE})
            emitted += len(_LOOP_LINE)
            time.sleep(0.001)

    def _answer_frames(self):
        yield self._chunk({"role": "assistant", "content": ""})
        for i in range(0, len(_FALLBACK_ANSWER), 40):
            yield self._chunk({"content": _FALLBACK_ANSWER[i:i + 40]})
        yield self._chunk({}, finish_reason="stop")
        yield "data: [DONE]\n\n"


@pytest.fixture
def endpoint():
    server = _TwoBackendEndpoint(loop_requests=1)
    try:
        yield server
    finally:
        server._stop.set()
        server._server.shutdown()


def _run_turn(server, *, fallback_chain) -> dict:
    """Run one turn against the fixture with an explicit fallback chain."""
    from run_agent import AIAgent

    agent = AIAgent(
        model="fixture-local", provider="custom", base_url=server.url, api_key="fixture",
        api_mode="chat_completions", quiet_mode=True, skip_memory=True, skip_context_files=True,
        enabled_toolsets=[], max_iterations=3,
    )
    # Install the chain the way config does, then pin every entry to the fixture socket so
    # the switch resolves without real credentials.
    agent._fallback_chain = list(fallback_chain)
    agent._fallback_index = 0
    agent._fallback_model = agent._fallback_chain[0] if agent._fallback_chain else None

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
        pytest.fail(f"turn still running after {_TURN_BOUND_S:.0f}s")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["result"]


def test_repetition_loop_escalates_to_the_fallback_chain(endpoint, monkeypatch):
    """The loop is cut and the answer arrives from the fallback provider."""
    # The chain resolves through the provider registry; point it at the fixture socket.
    monkeypatch.setattr(
        "agent.chat_completion_helpers._fallback_api_mode_hint",
        lambda fb, provider, base_url: (True, "chat_completions"),
    )
    monkeypatch.setattr(
        "hermes_cli.fallback_config.resolve_entry_api_key", lambda entry: "fixture"
    )

    result = _run_turn(endpoint, fallback_chain=[
        {"provider": "custom", "model": "fixture-fallback", "base_url": endpoint.url},
    ])

    # The fallback answered: the turn produced the healthy text, not the stopped notice.
    assert _FALLBACK_ANSWER[:20] in json.dumps(result, ensure_ascii=False), (
        f"fallback answer missing from the turn result: {result!r}"
    )
    assert "Repetition Detected" not in json.dumps(result, ensure_ascii=False), (
        "the turn still ended with the repetition-stopped notice despite a live fallback"
    )
    # Both backends were actually exercised.
    assert endpoint.loop_hits >= 1, "the looping backend was never called"
    assert endpoint.requests > endpoint.loop_hits, (
        "no request reached the fallback backend: the chain was never consulted"
    )


def test_exhausted_chain_still_ends_with_the_repetition_notice(endpoint):
    """No fallback configured: the existing end-the-turn copy is unchanged (no silent hang)."""
    result = _run_turn(endpoint, fallback_chain=[])

    assert "Repetition Detected" in json.dumps(result, ensure_ascii=False), (
        f"an exhausted chain must keep the existing user copy: {result!r}"
    )
