"""A plugin may ASK the core to cut a stream; only the core decides whether that happens.

Two invariants, one per test: with ``plugins.stream_interrupt_requests`` on, the wire's stream loop
stops where the plugin asked and the turn ends on the plugin's reason; with the setting off (the
default), the same ask is recorded and the stream runs to completion. Both drive a real turn against
a local endpoint, so the loop between the ask and the cut is covered rather than assumed.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

_CHUNK_TOTAL = 120
_CHUNK_GATE_AT = 8   # the stream pauses here until the hook has had its say
_CHUNKS = [f"item {n}: this backlog entry has its own wording, its own owner and its own date.\n"
           for n in range(_CHUNK_TOTAL)]
_FULL_TEXT = "".join(_CHUNKS)


class _WireEndpoint:
    """Local SSE endpoint. It holds the stream open at ``_CHUNK_GATE_AT`` until the hook has asked,
    so the ask always lands mid-stream: the test never races a clock."""

    def __init__(self) -> None:
        self.requests = 0
        self.gate = threading.Event()
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
                try:
                    for frame in endpoint._chat_frames():
                        self.wfile.write(frame.encode())
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass  # the core closed the stream early, which is the point of the cut test

            def log_message(self, *args):
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self._server.server_port}/v1"

    def _pause_for_the_ask(self, index: int) -> None:
        """Block at the gate until the plugin has spoken (bounded, so a broken hook cannot hang CI)."""
        if index + 1 == _CHUNK_GATE_AT:
            self.gate.wait(timeout=5.0)

    def _chat_frames(self):
        for index, chunk in enumerate(_CHUNKS):
            yield "data: " + json.dumps({
                "id": "chatcmpl-fixture", "object": "chat.completion.chunk", "created": 0,
                "model": "fixture-local",
                "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
            }) + "\n\n"
            self._pause_for_the_ask(index)
        yield "data: " + json.dumps({
            "id": "chatcmpl-fixture", "object": "chat.completion.chunk", "created": 0,
            "model": "fixture-local",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }) + "\n\n"
        yield "data: [DONE]\n\n"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


def _ask_at(nth: int, server: "_WireEndpoint"):
    """A stream hook that asks for a cut on its nth text delta, then lets the stream continue."""
    seen = {"deltas": 0}

    def on_stream_delta(delta=None, kind=None, interrupt=None, **_kwargs):
        if kind != "text":
            return
        seen["deltas"] += 1
        if seen["deltas"] == nth and interrupt is not None:
            interrupt.request("the same block is repeating", source="test-guard")
            server.gate.set()

    return on_stream_delta, seen


def _run_wire_turn(monkeypatch, server, *, enabled):
    from run_agent import AIAgent

    agent = AIAgent(
        model="fixture-local", provider="custom", base_url=server.url, api_key="fixture",
        api_mode="chat_completions", quiet_mode=True, skip_memory=True, skip_context_files=True,
        enabled_toolsets=[], max_iterations=1,
    )
    delivered = {"text": 0}

    def count(text):
        delivered["text"] += 1
        return True

    agent.stream_delta_callback = count
    # The config key is read per response, so patching it after construction exercises the real
    # lookup without handing a fake config to the agent's own setup.
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"plugins": {"stream_interrupt_requests": enabled}},
    )
    outcome: dict = {}

    def run():
        try:
            outcome["result"] = agent.run_conversation("Summarize the backlog.")
        except BaseException as exc:  # surfaced below on the test thread
            outcome["error"] = exc

    turn = threading.Thread(target=run, daemon=True)
    turn.start()
    turn.join(60)
    if turn.is_alive():
        agent.interrupt("test bound reached")
        turn.join(10)
        pytest.fail("turn still streaming after 60s")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["result"], delivered["text"]


@pytest.fixture()
def wire(monkeypatch):
    from agent.plugin_stream_hooks import shutdown_plugin_stream_hook_dispatcher

    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    shutdown_plugin_stream_hook_dispatcher()
    servers: list[_WireEndpoint] = []

    def start(hook_at: int | None = None) -> tuple[_WireEndpoint, dict]:
        server = _WireEndpoint()
        servers.append(server)
        seen = {}
        if hook_at is not None:
            callback, seen = _ask_at(hook_at, server)
            monkeypatch.setattr("hermes_cli.plugins.iter_hook_callbacks",
                                lambda name: (callback,) if name == "on_stream_delta" else ())
        return server, seen

    yield start
    shutdown_plugin_stream_hook_dispatcher()
    for server in servers:
        server.close()


def test_the_stream_loop_stops_where_the_plugin_asked(monkeypatch, wire):
    server, _seen = wire(hook_at=8)

    result, delivered = _run_wire_turn(monkeypatch, server, enabled=True)

    assert (result["completed"], result["failure_reason"]) == (False, "truncated")
    assert "A Plugin Asked to Cut the Stream" in result["final_response"]
    # The loop stopped near the ask instead of reading all 120 chunks: that is the link this test
    # exists for. Slack covers deltas already in flight when the ask landed.
    assert delivered < _CHUNK_TOTAL // 2
    assert server.requests == 1
    assert _CHUNKS[-1].strip() not in result["final_response"]


def test_the_same_ask_is_recorded_and_ignored_when_the_setting_is_off(monkeypatch, wire):
    server, seen = wire(hook_at=8)

    result, delivered = _run_wire_turn(monkeypatch, server, enabled=False)

    assert result["completed"] is True
    assert result["final_response"].strip() == _FULL_TEXT.strip()
    assert delivered == _CHUNK_TOTAL
    assert seen["deltas"] == _CHUNK_TOTAL  # the plugin still saw everything it asked about
