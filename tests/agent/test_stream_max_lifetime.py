"""Opt-in ``agent.stream_max_lifetime``: a stream that keeps sending chunks but
never finishes is cut at the cap (the stale detector resets on every chunk and
never fires). Off by default."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import run_agent
from agent.chat_completion_helpers import interruptible_streaming_api_call
from agent.stream_max_lifetime import StreamLifetimeExceeded, resolve_stream_max_lifetime
from hermes_cli.config_defaults import DEFAULT_CONFIG


class _DripWire:
    """Real OpenAI-wire SSE server: one content chunk every 50 ms for ~8 s, then a clean finish
    (a "short" prompt gets one chunk and the finish)."""

    def __init__(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
                short = (body.get("messages") or [{}])[-1].get("content") == "short"
                if not self.path.endswith("/chat/completions"):
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                chunk = {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m",
                         "choices": [{"index": 0, "delta": {"content": "."}, "finish_reason": None}]}
                fin = {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                try:
                    for _ in range(1 if short else 160):
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                        self.wfile.flush()
                        time.sleep(0.05)  # health: allow HX013 -- the drip cadence is the behaviour under test
                    self.wfile.write(f"data: {json.dumps(fin)}\n\ndata: [DONE]\n\n".encode())
                    self.wfile.flush()
                except OSError:
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"


@pytest.fixture
def drip_wire():
    w = _DripWire()
    yield w
    w.server.shutdown()
    w.server.server_close()


def test_cap_cuts_a_stream_that_never_goes_quiet(drip_wire, monkeypatch):
    monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "1.0")
    monkeypatch.setenv("HERMES_STREAM_MAX_LIFETIME", "1.5")
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")
    monkeypatch.setenv("HERMES_STREAM_STALE_GIVEUP", "2")
    agent = run_agent.AIAgent(
        api_key="test-key", base_url=drip_wire.base_url, model="m", provider="custom",
        platform="cli", quiet_mode=True, skip_context_files=True, skip_memory=True,
        enabled_toolsets=[], max_iterations=1,
    )
    def call(prompt):
        return interruptible_streaming_api_call(agent, {"model": "m", "messages": [{"role": "user", "content": prompt}]})

    # Without the cap the drip completes normally after ~8 s; with it each call ends early, with
    # an error naming the setting (the stream opened, so no "endpoint unreachable" copy).
    for _ in range(2):
        started = time.time()
        with pytest.raises(StreamLifetimeExceeded, match="agent.stream_max_lifetime"):
            call("hi")
        assert time.time() - started < 5.0
    # Cap kills are not stale kills: the breaker (giveup 2) must not refuse a healthy request.
    assert call("short").choices[0].message.content == "."


def test_cap_is_off_by_default_and_never_undercuts_the_stale_timeout(monkeypatch):
    monkeypatch.delenv("HERMES_STREAM_MAX_LIFETIME", raising=False)
    assert DEFAULT_CONFIG["agent"]["stream_max_lifetime"] == 0
    monkeypatch.setenv("HERMES_STREAM_MAX_LIFETIME", "60")
    assert resolve_stream_max_lifetime(180.0) == 180.0
    assert resolve_stream_max_lifetime(float("inf")) == 60.0
    monkeypatch.setenv("HERMES_STREAM_MAX_LIFETIME", "0")
    assert resolve_stream_max_lifetime(180.0) == 0.0
