"""SSE comment keepalives are transport liveness, not output (#151280).

The OpenAI SDK drops ``: keepalive`` comment lines before they become chunks/events, so
a proxy keeping a thinking model's connection alive with them looked silent to the
stale-stream watchdogs and was killed mid-think. Raw bytes on the wire now stretch the
patience by a bounded factor; a stream that only ever pings is still cut
(``test_inline_stream_stale_detector_still_fires_from_monitor_thread`` pins that).
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import httpx
import pytest

import run_agent
from agent.chat_completion_helpers import interruptible_streaming_api_call


def _chunk(delta, finish=None):
    return ("data: " + json.dumps({"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m",
                                   "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}) + "\n\n").encode()


@pytest.fixture
def thinking_proxy():
    """Role chunk, then only ``: heartbeat`` comments for ``think`` seconds, then the answer."""

    class Handler(BaseHTTPRequestHandler):
        think = 3.0

        def log_message(self, *_a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("content-length", 0)))
            if not self.path.endswith("/chat/completions"):
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            self.wfile.write(_chunk({"role": "assistant", "content": ""}))
            deadline = time.time() + self.think
            while time.time() < deadline:
                time.sleep(0.1)  # health: allow HX013 -- the keepalive cadence IS the behaviour under test
                self.wfile.write(b": opencodex heartbeat\n\n")
                self.wfile.flush()
            self.wfile.write(_chunk({"content": "thought it through"}) + _chunk({}, "stop") + b"data: [DONE]\n\n")
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()
    server.server_close()


def test_keepalive_comments_keep_a_thinking_chat_stream_alive(thinking_proxy, monkeypatch):
    """3s of keepalive-only silence against a 1s stale budget: the answer still arrives."""
    monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "1.0")
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")
    agent = run_agent.AIAgent(api_key="test-key", base_url=thinking_proxy, model="m", provider="custom",
                              platform="subagent", quiet_mode=True, skip_context_files=True, skip_memory=True,
                              enabled_toolsets=[], max_iterations=1)

    response = interruptible_streaming_api_call(agent, {"model": "m", "messages": [{"role": "user", "content": "hi"}]})

    assert response.choices[0].message.content == "thought it through"


@pytest.mark.parametrize("ttfb, killed", [("1.0", False), ("0.5", True)])
def test_keepalive_comments_count_as_codex_watchdog_activity(tmp_path, monkeypatch, ttfb, killed):
    """Codex Responses wire (contributed in #78818): 2.5s of keepalive comments every 0.25s,
    zero SDK events. They keep the stream alive past a 1s TTFB cutoff (4s stretched); the
    stretch is bounded, so a 0.5s cutoff (2s stretched) still kills a keepalive-only stream."""
    import openai
    from openai._streaming import Stream

    from agent import chat_completion_helpers as h
    from agent import codex_runtime

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", ttfb)
    monkeypatch.setenv("HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS", ttfb)
    agent = run_agent.AIAgent(model="grok-4.5", provider="xai-oauth", api_key="sk-dummy", base_url="https://api.x.ai/v1",
                              quiet_mode=True, skip_context_files=True, skip_memory=True, platform="cli")
    agent.api_mode = "codex_responses"
    monkeypatch.setattr(agent, "_compute_non_stream_stale_timeout", lambda *a, **k: 60.0)
    closes: list = []
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **k: SimpleNamespace())
    monkeypatch.setattr(agent, "_abort_request_openai_client", lambda c, reason=None: closes.append(reason))
    monkeypatch.setattr(agent, "_close_request_openai_client", lambda c, reason=None: closes.append(reason))

    class _Keepalives(httpx.SyncByteStream):
        def __iter__(self):
            for _ in range(10):  # 2.5s of bytes, no parseable event
                time.sleep(0.25)  # health: allow HX013 -- the keepalive cadence IS the behaviour under test
                yield b": keepalive\n\n"

    response = httpx.Response(200, headers={"Content-Type": "text/event-stream"}, stream=_Keepalives(),
                              request=httpx.Request("POST", "https://api.x.ai/v1/responses"))
    sdk_stream = Stream(cast_to=object, response=response, client=openai.OpenAI(api_key="sk-dummy"))
    stream_client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kw: sdk_stream))
    monkeypatch.setattr(agent, "_run_codex_stream", lambda api_kwargs, client=None, on_first_delta=None:
                        codex_runtime.run_codex_stream(agent, api_kwargs, client=stream_client))

    # Survives: the stream's own end (no terminal response). Bounded: the TTFB watchdog fires.
    with pytest.raises(TimeoutError if killed else RuntimeError):
        h.interruptible_api_call(agent, {"model": "grok-4.5", "input": "hi"})

    assert ("codex_ttfb_kill" in closes) is killed
