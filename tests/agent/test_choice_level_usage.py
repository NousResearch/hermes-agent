"""Usage an OpenAI-compatible provider reports only inside ``choices[0]`` is still counted.

Moonshot/Kimi chat endpoints (and gateways copying them) can put the usage object on the choice
instead of the chunk/response top level. The OpenAI SDK keeps it as an untyped extra, so a reader
of ``.usage`` saw ``None``: the turn's tokens, the context gauge and the compaction trigger all read
zero. The endpoint below is a real HTTP server, and the turn is a real ``AIAgent`` turn.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

_USAGE = {"prompt_tokens": 1234, "completion_tokens": 56, "total_tokens": 1290}


@pytest.fixture()
def endpoint(monkeypatch):
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, status, body, ctype="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send(200, json.dumps({"object": "list", "data": [{"id": "fixture", "object": "model"}]}).encode())

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
            base = {"id": "c1", "model": "fixture", "created": 0}
            if not req.get("stream"):
                self._send(200, json.dumps({**base, "object": "chat.completion", "choices": [{
                    "index": 0, "message": {"role": "assistant", "content": "Hello."}, "finish_reason": "stop",
                    "usage": _USAGE}]}).encode())
                return
            chunks = [
                {**base, "object": "chat.completion.chunk", "choices": [
                    {"index": 0, "delta": {"role": "assistant", "content": "Hello."}, "finish_reason": None}]},
                {**base, "object": "chat.completion.chunk", "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "stop", "usage": _USAGE}]},
            ]
            body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
            self._send(200, body.encode(), "text/event-stream")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/v1"
    server.shutdown()
    server.server_close()


@pytest.mark.parametrize("streaming", [True, False], ids=["streamed", "non-streamed"])
def test_choice_level_usage_is_counted(endpoint, streaming):
    from run_agent import AIAgent

    agent = AIAgent(model="fixture", provider="custom", base_url=endpoint, api_key="fixture",
                    quiet_mode=True, skip_memory=True, skip_context_files=True, enabled_toolsets=[],
                    max_iterations=2)
    if not streaming:
        agent._disable_streaming = True
    try:
        result = agent.run_conversation("Say hello.")
        counts = (agent.session_input_tokens, agent.session_output_tokens, agent.context_compressor.last_prompt_tokens)
    finally:
        agent.close()

    assert result["final_response"] == "Hello."
    assert counts == (1234, 56, 1234)
