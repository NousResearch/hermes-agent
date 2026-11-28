"""Regression for #153136: a server-default truncation must not shrink the retry's wire cap."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from run_agent import AIAgent


def test_length_continuation_wire_cap_exceeds_server_default_usage():
    """Exercise the actual stream collector, retry loop, and next request body."""
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.send_error(404)
                return
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(request)
            first = len(requests) == 1
            common = {"id": "local-length", "object": "chat.completion.chunk",
                      "created": 1, "model": "fake-model"}
            chunks = [
                {**common, "choices": [{"index": 0,
                    "delta": {"role": "assistant", "content": "partial" if first else "complete"},
                    "finish_reason": "length" if first else "stop"}]},
                {**common, "choices": [], "usage": {
                    "prompt_tokens": 12, "completion_tokens": 16384 if first else 5,
                    "total_tokens": 16396 if first else 17}},
            ]
            data = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
            payload = (data + "data: [DONE]\n\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    agent = None
    try:
        agent = AIAgent(
            model="fake-model", base_url=f"http://127.0.0.1:{server.server_port}/v1",
            api_key="local-only", provider="custom", api_mode="chat_completions",
            platform="subagent", quiet_mode=True, enabled_toolsets=[],
            max_iterations=3, skip_memory=True, skip_context_files=True,
            skip_background_review=True, save_trajectories=False,
        )
        assert agent.max_tokens is None
        agent.run_conversation("say hi")

        assert len(requests) >= 2
        assert requests[0].get("stream") is True
        assert "max_tokens" not in requests[0]
        assert requests[1]["max_tokens"] > 16384
    finally:
        if agent is not None:
            agent.client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
