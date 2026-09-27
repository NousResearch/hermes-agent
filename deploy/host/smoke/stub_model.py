#!/usr/bin/env python3
"""Stub OpenAI-compatible model for the host smoke test (stdlib only).

Answers every chat completion with a fixed sentence, streamed or not, so the
real Hermes agent inside the smoke container can run one turn end to end with
no provider key and no network. Listens on 127.0.0.1:18080 by default.
"""

from __future__ import annotations

import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "stub-model"
ANSWER = "SMOKE-OK: the matter host answered through the stub model."


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # keep the journal readable
        sys.stderr.write("stub-model: " + (fmt % args) + "\n")

    def _json(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._json(200, {"object": "list", "data": [
                {"id": MODEL, "object": "model", "owned_by": "smoke", "context_length": 200000}]})
        else:
            self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            req = {}
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._json(404, {"error": {"message": "not found"}})
            return
        usage = {"prompt_tokens": 42, "completion_tokens": 12, "total_tokens": 54}
        created = int(time.time())
        if not req.get("stream"):
            self._json(200, {"id": "chatcmpl-smoke", "object": "chat.completion", "created": created,
                             "model": MODEL, "usage": usage, "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": ANSWER}}]})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def chunk(delta, finish=None, extra=None):
            body = {"id": "chatcmpl-smoke", "object": "chat.completion.chunk", "created": created,
                    "model": MODEL, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            body.update(extra or {})
            self.wfile.write(f"data: {json.dumps(body)}\n\n".encode())
            self.wfile.flush()

        chunk({"role": "assistant", "content": ""})
        for word in ANSWER.split(" "):
            chunk({"content": word + " "})
        chunk({}, "stop", {"usage": usage})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 18080
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
