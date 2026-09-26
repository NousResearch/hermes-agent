"""Scriptable fake OpenAI-compatible HTTP server — a REAL local socket.

Drives genuine HTTP failure modes (429, 5xx, connection-reset mid-stream)
through the REAL ``openai`` SDK client construction path
(``agent.auxiliary_client.resolve_provider_client``), so fault-injection
tests exercise the actual error-surfacing code instead of a ``MagicMock``
standing in for the transport.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional

__all__ = ["FakeProviderServer"]


class FakeProviderServer:
    """A real ``HTTPServer`` bound to 127.0.0.1:0, serving a scripted queue.

    Each POST to ``/v1/chat/completions`` pops the next scripted response.
    Use :meth:`script` for a plain JSON response (any status code) or
    :meth:`script_truncated_stream` for an SSE stream that closes mid-way
    without a terminating ``[DONE]`` — the real shape of a server crash.
    """

    def __init__(self) -> None:
        self._responses: List[tuple] = []
        self._requests: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        handler = self._make_handler()
        self._httpd = HTTPServer(("127.0.0.1", 0), handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self._httpd.server_address
        return f"http://{host}:{port}/v1"

    @property
    def requests(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._requests)

    def script(self, status: int, body: Optional[Dict[str, Any]] = None) -> None:
        """Queue a plain JSON response."""
        with self._lock:
            self._responses.append(("json", status, body or {}))

    def script_truncated_stream(self, chunks: List[str]) -> None:
        """Queue an SSE stream that sends ``chunks`` then closes without [DONE]."""
        with self._lock:
            self._responses.append(("truncated_stream", 200, chunks))

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()

    def __enter__(self) -> "FakeProviderServer":
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()

    def _make_handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:  # silence stdlib access log
                pass

            def do_POST(self) -> None:  # noqa: N802 — stdlib handler contract
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except Exception:
                    payload = {}
                with server._lock:
                    server._requests.append({"path": self.path, "body": payload})
                    if not server._responses:
                        kind, status, data = (
                            "json", 500,
                            {"error": {"message": "fault_lab: no scripted response left"}},
                        )
                    else:
                        kind, status, data = server._responses.pop(0)

                if kind == "json":
                    body = json.dumps(data).encode("utf-8")
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                # truncated_stream: real SSE framing, then the connection
                # closes mid-stream — no [DONE], no final chunk. This is the
                # real shape of a provider crash, not a simulated one.
                self.send_response(status)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for chunk_text in data:
                    event = {
                        "id": "fault-lab",
                        "object": "chat.completion.chunk",
                        "model": "fault-lab-model",
                        "choices": [
                            {"delta": {"content": chunk_text}, "index": 0,
                             "finish_reason": None}
                        ],
                    }
                    self.wfile.write(f"data: {json.dumps(event)}\n\n".encode("utf-8"))
                    self.wfile.flush()
                self.close_connection = True

        return Handler
