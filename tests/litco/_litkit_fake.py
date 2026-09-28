"""A fake LitKit instance for the toolset tests: a real HTTP server on localhost.

Routes are registered as ``fake.route(method, path_regex, handler)``; a handler receives the
recorded request and returns ``(status, body)`` or ``(status, body, headers)``. ``body`` may be
a dict/list (sent as JSON), ``bytes``, or a ``str``. Every request is recorded with its method,
path, query, headers and raw body so tests can assert what the client sent.
"""

from __future__ import annotations

import json
import re
import socketserver
import threading
from dataclasses import dataclass, field
from email.parser import BytesParser
from email.policy import default as email_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Tuple
from urllib.parse import parse_qs, urlsplit

TOKEN = "lkm_" + "A" * 43
HOST_SECRET = "host-secret-for-tests"
MATTER_ID = "11111111-2222-3333-4444-555555555555"
USER_ID = "99999999-8888-7777-6666-555555555555"


@dataclass
class Recorded:
    method: str
    path: str
    query: Dict[str, List[str]]
    headers: Dict[str, str]
    body: bytes
    raw_headers: List[Tuple[str, str]] = field(default_factory=list)

    def json(self) -> Any:
        return json.loads(self.body.decode("utf-8") or "null")

    def form(self) -> Dict[str, Any]:
        """Parse a multipart body into ``{field: str | (filename, bytes)}``."""
        ctype = self.headers.get("content-type", "")
        msg = BytesParser(policy=email_policy).parsebytes(
            f"Content-Type: {ctype}\r\n\r\n".encode("utf-8") + self.body)
        out: Dict[str, Any] = {}
        for part in msg.iter_parts():
            name = part.get_param("name", header="content-disposition")
            filename = part.get_filename()
            payload = part.get_payload(decode=True) or b""
            out[name] = (filename, payload) if filename else payload.decode("utf-8")
        return out


class _QuickServer(ThreadingHTTPServer):
    """``HTTPServer.server_bind`` does a reverse DNS lookup (``getfqdn``) that can stall for
    tens of seconds on some hosts; skip it."""

    daemon_threads = True

    def server_bind(self) -> None:
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name, self.server_port = host, port


class FakeLitKit:
    def __init__(self) -> None:
        self.routes: List[Tuple[str, re.Pattern, Callable[[Recorded], Any]]] = []
        self.requests: List[Recorded] = []
        self._lock = threading.Lock()
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a: Any) -> None:  # quiet
                pass

            def _handle(self) -> None:
                length = int(self.headers.get("content-length") or 0)
                body = self.rfile.read(length) if length else b""
                parts = urlsplit(self.path)
                rec = Recorded(method=self.command, path=parts.path, query=parse_qs(parts.query, keep_blank_values=True),
                               headers={k.lower(): v for k, v in self.headers.items()}, body=body,
                               raw_headers=list(self.headers.items()))
                with fake._lock:
                    fake.requests.append(rec)
                result = fake._dispatch(rec)
                if not isinstance(result, tuple):
                    result = (200, result)
                status, payload, headers = result if len(result) == 3 else (result[0], result[1], {})
                if isinstance(payload, (dict, list)):
                    data = json.dumps(payload).encode("utf-8")
                    headers = {"Content-Type": "application/json", **headers}
                elif isinstance(payload, str):
                    data = payload.encode("utf-8")
                    headers = {"Content-Type": "text/plain", **headers}
                else:
                    data = payload or b""
                    headers = {"Content-Type": "application/octet-stream", **headers}
                self.send_response(status)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            do_GET = do_POST = do_PATCH = do_DELETE = do_PUT = _handle

        self.server = _QuickServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05},
                                       daemon=True)

    @property
    def url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> "FakeLitKit":
        self.thread.start()
        return self

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def route(self, method: str, pattern: str, handler: Any) -> None:
        fn = handler if callable(handler) else (lambda _req, _h=handler: _h)
        self.routes.insert(0, (method.upper(), re.compile("^" + pattern + "$"), fn))

    def _dispatch(self, rec: Recorded) -> Tuple:
        for method, pattern, fn in self.routes:
            if method == rec.method and pattern.match(rec.path):
                return fn(rec)
        return (404, {"error": "not_found"})

    def calls(self, method: str, pattern: str) -> List[Recorded]:
        rx = re.compile("^" + pattern + "$")
        return [r for r in self.requests if r.method == method and rx.match(r.path)]


def ndjson(rows: List[Dict[str, Any]]) -> Tuple[int, bytes, Dict[str, str]]:
    return 200, ("\n".join(json.dumps(r) for r in rows) + "\n").encode("utf-8"), {
        "Content-Type": "application/x-ndjson"}
