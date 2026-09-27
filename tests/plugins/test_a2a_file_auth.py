"""A2A bearer-file and query-route regressions, exercised through the HTTP client."""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from plugins.platforms.a2a import tools


@pytest.mark.parametrize("content,mode,valid", [
    ("x" * 40 + "\n", 0o600, True),
    ("", 0o600, False),
    ("short", 0o600, False),
    ("x" * 32 + "\ny", 0o600, False),
    ("x" * 4097, 0o600, False),
    ("x" * 40, 0o644, False),
], ids=["private-valid", "empty", "short", "embedded-newline", "oversized", "public-mode"])
def test_bearer_file_requires_private_valid_token(tmp_path, content, mode, valid):
    if os.name == "nt" and mode == 0o644:
        pytest.skip("POSIX mode bits are not enforced on Windows")
    token = tmp_path / "bearer"
    token.write_text(content)
    token.chmod(mode)
    if valid:
        assert tools._auth_header({"type": "bearer", "token_file": str(token)}) == {
            "Authorization": "Bearer " + "x" * 40
        }
    else:
        with pytest.raises(ValueError):
            tools._auth_header({"type": "bearer", "token_file": str(token)})


def test_file_auth_and_query_route_reach_real_transport(tmp_path):
    token = tmp_path / "bearer"
    token.write_text("x" * 40 + "\n")
    token.chmod(0o600)
    requests = []

    class Peer(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_GET(self):
            requests.append(("GET", self.path, self.headers.get("Authorization")))
            card = {"supportedInterfaces": [{"protocolBinding": "JSONRPC", "url": "https://fixture.invalid/other/"}]}
            payload = json.dumps(card).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            requests.append(("POST", self.path, self.headers.get("Authorization")))
            size = int(self.headers["Content-Length"])
            body = json.loads(self.rfile.read(size))
            payload = json.dumps({"jsonrpc": "2.0", "id": body["id"], "result": {
                "status": {"state": "TASK_STATE_COMPLETED"},
                "artifacts": [{"parts": [{"kind": "text", "text": "fixture reply"}]}],
            }}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Peer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        peer = {"url": f"http://127.0.0.1:{server.server_port}/?direct=1",
                "auth": {"type": "bearer", "token_file": str(token)}, "timeout": 5}
        reply, _context, _state = tools._send_task("fixture-peer", peer, "hello", "fixture-context")
        assert reply == "fixture reply"
        assert requests == [
            ("GET", "/.well-known/agent-card.json?direct=1", "Bearer " + "x" * 40),
            ("POST", "/?direct=1", "Bearer " + "x" * 40),
        ]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
