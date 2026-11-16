"""scripts/ci/check_catalog_cors.py against a loopback docs host that redirects like the real one."""

from __future__ import annotations

import json
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "ci"))

import check_catalog_cors as probe


class _DocsHost(BaseHTTPRequestHandler):
    """``/bare`` and ``/allowed`` 301 to ``/plugins.json``; only ``/allowed`` sends the CORS header."""

    def do_GET(self) -> None:
        if self.path == "/plugins.json":
            body = json.dumps([{"name": "fish-audio"}]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(301)
        self.send_header("Location", "/plugins.json")
        if self.path == "/allowed":
            self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def log_message(self, format: str, *args) -> None:
        pass


@pytest.fixture
def docs_host() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _DocsHost)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def test_redirect_without_cors_header_fails_the_probe(docs_host, capsys):
    assert probe.main(["probe", f"{docs_host}/bare"]) == 1
    out = capsys.readouterr().out
    assert out.startswith("blocked: 301 from ") and "/bare" in out


def test_redirect_with_cors_header_reaches_the_catalog(docs_host, capsys):
    assert probe.main(["probe", f"{docs_host}/allowed"]) == 0
    assert capsys.readouterr().out.strip() == "ok: 1 rows after 1 redirect(s)"
