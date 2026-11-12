"""getUpdates long-poll socket reuse, per OS (#87057 Windows deadlock vs #107880 TIME_WAIT exhaustion).

Drives the real ``build_ptb_requests`` limits into a real httpx client against a local HTTP/1.1 server
and counts the TCP connections the polls opened.
"""

import asyncio
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from hermes_agent.plugins.platforms.telegram.telegram_network import build_ptb_requests


class _Recorder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _tcp_connections_for_polls(polls: int) -> int:
    ports: set[int] = set()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            ports.add(self.client_address[1])
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            body = b'{"ok":true,"result":[]}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    _general, updates = build_ptb_requests(_Recorder, fallback_ips=[], proxy_url=None)
    httpx_kwargs = {k: v for k, v in updates.kwargs["httpx_kwargs"].items() if k != "verify"}

    async def run():
        async with httpx.AsyncClient(**httpx_kwargs) as client:
            for _ in range(polls):
                (await client.post(f"http://127.0.0.1:{server.server_port}/botT/getUpdates", json={})).raise_for_status()

    try:
        asyncio.run(run())
    finally:
        server.shutdown()
        server.server_close()
    return len(ports)


@pytest.mark.platforms("posix")
def test_getupdates_polls_reuse_one_socket_off_windows():
    assert _tcp_connections_for_polls(10) == 1


@pytest.mark.platforms("windows")
def test_getupdates_polls_never_reuse_a_socket_on_windows():
    assert _tcp_connections_for_polls(5) == 5
