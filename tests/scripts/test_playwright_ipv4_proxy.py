"""Behavioral test for the installer's temporary IPv4 CONNECT proxy."""

from __future__ import annotations

import base64
import os
import socket
import socketserver
import subprocess
import sys
import threading
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PROXY_SCRIPT = REPO_ROOT / "scripts" / "playwright_ipv4_proxy.py"


class EchoHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        while data := self.request.recv(65536):
            self.request.sendall(data)


class ThreadingServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def _wait_for_proxy(
    port_file: Path, token_file: Path, process: subprocess.Popen[str]
) -> tuple[int, str]:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if (
            port_file.exists()
            and port_file.stat().st_size
            and token_file.exists()
            and token_file.stat().st_size
        ):
            return (
                int(port_file.read_text(encoding="ascii")),
                token_file.read_text(encoding="ascii"),
            )
        if process.poll() is not None:
            _, stderr = process.communicate()
            raise AssertionError(f"proxy exited before startup: {stderr}")
        time.sleep(0.05)
    raise AssertionError("proxy did not publish its port within 5 seconds")


def _recv_headers(connection: socket.socket) -> bytes:
    response = bytearray()
    while b"\r\n\r\n" not in response and len(response) < 65536:
        chunk = connection.recv(4096)
        if not chunk:
            break
        response.extend(chunk)
    return bytes(response)


def test_connect_tunnel_resolves_and_forwards_over_ipv4(tmp_path: Path) -> None:
    """A CONNECT tunnel to localhost reaches the IPv4-only upstream server."""
    with ThreadingServer(("127.0.0.1", 0), EchoHandler) as upstream:
        upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        upstream_thread.start()

        port_file = tmp_path / "proxy-port"
        token_file = tmp_path / "proxy-token"
        proxy = subprocess.Popen(
            [
                sys.executable,
                str(PROXY_SCRIPT),
                str(port_file),
                str(token_file),
                str(os.getpid()),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        stderr = ""
        try:
            proxy_port, proxy_token = _wait_for_proxy(port_file, token_file, proxy)
            target_port = upstream.server_address[1]

            with socket.create_connection(
                ("127.0.0.1", proxy_port), timeout=5
            ) as client:
                client.sendall(
                    f"CONNECT localhost:{target_port} HTTP/1.1\r\n"
                    f"Host: localhost:{target_port}\r\n\r\n".encode("ascii")
                )
                assert _recv_headers(client).startswith(b"HTTP/1.1 407")

            credentials = base64.b64encode(
                f"hermes:{proxy_token}".encode("ascii")
            ).decode("ascii")
            with socket.create_connection(
                ("127.0.0.1", proxy_port), timeout=5
            ) as client:
                payload = b"playwright-ipv4-proxy-e2e"
                request = (
                    f"CONNECT localhost:{target_port} HTTP/1.1\r\n"
                    f"Host: localhost:{target_port}\r\n"
                    f"Proxy-Authorization: Basic {credentials}\r\n\r\n".encode("ascii")
                )
                # A robust CONNECT relay preserves tunnel bytes that arrive in
                # the same segment as the request headers.
                client.sendall(request + payload)
                response = _recv_headers(client)
                assert response.startswith(b"HTTP/1.1 200"), response
                _, _, received = response.partition(b"\r\n\r\n")
                while len(received) < len(payload):
                    received += client.recv(len(payload) - len(received))
                assert received == payload
        finally:
            proxy.terminate()
            _, stderr = proxy.communicate(timeout=5)
            upstream.shutdown()
            upstream_thread.join(timeout=5)

    assert "CONNECT localhost:" in stderr
    assert " via 127." in stderr
