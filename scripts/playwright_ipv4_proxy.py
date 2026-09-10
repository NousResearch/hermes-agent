#!/usr/bin/env python3
"""Loopback-only HTTP CONNECT proxy with IPv4-only upstream sockets.

Used by install.sh for one Playwright browser-download retry when the normal
IPv6-capable attempt fails. This process is intentionally short-lived and
exits when its parent installer exits.
"""

from __future__ import annotations

import base64
import os
import secrets
import socket
import socketserver
import sys
import threading
import time
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit


class ConnectHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        client = self.request
        client.settimeout(15)
        header = bytearray()
        while b"\r\n\r\n" not in header and len(header) < 65536:
            try:
                chunk = client.recv(4096)
            except OSError:
                return
            if not chunk:
                return
            header.extend(chunk)

        header_end = header.find(b"\r\n\r\n")
        if header_end < 0:
            client.sendall(
                b"HTTP/1.1 431 Request Header Fields Too Large\r\n"
                b"Connection: close\r\n\r\n"
            )
            return
        tunneled_data = bytes(header[header_end + 4 :])
        header_lines = bytes(header[:header_end]).split(b"\r\n")
        parts = header_lines[0].decode("ascii", "replace").split()
        if len(parts) != 3 or parts[0].upper() != "CONNECT":
            client.sendall(
                b"HTTP/1.1 405 Method Not Allowed\r\nConnection: close\r\n\r\n"
            )
            return

        proxy_authorization = b""
        for line in header_lines[1:]:
            name, separator, value = line.partition(b":")
            if separator and name.strip().lower() == b"proxy-authorization":
                proxy_authorization = value.strip()
                break
        if not secrets.compare_digest(
            proxy_authorization,
            cast("ThreadingServer", self.server).proxy_authorization,
        ):
            client.sendall(
                b"HTTP/1.1 407 Proxy Authentication Required\r\n"
                b'Proxy-Authenticate: Basic realm="hermes-installer"\r\n'
                b"Connection: close\r\n\r\n"
            )
            return

        host = None
        port = 443
        try:
            target = urlsplit("//" + parts[1])
            host, port = target.hostname, target.port or 443
        except ValueError:
            pass
        if not host:
            client.sendall(b"HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n")
            return

        upstream = None
        for family, socktype, proto, _, address in socket.getaddrinfo(
            host, port, socket.AF_INET, socket.SOCK_STREAM
        ):
            candidate = socket.socket(family, socktype, proto)
            candidate.settimeout(15)
            try:
                candidate.connect(address)
            except OSError:
                candidate.close()
                continue
            upstream = candidate
            print(
                f"CONNECT {host}:{port} via {address[0]}",
                file=sys.stderr,
                flush=True,
            )
            break

        if upstream is None:
            client.sendall(b"HTTP/1.1 502 Bad Gateway\r\nConnection: close\r\n\r\n")
            return

        client.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        if tunneled_data:
            upstream.sendall(tunneled_data)
        client.settimeout(None)
        upstream.settimeout(None)

        def pump(source: socket.socket, destination: socket.socket) -> None:
            try:
                while data := source.recv(65536):
                    destination.sendall(data)
            except OSError:
                pass
            finally:
                try:
                    destination.shutdown(socket.SHUT_WR)
                except OSError:
                    pass

        request_thread = threading.Thread(
            target=pump, args=(client, upstream), daemon=True
        )
        request_thread.start()
        pump(upstream, client)
        request_thread.join(timeout=1)
        upstream.close()


class ThreadingServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    proxy_authorization: bytes


def serve(port_file: Path, token_file: Path, parent_pid: int) -> None:
    with ThreadingServer(("127.0.0.1", 0), ConnectHandler) as server:
        token = secrets.token_urlsafe(24)
        credentials = base64.b64encode(f"hermes:{token}".encode("ascii"))
        server.proxy_authorization = b"Basic " + credentials
        port_file.write_text(str(server.server_address[1]), encoding="ascii")
        token_file.write_text(token, encoding="ascii")

        def stop_if_orphaned() -> None:
            while os.getppid() == parent_pid:
                time.sleep(0.5)
            server.shutdown()

        threading.Thread(target=stop_if_orphaned, daemon=True).start()
        server.serve_forever(poll_interval=0.1)


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: playwright_ipv4_proxy.py PORT_FILE TOKEN_FILE PARENT_INSTALLER_PID"
        )
    serve(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]))


if __name__ == "__main__":
    main()
