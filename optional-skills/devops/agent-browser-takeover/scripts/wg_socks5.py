#!/usr/bin/env python3
"""SOCKS5 CONNECT proxy bound to one WireGuard address.

Run this on the *peer* that should originate browser traffic (home NAS,
travel router, second VPS). Bind that host's wg0 IP, never 0.0.0.0.

SSH -D is not a substitute: many peers set AllowTcpForwarding no.
"""
from __future__ import annotations

import os
import select
import socket
import socketserver
import struct
import sys

TIMEOUT = 30


def bind_host() -> str:
    host = (os.environ.get("PEER_WG_IP") or os.environ.get("BIND_HOST") or "").strip()
    if not host or host in {"0.0.0.0", "::", "*", "localhost"}:
        print(
            "Set PEER_WG_IP to this machine's WireGuard address. Never 0.0.0.0.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return host


def bind_port() -> int:
    return int(os.environ.get("SOCKS_PORT", "1080"))


def _relay(a: socket.socket, b: socket.socket) -> None:
    sockets = [a, b]
    try:
        while True:
            r, _, x = select.select(sockets, [], sockets, TIMEOUT)
            if x or not r:
                break
            for src in r:
                dst = b if src is a else a
                data = src.recv(65536)
                if not data:
                    return
                dst.sendall(data)
    except OSError:
        return


class Socks5Handler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        sock = self.request
        sock.settimeout(TIMEOUT)
        try:
            header = sock.recv(2)
            if len(header) < 2 or header[0] != 5:
                return
            nmethods = header[1]
            sock.recv(nmethods)
            sock.sendall(b"\x05\x00")
            req = sock.recv(4)
            if len(req) < 4 or req[0] != 5 or req[1] != 1:
                sock.sendall(b"\x05\x07\x00\x01\x00\x00\x00\x00\x00\x00")
                return
            atyp = req[3]
            if atyp == 1:
                raw = sock.recv(4)
                host = socket.inet_ntoa(raw)
            elif atyp == 3:
                ln = sock.recv(1)[0]
                host = sock.recv(ln).decode("idna")
            elif atyp == 4:
                raw = sock.recv(16)
                host = socket.inet_ntop(socket.AF_INET6, raw)
            else:
                sock.sendall(b"\x05\x08\x00\x01\x00\x00\x00\x00\x00\x00")
                return
            port = struct.unpack("!H", sock.recv(2))[0]
            remote = socket.create_connection((host, port), timeout=TIMEOUT)
            sock.sendall(b"\x05\x00\x00\x01\x00\x00\x00\x00\x00\x00")
            _relay(sock, remote)
        except Exception:
            try:
                sock.sendall(b"\x05\x01\x00\x01\x00\x00\x00\x00\x00\x00")
            except OSError:
                pass
        finally:
            try:
                sock.close()
            except OSError:
                pass


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    host = bind_host()
    port = bind_port()
    srv = Server((host, port), Socks5Handler)
    print(f"socks5 {host}:{port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
