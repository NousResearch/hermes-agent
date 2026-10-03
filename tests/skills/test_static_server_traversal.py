"""The optional-skill static file servers must refuse path traversal.

`gitnexus-explorer/scripts/proxy.mjs` joined the raw request path onto the dist
dir with `path.join` (which collapses `..`) and had no confinement check;
`scrollcraft/scripts/serve.mjs` compared with a bare `startsWith(ROOT)`, so a
sibling directory sharing the root's name prefix passed. Both are exercised here
over a raw socket (the Node HTTP client would normalise `..` away).
"""
from __future__ import annotations

import socket
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROXY = REPO / "optional-skills/research/gitnexus-explorer/scripts/proxy.mjs"
SERVE = REPO / "optional-skills/web-development/scrollcraft/scripts/serve.mjs"


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _start(argv: list[str], cwd: Path) -> subprocess.Popen:
    proc = subprocess.Popen(
        ["node", *argv], cwd=str(cwd),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    deadline = time.time() + 20
    while time.time() < deadline:
        line = proc.stdout.readline() if proc.stdout else ""
        if "http" in line:
            return proc
    proc.terminate()
    raise AssertionError("server did not report listening")


def _raw_get(port: int, target: str) -> str:
    with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
        sock.sendall(
            f"GET {target} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n".encode()
        )
        data = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            data += chunk
    return data.decode("utf-8", "replace")


def test_gitnexus_proxy_refuses_path_traversal(tmp_path):
    root = tmp_path / "dist"
    root.mkdir()
    (root / "index.html").write_text("INDEX", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("TOPSECRET", encoding="utf-8")

    port = _free_port()
    proc = _start([str(PROXY), str(root), str(port)], tmp_path)
    try:
        resp = _raw_get(port, "/../secret.txt")
    finally:
        proc.terminate()
        proc.wait(timeout=10)

    assert "TOPSECRET" not in resp, resp
    assert "403" in resp, resp


def test_scrollcraft_serve_refuses_sibling_prefix_escape(tmp_path):
    root = tmp_path / "app"
    root.mkdir()
    (root / "index.html").write_text("INDEX", encoding="utf-8")
    sibling = tmp_path / "app-secret"
    sibling.mkdir()
    (sibling / "secret.txt").write_text("TOPSECRET", encoding="utf-8")

    port = _free_port()
    proc = _start([str(SERVE), "--root", str(root), "--port", str(port)], tmp_path)
    try:
        resp = _raw_get(port, "/../app-secret/secret.txt")
    finally:
        proc.terminate()
        proc.wait(timeout=10)

    assert "TOPSECRET" not in resp, resp
    assert "403" in resp, resp
