"""Local IPC bridge for Codex MCP clarify prompts.

The normal Hermes runtime calls ``AIAgent.clarify_callback`` directly. In the
Codex app-server runtime, the ``clarify`` tool is invoked inside the
``hermes-tools`` MCP subprocess, so it needs a small parent-process bridge back
to the foreground CLI/gateway/TUI callback.

This module intentionally uses only stdlib sockets and a bearer token carried in
the child environment. The socket is local to the machine, and Unix sockets are
created inside a 0700 temp directory when available.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import shutil
import socket
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

MAX_MESSAGE_BYTES = 1024 * 1024


class ClarifyBridgeServer:
    """Foreground-process server that services MCP clarify requests."""

    def __init__(
        self,
        callback: Optional[Callable[[str, Optional[list[str]]], str]],
        *,
        token: Optional[str] = None,
    ) -> None:
        self._callback = callback
        self.token = token or secrets.token_urlsafe(32)
        self.address = ""
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._closed = threading.Event()
        self._callback_lock = threading.Lock()
        self._tmpdir: Optional[Path] = None

    def start(self) -> "ClarifyBridgeServer":
        if self._sock is not None:
            return self

        if hasattr(socket, "AF_UNIX") and os.name != "nt":
            tmpdir = Path(tempfile.mkdtemp(prefix="hermes-clarify-"))
            try:
                tmpdir.chmod(0o700)
            except Exception:
                pass
            sock_path = tmpdir / "bridge.sock"
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.bind(str(sock_path))
            self._tmpdir = tmpdir
            self.address = f"unix:{sock_path}"
        else:  # pragma: no cover - Windows fallback
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            host, port = sock.getsockname()
            self.address = f"tcp:{host}:{port}"

        sock.listen(16)
        sock.settimeout(0.25)
        self._sock = sock
        self._thread = threading.Thread(
            target=self._serve,
            name="hermes-clarify-bridge",
            daemon=True,
        )
        self._thread.start()
        return self

    def set_callback(
        self,
        callback: Optional[Callable[[str, Optional[list[str]]], str]],
    ) -> None:
        with self._callback_lock:
            self._callback = callback

    def env(self) -> dict[str, str]:
        if not self.address:
            raise RuntimeError("clarify bridge has not been started")
        return {
            "HERMES_CLARIFY_BRIDGE_ADDR": self.address,
            "HERMES_CLARIFY_BRIDGE_TOKEN": self.token,
        }

    def close(self) -> None:
        self._closed.set()
        sock = self._sock
        self._sock = None
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        if self._tmpdir is not None:
            try:
                shutil.rmtree(self._tmpdir, ignore_errors=True)
            except Exception:
                pass
            self._tmpdir = None

    def _serve(self) -> None:
        while not self._closed.is_set():
            sock = self._sock
            if sock is None:
                return
            try:
                conn, _addr = sock.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            thread = threading.Thread(
                target=self._handle_connection,
                args=(conn,),
                name="hermes-clarify-request",
                daemon=True,
            )
            thread.start()

    def _handle_connection(self, conn: socket.socket) -> None:
        with conn:
            try:
                request = _recv_json_line(conn)
                response = self._handle_request(request)
            except Exception as exc:
                logger.debug("clarify bridge request failed", exc_info=True)
                response = {"ok": False, "error": str(exc)}
            _send_json_line(conn, response)

    def _handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        if request.get("token") != self.token:
            return {"ok": False, "error": "invalid clarify bridge token"}

        question = str(request.get("question") or "").strip()
        raw_choices = request.get("choices")
        choices: Optional[list[str]]
        if isinstance(raw_choices, list):
            choices = [str(choice) for choice in raw_choices]
        else:
            choices = None

        with self._callback_lock:
            callback = self._callback
        if callback is None:
            return {"ok": False, "error": "clarify callback is not available"}

        answer = callback(question, choices)
        return {"ok": True, "answer": str(answer)}


def request_clarify_via_bridge(
    *,
    address: str,
    token: str,
    question: str,
    choices: Optional[list[str]] = None,
    timeout: float = 900.0,
) -> str:
    """Ask the foreground Hermes process to run its clarify callback."""
    if not address or not token:
        raise RuntimeError("clarify bridge address/token is missing")

    conn = _connect(address, timeout=timeout)
    with conn:
        conn.settimeout(timeout)
        _send_json_line(
            conn,
            {
                "token": token,
                "question": question,
                "choices": list(choices) if choices else None,
            },
        )
        response = _recv_json_line(conn)

    if not response.get("ok"):
        raise RuntimeError(str(response.get("error") or "clarify bridge failed"))
    return str(response.get("answer") or "")


def _connect(address: str, *, timeout: float) -> socket.socket:
    if address.startswith("unix:"):
        path = address[5:]
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect(path)
        return sock
    if address.startswith("tcp:"):
        _prefix, host, port_s = address.split(":", 2)
        return socket.create_connection((host, int(port_s)), timeout=timeout)
    raise RuntimeError(f"unsupported clarify bridge address: {address!r}")


def _recv_json_line(conn: socket.socket) -> dict[str, Any]:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = conn.recv(4096)
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > MAX_MESSAGE_BYTES:
            raise RuntimeError("clarify bridge message too large")
        if b"\n" in chunk:
            break
    raw = b"".join(chunks).split(b"\n", 1)[0]
    if not raw:
        raise RuntimeError("empty clarify bridge message")
    parsed = json.loads(raw.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise RuntimeError("clarify bridge message must be an object")
    return parsed


def _send_json_line(conn: socket.socket, payload: dict[str, Any]) -> None:
    conn.sendall(json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n")
