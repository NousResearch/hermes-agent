"""Fail-closed local Unix-domain health endpoint for the control plane."""

from __future__ import annotations

import json
import os
import socket
import socketserver
import stat
import threading
from pathlib import Path
from typing import Any, Callable


class SocketPathInUseError(RuntimeError):
    """A live server owns the configured control socket."""


class SocketSecurityError(RuntimeError):
    """Socket or its parent violates the strict local ownership boundary."""


class _ThreadingUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = False


class _ControlAPIHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        raw_request = self.rfile.readline(8192)
        try:
            method, path, _protocol = raw_request.decode("ascii").strip().split(" ", 2)
        except ValueError:
            self._respond(400, {"error": "bad_request"})
            return
        if method != "GET":
            self._respond(405, {"error": "method_not_allowed"})
            return
        if path != "/v1/health":
            self._respond(404, {"error": "not_found"})
            return
        try:
            body = self.server.health_provider()  # type: ignore[attr-defined]
        except Exception:
            body = {
                "ready": False,
                "authority_mode": "observe_only",
                "safe_start_reasons": ["health_unavailable"],
                "store_available": False,
                "audit_chain_valid": None,
                "event_count": 0,
                "spool_depth": 0,
                "spool_bytes": 0,
                "spool_quarantine_bytes": 0,
                "spool_healthy": False,
                "global_write_enabled": False,
            }
        self._respond(200, body)

    def _respond(self, status: int, body: dict[str, Any]) -> None:
        encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        reason = {200: "OK", 400: "Bad Request", 404: "Not Found", 405: "Method Not Allowed"}[status]
        headers = (
            f"HTTP/1.1 {status} {reason}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(encoded)}\r\n"
            "Connection: close\r\n\r\n"
        ).encode("ascii")
        self.wfile.write(headers + encoded)


class ControlAPI:
    """Expose exactly one GET endpoint after UDS security checks pass."""

    def __init__(
        self,
        socket_path: Path,
        state_dir: Path,
        health_provider: Callable[[], dict[str, Any]],
        *,
        allow_stale_reclaim: bool,
    ):
        self.socket_path = Path(socket_path)
        self.state_dir = Path(state_dir)
        self.health_provider = health_provider
        self.allow_stale_reclaim = allow_stale_reclaim
        self._server: _ThreadingUnixServer | None = None
        self._thread: threading.Thread | None = None
        self._socket_inode: int | None = None

    def _validate_parent(self) -> None:
        try:
            state_status = self.state_dir.lstat()
            parent_status = self.socket_path.parent.lstat()
        except OSError as exc:
            raise SocketSecurityError("socket parent unavailable") from exc
        if (
            not stat.S_ISDIR(state_status.st_mode)
            or stat.S_ISLNK(state_status.st_mode)
            or not stat.S_ISDIR(parent_status.st_mode)
            or stat.S_ISLNK(parent_status.st_mode)
            or state_status.st_uid != os.getuid()
            or parent_status.st_uid != os.getuid()
            or stat.S_IMODE(state_status.st_mode) != 0o700
            or stat.S_IMODE(parent_status.st_mode) != 0o700
            or self.socket_path.parent != self.state_dir
        ):
            raise SocketSecurityError("socket parent rejected")

    def _reclaim_stale_socket(self) -> None:
        if not os.path.lexists(self.socket_path):
            return
        status = self.socket_path.lstat()
        if not stat.S_ISSOCK(status.st_mode) or stat.S_ISLNK(status.st_mode) or status.st_uid != os.getuid():
            raise SocketSecurityError("socket path rejected")
        try:
            request_health(self.socket_path)
        except (OSError, RuntimeError, ValueError):
            if not self.allow_stale_reclaim:
                raise SocketPathInUseError("stale socket requires daemon lock")
            self.socket_path.unlink()
            _fsync_directory(self.socket_path.parent)
            return
        raise SocketPathInUseError("control socket already serves health")

    def start(self) -> None:
        self._validate_parent()
        self._reclaim_stale_socket()
        try:
            self._server = _ThreadingUnixServer(str(self.socket_path), _ControlAPIHandler)
            self._server.health_provider = self.health_provider  # type: ignore[attr-defined]
            self._socket_inode = self.socket_path.lstat().st_ino
            os.chmod(self.socket_path, 0o600)
            status = self.socket_path.lstat()
            if (
                not stat.S_ISSOCK(status.st_mode)
                or stat.S_ISLNK(status.st_mode)
                or status.st_uid != os.getuid()
                or stat.S_IMODE(status.st_mode) != 0o600
            ):
                raise SocketSecurityError("socket permissions invalid")
        except Exception:
            self._discard_owned_socket()
            if self._server is not None:
                self._server.server_close()
            self._server = None
            raise
        self._thread = threading.Thread(target=self._server.serve_forever, name="agentops-uds", daemon=True)
        self._thread.start()

    def _discard_owned_socket(self) -> None:
        try:
            if self._socket_inode is not None and os.path.lexists(self.socket_path):
                status = self.socket_path.lstat()
                if stat.S_ISSOCK(status.st_mode) and status.st_ino == self._socket_inode:
                    self.socket_path.unlink()
                    _fsync_directory(self.socket_path.parent)
        except OSError:
            pass

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._discard_owned_socket()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def request_control_api(socket_path: Path, method: str, path: str) -> tuple[int, dict[str, Any]]:
    """Small test/operator client; it sends no request body or credentials."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(2)
        client.connect(str(socket_path))
        client.sendall(f"{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n".encode("ascii"))
        chunks: list[bytes] = []
        while True:
            chunk = client.recv(8192)
            if not chunk:
                break
            chunks.append(chunk)
    raw = b"".join(chunks)
    headers, body = raw.split(b"\r\n\r\n", 1)
    status = int(headers.splitlines()[0].split()[1])
    return status, json.loads(body.decode("utf-8"))


def request_health(socket_path: Path) -> dict[str, Any]:
    status, body = request_control_api(socket_path, "GET", "/v1/health")
    if status != 200:
        raise RuntimeError("health endpoint unavailable")
    return body
