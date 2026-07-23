"""Owner-only local IPC for injecting work into an exact gateway session.

The socket is scoped to the active ``HERMES_HOME``.  It is intentionally a
Unix-domain socket rather than an HTTP endpoint: session injection is an
operator-local control-plane operation and must never be network reachable.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import stat
import struct
from pathlib import Path
from typing import Any, Awaitable, Callable

from hermes_constants import get_hermes_home

_MAX_REQUEST_BYTES = 64 * 1024
_SOCKET_NAME = "gateway-session.sock"


class SessionIPCRequestError(RuntimeError):
    """A fail-closed session IPC request error with a stable machine code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code

    def to_response(self) -> dict[str, Any]:
        return {"ok": False, "error": {"code": self.code, "message": str(self)}}


def gateway_session_socket_path(hermes_home: Path | str | None = None) -> Path:
    home = Path(hermes_home) if hermes_home is not None else get_hermes_home()
    return home.resolve() / "run" / _SOCKET_NAME


def _validate_owner_only_socket(path: Path) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise SessionIPCRequestError(
            "gateway_unavailable",
            f"No live gateway session socket for this profile: {path}",
        ) from exc
    if not stat.S_ISSOCK(info.st_mode):
        raise SessionIPCRequestError(
            "unsafe_socket",
            f"Refusing non-socket gateway IPC path: {path}",
        )
    if hasattr(os, "geteuid") and info.st_uid != os.geteuid():
        raise SessionIPCRequestError(
            "unsafe_socket",
            f"Gateway IPC socket is not owned by the current user: {path}",
        )
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise SessionIPCRequestError(
            "unsafe_socket",
            f"Gateway IPC socket must be owner-only (mode 0600): {path}",
        )


def inject_gateway_session(
    *,
    profile: str,
    session_key: str,
    message: str,
    hermes_home: Path | str | None = None,
    timeout: float = 5.0,
) -> dict[str, Any]:
    """Send one exact-session injection request to the profile-local gateway."""
    path = gateway_session_socket_path(hermes_home)
    _validate_owner_only_socket(path)
    request = json.dumps(
        {
            "operation": "inject",
            "profile": profile,
            "session_key": session_key,
            "message": message,
        },
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    if len(request) > _MAX_REQUEST_BYTES:
        raise SessionIPCRequestError("request_too_large", "Gateway injection request is too large")

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(timeout)
        client.connect(str(path))
        client.sendall(request)
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = client.recv(min(8192, _MAX_REQUEST_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > _MAX_REQUEST_BYTES:
                raise SessionIPCRequestError("invalid_response", "Gateway IPC response is too large")
            if b"\n" in chunk:
                break
    raw = b"".join(chunks).split(b"\n", 1)[0]
    try:
        response = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SessionIPCRequestError("invalid_response", "Gateway returned invalid JSON") from exc
    if not isinstance(response, dict):
        raise SessionIPCRequestError("invalid_response", "Gateway returned a non-object response")
    return response


class GatewaySessionIPCServer:
    """Single-request NDJSON server bound to one profile's Hermes home."""

    def __init__(
        self,
        handler: Callable[..., Awaitable[dict[str, Any]]],
        *,
        profile: str,
        hermes_home: Path | str | None = None,
    ) -> None:
        self._handler = handler
        self.profile = str(profile or "default")
        self.socket_path = gateway_session_socket_path(hermes_home)
        self._server: asyncio.AbstractServer | None = None

    def _prepare_socket_directory(self) -> None:
        runtime_dir = self.socket_path.parent
        runtime_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
        info = runtime_dir.lstat()
        if not stat.S_ISDIR(info.st_mode) or runtime_dir.is_symlink():
            raise SessionIPCRequestError("unsafe_socket", f"Unsafe gateway IPC directory: {runtime_dir}")
        if hasattr(os, "geteuid") and info.st_uid != os.geteuid():
            raise SessionIPCRequestError(
                "unsafe_socket", f"Gateway IPC directory is not owned by the current user: {runtime_dir}"
            )
        os.chmod(runtime_dir, 0o700)

        try:
            existing = self.socket_path.lstat()
        except FileNotFoundError:
            return
        if not stat.S_ISSOCK(existing.st_mode):
            raise SessionIPCRequestError(
                "unsafe_socket", f"Refusing to replace non-socket IPC path: {self.socket_path}"
            )
        if hasattr(os, "geteuid") and existing.st_uid != os.geteuid():
            raise SessionIPCRequestError(
                "unsafe_socket", f"Refusing socket owned by another user: {self.socket_path}"
            )
        # Never unlink a live listener.  That would let a second gateway bind
        # the same pathname while the first keeps serving through its orphaned
        # inode, defeating the one-gateway/profile invariant.  Only a refused
        # connect proves this is crash-stale and safe to remove.
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.25)
            try:
                probe.connect(str(self.socket_path))
            except (ConnectionRefusedError, FileNotFoundError):
                pass
            except OSError as exc:
                raise SessionIPCRequestError(
                    "gateway_unavailable",
                    f"Cannot safely inspect existing gateway IPC socket: {exc}",
                ) from exc
            else:
                raise SessionIPCRequestError(
                    "gateway_already_running",
                    f"A live gateway already owns the profile IPC socket: {self.socket_path}",
                )
        self.socket_path.unlink()

    async def start(self) -> None:
        if not hasattr(socket, "AF_UNIX"):
            raise SessionIPCRequestError("unsupported", "Gateway session IPC requires Unix sockets")
        if self._server is not None:
            return
        self._prepare_socket_directory()
        self._server = await asyncio.start_unix_server(
            self._handle_connection,
            path=str(self.socket_path),
            limit=_MAX_REQUEST_BYTES,
        )
        os.chmod(self.socket_path, 0o600)

    async def stop(self) -> None:
        server = self._server
        self._server = None
        if server is not None:
            server.close()
            await server.wait_closed()
        try:
            info = self.socket_path.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISSOCK(info.st_mode) and (
            not hasattr(os, "geteuid") or info.st_uid == os.geteuid()
        ):
            self.socket_path.unlink()

    @staticmethod
    def _peer_uid(writer: asyncio.StreamWriter) -> int | None:
        transport_socket = writer.get_extra_info("socket")
        if transport_socket is None:
            return None
        if hasattr(socket, "SO_PEERCRED"):
            try:
                raw = transport_socket.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
                _pid, uid, _gid = struct.unpack("3i", raw)
                return uid
            except (OSError, struct.error):
                return None
        getpeereid = getattr(transport_socket, "getpeereid", None)
        if callable(getpeereid):
            try:
                peer_ids = getpeereid()
                if not isinstance(peer_ids, tuple) or len(peer_ids) != 2:
                    return None
                uid, _gid = peer_ids
                return int(uid)
            except OSError:
                return None
        return None

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        response: dict[str, Any]
        try:
            peer_uid = self._peer_uid(writer)
            if peer_uid is not None and hasattr(os, "geteuid") and peer_uid != os.geteuid():
                raise SessionIPCRequestError("permission_denied", "Gateway IPC peer user mismatch")
            try:
                raw = await asyncio.wait_for(reader.readline(), timeout=5.0)
            except (asyncio.LimitOverrunError, ValueError) as exc:
                raise SessionIPCRequestError("request_too_large", "Gateway IPC request is too large") from exc
            if not raw or len(raw) > _MAX_REQUEST_BYTES or not raw.endswith(b"\n"):
                raise SessionIPCRequestError("invalid_request", "Expected one bounded JSON request line")
            try:
                request = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise SessionIPCRequestError("invalid_request", "Request must be valid UTF-8 JSON") from exc
            if not isinstance(request, dict) or request.get("operation") != "inject":
                raise SessionIPCRequestError("invalid_request", "Unsupported gateway IPC operation")
            profile = request.get("profile")
            session_key = request.get("session_key")
            message = request.get("message")
            if profile != self.profile:
                raise SessionIPCRequestError(
                    "profile_mismatch",
                    f"Socket serves profile {self.profile!r}, not {profile!r}",
                )
            if not isinstance(session_key, str) or not session_key.strip():
                raise SessionIPCRequestError("invalid_request", "session_key is required")
            if not isinstance(message, str) or not message.strip():
                raise SessionIPCRequestError("invalid_request", "message is required")
            response = await self._handler(
                profile=profile,
                session_key=session_key.strip(),
                message=message.strip(),
            )
        except SessionIPCRequestError as exc:
            response = exc.to_response()
        except (asyncio.TimeoutError, Exception) as exc:
            response = SessionIPCRequestError("internal_error", str(exc) or type(exc).__name__).to_response()

        writer.write(json.dumps(response, separators=(",", ":")).encode("utf-8") + b"\n")
        try:
            await writer.drain()
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass
