"""Owner-local IPC for injecting work into an exact gateway session.

The socket is scoped to the active ``HERMES_HOME``.  It is intentionally a
Unix-domain socket rather than an HTTP endpoint: session injection is an
operator-local control-plane operation and must never be network reachable.

This boundary prevents accidental cross-profile and cross-session delivery by
requiring the target profile socket plus exact routing key and session id.  It
is owner-local, not hostile same-UID isolation: another process running as the
same OS user can access the profile's owner-only runtime directory.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import socket
import stat
import struct
from collections import OrderedDict
from pathlib import Path
from typing import Any, Awaitable, Callable

from hermes_constants import get_hermes_home

_MAX_REQUEST_BYTES = 64 * 1024
_SOCKET_NAME = "gateway-session.sock"
_MAX_IDEMPOTENCY_KEY_BYTES = 256
_MAX_IDEMPOTENCY_RECORDS = 1024
_HANDLER_TIMEOUT_SECONDS = 5.0


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
    expected_session_id: str,
    idempotency_key: str,
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
            "expected_session_id": expected_session_id,
            "idempotency_key": idempotency_key,
            "message": message,
        },
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    if len(request) > _MAX_REQUEST_BYTES:
        raise SessionIPCRequestError("request_too_large", "Gateway injection request is too large")

    phase = "connect"
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(timeout)
            client.connect(str(path))
            phase = "send"
            client.sendall(request)
            phase = "receive"
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = client.recv(min(8192, _MAX_REQUEST_BYTES + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > _MAX_REQUEST_BYTES:
                    raise SessionIPCRequestError(
                        "invalid_response",
                        "Gateway IPC response is too large",
                    )
                if b"\n" in chunk:
                    break
    except socket.timeout as exc:
        raise SessionIPCRequestError(
            "request_timeout",
            f"Gateway IPC {phase} timed out",
        ) from exc
    except OSError as exc:
        code = "gateway_unavailable" if phase == "connect" else "transport_error"
        raise SessionIPCRequestError(
            code,
            f"Gateway IPC {phase} failed: {exc}",
        ) from exc
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
        self._bound_identity: tuple[int, int] | None = None
        self._idempotency_lock = asyncio.Lock()
        self._idempotency_tasks: OrderedDict[str, asyncio.Task[dict[str, Any]]] = OrderedDict()
        self._idempotency_results: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._idempotency_fingerprints: dict[str, str] = {}

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
        try:
            current = self.socket_path.lstat()
        except FileNotFoundError:
            return
        if (current.st_dev, current.st_ino) != (existing.st_dev, existing.st_ino):
            raise SessionIPCRequestError(
                "socket_replaced",
                f"Gateway IPC socket changed during stale-socket inspection: {self.socket_path}",
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
        bound = self.socket_path.lstat()
        if not stat.S_ISSOCK(bound.st_mode):
            self._server.close()
            await self._server.wait_closed()
            self._server = None
            raise SessionIPCRequestError("unsafe_socket", "Gateway IPC bind did not create a socket")
        self._bound_identity = (bound.st_dev, bound.st_ino)

    async def stop(self) -> None:
        server = self._server
        self._server = None
        if server is not None:
            server.close()
            await server.wait_closed()
        pending = [task for task in self._idempotency_tasks.values() if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        try:
            info = self.socket_path.lstat()
        except FileNotFoundError:
            self._bound_identity = None
            return
        if (info.st_dev, info.st_ino) == self._bound_identity and stat.S_ISSOCK(info.st_mode) and (
            not hasattr(os, "geteuid") or info.st_uid == os.geteuid()
        ):
            self.socket_path.unlink()
        self._bound_identity = None

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

    @staticmethod
    def _request_fingerprint(request: dict[str, Any]) -> str:
        canonical = json.dumps(request, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    async def _run_handler(self, request: dict[str, str]) -> dict[str, Any]:
        try:
            return await asyncio.wait_for(
                self._handler(**request),
                timeout=_HANDLER_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            return SessionIPCRequestError(
                "request_timeout",
                "Gateway session acceptance timed out and was cancelled",
            ).to_response()
        except SessionIPCRequestError as exc:
            return exc.to_response()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return SessionIPCRequestError(
                "internal_error",
                str(exc) or type(exc).__name__,
            ).to_response()

    async def _finalize_idempotency_task(
        self,
        idempotency_key: str,
        task: asyncio.Task[dict[str, Any]],
    ) -> None:
        """Move a completed task out of the pending map without a new request."""
        async with self._idempotency_lock:
            if self._idempotency_tasks.get(idempotency_key) is not task:
                return
            self._idempotency_tasks.pop(idempotency_key, None)
            if task.cancelled():
                self._idempotency_fingerprints.pop(idempotency_key, None)
                return
            try:
                result = task.result()
            except Exception:
                self._idempotency_fingerprints.pop(idempotency_key, None)
                return
            self._idempotency_results[idempotency_key] = result
            self._idempotency_results.move_to_end(idempotency_key)

            while (
                len(self._idempotency_tasks) + len(self._idempotency_results)
                > _MAX_IDEMPOTENCY_RECORDS
            ):
                oldest_key, _ = self._idempotency_results.popitem(last=False)
                self._idempotency_fingerprints.pop(oldest_key, None)

    async def _dispatch_idempotent(
        self,
        idempotency_key: str,
        request: dict[str, str],
    ) -> dict[str, Any]:
        fingerprint = self._request_fingerprint(request)
        async with self._idempotency_lock:
            prior_fingerprint = self._idempotency_fingerprints.get(idempotency_key)
            if prior_fingerprint is not None and prior_fingerprint != fingerprint:
                raise SessionIPCRequestError(
                    "idempotency_conflict",
                    "Idempotency key was already used for a different request",
                )

            prior_result = self._idempotency_results.get(idempotency_key)
            if prior_result is not None:
                self._idempotency_results.move_to_end(idempotency_key)
                return prior_result

            task = self._idempotency_tasks.get(idempotency_key)
            if task is None:
                while (
                    len(self._idempotency_tasks) + len(self._idempotency_results)
                    >= _MAX_IDEMPOTENCY_RECORDS
                    and self._idempotency_results
                ):
                    oldest_key, _ = self._idempotency_results.popitem(last=False)
                    self._idempotency_fingerprints.pop(oldest_key, None)
                if (
                    len(self._idempotency_tasks) + len(self._idempotency_results)
                    >= _MAX_IDEMPOTENCY_RECORDS
                ):
                    raise SessionIPCRequestError(
                        "server_busy",
                        "Gateway session idempotency capacity is full",
                    )
                task = asyncio.create_task(self._run_handler(request))
                self._idempotency_tasks[idempotency_key] = task
                self._idempotency_fingerprints[idempotency_key] = fingerprint
                task.add_done_callback(
                    lambda completed, key=idempotency_key: asyncio.create_task(
                        self._finalize_idempotency_task(key, completed)
                    )
                )
            else:
                self._idempotency_tasks.move_to_end(idempotency_key)

        # Shield the shared task: a client disconnect/cancel must not cancel an
        # acceptance that a retry will join by idempotency key.
        return await asyncio.shield(task)

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        response: dict[str, Any]
        try:
            peer_uid = self._peer_uid(writer)
            if peer_uid is None or not hasattr(os, "geteuid"):
                raise SessionIPCRequestError(
                    "peer_credentials_unavailable",
                    "Gateway IPC peer credentials could not be established",
                )
            if peer_uid != os.geteuid():
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
            expected_session_id = request.get("expected_session_id")
            idempotency_key = request.get("idempotency_key")
            message = request.get("message")
            if profile != self.profile:
                raise SessionIPCRequestError(
                    "profile_mismatch",
                    f"Socket serves profile {self.profile!r}, not {profile!r}",
                )
            if not isinstance(session_key, str) or not session_key.strip():
                raise SessionIPCRequestError("invalid_request", "session_key is required")
            if not isinstance(expected_session_id, str) or not expected_session_id.strip():
                raise SessionIPCRequestError("invalid_request", "expected_session_id is required")
            if not isinstance(idempotency_key, str) or not idempotency_key.strip():
                raise SessionIPCRequestError("invalid_request", "idempotency_key is required")
            if len(idempotency_key.encode("utf-8")) > _MAX_IDEMPOTENCY_KEY_BYTES:
                raise SessionIPCRequestError("invalid_request", "idempotency_key is too large")
            if not isinstance(message, str) or not message.strip():
                raise SessionIPCRequestError("invalid_request", "message is required")
            response = await self._dispatch_idempotent(
                idempotency_key.strip(),
                {
                    "profile": profile,
                    "session_key": session_key.strip(),
                    "expected_session_id": expected_session_id.strip(),
                    "idempotency_key": idempotency_key.strip(),
                    "message": message.strip(),
                },
            )
        except SessionIPCRequestError as exc:
            response = exc.to_response()
        except asyncio.CancelledError:
            response = SessionIPCRequestError("request_cancelled", "Gateway IPC request cancelled").to_response()
        except Exception as exc:
            response = SessionIPCRequestError("internal_error", str(exc) or type(exc).__name__).to_response()

        try:
            writer.write(json.dumps(response, separators=(",", ":")).encode("utf-8") + b"\n")
            await writer.drain()
        except (BrokenPipeError, ConnectionError, OSError):
            # The request result remains in the idempotency cache for a retry.
            pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass
