"""Transport-only Unix service wrapper for :mod:`gateway.discord_edge_runtime`.

The service owns no protocol decisions and exposes no generic dispatcher.  It
accepts one bounded length-prefixed JSON frame, applies one of two fixed
Discord edge schemas, revalidates the connected peer's exact systemd MainPID,
and passes only a parsed mutation request or a mutation-free reconciliation
query to the injected runtime.

Invalid or unauthorized input closes the connection without fabricating an
unsigned receipt.  A valid response has exactly four fields: ``state``,
``blocker``, ``replayed``, and the runtime's signed ``receipt``.
"""

from __future__ import annotations

import os
import re
import socket
import stat
import struct
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol

from gateway.discord_edge_protocol import (
    MAX_REQUEST_BYTES,
    RECONCILIATION_NOT_AVAILABLE_ERROR,
    RECONCILIATION_PROTOCOL_VERSION,
    RECONCILIATION_RESPONSE_VERSION,
    DiscordEdgeErrorCode,
    DiscordEdgeProtocolError,
    canonical_json_bytes,
    decode_request_json,
    parse_request,
    parse_request_for_reconciliation,
    parse_reconciliation_query,
)
from gateway.discord_edge_runtime import (
    DiscordEdgeExecutionResult,
    DiscordEdgeReconciliationResult,
    DiscordEdgeRuntime,
    DiscordEdgeRuntimeError,
    DiscordEdgeRuntimeErrorCode,
)

MAX_RESPONSE_BYTES = 128 * 1024
SOCKET_MODE = 0o660
_FRAME_HEADER = struct.Struct("!I")
_PEER_CREDENTIALS = struct.Struct("3i")
_SYSTEMD_UNIT_RE = re.compile(r"^[A-Za-z0-9_.@:-]+\.service$")


class _FrameError(RuntimeError):
    pass


@dataclass(frozen=True)
class DiscordEdgePeerCredentials:
    pid: int
    uid: int
    gid: int


class DiscordEdgeMainPidProvider(Protocol):
    def main_pid(self, unit_name: str) -> int | None: ...


class SystemctlDiscordEdgeMainPidProvider:
    """Read one current systemd MainPID through a bounded absolute command."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 2.0,
        systemctl_path: str | os.PathLike[str] = "/usr/bin/systemctl",
    ) -> None:
        if not 0 < timeout_seconds <= 10:
            raise ValueError("systemctl timeout must be between 0 and 10 seconds")
        path = Path(systemctl_path)
        if not path.is_absolute():
            raise ValueError("systemctl path must be absolute")
        self.timeout_seconds = timeout_seconds
        self.systemctl_path = str(path)

    def main_pid(self, unit_name: str) -> int | None:
        if not _SYSTEMD_UNIT_RE.fullmatch(unit_name):
            return None
        try:
            completed = subprocess.run(
                [
                    self.systemctl_path,
                    "show",
                    "--property=MainPID",
                    "--value",
                    "--",
                    unit_name,
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if completed.returncode != 0:
            return None
        try:
            pid = int(completed.stdout.strip())
        except ValueError:
            return None
        return pid if pid > 0 else None


def linux_discord_edge_peer_credentials(
    conn: socket.socket,
) -> DiscordEdgePeerCredentials:
    """Read Linux ``SO_PEERCRED`` from one connected Unix socket."""

    so_peercred = getattr(socket, "SO_PEERCRED", None)
    if so_peercred is None:
        raise OSError("SO_PEERCRED is unavailable")
    raw = conn.getsockopt(socket.SOL_SOCKET, so_peercred, _PEER_CREDENTIALS.size)
    if len(raw) != _PEER_CREDENTIALS.size:
        raise OSError("SO_PEERCRED returned an invalid value")
    return DiscordEdgePeerCredentials(*_PEER_CREDENTIALS.unpack(raw))


PeerCredentialsGetter = Callable[[socket.socket], DiscordEdgePeerCredentials]


def _receive_exact(conn: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = conn.recv(remaining)
        if not chunk:
            raise _FrameError("connection closed during a frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _receive_request_body(conn: socket.socket) -> bytes:
    header = _receive_exact(conn, _FRAME_HEADER.size)
    (size,) = _FRAME_HEADER.unpack(header)
    if size == 0 or size > MAX_REQUEST_BYTES:
        raise _FrameError("request frame size is invalid")
    return _receive_exact(conn, size)


def _send_response(
    conn: socket.socket,
    response: Mapping[str, object],
) -> None:
    body = canonical_json_bytes(response)
    if not body or len(body) > MAX_RESPONSE_BYTES:
        raise _FrameError("response frame size is invalid")
    conn.sendall(_FRAME_HEADER.pack(len(body)) + body)


def _response_message(result: DiscordEdgeExecutionResult) -> dict[str, object]:
    if not isinstance(result, DiscordEdgeExecutionResult) or result.receipt is None:
        raise _FrameError("runtime did not return a signed fixed response")
    if not isinstance(result.replayed, bool):
        raise _FrameError("runtime replay marker is invalid")
    return {
        "state": result.state.value,
        "blocker": (
            result.blocker_code.value if result.blocker_code is not None else None
        ),
        "replayed": result.replayed,
        "receipt": result.receipt.to_message(),
    }


def _reconciliation_response_message(
    result: DiscordEdgeReconciliationResult,
) -> dict[str, object]:
    if not isinstance(result, DiscordEdgeReconciliationResult):
        raise _FrameError("runtime did not return a fixed reconciliation response")
    response = _response_message(result.execution)
    if response["replayed"] is not True:
        raise _FrameError("reconciliation response must be an exact replay")
    return {
        "protocol": RECONCILIATION_RESPONSE_VERSION,
        "request": result.request.to_message(),
        **response,
    }


def _reconciliation_not_available_message() -> dict[str, object]:
    return {
        "protocol": RECONCILIATION_RESPONSE_VERSION,
        "error": RECONCILIATION_NOT_AVAILABLE_ERROR,
    }


class DiscordEdgeUnixServer:
    """Threaded fixed-frame AF_UNIX boundary for one injected edge runtime."""

    def __init__(
        self,
        socket_path: str | os.PathLike[str],
        *,
        runtime: DiscordEdgeRuntime,
        expected_client_uid: int,
        gateway_unit: str,
        main_pid_provider: DiscordEdgeMainPidProvider,
        peer_credentials_getter: PeerCredentialsGetter = (
            linux_discord_edge_peer_credentials
        ),
        connection_timeout_seconds: float = 30.0,
        max_connections: int = 8,
    ) -> None:
        if not isinstance(runtime, DiscordEdgeRuntime):
            raise TypeError("runtime must be DiscordEdgeRuntime")
        if (
            isinstance(expected_client_uid, bool)
            or not isinstance(expected_client_uid, int)
            or expected_client_uid < 0
        ):
            raise ValueError("expected_client_uid is invalid")
        if not _SYSTEMD_UNIT_RE.fullmatch(gateway_unit):
            raise ValueError("gateway_unit must be one exact systemd service unit")
        if not callable(getattr(main_pid_provider, "main_pid", None)):
            raise TypeError("main_pid_provider is required")
        if not callable(peer_credentials_getter):
            raise TypeError("peer_credentials_getter is required")
        if not 0 < connection_timeout_seconds <= 300:
            raise ValueError("connection timeout must be between 0 and 300 seconds")
        if not 1 <= max_connections <= 64:
            raise ValueError("max_connections must be between 1 and 64")

        self.socket_path, self._parent_identity = self._validate_socket_parent(
            socket_path
        )
        self.runtime = runtime
        self.expected_client_uid = expected_client_uid
        self.gateway_unit = gateway_unit
        self.main_pid_provider = main_pid_provider
        self.peer_credentials_getter = peer_credentials_getter
        self.connection_timeout_seconds = connection_timeout_seconds
        self._listener: socket.socket | None = None
        self._listener_identity: tuple[int, int] | None = None
        self._stop_event = threading.Event()
        self._connection_slots = threading.BoundedSemaphore(max_connections)
        self._workers: set[threading.Thread] = set()
        self._connections: set[socket.socket] = set()
        self._state_lock = threading.Lock()

    @staticmethod
    def _validate_socket_parent(
        value: str | os.PathLike[str],
    ) -> tuple[Path, tuple[int, int]]:
        raw_path = Path(value)
        if not raw_path.is_absolute():
            raise ValueError("Discord edge socket path must be absolute")
        path = Path(os.path.normpath(os.fspath(raw_path)))
        if path != raw_path:
            raise ValueError("Discord edge socket path must be normalized")
        try:
            resolved_parent = path.parent.resolve(strict=True)
        except OSError as exc:
            raise ValueError("Discord edge socket parent must already exist") from exc
        if resolved_parent != path.parent:
            raise ValueError("Discord edge socket parent must be canonical and symlink-free")
        parent_stat = os.stat(resolved_parent, follow_symlinks=False)
        if not stat.S_ISDIR(parent_stat.st_mode):
            raise ValueError("Discord edge socket parent must be a directory")
        if parent_stat.st_uid != os.geteuid():
            raise PermissionError("Discord edge socket parent has the wrong owner")
        if parent_stat.st_mode & 0o022:
            raise PermissionError("Discord edge socket parent is group/world writable")
        if not path.name or path.name in {".", ".."}:
            raise ValueError("Discord edge socket filename is invalid")
        return path, (parent_stat.st_dev, parent_stat.st_ino)

    def _revalidate_socket_parent(self) -> None:
        parent_stat = os.stat(self.socket_path.parent, follow_symlinks=False)
        if (
            not stat.S_ISDIR(parent_stat.st_mode)
            or parent_stat.st_uid != os.geteuid()
            or parent_stat.st_mode & 0o022
            or (parent_stat.st_dev, parent_stat.st_ino) != self._parent_identity
        ):
            raise PermissionError("Discord edge socket parent identity or mode changed")

    @property
    def fileno(self) -> int:
        listener = self._listener
        return -1 if listener is None else listener.fileno()

    def start(self) -> None:
        if self._listener is not None:
            return
        self._revalidate_socket_parent()
        if self.socket_path.exists() or self.socket_path.is_symlink():
            raise FileExistsError("refusing to replace an existing Discord edge path")
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.set_inheritable(False)
        try:
            listener.bind(str(self.socket_path))
            socket_stat = self.socket_path.lstat()
            self._listener_identity = (socket_stat.st_dev, socket_stat.st_ino)
            if (
                not stat.S_ISSOCK(socket_stat.st_mode)
                or socket_stat.st_uid != os.geteuid()
                or socket_stat.st_nlink != 1
            ):
                raise PermissionError("bound Discord edge endpoint identity is invalid")
            os.chmod(self.socket_path, SOCKET_MODE)
            socket_stat = self.socket_path.lstat()
            if (
                (socket_stat.st_dev, socket_stat.st_ino) != self._listener_identity
                or stat.S_IMODE(socket_stat.st_mode) != SOCKET_MODE
            ):
                raise PermissionError("bound Discord edge endpoint mode changed")
            listener.listen()
            listener.settimeout(0.2)
        except BaseException:
            listener.close()
            self._unlink_owned_socket()
            raise
        self._listener = listener

    def serve_forever(self) -> None:
        self.start()
        listener = self._listener
        assert listener is not None
        try:
            while not self._stop_event.is_set():
                try:
                    conn, _address = listener.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if self._stop_event.is_set():
                        break
                    raise
                conn.set_inheritable(False)
                if not self._connection_slots.acquire(blocking=False):
                    conn.close()
                    continue
                worker = threading.Thread(
                    target=self._run_connection,
                    args=(conn,),
                    name="discord-edge-client",
                    daemon=True,
                )
                with self._state_lock:
                    self._workers.add(worker)
                    self._connections.add(conn)
                worker.start()
        finally:
            self._close_listener()
            self._unlink_owned_socket()

    def _run_connection(self, conn: socket.socket) -> None:
        try:
            try:
                conn.settimeout(self.connection_timeout_seconds)
                peer = self.peer_credentials_getter(conn)
            except (OSError, ValueError, TypeError):
                return
            if not self._is_authorized(peer):
                return

            while not self._stop_event.is_set():
                try:
                    body = _receive_request_body(conn)
                    message = decode_request_json(body)
                    reconciliation_query = None
                    request = None
                    if message.get("protocol") == RECONCILIATION_PROTOCOL_VERSION:
                        reconciliation_query = parse_reconciliation_query(message)
                    else:
                        now_unix_ms = self.runtime.clock_ms()
                        try:
                            request = parse_request(
                                message,
                                now_unix_ms=now_unix_ms,
                            )
                        except DiscordEdgeProtocolError as exc:
                            raw_deadline = message.get("deadline_unix_ms")
                            if (
                                exc.code is not DiscordEdgeErrorCode.INVALID_DEADLINE
                                or isinstance(raw_deadline, bool)
                                or not isinstance(raw_deadline, int)
                                or raw_deadline > now_unix_ms
                            ):
                                raise
                            request = parse_request_for_reconciliation(message)
                except (
                    OSError,
                    socket.timeout,
                    _FrameError,
                    DiscordEdgeProtocolError,
                    TypeError,
                    ValueError,
                ):
                    return

                # A persistent connection is never permanent authority.  The
                # current systemd MainPID is read immediately before each
                # runtime call, so a rotated process and its children fail.
                if not self._is_authorized(peer):
                    return
                try:
                    if reconciliation_query is not None:
                        try:
                            reconciliation = self.runtime.reconcile(
                                reconciliation_query
                            )
                        except DiscordEdgeRuntimeError as exc:
                            if (
                                exc.code
                                is DiscordEdgeRuntimeErrorCode.RECONCILIATION_NOT_AVAILABLE
                            ):
                                _send_response(
                                    conn,
                                    _reconciliation_not_available_message(),
                                )
                                continue
                            raise
                        else:
                            _send_response(
                                conn,
                                _reconciliation_response_message(reconciliation),
                            )
                    else:
                        assert request is not None
                        result = self.runtime.execute(request)
                        _send_response(conn, _response_message(result))
                except Exception:
                    return
        finally:
            with self._state_lock:
                self._connections.discard(conn)
                self._workers.discard(threading.current_thread())
            conn.close()
            self._connection_slots.release()

    def _is_authorized(self, peer: object) -> bool:
        if not isinstance(peer, DiscordEdgePeerCredentials):
            return False
        if (
            peer.pid <= 0
            or peer.uid != self.expected_client_uid
            or peer.gid < 0
        ):
            return False
        try:
            main_pid = self.main_pid_provider.main_pid(self.gateway_unit)
        except Exception:
            return False
        return (
            isinstance(main_pid, int)
            and not isinstance(main_pid, bool)
            and main_pid > 0
            and peer.pid == main_pid
        )

    def shutdown(self) -> None:
        self._stop_event.set()
        self._close_listener()
        with self._state_lock:
            connections = list(self._connections)
            workers = list(self._workers)
        for conn in connections:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            conn.close()
        for worker in workers:
            if worker is not threading.current_thread() and worker.ident is not None:
                worker.join(timeout=2)
        self._unlink_owned_socket()

    def _close_listener(self) -> None:
        listener = self._listener
        self._listener = None
        if listener is not None:
            listener.close()

    def _unlink_owned_socket(self) -> None:
        identity = self._listener_identity
        if identity is None:
            return
        try:
            socket_stat = self.socket_path.lstat()
        except FileNotFoundError:
            self._listener_identity = None
            return
        if (
            stat.S_ISSOCK(socket_stat.st_mode)
            and (socket_stat.st_dev, socket_stat.st_ino) == identity
        ):
            try:
                self.socket_path.unlink()
            except FileNotFoundError:
                pass
        self._listener_identity = None


__all__ = [
    "MAX_RESPONSE_BYTES",
    "SOCKET_MODE",
    "DiscordEdgeMainPidProvider",
    "DiscordEdgePeerCredentials",
    "DiscordEdgeUnixServer",
    "SystemctlDiscordEdgeMainPidProvider",
    "linux_discord_edge_peer_credentials",
]
