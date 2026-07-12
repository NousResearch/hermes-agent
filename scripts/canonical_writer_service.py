"""Privileged Canonical writer Unix-socket service framework.

This module owns transport authentication and typed dispatch only.  Database
credentials and SQL implementations belong to a separately privileged service
package; this framework deliberately exposes no raw-SQL operation.
"""

from __future__ import annotations

import os
import re
import socket
import stat
import struct
import subprocess
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol

from gateway.canonical_writer_protocol import (
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    SUCCESS_STATUSES,
    UNKNOWN_REQUEST_ID,
    CanonicalWriterOperation,
    ErrorCode,
    ProtocolError,
    WriterRequest,
    make_error_response,
    make_success_response,
    parse_request,
    receive_message,
    send_message,
)

_SYSTEMD_UNIT_RE = re.compile(r"^[A-Za-z0-9_.@:-]+\.service$")
_PEER_CREDENTIALS_STRUCT = struct.Struct("3i")


@dataclass(frozen=True)
class PeerCredentials:
    pid: int
    uid: int
    gid: int


class MainPidProvider(Protocol):
    def main_pid(self, unit_name: str) -> int | None: ...


class PeerAuthorizer(Protocol):
    def authorize(self, peer: PeerCredentials) -> bool: ...


class TypedDispatcher(Protocol):
    def dispatch(
        self,
        operation: CanonicalWriterOperation,
        payload: Mapping[str, Any],
        context: DispatchContext,
    ) -> DispatchResult: ...


class SystemctlMainPidProvider:
    """Read the exact MainPID property for one systemd service unit."""

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


class SystemdMainPidAuthorizer:
    """Authorize only the exact gateway service MainPID, never descendants."""

    def __init__(
        self,
        unit_name: str,
        main_pid_provider: MainPidProvider,
        *,
        expected_uid: int | None = None,
    ) -> None:
        if not _SYSTEMD_UNIT_RE.fullmatch(unit_name):
            raise ValueError("invalid systemd service unit name")
        self.unit_name = unit_name
        self.main_pid_provider = main_pid_provider
        self.expected_uid = expected_uid

    def authorize(self, peer: PeerCredentials) -> bool:
        if peer.pid <= 0 or peer.uid < 0 or peer.gid < 0:
            return False
        if self.expected_uid is not None and peer.uid != self.expected_uid:
            return False
        try:
            main_pid = self.main_pid_provider.main_pid(self.unit_name)
        except Exception:
            return False
        return main_pid is not None and peer.pid == main_pid


def linux_peer_credentials(conn: socket.socket) -> PeerCredentials:
    """Return Linux SO_PEERCRED for a connected Unix-domain socket."""

    so_peercred = getattr(socket, "SO_PEERCRED", None)
    if so_peercred is None:
        raise OSError("SO_PEERCRED is unavailable on this platform")
    raw = conn.getsockopt(socket.SOL_SOCKET, so_peercred, _PEER_CREDENTIALS_STRUCT.size)
    if len(raw) != _PEER_CREDENTIALS_STRUCT.size:
        raise OSError("SO_PEERCRED returned an invalid payload")
    return PeerCredentials(*_PEER_CREDENTIALS_STRUCT.unpack(raw))


@dataclass(frozen=True)
class DispatchContext:
    request_id: str
    sequence: int
    deadline_unix_ms: int
    idempotency_key: str | None
    peer: PeerCredentials
    runtime: Mapping[str, Any]


@dataclass(frozen=True)
class DispatchResult:
    status: str = "ok"
    result: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in SUCCESS_STATUSES:
            raise ValueError("dispatch result has an unsupported status")


class DispatchFailure(RuntimeError):
    def __init__(self, code: ErrorCode = ErrorCode.DISPATCH_FAILED) -> None:
        if code not in {ErrorCode.DISPATCH_FAILED, ErrorCode.DISPATCH_UNAVAILABLE}:
            raise ValueError("dispatch failure uses an invalid public error code")
        self.code = code
        super().__init__(code.value)


OperationHandler = Callable[[Mapping[str, Any], DispatchContext], DispatchResult]


class OperationDispatcher:
    """Map the fixed protocol operation enum to explicit handler functions."""

    def __init__(
        self,
        handlers: Mapping[CanonicalWriterOperation | str, OperationHandler],
    ) -> None:
        normalized: dict[CanonicalWriterOperation, OperationHandler] = {}
        for operation, handler in handlers.items():
            try:
                typed_operation = CanonicalWriterOperation(operation)
            except ValueError as exc:
                raise ValueError("dispatcher contains an unknown operation") from exc
            if not callable(handler) or typed_operation in normalized:
                raise ValueError("dispatcher contains an invalid handler")
            normalized[typed_operation] = handler
        self._handlers = normalized

    def dispatch(
        self,
        operation: CanonicalWriterOperation,
        payload: Mapping[str, Any],
        context: DispatchContext,
    ) -> DispatchResult:
        handler = self._handlers.get(operation)
        if handler is None:
            raise DispatchFailure(ErrorCode.DISPATCH_UNAVAILABLE)
        result = handler(payload, context)
        if not isinstance(result, DispatchResult):
            raise DispatchFailure()
        return result


class ReplayGuard:
    """Bounded process-wide request-ID replay cache."""

    def __init__(self, *, max_entries: int = 4096, ttl_seconds: float = 120.0) -> None:
        if max_entries < 1 or not 1 <= ttl_seconds <= 3600:
            raise ValueError("invalid replay guard bounds")
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._entries: OrderedDict[str, float] = OrderedDict()
        self._lock = threading.Lock()

    def seen_or_add(self, request_id: str, *, now: float | None = None) -> bool:
        current = time.monotonic() if now is None else now
        cutoff = current - self.ttl_seconds
        with self._lock:
            while self._entries:
                _oldest_id, oldest_seen = next(iter(self._entries.items()))
                if oldest_seen > cutoff:
                    break
                self._entries.popitem(last=False)
            if request_id in self._entries:
                return True
            self._entries[request_id] = current
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
        return False


PeerCredentialsGetter = Callable[[socket.socket], PeerCredentials]


def _safe_response_request_id(value: object) -> str:
    if not isinstance(value, str):
        return UNKNOWN_REQUEST_ID
    try:
        parsed = uuid.UUID(value)
    except ValueError:
        return UNKNOWN_REQUEST_ID
    return value if parsed.int != 0 and str(parsed) == value else UNKNOWN_REQUEST_ID


class CanonicalWriterServer:
    """Threaded AF_UNIX server with peer-PID auth and typed dispatch."""

    def __init__(
        self,
        socket_path: str | os.PathLike[str],
        *,
        authorizer: PeerAuthorizer,
        dispatcher: TypedDispatcher,
        peer_credentials_getter: PeerCredentialsGetter = linux_peer_credentials,
        replay_guard: ReplayGuard | None = None,
        socket_mode: int = 0o660,
        connection_timeout_seconds: float = 30.0,
        max_connections: int = 8,
    ) -> None:
        path = Path(socket_path)
        if not path.is_absolute():
            raise ValueError("Canonical writer socket path must be absolute")
        if socket_mode & ~0o777 or socket_mode & 0o007:
            raise ValueError("socket mode contains unsupported bits")
        if not 0 < connection_timeout_seconds <= 300:
            raise ValueError("connection timeout must be between 0 and 300 seconds")
        if not 1 <= max_connections <= 64:
            raise ValueError("max connections must be between 1 and 64")
        self.socket_path = path
        self.authorizer = authorizer
        self.dispatcher = dispatcher
        self.peer_credentials_getter = peer_credentials_getter
        self.replay_guard = replay_guard or ReplayGuard()
        self.socket_mode = socket_mode
        self.connection_timeout_seconds = connection_timeout_seconds
        self._listener: socket.socket | None = None
        self._listener_identity: tuple[int, int] | None = None
        self._stop_event = threading.Event()
        self._connection_slots = threading.BoundedSemaphore(max_connections)
        self._workers: set[threading.Thread] = set()
        self._connections: set[socket.socket] = set()
        self._state_lock = threading.Lock()

    @property
    def fileno(self) -> int:
        listener = self._listener
        return -1 if listener is None else listener.fileno()

    def start(self) -> None:
        if self._listener is not None:
            return
        if self.socket_path.exists() or self.socket_path.is_symlink():
            raise FileExistsError(f"refusing to replace existing socket path: {self.socket_path}")
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.set_inheritable(False)
        try:
            listener.bind(str(self.socket_path))
            os.chmod(self.socket_path, self.socket_mode)
            socket_stat = self.socket_path.lstat()
            if not stat.S_ISSOCK(socket_stat.st_mode):
                raise RuntimeError("bound Canonical writer endpoint is not a socket")
            self._listener_identity = (socket_stat.st_dev, socket_stat.st_ino)
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
                    self._safe_send_error(
                        conn,
                        UNKNOWN_REQUEST_ID,
                        ErrorCode.DISPATCH_UNAVAILABLE,
                        retryable=True,
                    )
                    conn.close()
                    continue
                worker = threading.Thread(
                    target=self._run_connection,
                    args=(conn,),
                    name="canonical-writer-client",
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
            except OSError:
                return
            try:
                peer = self.peer_credentials_getter(conn)
            except (OSError, ValueError):
                self._safe_send_error(conn, UNKNOWN_REQUEST_ID, ErrorCode.UNAUTHORIZED_PEER)
                return
            if not self._is_authorized(peer):
                self._safe_send_error(conn, UNKNOWN_REQUEST_ID, ErrorCode.UNAUTHORIZED_PEER)
                return

            last_sequence = 0
            while not self._stop_event.is_set():
                try:
                    message = receive_message(conn, max_bytes=MAX_REQUEST_BYTES)
                except socket.timeout:
                    return
                except OSError:
                    return
                except ProtocolError as exc:
                    if exc.code == ErrorCode.CONNECTION_CLOSED:
                        return
                    self._safe_send_error(conn, UNKNOWN_REQUEST_ID, exc.code)
                    if exc.fatal:
                        return
                    continue

                request_id = message.get("request_id")
                safe_request_id = _safe_response_request_id(request_id)
                try:
                    request = parse_request(message)
                except ProtocolError as exc:
                    self._safe_send_error(conn, safe_request_id, exc.code)
                    continue

                # A persistent connection is not a permanent authorization.
                # Re-read the current systemd MainPID before every dispatch so
                # an old gateway process loses access immediately on rotation.
                if not self._is_authorized(peer):
                    self._safe_send_error(
                        conn,
                        request.request_id,
                        ErrorCode.UNAUTHORIZED_PEER,
                    )
                    return

                if request.sequence <= last_sequence or self.replay_guard.seen_or_add(
                    request.request_id
                ):
                    self._safe_send_error(
                        conn,
                        request.request_id,
                        ErrorCode.REPLAYED_REQUEST,
                    )
                    continue
                last_sequence = request.sequence
                self._dispatch(conn, request, peer)
        finally:
            with self._state_lock:
                self._connections.discard(conn)
                self._workers.discard(threading.current_thread())
            conn.close()
            self._connection_slots.release()

    def _is_authorized(self, peer: PeerCredentials) -> bool:
        try:
            return self.authorizer.authorize(peer) is True
        except Exception:
            return False

    def _dispatch(
        self,
        conn: socket.socket,
        request: WriterRequest,
        peer: PeerCredentials,
    ) -> None:
        context = DispatchContext(
            request_id=request.request_id,
            sequence=request.sequence,
            deadline_unix_ms=request.deadline_unix_ms,
            idempotency_key=request.idempotency_key,
            peer=peer,
            runtime=MappingProxyType(
                {
                    **request.runtime,
                    "peer": MappingProxyType(
                        {"pid": peer.pid, "uid": peer.uid, "gid": peer.gid}
                    ),
                }
            ),
        )
        try:
            result = self.dispatcher.dispatch(request.operation, request.payload, context)
        except DispatchFailure as exc:
            self._safe_send_error(
                conn,
                request.request_id,
                exc.code,
                retryable=exc.code == ErrorCode.DISPATCH_UNAVAILABLE,
            )
            return
        except Exception:
            self._safe_send_error(conn, request.request_id, ErrorCode.INTERNAL_ERROR)
            return
        try:
            response = make_success_response(
                request.request_id,
                status=result.status,
                result=result.result,
            )
            send_message(conn, response, max_bytes=MAX_RESPONSE_BYTES)
        except (OSError, ProtocolError):
            return

    @staticmethod
    def _safe_send_error(
        conn: socket.socket,
        request_id: str,
        code: ErrorCode,
        *,
        retryable: bool = False,
    ) -> None:
        try:
            response = make_error_response(request_id, code, retryable=retryable)
            send_message(conn, response, max_bytes=MAX_RESPONSE_BYTES)
        except (OSError, ProtocolError):
            pass

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
        if (socket_stat.st_dev, socket_stat.st_ino) == identity and stat.S_ISSOCK(
            socket_stat.st_mode
        ):
            try:
                self.socket_path.unlink()
            except FileNotFoundError:
                pass
        self._listener_identity = None
