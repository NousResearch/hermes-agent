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
_SYSTEMD_SYSTEM_SLICE_PATH = Path("/sys/fs/cgroup/system.slice")
_CGROUP_PROCS_FILENAME = "cgroup.procs"
_MAX_CGROUP_PROCS_BYTES = 4096
_MAX_LINUX_TGID = (1 << 31) - 1


def _effective_uid() -> int:
    getter = getattr(os, "geteuid", None)
    if not callable(getter):
        raise RuntimeError("Canonical writer service requires POSIX UID support")
    return int(getter())


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


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_uid),
        int(value.st_gid),
    )


class SystemdCgroupV2MainPidProvider:
    """Read one service TGID directly from a root-owned cgroup-v2 boundary.

    ``systemctl show`` is unsuitable inside a peer-authentication heartbeat:
    the helper process temporarily joins the caller's service cgroup and can
    make an otherwise single-process boundary appear to contain a child.  This
    provider never creates a process.  It opens every attacker-selectable path
    component with ``O_NOFOLLOW`` and requires the same root-owned identity to
    remain reachable through the stable directory descriptors after the read.
    """

    def __init__(
        self,
        *,
        _system_slice_path: str | os.PathLike[str] = _SYSTEMD_SYSTEM_SLICE_PATH,
        _expected_owner_uid: int = 0,
        _expected_owner_gid: int = 0,
    ) -> None:
        system_slice_path = Path(_system_slice_path)
        if not system_slice_path.is_absolute():
            raise ValueError("systemd system.slice path must be absolute")
        if _expected_owner_uid < 0 or _expected_owner_gid < 0:
            raise ValueError("cgroup owner identity must be non-negative")
        self._system_slice_path = system_slice_path
        self._expected_owner_uid = int(_expected_owner_uid)
        self._expected_owner_gid = int(_expected_owner_gid)

    def _validate_identity(
        self,
        value: os.stat_result,
        *,
        directory: bool,
    ) -> tuple[int, int, int, int, int]:
        expected_type = stat.S_ISDIR if directory else stat.S_ISREG
        if (
            not expected_type(value.st_mode)
            or value.st_uid != self._expected_owner_uid
            or value.st_gid != self._expected_owner_gid
            or stat.S_IMODE(value.st_mode) & 0o022
        ):
            raise PermissionError("cgroup path lacks root-owned immutable identity")
        return _file_identity(value)

    @staticmethod
    def _directory_open_flags() -> int:
        if any(
            not hasattr(os, name)
            for name in ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW")
        ):
            raise OSError("required Linux secure-open flags are unavailable")
        return (
            os.O_RDONLY
            | os.O_CLOEXEC
            | os.O_DIRECTORY
            | os.O_NOFOLLOW
        )

    @staticmethod
    def _file_open_flags() -> int:
        if any(not hasattr(os, name) for name in ("O_CLOEXEC", "O_NOFOLLOW")):
            raise OSError("required Linux secure-open flags are unavailable")
        return (
            os.O_RDONLY
            | os.O_CLOEXEC
            | os.O_NOFOLLOW
        )

    def _open_system_slice(self) -> tuple[int, tuple[int, int, int, int, int]]:
        before = os.lstat(self._system_slice_path)
        before_identity = self._validate_identity(before, directory=True)
        descriptor = os.open(self._system_slice_path, self._directory_open_flags())
        try:
            opened_identity = self._validate_identity(
                os.fstat(descriptor),
                directory=True,
            )
            after_identity = self._validate_identity(
                os.lstat(self._system_slice_path),
                directory=True,
            )
            if before_identity != opened_identity or opened_identity != after_identity:
                raise PermissionError("system.slice identity changed during open")
            return descriptor, opened_identity
        except BaseException:
            os.close(descriptor)
            raise

    def _open_child(
        self,
        parent_descriptor: int,
        name: str,
        *,
        directory: bool,
    ) -> tuple[int, tuple[int, int, int, int, int]]:
        before = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        before_identity = self._validate_identity(before, directory=directory)
        flags = self._directory_open_flags() if directory else self._file_open_flags()
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
        try:
            opened_identity = self._validate_identity(
                os.fstat(descriptor),
                directory=directory,
            )
            after_identity = self._validate_identity(
                os.stat(
                    name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                ),
                directory=directory,
            )
            if before_identity != opened_identity or opened_identity != after_identity:
                raise PermissionError("cgroup path identity changed during open")
            return descriptor, opened_identity
        except BaseException:
            os.close(descriptor)
            raise

    def _read_bounded(self, descriptor: int) -> bytes:
        chunks: list[bytes] = []
        observed = 0
        while observed <= _MAX_CGROUP_PROCS_BYTES:
            chunk = os.read(descriptor, _MAX_CGROUP_PROCS_BYTES + 1 - observed)
            if not chunk:
                break
            chunks.append(chunk)
            observed += len(chunk)
        payload = b"".join(chunks)
        if len(payload) > _MAX_CGROUP_PROCS_BYTES:
            raise ValueError("cgroup.procs exceeds bounded input size")
        return payload

    def _identity_is_current(
        self,
        descriptor: int,
        expected_identity: tuple[int, int, int, int, int],
        *,
        directory: bool,
    ) -> bool:
        return (
            self._validate_identity(os.fstat(descriptor), directory=directory)
            == expected_identity
        )

    def main_pid(self, unit_name: str) -> int | None:
        if not _SYSTEMD_UNIT_RE.fullmatch(unit_name):
            return None
        system_slice_descriptor = unit_descriptor = procs_descriptor = -1
        try:
            system_slice_descriptor, system_slice_identity = self._open_system_slice()
            unit_descriptor, unit_identity = self._open_child(
                system_slice_descriptor,
                unit_name,
                directory=True,
            )
            procs_descriptor, procs_identity = self._open_child(
                unit_descriptor,
                _CGROUP_PROCS_FILENAME,
                directory=False,
            )
            payload = self._read_bounded(procs_descriptor)

            current_procs_identity = self._validate_identity(
                os.stat(
                    _CGROUP_PROCS_FILENAME,
                    dir_fd=unit_descriptor,
                    follow_symlinks=False,
                ),
                directory=False,
            )
            current_unit_identity = self._validate_identity(
                os.stat(
                    unit_name,
                    dir_fd=system_slice_descriptor,
                    follow_symlinks=False,
                ),
                directory=True,
            )
            current_system_slice_identity = self._validate_identity(
                os.lstat(self._system_slice_path),
                directory=True,
            )
            if (
                current_procs_identity != procs_identity
                or current_unit_identity != unit_identity
                or current_system_slice_identity != system_slice_identity
                or not self._identity_is_current(
                    procs_descriptor,
                    procs_identity,
                    directory=False,
                )
                or not self._identity_is_current(
                    unit_descriptor,
                    unit_identity,
                    directory=True,
                )
                or not self._identity_is_current(
                    system_slice_descriptor,
                    system_slice_identity,
                    directory=True,
                )
            ):
                return None

            match = re.fullmatch(rb"([1-9][0-9]*)\n", payload)
            if match is None:
                return None
            tgid = int(match.group(1))
            return tgid if 0 < tgid <= _MAX_LINUX_TGID else None
        except (OSError, ValueError):
            return None
        finally:
            for descriptor in (
                procs_descriptor,
                unit_descriptor,
                system_slice_descriptor,
            ):
                if descriptor >= 0:
                    os.close(descriptor)


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
        expected_socket_gid: int | None = None,
        recover_stale_socket: bool = False,
        socket_mode: int = 0o660,
        connection_timeout_seconds: float = 30.0,
        max_connections: int = 8,
    ) -> None:
        path = Path(socket_path)
        if not path.is_absolute():
            raise ValueError("Canonical writer socket path must be absolute")
        if socket_mode & ~0o777 or socket_mode & 0o007:
            raise ValueError("socket mode contains unsupported bits")
        if (
            expected_socket_gid is not None
            and (type(expected_socket_gid) is not int or expected_socket_gid <= 0)
        ):
            raise ValueError("expected socket GID is invalid")
        if type(recover_stale_socket) is not bool:
            raise TypeError("stale socket recovery policy must be boolean")
        if recover_stale_socket and expected_socket_gid is None:
            raise ValueError("stale socket recovery requires an exact socket GID")
        if not 0 < connection_timeout_seconds <= 300:
            raise ValueError("connection timeout must be between 0 and 300 seconds")
        if not 1 <= max_connections <= 64:
            raise ValueError("max connections must be between 1 and 64")
        self.socket_path = path
        self.authorizer = authorizer
        self.dispatcher = dispatcher
        self.peer_credentials_getter = peer_credentials_getter
        self.replay_guard = replay_guard or ReplayGuard()
        self.expected_socket_gid = expected_socket_gid
        self.recover_stale_socket = recover_stale_socket
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
        if self.expected_socket_gid is not None:
            parent_stat = self.socket_path.parent.stat()
            if (
                not stat.S_ISDIR(parent_stat.st_mode)
                or self.socket_path.parent.is_symlink()
                or parent_stat.st_uid != _effective_uid()
                or parent_stat.st_gid != self.expected_socket_gid
                or stat.S_IMODE(parent_stat.st_mode) != 0o2750
            ):
                raise PermissionError(
                    "Canonical writer runtime directory ownership is invalid"
                )
        self._prepare_socket_path()
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.set_inheritable(False)
        try:
            listener.bind(str(self.socket_path))
            os.chmod(self.socket_path, self.socket_mode)
            socket_stat = self.socket_path.lstat()
            if (
                not stat.S_ISSOCK(socket_stat.st_mode)
                or socket_stat.st_uid != _effective_uid()
                or stat.S_IMODE(socket_stat.st_mode) != self.socket_mode
            ):
                raise RuntimeError("bound Canonical writer endpoint is not a socket")
            if (
                self.expected_socket_gid is not None
                and socket_stat.st_gid != self.expected_socket_gid
            ):
                raise PermissionError(
                    "Canonical writer socket did not inherit the dedicated GID"
                )
            self._listener_identity = (socket_stat.st_dev, socket_stat.st_ino)
            listener.listen()
            listener.settimeout(0.2)
        except BaseException:
            listener.close()
            self._unlink_owned_socket()
            raise
        self._listener = listener

    def _prepare_socket_path(self) -> None:
        try:
            observed = self.socket_path.lstat()
        except FileNotFoundError:
            return
        if not self.recover_stale_socket:
            raise FileExistsError(
                f"refusing to replace existing socket path: {self.socket_path}"
            )
        if (
            not stat.S_ISSOCK(observed.st_mode)
            or observed.st_uid != _effective_uid()
            or observed.st_gid != self.expected_socket_gid
            or stat.S_IMODE(observed.st_mode) != self.socket_mode
            or observed.st_nlink != 1
        ):
            raise PermissionError(
                "Canonical writer stale socket identity is invalid"
            )
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        probe.settimeout(0.2)
        try:
            try:
                probe.connect(str(self.socket_path))
            except ConnectionRefusedError:
                pass
            except FileNotFoundError:
                return
            else:
                raise FileExistsError(
                    "Canonical writer socket is already accepting connections"
                )
        finally:
            probe.close()
        current = self.socket_path.lstat()
        if (current.st_dev, current.st_ino) != (observed.st_dev, observed.st_ino):
            raise RuntimeError("Canonical writer stale socket identity changed")
        self.socket_path.unlink()

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
