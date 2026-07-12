"""Persistent, credential-free client for the Canonical writer Unix socket."""

from __future__ import annotations

import os
import re
import socket
import struct
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from gateway.canonical_writer_protocol import (
    MAX_DEADLINE_SECONDS,
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    READ_ONLY_OPERATIONS,
    UNKNOWN_REQUEST_ID,
    CanonicalWriterOperation,
    ErrorCode,
    ProtocolError,
    make_request,
    parse_response,
    receive_message,
    send_message,
)


class CanonicalWriterClientError(RuntimeError):
    """A stable client/remote error safe to return through a tool boundary."""

    def __init__(self, code: ErrorCode, message: str, *, retryable: bool = False) -> None:
        self.code = code
        self.retryable = retryable
        super().__init__(message)


@dataclass(frozen=True)
class CanonicalWriterCallResult:
    request_id: str
    status: str
    result: Mapping[str, Any]


RequestContextBuilder = Callable[[Mapping[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True)
class ServerPeerCredentials:
    pid: int
    uid: int
    gid: int


class ServerPeerAuthorizer(Protocol):
    """Revalidate one connected writer peer against current service state."""

    def authorize(self, peer: ServerPeerCredentials) -> bool: ...


class ServerMainPidProvider(Protocol):
    def main_pid(self, unit_name: str) -> int | None: ...


_SYSTEMD_UNIT_RE = re.compile(r"^[A-Za-z0-9_.@:-]+\.service$")


class SystemctlServerMainPidProvider:
    """Read one exact writer MainPID through a bounded absolute systemctl call."""

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


class ExactServerMainPidAuthorizer:
    """Require the writer UID and exact current systemd service MainPID."""

    def __init__(
        self,
        *,
        server_unit: str,
        expected_server_uid: int,
        main_pid_provider: ServerMainPidProvider,
    ) -> None:
        if not _SYSTEMD_UNIT_RE.fullmatch(server_unit):
            raise ValueError("invalid Canonical writer systemd service unit")
        if isinstance(expected_server_uid, bool) or expected_server_uid < 0:
            raise ValueError("invalid Canonical writer service UID")
        if not callable(getattr(main_pid_provider, "main_pid", None)):
            raise ValueError("Canonical writer MainPID provider is required")
        self.server_unit = server_unit
        self.expected_server_uid = expected_server_uid
        self.main_pid_provider = main_pid_provider

    def authorize(self, peer: ServerPeerCredentials) -> bool:
        if peer.uid != self.expected_server_uid:
            return False
        try:
            current_main_pid = self.main_pid_provider.main_pid(self.server_unit)
        except Exception:
            return False
        return current_main_pid is not None and current_main_pid > 0 and (
            peer.pid == current_main_pid
        )


ServerPeerCredentialsGetter = Callable[[socket.socket], ServerPeerCredentials]
_PEER_CREDENTIALS_STRUCT = struct.Struct("3i")


def linux_server_peer_credentials(sock: socket.socket) -> ServerPeerCredentials:
    """Return the connected server's Linux SO_PEERCRED identity."""

    so_peercred = getattr(socket, "SO_PEERCRED", None)
    if so_peercred is None:
        raise OSError("SO_PEERCRED is unavailable on this platform")
    raw = sock.getsockopt(socket.SOL_SOCKET, so_peercred, _PEER_CREDENTIALS_STRUCT.size)
    if len(raw) != _PEER_CREDENTIALS_STRUCT.size:
        raise OSError("SO_PEERCRED returned an invalid payload")
    return ServerPeerCredentials(*_PEER_CREDENTIALS_STRUCT.unpack(raw))


def _copy_request_context(runtime: Mapping[str, Any]) -> Mapping[str, Any]:
    return dict(runtime)


class CanonicalWriterClient:
    """Serialize requests over one non-inheritable AF_UNIX connection.

    The client takes its socket path explicitly and never reads a bearer token
    or authentication secret from configuration or the environment.  Both
    sides authorize operating-system peer credentials against the exact current
    systemd MainPID before privileged work is dispatched.
    """

    def __init__(
        self,
        socket_path: str | os.PathLike[str],
        *,
        connect_timeout_seconds: float = 2.0,
        request_timeout_seconds: float = 15.0,
        max_reconnect_attempts: int = 1,
        request_context_builder: RequestContextBuilder = _copy_request_context,
        server_authorizer: ServerPeerAuthorizer,
        server_peer_credentials_getter: ServerPeerCredentialsGetter = (
            linux_server_peer_credentials
        ),
    ) -> None:
        path = Path(socket_path)
        if not path.is_absolute():
            raise ValueError("Canonical writer socket path must be absolute")
        if not 0 < connect_timeout_seconds <= MAX_DEADLINE_SECONDS:
            raise ValueError("connect timeout is outside the protocol bound")
        if not 0 < request_timeout_seconds <= MAX_DEADLINE_SECONDS:
            raise ValueError("request timeout is outside the protocol bound")
        if isinstance(max_reconnect_attempts, bool) or not 0 <= max_reconnect_attempts <= 3:
            raise ValueError("max reconnect attempts must be between 0 and 3")
        if not callable(request_context_builder):
            raise ValueError("request context builder must be callable")
        if not callable(getattr(server_authorizer, "authorize", None)):
            raise ValueError("a reciprocal Canonical writer server authorizer is required")
        if not callable(server_peer_credentials_getter):
            raise ValueError("server peer credentials getter must be callable")

        self.socket_path = str(path)
        self.connect_timeout_seconds = float(connect_timeout_seconds)
        self.request_timeout_seconds = float(request_timeout_seconds)
        self.max_reconnect_attempts = max_reconnect_attempts
        self.request_context_builder = request_context_builder
        self.server_authorizer = server_authorizer
        self.server_peer_credentials_getter = server_peer_credentials_getter
        self._owner_pid = os.getpid()
        self._sock: socket.socket | None = None
        self._server_peer: ServerPeerCredentials | None = None
        self._sequence = 0
        self._lock = threading.Lock()
        if hasattr(os, "register_at_fork"):
            os.register_at_fork(after_in_child=self._after_fork_child)

    def _after_fork_child(self) -> None:
        sock = self._sock
        self._sock = None
        self._server_peer = None
        self._owner_pid = -1
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def _ensure_owner(self) -> None:
        if os.getpid() != self._owner_pid:
            raise CanonicalWriterClientError(
                ErrorCode.UNAUTHORIZED_PEER,
                "Canonical writer clients cannot be used from a child process.",
            )

    def _connect(self, timeout_seconds: float) -> socket.socket:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.set_inheritable(False)
        try:
            sock.settimeout(min(self.connect_timeout_seconds, timeout_seconds))
            sock.connect(self.socket_path)
            peer = self.server_peer_credentials_getter(sock)
            if not self._server_is_authorized(peer):
                raise CanonicalWriterClientError(
                    ErrorCode.UNAUTHORIZED_PEER,
                    "Canonical writer server is not authorized.",
                )
        except BaseException:
            sock.close()
            raise
        self._sock = sock
        self._server_peer = peer
        return sock

    def _server_is_authorized(self, peer: ServerPeerCredentials) -> bool:
        if not isinstance(peer, ServerPeerCredentials):
            return False
        if peer.pid <= 0 or peer.uid < 0 or peer.gid < 0:
            return False
        try:
            return self.server_authorizer.authorize(peer) is True
        except Exception:
            return False

    def _reauthorize_server(self, sock: socket.socket) -> None:
        try:
            peer = self.server_peer_credentials_getter(sock)
        except Exception as exc:
            self._close_unlocked()
            raise CanonicalWriterClientError(
                ErrorCode.UNAUTHORIZED_PEER,
                "Canonical writer server identity is unavailable.",
            ) from exc
        if self._server_peer is None or peer != self._server_peer:
            self._close_unlocked()
            raise CanonicalWriterClientError(
                ErrorCode.UNAUTHORIZED_PEER,
                "Canonical writer server identity changed.",
            )
        if not self._server_is_authorized(peer):
            self._close_unlocked()
            raise CanonicalWriterClientError(
                ErrorCode.UNAUTHORIZED_PEER,
                "Canonical writer server is not authorized.",
            )

    def _close_unlocked(self) -> None:
        sock = self._sock
        self._sock = None
        self._server_peer = None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def close(self) -> None:
        with self._lock:
            self._close_unlocked()

    @property
    def fileno(self) -> int:
        with self._lock:
            return -1 if self._sock is None else self._sock.fileno()

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def call(
        self,
        operation: CanonicalWriterOperation | str,
        payload: Mapping[str, Any],
        *,
        runtime: Mapping[str, Any],
        timeout_seconds: float | None = None,
        idempotency_key: str | None = None,
    ) -> CanonicalWriterCallResult:
        """Perform one bounded call, reconnecting safely at most N times.

        A transport retry after bytes may have been sent is allowed only for a
        read operation or when the caller supplied an idempotency key.
        """

        self._ensure_owner()
        timeout = self.request_timeout_seconds if timeout_seconds is None else timeout_seconds
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise ValueError("timeout must be a number")
        if not 0 < timeout <= MAX_DEADLINE_SECONDS:
            raise ValueError("timeout is outside the protocol bound")
        try:
            typed_operation = CanonicalWriterOperation(operation)
        except ValueError as exc:
            raise CanonicalWriterClientError(
                ErrorCode.UNKNOWN_OPERATION,
                "Canonical writer operation is not allowed.",
            ) from exc
        built_runtime = self.request_context_builder(runtime)
        if not isinstance(built_runtime, Mapping):
            raise ValueError("request context builder must return a mapping")

        overall_deadline = time.monotonic() + float(timeout)
        with self._lock:
            for attempt in range(self.max_reconnect_attempts + 1):
                remaining = overall_deadline - time.monotonic()
                if remaining <= 0:
                    self._close_unlocked()
                    raise CanonicalWriterClientError(
                        ErrorCode.TIMEOUT,
                        "Canonical writer request timed out.",
                        retryable=True,
                    )

                request = make_request(
                    typed_operation,
                    payload,
                    runtime=built_runtime,
                    sequence=self._next_sequence(),
                    timeout_seconds=min(remaining, MAX_DEADLINE_SECONDS),
                    idempotency_key=idempotency_key,
                )
                may_have_been_sent = False
                try:
                    sock = self._sock or self._connect(remaining)
                    sock.settimeout(remaining)
                    self._reauthorize_server(sock)
                    may_have_been_sent = True
                    send_message(sock, request.to_message(), max_bytes=MAX_REQUEST_BYTES)
                    raw_response = receive_message(sock, max_bytes=MAX_RESPONSE_BYTES)
                    response = parse_response(raw_response)
                except socket.timeout as exc:
                    self._close_unlocked()
                    failure = CanonicalWriterClientError(
                        ErrorCode.TIMEOUT,
                        "Canonical writer request timed out.",
                        retryable=True,
                    )
                    if not self._may_retry(
                        attempt, typed_operation, idempotency_key, may_have_been_sent
                    ):
                        raise failure from exc
                    continue
                except (OSError, ProtocolError) as exc:
                    self._close_unlocked()
                    code = exc.code if isinstance(exc, ProtocolError) else ErrorCode.TRANSPORT_ERROR
                    failure = CanonicalWriterClientError(
                        code,
                        "Canonical writer transport failed.",
                        retryable=True,
                    )
                    if not self._may_retry(
                        attempt, typed_operation, idempotency_key, may_have_been_sent
                    ):
                        raise failure from exc
                    continue

                if not response.ok and response.request_id == UNKNOWN_REQUEST_ID:
                    assert response.error_code is not None
                    raise CanonicalWriterClientError(
                        response.error_code,
                        response.error_message or "Canonical writer request failed.",
                        retryable=response.retryable,
                    )
                if response.request_id != request.request_id:
                    self._close_unlocked()
                    raise CanonicalWriterClientError(
                        ErrorCode.RESPONSE_MISMATCH,
                        "Canonical writer response does not match the request.",
                    )
                if not response.ok:
                    assert response.error_code is not None
                    raise CanonicalWriterClientError(
                        response.error_code,
                        response.error_message or "Canonical writer request failed.",
                        retryable=response.retryable,
                    )
                assert response.status is not None and response.result is not None
                return CanonicalWriterCallResult(
                    request_id=response.request_id,
                    status=response.status,
                    result=response.result,
                )

        raise AssertionError("bounded reconnect loop exhausted without a result")

    def request(
        self,
        operation: str,
        payload: Mapping[str, Any],
        runtime: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Compatibility surface for synchronous Canonical integrations."""

        return dict(self.call(operation, payload, runtime=runtime).result)

    def _may_retry(
        self,
        attempt: int,
        operation: CanonicalWriterOperation,
        idempotency_key: str | None,
        may_have_been_sent: bool,
    ) -> bool:
        if attempt >= self.max_reconnect_attempts:
            return False
        if not may_have_been_sent:
            return True
        return operation in READ_ONLY_OPERATIONS or idempotency_key is not None

    def __enter__(self) -> CanonicalWriterClient:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()
