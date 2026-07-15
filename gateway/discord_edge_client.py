"""Credential-free client for the privileged Discord egress Unix socket.

The gateway uses this client only after the Canonical Writer has returned one
fully signed :class:`~gateway.discord_edge_protocol.DiscordEdgeRequest`.  The
client owns no Discord token or signing key and never retries after request
bytes may have reached the edge.  Receipt authenticity is deliberately
decided by the Canonical Writer, not by this unprivileged process.
"""

from __future__ import annotations

import os
import socket
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from gateway.canonical_writer_protocol import decode_json_object
from gateway.discord_edge_protocol import (
    MAX_REQUEST_BYTES,
    RECONCILIATION_NOT_AVAILABLE_ERROR,
    RECONCILIATION_RESPONSE_VERSION,
    DiscordEdgeErrorCode,
    DiscordEdgeReconciliationQuery,
    DiscordEdgeReceiptOutcome,
    DiscordEdgeRequest,
    SignedDiscordEdgeEnvelope,
    canonical_json_bytes,
    parse_request,
    parse_request_for_reconciliation,
    parse_reconciliation_query,
)

MAX_RESPONSE_BYTES = 128 * 1024
_FRAME_HEADER = struct.Struct("!I")
_PEER_CREDENTIALS = struct.Struct("3i")


@dataclass(frozen=True)
class DiscordEdgeServerPeer:
    pid: int
    uid: int
    gid: int


class DiscordEdgeServerAuthorizer(Protocol):
    def authorize(self, peer: DiscordEdgeServerPeer) -> bool: ...


ServerPeerGetter = Callable[[socket.socket], DiscordEdgeServerPeer]


def linux_discord_edge_server_peer(sock: socket.socket) -> DiscordEdgeServerPeer:
    """Read Linux ``SO_PEERCRED`` for the connected edge process."""

    so_peercred = getattr(socket, "SO_PEERCRED", None)
    if so_peercred is None:
        raise OSError("SO_PEERCRED is unavailable")
    raw = sock.getsockopt(socket.SOL_SOCKET, so_peercred, _PEER_CREDENTIALS.size)
    if len(raw) != _PEER_CREDENTIALS.size:
        raise OSError("SO_PEERCRED returned an invalid value")
    return DiscordEdgeServerPeer(*_PEER_CREDENTIALS.unpack(raw))


class DiscordEdgeClientError(RuntimeError):
    """Secret-free local boundary failure.

    ``dispatch_uncertain`` is true only when request bytes may already have
    reached the token-owning process.  Callers must reconcile the durable edge
    journal and must never blindly re-execute in that case.
    """

    def __init__(self, code: str, *, dispatch_uncertain: bool = False) -> None:
        self.code = code
        self.dispatch_uncertain = dispatch_uncertain
        super().__init__(code)


@dataclass(frozen=True)
class DiscordEdgeCallResult:
    state: str
    blocker: str | None
    replayed: bool
    receipt: SignedDiscordEdgeEnvelope


@dataclass(frozen=True)
class DiscordEdgeReconciliationCallResult:
    request: DiscordEdgeRequest
    state: str
    blocker: str | None
    replayed: bool
    receipt: SignedDiscordEdgeEnvelope


def _receive_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise OSError("edge connection closed during response")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _receive_response(sock: socket.socket) -> dict[str, Any]:
    header = _receive_exact(sock, _FRAME_HEADER.size)
    (size,) = _FRAME_HEADER.unpack(header)
    if size == 0 or size > MAX_RESPONSE_BYTES:
        raise ValueError("discord_edge_response_frame_invalid")
    return decode_json_object(_receive_exact(sock, size))


def _parse_response(
    value: Mapping[str, Any],
    *,
    expected_request: DiscordEdgeRequest,
) -> DiscordEdgeCallResult:
    if set(value) != {"state", "blocker", "replayed", "receipt"}:
        raise ValueError("discord_edge_response_shape_invalid")
    state = value.get("state")
    blocker = value.get("blocker")
    replayed = value.get("replayed")
    if state not in {"dispatching", "verified", "blocked"}:
        raise ValueError("discord_edge_response_state_invalid")
    if blocker is not None and (
        not isinstance(blocker, str) or not blocker or len(blocker) > 128
    ):
        raise ValueError("discord_edge_response_blocker_invalid")
    if type(replayed) is not bool:
        raise ValueError("discord_edge_response_replay_invalid")
    receipt = SignedDiscordEdgeEnvelope.from_mapping(
        value.get("receipt"),
        code=DiscordEdgeErrorCode.INVALID_RECEIPT,
        label="receipt",
    )
    payload = receipt.payload
    exact_binding = {
        "edge_request_id": expected_request.request_id,
        "operation": expected_request.intent.operation.value,
        "target": expected_request.intent.target.to_dict(),
        "idempotency_key": expected_request.intent.idempotency_key,
        "request_sha256": expected_request.intent.request_sha256,
        "content_sha256": expected_request.intent.content_sha256,
    }
    if any(payload.get(key) != expected for key, expected in exact_binding.items()):
        raise ValueError("discord_edge_response_binding_invalid")
    try:
        outcome = DiscordEdgeReceiptOutcome(payload.get("outcome"))
    except (TypeError, ValueError) as exc:
        raise ValueError("discord_edge_response_outcome_invalid") from exc
    state_outcomes = {
        "verified": {DiscordEdgeReceiptOutcome.VERIFIED},
        "blocked": {
            DiscordEdgeReceiptOutcome.BLOCKED_BEFORE_DISPATCH,
            DiscordEdgeReceiptOutcome.FAILED_BEFORE_DISPATCH,
        },
        "dispatching": {
            DiscordEdgeReceiptOutcome.ACCEPTED_UNVERIFIED,
            DiscordEdgeReceiptOutcome.DISPATCH_UNCERTAIN,
        },
    }
    if outcome not in state_outcomes[state]:
        raise ValueError("discord_edge_response_state_outcome_mismatch")
    return DiscordEdgeCallResult(
        state=str(state),
        blocker=blocker,
        replayed=replayed,
        receipt=receipt,
    )


def _parse_reconciliation_response(
    value: Mapping[str, Any],
    *,
    expected_query: DiscordEdgeReconciliationQuery,
) -> DiscordEdgeReconciliationCallResult:
    if set(value) == {"protocol", "error"}:
        if (
            value.get("protocol") == RECONCILIATION_RESPONSE_VERSION
            and value.get("error") == RECONCILIATION_NOT_AVAILABLE_ERROR
        ):
            raise DiscordEdgeClientError(
                "discord_edge_reconciliation_not_available",
                dispatch_uncertain=False,
            )
        raise ValueError("discord_edge_reconciliation_error_invalid")
    if set(value) != {
        "protocol",
        "request",
        "state",
        "blocker",
        "replayed",
        "receipt",
    }:
        raise ValueError("discord_edge_reconciliation_response_shape_invalid")
    if value.get("protocol") != RECONCILIATION_RESPONSE_VERSION:
        raise ValueError("discord_edge_reconciliation_response_version_invalid")
    request = parse_request_for_reconciliation(value.get("request"))
    serialized_request = canonical_json_bytes(request.to_message())
    if not serialized_request or len(serialized_request) > MAX_REQUEST_BYTES:
        raise ValueError("discord_edge_reconciliation_request_invalid")
    if not expected_query.matches_request(request):
        raise ValueError("discord_edge_reconciliation_binding_invalid")
    call = _parse_response(
        {
            "state": value.get("state"),
            "blocker": value.get("blocker"),
            "replayed": value.get("replayed"),
            "receipt": value.get("receipt"),
        },
        expected_request=request,
    )
    if call.replayed is not True:
        raise ValueError("discord_edge_reconciliation_replay_invalid")
    return DiscordEdgeReconciliationCallResult(
        request=request,
        state=call.state,
        blocker=call.blocker,
        replayed=call.replayed,
        receipt=call.receipt,
    )


class DiscordEdgeClient:
    """One-process, no-resend client with reciprocal exact-PID auth."""

    def __init__(
        self,
        socket_path: str | os.PathLike[str],
        *,
        server_authorizer: DiscordEdgeServerAuthorizer,
        server_peer_getter: ServerPeerGetter = linux_discord_edge_server_peer,
        connect_timeout_seconds: float = 2.0,
        request_timeout_seconds: float = 15.0,
    ) -> None:
        path = Path(socket_path)
        if not path.is_absolute() or path != Path(os.path.normpath(path)):
            raise ValueError("Discord edge socket path must be absolute and normalized")
        if not callable(getattr(server_authorizer, "authorize", None)):
            raise TypeError("Discord edge server authorizer is required")
        if not callable(server_peer_getter):
            raise TypeError("Discord edge peer getter is required")
        if not 0 < connect_timeout_seconds <= 30:
            raise ValueError("Discord edge connect timeout is invalid")
        if not 0 < request_timeout_seconds <= 30:
            raise ValueError("Discord edge request timeout is invalid")
        self.socket_path = str(path)
        self.server_authorizer = server_authorizer
        self.server_peer_getter = server_peer_getter
        self.connect_timeout_seconds = float(connect_timeout_seconds)
        self.request_timeout_seconds = float(request_timeout_seconds)
        self._owner_pid = os.getpid()
        self._sock: socket.socket | None = None
        self._peer: DiscordEdgeServerPeer | None = None
        self._lock = threading.Lock()
        if hasattr(os, "register_at_fork"):
            os.register_at_fork(after_in_child=self._after_fork_child)

    def _after_fork_child(self) -> None:
        sock = self._sock
        self._sock = None
        self._peer = None
        self._owner_pid = -1
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def _require_owner(self) -> None:
        if os.getpid() != self._owner_pid:
            raise DiscordEdgeClientError("discord_edge_client_wrong_process")

    def _authorized(self, peer: object) -> bool:
        if not isinstance(peer, DiscordEdgeServerPeer):
            return False
        if peer.pid <= 0 or peer.uid < 0 or peer.gid < 0:
            return False
        try:
            return self.server_authorizer.authorize(peer) is True
        except Exception:
            return False

    def _close_unlocked(self) -> None:
        sock = self._sock
        self._sock = None
        self._peer = None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def close(self) -> None:
        with self._lock:
            self._close_unlocked()

    def __enter__(self) -> "DiscordEdgeClient":
        self.connect()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _reauthorize(self, sock: socket.socket) -> None:
        peer = self.server_peer_getter(sock)
        if self._peer is None or peer != self._peer or not self._authorized(peer):
            self._close_unlocked()
            raise DiscordEdgeClientError("discord_edge_server_unauthorized")

    def connect(self) -> None:
        """Create a fresh authenticated connection before a writer claim.

        A cached Unix socket can retain peer credentials after the server has
        closed its side.  Reusing that half-closed socket would let the writer
        commit a claim before liveness is discovered.  Every preclaim check
        therefore replaces the prior connection instead of treating peer
        credentials as a health probe.
        """

        self._require_owner()
        with self._lock:
            self._close_unlocked()
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.set_inheritable(False)
            try:
                sock.settimeout(self.connect_timeout_seconds)
                sock.connect(self.socket_path)
                peer = self.server_peer_getter(sock)
                if not self._authorized(peer):
                    raise DiscordEdgeClientError("discord_edge_server_unauthorized")
            except BaseException:
                sock.close()
                raise
            self._sock = sock
            self._peer = peer

    def execute(
        self,
        request_value: Mapping[str, Any],
        *,
        require_preconnected: bool = True,
    ) -> DiscordEdgeCallResult:
        """Send one signed request exactly once and return its signed receipt."""

        self._require_owner()
        request = parse_request(request_value)
        body = canonical_json_bytes(request.to_message())
        if not body or len(body) > MAX_REQUEST_BYTES:
            raise ValueError("discord_edge_request_frame_invalid")
        deadline_seconds = max(
            0.001,
            (request.deadline_unix_ms - int(time.time() * 1000)) / 1000,
        )
        with self._lock:
            if self._sock is None:
                if require_preconnected:
                    raise DiscordEdgeClientError("discord_edge_not_preconnected")
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.set_inheritable(False)
                try:
                    sock.settimeout(min(self.connect_timeout_seconds, deadline_seconds))
                    sock.connect(self.socket_path)
                    peer = self.server_peer_getter(sock)
                    if not self._authorized(peer):
                        raise DiscordEdgeClientError("discord_edge_server_unauthorized")
                except BaseException:
                    sock.close()
                    raise
                self._sock = sock
                self._peer = peer
            sock = self._sock
            assert sock is not None
            may_have_dispatched = False
            try:
                self._reauthorize(sock)
                sock.settimeout(min(self.request_timeout_seconds, deadline_seconds))
                may_have_dispatched = True
                sock.sendall(_FRAME_HEADER.pack(len(body)) + body)
                raw_response = _receive_response(sock)
                self._reauthorize(sock)
                return _parse_response(raw_response, expected_request=request)
            except DiscordEdgeClientError:
                raise
            except (OSError, socket.timeout, TypeError, ValueError) as exc:
                self._close_unlocked()
                raise DiscordEdgeClientError(
                    "discord_edge_transport_failed",
                    dispatch_uncertain=may_have_dispatched,
                ) from exc

    def reconcile(
        self,
        query_value: DiscordEdgeReconciliationQuery | Mapping[str, Any],
        *,
        require_preconnected: bool = False,
    ) -> DiscordEdgeReconciliationCallResult:
        """Read one exact journaled result; this frame cannot dispatch Discord."""

        self._require_owner()
        query = (
            query_value
            if isinstance(query_value, DiscordEdgeReconciliationQuery)
            else parse_reconciliation_query(query_value)
        )
        body = canonical_json_bytes(query.to_message())
        if not body or len(body) > MAX_REQUEST_BYTES:
            raise ValueError("discord_edge_reconciliation_frame_invalid")
        with self._lock:
            if self._sock is None:
                if require_preconnected:
                    raise DiscordEdgeClientError("discord_edge_not_preconnected")
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.set_inheritable(False)
                try:
                    sock.settimeout(self.connect_timeout_seconds)
                    sock.connect(self.socket_path)
                    peer = self.server_peer_getter(sock)
                    if not self._authorized(peer):
                        raise DiscordEdgeClientError(
                            "discord_edge_server_unauthorized"
                        )
                except BaseException:
                    sock.close()
                    raise
                self._sock = sock
                self._peer = peer
            sock = self._sock
            assert sock is not None
            try:
                self._reauthorize(sock)
                sock.settimeout(self.request_timeout_seconds)
                sock.sendall(_FRAME_HEADER.pack(len(body)) + body)
                raw_response = _receive_response(sock)
                self._reauthorize(sock)
                return _parse_reconciliation_response(
                    raw_response,
                    expected_query=query,
                )
            except DiscordEdgeClientError:
                raise
            except (OSError, socket.timeout, TypeError, ValueError) as exc:
                self._close_unlocked()
                raise DiscordEdgeClientError(
                    "discord_edge_reconciliation_failed",
                    dispatch_uncertain=False,
                ) from exc


__all__ = [
    "MAX_RESPONSE_BYTES",
    "DiscordEdgeCallResult",
    "DiscordEdgeClient",
    "DiscordEdgeClientError",
    "DiscordEdgeReconciliationCallResult",
    "DiscordEdgeServerAuthorizer",
    "DiscordEdgeServerPeer",
    "linux_discord_edge_server_peer",
]
