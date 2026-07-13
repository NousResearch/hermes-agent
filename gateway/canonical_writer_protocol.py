"""Wire protocol for the privileged Canonical Brain writer.

The protocol intentionally has a small, fixed operation surface.  It carries
typed JSON payloads over a Unix stream socket, but it never accepts SQL or an
authentication secret.  Authentication belongs to the server's operating
system peer-credential boundary.
"""

from __future__ import annotations

import json
import socket
import struct
import time
import uuid
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping

PROTOCOL_VERSION = "canonical-writer.v1"
UNKNOWN_REQUEST_ID = "00000000-0000-0000-0000-000000000000"

MAX_REQUEST_BYTES = 128 * 1024
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_DEADLINE_SECONDS = 30.0
MAX_IDEMPOTENCY_KEY_BYTES = 256
MAX_SEQUENCE = (1 << 63) - 1

_FRAME_HEADER = struct.Struct("!I")
_REQUEST_REQUIRED_FIELDS = frozenset(
    {
        "protocol",
        "request_id",
        "sequence",
        "operation",
        "deadline_unix_ms",
        "runtime",
        "payload",
    }
)
_REQUEST_OPTIONAL_FIELDS = frozenset({"idempotency_key"})


class CanonicalWriterOperation(StrEnum):
    """Operations implemented by the privileged writer boundary."""

    PING = "ping"
    CANARY_SCOPE_CLAIM = "canary.scope_claim"
    CASE_QUERY = "case.query"
    ROUTEBACK_CONTEXT = "routeback.context"
    PLAN_ACTIVE_MATCH = "plan.active_match"
    EVENT_APPEND_MODEL = "event.append_model"
    PLAN_TRANSITION = "plan.transition"
    VERIFICATION_APPEND = "verification.append"
    ROUTEBACK_CLAIM = "routeback.claim"
    ROUTEBACK_RECOVER = "routeback.recover"
    ROUTEBACK_FINALIZE_SENT = "routeback.finalize_sent"
    ROUTEBACK_FINALIZE_BLOCKED = "routeback.finalize_blocked"
    LEASE_SHADOW_RECORD = "lease_shadow.record"
    CAPABILITY_GRANT = "capability.grant"
    CAPABILITY_CONSUME = "capability.consume"
    CAPABILITY_REVOKE = "capability.revoke"
    CAPABILITY_REVOKE_SESSION = "capability.revoke_session"
    PROJECTION_READ_EVENTS = "projection.read_events"


CANONICAL_WRITER_OPERATIONS = frozenset(item.value for item in CanonicalWriterOperation)
READ_ONLY_OPERATIONS = frozenset(
    {
        CanonicalWriterOperation.PING,
        CanonicalWriterOperation.CASE_QUERY,
        CanonicalWriterOperation.ROUTEBACK_CONTEXT,
        CanonicalWriterOperation.PLAN_ACTIVE_MATCH,
        CanonicalWriterOperation.PROJECTION_READ_EVENTS,
    }
)

# These values are derived from the authenticated connection by the service.
# Neither a model payload nor the client-supplied runtime envelope may claim
# them on the wire.
PEER_IDENTITY_FIELDS = frozenset(
    {"peer", "peer_identity", "peer_pid", "peer_uid", "peer_gid", "systemd_main_pid"}
)
RESERVED_PAYLOAD_FIELDS = PEER_IDENTITY_FIELDS | frozenset({"runtime", "request_context"})


class ErrorCode(StrEnum):
    """Stable machine-readable errors returned by the boundary."""

    MALFORMED_FRAME = "malformed_frame"
    FRAME_TOO_LARGE = "frame_too_large"
    INVALID_JSON = "invalid_json"
    INVALID_REQUEST = "invalid_request"
    UNSUPPORTED_VERSION = "unsupported_version"
    INVALID_REQUEST_ID = "invalid_request_id"
    INVALID_DEADLINE = "invalid_deadline"
    DEADLINE_EXPIRED = "deadline_expired"
    DEADLINE_TOO_FAR = "deadline_too_far"
    UNKNOWN_OPERATION = "unknown_operation"
    REPLAYED_REQUEST = "replayed_request"
    UNAUTHORIZED_PEER = "unauthorized_peer"
    DISPATCH_UNAVAILABLE = "dispatch_unavailable"
    DISPATCH_FAILED = "dispatch_failed"
    INTERNAL_ERROR = "internal_error"
    CONNECTION_CLOSED = "connection_closed"
    TIMEOUT = "timeout"
    TRANSPORT_ERROR = "transport_error"
    RESPONSE_MISMATCH = "response_mismatch"
    INVALID_RESPONSE = "invalid_response"


ERROR_MESSAGES: Mapping[ErrorCode, str] = {
    ErrorCode.MALFORMED_FRAME: "The Canonical writer frame is malformed.",
    ErrorCode.FRAME_TOO_LARGE: "The Canonical writer frame exceeds its size limit.",
    ErrorCode.INVALID_JSON: "The Canonical writer message is not valid canonical JSON.",
    ErrorCode.INVALID_REQUEST: "The Canonical writer request has an invalid shape.",
    ErrorCode.UNSUPPORTED_VERSION: "The Canonical writer protocol version is unsupported.",
    ErrorCode.INVALID_REQUEST_ID: "The Canonical writer request ID is invalid.",
    ErrorCode.INVALID_DEADLINE: "The Canonical writer request deadline is invalid.",
    ErrorCode.DEADLINE_EXPIRED: "The Canonical writer request deadline has expired.",
    ErrorCode.DEADLINE_TOO_FAR: "The Canonical writer request deadline is too far away.",
    ErrorCode.UNKNOWN_OPERATION: "The Canonical writer operation is not allowed.",
    ErrorCode.REPLAYED_REQUEST: "The Canonical writer request was already observed.",
    ErrorCode.UNAUTHORIZED_PEER: "The Canonical writer peer is not authorized.",
    ErrorCode.DISPATCH_UNAVAILABLE: "The Canonical writer operation is unavailable.",
    ErrorCode.DISPATCH_FAILED: "The Canonical writer operation failed.",
    ErrorCode.INTERNAL_ERROR: "The Canonical writer encountered an internal error.",
    ErrorCode.CONNECTION_CLOSED: "The Canonical writer connection closed unexpectedly.",
    ErrorCode.TIMEOUT: "The Canonical writer request timed out.",
    ErrorCode.TRANSPORT_ERROR: "The Canonical writer transport failed.",
    ErrorCode.RESPONSE_MISMATCH: "The Canonical writer response does not match the request.",
    ErrorCode.INVALID_RESPONSE: "The Canonical writer response has an invalid shape.",
}

SUCCESS_STATUSES = frozenset(
    {"ok", "inserted", "deduplicated", "conflict", "blocked", "unavailable"}
)


class ProtocolError(ValueError):
    """A safe protocol failure with a stable public code and message."""

    def __init__(self, code: ErrorCode, *, fatal: bool = False) -> None:
        self.code = code
        self.public_message = ERROR_MESSAGES[code]
        self.fatal = fatal
        super().__init__(f"{code.value}: {self.public_message}")


@dataclass(frozen=True)
class WriterRequest:
    protocol: str
    request_id: str
    sequence: int
    operation: CanonicalWriterOperation
    deadline_unix_ms: int
    runtime: Mapping[str, Any]
    payload: Mapping[str, Any]
    idempotency_key: str | None = None

    def to_message(self) -> dict[str, Any]:
        message: dict[str, Any] = {
            "protocol": self.protocol,
            "request_id": self.request_id,
            "sequence": self.sequence,
            "operation": self.operation.value,
            "deadline_unix_ms": self.deadline_unix_ms,
            "runtime": dict(self.runtime),
            "payload": dict(self.payload),
        }
        if self.idempotency_key is not None:
            message["idempotency_key"] = self.idempotency_key
        return message


@dataclass(frozen=True)
class WriterResponse:
    protocol: str
    request_id: str
    ok: bool
    status: str | None
    result: Mapping[str, Any] | None
    error_code: ErrorCode | None
    error_message: str | None
    retryable: bool


def _reject_non_json_constant(value: str) -> None:
    raise ValueError(f"non-JSON numeric constant: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def canonical_json_bytes(message: Mapping[str, Any]) -> bytes:
    """Serialize an object deterministically, rejecting NaN and infinity."""

    try:
        return json.dumps(
            dict(message),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProtocolError(ErrorCode.INVALID_JSON) from exc


def decode_json_object(body: bytes) -> dict[str, Any]:
    """Decode one strict UTF-8 JSON object without duplicate keys."""

    try:
        text = body.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProtocolError(ErrorCode.INVALID_JSON) from exc
    if not isinstance(value, dict):
        raise ProtocolError(ErrorCode.INVALID_JSON)
    return value


def encode_frame(message: Mapping[str, Any], *, max_bytes: int) -> bytes:
    """Encode one canonical JSON message with a four-byte network-order size."""

    body = canonical_json_bytes(message)
    if not body:
        raise ProtocolError(ErrorCode.MALFORMED_FRAME)
    if len(body) > max_bytes:
        raise ProtocolError(ErrorCode.FRAME_TOO_LARGE, fatal=True)
    return _FRAME_HEADER.pack(len(body)) + body


def _receive_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            code = ErrorCode.CONNECTION_CLOSED if not chunks else ErrorCode.MALFORMED_FRAME
            raise ProtocolError(code, fatal=True)
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def receive_message(sock: socket.socket, *, max_bytes: int) -> dict[str, Any]:
    """Receive and decode one bounded frame from a stream socket."""

    header = _receive_exact(sock, _FRAME_HEADER.size)
    (body_size,) = _FRAME_HEADER.unpack(header)
    if body_size == 0:
        raise ProtocolError(ErrorCode.MALFORMED_FRAME, fatal=True)
    if body_size > max_bytes:
        raise ProtocolError(ErrorCode.FRAME_TOO_LARGE, fatal=True)
    return decode_json_object(_receive_exact(sock, body_size))


def send_message(sock: socket.socket, message: Mapping[str, Any], *, max_bytes: int) -> None:
    sock.sendall(encode_frame(message, max_bytes=max_bytes))


def _canonical_request_id(value: Any) -> str:
    if not isinstance(value, str):
        raise ProtocolError(ErrorCode.INVALID_REQUEST_ID)
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ProtocolError(ErrorCode.INVALID_REQUEST_ID) from exc
    if parsed.int == 0 or str(parsed) != value:
        raise ProtocolError(ErrorCode.INVALID_REQUEST_ID)
    return value


def _request_operation(value: Any) -> CanonicalWriterOperation:
    if not isinstance(value, str):
        raise ProtocolError(ErrorCode.UNKNOWN_OPERATION)
    try:
        return CanonicalWriterOperation(value)
    except ValueError as exc:
        raise ProtocolError(ErrorCode.UNKNOWN_OPERATION) from exc


def parse_request(
    message: Mapping[str, Any],
    *,
    now: float | None = None,
) -> WriterRequest:
    """Validate a request strictly and return its typed representation."""

    keys = frozenset(message)
    if not _REQUEST_REQUIRED_FIELDS.issubset(keys) or not keys.issubset(
        _REQUEST_REQUIRED_FIELDS | _REQUEST_OPTIONAL_FIELDS
    ):
        raise ProtocolError(ErrorCode.INVALID_REQUEST)
    if message.get("protocol") != PROTOCOL_VERSION:
        raise ProtocolError(ErrorCode.UNSUPPORTED_VERSION)

    request_id = _canonical_request_id(message.get("request_id"))
    sequence = message.get("sequence")
    if isinstance(sequence, bool) or not isinstance(sequence, int):
        raise ProtocolError(ErrorCode.INVALID_REQUEST)
    if sequence < 1 or sequence > MAX_SEQUENCE:
        raise ProtocolError(ErrorCode.INVALID_REQUEST)

    operation = _request_operation(message.get("operation"))
    deadline_unix_ms = message.get("deadline_unix_ms")
    if isinstance(deadline_unix_ms, bool) or not isinstance(deadline_unix_ms, int):
        raise ProtocolError(ErrorCode.INVALID_DEADLINE)
    current_ms = int((time.time() if now is None else now) * 1000)
    if deadline_unix_ms <= current_ms:
        raise ProtocolError(ErrorCode.DEADLINE_EXPIRED)
    if deadline_unix_ms > current_ms + int(MAX_DEADLINE_SECONDS * 1000):
        raise ProtocolError(ErrorCode.DEADLINE_TOO_FAR)

    payload = message.get("payload")
    if not isinstance(payload, dict):
        raise ProtocolError(ErrorCode.INVALID_REQUEST)
    if RESERVED_PAYLOAD_FIELDS.intersection(payload):
        raise ProtocolError(ErrorCode.INVALID_REQUEST)

    runtime = message.get("runtime")
    if not isinstance(runtime, dict) or PEER_IDENTITY_FIELDS.intersection(runtime):
        raise ProtocolError(ErrorCode.INVALID_REQUEST)

    idempotency_key = message.get("idempotency_key")
    if idempotency_key is not None:
        if not isinstance(idempotency_key, str) or not idempotency_key:
            raise ProtocolError(ErrorCode.INVALID_REQUEST)
        if len(idempotency_key.encode("utf-8")) > MAX_IDEMPOTENCY_KEY_BYTES:
            raise ProtocolError(ErrorCode.INVALID_REQUEST)

    return WriterRequest(
        protocol=PROTOCOL_VERSION,
        request_id=request_id,
        sequence=sequence,
        operation=operation,
        deadline_unix_ms=deadline_unix_ms,
        runtime=dict(runtime),
        payload=dict(payload),
        idempotency_key=idempotency_key,
    )


def make_request(
    operation: CanonicalWriterOperation | str,
    payload: Mapping[str, Any],
    *,
    runtime: Mapping[str, Any],
    sequence: int,
    timeout_seconds: float,
    request_id: str | None = None,
    idempotency_key: str | None = None,
    now: float | None = None,
) -> WriterRequest:
    """Build and validate a request with a bounded absolute deadline."""

    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
        raise ProtocolError(ErrorCode.INVALID_DEADLINE)
    if timeout_seconds <= 0 or timeout_seconds > MAX_DEADLINE_SECONDS:
        raise ProtocolError(ErrorCode.INVALID_DEADLINE)
    current = time.time() if now is None else now
    operation_value = operation.value if isinstance(operation, CanonicalWriterOperation) else operation
    message: dict[str, Any] = {
        "protocol": PROTOCOL_VERSION,
        "request_id": request_id or str(uuid.uuid4()),
        "sequence": sequence,
        "operation": operation_value,
        "deadline_unix_ms": int((current + timeout_seconds) * 1000),
        "runtime": dict(runtime),
        "payload": dict(payload),
    }
    if idempotency_key is not None:
        message["idempotency_key"] = idempotency_key
    return parse_request(message, now=current)


def make_success_response(
    request_id: str,
    *,
    status: str,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    if status not in SUCCESS_STATUSES:
        raise ProtocolError(ErrorCode.INVALID_RESPONSE)
    return {
        "protocol": PROTOCOL_VERSION,
        "request_id": _canonical_request_id(request_id),
        "ok": True,
        "status": status,
        "result": dict(result),
    }


def make_error_response(
    request_id: str,
    code: ErrorCode,
    *,
    retryable: bool = False,
) -> dict[str, Any]:
    if request_id != UNKNOWN_REQUEST_ID:
        _canonical_request_id(request_id)
    return {
        "protocol": PROTOCOL_VERSION,
        "request_id": request_id,
        "ok": False,
        "error": {
            "code": code.value,
            "message": ERROR_MESSAGES[code],
            "retryable": bool(retryable),
        },
    }


def parse_response(message: Mapping[str, Any]) -> WriterResponse:
    """Strictly validate a writer response."""

    if message.get("protocol") != PROTOCOL_VERSION:
        raise ProtocolError(ErrorCode.INVALID_RESPONSE)
    request_id = message.get("request_id")
    if request_id != UNKNOWN_REQUEST_ID:
        try:
            request_id = _canonical_request_id(request_id)
        except ProtocolError as exc:
            raise ProtocolError(ErrorCode.INVALID_RESPONSE) from exc
    if not isinstance(message.get("ok"), bool):
        raise ProtocolError(ErrorCode.INVALID_RESPONSE)

    if message["ok"]:
        if frozenset(message) != frozenset(
            {"protocol", "request_id", "ok", "status", "result"}
        ):
            raise ProtocolError(ErrorCode.INVALID_RESPONSE)
        status = message.get("status")
        result = message.get("result")
        if status not in SUCCESS_STATUSES or not isinstance(result, dict):
            raise ProtocolError(ErrorCode.INVALID_RESPONSE)
        return WriterResponse(
            protocol=PROTOCOL_VERSION,
            request_id=request_id,
            ok=True,
            status=status,
            result=dict(result),
            error_code=None,
            error_message=None,
            retryable=False,
        )

    if frozenset(message) != frozenset({"protocol", "request_id", "ok", "error"}):
        raise ProtocolError(ErrorCode.INVALID_RESPONSE)
    error = message.get("error")
    if not isinstance(error, dict) or frozenset(error) != frozenset(
        {"code", "message", "retryable"}
    ):
        raise ProtocolError(ErrorCode.INVALID_RESPONSE)
    try:
        code = ErrorCode(error.get("code"))
    except (TypeError, ValueError) as exc:
        raise ProtocolError(ErrorCode.INVALID_RESPONSE) from exc
    if error.get("message") != ERROR_MESSAGES[code] or not isinstance(
        error.get("retryable"), bool
    ):
        raise ProtocolError(ErrorCode.INVALID_RESPONSE)
    return WriterResponse(
        protocol=PROTOCOL_VERSION,
        request_id=request_id,
        ok=False,
        status=None,
        result=None,
        error_code=code,
        error_message=ERROR_MESSAGES[code],
        retryable=error["retryable"],
    )
