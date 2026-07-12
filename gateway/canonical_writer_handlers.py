"""Typed privileged dispatch boundary for the Canonical Writer service.

The wire protocol authenticates a peer and constructs :class:`RuntimeContext`.
This module accepts that trusted envelope separately from caller payloads,
validates exact operation schemas, and invokes a transactional backend.  It
contains no raw-SQL operation and makes no semantic/routing decisions.

``InMemoryCanonicalWriterBackend`` is a reference/test backend.  Production
backends must provide the same atomic capability and route-back methods over a
durable store; handlers never emulate those transactions outside the backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
import datetime as dt
import hashlib
import json
import re
import threading
import uuid
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from gateway.canonical_writer_protocol import (
    MAX_IDEMPOTENCY_KEY_BYTES,
    CanonicalWriterOperation,
    WriterRequest,
)


OP_PING = CanonicalWriterOperation.PING.value
OP_EVENT_APPEND_MODEL = CanonicalWriterOperation.EVENT_APPEND_MODEL.value
OP_PLAN_TRANSITION = CanonicalWriterOperation.PLAN_TRANSITION.value
OP_VERIFICATION_APPEND = CanonicalWriterOperation.VERIFICATION_APPEND.value
OP_CASE_QUERY = CanonicalWriterOperation.CASE_QUERY.value
OP_PLAN_ACTIVE_MATCH = CanonicalWriterOperation.PLAN_ACTIVE_MATCH.value
OP_ROUTEBACK_CONTEXT = CanonicalWriterOperation.ROUTEBACK_CONTEXT.value
OP_ROUTEBACK_CLAIM = CanonicalWriterOperation.ROUTEBACK_CLAIM.value
OP_ROUTEBACK_FINALIZE_SENT = CanonicalWriterOperation.ROUTEBACK_FINALIZE_SENT.value
OP_ROUTEBACK_FINALIZE_BLOCKED = CanonicalWriterOperation.ROUTEBACK_FINALIZE_BLOCKED.value
OP_LEASE_SHADOW_RECORD = CanonicalWriterOperation.LEASE_SHADOW_RECORD.value
OP_CAPABILITY_GRANT = CanonicalWriterOperation.CAPABILITY_GRANT.value
OP_CAPABILITY_CONSUME = CanonicalWriterOperation.CAPABILITY_CONSUME.value
OP_CAPABILITY_REVOKE = CanonicalWriterOperation.CAPABILITY_REVOKE.value
OP_CAPABILITY_REVOKE_SESSION = CanonicalWriterOperation.CAPABILITY_REVOKE_SESSION.value
OP_PROJECTION_READ_EVENTS = CanonicalWriterOperation.PROJECTION_READ_EVENTS.value

# Backward-compatible Python names only; wire values always come from the
# centralized CanonicalWriterOperation enum.
OP_EVENT_APPEND = OP_EVENT_APPEND_MODEL
OP_QUERY = OP_CASE_QUERY
OP_ROUTEBACK_AUTHORIZE = OP_ROUTEBACK_CLAIM
OP_PROJECTOR_READ = OP_PROJECTION_READ_EVENTS

SUPPORTED_OPERATIONS = frozenset(item.value for item in CanonicalWriterOperation)

WRITER_OWNED_EVENT_TYPES = frozenset({
    "task.plan.updated",
    "task.verification.recorded",
    "route_back.intent.created",
    "route_back.sent",
    "route_back.blocked",
    "approval.capability.recorded",
    "approval.capability.revoked",
    "approval.capability.session_revoked",
    "capability.check.recorded",
    "lease.shadow.recorded",
})
# Compatibility name for downstream imports.  This is deliberately the full
# fixed-writer namespace: model append must never occupy a deterministic event
# identity later needed by a typed writer operation.
MODEL_FORBIDDEN_EVENT_TYPES = WRITER_OWNED_EVENT_TYPES

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RUNTIME_PAYLOAD_KEYS = frozenset({
    "runtime",
    "runtime_context",
    "trusted_runtime",
    "session_key_sha256",
    "capability_epoch_sha256",
    "observed_session",
})
_MAX_PAYLOAD_BYTES = 128_000
_MAX_TEXT = 4_000
_MAX_COMMANDS = 64
_MAX_USES = 1_000
_FORBIDDEN_DM_KEYS = {
    "dm_channel_id",
    "direct_message_channel_id",
    "recipient_id",
    "dm_recipient_id",
}
_FORBIDDEN_DM_VALUES = {
    "dm",
    "direct_message",
    "private_dm",
    "user_dm",
    "private",
    "group",
    "group_dm",
    "private_channel",
    "private_thread",
}


class CanonicalWriterError(Exception):
    """Bounded typed request/backend failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message[:1_000]


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _json_copy(value: Any) -> Any:
    return json.loads(_stable_json(value))


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CanonicalWriterError("invalid_request", f"{path} must be an object")
    return dict(value)


def _strict_payload(
    payload: Any,
    *,
    allowed: set[str],
    required: set[str] = frozenset(),
) -> dict[str, Any]:
    value = _require_mapping(payload, "payload")
    encoded = _stable_json(value).encode("utf-8")
    if len(encoded) > _MAX_PAYLOAD_BYTES:
        raise CanonicalWriterError("invalid_request", "payload exceeds bounded size")
    if _RUNTIME_PAYLOAD_KEYS.intersection(value):
        raise CanonicalWriterError(
            "runtime_override_forbidden",
            "trusted runtime context must not appear in caller payload",
        )
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise CanonicalWriterError(
            "invalid_request",
            "unknown payload fields:" + ",".join(unknown),
        )
    missing = sorted(required - set(value))
    if missing:
        raise CanonicalWriterError(
            "invalid_request",
            "missing payload fields:" + ",".join(missing),
        )
    return value


def _text(value: Any, path: str, *, maximum: int = _MAX_TEXT) -> str:
    text = str(value or "").strip()
    if not text or len(text) > maximum or any(ord(char) < 32 for char in text):
        raise CanonicalWriterError("invalid_request", f"{path} is invalid")
    return text


def _identifier(value: Any, path: str) -> str:
    text = _text(value, path, maximum=240)
    if not _ID_RE.fullmatch(text):
        raise CanonicalWriterError("invalid_request", f"{path} is not a safe identifier")
    return text


def _case_identifier(value: Any, path: str) -> str:
    text = _identifier(value, path)
    if not text.startswith("case:"):
        raise CanonicalWriterError("invalid_request", f"{path} must start with case:")
    return text


def _sha256(value: Any, path: str) -> str:
    text = str(value or "").strip().lower()
    if not _SHA256_RE.fullmatch(text):
        raise CanonicalWriterError("invalid_request", f"{path} must be a sha256 digest")
    return text


def _idempotency_key(value: Any, path: str) -> str:
    text = _text(value, path, maximum=MAX_IDEMPOTENCY_KEY_BYTES)
    if len(text.encode("utf-8")) > MAX_IDEMPOTENCY_KEY_BYTES:
        raise CanonicalWriterError(
            "invalid_request",
            f"{path} exceeds the protocol byte bound",
        )
    return text


def _positive_int(value: Any, path: str, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise CanonicalWriterError("invalid_request", f"{path} is out of bounds")
    return value


def _utc(value: Any, path: str) -> dt.datetime:
    try:
        parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        raise CanonicalWriterError("invalid_request", f"{path} must be ISO-8601") from None
    if parsed.tzinfo is None:
        raise CanonicalWriterError("invalid_request", f"{path} must include a timezone")
    return parsed.astimezone(dt.timezone.utc)


def _contains_forbidden_dm_ref(value: Any) -> bool:
    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key or "").strip().casefold().replace("-", "_")
            if key in _FORBIDDEN_DM_KEYS and nested not in (None, "", False, [], {}):
                return True
            if key in {
                "channel_type",
                "target_type",
                "target_kind",
                "delivery_type",
                "lane",
                "role",
            }:
                normalized = str(nested or "").strip().casefold()
                if normalized in _FORBIDDEN_DM_VALUES or normalized.endswith("_dm"):
                    return True
            if _contains_forbidden_dm_ref(nested):
                return True
        return False
    if isinstance(value, (list, tuple, set)):
        return any(_contains_forbidden_dm_ref(item) for item in value)
    return False


@dataclass(frozen=True)
class RuntimeContext:
    """Trusted server-owned envelope; never decoded from an operation payload."""

    request_id: str
    platform: str = ""
    session_key_sha256: str = ""
    capability_epoch_sha256: str = ""
    user_id: str = ""
    chat_id: str = ""
    thread_id: str = ""
    message_id: str = ""
    owner_authenticated: bool = False
    service_internal: bool = False

    def __post_init__(self) -> None:
        _identifier(self.request_id, "runtime.request_id")
        if self.platform:
            _identifier(self.platform, "runtime.platform")
        if self.session_key_sha256:
            _sha256(self.session_key_sha256, "runtime.session_key_sha256")
        if self.capability_epoch_sha256:
            _sha256(
                self.capability_epoch_sha256,
                "runtime.capability_epoch_sha256",
            )
        if type(self.owner_authenticated) is not bool:
            raise CanonicalWriterError(
                "invalid_runtime",
                "runtime.owner_authenticated must be boolean",
            )
        if type(self.service_internal) is not bool:
            raise CanonicalWriterError(
                "invalid_runtime",
                "runtime.service_internal must be boolean",
            )
        for name in ("user_id", "chat_id", "thread_id", "message_id"):
            value = str(getattr(self, name) or "")
            if len(value) > 240 or any(ord(char) < 32 for char in value):
                raise CanonicalWriterError("invalid_runtime", f"runtime.{name} is invalid")

    def observed_session(self) -> dict[str, str]:
        result = {}
        if self.platform:
            result["platform"] = self.platform
        if self.session_key_sha256:
            result["session_key_sha256"] = self.session_key_sha256
        if self.capability_epoch_sha256:
            result["capability_epoch_sha256"] = self.capability_epoch_sha256
        for key in ("user_id", "chat_id", "thread_id", "message_id"):
            value = str(getattr(self, key) or "")
            if value:
                result[key] = value
        return result


def _require_exact_runtime_epoch(runtime: RuntimeContext) -> None:
    """Require the exact gateway session generation for initiating mutations."""

    if (
        not _SHA256_RE.fullmatch(runtime.session_key_sha256)
        or not _SHA256_RE.fullmatch(runtime.capability_epoch_sha256)
    ):
        raise CanonicalWriterError(
            "invalid_runtime",
            "initiating mutation requires exact session and routing-epoch binding",
        )


@dataclass(frozen=True)
class EventAppendRequest:
    event_type: str
    case_id: str
    summary: str
    source_refs: Mapping[str, Any]
    actors: Mapping[str, Any]
    body: Mapping[str, Any]
    safety: Mapping[str, Any]
    idempotency_key: str


@dataclass(frozen=True)
class QueryRequest:
    case_id: str
    thread_id: str
    limit: int
    view: str


@dataclass(frozen=True)
class PlanActiveMatchRequest:
    case_id: str
    plan_id: str


@dataclass(frozen=True)
class RouteBackContextRequest:
    thread_id: str


@dataclass(frozen=True)
class RouteBackAuthorizeRequest:
    case_id: str
    target_ref: Mapping[str, Any]
    message_summary: str
    source_refs: Mapping[str, Any]
    content_sha256: str
    idempotency_key: str


@dataclass(frozen=True)
class RouteBackTerminalRequest:
    authorization_id: str
    outcome: str
    receipt: Mapping[str, Any]
    blocker_reason: str
    preclaim: bool = False
    case_id: str = ""
    target_ref: Mapping[str, Any] = field(default_factory=dict)
    message_summary: str = ""
    source_refs: Mapping[str, Any] = field(default_factory=dict)
    idempotency_key: str = ""


@dataclass(frozen=True)
class CapabilityGrantRequest:
    approval_id: str
    case_id: str
    plan_id: str
    plan_revision: int
    approval_source_sha256: str
    command_hashes: tuple[str, ...]
    expires_at: dt.datetime
    max_uses: int


@dataclass(frozen=True)
class CapabilityConsumeRequest:
    command_sha256: str
    idempotency_key: str


@dataclass(frozen=True)
class CapabilityRevokeRequest:
    plan_id: str
    reason: str


@dataclass(frozen=True)
class ProjectorReadRequest:
    case_id: str
    after_event_id: str
    limit: int


@runtime_checkable
class CanonicalWriterBackend(Protocol):
    """Production implementations make every mutation durable and atomic."""

    def ping(self, runtime: RuntimeContext) -> Mapping[str, Any]: ...
    def event_append(self, request: EventAppendRequest, runtime: RuntimeContext) -> Mapping[str, Any]: ...
    def query(self, request: QueryRequest, runtime: RuntimeContext) -> Mapping[str, Any]: ...
    def plan_active_match(self, request: PlanActiveMatchRequest, runtime: RuntimeContext) -> Mapping[str, Any]: ...
    def routeback_context(self, request: RouteBackContextRequest, runtime: RuntimeContext) -> Mapping[str, Any]: ...
    def routeback_authorize(self, request: RouteBackAuthorizeRequest, runtime: RuntimeContext) -> Mapping[str, Any]: ...
    def routeback_terminal(self, request: RouteBackTerminalRequest, runtime: RuntimeContext) -> Mapping[str, Any]: ...
    def lease_shadow_record(self, payload: Mapping[str, Any], runtime: RuntimeContext) -> Mapping[str, Any]: ...
    def capability_grant(self, request: CapabilityGrantRequest, runtime: RuntimeContext) -> Mapping[str, Any]: ...
    def capability_consume(self, request: CapabilityConsumeRequest, runtime: RuntimeContext) -> Mapping[str, Any]: ...
    def capability_revoke(self, request: CapabilityRevokeRequest, runtime: RuntimeContext) -> Mapping[str, Any]: ...
    def capability_revoke_session(self, session_key_sha256: str, reason: str, runtime: RuntimeContext) -> Mapping[str, Any]: ...
    def projector_read(self, request: ProjectorReadRequest, runtime: RuntimeContext) -> Mapping[str, Any]: ...


# Exported strict payload shapes for the protocol/service handshake and tests.
REQUEST_SCHEMAS: Mapping[str, Mapping[str, frozenset[str]]] = {
    OP_PING: {"allowed": frozenset(), "required": frozenset()},
    OP_EVENT_APPEND_MODEL: {
        "allowed": frozenset({"event_type", "case_id", "summary", "source_refs", "actors", "payload", "safety", "idempotency_key"}),
        "required": frozenset({"event_type", "case_id", "summary", "source_refs"}),
    },
    OP_PLAN_TRANSITION: {
        "allowed": frozenset({"case_id", "summary", "source_refs", "actors", "payload", "safety", "idempotency_key"}),
        "required": frozenset({"case_id", "summary", "source_refs", "payload"}),
    },
    OP_VERIFICATION_APPEND: {
        "allowed": frozenset({"case_id", "summary", "source_refs", "actors", "payload", "safety", "idempotency_key"}),
        "required": frozenset({"case_id", "summary", "source_refs", "payload"}),
    },
    OP_CASE_QUERY: {
        "allowed": frozenset({"case_id", "thread_id", "limit", "view"}),
        "required": frozenset(),
    },
    OP_PLAN_ACTIVE_MATCH: {
        "allowed": frozenset({"case_id", "plan_id"}),
        "required": frozenset({"case_id", "plan_id"}),
    },
    OP_ROUTEBACK_CONTEXT: {
        "allowed": frozenset({"thread_id"}),
        "required": frozenset({"thread_id"}),
    },
    OP_ROUTEBACK_CLAIM: {
        "allowed": frozenset({"case_id", "target_ref", "message_summary", "source_refs", "idempotency_key", "execution_binding"}),
        "required": frozenset({"case_id", "target_ref", "message_summary", "source_refs", "idempotency_key", "execution_binding"}),
    },
    OP_ROUTEBACK_FINALIZE_SENT: {
        "allowed": frozenset({"case_id", "target_ref", "message_summary", "source_refs", "receipt", "idempotency_key", "execution_binding"}),
        "required": frozenset({"case_id", "target_ref", "message_summary", "source_refs", "receipt", "idempotency_key", "execution_binding"}),
    },
    OP_ROUTEBACK_FINALIZE_BLOCKED: {
        "allowed": frozenset({"case_id", "target_ref", "message_summary", "source_refs", "blocker_reason", "partial_receipt", "idempotency_key", "execution_binding", "preclaim"}),
        "required": frozenset({"case_id", "target_ref", "message_summary", "source_refs", "blocker_reason", "idempotency_key"}),
    },
    OP_LEASE_SHADOW_RECORD: {
        "allowed": frozenset({"intent_event_id", "intent_kind", "case", "runtime_lease_enforcement", "enforcement_enabled", "send_path_blocking_enabled", "audit_runtime_id", "source_platform", "session_key_ref"}),
        "required": frozenset({"intent_event_id", "intent_kind", "case", "runtime_lease_enforcement", "enforcement_enabled", "send_path_blocking_enabled", "audit_runtime_id", "source_platform", "session_key_ref"}),
    },
    OP_CAPABILITY_GRANT: {
        "allowed": frozenset({"approval_id", "case_id", "plan_id", "plan_revision", "approval_source_sha256", "command_hashes", "expires_at", "max_uses"}),
        "required": frozenset({"approval_id", "case_id", "plan_id", "plan_revision", "approval_source_sha256", "command_hashes", "expires_at", "max_uses"}),
    },
    OP_CAPABILITY_CONSUME: {
        "allowed": frozenset({"command_sha256", "idempotency_key"}),
        "required": frozenset({"command_sha256", "idempotency_key"}),
    },
    OP_CAPABILITY_REVOKE: {
        "allowed": frozenset({"plan_id", "reason"}),
        "required": frozenset({"plan_id", "reason"}),
    },
    OP_CAPABILITY_REVOKE_SESSION: {
        "allowed": frozenset({"reason"}),
        "required": frozenset({"reason"}),
    },
    OP_PROJECTION_READ_EVENTS: {
        "allowed": frozenset({"case_id", "after_event_id", "limit"}),
        "required": frozenset({"case_id"}),
    },
}


class CanonicalWriterHandlers:
    """Exact typed dispatcher used by the authenticated writer service."""

    def __init__(self, backend: CanonicalWriterBackend):
        self.backend = backend
        self._handlers: Mapping[str, Callable[[dict[str, Any], RuntimeContext], Mapping[str, Any]]] = {
            OP_PING: self._ping,
            OP_EVENT_APPEND_MODEL: self._event_append,
            OP_PLAN_TRANSITION: lambda payload, runtime: self._typed_event_append(
                "task.plan.updated", payload, runtime
            ),
            OP_VERIFICATION_APPEND: lambda payload, runtime: self._typed_event_append(
                "task.verification.recorded", payload, runtime
            ),
            OP_CASE_QUERY: self._query,
            OP_PLAN_ACTIVE_MATCH: self._plan_active_match,
            OP_ROUTEBACK_CONTEXT: self._routeback_context,
            OP_ROUTEBACK_CLAIM: self._routeback_authorize,
            OP_ROUTEBACK_FINALIZE_SENT: lambda payload, runtime: self._routeback_terminal(
                payload, runtime, outcome="sent"
            ),
            OP_ROUTEBACK_FINALIZE_BLOCKED: lambda payload, runtime: self._routeback_terminal(
                payload, runtime, outcome="blocked"
            ),
            OP_LEASE_SHADOW_RECORD: self._lease_shadow_record,
            OP_CAPABILITY_GRANT: self._capability_grant,
            OP_CAPABILITY_CONSUME: self._capability_consume,
            OP_CAPABILITY_REVOKE: self._capability_revoke,
            OP_CAPABILITY_REVOKE_SESSION: self._capability_revoke_session,
            OP_PROJECTION_READ_EVENTS: self._projector_read,
        }

    def dispatch(
        self,
        operation: str,
        payload: Mapping[str, Any],
        *,
        runtime: RuntimeContext,
    ) -> dict[str, Any]:
        op = str(operation or "")
        if type(runtime) is not RuntimeContext:
            return self._error(op, "trusted_runtime_required", "server-owned RuntimeContext is required")
        handler = self._handlers.get(op)
        if handler is None:
            return self._error(op, "unsupported_operation", "operation is not supported")
        schema = REQUEST_SCHEMAS[op]
        try:
            strict = _strict_payload(
                payload,
                allowed=set(schema["allowed"]),
                required=set(schema["required"]),
            )
            result = dict(handler(strict, runtime))
            return {"ok": True, "operation": op, "result": result}
        except CanonicalWriterError as exc:
            return self._error(op, exc.code, exc.message)
        except Exception:
            return self._error(op, "backend_failure", "privileged backend operation failed")

    def dispatch_request(
        self,
        request: WriterRequest,
        *,
        runtime: RuntimeContext,
    ) -> dict[str, Any]:
        """Dispatch a parsed protocol request without trusting its runtime map."""
        payload = dict(request.payload)
        if request.idempotency_key and "idempotency_key" not in payload:
            payload["idempotency_key"] = request.idempotency_key
        return self.dispatch(request.operation.value, payload, runtime=runtime)

    @staticmethod
    def _error(operation: str, code: str, message: str) -> dict[str, Any]:
        return {
            "ok": False,
            "operation": operation,
            "error": {"code": code, "message": str(message)[:1_000]},
        }

    def _ping(self, payload: dict[str, Any], runtime: RuntimeContext) -> Mapping[str, Any]:
        return self.backend.ping(runtime)

    def _event_append(self, p: dict[str, Any], runtime: RuntimeContext) -> Mapping[str, Any]:
        _require_exact_runtime_epoch(runtime)
        event_type = _identifier(p["event_type"], "payload.event_type")
        if event_type in MODEL_FORBIDDEN_EVENT_TYPES:
            raise CanonicalWriterError(
                "privileged_event_forbidden",
                f"{event_type} requires its typed privileged operation",
            )
        request = self._validated_event_request(EventAppendRequest(
            event_type=event_type,
            case_id=_case_identifier(p["case_id"], "payload.case_id"),
            summary=_text(p["summary"], "payload.summary"),
            source_refs=_require_mapping(p["source_refs"], "payload.source_refs"),
            actors=_require_mapping(p.get("actors") or {}, "payload.actors"),
            body=_require_mapping(p.get("payload") or {}, "payload.payload"),
            safety=_require_mapping(p.get("safety") or {}, "payload.safety"),
            idempotency_key=_idempotency_key(
                p.get("idempotency_key") or _digest(p),
                "payload.idempotency_key",
            ),
        ), runtime)
        return self.backend.event_append(request, runtime)

    def _typed_event_append(
        self,
        event_type: str,
        p: dict[str, Any],
        runtime: RuntimeContext,
    ) -> Mapping[str, Any]:
        _require_exact_runtime_epoch(runtime)
        request = self._validated_event_request(EventAppendRequest(
            event_type=event_type,
            case_id=_case_identifier(p["case_id"], "payload.case_id"),
            summary=_text(p["summary"], "payload.summary"),
            source_refs=_require_mapping(p["source_refs"], "payload.source_refs"),
            actors=_require_mapping(p.get("actors") or {}, "payload.actors"),
            body=_require_mapping(p["payload"], "payload.payload"),
            safety=_require_mapping(p.get("safety") or {}, "payload.safety"),
            idempotency_key=_idempotency_key(
                p.get("idempotency_key") or _digest(p),
                "payload.idempotency_key",
            ),
        ), runtime)
        return self.backend.event_append(request, runtime)

    @staticmethod
    def _validated_event_request(
        request: EventAppendRequest,
        runtime: RuntimeContext,
    ) -> EventAppendRequest:
        source_refs = dict(request.source_refs)
        if runtime.platform:
            source_refs.setdefault("platform", runtime.platform)
        if runtime.thread_id:
            source_refs.setdefault("thread_id", runtime.thread_id)
        if runtime.chat_id:
            source_refs.setdefault("chat_id", runtime.chat_id)
        if runtime.message_id:
            source_refs.setdefault("message_id", runtime.message_id)
        if not any(source_refs.get(key) for key in ("message_id", "event_ref", "manual_ref")):
            source_refs["manual_ref"] = f"writer_request:{runtime.request_id}"
        try:
            from tools.canonical_brain_tool import (
                _block_secret_like_fields,
                _validate_append_request,
            )

            _validate_append_request(
                event_type=request.event_type,
                case_id=request.case_id,
                summary=request.summary,
                source_refs=source_refs,
                actors=dict(request.actors),
                payload=dict(request.body),
                safety=dict(request.safety),
            )
            _block_secret_like_fields(
                summary=request.summary,
                source_refs=source_refs,
                actors=dict(request.actors),
                payload=dict(request.body),
                safety=dict(request.safety),
            )
        except (TypeError, ValueError) as exc:
            raise CanonicalWriterError("invalid_event", str(exc)) from None
        return EventAppendRequest(
            event_type=request.event_type,
            case_id=request.case_id,
            summary=request.summary,
            source_refs=source_refs,
            actors=request.actors,
            body=request.body,
            safety=request.safety,
            idempotency_key=request.idempotency_key,
        )

    def _query(self, p: dict[str, Any], runtime: RuntimeContext) -> Mapping[str, Any]:
        case_id = str(p.get("case_id") or "").strip()
        thread_id = str(p.get("thread_id") or "").strip()
        if bool(case_id) == bool(thread_id):
            raise CanonicalWriterError("invalid_request", "provide exactly one of case_id or thread_id")
        request = QueryRequest(
            case_id=_case_identifier(case_id, "payload.case_id") if case_id else "",
            thread_id=_identifier(thread_id, "payload.thread_id") if thread_id else "",
            limit=_positive_int(p.get("limit", 80), "payload.limit", maximum=200),
            view=str(p.get("view") or "summary"),
        )
        if request.view not in {"summary", "resume_bundle"}:
            raise CanonicalWriterError("invalid_request", "payload.view is invalid")
        return self.backend.query(request, runtime)

    def _plan_active_match(self, p: dict[str, Any], runtime: RuntimeContext) -> Mapping[str, Any]:
        return self.backend.plan_active_match(PlanActiveMatchRequest(
            case_id=_case_identifier(p["case_id"], "payload.case_id"),
            plan_id=_identifier(p["plan_id"], "payload.plan_id"),
        ), runtime)

    def _routeback_context(self, payload: dict[str, Any], runtime: RuntimeContext) -> Mapping[str, Any]:
        thread_id = _identifier(payload["thread_id"], "payload.thread_id")
        observed_thread = runtime.thread_id or runtime.chat_id
        if not observed_thread or observed_thread != thread_id:
            raise CanonicalWriterError("scope_mismatch", "route-back thread differs from observed runtime")
        return self.backend.routeback_context(RouteBackContextRequest(thread_id), runtime)

    def _routeback_authorize(self, p: dict[str, Any], runtime: RuntimeContext) -> Mapping[str, Any]:
        _require_exact_runtime_epoch(runtime)
        target = _require_mapping(p["target_ref"], "payload.target_ref")
        if _contains_forbidden_dm_ref(target):
            raise CanonicalWriterError("dm_forbidden", "Discord DM route-back is forbidden")
        target_id = str(target.get("thread_id") or target.get("channel_id") or "").strip()
        if not target_id:
            raise CanonicalWriterError("invalid_request", "target_ref requires public thread_id or channel_id")
        channel_type = str(target.get("channel_type") or target.get("target_type") or "").casefold()
        if channel_type in {"dm", "direct_message", "private"}:
            raise CanonicalWriterError("dm_forbidden", "Discord DM route-back is forbidden")
        binding = _require_mapping(p["execution_binding"], "payload.execution_binding")
        content_sha256 = _sha256(binding.get("content_sha256"), "payload.execution_binding.content_sha256")
        target_channel_id = _identifier(
            binding.get("target_channel_id"),
            "payload.execution_binding.target_channel_id",
        )
        if target_channel_id != target_id:
            raise CanonicalWriterError("scope_mismatch", "execution binding target differs from target_ref")
        return self.backend.routeback_authorize(RouteBackAuthorizeRequest(
            case_id=_case_identifier(p["case_id"], "payload.case_id"),
            target_ref=target,
            message_summary=_text(p["message_summary"], "payload.message_summary"),
            source_refs=_require_mapping(p["source_refs"], "payload.source_refs"),
            content_sha256=content_sha256,
            idempotency_key=_idempotency_key(
                p["idempotency_key"], "payload.idempotency_key"
            ),
        ), runtime)

    def _routeback_terminal(
        self,
        p: dict[str, Any],
        runtime: RuntimeContext,
        *,
        outcome: str,
    ) -> Mapping[str, Any]:
        preclaim = p.get("preclaim", False)
        if type(preclaim) is not bool:
            raise CanonicalWriterError(
                "invalid_request",
                "payload.preclaim must be boolean",
            )
        if outcome != "blocked" and preclaim:
            raise CanonicalWriterError(
                "invalid_request",
                "preclaim is available only for blocked finalization",
            )
        receipt_field = "receipt" if outcome == "sent" else "partial_receipt"
        receipt = _require_mapping(
            p.get(receipt_field, {}),
            f"payload.{receipt_field}",
        )
        if _contains_forbidden_dm_ref(p):
            raise CanonicalWriterError("dm_forbidden", "Discord DM route-back is forbidden")
        blocker = str(p.get("blocker_reason") or "").strip()
        if outcome == "sent" and not receipt:
            raise CanonicalWriterError("invalid_request", "sent outcome requires receipt")
        if outcome == "sent":
            allowed_receipt_keys = {
                "platform",
                "adapter_receipt",
                "receipt_readback_verified",
                "message_id",
                "channel_id",
                "chat_id",
                "channel_type",
                "target_kind",
                "content_sha256",
            }
            required_receipt_keys = {
                "platform",
                "adapter_receipt",
                "receipt_readback_verified",
                "message_id",
                "channel_id",
                "content_sha256",
            }
            if (
                not required_receipt_keys.issubset(receipt)
                or not set(receipt).issubset(allowed_receipt_keys)
                or receipt.get("platform") != "discord"
                or receipt.get("adapter_receipt") is not True
                or receipt.get("receipt_readback_verified") is not True
                or not isinstance(receipt.get("message_id"), str)
                or _identifier(
                    receipt.get("message_id"),
                    "payload.receipt.message_id",
                )
                != receipt.get("message_id")
                or not isinstance(receipt.get("channel_id"), str)
                or _identifier(
                    receipt.get("channel_id"),
                    "payload.receipt.channel_id",
                )
                != receipt.get("channel_id")
                or not isinstance(receipt.get("content_sha256"), str)
                or _sha256(
                    receipt.get("content_sha256"),
                    "payload.receipt.content_sha256",
                )
                != receipt.get("content_sha256")
                or (
                    "chat_id" in receipt
                    and receipt.get("chat_id") != receipt.get("channel_id")
                )
                or any(
                    _identifier(receipt[key], f"payload.receipt.{key}")
                    != receipt[key]
                    for key in ("channel_type", "target_kind")
                    if key in receipt
                )
            ):
                raise CanonicalWriterError(
                    "invalid_receipt",
                    "sent outcome requires an exact verified Discord adapter receipt",
                )
        if outcome == "blocked" and not blocker:
            raise CanonicalWriterError("invalid_request", "blocked outcome requires blocker_reason")
        if preclaim and (receipt or "execution_binding" in p):
            raise CanonicalWriterError(
                "invalid_request",
                "preclaim blocked finalization forbids send or authorization evidence",
            )
        if not preclaim and "execution_binding" not in p:
            raise CanonicalWriterError(
                "invalid_request",
                "claimed blocked finalization requires execution_binding",
            )
        if outcome == "blocked" and receipt:
            expected_keys = {
                "platform",
                "adapter_receipt",
                "receipt_readback_verified",
                "message_id",
                "channel_id",
                "content_sha256",
            }
            if set(receipt) != expected_keys:
                raise CanonicalWriterError(
                    "invalid_receipt",
                    "blocked partial_receipt has an invalid exact shape",
                )
            binding = _require_mapping(
                p["execution_binding"],
                "payload.execution_binding",
            )
            target = _require_mapping(p["target_ref"], "payload.target_ref")
            target_id = _identifier(
                target.get("thread_id") or target.get("channel_id"),
                "payload.target_ref.channel_id",
            )
            bound_channel_id = _identifier(
                binding.get("target_channel_id"),
                "payload.execution_binding.target_channel_id",
            )
            bound_content_sha256 = _sha256(
                binding.get("content_sha256"),
                "payload.execution_binding.content_sha256",
            )
            message_id = receipt.get("message_id")
            channel_id = receipt.get("channel_id")
            content_sha256 = receipt.get("content_sha256")
            readback_verified = receipt.get("receipt_readback_verified")
            if (
                receipt.get("platform") != "discord"
                or receipt.get("adapter_receipt") is not True
                or type(readback_verified) is not bool
                or (
                    readback_verified is True
                    and blocker
                    != "route_back_sent_receipt_persistence_failed"
                )
                or not isinstance(message_id, str)
                or _identifier(
                    message_id,
                    "payload.partial_receipt.message_id",
                )
                != message_id
                or not isinstance(channel_id, str)
                or _identifier(
                    channel_id,
                    "payload.partial_receipt.channel_id",
                )
                != target_id
                or channel_id != bound_channel_id
                or not isinstance(content_sha256, str)
                or _sha256(
                    content_sha256,
                    "payload.partial_receipt.content_sha256",
                )
                != bound_content_sha256
                or content_sha256 != bound_content_sha256
            ):
                raise CanonicalWriterError(
                    "invalid_receipt",
                    "blocked partial_receipt must exactly bind the accepted Discord send and blocker",
                )
        case_id = _case_identifier(p["case_id"], "payload.case_id")
        idempotency_key = _idempotency_key(
            p["idempotency_key"], "payload.idempotency_key"
        )
        _require_exact_runtime_epoch(runtime)
        authorization_id = "routeauth:" + _digest({
            "case_id": case_id,
            "idempotency_key": idempotency_key,
        })[:40]
        if preclaim:
            authorization_id = ""
        result = dict(self.backend.routeback_terminal(RouteBackTerminalRequest(
            authorization_id=authorization_id,
            outcome=outcome,
            receipt=receipt,
            blocker_reason=blocker,
            preclaim=preclaim,
            case_id=case_id,
            target_ref=_require_mapping(p["target_ref"], "payload.target_ref"),
            message_summary=_text(
                p["message_summary"],
                "payload.message_summary",
            ),
            source_refs=_require_mapping(p["source_refs"], "payload.source_refs"),
            idempotency_key=idempotency_key,
        ), runtime))
        if outcome == "blocked":
            result.setdefault("partial_receipt", receipt)
        return result

    def _lease_shadow_record(self, p: dict[str, Any], runtime: RuntimeContext) -> Mapping[str, Any]:
        case = _require_mapping(p["case"], "payload.case")
        case_id = _case_identifier(case.get("case_id"), "payload.case.case_id")
        enforcement = _require_mapping(
            p["runtime_lease_enforcement"],
            "payload.runtime_lease_enforcement",
        )
        for key in ("enforcement_enabled", "send_path_blocking_enabled"):
            if type(p[key]) is not bool:
                raise CanonicalWriterError("invalid_request", f"payload.{key} must be boolean")
        payload = {
            "intent_event_id": _identifier(p["intent_event_id"], "payload.intent_event_id"),
            "intent_kind": _identifier(p["intent_kind"], "payload.intent_kind"),
            "case": case,
            "case_id": case_id,
            "runtime_lease_enforcement": enforcement,
            "enforcement_enabled": p["enforcement_enabled"],
            "send_path_blocking_enabled": p["send_path_blocking_enabled"],
            "audit_runtime_id": _identifier(p["audit_runtime_id"], "payload.audit_runtime_id"),
            "source_platform": _identifier(p["source_platform"], "payload.source_platform"),
            "session_key_ref": _identifier(p["session_key_ref"], "payload.session_key_ref"),
        }
        return self.backend.lease_shadow_record(payload, runtime)

    def _capability_grant(self, p: dict[str, Any], runtime: RuntimeContext) -> Mapping[str, Any]:
        hashes = p["command_hashes"]
        if not isinstance(hashes, list) or not 1 <= len(hashes) <= _MAX_COMMANDS:
            raise CanonicalWriterError("invalid_request", "command_hashes is out of bounds")
        command_hashes = tuple(_sha256(value, "payload.command_hashes") for value in hashes)
        if len(set(command_hashes)) != len(command_hashes):
            raise CanonicalWriterError("invalid_request", "command_hashes must be unique")
        if (
            not runtime.user_id
            or not runtime.session_key_sha256
            or not runtime.capability_epoch_sha256
        ):
            raise CanonicalWriterError(
                "invalid_runtime",
                "capability grant requires exact observed user, session, and epoch bindings",
            )
        if runtime.platform != "discord" or not runtime.owner_authenticated:
            raise CanonicalWriterError(
                "owner_required",
                "capability grant requires a configured authenticated Discord owner",
            )
        return self.backend.capability_grant(CapabilityGrantRequest(
            approval_id=_identifier(p["approval_id"], "payload.approval_id"),
            case_id=_case_identifier(p["case_id"], "payload.case_id"),
            plan_id=_identifier(p["plan_id"], "payload.plan_id"),
            plan_revision=_positive_int(
                p["plan_revision"],
                "payload.plan_revision",
                maximum=999_999_999,
            ),
            approval_source_sha256=_sha256(p["approval_source_sha256"], "payload.approval_source_sha256"),
            command_hashes=command_hashes,
            expires_at=_utc(p["expires_at"], "payload.expires_at"),
            max_uses=_positive_int(p["max_uses"], "payload.max_uses", maximum=_MAX_USES),
        ), runtime)

    def _capability_consume(self, p: dict[str, Any], runtime: RuntimeContext) -> Mapping[str, Any]:
        if (
            not runtime.user_id
            or not runtime.session_key_sha256
            or not runtime.capability_epoch_sha256
        ):
            raise CanonicalWriterError(
                "invalid_runtime",
                "capability consume requires exact user, session, and epoch bindings",
            )
        return self.backend.capability_consume(CapabilityConsumeRequest(
            command_sha256=_sha256(p["command_sha256"], "payload.command_sha256"),
            idempotency_key=_idempotency_key(
                p["idempotency_key"],
                "payload.idempotency_key",
            ),
        ), runtime)

    def _capability_revoke(self, p: dict[str, Any], runtime: RuntimeContext) -> Mapping[str, Any]:
        if not runtime.session_key_sha256 or not runtime.capability_epoch_sha256:
            raise CanonicalWriterError(
                "invalid_runtime",
                "capability revoke requires session and epoch bindings",
            )
        return self.backend.capability_revoke(CapabilityRevokeRequest(
            plan_id=_identifier(p["plan_id"], "payload.plan_id"),
            reason=_text(p["reason"], "payload.reason", maximum=1_000),
        ), runtime)

    def _capability_revoke_session(self, p: dict[str, Any], runtime: RuntimeContext) -> Mapping[str, Any]:
        if not runtime.session_key_sha256 or not runtime.capability_epoch_sha256:
            raise CanonicalWriterError(
                "invalid_runtime",
                "session revoke requires session and epoch bindings",
            )
        reason = _text(p["reason"], "payload.reason", maximum=1_000)
        # Reference and production backends expose the session-wide operation
        # as one transaction; fallbacks are intentionally not synthesized.
        method = getattr(self.backend, "capability_revoke_session", None)
        if not callable(method):
            raise CanonicalWriterError("backend_unavailable", "session revoke is unavailable")
        result = method(runtime.session_key_sha256, reason, runtime)
        if isinstance(result, Mapping):
            return result
        return {"revoked": 0}

    def _projector_read(self, p: dict[str, Any], runtime: RuntimeContext) -> Mapping[str, Any]:
        return self.backend.projector_read(ProjectorReadRequest(
            case_id=_case_identifier(p["case_id"], "payload.case_id"),
            after_event_id=_identifier(p["after_event_id"], "payload.after_event_id") if p.get("after_event_id") else "",
            limit=_positive_int(p.get("limit", 100), "payload.limit", maximum=500),
        ), runtime)


class CanonicalWriterTypedDispatcher:
    """Adapter for ``scripts.canonical_writer_service.TypedDispatcher``."""

    def __init__(
        self,
        handlers: CanonicalWriterHandlers,
        *,
        owner_user_ids: frozenset[str] = frozenset(),
    ):
        self.handlers = handlers
        self.owner_user_ids = frozenset(
            str(value) for value in owner_user_ids if str(value)
        )

    def dispatch(self, operation: Any, payload: Mapping[str, Any], context: Any) -> Any:
        from scripts.canonical_writer_service import DispatchResult

        try:
            typed_operation = CanonicalWriterOperation(operation)
            supplied_runtime = context.runtime
            supplied_runtime = (
                supplied_runtime if isinstance(supplied_runtime, Mapping) else {}
            )
            # Only the service-injected runtime envelope is copied. Peer
            # credentials remain in DispatchContext and never enter events or
            # capability bindings.
            platform = str(supplied_runtime.get("platform") or "")
            user_id = str(supplied_runtime.get("user_id") or "")
            runtime = RuntimeContext(
                request_id=str(context.request_id),
                platform=platform,
                session_key_sha256=str(
                    supplied_runtime.get("session_key_sha256") or ""
                ),
                capability_epoch_sha256=str(
                    supplied_runtime.get("capability_epoch_sha256") or ""
                ),
                user_id=user_id,
                chat_id=str(supplied_runtime.get("chat_id") or ""),
                thread_id=str(supplied_runtime.get("thread_id") or ""),
                message_id=str(supplied_runtime.get("message_id") or ""),
                owner_authenticated=bool(
                    platform == "discord"
                    and user_id
                    and user_id in self.owner_user_ids
                ),
                # Wire/runtime payloads can never opt into service-internal
                # authority. One-shot writer jobs construct it in-process.
                service_internal=False,
            )
            clean_payload = dict(payload)
            schema = REQUEST_SCHEMAS[typed_operation.value]
            if "idempotency_key" in schema["allowed"]:
                transport_key = str(context.idempotency_key or "")
                payload_key = str(clean_payload.get("idempotency_key") or "")
                if (
                    (transport_key and payload_key and transport_key != payload_key)
                    or (
                        "idempotency_key" in schema["required"]
                        and not transport_key
                    )
                ):
                    return DispatchResult(
                        status="blocked",
                        result={
                            "success": False,
                            "error_code": "idempotency_binding_mismatch",
                            "error_message": (
                                "payload and transport idempotency keys must match"
                            ),
                        },
                    )
                if transport_key:
                    clean_payload["idempotency_key"] = transport_key
            response = self.handlers.dispatch(
                typed_operation.value,
                clean_payload,
                runtime=runtime,
            )
            if not response.get("ok"):
                error = response.get("error") or {}
                return DispatchResult(
                    status="blocked",
                    result={
                        "success": False,
                        "error_code": str(error.get("code") or "dispatch_failed"),
                        "error_message": str(error.get("message") or "")[:1_000],
                    },
                )
            result = dict(response.get("result") or {})
            if result.get("inserted") is True:
                status = "inserted"
            elif result.get("deduped") is True:
                status = "deduplicated"
            elif result.get("outcome") == "blocked":
                status = "blocked"
            else:
                status = "ok"
            return DispatchResult(status=status, result=result)
        except Exception:
            return DispatchResult(
                status="blocked",
                result={
                    "success": False,
                    "error_code": "dispatch_failed",
                    "error_message": "privileged operation was blocked",
                },
            )


@dataclass
class InMemoryCanonicalWriterStore:
    """Serializable reference state; sharing/restoring it simulates durability."""

    events: list[dict[str, Any]] = field(default_factory=list)
    event_by_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    active_plans: dict[str, dict[str, str]] = field(default_factory=dict)
    capabilities: dict[str, dict[str, Any]] = field(default_factory=dict)
    capability_scope_revocations: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )
    approval_sources: dict[str, str] = field(default_factory=dict)
    consume_attempts: dict[str, dict[str, Any]] = field(default_factory=dict)
    routeback_authorizations: dict[str, dict[str, Any]] = field(default_factory=dict)
    routeback_lifecycle_terminals: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return _json_copy({
                "events": self.events,
                "event_by_id": self.event_by_id,
                "active_plans": self.active_plans,
                "capabilities": self.capabilities,
                "capability_scope_revocations": self.capability_scope_revocations,
                "approval_sources": self.approval_sources,
                "consume_attempts": self.consume_attempts,
                "routeback_authorizations": self.routeback_authorizations,
                "routeback_lifecycle_terminals": (
                    self.routeback_lifecycle_terminals
                ),
            })

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, Any]) -> "InMemoryCanonicalWriterStore":
        data = _json_copy(snapshot)
        return cls(
            events=list(data.get("events") or []),
            event_by_id=dict(data.get("event_by_id") or {}),
            active_plans=dict(data.get("active_plans") or {}),
            capabilities=dict(data.get("capabilities") or {}),
            capability_scope_revocations=dict(
                data.get("capability_scope_revocations") or {}
            ),
            approval_sources=dict(data.get("approval_sources") or {}),
            consume_attempts=dict(data.get("consume_attempts") or {}),
            routeback_authorizations=dict(data.get("routeback_authorizations") or {}),
            routeback_lifecycle_terminals=dict(
                data.get("routeback_lifecycle_terminals") or {}
            ),
        )


class InMemoryCanonicalWriterBackend:
    """Transactional reference backend for handler and protocol tests only."""

    def __init__(
        self,
        store: InMemoryCanonicalWriterStore | None = None,
        *,
        clock: Callable[[], dt.datetime] | None = None,
    ):
        self.store = store or InMemoryCanonicalWriterStore()
        self._clock = clock or (lambda: dt.datetime.now(dt.timezone.utc))

    def _now(self) -> dt.datetime:
        value = self._clock()
        if value.tzinfo is None:
            value = value.replace(tzinfo=dt.timezone.utc)
        return value.astimezone(dt.timezone.utc)

    def ping(self, runtime: RuntimeContext) -> Mapping[str, Any]:
        return {"status": "ok", "request_id": runtime.request_id}

    def _require_initiating_epoch_active_locked(
        self,
        runtime: RuntimeContext,
    ) -> None:
        """Fence new authority against the durable exact-epoch tombstone.

        Callers hold ``store.lock``.  Session revocation uses that same lock, so
        an initiation either commits before the retirement or observes it and
        fails; there is no check/use race in the reference backend.
        """

        _require_exact_runtime_epoch(runtime)
        session_scope = (
            "session:"
            + runtime.session_key_sha256
            + ":"
            + runtime.capability_epoch_sha256
        )
        revocation = self.store.capability_scope_revocations.get(session_scope)
        if revocation and revocation.get("scope_type") == "session":
            raise CanonicalWriterError(
                "session_epoch_retired",
                "session authority epoch has been durably retired",
            )

    def _append_locked(
        self,
        *,
        event_type: str,
        case_id: str,
        body: Mapping[str, Any],
        runtime: RuntimeContext,
        identity: str,
        origin: str,
    ) -> dict[str, Any]:
        event_id = str(uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"canonical-writer:{case_id}:{event_type}:{identity}",
        ))
        content_sha256 = _digest({
            "event_type": event_type,
            "case_id": case_id,
            "body": body,
            "origin": origin,
        })
        existing = self.store.event_by_id.get(event_id)
        if existing:
            if existing.get("content_sha256") != content_sha256:
                raise CanonicalWriterError("idempotency_conflict", "event identity has different content")
            return {"event_id": event_id, "inserted": False, "deduped": True}
        event = {
            "event_id": event_id,
            "event_type": event_type,
            "case_id": case_id,
            "occurred_at": self._now().isoformat(),
            "origin": origin,
            "runtime": runtime.observed_session(),
            "body": _json_copy(body),
            "content_sha256": content_sha256,
        }
        self.store.events.append(event)
        self.store.event_by_id[event_id] = event
        return {"event_id": event_id, "inserted": True, "deduped": False}

    def event_append(self, request: EventAppendRequest, runtime: RuntimeContext) -> Mapping[str, Any]:
        if request.event_type in (
            WRITER_OWNED_EVENT_TYPES
            - {"task.plan.updated", "task.verification.recorded"}
        ):
            raise CanonicalWriterError("privileged_event_forbidden", "typed privileged operation required")
        with self.store.lock:
            self._require_initiating_epoch_active_locked(runtime)
            result = self._append_locked(
                event_type=request.event_type,
                case_id=request.case_id,
                body={
                    "summary": request.summary,
                    "source_refs": request.source_refs,
                    "actors": request.actors,
                    "payload": request.body,
                    "safety": request.safety,
                },
                runtime=runtime,
                identity=request.idempotency_key,
                origin="model_event_append",
            )
            if result["inserted"] and request.event_type == "task.plan.updated":
                self._apply_plan_event_locked(request.case_id, request.body)
            return {"success": True, **result}

    def lease_shadow_record(self, payload: Mapping[str, Any], runtime: RuntimeContext) -> Mapping[str, Any]:
        with self.store.lock:
            result = self._append_locked(
                event_type="lease.shadow.recorded",
                case_id=str(payload["case_id"]),
                body={"lease_shadow": payload},
                runtime=runtime,
                identity=str(payload["intent_event_id"]),
                origin="lease_shadow_record",
            )
            return {"success": True, **result}

    def _apply_plan_event_locked(self, case_id: str, body: Mapping[str, Any]) -> None:
        plan = body.get("plan") if isinstance(body, Mapping) else None
        if not isinstance(plan, Mapping):
            return
        plan_id = str(plan.get("plan_id") or "")
        state = str(plan.get("state") or "")
        revision = int(plan.get("revision") or 0)
        previous = self.store.active_plans.get(case_id)
        supersedes = str(plan.get("supersedes_plan_id") or "")
        if previous and plan_id != previous.get("plan_id") and supersedes == previous.get("plan_id"):
            self._revoke_plan_locked(case_id, previous["plan_id"], "plan_superseded")
        elif (
            previous
            and plan_id == previous.get("plan_id")
            and revision > int(previous.get("revision") or 0)
        ):
            self._revoke_plan_locked(case_id, plan_id, "plan_revision_advanced")
        if state == "active":
            self.store.active_plans[case_id] = {
                "plan_id": plan_id,
                "revision": revision,
                "state": state,
            }
        else:
            self.store.active_plans.pop(case_id, None)
            if state in {"completed", "cancelled", "blocked"}:
                self._revoke_plan_locked(case_id, plan_id, f"plan_{state}")

    def query(self, request: QueryRequest, runtime: RuntimeContext) -> Mapping[str, Any]:
        with self.store.lock:
            rows = [
                copy.deepcopy(event)
                for event in self.store.events
                if (
                    (request.case_id and event["case_id"] == request.case_id)
                    or (
                        request.thread_id
                        and request.thread_id in {
                            str(event.get("runtime", {}).get("thread_id") or ""),
                            str(event.get("runtime", {}).get("chat_id") or ""),
                        }
                    )
                )
            ]
            return {"events": rows[-request.limit:], "view": request.view}

    def plan_active_match(self, request: PlanActiveMatchRequest, runtime: RuntimeContext) -> Mapping[str, Any]:
        with self.store.lock:
            plan = self.store.active_plans.get(request.case_id)
            matches = bool(plan and plan.get("plan_id") == request.plan_id)
            revision = int(plan.get("revision") or 0) if matches and plan else 0
            return {
                "matches": matches,
                "active": matches,
                "plan_revision": revision if revision > 0 else None,
            }

    def routeback_context(self, request: RouteBackContextRequest, runtime: RuntimeContext) -> Mapping[str, Any]:
        thread_id = request.thread_id
        with self.store.lock:
            cases = []
            for authorization in self.store.routeback_authorizations.values():
                terminal = authorization.get("terminal") or {}
                target = authorization.get("target_ref") or {}
                if terminal.get("outcome") != "sent" or thread_id not in {
                    str(target.get("thread_id") or ""),
                    str(target.get("channel_id") or ""),
                }:
                    continue
                source_refs = authorization.get("source_refs") or {}
                source_thread_id = str(
                    source_refs.get("thread_id") or source_refs.get("chat_id") or ""
                )
                if source_thread_id and source_thread_id != thread_id:
                    cases.append({
                        "case_id": authorization["case_id"],
                        "source_thread_id": source_thread_id,
                    })
            return {"thread_id": thread_id, "cases": cases[:3], "truncated": len(cases) > 3}

    def routeback_authorize(self, request: RouteBackAuthorizeRequest, runtime: RuntimeContext) -> Mapping[str, Any]:
        with self.store.lock:
            self._require_initiating_epoch_active_locked(runtime)
            lifecycle_id = "routeblock:" + _digest({
                "case_id": request.case_id,
                "idempotency_key": request.idempotency_key,
            })[:40]
            lifecycle = self.store.routeback_lifecycle_terminals.get(
                lifecycle_id
            )
            if lifecycle:
                expected = {
                    "case_id": request.case_id,
                    "target_ref": _json_copy(request.target_ref),
                    "message_summary": request.message_summary,
                    "source_refs": _json_copy(request.source_refs),
                }
                if any(
                    lifecycle.get(key) != value
                    for key, value in expected.items()
                ):
                    raise CanonicalWriterError(
                        "idempotency_conflict",
                        "route-back lifecycle identity conflicts",
                    )
                terminal = {
                    "outcome": "blocked",
                    "receipt": {},
                    "blocker_reason": lifecycle["blocker_reason"],
                }
                return {
                    "success": True,
                    "preclaim": True,
                    "preclaim_block_id": lifecycle_id,
                    "inserted": False,
                    "deduped": True,
                    "terminal_event_type": "route_back.blocked",
                    "terminal_payload": terminal,
                    **copy.deepcopy(terminal),
                }
            authorization_id = "routeauth:" + _digest({
                "case_id": request.case_id,
                "idempotency_key": request.idempotency_key,
            })[:40]
            candidate = {
                "authorization_id": authorization_id,
                "case_id": request.case_id,
                "target_ref": _json_copy(request.target_ref),
                "message_summary": request.message_summary,
                "source_refs": _json_copy(request.source_refs),
                "content_sha256": request.content_sha256,
                "session_key_sha256": runtime.session_key_sha256,
                "capability_epoch_sha256": runtime.capability_epoch_sha256,
                "state": "authorized",
                "terminal": None,
            }
            existing = self.store.routeback_authorizations.get(authorization_id)
            if existing:
                semantic_keys = {
                    "authorization_id",
                    "case_id",
                    "target_ref",
                    "message_summary",
                    "source_refs",
                    "content_sha256",
                }
                if any(
                    existing.get(key) != candidate.get(key)
                    for key in semantic_keys
                ):
                    raise CanonicalWriterError("idempotency_conflict", "route-back authorization conflicts")
                terminal = existing.get("terminal") or {}
                terminal_outcome = str(terminal.get("outcome") or "")
                return {
                    **copy.deepcopy(existing),
                    "success": True,
                    "inserted": False,
                    "deduped": True,
                    "terminal_event_type": (
                        f"route_back.{terminal_outcome}" if terminal_outcome else ""
                    ),
                    "terminal_payload": copy.deepcopy(terminal),
                }
            candidate["claimed_at"] = self._now().isoformat()
            self.store.routeback_authorizations[authorization_id] = candidate
            appended = self._append_locked(
                event_type="route_back.intent.created",
                case_id=request.case_id,
                body={"authorization": candidate},
                runtime=runtime,
                identity=authorization_id,
                origin="routeback_authorize",
            )
            return {"success": True, **copy.deepcopy(candidate), **appended}

    def routeback_terminal(self, request: RouteBackTerminalRequest, runtime: RuntimeContext) -> Mapping[str, Any]:
        with self.store.lock:
            if request.preclaim:
                self._require_initiating_epoch_active_locked(runtime)
                if request.authorization_id or request.outcome != "blocked" or request.receipt:
                    raise CanonicalWriterError(
                        "invalid_request",
                        "preclaim blocked request contains authorization evidence",
                    )
                authorization_id = "routeauth:" + _digest({
                    "case_id": request.case_id,
                    "idempotency_key": request.idempotency_key,
                })[:40]
                preclaim_id = "routeblock:" + _digest({
                    "case_id": request.case_id,
                    "idempotency_key": request.idempotency_key,
                })[:40]
                existing = self.store.routeback_authorizations.get(
                    authorization_id
                )
                lifecycle = self.store.routeback_lifecycle_terminals.get(
                    preclaim_id
                )
                if lifecycle:
                    expected = {
                        "case_id": request.case_id,
                        "target_ref": _json_copy(request.target_ref),
                        "message_summary": request.message_summary,
                        "source_refs": _json_copy(request.source_refs),
                        "blocker_reason": request.blocker_reason,
                    }
                    if any(
                        lifecycle.get(key) != value
                        for key, value in expected.items()
                    ):
                        raise CanonicalWriterError(
                            "idempotency_conflict",
                            "route-back lifecycle identity conflicts",
                        )
                    return {
                        "success": True,
                        "preclaim": True,
                        "preclaim_block_id": preclaim_id,
                        "outcome": "blocked",
                        "receipt": {},
                        "partial_receipt": {},
                        "blocker_reason": lifecycle["blocker_reason"],
                        "deduped": True,
                    }
                if existing:
                    expected = {
                        "case_id": request.case_id,
                        "target_ref": _json_copy(request.target_ref),
                        "message_summary": request.message_summary,
                        "source_refs": _json_copy(request.source_refs),
                    }
                    if any(
                        existing.get(key) != value
                        for key, value in expected.items()
                    ):
                        raise CanonicalWriterError(
                            "idempotency_conflict",
                            "route-back lifecycle identity conflicts",
                        )
                    terminal = existing.get("terminal") or {}
                    if not terminal:
                        raise CanonicalWriterError(
                            "routeback_outcome_uncertain",
                            "route-back claim is pending reconciliation",
                        )
                    if terminal.get("outcome") != "blocked":
                        raise CanonicalWriterError(
                            "terminal_conflict",
                            "route-back lifecycle is already finalized differently",
                        )
                    return {
                        "success": True,
                        "preclaim": False,
                        "preclaim_block_id": preclaim_id,
                        **copy.deepcopy(terminal),
                        "partial_receipt": copy.deepcopy(
                            terminal.get("receipt") or {}
                        ),
                        "deduped": True,
                    }
                result = self._append_locked(
                    event_type="route_back.blocked",
                    case_id=request.case_id,
                    body={
                        "preclaim": True,
                        "preclaim_block_id": preclaim_id,
                        "target_ref": _json_copy(request.target_ref),
                        "message_summary": request.message_summary,
                        "source_refs": _json_copy(request.source_refs),
                        "blocker_reason": request.blocker_reason,
                        "partial_receipt": {},
                    },
                    runtime=runtime,
                    identity=preclaim_id,
                    origin="routeback_preclaim_blocked",
                )
                self.store.routeback_lifecycle_terminals[preclaim_id] = {
                    "preclaim_block_id": preclaim_id,
                    "case_id": request.case_id,
                    "target_ref": _json_copy(request.target_ref),
                    "message_summary": request.message_summary,
                    "source_refs": _json_copy(request.source_refs),
                    "session_key_sha256": runtime.session_key_sha256,
                    "capability_epoch_sha256": (
                        runtime.capability_epoch_sha256
                    ),
                    "blocker_reason": request.blocker_reason,
                }
                return {
                    "success": True,
                    "preclaim": True,
                    "preclaim_block_id": preclaim_id,
                    "outcome": "blocked",
                    "receipt": {},
                    "partial_receipt": {},
                    "blocker_reason": request.blocker_reason,
                    **result,
                }
            authorization = self.store.routeback_authorizations.get(request.authorization_id)
            if not authorization:
                raise CanonicalWriterError("authorization_missing", "route-back authorization not found")
            if authorization["session_key_sha256"] != runtime.session_key_sha256:
                raise CanonicalWriterError("scope_mismatch", "route-back session does not match")
            if (
                authorization.get("capability_epoch_sha256")
                != runtime.capability_epoch_sha256
            ):
                raise CanonicalWriterError(
                    "scope_mismatch",
                    "route-back routing epoch does not match",
                )
            terminal = {
                "outcome": request.outcome,
                "receipt": _json_copy(request.receipt),
                "blocker_reason": request.blocker_reason,
            }
            if authorization.get("terminal"):
                if authorization["terminal"] != terminal:
                    raise CanonicalWriterError("terminal_conflict", "route-back already finalized differently")
                return {
                    "success": True,
                    "authorization_id": request.authorization_id,
                    **terminal,
                    "deduped": True,
                }
            if request.outcome == "sent":
                receipt = request.receipt
                for key in ("message_id", "channel_id", "content_sha256"):
                    if not str(receipt.get(key) or ""):
                        raise CanonicalWriterError("invalid_receipt", f"receipt.{key} is required")
                if (
                    receipt.get("platform") != "discord"
                    or receipt.get("adapter_receipt") is not True
                    or receipt.get("receipt_readback_verified") is not True
                ):
                    raise CanonicalWriterError(
                        "invalid_receipt",
                        "verified Discord adapter receipt is required",
                    )
                if str(receipt["content_sha256"]) != authorization["content_sha256"]:
                    raise CanonicalWriterError("invalid_receipt", "receipt content hash does not match authorization")
                target_id = str(
                    authorization["target_ref"].get("thread_id")
                    or authorization["target_ref"].get("channel_id")
                    or ""
                )
                if str(receipt["channel_id"]) != target_id:
                    raise CanonicalWriterError("invalid_receipt", "receipt channel does not match authorization")
            authorization["terminal"] = terminal
            authorization["state"] = request.outcome
            event_type = "route_back.sent" if request.outcome == "sent" else "route_back.blocked"
            body = {
                "authorization_id": request.authorization_id,
                "target_ref": authorization["target_ref"],
                "receipt": terminal["receipt"],
                "blocker_reason": terminal["blocker_reason"],
            }
            result = self._append_locked(
                event_type=event_type,
                case_id=authorization["case_id"],
                body=body,
                runtime=runtime,
                identity=request.authorization_id,
                origin="routeback_terminal",
            )
            return {
                "success": True,
                "authorization_id": request.authorization_id,
                **terminal,
                **result,
            }

    def capability_grant(self, request: CapabilityGrantRequest, runtime: RuntimeContext) -> Mapping[str, Any]:
        with self.store.lock:
            session_scope = (
                "session:" + runtime.session_key_sha256 + ":"
                + runtime.capability_epoch_sha256
            )
            plan_scope = session_scope + ":plan:" + request.plan_id
            if (
                session_scope in self.store.capability_scope_revocations
                or plan_scope in self.store.capability_scope_revocations
            ):
                raise CanonicalWriterError(
                    "capability_scope_revoked",
                    "capability scope was durably revoked for this routing epoch",
                )
            active = self.store.active_plans.get(request.case_id)
            if (
                not active
                or active.get("plan_id") != request.plan_id
                or int(active.get("revision") or 0) != request.plan_revision
            ):
                raise CanonicalWriterError("plan_not_active", "exact canonical plan is not active")
            if request.expires_at <= self._now():
                raise CanonicalWriterError("approval_expired", "capability expiry must be in the future")
            if request.expires_at > self._now() + dt.timedelta(hours=8):
                raise CanonicalWriterError(
                    "approval_expiry_out_of_bounds",
                    "capability expiry cannot exceed eight hours",
                )
            candidate = {
                "approval_id": request.approval_id,
                "case_id": request.case_id,
                "plan_id": request.plan_id,
                "plan_revision": request.plan_revision,
                "session_key_sha256": runtime.session_key_sha256,
                "capability_epoch_sha256": runtime.capability_epoch_sha256,
                "approved_by_user_id": runtime.user_id,
                "approval_source_sha256": request.approval_source_sha256,
                "command_hashes": list(request.command_hashes),
                "max_uses": request.max_uses,
                "expires_at": request.expires_at.isoformat(),
                "remaining_uses": {value: request.max_uses for value in request.command_hashes},
                "state": "granted",
            }
            replay_id = self.store.approval_sources.get(request.approval_source_sha256)
            if replay_id:
                existing = self.store.capabilities.get(replay_id)
                immutable_keys = (
                    "approval_id",
                    "case_id",
                    "plan_id",
                    "plan_revision",
                    "session_key_sha256",
                    "capability_epoch_sha256",
                    "approved_by_user_id",
                    "approval_source_sha256",
                    "command_hashes",
                    "expires_at",
                    "max_uses",
                )
                if existing and all(
                    existing.get(key) == candidate.get(key)
                    for key in immutable_keys
                ):
                    if self._now() >= _utc(
                        existing["expires_at"],
                        "capability.expires_at",
                    ):
                        existing["state"] = "expired"
                    return {
                        "success": True,
                        **copy.deepcopy(existing),
                        "authority_active": existing.get("state") == "granted",
                        "deduped": True,
                    }
                raise CanonicalWriterError("approval_source_replay", "approval source was already consumed")
            if request.approval_id in self.store.capabilities:
                raise CanonicalWriterError("approval_id_conflict", "approval_id already exists")
            self.store.capabilities[request.approval_id] = candidate
            self.store.approval_sources[request.approval_source_sha256] = request.approval_id
            result = self._append_locked(
                event_type="approval.capability.recorded",
                case_id=request.case_id,
                body={"approval_receipt": candidate},
                runtime=runtime,
                identity=f"grant:{request.approval_id}",
                origin="capability_grant",
            )
            return {
                "success": True,
                **copy.deepcopy(candidate),
                **result,
                "authority_active": True,
                "deduped": False,
            }

    def capability_consume(self, request: CapabilityConsumeRequest, runtime: RuntimeContext) -> Mapping[str, Any]:
        # Active-plan recheck, scope match, decrement, and receipt append share
        # one backend transaction/lock.  Handlers never perform a split check.
        with self.store.lock:
            attempt_key = _digest({
                "session_key_sha256": runtime.session_key_sha256,
                "capability_epoch_sha256": runtime.capability_epoch_sha256,
                "user_id": runtime.user_id,
                "idempotency_key": request.idempotency_key,
            })
            previous_attempt = self.store.consume_attempts.get(attempt_key)
            if previous_attempt:
                if previous_attempt.get("command_sha256") != request.command_sha256:
                    raise CanonicalWriterError(
                        "idempotency_conflict",
                        "consume idempotency key was used for another command",
                    )
                return {
                    **copy.deepcopy(previous_attempt["result"]),
                    "deduped": True,
                }
            matched = [
                capability
                for capability in self.store.capabilities.values()
                if capability.get("session_key_sha256") == runtime.session_key_sha256
                and capability.get("capability_epoch_sha256")
                    == runtime.capability_epoch_sha256
                and capability.get("approved_by_user_id") == runtime.user_id
                and request.command_sha256 in capability.get("command_hashes", [])
                and capability.get("state") == "granted"
            ]
            failure_code = "capability_missing"
            for capability in matched:
                if self._now() >= _utc(capability["expires_at"], "capability.expires_at"):
                    capability["state"] = "expired"
                    failure_code = "capability_expired"
                    continue
                active = self.store.active_plans.get(str(capability["case_id"]))
                if (
                    not active
                    or active.get("plan_id") != capability.get("plan_id")
                    or int(active.get("revision") or 0)
                        != int(capability.get("plan_revision") or 0)
                ):
                    capability["state"] = "revoked"
                    failure_code = "plan_not_active"
            candidates = [
                capability for capability in matched
                if capability.get("state") == "granted"
            ]
            if not candidates:
                messages = {
                    "capability_expired": "matching durable capability has expired",
                    "plan_not_active": "matching canonical plan is not active",
                    "capability_missing": "no durable capability matches command and session",
                }
                raise CanonicalWriterError(failure_code, messages[failure_code])
            if len(candidates) > 1:
                raise CanonicalWriterError("capability_ambiguous", "multiple durable capabilities match command and session")
            capability = candidates[0]
            case_id = str(capability["case_id"])
            plan_id = str(capability["plan_id"])
            plan_revision = int(capability["plan_revision"])
            approval_id = str(capability["approval_id"])
            remaining = capability["remaining_uses"].get(request.command_sha256)
            if remaining is None:
                raise CanonicalWriterError("command_not_approved", "command hash is not approved")
            if int(remaining) <= 0:
                raise CanonicalWriterError("capability_exhausted", "command use counter is exhausted")
            remaining = int(remaining) - 1
            capability["remaining_uses"][request.command_sha256] = remaining
            result = self._append_locked(
                event_type="capability.check.recorded",
                case_id=case_id,
                body={"capability_receipt": {
                    "approval_id": approval_id,
                    "plan_id": plan_id,
                    "plan_revision": plan_revision,
                    "approved_by_user_id": runtime.user_id,
                    "session_key_sha256": runtime.session_key_sha256,
                    "capability_epoch_sha256": runtime.capability_epoch_sha256,
                    "command_sha256": request.command_sha256,
                    "remaining_uses_for_command": remaining,
                    "state": "authorized",
                }},
                runtime=runtime,
                identity=f"consume:{attempt_key}",
                origin="capability_consume",
            )
            response = {
                "success": True,
                "authorized": True,
                "approval_id": approval_id,
                "case_id": case_id,
                "plan_id": plan_id,
                "plan_revision": plan_revision,
                "approved_by_user_id": runtime.user_id,
                "capability_epoch_sha256": runtime.capability_epoch_sha256,
                "command_sha256": request.command_sha256,
                "remaining_uses": remaining,
                **result,
            }
            self.store.consume_attempts[attempt_key] = {
                "command_sha256": request.command_sha256,
                "result": copy.deepcopy(response),
            }
            return response

    def _revoke_plan_locked(self, case_id: str, plan_id: str, reason: str) -> int:
        count = 0
        for capability in self.store.capabilities.values():
            if (
                capability.get("case_id") == case_id
                and capability.get("plan_id") == plan_id
                and capability.get("state") == "granted"
            ):
                capability["state"] = "revoked"
                capability["revoke_reason"] = reason
                count += 1
        return count

    def capability_revoke(self, request: CapabilityRevokeRequest, runtime: RuntimeContext) -> Mapping[str, Any]:
        with self.store.lock:
            session_scope = (
                "session:" + runtime.session_key_sha256 + ":"
                + runtime.capability_epoch_sha256
            )
            plan_scope = session_scope + ":plan:" + request.plan_id
            self.store.capability_scope_revocations.setdefault(
                plan_scope,
                {
                    "scope_type": "plan",
                    "session_key_sha256": runtime.session_key_sha256,
                    "capability_epoch_sha256": runtime.capability_epoch_sha256,
                    "plan_id": request.plan_id,
                    "reason": request.reason,
                },
            )
            revoked_by_case: dict[str, list[str]] = {}
            for capability in self.store.capabilities.values():
                if (
                    capability.get("plan_id") != request.plan_id
                    or capability.get("session_key_sha256") != runtime.session_key_sha256
                    or capability.get("capability_epoch_sha256")
                        != runtime.capability_epoch_sha256
                ):
                    continue
                if capability.get("state") == "granted":
                    capability["state"] = "revoked"
                    capability["revoke_reason"] = request.reason
                    revoked_by_case.setdefault(
                        str(capability.get("case_id") or ""), []
                    ).append(str(capability.get("approval_id") or ""))
            for case_id, approval_ids in sorted(revoked_by_case.items()):
                set_sha256 = _digest(sorted(approval_ids))
                self._append_locked(
                    event_type="approval.capability.revoked",
                    case_id=case_id,
                    body={
                        "plan_id": request.plan_id,
                        "session_key_sha256": runtime.session_key_sha256,
                        "capability_epoch_sha256": runtime.capability_epoch_sha256,
                        "reason": request.reason,
                        "revoked": len(approval_ids),
                        "revocation_set_sha256": set_sha256,
                    },
                    runtime=runtime,
                    identity=(
                        "revoke:" + request.plan_id + ":"
                        + runtime.capability_epoch_sha256 + ":" + set_sha256
                    ),
                    origin="capability_revoke",
                )
            revoked = sum(len(items) for items in revoked_by_case.values())
            return {
                "success": True,
                "capability_epoch_sha256": runtime.capability_epoch_sha256,
                "scope_type": "plan",
                "scope_revoked": True,
                "revoked": revoked,
            }

    def capability_revoke_session(
        self,
        session_key_sha256: str,
        reason: str,
        runtime: RuntimeContext,
    ) -> Mapping[str, Any]:
        with self.store.lock:
            session_scope = (
                "session:" + session_key_sha256 + ":"
                + runtime.capability_epoch_sha256
            )
            self.store.capability_scope_revocations.setdefault(
                session_scope,
                {
                    "scope_type": "session",
                    "session_key_sha256": session_key_sha256,
                    "capability_epoch_sha256": runtime.capability_epoch_sha256,
                    "plan_id": "",
                    "reason": reason,
                },
            )
            revoked_by_case: dict[str, list[str]] = {}
            for capability in self.store.capabilities.values():
                if (
                    capability.get("session_key_sha256") == session_key_sha256
                    and capability.get("capability_epoch_sha256")
                        == runtime.capability_epoch_sha256
                    and capability.get("state") == "granted"
                ):
                    capability["state"] = "revoked"
                    capability["revoke_reason"] = reason
                    revoked_by_case.setdefault(
                        str(capability.get("case_id") or ""), []
                    ).append(str(capability.get("approval_id") or ""))
            for case_id, approval_ids in sorted(revoked_by_case.items()):
                set_sha256 = _digest(sorted(approval_ids))
                self._append_locked(
                    event_type="approval.capability.session_revoked",
                    case_id=case_id,
                    body={
                        "session_key_sha256": session_key_sha256,
                        "capability_epoch_sha256": runtime.capability_epoch_sha256,
                        "reason": reason,
                        "revoked": len(approval_ids),
                        "revocation_set_sha256": set_sha256,
                    },
                    runtime=runtime,
                    identity=(
                        "session-revoke:" + session_key_sha256 + ":"
                        + runtime.capability_epoch_sha256 + ":" + set_sha256
                    ),
                    origin="capability_revoke_session",
                )
            revoked = sum(len(items) for items in revoked_by_case.values())
            return {
                "success": True,
                "session_key_sha256": session_key_sha256,
                "capability_epoch_sha256": runtime.capability_epoch_sha256,
                "scope_type": "session",
                "scope_revoked": True,
                "revoked": revoked,
            }

    def projector_read(self, request: ProjectorReadRequest, runtime: RuntimeContext) -> Mapping[str, Any]:
        with self.store.lock:
            rows = [event for event in self.store.events if event["case_id"] == request.case_id]
            if request.after_event_id:
                indexes = [
                    index for index, event in enumerate(rows)
                    if event["event_id"] == request.after_event_id
                ]
                rows = rows[indexes[-1] + 1:] if indexes else []
            return {"events": copy.deepcopy(rows[:request.limit])}


__all__ = [
    "CanonicalWriterBackend",
    "CanonicalWriterError",
    "CanonicalWriterHandlers",
    "CanonicalWriterTypedDispatcher",
    "RuntimeContext",
    "EventAppendRequest",
    "QueryRequest",
    "PlanActiveMatchRequest",
    "RouteBackContextRequest",
    "RouteBackAuthorizeRequest",
    "RouteBackTerminalRequest",
    "CapabilityGrantRequest",
    "CapabilityConsumeRequest",
    "CapabilityRevokeRequest",
    "ProjectorReadRequest",
    "InMemoryCanonicalWriterBackend",
    "InMemoryCanonicalWriterStore",
    "SUPPORTED_OPERATIONS",
    "REQUEST_SCHEMAS",
    "MODEL_FORBIDDEN_EVENT_TYPES",
    "OP_PING",
    "OP_EVENT_APPEND",
    "OP_EVENT_APPEND_MODEL",
    "OP_PLAN_TRANSITION",
    "OP_VERIFICATION_APPEND",
    "OP_QUERY",
    "OP_CASE_QUERY",
    "OP_PLAN_ACTIVE_MATCH",
    "OP_ROUTEBACK_CONTEXT",
    "OP_ROUTEBACK_AUTHORIZE",
    "OP_ROUTEBACK_CLAIM",
    "OP_ROUTEBACK_FINALIZE_SENT",
    "OP_ROUTEBACK_FINALIZE_BLOCKED",
    "OP_LEASE_SHADOW_RECORD",
    "OP_CAPABILITY_GRANT",
    "OP_CAPABILITY_CONSUME",
    "OP_CAPABILITY_REVOKE",
    "OP_CAPABILITY_REVOKE_SESSION",
    "OP_PROJECTOR_READ",
    "OP_PROJECTION_READ_EVENTS",
]
