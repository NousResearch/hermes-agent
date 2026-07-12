"""Strict protocol primitives for a future privileged Discord edge service.

This module is deliberately transport- and Discord-SDK-free.  It defines the
small fixed operation surface that an unprivileged gateway may eventually send
to a token-owning edge process, plus Ed25519-signed capability and receipt
envelopes.  It does not load keys, open sockets, call Discord, or inspect text
for meaning.

Hermes/GPT remains responsible for choosing an operation, target, and content.
These primitives perform only schema validation, explicit public-target
validation, exact request binding, expiry checks, and signature verification.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

PROTOCOL_VERSION = "discord-edge.v1"
RECONCILIATION_PROTOCOL_VERSION = "discord-edge-reconcile.v1"
RECONCILIATION_RESPONSE_VERSION = "discord-edge-reconcile-result.v1"
RECONCILIATION_NOT_AVAILABLE_ERROR = "reconciliation_not_available"
CAPABILITY_VERSION = "discord-edge-capability.v1"
RECEIPT_VERSION = "discord-edge-receipt.v1"

MAX_DEADLINE_SECONDS = 30
MAX_REQUEST_BYTES = 64 * 1024
MAX_SEQUENCE = (1 << 63) - 1
MAX_IDEMPOTENCY_KEY_BYTES = 256
MAX_CONTENT_CHARS = 2_000
MAX_CONTENT_BYTES = 8_000
MAX_THREAD_NAME_CHARS = 100
MAX_THREAD_NAME_BYTES = 400
MAX_CAPABILITY_LIFETIME_MS = 8 * 60 * 60 * 1000
MAX_CLOCK_SKEW_MS = 30 * 1000

_CAPABILITY_DOMAIN = b"muncho-discord-edge-capability-v1\x00"
_RECEIPT_DOMAIN = b"muncho-discord-edge-receipt-v1\x00"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_KEY_ID_RE = _SHA256_RE
_SNOWFLAKE_RE = re.compile(r"^[0-9]{1,25}$")
_IDEMPOTENCY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")
_BLOCKER_CODE_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$")
_SIGNATURE_RE = re.compile(r"^[A-Za-z0-9_-]{86}$")
_AUTO_ARCHIVE_MINUTES = frozenset({60, 1_440, 4_320, 10_080})

_FORBIDDEN_TARGET_TYPE_NAMES = frozenset(
    {
        "dm",
        "direct_message",
        "group_dm",
        "private",
        "private_channel",
        "private_thread",
    }
)


class DiscordEdgeErrorCode(StrEnum):
    INVALID_JSON = "invalid_json"
    REQUEST_TOO_LARGE = "request_too_large"
    INVALID_SHAPE = "invalid_shape"
    UNSUPPORTED_VERSION = "unsupported_version"
    UNKNOWN_OPERATION = "unknown_operation"
    INVALID_REQUEST_ID = "invalid_request_id"
    INVALID_SEQUENCE = "invalid_sequence"
    INVALID_DEADLINE = "invalid_deadline"
    FORBIDDEN_TARGET = "forbidden_target"
    INVALID_TARGET = "invalid_target"
    INVALID_PAYLOAD = "invalid_payload"
    INVALID_IDEMPOTENCY_KEY = "invalid_idempotency_key"
    INVALID_CAPABILITY = "invalid_capability"
    CAPABILITY_EXPIRED = "capability_expired"
    CAPABILITY_BINDING_MISMATCH = "capability_binding_mismatch"
    INVALID_RECEIPT = "invalid_receipt"
    RECEIPT_BINDING_MISMATCH = "receipt_binding_mismatch"
    SIGNATURE_INVALID = "signature_invalid"


class DiscordEdgeProtocolError(ValueError):
    """Stable, secret-free validation failure."""

    def __init__(self, code: DiscordEdgeErrorCode, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code.value}: {detail}")


class DiscordEdgeOperation(StrEnum):
    """The initial fixed mutation surface; there is no raw HTTP operation."""

    PUBLIC_MESSAGE_SEND = "public.message.send"
    PUBLIC_MESSAGE_EDIT = "public.message.edit"
    PUBLIC_THREAD_CREATE = "public.thread.create"


class DiscordPublicTargetType(StrEnum):
    """Discord guild surfaces that can potentially be proven public live."""

    PUBLIC_GUILD_CHANNEL = "public_guild_channel"
    PUBLIC_GUILD_THREAD = "public_guild_thread"
    PUBLIC_GUILD_FORUM = "public_guild_forum"


class DiscordEdgeAuthorityKind(StrEnum):
    """Structured authority sources; none is inferred from message prose."""

    SESSION_REPLY = "session_reply"
    CANONICAL_ROUTEBACK = "canonical_routeback"
    CANONICAL_PLAN = "canonical_plan"
    ROOT_JOB = "root_job"


class DiscordEdgeReceiptOutcome(StrEnum):
    VERIFIED = "verified"
    ACCEPTED_UNVERIFIED = "accepted_unverified"
    DISPATCH_UNCERTAIN = "dispatch_uncertain"
    BLOCKED_BEFORE_DISPATCH = "blocked_before_dispatch"
    FAILED_BEFORE_DISPATCH = "failed_before_dispatch"


_OPERATION_PAYLOAD_FIELDS: Mapping[
    DiscordEdgeOperation, tuple[frozenset[str], frozenset[str]]
] = {
    DiscordEdgeOperation.PUBLIC_MESSAGE_SEND: (
        frozenset({"content"}),
        frozenset({"reply_to_message_id"}),
    ),
    DiscordEdgeOperation.PUBLIC_MESSAGE_EDIT: (
        frozenset({"message_id", "content"}),
        frozenset(),
    ),
    DiscordEdgeOperation.PUBLIC_THREAD_CREATE: (
        frozenset({"name"}),
        frozenset({"initial_message", "auto_archive_minutes"}),
    ),
}

_OPERATION_TARGET_TYPES: Mapping[
    DiscordEdgeOperation, frozenset[DiscordPublicTargetType]
] = {
    DiscordEdgeOperation.PUBLIC_MESSAGE_SEND: frozenset(
        {
            DiscordPublicTargetType.PUBLIC_GUILD_CHANNEL,
            DiscordPublicTargetType.PUBLIC_GUILD_THREAD,
        }
    ),
    DiscordEdgeOperation.PUBLIC_MESSAGE_EDIT: frozenset(
        {
            DiscordPublicTargetType.PUBLIC_GUILD_CHANNEL,
            DiscordPublicTargetType.PUBLIC_GUILD_THREAD,
        }
    ),
    DiscordEdgeOperation.PUBLIC_THREAD_CREATE: frozenset(
        {
            DiscordPublicTargetType.PUBLIC_GUILD_CHANNEL,
            DiscordPublicTargetType.PUBLIC_GUILD_FORUM,
        }
    ),
}

_OPERATION_AUTHORITY_KINDS: Mapping[
    DiscordEdgeOperation, frozenset[DiscordEdgeAuthorityKind]
] = {
    DiscordEdgeOperation.PUBLIC_MESSAGE_SEND: frozenset(DiscordEdgeAuthorityKind),
    DiscordEdgeOperation.PUBLIC_MESSAGE_EDIT: frozenset(
        {
            DiscordEdgeAuthorityKind.SESSION_REPLY,
            DiscordEdgeAuthorityKind.CANONICAL_PLAN,
            DiscordEdgeAuthorityKind.ROOT_JOB,
        }
    ),
    DiscordEdgeOperation.PUBLIC_THREAD_CREATE: frozenset(
        {
            DiscordEdgeAuthorityKind.CANONICAL_PLAN,
            DiscordEdgeAuthorityKind.ROOT_JOB,
        }
    ),
}


def _fail(code: DiscordEdgeErrorCode, detail: str) -> None:
    raise DiscordEdgeProtocolError(code, detail)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _strict_object(
    value: Any,
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
    code: DiscordEdgeErrorCode,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(code, f"{label} must be an object")
    if any(not isinstance(key, str) for key in value):
        _fail(code, f"{label} keys must be strings")
    result = dict(value)
    unknown = set(result) - required - optional
    missing = required - set(result)
    if unknown:
        _fail(code, f"{label} has unknown fields: {', '.join(sorted(unknown))}")
    if missing:
        _fail(code, f"{label} is missing fields: {', '.join(sorted(missing))}")
    return result


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Return deterministic JSON bytes, rejecting non-JSON values."""

    try:
        return json.dumps(
            dict(value),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DiscordEdgeProtocolError(
            DiscordEdgeErrorCode.INVALID_SHAPE,
            "value is not canonical JSON",
        ) from exc


def _reject_non_json_constant(value: str) -> None:
    raise ValueError(f"non-JSON numeric constant: {value}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def decode_request_json(body: bytes) -> dict[str, Any]:
    """Decode one fixed-size strict UTF-8 request object before typed parsing."""

    if not isinstance(body, bytes) or not body:
        _fail(DiscordEdgeErrorCode.INVALID_JSON, "request body must be non-empty bytes")
    if len(body) > MAX_REQUEST_BYTES:
        _fail(
            DiscordEdgeErrorCode.REQUEST_TOO_LARGE,
            "request body exceeds the fixed Discord edge limit",
        )
    try:
        value = json.loads(
            body.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_non_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise DiscordEdgeProtocolError(
            DiscordEdgeErrorCode.INVALID_JSON,
            "request body is not strict UTF-8 JSON",
        ) from exc
    if not isinstance(value, dict):
        _fail(DiscordEdgeErrorCode.INVALID_JSON, "request JSON root must be an object")
    return value


def _json_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(canonical_json_bytes(value).decode("utf-8"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _validate_sha256(value: Any, label: str, code: DiscordEdgeErrorCode) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        _fail(code, f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _validate_uuid(value: Any, label: str, code: DiscordEdgeErrorCode) -> str:
    if not isinstance(value, str):
        _fail(code, f"{label} must be a canonical UUID")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise DiscordEdgeProtocolError(code, f"{label} must be a canonical UUID") from exc
    canonical = str(parsed)
    if value != canonical:
        _fail(code, f"{label} must be a lowercase canonical UUID")
    return canonical


def _validate_int(
    value: Any,
    label: str,
    *,
    minimum: int,
    maximum: int,
    code: DiscordEdgeErrorCode,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(code, f"{label} must be an integer")
    if value < minimum or value > maximum:
        _fail(code, f"{label} is outside its allowed range")
    return value


def _validate_snowflake(
    value: Any,
    label: str,
    *,
    code: DiscordEdgeErrorCode = DiscordEdgeErrorCode.INVALID_TARGET,
) -> str:
    if (
        not isinstance(value, str)
        or not _SNOWFLAKE_RE.fullmatch(value)
        or int(value) == 0
    ):
        _fail(code, f"{label} must be a non-zero numeric Discord snowflake")
    return value


def _validate_text(
    value: Any,
    label: str,
    *,
    allow_empty: bool,
    max_chars: int,
    max_bytes: int,
) -> str:
    if not isinstance(value, str):
        _fail(DiscordEdgeErrorCode.INVALID_PAYLOAD, f"{label} must be a string")
    if not allow_empty and not value:
        _fail(DiscordEdgeErrorCode.INVALID_PAYLOAD, f"{label} must not be empty")
    if "\x00" in value:
        _fail(DiscordEdgeErrorCode.INVALID_PAYLOAD, f"{label} contains NUL")
    if len(value) > max_chars:
        _fail(DiscordEdgeErrorCode.INVALID_PAYLOAD, f"{label} exceeds its character limit")
    if len(value.encode("utf-8")) > max_bytes:
        _fail(DiscordEdgeErrorCode.INVALID_PAYLOAD, f"{label} exceeds its byte limit")
    return value


def _validate_idempotency_key(value: Any) -> str:
    if not isinstance(value, str):
        _fail(
            DiscordEdgeErrorCode.INVALID_IDEMPOTENCY_KEY,
            "idempotency_key must be a string",
        )
    size = len(value.encode("utf-8"))
    if (
        size < 1
        or size > MAX_IDEMPOTENCY_KEY_BYTES
        or not _IDEMPOTENCY_RE.fullmatch(value)
    ):
        _fail(
            DiscordEdgeErrorCode.INVALID_IDEMPOTENCY_KEY,
            "idempotency_key has an invalid format or length",
        )
    return value


def _validate_authority_ref(value: Any) -> str:
    if not isinstance(value, str):
        _fail(DiscordEdgeErrorCode.INVALID_CAPABILITY, "authority_ref must be a string")
    size = len(value.encode("utf-8"))
    if (
        size < 1
        or size > MAX_IDEMPOTENCY_KEY_BYTES
        or not _IDEMPOTENCY_RE.fullmatch(value)
    ):
        _fail(
            DiscordEdgeErrorCode.INVALID_CAPABILITY,
            "authority_ref has an invalid format or length",
        )
    return value


def _parse_operation(value: Any) -> DiscordEdgeOperation:
    if not isinstance(value, str):
        _fail(DiscordEdgeErrorCode.UNKNOWN_OPERATION, "operation must be a string")
    try:
        return DiscordEdgeOperation(value)
    except ValueError as exc:
        raise DiscordEdgeProtocolError(
            DiscordEdgeErrorCode.UNKNOWN_OPERATION,
            "operation is not in the fixed Discord edge operation set",
        ) from exc


def _validate_operation_target(
    operation: DiscordEdgeOperation,
    target: "DiscordPublicTarget",
) -> None:
    if target.target_type not in _OPERATION_TARGET_TYPES[operation]:
        _fail(
            DiscordEdgeErrorCode.INVALID_TARGET,
            f"{target.target_type.value} cannot be used with {operation.value}",
        )


def _validate_operation_authority(
    operation: DiscordEdgeOperation,
    authority_kind: DiscordEdgeAuthorityKind,
) -> None:
    if authority_kind not in _OPERATION_AUTHORITY_KINDS[operation]:
        _fail(
            DiscordEdgeErrorCode.INVALID_CAPABILITY,
            f"{authority_kind.value} cannot authorize {operation.value}",
        )


@dataclass(frozen=True)
class DiscordPublicTarget:
    target_type: DiscordPublicTargetType
    guild_id: str
    channel_id: str
    parent_channel_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.target_type, DiscordPublicTargetType):
            _fail(
                DiscordEdgeErrorCode.INVALID_TARGET,
                "target_type must be an explicit public Discord target enum",
            )
        object.__setattr__(
            self,
            "guild_id",
            _validate_snowflake(self.guild_id, "guild_id"),
        )
        object.__setattr__(
            self,
            "channel_id",
            _validate_snowflake(self.channel_id, "channel_id"),
        )
        if self.parent_channel_id is not None:
            object.__setattr__(
                self,
                "parent_channel_id",
                _validate_snowflake(self.parent_channel_id, "parent_channel_id"),
            )
        if self.target_type is DiscordPublicTargetType.PUBLIC_GUILD_THREAD:
            if self.parent_channel_id is None:
                _fail(
                    DiscordEdgeErrorCode.INVALID_TARGET,
                    "public guild threads require parent_channel_id",
                )
            if self.parent_channel_id == self.channel_id:
                _fail(
                    DiscordEdgeErrorCode.INVALID_TARGET,
                    "thread channel_id and parent_channel_id must differ",
                )
        elif self.parent_channel_id is not None:
            _fail(
                DiscordEdgeErrorCode.INVALID_TARGET,
                "parent_channel_id is valid only for public guild threads",
            )

    @classmethod
    def from_mapping(cls, value: Any) -> "DiscordPublicTarget":
        fields = _strict_object(
            value,
            required=frozenset({"target_type", "guild_id", "channel_id"}),
            optional=frozenset({"parent_channel_id"}),
            code=DiscordEdgeErrorCode.INVALID_TARGET,
            label="target",
        )
        raw_type = fields["target_type"]
        if not isinstance(raw_type, str):
            _fail(DiscordEdgeErrorCode.INVALID_TARGET, "target_type must be a string")
        if raw_type in _FORBIDDEN_TARGET_TYPE_NAMES:
            _fail(
                DiscordEdgeErrorCode.FORBIDDEN_TARGET,
                "Discord DMs and private targets are forbidden",
            )
        try:
            target_type = DiscordPublicTargetType(raw_type)
        except ValueError as exc:
            raise DiscordEdgeProtocolError(
                DiscordEdgeErrorCode.INVALID_TARGET,
                "target_type is not an explicit public Discord target type",
            ) from exc

        guild_id = _validate_snowflake(fields["guild_id"], "guild_id")
        channel_id = _validate_snowflake(fields["channel_id"], "channel_id")
        raw_parent = fields.get("parent_channel_id")
        parent_channel_id = (
            _validate_snowflake(raw_parent, "parent_channel_id")
            if raw_parent is not None
            else None
        )
        if target_type is DiscordPublicTargetType.PUBLIC_GUILD_THREAD:
            if parent_channel_id is None:
                _fail(
                    DiscordEdgeErrorCode.INVALID_TARGET,
                    "public guild threads require parent_channel_id",
                )
            if parent_channel_id == channel_id:
                _fail(
                    DiscordEdgeErrorCode.INVALID_TARGET,
                    "thread channel_id and parent_channel_id must differ",
                )
        elif parent_channel_id is not None:
            _fail(
                DiscordEdgeErrorCode.INVALID_TARGET,
                "parent_channel_id is valid only for public guild threads",
            )
        return cls(target_type, guild_id, channel_id, parent_channel_id)

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "target_type": self.target_type.value,
            "guild_id": self.guild_id,
            "channel_id": self.channel_id,
        }
        if self.parent_channel_id is not None:
            value["parent_channel_id"] = self.parent_channel_id
        return value


def _validate_payload(
    operation: DiscordEdgeOperation,
    value: Any,
) -> dict[str, Any]:
    required, optional = _OPERATION_PAYLOAD_FIELDS[operation]
    payload = _strict_object(
        value,
        required=required,
        optional=optional,
        code=DiscordEdgeErrorCode.INVALID_PAYLOAD,
        label=f"{operation.value} payload",
    )

    if operation is DiscordEdgeOperation.PUBLIC_MESSAGE_SEND:
        payload["content"] = _validate_text(
            payload["content"],
            "content",
            allow_empty=False,
            max_chars=MAX_CONTENT_CHARS,
            max_bytes=MAX_CONTENT_BYTES,
        )
        if "reply_to_message_id" in payload:
            payload["reply_to_message_id"] = _validate_snowflake(
                payload["reply_to_message_id"],
                "reply_to_message_id",
                code=DiscordEdgeErrorCode.INVALID_PAYLOAD,
            )
    elif operation is DiscordEdgeOperation.PUBLIC_MESSAGE_EDIT:
        payload["message_id"] = _validate_snowflake(
            payload["message_id"],
            "message_id",
            code=DiscordEdgeErrorCode.INVALID_PAYLOAD,
        )
        payload["content"] = _validate_text(
            payload["content"],
            "content",
            allow_empty=False,
            max_chars=MAX_CONTENT_CHARS,
            max_bytes=MAX_CONTENT_BYTES,
        )
    else:
        payload["name"] = _validate_text(
            payload["name"],
            "name",
            allow_empty=False,
            max_chars=MAX_THREAD_NAME_CHARS,
            max_bytes=MAX_THREAD_NAME_BYTES,
        )
        if "initial_message" in payload:
            payload["initial_message"] = _validate_text(
                payload["initial_message"],
                "initial_message",
                allow_empty=True,
                max_chars=MAX_CONTENT_CHARS,
                max_bytes=MAX_CONTENT_BYTES,
            )
        if "auto_archive_minutes" in payload:
            archive = _validate_int(
                payload["auto_archive_minutes"],
                "auto_archive_minutes",
                minimum=1,
                maximum=10_080,
                code=DiscordEdgeErrorCode.INVALID_PAYLOAD,
            )
            if archive not in _AUTO_ARCHIVE_MINUTES:
                _fail(
                    DiscordEdgeErrorCode.INVALID_PAYLOAD,
                    "auto_archive_minutes is not a supported Discord value",
                )
            payload["auto_archive_minutes"] = archive
    return payload


@dataclass(frozen=True)
class DiscordEdgeIntent:
    """One exact semantic-free mutation binding."""

    operation: DiscordEdgeOperation
    target: DiscordPublicTarget
    payload: Mapping[str, Any]
    idempotency_key: str

    def __post_init__(self) -> None:
        if not isinstance(self.operation, DiscordEdgeOperation):
            _fail(DiscordEdgeErrorCode.UNKNOWN_OPERATION, "operation enum is required")
        if not isinstance(self.target, DiscordPublicTarget):
            _fail(DiscordEdgeErrorCode.INVALID_TARGET, "typed public target is required")
        _validate_operation_target(self.operation, self.target)
        normalized_payload = _validate_payload(self.operation, self.payload)
        if (
            self.operation is DiscordEdgeOperation.PUBLIC_THREAD_CREATE
            and self.target.target_type
            is DiscordPublicTargetType.PUBLIC_GUILD_FORUM
            and not normalized_payload.get("initial_message")
        ):
            _fail(
                DiscordEdgeErrorCode.INVALID_PAYLOAD,
                "public forum thread creation requires a non-empty initial_message",
            )
        if (
            self.operation is DiscordEdgeOperation.PUBLIC_THREAD_CREATE
            and self.target.target_type
            is DiscordPublicTargetType.PUBLIC_GUILD_CHANNEL
            and normalized_payload.get("initial_message")
        ):
            _fail(
                DiscordEdgeErrorCode.INVALID_PAYLOAD,
                "public channel thread creation requires a separately receipted initial message",
            )
        object.__setattr__(self, "payload", MappingProxyType(normalized_payload))
        object.__setattr__(
            self,
            "idempotency_key",
            _validate_idempotency_key(self.idempotency_key),
        )

    @classmethod
    def from_parts(
        cls,
        *,
        operation: Any,
        target: Any,
        payload: Any,
        idempotency_key: Any,
    ) -> "DiscordEdgeIntent":
        return cls(
            operation=_parse_operation(operation),
            target=DiscordPublicTarget.from_mapping(target),
            payload=payload,
            idempotency_key=_validate_idempotency_key(idempotency_key),
        )

    @property
    def content(self) -> str:
        if self.operation in {
            DiscordEdgeOperation.PUBLIC_MESSAGE_SEND,
            DiscordEdgeOperation.PUBLIC_MESSAGE_EDIT,
        }:
            return str(self.payload["content"])
        return str(self.payload.get("initial_message") or "")

    @property
    def content_sha256(self) -> str:
        return _sha256_text(self.content)

    def binding_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation.value,
            "target": self.target.to_dict(),
            "payload": dict(self.payload),
            "idempotency_key": self.idempotency_key,
        }

    @property
    def request_sha256(self) -> str:
        return _sha256_bytes(canonical_json_bytes(self.binding_dict()))


@dataclass(frozen=True)
class SignedDiscordEdgeEnvelope:
    key_id: str
    payload: Mapping[str, Any]
    signature: str

    @classmethod
    def from_mapping(
        cls,
        value: Any,
        *,
        code: DiscordEdgeErrorCode,
        label: str,
    ) -> "SignedDiscordEdgeEnvelope":
        fields = _strict_object(
            value,
            required=frozenset({"key_id", "payload", "signature"}),
            code=code,
            label=label,
        )
        key_id = fields["key_id"]
        if not isinstance(key_id, str) or not _KEY_ID_RE.fullmatch(key_id):
            _fail(code, f"{label}.key_id must be a SHA-256 key identifier")
        signature = fields["signature"]
        if not isinstance(signature, str) or not _SIGNATURE_RE.fullmatch(signature):
            _fail(code, f"{label}.signature must be an Ed25519 base64url signature")
        if not isinstance(fields["payload"], Mapping):
            _fail(code, f"{label}.payload must be an object")
        return cls(key_id, MappingProxyType(_json_copy(fields["payload"])), signature)

    def to_message(self) -> dict[str, Any]:
        return {
            "key_id": self.key_id,
            "payload": _json_copy(self.payload),
            "signature": self.signature,
        }


@dataclass(frozen=True)
class DiscordEdgeRequest:
    request_id: str
    sequence: int
    deadline_unix_ms: int
    intent: DiscordEdgeIntent
    capability: SignedDiscordEdgeEnvelope

    def to_message(self) -> dict[str, Any]:
        return {
            "protocol": PROTOCOL_VERSION,
            "request_id": self.request_id,
            "sequence": self.sequence,
            "deadline_unix_ms": self.deadline_unix_ms,
            "operation": self.intent.operation.value,
            "target": self.intent.target.to_dict(),
            "payload": dict(self.intent.payload),
            "idempotency_key": self.intent.idempotency_key,
            "capability": self.capability.to_message(),
        }


@dataclass(frozen=True)
class DiscordEdgeReconciliationQuery:
    """Exact, mutation-free lookup binding for one durable edge outcome.

    The query deliberately carries neither payload content nor a capability.
    Possession of it cannot authorize a Discord write.  The privileged edge
    may only return the already journaled request/outcome or perform the
    readback-only upgrade of a previously accepted mutation.
    """

    idempotency_key: str
    operation: DiscordEdgeOperation
    target: DiscordPublicTarget
    request_sha256: str
    content_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "idempotency_key",
            _validate_idempotency_key(self.idempotency_key),
        )
        if not isinstance(self.operation, DiscordEdgeOperation):
            _fail(
                DiscordEdgeErrorCode.UNKNOWN_OPERATION,
                "reconciliation operation enum is required",
            )
        if not isinstance(self.target, DiscordPublicTarget):
            _fail(
                DiscordEdgeErrorCode.INVALID_TARGET,
                "reconciliation requires a typed public target",
            )
        _validate_operation_target(self.operation, self.target)
        object.__setattr__(
            self,
            "request_sha256",
            _validate_sha256(
                self.request_sha256,
                "request_sha256",
                DiscordEdgeErrorCode.INVALID_SHAPE,
            ),
        )
        object.__setattr__(
            self,
            "content_sha256",
            _validate_sha256(
                self.content_sha256,
                "content_sha256",
                DiscordEdgeErrorCode.INVALID_SHAPE,
            ),
        )

    @classmethod
    def from_mapping(cls, value: Any) -> "DiscordEdgeReconciliationQuery":
        fields = _strict_object(
            value,
            required=frozenset(
                {
                    "protocol",
                    "idempotency_key",
                    "operation",
                    "target",
                    "request_sha256",
                    "content_sha256",
                }
            ),
            code=DiscordEdgeErrorCode.INVALID_SHAPE,
            label="reconciliation query",
        )
        if fields["protocol"] != RECONCILIATION_PROTOCOL_VERSION:
            _fail(
                DiscordEdgeErrorCode.UNSUPPORTED_VERSION,
                "unsupported reconciliation protocol",
            )
        return cls(
            idempotency_key=_validate_idempotency_key(fields["idempotency_key"]),
            operation=_parse_operation(fields["operation"]),
            target=DiscordPublicTarget.from_mapping(fields["target"]),
            request_sha256=_validate_sha256(
                fields["request_sha256"],
                "request_sha256",
                DiscordEdgeErrorCode.INVALID_SHAPE,
            ),
            content_sha256=_validate_sha256(
                fields["content_sha256"],
                "content_sha256",
                DiscordEdgeErrorCode.INVALID_SHAPE,
            ),
        )

    def to_message(self) -> dict[str, Any]:
        return {
            "protocol": RECONCILIATION_PROTOCOL_VERSION,
            "idempotency_key": self.idempotency_key,
            "operation": self.operation.value,
            "target": self.target.to_dict(),
            "request_sha256": self.request_sha256,
            "content_sha256": self.content_sha256,
        }

    def matches_request(self, request: DiscordEdgeRequest) -> bool:
        return (
            isinstance(request, DiscordEdgeRequest)
            and request.intent.idempotency_key == self.idempotency_key
            and request.intent.operation is self.operation
            and request.intent.target == self.target
            and request.intent.request_sha256 == self.request_sha256
            and request.intent.content_sha256 == self.content_sha256
        )


def _parse_request(
    value: Any,
    *,
    now_unix_ms: int | None = None,
    enforce_deadline_window: bool,
) -> DiscordEdgeRequest:
    fields = _strict_object(
        value,
        required=frozenset(
            {
                "protocol",
                "request_id",
                "sequence",
                "deadline_unix_ms",
                "operation",
                "target",
                "payload",
                "idempotency_key",
                "capability",
            }
        ),
        code=DiscordEdgeErrorCode.INVALID_SHAPE,
        label="request",
    )
    if fields["protocol"] != PROTOCOL_VERSION:
        _fail(DiscordEdgeErrorCode.UNSUPPORTED_VERSION, "unsupported protocol")
    request_id = _validate_uuid(
        fields["request_id"],
        "request_id",
        DiscordEdgeErrorCode.INVALID_REQUEST_ID,
    )
    sequence = _validate_int(
        fields["sequence"],
        "sequence",
        minimum=1,
        maximum=MAX_SEQUENCE,
        code=DiscordEdgeErrorCode.INVALID_SEQUENCE,
    )
    deadline = _validate_int(
        fields["deadline_unix_ms"],
        "deadline_unix_ms",
        minimum=1,
        maximum=(1 << 63) - 1,
        code=DiscordEdgeErrorCode.INVALID_DEADLINE,
    )
    if enforce_deadline_window:
        now = _now_ms() if now_unix_ms is None else now_unix_ms
        if deadline <= now:
            _fail(DiscordEdgeErrorCode.INVALID_DEADLINE, "request deadline expired")
        if deadline > now + MAX_DEADLINE_SECONDS * 1000:
            _fail(DiscordEdgeErrorCode.INVALID_DEADLINE, "request deadline is too far away")
    intent = DiscordEdgeIntent.from_parts(
        operation=fields["operation"],
        target=fields["target"],
        payload=fields["payload"],
        idempotency_key=fields["idempotency_key"],
    )
    capability = SignedDiscordEdgeEnvelope.from_mapping(
        fields["capability"],
        code=DiscordEdgeErrorCode.INVALID_CAPABILITY,
        label="capability",
    )
    return DiscordEdgeRequest(request_id, sequence, deadline, intent, capability)


def parse_request(
    value: Any,
    *,
    now_unix_ms: int | None = None,
) -> DiscordEdgeRequest:
    return _parse_request(
        value,
        now_unix_ms=now_unix_ms,
        enforce_deadline_window=True,
    )


def parse_request_for_reconciliation(value: Any) -> DiscordEdgeRequest:
    """Parse an old exact envelope without reviving its mutation authority.

    This exists only so the durable runtime can return or finish a previously
    journaled outcome after the request deadline.  The runtime still rejects
    an expired envelope when no non-prepared journal record exists.
    """

    return _parse_request(
        value,
        enforce_deadline_window=False,
    )


def parse_reconciliation_query(value: Any) -> DiscordEdgeReconciliationQuery:
    """Parse the one fixed read-only reconciliation frame."""

    return DiscordEdgeReconciliationQuery.from_mapping(value)


def make_request(
    intent: DiscordEdgeIntent,
    capability: SignedDiscordEdgeEnvelope,
    *,
    sequence: int = 1,
    timeout_seconds: int = 15,
    request_id: str | None = None,
    now_unix_ms: int | None = None,
) -> DiscordEdgeRequest:
    now = _now_ms() if now_unix_ms is None else now_unix_ms
    timeout = _validate_int(
        timeout_seconds,
        "timeout_seconds",
        minimum=1,
        maximum=MAX_DEADLINE_SECONDS,
        code=DiscordEdgeErrorCode.INVALID_DEADLINE,
    )
    return parse_request(
        {
            "protocol": PROTOCOL_VERSION,
            "request_id": request_id or str(uuid.uuid4()),
            "sequence": sequence,
            "deadline_unix_ms": now + timeout * 1000,
            "operation": intent.operation.value,
            "target": intent.target.to_dict(),
            "payload": dict(intent.payload),
            "idempotency_key": intent.idempotency_key,
            "capability": capability.to_message(),
        },
        now_unix_ms=now,
    )


def ed25519_public_key_id(public_key: Ed25519PublicKey) -> str:
    raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return _sha256_bytes(raw)


def _encode_signature(signature: bytes) -> str:
    return base64.urlsafe_b64encode(signature).rstrip(b"=").decode("ascii")


def _decode_signature(value: str) -> bytes:
    if not _SIGNATURE_RE.fullmatch(value):
        _fail(DiscordEdgeErrorCode.SIGNATURE_INVALID, "invalid signature encoding")
    try:
        decoded = base64.urlsafe_b64decode(value + "==")
    except (ValueError, TypeError) as exc:
        raise DiscordEdgeProtocolError(
            DiscordEdgeErrorCode.SIGNATURE_INVALID,
            "invalid signature encoding",
        ) from exc
    if len(decoded) != 64 or _encode_signature(decoded) != value:
        _fail(DiscordEdgeErrorCode.SIGNATURE_INVALID, "invalid Ed25519 signature length")
    return decoded


def _sign_envelope(
    private_key: Ed25519PrivateKey,
    payload: Mapping[str, Any],
    *,
    domain: bytes,
) -> SignedDiscordEdgeEnvelope:
    if not isinstance(private_key, Ed25519PrivateKey):
        raise TypeError("private_key must be Ed25519PrivateKey")
    normalized = _json_copy(payload)
    signature = private_key.sign(domain + canonical_json_bytes(normalized))
    return SignedDiscordEdgeEnvelope(
        ed25519_public_key_id(private_key.public_key()),
        MappingProxyType(normalized),
        _encode_signature(signature),
    )


def _verify_envelope(
    envelope: SignedDiscordEdgeEnvelope,
    public_key: Ed25519PublicKey,
    *,
    domain: bytes,
) -> Mapping[str, Any]:
    if not isinstance(public_key, Ed25519PublicKey):
        raise TypeError("public_key must be Ed25519PublicKey")
    if envelope.key_id != ed25519_public_key_id(public_key):
        _fail(DiscordEdgeErrorCode.SIGNATURE_INVALID, "signing key identifier mismatch")
    try:
        public_key.verify(
            _decode_signature(envelope.signature),
            domain + canonical_json_bytes(envelope.payload),
        )
    except InvalidSignature as exc:
        raise DiscordEdgeProtocolError(
            DiscordEdgeErrorCode.SIGNATURE_INVALID,
            "Ed25519 signature verification failed",
        ) from exc
    return envelope.payload


@dataclass(frozen=True)
class DiscordEdgeCapability:
    capability_id: str
    authority_kind: DiscordEdgeAuthorityKind
    authority_ref: str
    operation: DiscordEdgeOperation
    target: DiscordPublicTarget
    idempotency_key: str
    request_sha256: str
    content_sha256: str
    issued_at_unix_ms: int
    expires_at_unix_ms: int
    max_uses: int


def _parse_capability_payload(
    value: Any,
    *,
    now_unix_ms: int,
) -> DiscordEdgeCapability:
    fields = _strict_object(
        value,
        required=frozenset(
            {
                "protocol",
                "capability_id",
                "authority_kind",
                "authority_ref",
                "operation",
                "target",
                "idempotency_key",
                "request_sha256",
                "content_sha256",
                "issued_at_unix_ms",
                "expires_at_unix_ms",
                "max_uses",
            }
        ),
        code=DiscordEdgeErrorCode.INVALID_CAPABILITY,
        label="capability payload",
    )
    if fields["protocol"] != CAPABILITY_VERSION:
        _fail(DiscordEdgeErrorCode.INVALID_CAPABILITY, "unsupported capability protocol")
    capability_id = _validate_uuid(
        fields["capability_id"],
        "capability_id",
        DiscordEdgeErrorCode.INVALID_CAPABILITY,
    )
    try:
        authority_kind = DiscordEdgeAuthorityKind(fields["authority_kind"])
    except (ValueError, TypeError) as exc:
        raise DiscordEdgeProtocolError(
            DiscordEdgeErrorCode.INVALID_CAPABILITY,
            "unknown authority_kind",
        ) from exc
    operation = _parse_operation(fields["operation"])
    target = DiscordPublicTarget.from_mapping(fields["target"])
    _validate_operation_target(operation, target)
    _validate_operation_authority(operation, authority_kind)
    issued = _validate_int(
        fields["issued_at_unix_ms"],
        "issued_at_unix_ms",
        minimum=1,
        maximum=(1 << 63) - 1,
        code=DiscordEdgeErrorCode.INVALID_CAPABILITY,
    )
    expires = _validate_int(
        fields["expires_at_unix_ms"],
        "expires_at_unix_ms",
        minimum=1,
        maximum=(1 << 63) - 1,
        code=DiscordEdgeErrorCode.INVALID_CAPABILITY,
    )
    if expires <= issued or expires - issued > MAX_CAPABILITY_LIFETIME_MS:
        _fail(DiscordEdgeErrorCode.INVALID_CAPABILITY, "invalid capability lifetime")
    if issued > now_unix_ms + MAX_CLOCK_SKEW_MS:
        _fail(DiscordEdgeErrorCode.INVALID_CAPABILITY, "capability issued in the future")
    if expires <= now_unix_ms:
        _fail(DiscordEdgeErrorCode.CAPABILITY_EXPIRED, "capability expired")
    max_uses = _validate_int(
        fields["max_uses"],
        "max_uses",
        minimum=1,
        maximum=1,
        code=DiscordEdgeErrorCode.INVALID_CAPABILITY,
    )
    return DiscordEdgeCapability(
        capability_id=capability_id,
        authority_kind=authority_kind,
        authority_ref=_validate_authority_ref(fields["authority_ref"]),
        operation=operation,
        target=target,
        idempotency_key=_validate_idempotency_key(fields["idempotency_key"]),
        request_sha256=_validate_sha256(
            fields["request_sha256"],
            "request_sha256",
            DiscordEdgeErrorCode.INVALID_CAPABILITY,
        ),
        content_sha256=_validate_sha256(
            fields["content_sha256"],
            "content_sha256",
            DiscordEdgeErrorCode.INVALID_CAPABILITY,
        ),
        issued_at_unix_ms=issued,
        expires_at_unix_ms=expires,
        max_uses=max_uses,
    )


def sign_capability(
    private_key: Ed25519PrivateKey,
    intent: DiscordEdgeIntent,
    *,
    authority_kind: DiscordEdgeAuthorityKind,
    authority_ref: str,
    issued_at_unix_ms: int | None = None,
    expires_at_unix_ms: int | None = None,
    capability_id: str | None = None,
) -> SignedDiscordEdgeEnvelope:
    if not isinstance(authority_kind, DiscordEdgeAuthorityKind):
        _fail(DiscordEdgeErrorCode.INVALID_CAPABILITY, "authority_kind enum is required")
    normalized_authority_ref = _validate_authority_ref(authority_ref)
    now = _now_ms() if issued_at_unix_ms is None else issued_at_unix_ms
    expires = now + 60_000 if expires_at_unix_ms is None else expires_at_unix_ms
    payload = {
        "protocol": CAPABILITY_VERSION,
        "capability_id": capability_id or str(uuid.uuid4()),
        "authority_kind": authority_kind.value,
        "authority_ref": normalized_authority_ref,
        "operation": intent.operation.value,
        "target": intent.target.to_dict(),
        "idempotency_key": intent.idempotency_key,
        "request_sha256": intent.request_sha256,
        "content_sha256": intent.content_sha256,
        "issued_at_unix_ms": now,
        "expires_at_unix_ms": expires,
        "max_uses": 1,
    }
    _parse_capability_payload(payload, now_unix_ms=now)
    return _sign_envelope(private_key, payload, domain=_CAPABILITY_DOMAIN)


def _require_capability_binding(
    capability: DiscordEdgeCapability,
    intent: DiscordEdgeIntent,
) -> None:
    expected = {
        "operation": intent.operation.value,
        "target": intent.target.to_dict(),
        "idempotency_key": intent.idempotency_key,
        "request_sha256": intent.request_sha256,
        "content_sha256": intent.content_sha256,
    }
    actual = {
        "operation": capability.operation.value,
        "target": capability.target.to_dict(),
        "idempotency_key": capability.idempotency_key,
        "request_sha256": capability.request_sha256,
        "content_sha256": capability.content_sha256,
    }
    if actual != expected:
        _fail(
            DiscordEdgeErrorCode.CAPABILITY_BINDING_MISMATCH,
            "capability does not bind this exact operation, target, content, and idempotency key",
        )


def verify_request_capability(
    request: DiscordEdgeRequest,
    public_key: Ed25519PublicKey,
    *,
    now_unix_ms: int | None = None,
) -> DiscordEdgeCapability:
    payload = _verify_envelope(
        request.capability,
        public_key,
        domain=_CAPABILITY_DOMAIN,
    )
    capability = _parse_capability_payload(
        payload,
        now_unix_ms=_now_ms() if now_unix_ms is None else now_unix_ms,
    )
    _require_capability_binding(capability, request.intent)
    return capability


def verify_request_capability_for_reconciliation(
    request: DiscordEdgeRequest,
    public_key: Ed25519PublicKey,
) -> DiscordEdgeCapability:
    """Verify a historical one-use capability without reviving dispatch authority.

    This is only for reconciling an already durable non-``prepared`` journal
    record after its request deadline or capability expiry.  The signature,
    lifetime shape, operation authority, and exact request binding remain
    mandatory; only comparison to the current wall clock is omitted.
    """

    payload = _verify_envelope(
        request.capability,
        public_key,
        domain=_CAPABILITY_DOMAIN,
    )
    raw_issued = payload.get("issued_at_unix_ms")
    reference_time = (
        raw_issued
        if isinstance(raw_issued, int) and not isinstance(raw_issued, bool)
        else 1
    )
    capability = _parse_capability_payload(
        payload,
        now_unix_ms=reference_time,
    )
    _require_capability_binding(capability, request.intent)
    return capability


@dataclass(frozen=True)
class DiscordEdgeThreadReadback:
    """External Discord evidence for one created public guild thread."""

    target: DiscordPublicTarget
    name: str
    auto_archive_minutes: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.target, DiscordPublicTarget)
            or self.target.target_type
            is not DiscordPublicTargetType.PUBLIC_GUILD_THREAD
        ):
            _fail(
                DiscordEdgeErrorCode.INVALID_RECEIPT,
                "thread readback requires an explicit public guild thread target",
            )
        if (
            not isinstance(self.name, str)
            or not self.name
            or "\x00" in self.name
            or len(self.name) > MAX_THREAD_NAME_CHARS
            or len(self.name.encode("utf-8")) > MAX_THREAD_NAME_BYTES
        ):
            _fail(DiscordEdgeErrorCode.INVALID_RECEIPT, "invalid thread readback name")
        if (
            isinstance(self.auto_archive_minutes, bool)
            or self.auto_archive_minutes not in _AUTO_ARCHIVE_MINUTES
        ):
            _fail(
                DiscordEdgeErrorCode.INVALID_RECEIPT,
                "invalid thread readback auto-archive value",
            )

    @classmethod
    def from_mapping(cls, value: Any) -> "DiscordEdgeThreadReadback":
        fields = _strict_object(
            value,
            required=frozenset({"target", "name", "auto_archive_minutes"}),
            code=DiscordEdgeErrorCode.INVALID_RECEIPT,
            label="thread readback",
        )
        return cls(
            target=DiscordPublicTarget.from_mapping(fields["target"]),
            name=fields["name"],
            auto_archive_minutes=fields["auto_archive_minutes"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target.to_dict(),
            "name": self.name,
            "auto_archive_minutes": self.auto_archive_minutes,
        }


@dataclass(frozen=True)
class DiscordEdgeReceipt:
    receipt_id: str
    edge_request_id: str
    capability_id: str
    operation: DiscordEdgeOperation
    target: DiscordPublicTarget
    idempotency_key: str
    request_sha256: str
    content_sha256: str
    outcome: DiscordEdgeReceiptOutcome
    discord_object_id: str | None
    bot_user_id: str | None
    adapter_accepted: bool | None
    readback_verified: bool
    readback_content_sha256: str | None
    readback_thread: DiscordEdgeThreadReadback | None
    blocker_code: str | None
    occurred_at_unix_ms: int


def _parse_receipt_payload(
    value: Any,
    *,
    now_unix_ms: int,
) -> DiscordEdgeReceipt:
    fields = _strict_object(
        value,
        required=frozenset(
            {
                "protocol",
                "receipt_id",
                "edge_request_id",
                "capability_id",
                "operation",
                "target",
                "idempotency_key",
                "request_sha256",
                "content_sha256",
                "outcome",
                "discord_object_id",
                "bot_user_id",
                "adapter_accepted",
                "readback_verified",
                "readback_content_sha256",
                "readback_thread",
                "blocker_code",
                "occurred_at_unix_ms",
            }
        ),
        code=DiscordEdgeErrorCode.INVALID_RECEIPT,
        label="receipt payload",
    )
    if fields["protocol"] != RECEIPT_VERSION:
        _fail(DiscordEdgeErrorCode.INVALID_RECEIPT, "unsupported receipt protocol")
    operation = _parse_operation(fields["operation"])
    target = DiscordPublicTarget.from_mapping(fields["target"])
    _validate_operation_target(operation, target)
    try:
        outcome = DiscordEdgeReceiptOutcome(fields["outcome"])
    except (ValueError, TypeError) as exc:
        raise DiscordEdgeProtocolError(
            DiscordEdgeErrorCode.INVALID_RECEIPT,
            "unknown receipt outcome",
        ) from exc
    accepted = fields["adapter_accepted"]
    readback = fields["readback_verified"]
    if (accepted is not None and not isinstance(accepted, bool)) or not isinstance(
        readback,
        bool,
    ):
        _fail(
            DiscordEdgeErrorCode.INVALID_RECEIPT,
            "receipt acceptance must be true, false, or null and readback must be boolean",
        )
    raw_object_id = fields["discord_object_id"]
    object_id = (
        _validate_snowflake(
            raw_object_id,
            "discord_object_id",
            code=DiscordEdgeErrorCode.INVALID_RECEIPT,
        )
        if raw_object_id is not None
        else None
    )
    raw_bot_user_id = fields["bot_user_id"]
    bot_user_id = (
        _validate_snowflake(
            raw_bot_user_id,
            "bot_user_id",
            code=DiscordEdgeErrorCode.INVALID_RECEIPT,
        )
        if raw_bot_user_id is not None
        else None
    )
    raw_readback_sha256 = fields["readback_content_sha256"]
    readback_content_sha256 = (
        _validate_sha256(
            raw_readback_sha256,
            "readback_content_sha256",
            DiscordEdgeErrorCode.INVALID_RECEIPT,
        )
        if raw_readback_sha256 is not None
        else None
    )
    raw_thread_readback = fields["readback_thread"]
    thread_readback = (
        DiscordEdgeThreadReadback.from_mapping(raw_thread_readback)
        if raw_thread_readback is not None
        else None
    )
    if thread_readback is not None and (
        object_id is None
        or thread_readback.target.channel_id != object_id
        or thread_readback.target.guild_id != target.guild_id
        or thread_readback.target.parent_channel_id != target.channel_id
    ):
        _fail(
            DiscordEdgeErrorCode.INVALID_RECEIPT,
            "thread readback does not bind the returned object, guild, and parent",
        )
    blocker = fields["blocker_code"]
    if blocker is not None and (
        not isinstance(blocker, str) or not _BLOCKER_CODE_RE.fullmatch(blocker)
    ):
        _fail(DiscordEdgeErrorCode.INVALID_RECEIPT, "invalid blocker_code")

    if outcome is DiscordEdgeReceiptOutcome.VERIFIED:
        if (
            accepted is not True
            or readback is not True
            or object_id is None
            or bot_user_id is None
            or readback_content_sha256 != fields["content_sha256"]
            or (
                operation is DiscordEdgeOperation.PUBLIC_THREAD_CREATE
                and thread_readback is None
            )
            or (
                operation is not DiscordEdgeOperation.PUBLIC_THREAD_CREATE
                and thread_readback is not None
            )
            or blocker is not None
        ):
            _fail(
                DiscordEdgeErrorCode.INVALID_RECEIPT,
                "verified receipt requires exact bot-authored content readback",
            )
    elif outcome is DiscordEdgeReceiptOutcome.ACCEPTED_UNVERIFIED:
        if (
            accepted is not True
            or readback is not False
            or bot_user_id is None
            or readback_content_sha256 is not None
            or thread_readback is not None
            or blocker is None
        ):
            _fail(
                DiscordEdgeErrorCode.INVALID_RECEIPT,
                "accepted_unverified receipt has inconsistent evidence",
            )
    elif outcome is DiscordEdgeReceiptOutcome.DISPATCH_UNCERTAIN:
        if (
            accepted is not None
            or readback is not False
            or object_id is not None
            or bot_user_id is not None
            or readback_content_sha256 is not None
            or thread_readback is not None
            or blocker is None
        ):
            _fail(
                DiscordEdgeErrorCode.INVALID_RECEIPT,
                "dispatch_uncertain receipt requires unknown acceptance and no claimed evidence",
            )
    elif (
        accepted is not False
        or readback is not False
        or object_id is not None
        or bot_user_id is not None
        or readback_content_sha256 is not None
        or thread_readback is not None
        or blocker is None
    ):
        _fail(
            DiscordEdgeErrorCode.INVALID_RECEIPT,
            "pre-dispatch failure receipt has inconsistent evidence",
        )

    occurred = _validate_int(
        fields["occurred_at_unix_ms"],
        "occurred_at_unix_ms",
        minimum=1,
        maximum=(1 << 63) - 1,
        code=DiscordEdgeErrorCode.INVALID_RECEIPT,
    )
    if occurred > now_unix_ms + MAX_CLOCK_SKEW_MS:
        _fail(DiscordEdgeErrorCode.INVALID_RECEIPT, "receipt occurred in the future")
    return DiscordEdgeReceipt(
        receipt_id=_validate_uuid(
            fields["receipt_id"],
            "receipt_id",
            DiscordEdgeErrorCode.INVALID_RECEIPT,
        ),
        edge_request_id=_validate_uuid(
            fields["edge_request_id"],
            "edge_request_id",
            DiscordEdgeErrorCode.INVALID_RECEIPT,
        ),
        capability_id=_validate_uuid(
            fields["capability_id"],
            "capability_id",
            DiscordEdgeErrorCode.INVALID_RECEIPT,
        ),
        operation=operation,
        target=target,
        idempotency_key=_validate_idempotency_key(fields["idempotency_key"]),
        request_sha256=_validate_sha256(
            fields["request_sha256"],
            "request_sha256",
            DiscordEdgeErrorCode.INVALID_RECEIPT,
        ),
        content_sha256=_validate_sha256(
            fields["content_sha256"],
            "content_sha256",
            DiscordEdgeErrorCode.INVALID_RECEIPT,
        ),
        outcome=outcome,
        discord_object_id=object_id,
        bot_user_id=bot_user_id,
        adapter_accepted=accepted,
        readback_verified=readback,
        readback_content_sha256=readback_content_sha256,
        readback_thread=thread_readback,
        blocker_code=blocker,
        occurred_at_unix_ms=occurred,
    )


def sign_receipt(
    private_key: Ed25519PrivateKey,
    request: DiscordEdgeRequest,
    capability: DiscordEdgeCapability,
    *,
    outcome: DiscordEdgeReceiptOutcome,
    discord_object_id: str | None,
    bot_user_id: str | None,
    adapter_accepted: bool | None,
    readback_verified: bool,
    readback_content_sha256: str | None,
    readback_thread: DiscordEdgeThreadReadback | None = None,
    blocker_code: str | None = None,
    occurred_at_unix_ms: int | None = None,
    receipt_id: str | None = None,
) -> SignedDiscordEdgeEnvelope:
    if not isinstance(outcome, DiscordEdgeReceiptOutcome):
        _fail(DiscordEdgeErrorCode.INVALID_RECEIPT, "receipt outcome enum is required")
    if readback_thread is not None and not isinstance(
        readback_thread,
        DiscordEdgeThreadReadback,
    ):
        _fail(
            DiscordEdgeErrorCode.INVALID_RECEIPT,
            "readback_thread must be typed external thread evidence",
        )
    _require_capability_binding(capability, request.intent)
    occurred = _now_ms() if occurred_at_unix_ms is None else occurred_at_unix_ms
    payload = {
        "protocol": RECEIPT_VERSION,
        "receipt_id": receipt_id or str(uuid.uuid4()),
        "edge_request_id": request.request_id,
        "capability_id": capability.capability_id,
        "operation": request.intent.operation.value,
        "target": request.intent.target.to_dict(),
        "idempotency_key": request.intent.idempotency_key,
        "request_sha256": request.intent.request_sha256,
        "content_sha256": request.intent.content_sha256,
        "outcome": outcome.value,
        "discord_object_id": discord_object_id,
        "bot_user_id": bot_user_id,
        "adapter_accepted": adapter_accepted,
        "readback_verified": readback_verified,
        "readback_content_sha256": readback_content_sha256,
        "readback_thread": (
            readback_thread.to_dict() if readback_thread is not None else None
        ),
        "blocker_code": blocker_code,
        "occurred_at_unix_ms": occurred,
    }
    parsed_receipt = _parse_receipt_payload(payload, now_unix_ms=occurred)
    _require_receipt_binding(parsed_receipt, request, capability)
    return _sign_envelope(private_key, payload, domain=_RECEIPT_DOMAIN)


def _require_receipt_binding(
    receipt: DiscordEdgeReceipt,
    request: DiscordEdgeRequest,
    capability: DiscordEdgeCapability | None,
) -> None:
    expected = {
        "edge_request_id": request.request_id,
        "operation": request.intent.operation.value,
        "target": request.intent.target.to_dict(),
        "idempotency_key": request.intent.idempotency_key,
        "request_sha256": request.intent.request_sha256,
        "content_sha256": request.intent.content_sha256,
    }
    actual = {
        "edge_request_id": receipt.edge_request_id,
        "operation": receipt.operation.value,
        "target": receipt.target.to_dict(),
        "idempotency_key": receipt.idempotency_key,
        "request_sha256": receipt.request_sha256,
        "content_sha256": receipt.content_sha256,
    }
    if actual != expected or (
        capability is not None and receipt.capability_id != capability.capability_id
    ):
        _fail(
            DiscordEdgeErrorCode.RECEIPT_BINDING_MISMATCH,
            "receipt does not bind this exact request and capability",
        )
    if (
        request.intent.operation is DiscordEdgeOperation.PUBLIC_THREAD_CREATE
        and receipt.outcome is DiscordEdgeReceiptOutcome.VERIFIED
    ):
        evidence = receipt.readback_thread
        requested_archive = request.intent.payload.get("auto_archive_minutes")
        if (
            evidence is None
            or evidence.name != request.intent.payload["name"]
            or (
                requested_archive is not None
                and evidence.auto_archive_minutes != requested_archive
            )
        ):
            _fail(
                DiscordEdgeErrorCode.RECEIPT_BINDING_MISMATCH,
                "verified thread receipt lacks exact external name/archive evidence",
            )


def verify_receipt(
    envelope: SignedDiscordEdgeEnvelope,
    public_key: Ed25519PublicKey,
    *,
    expected_request: DiscordEdgeRequest,
    expected_capability: DiscordEdgeCapability | None = None,
    now_unix_ms: int | None = None,
) -> DiscordEdgeReceipt:
    payload = _verify_envelope(envelope, public_key, domain=_RECEIPT_DOMAIN)
    receipt = _parse_receipt_payload(
        payload,
        now_unix_ms=_now_ms() if now_unix_ms is None else now_unix_ms,
    )
    _require_receipt_binding(receipt, expected_request, expected_capability)
    return receipt


__all__ = [
    "CAPABILITY_VERSION",
    "MAX_REQUEST_BYTES",
    "PROTOCOL_VERSION",
    "RECONCILIATION_PROTOCOL_VERSION",
    "RECONCILIATION_NOT_AVAILABLE_ERROR",
    "RECONCILIATION_RESPONSE_VERSION",
    "RECEIPT_VERSION",
    "DiscordEdgeAuthorityKind",
    "DiscordEdgeCapability",
    "DiscordEdgeErrorCode",
    "DiscordEdgeIntent",
    "DiscordEdgeOperation",
    "DiscordEdgeProtocolError",
    "DiscordEdgeReceipt",
    "DiscordEdgeReceiptOutcome",
    "DiscordEdgeReconciliationQuery",
    "DiscordEdgeRequest",
    "DiscordEdgeThreadReadback",
    "DiscordPublicTarget",
    "DiscordPublicTargetType",
    "SignedDiscordEdgeEnvelope",
    "canonical_json_bytes",
    "decode_request_json",
    "ed25519_public_key_id",
    "make_request",
    "parse_request",
    "parse_request_for_reconciliation",
    "parse_reconciliation_query",
    "sign_capability",
    "sign_receipt",
    "verify_receipt",
    "verify_request_capability",
    "verify_request_capability_for_reconciliation",
]
