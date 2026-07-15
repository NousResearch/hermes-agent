"""Strict protocol for credential-scoped operational edge services.

The model selects an explicit operation identifier and authors its arguments.
This module never reads prose, infers intent, or chooses a domain.  It only
validates exact JSON shapes, binds arguments to an operation capability, and
signs/verifies execution receipts.

Read-only calls are authenticated by the Unix peer boundary.  Mutation calls
also require a short-lived Ed25519 capability issued by the privileged
Canonical Writer for the exact operation, arguments, and idempotency key.
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
from typing import Any, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


PROTOCOL_SCHEMA = "muncho-operational-edge-request.v1"
CAPABILITY_SCHEMA = "muncho-operational-edge-capability.v1"
RECEIPT_SCHEMA = "muncho-operational-edge-receipt.v2"
PREDISPATCH_MUTATION_BLOCKERS = frozenset(
    {"mutation_capability_required", "mutation_capability_invalid"}
)
SIGNED_ENVELOPE_SCHEMA = "muncho-ed25519-envelope.v1"
COMMAND_AUTHORIZATION_SCHEMA = "muncho-operational-edge-command-authorization.v1"

MAX_REQUEST_BYTES = 256 * 1024
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_ARGUMENT_BYTES = 128 * 1024
MAX_DEADLINE_SECONDS = 60
MAX_CAPABILITY_SECONDS = 8 * 60 * 60

_OPERATION = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z0-9_]+){1,7}$")
_IDEMPOTENCY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_KEY_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class OperationalAccess(StrEnum):
    READ = "read"
    MECHANICAL = "mechanical"
    MUTATION = "mutation"


class OperationalOutcome(StrEnum):
    SUCCEEDED = "succeeded"
    BLOCKED = "blocked"
    DISPATCH_UNCERTAIN = "dispatch_uncertain"


class OperationalProtocolError(ValueError):
    """One stable, secret-free protocol failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _fail(code: str) -> None:
    raise OperationalProtocolError(code)


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            dict(value),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise OperationalProtocolError("invalid_json_value") from exc


def sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def operational_command_sha256(intent: "OperationalIntent") -> str:
    """Domain-separate the exact intent hash stored in owner plan approvals."""

    if not isinstance(intent, OperationalIntent):
        _fail("invalid_intent")
    return sha256_json(
        {
            "schema": COMMAND_AUTHORIZATION_SCHEMA,
            "intent": intent.to_mapping(),
        }
    )


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate_key")
        result[key] = value
    return result


def decode_json_object(raw: bytes, *, maximum: int) -> dict[str, Any]:
    if not isinstance(raw, bytes) or not raw or len(raw) > maximum:
        _fail("invalid_json_frame")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise OperationalProtocolError("invalid_json_frame") from exc
    if not isinstance(value, dict):
        _fail("invalid_json_frame")
    return value


def _exact(value: Any, fields: frozenset[str], code: str) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != fields
    ):
        _fail(code)
    return dict(value)


def _uuid4(value: Any) -> str:
    try:
        parsed = uuid.UUID(str(value))
    except (TypeError, ValueError, AttributeError) as exc:
        raise OperationalProtocolError("invalid_request_id") from exc
    if parsed.version != 4 or str(parsed) != value:
        _fail("invalid_request_id")
    return str(parsed)


def _operation(value: Any) -> str:
    if not isinstance(value, str) or _OPERATION.fullmatch(value) is None:
        _fail("invalid_operation_id")
    return value


def _idempotency(value: Any) -> str:
    if not isinstance(value, str) or _IDEMPOTENCY.fullmatch(value) is None:
        _fail("invalid_idempotency_key")
    return value


def _digest(value: Any, code: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(code)
    return value


def _timestamp(value: Any, code: str) -> int:
    if type(value) is not int or value < 1:
        _fail(code)
    return value


@dataclass(frozen=True)
class OperationalIntent:
    operation_id: str
    arguments: Mapping[str, Any]
    arguments_sha256: str
    idempotency_key: str

    @classmethod
    def from_mapping(cls, value: Any) -> "OperationalIntent":
        raw = _exact(
            value,
            frozenset(
                {
                    "operation_id",
                    "arguments",
                    "arguments_sha256",
                    "idempotency_key",
                }
            ),
            "invalid_intent",
        )
        operation_id = _operation(raw["operation_id"])
        arguments = raw["arguments"]
        if (
            not isinstance(arguments, Mapping)
            or any(not isinstance(key, str) for key in arguments)
            or len(canonical_json_bytes(dict(arguments))) > MAX_ARGUMENT_BYTES
        ):
            _fail("invalid_arguments")
        digest = _digest(raw["arguments_sha256"], "invalid_arguments_sha256")
        if sha256_json(dict(arguments)) != digest:
            _fail("arguments_sha256_mismatch")
        return cls(
            operation_id=operation_id,
            arguments=dict(arguments),
            arguments_sha256=digest,
            idempotency_key=_idempotency(raw["idempotency_key"]),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "arguments": dict(self.arguments),
            "arguments_sha256": self.arguments_sha256,
            "idempotency_key": self.idempotency_key,
        }


@dataclass(frozen=True)
class OperationalCapability:
    authority_kind: str
    authority_ref: str
    operation_id: str
    arguments_sha256: str
    idempotency_key: str
    issued_at_unix_ms: int
    expires_at_unix_ms: int

    @classmethod
    def from_mapping(cls, value: Any) -> "OperationalCapability":
        raw = _exact(
            value,
            frozenset(
                {
                    "schema",
                    "authority_kind",
                    "authority_ref",
                    "operation_id",
                    "arguments_sha256",
                    "idempotency_key",
                    "issued_at_unix_ms",
                    "expires_at_unix_ms",
                }
            ),
            "invalid_capability",
        )
        if raw["schema"] != CAPABILITY_SCHEMA:
            _fail("invalid_capability_schema")
        if raw["authority_kind"] != "canonical_owner_plan":
            _fail("invalid_capability_authority")
        authority_ref = raw["authority_ref"]
        if (
            not isinstance(authority_ref, str)
            or not authority_ref
            or len(authority_ref) > 240
            or "\x00" in authority_ref
        ):
            _fail("invalid_capability_authority")
        issued = _timestamp(raw["issued_at_unix_ms"], "invalid_capability_time")
        expires = _timestamp(raw["expires_at_unix_ms"], "invalid_capability_time")
        if expires <= issued or expires - issued > MAX_CAPABILITY_SECONDS * 1000:
            _fail("invalid_capability_time")
        return cls(
            authority_kind="canonical_owner_plan",
            authority_ref=authority_ref,
            operation_id=_operation(raw["operation_id"]),
            arguments_sha256=_digest(
                raw["arguments_sha256"], "invalid_arguments_sha256"
            ),
            idempotency_key=_idempotency(raw["idempotency_key"]),
            issued_at_unix_ms=issued,
            expires_at_unix_ms=expires,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": CAPABILITY_SCHEMA,
            "authority_kind": self.authority_kind,
            "authority_ref": self.authority_ref,
            "operation_id": self.operation_id,
            "arguments_sha256": self.arguments_sha256,
            "idempotency_key": self.idempotency_key,
            "issued_at_unix_ms": self.issued_at_unix_ms,
            "expires_at_unix_ms": self.expires_at_unix_ms,
        }

    def require(self, intent: OperationalIntent, *, now_unix_ms: int) -> None:
        if (
            self.operation_id != intent.operation_id
            or self.arguments_sha256 != intent.arguments_sha256
            or self.idempotency_key != intent.idempotency_key
        ):
            _fail("capability_intent_mismatch")
        if now_unix_ms < self.issued_at_unix_ms or now_unix_ms >= self.expires_at_unix_ms:
            _fail("capability_expired")


@dataclass(frozen=True)
class SignedEnvelope:
    key_id: str
    payload: Mapping[str, Any]
    signature_b64: str

    @classmethod
    def from_mapping(cls, value: Any, *, code: str) -> "SignedEnvelope":
        raw = _exact(
            value,
            frozenset({"schema", "key_id", "payload", "signature_b64"}),
            code,
        )
        if raw["schema"] != SIGNED_ENVELOPE_SCHEMA:
            _fail(code)
        key_id = raw["key_id"]
        if not isinstance(key_id, str) or _KEY_ID.fullmatch(key_id) is None:
            _fail(code)
        payload = raw["payload"]
        if not isinstance(payload, Mapping):
            _fail(code)
        signature = raw["signature_b64"]
        if not isinstance(signature, str):
            _fail(code)
        try:
            decoded = base64.b64decode(signature, validate=True)
        except (ValueError, TypeError) as exc:
            raise OperationalProtocolError(code) from exc
        if len(decoded) != 64:
            _fail(code)
        return cls(key_id=key_id, payload=dict(payload), signature_b64=signature)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": SIGNED_ENVELOPE_SCHEMA,
            "key_id": self.key_id,
            "payload": dict(self.payload),
            "signature_b64": self.signature_b64,
        }


def sign_envelope(
    payload: Mapping[str, Any],
    *,
    key_id: str,
    private_key: Ed25519PrivateKey,
) -> SignedEnvelope:
    if not isinstance(key_id, str) or _KEY_ID.fullmatch(key_id) is None:
        _fail("invalid_signing_key_id")
    if not isinstance(private_key, Ed25519PrivateKey):
        _fail("invalid_signing_key")
    signature = private_key.sign(canonical_json_bytes(payload))
    return SignedEnvelope(
        key_id=key_id,
        payload=dict(payload),
        signature_b64=base64.b64encode(signature).decode("ascii"),
    )


def verify_envelope(
    value: Any,
    *,
    key_id: str,
    public_key: Ed25519PublicKey,
    code: str,
) -> Mapping[str, Any]:
    envelope = SignedEnvelope.from_mapping(value, code=code)
    if envelope.key_id != key_id or not isinstance(public_key, Ed25519PublicKey):
        _fail(code)
    try:
        public_key.verify(
            base64.b64decode(envelope.signature_b64, validate=True),
            canonical_json_bytes(envelope.payload),
        )
    except (InvalidSignature, ValueError, TypeError) as exc:
        raise OperationalProtocolError(code) from exc
    return dict(envelope.payload)


@dataclass(frozen=True)
class OperationalRequest:
    request_id: str
    sequence: int
    deadline_unix_ms: int
    intent: OperationalIntent
    capability: SignedEnvelope | None

    @classmethod
    def from_mapping(
        cls,
        value: Any,
        *,
        now_unix_ms: int | None = None,
    ) -> "OperationalRequest":
        raw = _exact(
            value,
            frozenset(
                {
                    "schema",
                    "request_id",
                    "sequence",
                    "deadline_unix_ms",
                    "intent",
                    "capability",
                }
            ),
            "invalid_request",
        )
        if raw["schema"] != PROTOCOL_SCHEMA:
            _fail("invalid_request_schema")
        sequence = raw["sequence"]
        if type(sequence) is not int or not 0 <= sequence < (1 << 63):
            _fail("invalid_sequence")
        deadline = _timestamp(raw["deadline_unix_ms"], "invalid_deadline")
        now = int(time.time() * 1000) if now_unix_ms is None else now_unix_ms
        if deadline <= now or deadline > now + MAX_DEADLINE_SECONDS * 1000:
            _fail("invalid_deadline")
        capability = (
            None
            if raw["capability"] is None
            else SignedEnvelope.from_mapping(
                raw["capability"], code="invalid_capability_envelope"
            )
        )
        return cls(
            request_id=_uuid4(raw["request_id"]),
            sequence=sequence,
            deadline_unix_ms=deadline,
            intent=OperationalIntent.from_mapping(raw["intent"]),
            capability=capability,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": PROTOCOL_SCHEMA,
            "request_id": self.request_id,
            "sequence": self.sequence,
            "deadline_unix_ms": self.deadline_unix_ms,
            "intent": self.intent.to_mapping(),
            "capability": self.capability.to_mapping() if self.capability else None,
        }


def verify_mutation_capability(
    request: OperationalRequest,
    *,
    key_id: str,
    public_key: Ed25519PublicKey,
    now_unix_ms: int | None = None,
) -> OperationalCapability:
    if request.capability is None:
        _fail("mutation_capability_required")
    payload = verify_envelope(
        request.capability.to_mapping(),
        key_id=key_id,
        public_key=public_key,
        code="mutation_capability_invalid",
    )
    capability = OperationalCapability.from_mapping(payload)
    capability.require(
        request.intent,
        now_unix_ms=(
            int(time.time() * 1000) if now_unix_ms is None else now_unix_ms
        ),
    )
    return capability


def receipt_payload(
    *,
    request: OperationalRequest,
    domain: str,
    service_unit: str,
    release_revision: str,
    request_sha256: str,
    access: OperationalAccess,
    outcome: OperationalOutcome,
    service_pid: int,
    executable_sha256: str,
    return_code: int | None,
    stdout_b64: str,
    stderr_b64: str,
    started_at_unix_ms: int,
    finished_at_unix_ms: int,
    blocker_code: str | None,
    dispatched: bool,
    executable_started: bool,
    mutation_performed: bool | None,
    readback_verified: bool,
) -> dict[str, Any]:
    return {
        "schema": RECEIPT_SCHEMA,
        "request_id": request.request_id,
        "operation_id": request.intent.operation_id,
        "arguments_sha256": request.intent.arguments_sha256,
        "idempotency_key": request.intent.idempotency_key,
        "domain": domain,
        "service_unit": service_unit,
        "release_revision": release_revision,
        "request_sha256": request_sha256,
        "access": access.value,
        "outcome": outcome.value,
        "service_pid": service_pid,
        "executable_sha256": executable_sha256,
        "return_code": return_code,
        "stdout_b64": stdout_b64,
        "stderr_b64": stderr_b64,
        "started_at_unix_ms": started_at_unix_ms,
        "finished_at_unix_ms": finished_at_unix_ms,
        "blocker_code": blocker_code,
        "dispatched": dispatched,
        "executable_started": executable_started,
        "mutation_performed": mutation_performed,
        "readback_verified": readback_verified,
        "secret_material_recorded": False,
    }


__all__ = [
    "CAPABILITY_SCHEMA",
    "COMMAND_AUTHORIZATION_SCHEMA",
    "MAX_REQUEST_BYTES",
    "MAX_RESPONSE_BYTES",
    "OperationalAccess",
    "OperationalCapability",
    "OperationalIntent",
    "OperationalOutcome",
    "OperationalProtocolError",
    "OperationalRequest",
    "PREDISPATCH_MUTATION_BLOCKERS",
    "RECEIPT_SCHEMA",
    "SignedEnvelope",
    "canonical_json_bytes",
    "decode_json_object",
    "receipt_payload",
    "operational_command_sha256",
    "sha256_json",
    "sign_envelope",
    "verify_envelope",
    "verify_mutation_capability",
]
