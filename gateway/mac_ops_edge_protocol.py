"""Strict mechanical protocol for the privileged Mac operations edge.

The model authors the task contract and selects one explicit read-only task
class.  This module does not inspect prose, choose a worker, infer intent, or
manufacture approvals.  It only validates exact JSON shapes, bounds, hashes,
deadlines, and idempotency identifiers.

The first production-shaped capability gate intentionally exposes read-only
Mac/browser work only.  Mutation authority is a separate future protocol and
cannot be self-asserted through this one.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping


PROTOCOL_VERSION = "muncho-mac-ops-edge.v1"
RESPONSE_VERSION = "muncho-mac-ops-edge-response.v1"
RECEIPT_VERSION = "muncho-mac-ops-edge-receipt.v1"

MAX_REQUEST_BYTES = 64 * 1024
MAX_RESPONSE_BYTES = 256 * 1024
MAX_CONTRACT_BYTES = 48 * 1024
MAX_TITLE_CHARS = 240
MAX_IDEMPOTENCY_KEY_BYTES = 240
MAX_DEADLINE_SECONDS = 30

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDEMPOTENCY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$")
_BLOCKER_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$")


class MacOpsEdgeOperation(StrEnum):
    PING = "ping"
    READONLY_SUBMIT = "readonly.submit"
    TASK_READ = "task.read"


class MacOpsReadOnlyClass(StrEnum):
    DISCOVERY = "readonly.discovery"
    BROWSER = "readonly.browser"
    LOCAL_FILES = "readonly.local_files"
    CLI_STATUS = "readonly.cli_status"
    CODE = "code.readonly"


class MacOpsEdgeState(StrEnum):
    QUEUED = "queued"
    OBSERVED = "observed"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    DISPATCH_UNCERTAIN = "dispatch_uncertain"


class MacOpsEdgeProtocolError(ValueError):
    """One stable validation failure that contains no task or secret value."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _fail(code: str) -> None:
    raise MacOpsEdgeProtocolError(code)


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            dict(value),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MacOpsEdgeProtocolError("invalid_json_value") from exc


def sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _reject_constant(_value: str) -> None:
    raise ValueError("non_json_number")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
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
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise MacOpsEdgeProtocolError("invalid_json_frame") from exc
    if not isinstance(value, dict):
        _fail("invalid_json_frame")
    return value


def _strict_mapping(
    value: Any,
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
    code: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        _fail(code)
    result = dict(value)
    if set(result) - required - optional or required - set(result):
        _fail(code)
    return result


def _text(value: Any, *, maximum_chars: int, code: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > maximum_chars
        or "\x00" in value
    ):
        _fail(code)
    return value


def _sha256(value: Any, code: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        _fail(code)
    return value


def _idempotency_key(value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value.encode("utf-8")) > MAX_IDEMPOTENCY_KEY_BYTES
        or _IDEMPOTENCY_RE.fullmatch(value) is None
    ):
        _fail("invalid_idempotency_key")
    return value


def _request_id(value: Any) -> str:
    try:
        parsed = uuid.UUID(str(value))
    except (TypeError, ValueError, AttributeError) as exc:
        raise MacOpsEdgeProtocolError("invalid_request_id") from exc
    if parsed.version != 4 or str(parsed) != value:
        _fail("invalid_request_id")
    return str(parsed)


def _sequence(value: Any) -> int:
    if type(value) is not int or not 0 <= value < (1 << 63):
        _fail("invalid_sequence")
    return value


def _deadline(value: Any, *, now_ms: int) -> int:
    if type(value) is not int:
        _fail("invalid_deadline")
    if value <= now_ms or value > now_ms + MAX_DEADLINE_SECONDS * 1000:
        _fail("invalid_deadline")
    return value


@dataclass(frozen=True)
class MacOpsReadOnlySubmit:
    title: str
    task_class: MacOpsReadOnlyClass
    contract: str
    contract_sha256: str

    @classmethod
    def from_mapping(cls, value: Any) -> "MacOpsReadOnlySubmit":
        item = _strict_mapping(
            value,
            required=frozenset(
                {"title", "task_class", "contract", "contract_sha256"}
            ),
            code="invalid_submit_payload",
        )
        title = _text(
            item["title"], maximum_chars=MAX_TITLE_CHARS, code="invalid_title"
        )
        try:
            task_class = MacOpsReadOnlyClass(item["task_class"])
        except (TypeError, ValueError) as exc:
            raise MacOpsEdgeProtocolError("invalid_readonly_task_class") from exc
        contract = _text(
            item["contract"],
            maximum_chars=MAX_CONTRACT_BYTES,
            code="invalid_contract",
        )
        if len(contract.encode("utf-8")) > MAX_CONTRACT_BYTES:
            _fail("invalid_contract")
        digest = _sha256(item["contract_sha256"], "invalid_contract_sha256")
        if hashlib.sha256(contract.encode("utf-8")).hexdigest() != digest:
            _fail("contract_sha256_mismatch")
        return cls(
            title=title,
            task_class=task_class,
            contract=contract,
            contract_sha256=digest,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "task_class": self.task_class.value,
            "contract": self.contract,
            "contract_sha256": self.contract_sha256,
        }


@dataclass(frozen=True)
class MacOpsTaskRead:
    issue_iid: int

    @classmethod
    def from_mapping(cls, value: Any) -> "MacOpsTaskRead":
        item = _strict_mapping(
            value,
            required=frozenset({"issue_iid"}),
            code="invalid_read_payload",
        )
        issue_iid = item["issue_iid"]
        if type(issue_iid) is not int or not 1 <= issue_iid < (1 << 63):
            _fail("invalid_issue_iid")
        return cls(issue_iid=issue_iid)

    def to_mapping(self) -> dict[str, Any]:
        return {"issue_iid": self.issue_iid}


@dataclass(frozen=True)
class MacOpsPing:
    nonce: str

    @classmethod
    def from_mapping(cls, value: Any) -> "MacOpsPing":
        item = _strict_mapping(
            value,
            required=frozenset({"nonce"}),
            code="invalid_ping_payload",
        )
        nonce = item["nonce"]
        if not isinstance(nonce, str) or _SHA256_RE.fullmatch(nonce) is None:
            _fail("invalid_ping_nonce")
        return cls(nonce=nonce)

    def to_mapping(self) -> dict[str, Any]:
        return {"nonce": self.nonce}


Payload = MacOpsPing | MacOpsReadOnlySubmit | MacOpsTaskRead


@dataclass(frozen=True)
class MacOpsEdgeRequest:
    request_id: str
    sequence: int
    deadline_unix_ms: int
    operation: MacOpsEdgeOperation
    idempotency_key: str
    payload: Payload

    @classmethod
    def from_mapping(
        cls,
        value: Any,
        *,
        now_ms: int | None = None,
    ) -> "MacOpsEdgeRequest":
        item = _strict_mapping(
            value,
            required=frozenset(
                {
                    "protocol",
                    "request_id",
                    "sequence",
                    "deadline_unix_ms",
                    "operation",
                    "idempotency_key",
                    "payload",
                }
            ),
            code="invalid_request_shape",
        )
        if item["protocol"] != PROTOCOL_VERSION:
            _fail("unsupported_protocol")
        current = int(time.time() * 1000) if now_ms is None else int(now_ms)
        try:
            operation = MacOpsEdgeOperation(item["operation"])
        except (TypeError, ValueError) as exc:
            raise MacOpsEdgeProtocolError("unknown_operation") from exc
        if operation is MacOpsEdgeOperation.PING:
            payload: Payload = MacOpsPing.from_mapping(item["payload"])
        elif operation is MacOpsEdgeOperation.READONLY_SUBMIT:
            payload: Payload = MacOpsReadOnlySubmit.from_mapping(item["payload"])
        else:
            payload = MacOpsTaskRead.from_mapping(item["payload"])
        return cls(
            request_id=_request_id(item["request_id"]),
            sequence=_sequence(item["sequence"]),
            deadline_unix_ms=_deadline(item["deadline_unix_ms"], now_ms=current),
            operation=operation,
            idempotency_key=_idempotency_key(item["idempotency_key"]),
            payload=payload,
        )

    @property
    def request_sha256(self) -> str:
        return sha256_json(self.to_mapping())

    @property
    def intent_sha256(self) -> str:
        """Stable binding for safe retries with fresh transport envelopes."""

        return sha256_json(
            {
                "operation": self.operation.value,
                "idempotency_key": self.idempotency_key,
                "payload": self.payload.to_mapping(),
            }
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "protocol": PROTOCOL_VERSION,
            "request_id": self.request_id,
            "sequence": self.sequence,
            "deadline_unix_ms": self.deadline_unix_ms,
            "operation": self.operation.value,
            "idempotency_key": self.idempotency_key,
            "payload": self.payload.to_mapping(),
        }


@dataclass(frozen=True)
class MacOpsEdgeReceipt:
    value: Mapping[str, Any]

    @classmethod
    def build(
        cls,
        *,
        request: MacOpsEdgeRequest,
        state: MacOpsEdgeState,
        issue_iid: int | None,
        external_updated_at: str | None,
        service_identity_sha256: str,
        recorded_at_unix_ms: int,
    ) -> "MacOpsEdgeReceipt":
        _sha256(service_identity_sha256, "invalid_service_identity")
        if issue_iid is not None and (
            type(issue_iid) is not int or issue_iid < 1
        ):
            _fail("invalid_issue_iid")
        unsigned = {
            "schema": RECEIPT_VERSION,
            "request_id": request.request_id,
            "request_sha256": request.request_sha256,
            "intent_sha256": request.intent_sha256,
            "operation": request.operation.value,
            "idempotency_key": request.idempotency_key,
            "state": state.value,
            "issue_iid": issue_iid,
            "external_updated_at": external_updated_at,
            "service_identity_sha256": service_identity_sha256,
            "recorded_at_unix_ms": recorded_at_unix_ms,
        }
        return cls(MappingProxyType({**unsigned, "sha256": sha256_json(unsigned)}))

    @classmethod
    def from_mapping(
        cls,
        value: Any,
        *,
        request: MacOpsEdgeRequest | None = None,
    ) -> "MacOpsEdgeReceipt":
        required = frozenset(
            {
                "schema",
                "request_id",
                "request_sha256",
                "intent_sha256",
                "operation",
                "idempotency_key",
                "state",
                "issue_iid",
                "external_updated_at",
                "service_identity_sha256",
                "recorded_at_unix_ms",
                "sha256",
            }
        )
        item = _strict_mapping(value, required=required, code="invalid_receipt")
        if item["schema"] != RECEIPT_VERSION:
            _fail("invalid_receipt")
        digest = _sha256(item["sha256"], "invalid_receipt")
        unsigned = {key: item[key] for key in required if key != "sha256"}
        if sha256_json(unsigned) != digest:
            _fail("invalid_receipt")
        _sha256(item["request_sha256"], "invalid_receipt")
        _sha256(item["intent_sha256"], "invalid_receipt")
        _sha256(item["service_identity_sha256"], "invalid_receipt")
        try:
            MacOpsEdgeOperation(item["operation"])
            MacOpsEdgeState(item["state"])
        except (TypeError, ValueError) as exc:
            raise MacOpsEdgeProtocolError("invalid_receipt") from exc
        if request is not None and (
            item["request_id"] != request.request_id
            or item["request_sha256"] != request.request_sha256
            or item["intent_sha256"] != request.intent_sha256
            or item["operation"] != request.operation.value
            or item["idempotency_key"] != request.idempotency_key
        ):
            _fail("receipt_binding_mismatch")
        return cls(MappingProxyType(item))

    def to_mapping(self) -> dict[str, Any]:
        return dict(self.value)


def validate_response(
    value: Any,
    *,
    request: MacOpsEdgeRequest,
) -> dict[str, Any]:
    item = _strict_mapping(
        value,
        required=frozenset(
            {
                "protocol",
                "request_id",
                "sequence",
                "state",
                "replayed",
                "blocker",
                "result",
                "receipt",
            }
        ),
        code="invalid_response",
    )
    if (
        item["protocol"] != RESPONSE_VERSION
        or item["request_id"] != request.request_id
        or item["sequence"] != request.sequence
        or type(item["replayed"]) is not bool
        or not isinstance(item["result"], Mapping)
    ):
        _fail("invalid_response")
    try:
        MacOpsEdgeState(item["state"])
    except (TypeError, ValueError) as exc:
        raise MacOpsEdgeProtocolError("invalid_response") from exc
    blocker = item["blocker"]
    if blocker is not None and (
        not isinstance(blocker, str) or _BLOCKER_RE.fullmatch(blocker) is None
    ):
        _fail("invalid_response")
    MacOpsEdgeReceipt.from_mapping(item["receipt"], request=request)
    return dict(item)


__all__ = [
    "MAX_REQUEST_BYTES",
    "MAX_RESPONSE_BYTES",
    "MacOpsEdgeOperation",
    "MacOpsEdgeProtocolError",
    "MacOpsEdgeReceipt",
    "MacOpsEdgeRequest",
    "MacOpsEdgeState",
    "MacOpsReadOnlyClass",
    "MacOpsReadOnlySubmit",
    "MacOpsPing",
    "MacOpsTaskRead",
    "PROTOCOL_VERSION",
    "RECEIPT_VERSION",
    "RESPONSE_VERSION",
    "canonical_json_bytes",
    "decode_json_object",
    "sha256_json",
    "validate_response",
]
