"""Frozen Phase 1 JSON contracts and local digest validation.

This module intentionally uses only the Python standard library.  In
particular, validating an operation, record, or sealed receipt must not import
Olympus Engine; replay is a local, side-effect-free operation.
"""

from __future__ import annotations

import hashlib
import json
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

JSONScalar: TypeAlias = None | bool | int | str
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

GATE1_PACKET_SHA256 = (
    "6ef02d1e63d19853e0794b6c324f0e7233ec082744419fae529ed16e6e11cacf"
)
GATE2_PACKET_SHA256 = (
    "445f0ce7bbf3c063ce84d7b1ae47154241e5bb6f148969790a5d63dfca4f6d25"
)
ENGINE_CONTRACT_DIGEST = (
    "2c5860582fcc73192b4e301544e043189dac1079b47078bad3c1f93371b8ee85"
)
PHASE1_CONTRACT_DIGEST = (
    "e5db4065e1e39134a5274152a01363d2ed3a0c79fb5d4bdb9c1773d36a2b1de1"
)
PHASE0_RUNTIME_BINDING_DIGEST = (
    "366f41af5292272b183605cb1f6f5ab75b3a259c453d3396eab34a34bb83cb12"
)
ISOLATION_PROFILE_DIGEST = (
    "1463fab53b426fee1b11c61745e439f124f69e54c3af886b53d3a94c3f2ff67a"
)

OPERATION_SCHEMA_ID = "olympus.hermes.phase1.operation/v1"
RECEIPT_SCHEMA_ID = "olympus.hermes.phase1.receipt/v1"
OWNERSHIP_SCHEMA_ID = "olympus.hermes.phase1.ownership-record/v1"

MAX_ENVELOPE_BYTES = 65_536
MAX_RECEIPT_BYTES = 65_536
MAX_RECORD_BYTES = 16_384
MAX_IDEMPOTENCY_KEY_BYTES = 128
MAX_RECORDS_PER_KEY = 8
MAX_DIRECTORY_ENTRIES_PER_KEY = 24
MAX_JSON_NESTING = 64

_HASH_PREFIX = b"OLYMPUS_ENGINE_SHA256_V1\x00"
_DOMAIN_RE = re.compile(r"^[a-z][a-z0-9-]{0,63}$")
_HEX_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_RECORD_NAME_RE = re.compile(
    r"^(?P<sequence>[0-9]{6})-(?P<state>[A-Z][A-Z0-9_]*)\.json$"
)

_SCHEMA_ROOT = Path(__file__).with_name("schemas") / "v1"
_SCHEMA_SPECS = {
    "operation": (
        _SCHEMA_ROOT / "phase1-operation-v1.schema.json",
        5_064,
        "c154d9b90e0b409b88271bcab13c0947e49b973578a0cb3bce83bee659c46708",
        OPERATION_SCHEMA_ID,
        "9acd65e2814a6229b62760ecd20aa71e232974dd0532c6557d6273377b66ad46",
    ),
    "receipt": (
        _SCHEMA_ROOT / "phase1-receipt-v1.schema.json",
        22_706,
        "338ee017b61aa43c87e6999d64266d6963dde13e4c1c9ce522fb0b7770d42ce9",
        RECEIPT_SCHEMA_ID,
        "a8f941641570e8f0f96dd2ff368e94ee97b2e20fbf59fcb5c351dd73420f8a2a",
    ),
    "record": (
        _SCHEMA_ROOT / "phase1-ownership-record-v1.schema.json",
        9_478,
        "3b107dc9578c0a99df4d6aa8ee6d733174a28f7f1828d8dfe96862e91f822f83",
        OWNERSHIP_SCHEMA_ID,
        "75a2e09260f7565514ac67539da295cd642afb3302f9f6f2c42ac8102af07eec",
    ),
}
_SCHEMA_CACHE: dict[str, dict[str, JSONValue]] = {}

PHASE1_REASON_PRECEDENCE = (
    "ENVELOPE_TOO_LARGE",
    "ENVELOPE_INVALID_JSON",
    "ENGINE_CONTRACT_MISMATCH",
    "ENVELOPE_SCHEMA_MISMATCH",
    "UNSAFE_RECEIPT_ROOT",
    "ROOTS_OVERLAP",
    "UNSAFE_EVIDENCE_ROOT",
    "UNSAFE_REPOSITORY_ROOT",
    "PHASE0_PROVENANCE_MISMATCH",
    "FAKE_TRANSPORT_INVALID",
    "PERSISTENCE_CORRUPTION",
    "KEY_BOUND_TO_DIFFERENT_OPERATION",
    "ACTIVE_OWNER",
    "CANCELLED_BEFORE_INVOCATION_CLAIM",
    "RECOVERED_BEFORE_INVOCATION_CLAIM",
    "WORKFLOW_CONSTRUCTION_FAILED",
    "CANCELLED_AFTER_INVOCATION_CLAIM",
    "ENGINE_EXCEPTION_WITHOUT_SEALED_RECEIPT",
    "EVIDENCE_MISSING",
    "EVIDENCE_VERIFICATION_FAILED",
    "RECOVERED_AFTER_INVOCATION_CLAIM",
    "PHASE0_TERMINAL",
)

OWNERSHIP_STATES = (
    "OWNERSHIP_ACQUIRED",
    "INVOCATION_CLAIMED",
    "ENGINE_EVIDENCE_VERIFIED",
    "PREINVOKE_REJECTED",
    "INDETERMINATE_NO_RETRY",
    "RECEIPT_FINALIZED",
)
ALLOWED_TRANSITIONS = frozenset(
    {
        (None, "OWNERSHIP_ACQUIRED"),
        ("OWNERSHIP_ACQUIRED", "PREINVOKE_REJECTED"),
        ("OWNERSHIP_ACQUIRED", "INVOCATION_CLAIMED"),
        ("INVOCATION_CLAIMED", "ENGINE_EVIDENCE_VERIFIED"),
        ("INVOCATION_CLAIMED", "INDETERMINATE_NO_RETRY"),
        ("ENGINE_EVIDENCE_VERIFIED", "INDETERMINATE_NO_RETRY"),
        ("ENGINE_EVIDENCE_VERIFIED", "RECEIPT_FINALIZED"),
        ("PREINVOKE_REJECTED", "RECEIPT_FINALIZED"),
        ("INDETERMINATE_NO_RETRY", "RECEIPT_FINALIZED"),
    }
)
SEALED_RECEIPT_STATES = frozenset(
    {"REJECTED_PRE_INVOKE", "ENGINE_TERMINAL", "INDETERMINATE_NO_RETRY"}
)
_SEALED_RECEIPT_STATE_BY_PREDECESSOR = {
    "PREINVOKE_REJECTED": "REJECTED_PRE_INVOKE",
    "ENGINE_EVIDENCE_VERIFIED": "ENGINE_TERMINAL",
    "INDETERMINATE_NO_RETRY": "INDETERMINATE_NO_RETRY",
}
_ENGINE_EVIDENCE_RECORD_FIELDS = (
    "phase0_terminal_report_digest",
    "phase0_evidence_manifest_digest",
    "phase0_evidence_directory_name",
)

_PHASE0_ARTIFACT_SPECS = {
    "01-request.json": ("request", "REQUEST_CAPTURED"),
    "02-authorization.json": ("authorization", "AUTHORIZATION_RECORDED"),
    "03-packet.json": ("sealed-packet", "PACKET_SEALED"),
    "04-worker-result.json": ("worker-result", "PAIR_A_RECORDED"),
    "05-reviewer-result.json": ("reviewer-result", "PAIR_B_RECORDED"),
    "06-comparison.json": ("comparison", "COMPARISON_RECORDED"),
    "07-terminal-report.json": ("terminal-report", "TERMINAL_RECORDED"),
}
_PHASE0_REQUIRED_ARTIFACTS = {
    "01-request.json",
    "02-authorization.json",
    "06-comparison.json",
    "07-terminal-report.json",
}


class CanonicalJSONError(ValueError):
    """Input is outside the frozen deterministic JSON profile."""


class JSONSizeError(CanonicalJSONError):
    """JSON input exceeds its contract byte limit."""


class DuplicateKeyError(CanonicalJSONError):
    """A JSON object repeated a key."""


class ContractValidationError(ValueError):
    """A decoded value violates a frozen Phase 1 contract."""


class EnvelopeValidationError(ContractValidationError):
    """Operation envelope rejection with its deterministic reason code."""

    def __init__(self, reason_code: str) -> None:
        if reason_code not in {
            "ENVELOPE_TOO_LARGE",
            "ENVELOPE_INVALID_JSON",
            "ENGINE_CONTRACT_MISMATCH",
            "ENVELOPE_SCHEMA_MISMATCH",
        }:
            raise ValueError("invalid envelope rejection reason")
        self.reason_code = reason_code
        super().__init__(reason_code)


def _reject_constant(value: str) -> None:
    raise CanonicalJSONError(f"non-finite JSON number is forbidden: {value}")


def _reject_float(value: str) -> None:
    raise CanonicalJSONError(f"floating-point JSON numbers are forbidden: {value}")


def _parse_integer(value: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise CanonicalJSONError(
            "JSON integer exceeds the deterministic conversion limit"
        ) from exc


def _object_from_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def validate_json_value(value: Any, *, _depth: int = 0) -> None:
    if _depth > MAX_JSON_NESTING:
        raise CanonicalJSONError("JSON nesting limit exceeded")
    if value is None or isinstance(value, bool) or isinstance(value, int):
        return
    if isinstance(value, float):
        raise CanonicalJSONError("floating-point values are forbidden")
    if isinstance(value, str):
        try:
            value.encode("utf-8", "strict")
        except UnicodeEncodeError as exc:
            raise CanonicalJSONError("unpaired Unicode surrogate is forbidden") from exc
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise CanonicalJSONError("JSON object keys must be strings")
            validate_json_value(key, _depth=_depth + 1)
            validate_json_value(child, _depth=_depth + 1)
        return
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        for child in value:
            validate_json_value(child, _depth=_depth + 1)
        return
    raise CanonicalJSONError(f"unsupported JSON value type: {type(value).__name__}")


def strict_loads(data: str | bytes, *, max_bytes: int | None = None) -> JSONValue:
    if isinstance(data, bytes):
        encoded = data
        try:
            text = data.decode("utf-8", "strict")
        except UnicodeDecodeError as exc:
            raise CanonicalJSONError("JSON input is not valid UTF-8") from exc
    elif isinstance(data, str):
        text = data
        try:
            encoded = data.encode("utf-8", "strict")
        except UnicodeEncodeError as exc:
            raise CanonicalJSONError(
                "JSON input contains an unpaired surrogate"
            ) from exc
    else:
        raise TypeError("JSON input must be str or bytes")
    if max_bytes is not None and len(encoded) > max_bytes:
        raise JSONSizeError(
            f"JSON input is {len(encoded)} bytes; maximum is {max_bytes} bytes"
        )
    if text.startswith("\ufeff"):
        raise CanonicalJSONError("UTF-8 BOM is forbidden")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_object_from_pairs,
            parse_constant=_reject_constant,
            parse_float=_reject_float,
            parse_int=_parse_integer,
        )
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        if isinstance(exc, CanonicalJSONError):
            raise
        raise CanonicalJSONError("malformed JSON input") from exc
    validate_json_value(value)
    return value


def canonical_bytes(value: Any) -> bytes:
    validate_json_value(value)
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise CanonicalJSONError("value cannot be serialized canonically") from exc
    return text.encode("utf-8", "strict")


def domain_separated_sha256(domain: str, payload: bytes) -> str:
    if not isinstance(domain, str) or _DOMAIN_RE.fullmatch(domain) is None:
        raise ValueError("invalid digest domain")
    if not isinstance(payload, bytes):
        raise TypeError("digest payload must be bytes")
    domain_bytes = domain.encode("ascii")
    framed = b"".join(
        (
            _HASH_PREFIX,
            len(domain_bytes).to_bytes(2, "big"),
            domain_bytes,
            len(payload).to_bytes(8, "big"),
            payload,
        )
    )
    return hashlib.sha256(framed).hexdigest()


def canonical_digest(domain: str, value: Any) -> str:
    return domain_separated_sha256(domain, canonical_bytes(value))


def validate_hex_digest(value: Any, path: str) -> str:
    if not isinstance(value, str) or _HEX_DIGEST_RE.fullmatch(value) is None:
        raise ContractValidationError(f"{path}: expected 64 lowercase hex")
    return value


def _json_type_matches(expected: str, value: Any) -> bool:
    if expected == "null":
        return value is None
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "string":
        return isinstance(value, str)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, Mapping)
    return False


def _json_equal(left: Any, right: Any) -> bool:
    try:
        return canonical_bytes(left) == canonical_bytes(right)
    except CanonicalJSONError:
        return False


def _schema_document(name: str) -> dict[str, JSONValue]:
    cached = _SCHEMA_CACHE.get(name)
    if cached is not None:
        return cached
    try:
        path, expected_size, expected_sha, expected_id, expected_digest = (
            _SCHEMA_SPECS[name]
        )
    except KeyError as exc:
        raise ContractValidationError("unknown frozen schema") from exc
    entry = path.lstat()
    if (
        not stat.S_ISREG(entry.st_mode)
        or entry.st_nlink != 1
        or stat.S_ISLNK(entry.st_mode)
    ):
        raise ContractValidationError(f"{name} schema is not a regular unique file")
    raw = path.read_bytes()
    if len(raw) != expected_size or hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ContractValidationError(f"{name} schema raw bytes do not match Gate 2")
    parsed = strict_loads(raw, max_bytes=MAX_RECEIPT_BYTES)
    if not isinstance(parsed, dict):
        raise ContractValidationError(f"{name} schema is not an object")
    if canonical_digest("json-schema", parsed) != expected_digest:
        raise ContractValidationError(f"{name} schema digest does not match Gate 2")
    if parsed.get("$id") != expected_id:
        raise ContractValidationError(f"{name} schema identifier does not match Gate 2")
    expected_raw = (
        json.dumps(
            parsed,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        ).encode("utf-8")
        + b"\n"
    )
    if raw != expected_raw:
        raise ContractValidationError(f"{name} schema serialization is not frozen")
    _SCHEMA_CACHE[name] = parsed
    return parsed


def verify_frozen_schemas() -> None:
    for name in _SCHEMA_SPECS:
        _schema_document(name)


def _resolve_schema_ref(root: Mapping[str, Any], reference: str) -> Mapping[str, Any]:
    if not reference.startswith("#/"):
        raise ContractValidationError("external schema references are forbidden")
    node: Any = root
    for component in reference[2:].split("/"):
        component = component.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, Mapping) or component not in node:
            raise ContractValidationError("invalid local schema reference")
        node = node[component]
    if not isinstance(node, Mapping):
        raise ContractValidationError("schema reference does not resolve to an object")
    return node


def _validate_schema_node(
    value: Any,
    schema: Mapping[str, Any],
    root: Mapping[str, Any],
    path: str,
) -> None:
    reference = schema.get("$ref")
    if reference is not None:
        if not isinstance(reference, str):
            raise ContractValidationError("invalid schema reference")
        _validate_schema_node(value, _resolve_schema_ref(root, reference), root, path)

    for child in schema.get("allOf", []):
        _validate_schema_node(value, child, root, path)

    one_of = schema.get("oneOf")
    if one_of is not None:
        matches = 0
        for child in one_of:
            try:
                _validate_schema_node(value, child, root, path)
            except ContractValidationError:
                continue
            matches += 1
        if matches != 1:
            raise ContractValidationError(f"{path}: expected exactly one schema branch")

    conditional = schema.get("if")
    if conditional is not None:
        try:
            _validate_schema_node(value, conditional, root, path)
        except ContractValidationError:
            alternate = schema.get("else")
            if alternate is not None:
                _validate_schema_node(value, alternate, root, path)
        else:
            consequence = schema.get("then")
            if consequence is not None:
                _validate_schema_node(value, consequence, root, path)

    declared = schema.get("type")
    if declared is not None:
        types = [declared] if isinstance(declared, str) else declared
        if not isinstance(types, list) or not all(isinstance(item, str) for item in types):
            raise ContractValidationError("invalid frozen schema type declaration")
        if not any(_json_type_matches(item, value) for item in types):
            raise ContractValidationError(f"{path}: value has the wrong JSON type")

    if "const" in schema and not _json_equal(value, schema["const"]):
        raise ContractValidationError(f"{path}: value does not match the frozen constant")
    if "enum" in schema and not any(
        _json_equal(value, candidate) for candidate in schema["enum"]
    ):
        raise ContractValidationError(f"{path}: value is outside the frozen enumeration")

    if isinstance(value, Mapping):
        required = schema.get("required", [])
        missing = [field for field in required if field not in value]
        if missing:
            raise ContractValidationError(f"{path}: missing fields {missing!r}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(properties))
            if extra:
                raise ContractValidationError(f"{path}: unexpected fields {extra!r}")
        for field, child in properties.items():
            if field in value:
                _validate_schema_node(value[field], child, root, f"{path}.{field}")

    if isinstance(value, list):
        minimum = schema.get("minItems")
        maximum = schema.get("maxItems")
        if minimum is not None and len(value) < minimum:
            raise ContractValidationError(f"{path}: array is too short")
        if maximum is not None and len(value) > maximum:
            raise ContractValidationError(f"{path}: array is too long")
        if schema.get("uniqueItems"):
            encoded = [canonical_bytes(item) for item in value]
            if len(encoded) != len(set(encoded)):
                raise ContractValidationError(f"{path}: array items are not unique")
        child = schema.get("items")
        if child is not None:
            for index, item in enumerate(value):
                _validate_schema_node(item, child, root, f"{path}[{index}]")

    if isinstance(value, str):
        minimum = schema.get("minLength")
        maximum = schema.get("maxLength")
        if minimum is not None and len(value) < minimum:
            raise ContractValidationError(f"{path}: string is too short")
        if maximum is not None and len(value) > maximum:
            raise ContractValidationError(f"{path}: string is too long")
        pattern = schema.get("pattern")
        if pattern is not None and re.search(pattern, value) is None:
            raise ContractValidationError(f"{path}: string does not match the pattern")

    if isinstance(value, int) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and value < minimum:
            raise ContractValidationError(f"{path}: integer is too small")
        if maximum is not None and value > maximum:
            raise ContractValidationError(f"{path}: integer is too large")


def validate_schema(name: str, value: Any) -> None:
    document = _schema_document(name)
    _validate_schema_node(value, document, document, name)


@dataclass(frozen=True, slots=True)
class Operation:
    envelope: dict[str, JSONValue]
    semantic_body: dict[str, JSONValue]
    correlation_id: str
    idempotency_key: str
    idempotency_key_digest: str
    operation_digest: str
    payload: dict[str, JSONValue]
    payload_request_id: str
    payload_request_digest: str


def parse_operation(envelope_json: str | bytes) -> Operation:
    if not isinstance(envelope_json, (str, bytes)):
        raise TypeError("envelope_json must be str or bytes")
    try:
        encoded = (
            envelope_json
            if isinstance(envelope_json, bytes)
            else envelope_json.encode("utf-8", "strict")
        )
    except UnicodeEncodeError as exc:
        raise EnvelopeValidationError("ENVELOPE_INVALID_JSON") from exc
    if len(encoded) > MAX_ENVELOPE_BYTES:
        raise EnvelopeValidationError("ENVELOPE_TOO_LARGE")
    try:
        parsed = strict_loads(envelope_json, max_bytes=MAX_ENVELOPE_BYTES)
    except CanonicalJSONError as exc:
        raise EnvelopeValidationError("ENVELOPE_INVALID_JSON") from exc
    if not isinstance(parsed, dict):
        raise EnvelopeValidationError("ENVELOPE_SCHEMA_MISMATCH")
    engine_digest = parsed.get("engine_contract_digest")
    if not isinstance(engine_digest, str):
        raise EnvelopeValidationError("ENVELOPE_SCHEMA_MISMATCH")
    if engine_digest != ENGINE_CONTRACT_DIGEST:
        raise EnvelopeValidationError("ENGINE_CONTRACT_MISMATCH")
    try:
        validate_schema("operation", parsed)
    except ContractValidationError as exc:
        raise EnvelopeValidationError("ENVELOPE_SCHEMA_MISMATCH") from exc
    idempotency_key = parsed["idempotency_key"]
    payload = parsed["payload"]
    correlation_id = parsed["correlation_id"]
    if (
        not isinstance(idempotency_key, str)
        or not isinstance(payload, dict)
        or not isinstance(correlation_id, str)
    ):
        raise EnvelopeValidationError("ENVELOPE_SCHEMA_MISMATCH")
    key_bytes = idempotency_key.encode("utf-8")
    if len(key_bytes) > MAX_IDEMPOTENCY_KEY_BYTES:
        raise EnvelopeValidationError("ENVELOPE_SCHEMA_MISMATCH")
    semantic_body = {
        key: value for key, value in parsed.items() if key != "idempotency_key"
    }
    request_id = payload.get("request_id")
    if not isinstance(request_id, str):
        raise EnvelopeValidationError("ENVELOPE_SCHEMA_MISMATCH")
    return Operation(
        envelope=parsed,
        semantic_body=semantic_body,
        correlation_id=correlation_id,
        idempotency_key=idempotency_key,
        idempotency_key_digest=domain_separated_sha256(
            "phase1-idempotency-key", key_bytes
        ),
        operation_digest=canonical_digest("phase1-operation", semantic_body),
        payload=payload,
        payload_request_id=request_id,
        payload_request_digest=canonical_digest("request", payload),
    )


def _receipt_base() -> dict[str, JSONValue]:
    return {
        "schema_version": RECEIPT_SCHEMA_ID,
        "gate1_packet_sha256": GATE1_PACKET_SHA256,
        "phase1_contract_digest": PHASE1_CONTRACT_DIGEST,
        "engine_contract_digest": ENGINE_CONTRACT_DIGEST,
        "receipt_state": "REJECTED_PRE_INVOKE",
        "durability": "TRANSIENT",
        "correlation_id": None,
        "idempotency_key_digest": None,
        "operation_digest": None,
        "payload_request_id": None,
        "payload_request_digest": None,
        "bound_operation_digest": None,
        "ownership_record_digest": None,
        "reason_codes": ["PERSISTENCE_CORRUPTION"],
        "phase0_terminal_report": None,
        "phase0_terminal_report_digest": None,
        "phase0_evidence_manifest": None,
        "phase0_evidence_manifest_digest": None,
        "phase0_evidence_package_id": None,
        "phase0_evidence_directory_name": None,
        "automatic_engine_retry_permitted": False,
        "receipt_digest": "",
    }


def _finish_receipt(receipt: dict[str, JSONValue]) -> bytes:
    body = {key: value for key, value in receipt.items() if key != "receipt_digest"}
    receipt["receipt_digest"] = canonical_digest("phase1-receipt", body)
    validate_receipt(receipt)
    return canonical_bytes(receipt)


def transient_rejection(reason_code: str) -> bytes:
    receipt = _receipt_base()
    receipt["reason_codes"] = [reason_code]
    return _finish_receipt(receipt)


def transient_conflict(
    operation: Operation,
    *,
    bound_operation_digest: str,
    ownership_record_digest: str,
) -> bytes:
    validate_hex_digest(bound_operation_digest, "bound_operation_digest")
    validate_hex_digest(ownership_record_digest, "ownership_record_digest")
    different = bound_operation_digest != operation.operation_digest
    receipt = _receipt_base()
    receipt.update(
        {
            "receipt_state": (
                "IDEMPOTENCY_CONFLICT" if different else "CONFLICT_IN_PROGRESS"
            ),
            "correlation_id": operation.correlation_id,
            "idempotency_key_digest": operation.idempotency_key_digest,
            "operation_digest": operation.operation_digest,
            "payload_request_id": operation.payload_request_id,
            "payload_request_digest": operation.payload_request_digest,
            "bound_operation_digest": bound_operation_digest,
            "ownership_record_digest": ownership_record_digest,
            "reason_codes": [
                "KEY_BOUND_TO_DIFFERENT_OPERATION" if different else "ACTIVE_OWNER"
            ],
        }
    )
    return _finish_receipt(receipt)


def build_record(
    operation: Operation,
    *,
    sequence: int,
    state: str,
    previous_record_digest: str | None,
    reason_code: str | None = None,
    phase0_terminal_report_digest: str | None = None,
    phase0_evidence_manifest_digest: str | None = None,
    phase0_evidence_directory_name: str | None = None,
    receipt_digest: str | None = None,
) -> tuple[dict[str, JSONValue], bytes]:
    record: dict[str, JSONValue] = {
        "schema_version": OWNERSHIP_SCHEMA_ID,
        "gate1_packet_sha256": GATE1_PACKET_SHA256,
        "phase1_contract_digest": PHASE1_CONTRACT_DIGEST,
        "engine_contract_digest": ENGINE_CONTRACT_DIGEST,
        "sequence": sequence,
        "state": state,
        "idempotency_key_digest": operation.idempotency_key_digest,
        "operation_digest": operation.operation_digest,
        "correlation_id": operation.correlation_id,
        "payload_request_id": operation.payload_request_id,
        "payload_request_digest": operation.payload_request_digest,
        "previous_record_digest": previous_record_digest,
        "reason_code": reason_code,
        "phase0_terminal_report_digest": phase0_terminal_report_digest,
        "phase0_evidence_manifest_digest": phase0_evidence_manifest_digest,
        "phase0_evidence_directory_name": phase0_evidence_directory_name,
        "receipt_digest": receipt_digest,
        "record_digest": "",
    }
    body = {key: value for key, value in record.items() if key != "record_digest"}
    record["record_digest"] = canonical_digest("phase1-ownership-record", body)
    validate_record(record)
    return record, canonical_bytes(record)


def sealed_receipt(
    operation: Operation,
    *,
    receipt_state: str,
    predecessor: Mapping[str, JSONValue],
    reason_code: str,
    phase0_terminal_report: Mapping[str, JSONValue] | None = None,
    phase0_evidence_manifest: Mapping[str, JSONValue] | None = None,
) -> tuple[dict[str, JSONValue], bytes]:
    if receipt_state not in SEALED_RECEIPT_STATES:
        raise ContractValidationError("invalid sealed receipt state")
    validated_predecessor = validate_record(predecessor)
    expected_receipt_state = _SEALED_RECEIPT_STATE_BY_PREDECESSOR.get(
        validated_predecessor["state"]
    )
    if expected_receipt_state is None or receipt_state != expected_receipt_state:
        raise ContractValidationError("sealed receipt predecessor state mismatch")
    if reason_code != validated_predecessor["reason_code"]:
        raise ContractValidationError("sealed receipt predecessor reason mismatch")
    predecessor_digest = validate_hex_digest(
        validated_predecessor["record_digest"], "predecessor.record_digest"
    )
    terminal = None if phase0_terminal_report is None else dict(phase0_terminal_report)
    manifest = None if phase0_evidence_manifest is None else dict(
        phase0_evidence_manifest
    )
    terminal_digest = (
        None if terminal is None else canonical_digest("terminal-report", terminal)
    )
    manifest_digest = (
        None if manifest is None else canonical_digest("evidence-manifest", manifest)
    )
    package_id = None if manifest is None else manifest.get("package_id")
    directory_name = (
        None
        if receipt_state != "ENGINE_TERMINAL"
        else f"phase0-{operation.idempotency_key_digest}"
    )
    if receipt_state == "ENGINE_TERMINAL":
        engine_bindings = {
            "phase0_terminal_report_digest": terminal_digest,
            "phase0_evidence_manifest_digest": manifest_digest,
            "phase0_evidence_directory_name": directory_name,
        }
        for field in _ENGINE_EVIDENCE_RECORD_FIELDS:
            if engine_bindings[field] != validated_predecessor[field]:
                raise ContractValidationError(
                    f"sealed receipt predecessor evidence mismatch: {field}"
                )
    receipt = _receipt_base()
    receipt.update(
        {
            "receipt_state": receipt_state,
            "durability": "SEALED",
            "correlation_id": operation.correlation_id,
            "idempotency_key_digest": operation.idempotency_key_digest,
            "operation_digest": operation.operation_digest,
            "payload_request_id": operation.payload_request_id,
            "payload_request_digest": operation.payload_request_digest,
            "bound_operation_digest": operation.operation_digest,
            "ownership_record_digest": predecessor_digest,
            "reason_codes": [reason_code],
            "phase0_terminal_report": terminal,
            "phase0_terminal_report_digest": terminal_digest,
            "phase0_evidence_manifest": manifest,
            "phase0_evidence_manifest_digest": manifest_digest,
            "phase0_evidence_package_id": package_id,
            "phase0_evidence_directory_name": directory_name,
        }
    )
    raw = _finish_receipt(receipt)
    validate_receipt(receipt, operation=operation)
    return receipt, raw


def validate_record(record: Mapping[str, Any]) -> dict[str, JSONValue]:
    value = dict(record)
    validate_schema("record", value)
    if value["gate1_packet_sha256"] != GATE1_PACKET_SHA256:
        raise ContractValidationError("record Gate 1 binding mismatch")
    if value["phase1_contract_digest"] != PHASE1_CONTRACT_DIGEST:
        raise ContractValidationError("record Phase 1 binding mismatch")
    if value["engine_contract_digest"] != ENGINE_CONTRACT_DIGEST:
        raise ContractValidationError("record engine binding mismatch")
    claimed = validate_hex_digest(value["record_digest"], "record.record_digest")
    body = {key: item for key, item in value.items() if key != "record_digest"}
    if canonical_digest("phase1-ownership-record", body) != claimed:
        raise ContractValidationError("record digest mismatch")
    return value


def parse_record_bytes(raw: bytes) -> dict[str, JSONValue]:
    if len(raw) > MAX_RECORD_BYTES:
        raise ContractValidationError("ownership record exceeds byte limit")
    try:
        parsed = strict_loads(raw, max_bytes=MAX_RECORD_BYTES)
    except CanonicalJSONError as exc:
        raise ContractValidationError("ownership record is invalid JSON") from exc
    if not isinstance(parsed, dict) or canonical_bytes(parsed) != raw:
        raise ContractValidationError("ownership record is not canonical")
    return validate_record(parsed)


def record_filename(record: Mapping[str, Any]) -> str:
    return f"{int(record['sequence']):06d}-{record['state']}.json"


def validate_record_chain(
    records: Sequence[Mapping[str, Any]],
    *,
    filenames: Sequence[str] | None = None,
) -> list[dict[str, JSONValue]]:
    if not records or len(records) > MAX_RECORDS_PER_KEY:
        raise ContractValidationError("ownership chain length is invalid")
    if filenames is not None and len(filenames) != len(records):
        raise ContractValidationError("record filename count mismatch")
    validated: list[dict[str, JSONValue]] = []
    previous_state: str | None = None
    previous_digest: str | None = None
    invariant_fields = (
        "gate1_packet_sha256",
        "phase1_contract_digest",
        "engine_contract_digest",
        "idempotency_key_digest",
        "operation_digest",
        "correlation_id",
        "payload_request_id",
        "payload_request_digest",
    )
    anchor: dict[str, JSONValue] | None = None
    invocation_count = 0
    for index, raw_record in enumerate(records, 1):
        record = validate_record(raw_record)
        if record["sequence"] != index:
            raise ContractValidationError("ownership sequence is not contiguous")
        if filenames is not None:
            match = _RECORD_NAME_RE.fullmatch(filenames[index - 1])
            if (
                match is None
                or int(match.group("sequence")) != index
                or match.group("state") != record["state"]
                or filenames[index - 1] != record_filename(record)
            ):
                raise ContractValidationError("record filename binding mismatch")
        if record["previous_record_digest"] != previous_digest:
            raise ContractValidationError("ownership predecessor digest mismatch")
        state = record["state"]
        if (previous_state, state) not in ALLOWED_TRANSITIONS:
            raise ContractValidationError("forbidden ownership transition")
        if anchor is None:
            anchor = record
        else:
            for field in invariant_fields:
                if record[field] != anchor[field]:
                    raise ContractValidationError(
                        f"ownership invariant changed: {field}"
                    )
        if state == "INVOCATION_CLAIMED":
            invocation_count += 1
            if invocation_count > 1:
                raise ContractValidationError("multiple invocation claims")
        if previous_state == "RECEIPT_FINALIZED":
            raise ContractValidationError("record follows finalization")
        validated.append(record)
        previous_state = str(state)
        previous_digest = str(record["record_digest"])
    return validated


def _validate_phase0_embedded(
    receipt: Mapping[str, Any],
    operation: Operation | None,
) -> None:
    terminal = receipt["phase0_terminal_report"]
    manifest = receipt["phase0_evidence_manifest"]
    if not isinstance(terminal, dict) or not isinstance(manifest, dict):
        raise ContractValidationError("terminal receipt lacks embedded evidence")
    terminal_digest = canonical_digest("terminal-report", terminal)
    manifest_digest = canonical_digest("evidence-manifest", manifest)
    if receipt["phase0_terminal_report_digest"] != terminal_digest:
        raise ContractValidationError("embedded terminal digest mismatch")
    if receipt["phase0_evidence_manifest_digest"] != manifest_digest:
        raise ContractValidationError("embedded manifest digest mismatch")
    request_id = receipt["payload_request_id"]
    if terminal.get("request_id") != request_id or manifest.get("request_id") != request_id:
        raise ContractValidationError("embedded Phase 0 request identity mismatch")
    if manifest.get("terminal_report_digest") != terminal_digest:
        raise ContractValidationError("manifest terminal binding mismatch")

    identity_fields = (
        "request_id",
        "packetization_status",
        "pair_a_status",
        "pair_b_status",
        "repository_postcheck_status",
        "artifacts",
        "events",
        "repository_pre_digest",
        "repository_post_digest",
        "terminal_report_digest",
    )
    identity_body = {field: manifest.get(field) for field in identity_fields}
    package_id = canonical_digest("evidence-package", identity_body)
    if (
        manifest.get("package_id") != package_id
        or receipt["phase0_evidence_package_id"] != package_id
    ):
        raise ContractValidationError("evidence package identity mismatch")
    expected_directory = f"phase0-{receipt['idempotency_key_digest']}"
    if receipt["phase0_evidence_directory_name"] != expected_directory:
        raise ContractValidationError("evidence directory identity mismatch")

    artifacts = manifest.get("artifacts")
    events = manifest.get("events")
    if not isinstance(artifacts, list) or not isinstance(events, list):
        raise ContractValidationError("manifest arrays are invalid")
    if len(artifacts) != len(events):
        raise ContractValidationError("manifest event and artifact counts differ")
    names = [item.get("name") for item in artifacts if isinstance(item, dict)]
    if len(names) != len(artifacts) or not _PHASE0_REQUIRED_ARTIFACTS <= set(names):
        raise ContractValidationError("manifest required artifacts are missing")
    expected_order = [
        name for name in _PHASE0_ARTIFACT_SPECS if name in set(names)
    ]
    if names != expected_order:
        raise ContractValidationError("manifest artifact order is invalid")
    previous_event: str | None = None
    for sequence, (artifact, event) in enumerate(zip(artifacts, events, strict=True), 1):
        if not isinstance(artifact, dict) or not isinstance(event, dict):
            raise ContractValidationError("manifest item is invalid")
        name = artifact["name"]
        expected_domain, expected_event_type = _PHASE0_ARTIFACT_SPECS[name]
        if (
            artifact["sequence"] != sequence
            or event["sequence"] != sequence
            or artifact["domain"] != expected_domain
            or event["event_type"] != expected_event_type
            or event["artifact_name"] != name
            or event["artifact_digest"] != artifact["digest"]
            or event["previous_event_digest"] != previous_event
        ):
            raise ContractValidationError("manifest event chain binding mismatch")
        body = {
            "sequence": event["sequence"],
            "event_type": event["event_type"],
            "artifact_name": event["artifact_name"],
            "artifact_digest": event["artifact_digest"],
            "previous_event_digest": event["previous_event_digest"],
        }
        expected_event = canonical_digest("evidence-event", body)
        if event["event_digest"] != expected_event:
            raise ContractValidationError("manifest event digest mismatch")
        previous_event = expected_event

    by_name = {item["name"]: item for item in artifacts}
    terminal_record = by_name["07-terminal-report.json"]
    if (
        terminal_record["digest"] != terminal_digest
        or terminal_record["size"] != len(canonical_bytes(terminal))
    ):
        raise ContractValidationError("terminal artifact binding mismatch")
    request_record = by_name["01-request.json"]
    if request_record["digest"] != receipt["payload_request_digest"]:
        raise ContractValidationError("request artifact binding mismatch")
    if operation is not None:
        if operation.payload_request_id != request_id:
            raise ContractValidationError("receipt request identifier mismatch")
        if receipt["payload_request_digest"] != operation.payload_request_digest:
            raise ContractValidationError("receipt request digest mismatch")
        if request_record["size"] != len(canonical_bytes(operation.payload)):
            raise ContractValidationError("request artifact binding mismatch")


def validate_receipt(
    receipt: Mapping[str, Any],
    *,
    chain: Sequence[Mapping[str, Any]] | None = None,
    operation: Operation | None = None,
    require_sealed: bool = False,
) -> dict[str, JSONValue]:
    value = dict(receipt)
    validate_schema("receipt", value)
    if value["gate1_packet_sha256"] != GATE1_PACKET_SHA256:
        raise ContractValidationError("receipt Gate 1 binding mismatch")
    if value["phase1_contract_digest"] != PHASE1_CONTRACT_DIGEST:
        raise ContractValidationError("receipt Phase 1 binding mismatch")
    if value["engine_contract_digest"] != ENGINE_CONTRACT_DIGEST:
        raise ContractValidationError("receipt engine binding mismatch")
    claimed = validate_hex_digest(value["receipt_digest"], "receipt.receipt_digest")
    body = {key: item for key, item in value.items() if key != "receipt_digest"}
    if canonical_digest("phase1-receipt", body) != claimed:
        raise ContractValidationError("receipt digest mismatch")
    if require_sealed and value["durability"] != "SEALED":
        raise ContractValidationError("stored receipt is not sealed")
    reason_codes = value["reason_codes"]
    if not isinstance(reason_codes, list) or len(reason_codes) != 1:
        raise ContractValidationError("receipt must have one reason")

    state = value["receipt_state"]
    if state == "IDEMPOTENCY_CONFLICT":
        if value["bound_operation_digest"] == value["operation_digest"]:
            raise ContractValidationError("idempotency conflict digests must differ")
    elif value["idempotency_key_digest"] is not None:
        if value["bound_operation_digest"] != value["operation_digest"]:
            raise ContractValidationError("keyed receipt operation binding mismatch")

    if operation is not None and value["idempotency_key_digest"] is not None:
        expected = {
            "correlation_id": operation.correlation_id,
            "idempotency_key_digest": operation.idempotency_key_digest,
            "operation_digest": operation.operation_digest,
            "payload_request_id": operation.payload_request_id,
            "payload_request_digest": operation.payload_request_digest,
        }
        for field, expected_value in expected.items():
            if value[field] != expected_value:
                raise ContractValidationError(f"receipt current operation mismatch: {field}")

    validated_chain: list[dict[str, JSONValue]] | None = None
    if chain is not None:
        validated_chain = validate_record_chain(chain)
        anchor = validated_chain[0]
        if value["durability"] == "TRANSIENT":
            if value["idempotency_key_digest"] != anchor["idempotency_key_digest"]:
                raise ContractValidationError(
                    "receipt anchor mismatch: idempotency_key_digest"
                )
            if value["ownership_record_digest"] != anchor["record_digest"]:
                raise ContractValidationError("conflict anchor digest mismatch")
            if value["bound_operation_digest"] != anchor["operation_digest"]:
                raise ContractValidationError("conflict bound operation mismatch")
        else:
            for field in (
                "idempotency_key_digest",
                "correlation_id",
                "payload_request_id",
                "payload_request_digest",
            ):
                if value[field] != anchor[field]:
                    raise ContractValidationError(f"receipt anchor mismatch: {field}")
            predecessor = (
                validated_chain[-2]
                if validated_chain[-1]["state"] == "RECEIPT_FINALIZED"
                else validated_chain[-1]
            )
            if value["ownership_record_digest"] != predecessor["record_digest"]:
                raise ContractValidationError("sealed receipt predecessor mismatch")
            if value["operation_digest"] != anchor["operation_digest"]:
                raise ContractValidationError("sealed receipt operation mismatch")
            if reason_codes[0] != predecessor["reason_code"]:
                raise ContractValidationError("sealed receipt reason mismatch")
            expected_receipt_state = _SEALED_RECEIPT_STATE_BY_PREDECESSOR.get(
                predecessor["state"]
            )
            if expected_receipt_state is None or state != expected_receipt_state:
                raise ContractValidationError(
                    "sealed receipt predecessor state mismatch"
                )
            if state == "ENGINE_TERMINAL":
                for field in _ENGINE_EVIDENCE_RECORD_FIELDS:
                    if value[field] != predecessor[field]:
                        raise ContractValidationError(
                            f"sealed receipt predecessor evidence mismatch: {field}"
                        )
            if validated_chain[-1]["state"] == "RECEIPT_FINALIZED":
                final = validated_chain[-1]
                if (
                    final["previous_record_digest"] != predecessor["record_digest"]
                    or final["receipt_digest"] != value["receipt_digest"]
                ):
                    raise ContractValidationError("final record receipt binding mismatch")

    if state == "ENGINE_TERMINAL":
        _validate_phase0_embedded(value, operation)
    return value


def parse_receipt_bytes(
    raw: bytes,
    *,
    chain: Sequence[Mapping[str, Any]] | None = None,
    operation: Operation | None = None,
    require_sealed: bool = True,
) -> dict[str, JSONValue]:
    if len(raw) > MAX_RECEIPT_BYTES:
        raise ContractValidationError("receipt exceeds byte limit")
    try:
        parsed = strict_loads(raw, max_bytes=MAX_RECEIPT_BYTES)
    except CanonicalJSONError as exc:
        raise ContractValidationError("receipt is invalid JSON") from exc
    if not isinstance(parsed, dict) or canonical_bytes(parsed) != raw:
        raise ContractValidationError("receipt is not canonical")
    return validate_receipt(
        parsed,
        chain=chain,
        operation=operation,
        require_sealed=require_sealed,
    )


def digest_vectors() -> dict[str, str]:
    """Return the three primary Gate 2 vector identities for focused tests."""

    envelope = {
        "schema_version": OPERATION_SCHEMA_ID,
        "correlation_id": "corr-phase1-gate2-vector",
        "idempotency_key": "idem-phase1-gate2-vector",
        "engine_contract_digest": ENGINE_CONTRACT_DIGEST,
        "payload": {
            "contract_id": "olympus.repo-analysis.request/v1",
            "request_id": "req-phase1-gate2-vector",
            "repository_id": "synthetic-phase1-gate2-vector",
            "repository_kind": "SYNTHETIC",
            "repository_path": ".",
            "requested_paths": ["README.md"],
            "purpose": "Hermetic Phase 1 Gate 2 digest vector",
            "authority": {
                "scope": "OFFLINE_SYNTHETIC_REPOSITORY_ANALYSIS",
                "granted": True,
            },
            "limits": {
                "max_files": 96,
                "max_file_bytes": 32_768,
                "max_total_bytes": 196_608,
                "max_findings": 64,
                "max_model_json_bytes": 65_536,
            },
            "controls": {
                "offline": True,
                "fake_transports_only": True,
                "tools": [],
                "max_pair_a_attempts": 1,
                "max_pair_b_attempts": 1,
                "allow_retries": False,
                "allow_response_repair": False,
                "allow_cloud_fallback": False,
            },
        },
    }
    operation = parse_operation(canonical_bytes(envelope))
    first, _ = build_record(
        operation,
        sequence=1,
        state="OWNERSHIP_ACQUIRED",
        previous_record_digest=None,
    )
    return {
        "idempotency_key_digest": operation.idempotency_key_digest,
        "operation_digest": operation.operation_digest,
        "payload_request_digest": operation.payload_request_digest,
        "first_record_digest": str(first["record_digest"]),
        "transient_invalid_json_receipt_digest": str(
            strict_loads(transient_rejection("ENVELOPE_INVALID_JSON"))[
                "receipt_digest"
            ]
        ),
    }
