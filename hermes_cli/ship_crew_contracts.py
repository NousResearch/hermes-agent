"""Versioned Ship's Crew handoff contracts and strict validation.

Contracts are deliberately data-only: validation is deterministic and does not
invoke a model.  The runtime keeps payloads separate from their envelope so a
receipt can identify an immutable, hashed artifact without copying the artifact
into every downstream task.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import jsonschema


SCHEMA_VERSION = "crew-handoff/v1"
CONTRACT_TYPES = frozenset(
    {
        "navigator-evidence/v1",
        "engineer-delivery/v1",
        "pirate-review/v1",
        "captain-disposition/v1",
        "crew-block/v1",
        "completion-metadata/v1",
    }
)

_ROLE_PAIRS = {
    "navigator-evidence/v1": {("navigator", "captain"), ("navigator", "engineer")},
    "engineer-delivery/v1": {("engineer", "pirate"), ("engineer", "captain")},
    "pirate-review/v1": {("pirate", "captain")},
    "captain-disposition/v1": {("captain", "user"), ("captain", "engineer")},
    "crew-block/v1": {
        ("navigator", "captain"),
        ("engineer", "captain"),
        ("pirate", "captain"),
        ("captain", "user"),
    },
}


class ContractValidationError(ValueError):
    """Raised when an envelope or payload is not safe to persist/dispatch."""

    def __init__(self, issues: list[str]):
        self.issues = tuple(issues)
        super().__init__("; ".join(issues))


@dataclass(frozen=True)
class ContractIssue:
    path: str
    message: str


_COMMON_ENVELOPE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "mission_id",
        "task_id",
        "producer",
        "consumer",
        "contract_type",
        "governance_class",
        "governance_version",
        "idempotency_key",
        "data_class",
        "authority",
        "payload_ref",
        "payload_sha256",
    ],
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION},
        "mission_id": {"type": "string", "minLength": 1, "maxLength": 128},
        "task_id": {"type": "string", "minLength": 1, "maxLength": 128},
        "source_task_id": {"type": "string", "minLength": 1, "maxLength": 128},
        "producer": {"type": "string", "enum": ["navigator", "engineer", "pirate", "captain", "user"]},
        "consumer": {"type": "string", "enum": ["navigator", "engineer", "pirate", "captain", "user"]},
        "contract_type": {"type": "string", "enum": sorted(CONTRACT_TYPES)},
        "governance_class": {"type": "string", "enum": ["lite", "standard", "constitutional"]},
        "governance_version": {"type": "string", "minLength": 1, "maxLength": 32},
        "idempotency_key": {"type": "string", "minLength": 1, "maxLength": 256},
        "data_class": {"type": "string", "enum": ["ordinary", "sensitive"]},
        "authority": {
            "type": "object",
            "additionalProperties": False,
            "required": ["write_scope", "external_effect"],
            "properties": {
                "write_scope": {"type": "string", "enum": ["read-only", "worktree", "scoped-internal", "broad"]},
                "external_effect": {"type": "boolean"},
            },
        },
        "payload_ref": {"type": "string", "pattern": r"^/[^\x00]*$"},
        "payload_sha256": {"type": "string", "pattern": r"^[0-9a-f]{64}$"},
        "supersedes": {"type": "string", "minLength": 1, "maxLength": 256},
    },
}

_PAYLOAD_SCHEMAS: dict[str, dict[str, Any]] = {
    "navigator-evidence/v1": {
        "type": "object",
        "additionalProperties": False,
        "required": ["evidence", "findings"],
        "properties": {
            "evidence": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["id", "kind", "ref"], "properties": {"id": {"type": "string", "minLength": 1}, "kind": {"type": "string", "minLength": 1}, "ref": {"type": "string", "pattern": r"^/[^\x00]*$"}, "sha256": {"type": "string", "pattern": r"^[0-9a-f]{64}$"}}}},
            "findings": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["id", "severity", "summary"], "properties": {"id": {"type": "string", "minLength": 1}, "severity": {"type": "string", "enum": ["info", "low", "medium", "high", "critical"]}, "summary": {"type": "string", "minLength": 1, "maxLength": 2000}}}},
        },
    },
    "engineer-delivery/v1": {
        "type": "object",
        "additionalProperties": False,
        "required": ["status", "changed_paths", "verification"],
        "properties": {
            "status": {"type": "string", "enum": ["completed", "partial", "blocked"]},
            "changed_paths": {"type": "array", "items": {"type": "string", "pattern": r"^/[^\x00]*$"}},
            "verification": {"type": "array", "minItems": 1, "items": {"type": "object", "additionalProperties": False, "required": ["id", "status"], "properties": {"id": {"type": "string", "minLength": 1}, "status": {"type": "string", "enum": ["passed", "failed", "skipped", "manual", "environment-blocked"]}, "ref": {"type": "string", "pattern": r"^/[^\x00]*$"}}}},
        },
    },
    "pirate-review/v1": {
        "type": "object",
        "additionalProperties": False,
        "required": ["decision", "findings", "required_action"],
        "properties": {
            "decision": {"type": "string", "enum": ["approve", "reject", "request-changes", "block"]},
            "findings": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["id", "severity", "summary"], "properties": {"id": {"type": "string", "minLength": 1}, "severity": {"type": "string", "enum": ["info", "low", "medium", "high", "critical"]}, "summary": {"type": "string", "minLength": 1, "maxLength": 2000}}}},
            "required_action": {"type": "string", "minLength": 1},
        },
    },
    "captain-disposition/v1": {
        "type": "object",
        "additionalProperties": False,
        "required": ["decision", "rationale", "sail_required"],
        "properties": {
            "decision": {"type": "string", "enum": ["sail", "amend", "reconvene", "approve", "reject", "block"]},
            "rationale": {"type": "string", "minLength": 1, "maxLength": 4000},
            "sail_required": {"type": "boolean"},
        },
    },
    "crew-block/v1": {
        "type": "object",
        "additionalProperties": False,
        "required": ["kind", "retryable", "required_action", "required_evidence"],
        "properties": {
            "kind": {"type": "string", "enum": ["dependency", "needs_input", "capability", "transient", "contract", "quota"]},
            "retryable": {"type": "boolean"},
            "blocked_until": {"type": ["integer", "null"], "minimum": 0},
            "quota_domain": {"type": ["string", "null"], "minLength": 1},
            "required_action": {"type": "string", "minLength": 1, "maxLength": 2000},
            "required_evidence": {"type": "array", "items": {"type": "string", "pattern": r"^/[^\x00]*$"}},
        },
    },
    "completion-metadata/v1": {
        "type": "object",
        "additionalProperties": False,
        "required": ["status", "output_class", "summary", "evidence_refs"],
        "properties": {
            "status": {"type": "string", "enum": ["completed", "blocked", "failed"]},
            "output_class": {"type": "string", "enum": ["O0", "O1", "O2", "O3+"]},
            "summary": {"type": "string", "minLength": 1, "maxLength": 4000},
            "evidence_refs": {"type": "array", "items": {"type": "string", "pattern": r"^/[^\x00]*$"}},
            "artifact_ref": {"type": ["string", "null"], "pattern": r"^/[^\x00]*$"},
        },
    },
}


def schema_for(contract_type: str) -> dict[str, Any]:
    if contract_type not in _PAYLOAD_SCHEMAS:
        raise ContractValidationError([f"contract_type: unsupported contract {contract_type!r}"])
    return _PAYLOAD_SCHEMAS[contract_type]


def _errors(instance: Any, schema: Mapping[str, Any]) -> list[str]:
    validator = jsonschema.Draft202012Validator(schema)
    result: list[str] = []
    for error in sorted(validator.iter_errors(instance), key=lambda item: list(item.absolute_path)):
        path = ".".join(str(part) for part in error.absolute_path) or "$"
        result.append(f"{path}: {error.message}")
    return result


def canonical_sha256(payload: Any) -> str:
    """Hash canonical JSON without persisting or logging the payload."""
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate_envelope(envelope: Mapping[str, Any]) -> None:
    issues = _errors(dict(envelope), _COMMON_ENVELOPE_SCHEMA)
    contract_type = envelope.get("contract_type") if isinstance(envelope, Mapping) else None
    if not issues and contract_type in _ROLE_PAIRS:
        pair = (envelope.get("producer"), envelope.get("consumer"))
        if pair not in _ROLE_PAIRS[contract_type]:
            issues.append(f"producer/consumer: role pair {pair!r} is not allowed for {contract_type}")
    if issues:
        raise ContractValidationError(issues)


def validate_payload(contract_type: str, payload: Mapping[str, Any]) -> None:
    issues = _errors(dict(payload), schema_for(contract_type))
    if issues:
        raise ContractValidationError(issues)


def validate_contract(envelope: Mapping[str, Any], payload: Mapping[str, Any] | None = None) -> None:
    """Validate an envelope and, when supplied, its separately stored payload."""
    validate_envelope(envelope)
    if payload is not None:
        contract_type = str(envelope["contract_type"])
        validate_payload(contract_type, payload)
        actual = canonical_sha256(payload)
        if actual != envelope["payload_sha256"]:
            raise ContractValidationError([f"payload_sha256: expected {envelope['payload_sha256']}, observed {actual}"])


def validate_completion_metadata(metadata: Mapping[str, Any]) -> None:
    validate_payload("completion-metadata/v1", metadata)


def contract_metadata(envelope: Mapping[str, Any]) -> dict[str, Any]:
    """Return the safe, JSON-serializable task/run metadata projection."""
    validate_envelope(envelope)
    return {
        "schema_version": envelope["schema_version"],
        "contract_type": envelope["contract_type"],
        "mission_id": envelope["mission_id"],
        "source_task_id": envelope.get("source_task_id"),
        "governance_class": envelope["governance_class"],
        "idempotency_key": envelope["idempotency_key"],
        "data_class": envelope["data_class"],
        "authority": dict(envelope["authority"]),
        "payload_ref": envelope["payload_ref"],
        "payload_sha256": envelope["payload_sha256"],
    }


def load_json_contract(path: str | Path) -> dict[str, Any]:
    """Load strict UTF-8 JSON for a contract fixture without accepting NaN."""
    try:
        raw = Path(path).read_text(encoding="utf-8")
        value = json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ContractValidationError([f"fixture: invalid strict JSON ({exc})"]) from exc
    if not isinstance(value, dict):
        raise ContractValidationError(["fixture: top-level value must be an object"])
    return value
