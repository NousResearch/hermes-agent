"""FND-05 Evidence Model — Phase 1 semantic/in-memory boundary."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from typing import Any, Mapping

from .execution_identity import ExecutionIdentity, ExecutionIdentityError

SCHEMA_VERSION = "1.0"
EVIDENCE_TYPES = frozenset({
    "EXECUTION",
    "AUTHORIZATION",
    "STATE_TRANSITION",
    "RESULT",
    "FAILURE",
    "ARTIFACT_REFERENCE",
    "VERIFICATION_REFERENCE",
})
IDENTITY_APPLICABILITY = frozenset({"EXECUTION", "NON_EXECUTION"})
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_MAX_PAYLOAD_BYTES = 16384

ERROR_CODES = frozenset({
    "INVALID_IDENTITY",
    "INVALID_EVIDENCE_INPUT",
    "INVALID_PROVENANCE",
    "INVALID_INTEGRITY",
    "IMMUTABLE_EVIDENCE_MUTATION",
    "SCHEMA_ERROR",
    "AMBIGUOUS_EVIDENCE",
    "UNSUPPORTED_EVIDENCE_TYPE",
    "EVIDENCE_SERIALIZATION_ERROR",
})


class EvidenceModelError(ValueError):
    code = "INVALID_EVIDENCE_INPUT"


class InvalidIdentityError(EvidenceModelError):
    code = "INVALID_IDENTITY"


class InvalidEvidenceInputError(EvidenceModelError):
    code = "INVALID_EVIDENCE_INPUT"


class InvalidProvenanceError(EvidenceModelError):
    code = "INVALID_PROVENANCE"


class InvalidIntegrityError(EvidenceModelError):
    code = "INVALID_INTEGRITY"


class ImmutableEvidenceMutationError(EvidenceModelError):
    code = "IMMUTABLE_EVIDENCE_MUTATION"


class SchemaError(EvidenceModelError):
    code = "SCHEMA_ERROR"


class AmbiguousEvidenceError(EvidenceModelError):
    code = "AMBIGUOUS_EVIDENCE"


class UnsupportedEvidenceTypeError(EvidenceModelError):
    code = "UNSUPPORTED_EVIDENCE_TYPE"


class EvidenceSerializationError(EvidenceModelError):
    code = "EVIDENCE_SERIALIZATION_ERROR"


@dataclasses.dataclass(frozen=True)
class EvidenceProvenance:
    source_boundary: str
    source_type: str
    creation_context: str
    parent_evidence_id: str | None = None

    def validate(self) -> None:
        for name, value in (
            ("source_boundary", self.source_boundary),
            ("source_type", self.source_type),
            ("creation_context", self.creation_context),
        ):
            if not isinstance(value, str) or not value.strip():
                raise InvalidProvenanceError(f"{name} must be a non-empty string.")
        if self.parent_evidence_id is not None:
            _validate_identifier("parent_evidence_id", self.parent_evidence_id)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        result = {
            "creation_context": self.creation_context,
            "source_boundary": self.source_boundary,
            "source_type": self.source_type,
        }
        if self.parent_evidence_id is not None:
            result["parent_evidence_id"] = self.parent_evidence_id
        return result


@dataclasses.dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    evidence_type: str
    provenance: EvidenceProvenance
    identity_applicability: str
    identity: ExecutionIdentity | None = None
    metadata: Mapping[str, Any] | None = None
    reference: str | None = None
    payload: Mapping[str, Any] | None = None
    timestamp: str | None = None
    supersedes: str | None = None
    integrity_sha256: str | None = None
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        _validate_identifier("evidence_id", self.evidence_id)
        if self.evidence_type not in EVIDENCE_TYPES:
            raise UnsupportedEvidenceTypeError(
                "evidence_type must be one of the seven canonical Phase 1 types."
            )
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaError(f"schema_version must be {SCHEMA_VERSION!r}.")
        if self.identity_applicability not in IDENTITY_APPLICABILITY:
            raise AmbiguousEvidenceError(
                "identity_applicability must be exactly EXECUTION or NON_EXECUTION."
            )
        if not isinstance(self.provenance, EvidenceProvenance):
            raise InvalidProvenanceError("provenance must be an EvidenceProvenance.")
        self.provenance.validate()

        if self.identity_applicability == "EXECUTION":
            if not isinstance(self.identity, ExecutionIdentity):
                raise InvalidIdentityError(
                    "execution-related evidence requires exactly one ExecutionIdentity."
                )
            try:
                self.identity.validate()
            except ExecutionIdentityError as exc:
                raise InvalidIdentityError(str(exc)) from exc
        elif self.identity is not None:
            raise AmbiguousEvidenceError(
                "NON_EXECUTION evidence must not carry an execution identity."
            )

        if self.supersedes is not None:
            _validate_identifier("supersedes", self.supersedes)
            if self.supersedes == self.evidence_id:
                raise AmbiguousEvidenceError("evidence cannot supersede itself.")

        for name, value in (("reference", self.reference), ("timestamp", self.timestamp)):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise InvalidEvidenceInputError(f"{name} must be a non-empty string when supplied.")

        _validate_json_mapping("metadata", self.metadata)
        _validate_json_mapping("payload", self.payload)
        if self.payload is not None:
            payload_bytes = _canonical_json(self.payload).encode("utf-8")
            if len(payload_bytes) > _MAX_PAYLOAD_BYTES:
                raise InvalidEvidenceInputError("payload exceeds the Phase 1 bounded size limit.")

        if self.integrity_sha256 is not None:
            if not isinstance(self.integrity_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.integrity_sha256):
                raise InvalidIntegrityError("integrity_sha256 must be a lowercase SHA-256 hex digest.")
            if self.integrity_sha256 != self.compute_sha256():
                raise InvalidIntegrityError("integrity_sha256 does not match canonical evidence content.")

    def _canonical_dict(self, *, include_integrity: bool) -> dict[str, Any]:
        self.provenance.validate()
        result: dict[str, Any] = {
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type,
            "identity_applicability": self.identity_applicability,
            "provenance": self.provenance.to_dict(),
            "schema_version": self.schema_version,
        }
        if self.identity is not None:
            result["identity"] = self.identity.to_dict()
        if self.metadata is not None:
            result["metadata"] = _json_safe(self.metadata)
        if self.reference is not None:
            result["reference"] = self.reference
        if self.payload is not None:
            result["payload"] = _json_safe(self.payload)
        if self.timestamp is not None:
            result["timestamp"] = self.timestamp
        if self.supersedes is not None:
            result["supersedes"] = self.supersedes
        if include_integrity and self.integrity_sha256 is not None:
            result["integrity_sha256"] = self.integrity_sha256
        return result

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return self._canonical_dict(include_integrity=True)

    def canonical_bytes(self) -> bytes:
        self.validate_without_integrity()
        return _canonical_json(self._canonical_dict(include_integrity=False)).encode("utf-8")

    def compute_sha256(self) -> str:
        try:
            return hashlib.sha256(self.canonical_bytes()).hexdigest()
        except EvidenceModelError:
            raise
        except Exception as exc:
            raise EvidenceSerializationError("canonical evidence serialization failed.") from exc

    def with_integrity(self) -> "EvidenceRecord":
        self.validate_without_integrity()
        return dataclasses.replace(self, integrity_sha256=self.compute_sha256())

    def serialize(self) -> str:
        self.validate()
        try:
            return _canonical_json(self._canonical_dict(include_integrity=True))
        except EvidenceModelError:
            raise
        except Exception as exc:
            raise EvidenceSerializationError("evidence serialization failed.") from exc

    @classmethod
    def deserialize(cls, serialized: str) -> "EvidenceRecord":
        if not isinstance(serialized, str):
            raise EvidenceSerializationError("serialized evidence must be a string.")
        try:
            raw = json.loads(serialized)
            if not isinstance(raw, dict):
                raise SchemaError("serialized evidence must decode to an object.")
            provenance_raw = raw.pop("provenance", None)
            identity_raw = raw.pop("identity", None)
            if not isinstance(provenance_raw, dict):
                raise InvalidProvenanceError("provenance must be an object.")
            provenance = EvidenceProvenance(**provenance_raw)
            identity = None
            if identity_raw is not None:
                identity = ExecutionIdentity.deserialize(_canonical_json(identity_raw))
            item = cls(provenance=provenance, identity=identity, **raw)
            item.validate()
            return item
        except EvidenceModelError:
            raise
        except (ExecutionIdentityError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise EvidenceSerializationError("evidence deserialization failed.") from exc

    def validate_without_integrity(self) -> None:
        original = self.integrity_sha256
        object.__setattr__(self, "integrity_sha256", None)
        try:
            self.validate()
        finally:
            object.__setattr__(self, "integrity_sha256", original)

    def assert_can_supersede(self, previous: "EvidenceRecord") -> None:
        if not isinstance(previous, EvidenceRecord):
            raise ImmutableEvidenceMutationError("previous evidence must be an EvidenceRecord.")
        self.validate_without_integrity()
        previous.validate_without_integrity()
        if self.supersedes != previous.evidence_id:
            raise ImmutableEvidenceMutationError("supersedes must identify the previous evidence record.")
        if (self.evidence_type, self.identity_applicability) != (
            previous.evidence_type,
            previous.identity_applicability,
        ):
            raise ImmutableEvidenceMutationError("superseding evidence must remain in the same semantic domain.")
        if self.identity_applicability == "EXECUTION" and self.identity != previous.identity:
            raise ImmutableEvidenceMutationError(
                "execution evidence cannot supersede evidence from another execution identity."
            )


def _validate_identifier(name: str, value: Any) -> None:
    if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value):
        raise InvalidEvidenceInputError(
            f"{name} must match ^[A-Za-z0-9][A-Za-z0-9._:-]*$."
        )


def _validate_json_mapping(name: str, value: Mapping[str, Any] | None) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise InvalidEvidenceInputError(f"{name} must be a JSON-compatible mapping.")
    try:
        _canonical_json(value)
    except EvidenceSerializationError as exc:
        raise InvalidEvidenceInputError(f"{name} must contain only deterministic JSON values.") from exc


def _json_safe(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(_canonical_json(value))
    except EvidenceSerializationError as exc:
        raise InvalidEvidenceInputError("value is not deterministically JSON serializable.") from exc


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise EvidenceSerializationError("value is not deterministically JSON serializable.") from exc
