"""Versioned, canonical contracts for Ares installed releases.

The contracts are intentionally stdlib-only.  The stable bootstrap resolver
must be able to validate the selected release before importing Hermes code.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .errors import AresRuntimeError

ACTIVATION_GRANT_SCHEMA = "AresActivationGrantV1"
INSTALLED_RUNTIME_POINTER_SCHEMA = "AresInstalledRuntimePointerV1"
RUNTIME_IDENTITY_SCHEMA = "AresRuntimeIdentityV1"


def canonical_json(value: Any) -> bytes:
    """Encode a contract as newline-terminated canonical JSON."""

    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError) as exc:
        raise AresRuntimeError("CONTRACT_NOT_CANONICAL", str(exc)) from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise AresRuntimeError("INVALID_IDENTITY", field)
    if set(value) - set("0123456789abcdef"):
        raise AresRuntimeError("INVALID_IDENTITY", field)
    return value


def require_absolute_path(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise AresRuntimeError("INVALID_PATH", field)
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise AresRuntimeError("INVALID_PATH", field)
    return str(path)


def _require_nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AresRuntimeError("INVALID_CONTRACT", field)
    return value


def _strict_object(raw: bytes) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AresRuntimeError("DUPLICATE_JSON_KEY", key)
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AresRuntimeError("INVALID_JSON", str(exc)) from exc
    if not isinstance(value, dict):
        raise AresRuntimeError("INVALID_CONTRACT", "root must be an object")
    if canonical_json(value) != raw:
        raise AresRuntimeError("CONTRACT_NOT_CANONICAL")
    return value


@dataclass(frozen=True)
class ReleaseReference:
    """One immutable release selected by the installed-runtime pointer."""

    kind: str
    release_id: str
    release_manifest_sha256: str
    runtime_tree_sha256: str

    def __post_init__(self) -> None:
        if self.kind == "sealed_candidate":
            require_sha256(self.release_id, "release_id")
        elif self.kind == "legacy_import":
            if not self.release_id.startswith("legacy-"):
                raise AresRuntimeError("INVALID_RELEASE_REFERENCE", "legacy release id")
            require_sha256(self.release_id.removeprefix("legacy-"), "legacy release id")
        else:
            raise AresRuntimeError("INVALID_RELEASE_REFERENCE", "kind")
        require_sha256(self.release_manifest_sha256, "release_manifest_sha256")
        require_sha256(self.runtime_tree_sha256, "runtime_tree_sha256")

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "release_id": self.release_id,
            "release_manifest_sha256": self.release_manifest_sha256,
            "runtime_tree_sha256": self.runtime_tree_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReleaseReference":
        if set(value) != {
            "kind",
            "release_id",
            "release_manifest_sha256",
            "runtime_tree_sha256",
        }:
            raise AresRuntimeError("INVALID_RELEASE_REFERENCE", "fields")
        return cls(
            kind=value["kind"],
            release_id=value["release_id"],
            release_manifest_sha256=value["release_manifest_sha256"],
            runtime_tree_sha256=value["runtime_tree_sha256"],
        )


@dataclass(frozen=True)
class InstalledRuntimePointer:
    """The only authority for stable current and previous release selection."""

    generation: int
    current: ReleaseReference
    previous: ReleaseReference | None
    committed_transaction_id: str
    state_root: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.generation, int)
            or isinstance(self.generation, bool)
            or self.generation < 1
        ):
            raise AresRuntimeError("INVALID_POINTER", "generation")
        require_sha256(self.committed_transaction_id, "committed_transaction_id")
        require_absolute_path(self.state_root, "state_root")
        if self.previous is not None and self.previous == self.current:
            raise AresRuntimeError("INVALID_POINTER", "current equals previous")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": INSTALLED_RUNTIME_POINTER_SCHEMA,
            "generation": self.generation,
            "current": self.current.to_dict(),
            "previous": self.previous.to_dict() if self.previous else None,
            "committed_transaction_id": self.committed_transaction_id,
            "state_root": self.state_root,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InstalledRuntimePointer":
        expected = {
            "schema",
            "generation",
            "current",
            "previous",
            "committed_transaction_id",
            "state_root",
        }
        if (
            set(value) != expected
            or value.get("schema") != INSTALLED_RUNTIME_POINTER_SCHEMA
        ):
            raise AresRuntimeError("INVALID_POINTER", "schema or fields")
        current = value.get("current")
        previous = value.get("previous")
        if not isinstance(current, dict):
            raise AresRuntimeError("INVALID_POINTER", "current")
        if previous is not None and not isinstance(previous, dict):
            raise AresRuntimeError("INVALID_POINTER", "previous")
        return cls(
            generation=value["generation"],
            current=ReleaseReference.from_dict(current),
            previous=ReleaseReference.from_dict(previous) if previous else None,
            committed_transaction_id=value["committed_transaction_id"],
            state_root=value["state_root"],
        )

    @classmethod
    def parse(cls, raw: bytes) -> "InstalledRuntimePointer":
        return cls.from_dict(_strict_object(raw))


@dataclass(frozen=True)
class ActivationGrant:
    """CandidateStore-issued authorization for one exact installed release."""

    candidate_id: str
    certification_set_id: str
    sealed_candidate_id: str
    audit_subject_id: str
    audit_subject_sha256: str
    audit_result_sha256: str
    archive_sha256: str
    candidate_core_sha256: str
    sealed_manifest_sha256: str
    release_manifest_sha256: str
    runtime_tree_sha256: str
    custody_event_sequence: int
    target_platform: str
    target_release_root: str
    materializer_contract: str
    activator_contract: str
    resolver_contract: str
    grant_id: str | None = None

    def __post_init__(self) -> None:
        for field in (
            "candidate_id",
            "certification_set_id",
            "sealed_candidate_id",
            "audit_subject_id",
            "audit_subject_sha256",
            "audit_result_sha256",
            "archive_sha256",
            "candidate_core_sha256",
            "sealed_manifest_sha256",
            "release_manifest_sha256",
            "runtime_tree_sha256",
        ):
            require_sha256(getattr(self, field), field)
        if (
            not isinstance(self.custody_event_sequence, int)
            or isinstance(self.custody_event_sequence, bool)
            or self.custody_event_sequence < 1
        ):
            raise AresRuntimeError("INVALID_GRANT", "custody_event_sequence")
        _require_nonempty_string(self.target_platform, "target_platform")
        require_absolute_path(self.target_release_root, "target_release_root")
        for field in (
            "materializer_contract",
            "activator_contract",
            "resolver_contract",
        ):
            _require_nonempty_string(getattr(self, field), field)
        if self.grant_id is not None:
            require_sha256(self.grant_id, "grant_id")
            if self.grant_id != self.computed_grant_id():
                raise AresRuntimeError("INVALID_GRANT", "grant_id")

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema": ACTIVATION_GRANT_SCHEMA,
            "candidate_id": self.candidate_id,
            "certification_set_id": self.certification_set_id,
            "sealed_candidate_id": self.sealed_candidate_id,
            "audit_subject_id": self.audit_subject_id,
            "audit_subject_sha256": self.audit_subject_sha256,
            "audit_result": "PASS",
            "audit_result_sha256": self.audit_result_sha256,
            "archive_sha256": self.archive_sha256,
            "candidate_core_sha256": self.candidate_core_sha256,
            "sealed_manifest_sha256": self.sealed_manifest_sha256,
            "release_manifest_sha256": self.release_manifest_sha256,
            "runtime_tree_sha256": self.runtime_tree_sha256,
            "custody_event_sequence": self.custody_event_sequence,
            "target_platform": self.target_platform,
            "target_release_root": self.target_release_root,
            "materializer_contract": self.materializer_contract,
            "activator_contract": self.activator_contract,
            "resolver_contract": self.resolver_contract,
        }

    def computed_grant_id(self) -> str:
        return sha256_bytes(canonical_json(self.unsigned_dict()))

    def with_grant_id(self) -> "ActivationGrant":
        values = dict(self.__dict__)
        values["grant_id"] = self.computed_grant_id()
        return ActivationGrant(**values)

    def to_dict(self) -> dict[str, Any]:
        if self.grant_id is None:
            raise AresRuntimeError("INVALID_GRANT", "grant_id missing")
        return {**self.unsigned_dict(), "grant_id": self.grant_id}

    def canonical_bytes(self) -> bytes:
        return canonical_json(self.to_dict())

    @classmethod
    def parse(cls, raw: bytes) -> "ActivationGrant":
        value = _strict_object(raw)
        expected = {
            "schema",
            "candidate_id",
            "certification_set_id",
            "sealed_candidate_id",
            "audit_subject_id",
            "audit_subject_sha256",
            "audit_result",
            "audit_result_sha256",
            "archive_sha256",
            "candidate_core_sha256",
            "sealed_manifest_sha256",
            "release_manifest_sha256",
            "runtime_tree_sha256",
            "custody_event_sequence",
            "target_platform",
            "target_release_root",
            "materializer_contract",
            "activator_contract",
            "resolver_contract",
            "grant_id",
        }
        if set(value) != expected or value.get("schema") != ACTIVATION_GRANT_SCHEMA:
            raise AresRuntimeError("INVALID_GRANT", "schema or fields")
        if value.get("audit_result") != "PASS":
            raise AresRuntimeError("INVALID_GRANT", "audit_result")
        value = dict(value)
        value.pop("schema")
        value.pop("audit_result")
        return cls(**value)


@dataclass(frozen=True)
class RuntimeIdentity:
    """Minimal release identity a running process must report."""

    sealed_candidate_id: str
    release_manifest_sha256: str
    runtime_tree_sha256: str
    resolver_sha256: str
    role: str
    generation: int

    def __post_init__(self) -> None:
        for field in (
            "sealed_candidate_id",
            "release_manifest_sha256",
            "runtime_tree_sha256",
            "resolver_sha256",
        ):
            require_sha256(getattr(self, field), field)
        _require_nonempty_string(self.role, "role")
        if (
            not isinstance(self.generation, int)
            or isinstance(self.generation, bool)
            or self.generation < 1
        ):
            raise AresRuntimeError("INVALID_RUNTIME_IDENTITY", "generation")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RUNTIME_IDENTITY_SCHEMA,
            "sealed_candidate_id": self.sealed_candidate_id,
            "release_manifest_sha256": self.release_manifest_sha256,
            "runtime_tree_sha256": self.runtime_tree_sha256,
            "resolver_sha256": self.resolver_sha256,
            "role": self.role,
            "generation": self.generation,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json(self.to_dict())
