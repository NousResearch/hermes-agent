"""Owner-preserving Ares collaboration artifacts and enforcement seams.

This module owns immutable contracts and rebuildable projections only. Existing
profile, task, claim, permit, effect, and model-route stores remain canonical.
"""
from __future__ import annotations

import contextvars
import hashlib
import json
import os
import re
import secrets
import socket
import struct
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence

SCHEMA_VERSION = "1.0.0"
COMPILER_VERSION = "ares-context-compiler-1"
EPOCH = "1970-01-01T00:00:00Z"


def canonical_json(value: Any) -> bytes:
    return (json.dumps(_thaw(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode()


def digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value)).hexdigest()


def _freeze(value: Any) -> Any:
    """Recursively freeze artifact payloads so callers cannot alter a projection."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    """Return JSON-compatible copies without exposing an artifact's internals."""
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class ContractError(ValueError):
    def __init__(self, code: str, field: str = ""):
        self.code, self.field = code, field
        super().__init__(f"{code}{(': ' + field) if field else ''}")


def _check_ref(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._:/@#-" for c in value):
        raise ContractError("INVALID_REFERENCE", field)


def _strict_payload(raw: Mapping[str, Any], required: set[str], allowed: set[str], digest_field: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ContractError("INVALID_ARTIFACT")
    keys = set(raw)
    missing, unknown = required - keys, keys - allowed
    if missing:
        raise ContractError("MISSING_FIELD", sorted(missing)[0])
    if unknown:
        raise ContractError("UNKNOWN_FIELD", sorted(unknown)[0])
    value = dict(raw)
    supplied = value.pop(digest_field)
    if supplied != digest(value):
        raise ContractError("DIGEST_MISMATCH", digest_field)
    return value


_SCHEMA_DIR = os.path.join(os.path.dirname(__file__), "schemas")
_SCHEMA_FILES = {
    "role": "role_contract_v1.json",
    "specialist_descriptor": "specialist_descriptor_v1.json",
    "mission": "mission_contract_v1.json",
    "finding": "finding_v1.json",
    "evidence": "evidence_item_v1.json",
    "context": "context_packet_v1.json",
    "handoff": "handoff_packet_v1.json",
    "test_request": "test_request_v1.json",
    "witness": "witness_v1.json",
    "closure": "closure_state_v1.json",
    "transition": "transition_receipt_v1.json",
    "evaluation": "evaluation_outcome_v1.json",
}


def _validate_schema(kind: str, value: Mapping[str, Any]) -> None:
    """Validate against the dossier schema; never silently downgrade validation."""
    filename = _SCHEMA_FILES.get(kind)
    if filename is None:
        return
    try:
        import jsonschema
    except ImportError as exc:
        raise ContractError("SCHEMA_VALIDATOR_UNAVAILABLE", kind) from exc
    try:
        with open(os.path.join(_SCHEMA_DIR, filename), encoding="utf-8") as handle:
            schema = json.load(handle)
        jsonschema.Draft202012Validator(schema).validate(_thaw(value))
    except FileNotFoundError as exc:
        raise ContractError("SCHEMA_UNAVAILABLE", filename) from exc
    except jsonschema.ValidationError as exc:
        raise ContractError("SCHEMA_INVALID", str(exc.absolute_path)) from exc


def _finalize(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    result[field] = digest(result)
    return result


@dataclass(frozen=True)
class ImmutableArtifact:
    payload: Mapping[str, Any]
    digest_field: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _freeze(self.payload))

    def to_dict(self) -> dict[str, Any]:
        return _thaw(self.payload)

    def canonical_bytes(self) -> bytes:
        return canonical_json(self.payload)

    @property
    def artifact_digest(self) -> str:
        return self.payload[self.digest_field]


ROLE_REQUIRED = {"schema_version", "role_id", "role_kind", "durable_ownership", "objective", "unique_questions", "mandatory_triggers", "exclusions", "context_policy", "capability_profile", "mutation_authority", "output_schema_ref", "stop_conditions", "typed_failures", "handoff_rules", "model_eligibility", "evaluation", "recorded_at", "contract_digest"}
ROLE_ALLOWED = ROLE_REQUIRED | {"profile_ref", "merge_prohibition", "supersedes_contract_ref"}
SPECIALIST_DESCRIPTOR_REQUIRED = {
    "schema_version",
    "profile_id",
    "semantic_role_id",
    "enabled",
    "narrow_purpose",
    "capability_classes",
    "tool_classes",
    "required_artifact_ids",
    "input_evidence_classes",
    "required_outputs",
    "explicit_exclusions",
    "mandatory_deferrals",
    "handoff_rules",
    "failure_and_abstention_behavior",
    "activation_evidence_refs",
    "provenance",
    "recorded_at",
    "descriptor_digest",
}
SPECIALIST_DESCRIPTOR_ALLOWED = SPECIALIST_DESCRIPTOR_REQUIRED | {
    "supersedes_descriptor_ref"
}
_PROFILE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
MISSION_REQUIRED = {"schema_version", "mission_id", "kanban_root_task_ref", "objective", "source_freeze", "closure_profile", "risk_class", "effect_class", "boundaries", "required_evidence", "topology_policy", "stop_conditions", "recorded_at", "contract_digest"}
MISSION_ALLOWED = MISSION_REQUIRED | {"session_ref", "goal_ref", "non_goals", "budget", "supersedes_contract_ref"}


class RoleContractV1(ImmutableArtifact):
    @classmethod
    def create(cls, values: Mapping[str, Any], profile_exists: Callable[[str], bool] | None = None) -> "RoleContractV1":
        value = dict(values); value.setdefault("schema_version", SCHEMA_VERSION); value.setdefault("recorded_at", EPOCH)
        if value["schema_version"] != SCHEMA_VERSION: raise ContractError("UNSUPPORTED_SCHEMA", "schema_version")
        if value.get("profile_ref") is not None:
            _check_ref(value["profile_ref"], "profile_ref")
            if profile_exists is not None and not profile_exists(value["profile_ref"]): raise ContractError("UNKNOWN_PROFILE", value["profile_ref"])
        if value.get("supersedes_contract_ref") is not None:
            _check_ref(value["supersedes_contract_ref"], "supersedes_contract_ref")
        value.pop("contract_digest", None); value = _finalize(value, "contract_digest")
        _strict_payload(value, ROLE_REQUIRED, ROLE_ALLOWED, "contract_digest")
        _validate_schema("role", value)
        return cls(value, "contract_digest")

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "RoleContractV1":
        value = _strict_payload(raw, ROLE_REQUIRED, ROLE_ALLOWED, "contract_digest")
        _validate_schema("role", raw)
        return cls({**value, "contract_digest": raw["contract_digest"]}, "contract_digest")


def _validate_specialist_descriptor_references(
    value: Mapping[str, Any],
    *,
    profile_exists: Callable[[str], bool] | None = None,
    semantic_role_artifacts: Mapping[str, Sequence[str]] | None = None,
) -> None:
    """Validate external identities without creating a second owner or router."""
    profile_id = value.get("profile_id")
    if not isinstance(profile_id, str) or not _PROFILE_ID_RE.fullmatch(profile_id):
        raise ContractError("INVALID_PROFILE", "profile_id")
    if profile_exists is not None and not profile_exists(profile_id):
        raise ContractError("UNKNOWN_PROFILE", profile_id)

    role_id = value.get("semantic_role_id")
    if not isinstance(role_id, str) or not role_id.startswith("role."):
        raise ContractError("INVALID_SEMANTIC_ROLE", "semantic_role_id")
    if semantic_role_artifacts is None:
        return
    required = semantic_role_artifacts.get(role_id)
    if required is None:
        raise ContractError("UNKNOWN_SEMANTIC_ROLE", role_id)
    actual = value.get("required_artifact_ids")
    if (
        not isinstance(actual, list)
        or set(actual) != set(required)
        or len(actual) != len(set(actual))
    ):
        raise ContractError("REQUIRED_ARTIFACT_MISMATCH", role_id)


class SpecialistDescriptorV1(ImmutableArtifact):
    """Static, content-addressed profile declaration; never a routing decision.

    The descriptor is intentionally separate from ``RoleContractV1``. The
    latter is a general collaboration artifact, while this type fixes a
    profile-to-existing-semantic-role mapping and rejects runtime state such as
    provider health, credentials, current capacity, reservations, cost, or
    latency by strict field admission.
    """

    @classmethod
    def create(
        cls,
        values: Mapping[str, Any],
        *,
        profile_exists: Callable[[str], bool] | None = None,
        semantic_role_artifacts: Mapping[str, Sequence[str]] | None = None,
    ) -> "SpecialistDescriptorV1":
        value = dict(values)
        value.setdefault("schema_version", SCHEMA_VERSION)
        value.setdefault("recorded_at", EPOCH)
        if value["schema_version"] != SCHEMA_VERSION:
            raise ContractError("UNSUPPORTED_SCHEMA", "schema_version")
        value.pop("descriptor_digest", None)
        value = _finalize(value, "descriptor_digest")
        _strict_payload(
            value,
            SPECIALIST_DESCRIPTOR_REQUIRED,
            SPECIALIST_DESCRIPTOR_ALLOWED,
            "descriptor_digest",
        )
        _validate_schema("specialist_descriptor", value)
        _validate_specialist_descriptor_references(
            value,
            profile_exists=profile_exists,
            semantic_role_artifacts=semantic_role_artifacts,
        )
        return cls(value, "descriptor_digest")

    @classmethod
    def parse(
        cls,
        raw: Mapping[str, Any],
        *,
        profile_exists: Callable[[str], bool] | None = None,
        semantic_role_artifacts: Mapping[str, Sequence[str]] | None = None,
    ) -> "SpecialistDescriptorV1":
        value = _strict_payload(
            raw,
            SPECIALIST_DESCRIPTOR_REQUIRED,
            SPECIALIST_DESCRIPTOR_ALLOWED,
            "descriptor_digest",
        )
        _validate_schema("specialist_descriptor", raw)
        _validate_specialist_descriptor_references(
            raw,
            profile_exists=profile_exists,
            semantic_role_artifacts=semantic_role_artifacts,
        )
        return cls(
            {**value, "descriptor_digest": raw["descriptor_digest"]},
            "descriptor_digest",
        )


def specialist_descriptor_ref(descriptor: SpecialistDescriptorV1) -> str:
    """Return an unambiguous reference without binding it to profile metadata."""
    if not isinstance(descriptor, SpecialistDescriptorV1):
        raise ContractError("INVALID_SPECIALIST_DESCRIPTOR")
    return "specialist-descriptor:" + descriptor.artifact_digest.removeprefix("sha256:")


def validate_specialist_descriptor_set(
    raw_descriptors: Sequence[Mapping[str, Any]],
    *,
    profile_ids: Sequence[str],
    semantic_role_artifacts: Mapping[str, Sequence[str]],
    require_disabled: bool = True,
) -> list[str]:
    """Return stable errors for a static descriptor set, with no side effects.

    This validates only profile and semantic-registry identity passed by the
    caller. It deliberately does not look up provider/model state, live tools,
    desktop capacity, reservations, credentials, or any router-owned evidence.
    """
    errors: list[str] = []
    expected_profiles = set(profile_ids)
    if len(expected_profiles) != len(profile_ids):
        errors.append("profile roster contains duplicate profile_id")
    seen_profiles: set[str] = set()
    for index, raw in enumerate(raw_descriptors):
        if not isinstance(raw, Mapping):
            errors.append(f"descriptor[{index}] is not an object")
            continue
        try:
            descriptor = SpecialistDescriptorV1.parse(
                raw,
                profile_exists=lambda profile_id: profile_id in expected_profiles,
                semantic_role_artifacts=semantic_role_artifacts,
            )
        except ContractError as exc:
            errors.append(f"descriptor[{index}] {exc}")
            continue
        payload = descriptor.to_dict()
        profile_id = payload["profile_id"]
        if profile_id in seen_profiles:
            errors.append(f"duplicate descriptor profile_id: {profile_id}")
        seen_profiles.add(profile_id)
        if require_disabled and payload["enabled"] is not False:
            errors.append(f"descriptor must remain disabled: {profile_id}")
    missing = sorted(expected_profiles - seen_profiles)
    unexpected = sorted(seen_profiles - expected_profiles)
    if missing:
        errors.append("missing descriptor profile_ids: " + ", ".join(missing))
    if unexpected:
        errors.append("unexpected descriptor profile_ids: " + ", ".join(unexpected))
    return errors


class MissionContractV1(ImmutableArtifact):
    @classmethod
    def create(cls, values: Mapping[str, Any], task_exists: Callable[[str], bool] | None = None) -> "MissionContractV1":
        value = dict(values); value.setdefault("schema_version", SCHEMA_VERSION); value.setdefault("recorded_at", EPOCH)
        if value["schema_version"] != SCHEMA_VERSION: raise ContractError("UNSUPPORTED_SCHEMA", "schema_version")
        _check_ref(value.get("kanban_root_task_ref"), "kanban_root_task_ref")
        if task_exists is not None and not task_exists(value["kanban_root_task_ref"]): raise ContractError("UNKNOWN_OWNER_REFERENCE", value["kanban_root_task_ref"])
        for field in ("session_ref", "goal_ref", "supersedes_contract_ref"):
            if value.get(field) is not None: _check_ref(value[field], field)
        value.pop("contract_digest", None); value = _finalize(value, "contract_digest")
        _strict_payload(value, MISSION_REQUIRED, MISSION_ALLOWED, "contract_digest")
        _validate_schema("mission", value)
        return cls(value, "contract_digest")

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "MissionContractV1":
        value = _strict_payload(raw, MISSION_REQUIRED, MISSION_ALLOWED, "contract_digest")
        _validate_schema("mission", raw)
        return cls({**value, "contract_digest": raw["contract_digest"]}, "contract_digest")


def contract_ref(contract: ImmutableArtifact) -> str:
    """Return the stable owner reference for an immutable contract artifact."""
    if not isinstance(contract, (RoleContractV1, MissionContractV1)):
        raise ContractError("INVALID_CONTRACT")
    return "contract:" + contract.artifact_digest.removeprefix("sha256:")


class ContractBindings:
    """Thin Phase-1 binding adapter over existing profile/task/goal owners.

    It owns neither rows nor credentials. Callers provide the existing owner
    operations; with the flag disabled every method is a strict no-op.
    """
    def __init__(
        self,
        *,
        enabled: Callable[[], bool],
        profile_exists: Callable[[str], bool],
        task_exists: Callable[[str], bool],
        set_profile_contract_ref: Callable[[str, str, str], None],
        attach_task_contract_ref: Callable[[str, str], None],
        attach_goal_contract_ref: Callable[[str, str], None] | None = None,
    ) -> None:
        self._enabled = enabled
        self._profile_exists = profile_exists
        self._task_exists = task_exists
        self._set_profile_contract_ref = set_profile_contract_ref
        self._attach_task_contract_ref = attach_task_contract_ref
        self._attach_goal_contract_ref = attach_goal_contract_ref

    def bind_role(self, contract: RoleContractV1) -> str:
        ref = contract_ref(contract)
        if not self._enabled():
            return ref
        profile_ref = contract.to_dict().get("profile_ref")
        if not isinstance(profile_ref, str) or not self._profile_exists(profile_ref):
            raise ContractError("UNKNOWN_PROFILE", str(profile_ref or ""))
        self._set_profile_contract_ref(profile_ref, contract.to_dict()["role_id"], ref)
        return ref

    def bind_mission(self, contract: MissionContractV1) -> str:
        ref = contract_ref(contract)
        if not self._enabled():
            return ref
        payload = contract.to_dict()
        task_ref = payload["kanban_root_task_ref"]
        if not self._task_exists(task_ref):
            raise ContractError("UNKNOWN_OWNER_REFERENCE", task_ref)
        prior = payload.get("supersedes_contract_ref")
        if prior is not None and prior == ref:
            raise ContractError("INVALID_SUPERSESSION", "supersedes_contract_ref")
        self._attach_task_contract_ref(task_ref, ref)
        goal_ref = payload.get("goal_ref")
        if goal_ref is not None:
            if self._attach_goal_contract_ref is None:
                raise ContractError("UNKNOWN_OWNER_REFERENCE", goal_ref)
            self._attach_goal_contract_ref(goal_ref, ref)
        return ref


ARTIFACT_FIELDS: dict[str, tuple[set[str], str]] = {
    "finding": ({"schema_version", "finding_id", "mission_ref", "role_contract_ref", "severity", "release_impact", "surface", "evidence_refs", "consequence", "root_cause", "owner_preserving_fix", "acceptance_test", "rollback_or_quarantine", "confidence_basis", "status", "recorded_at", "finding_digest"}, "finding_digest"),
    "evidence": ({"schema_version", "evidence_id", "mission_ref", "kind", "source_ref", "artifact_digest", "recorded_at", "evidence_state", "authority_class", "taint", "acquisition_receipt_ref"}, "artifact_digest"),
    "context": ({"schema_version", "context_packet_id", "mission_ref", "role_contract_ref", "included_refs", "omitted_classes", "withheld_until_commit", "compiler_version", "compiled_at", "context_digest"}, "context_digest"),
    "handoff": ({"schema_version", "handoff_id", "mission_ref", "from_role_ref", "to_role_ref", "owned_question", "context_packet_ref", "evidence_refs", "unresolved_claim_refs", "withheld_classes", "permit_refs", "required_output_schema_ref", "stop_conditions", "recorded_at", "handoff_digest"}, "handoff_digest"),
    "test_request": ({"schema_version", "test_request_id", "mission_ref", "question", "oracle_class", "procedure", "expected_discriminators", "required_environment", "authority_requirements", "stop_conditions", "recorded_at"}, ""),
    "witness": ({"schema_version", "witness_id", "mission_ref", "test_request_ref", "role_contract_ref", "independence", "verdict", "evidence_refs", "coverage", "limitations", "recorded_at", "witness_digest"}, "witness_digest"),
    "closure": ({"schema_version", "mission_ref", "closure_profile", "state", "satisfied_gate_ids", "unsatisfied_gate_ids", "blocking_refs", "source_event_refs", "projected_at", "projection_digest"}, "projection_digest"),
}
ARTIFACT_ALLOWED = {k: set(fields) for k, (fields, _) in ARTIFACT_FIELDS.items()}
ARTIFACT_ALLOWED["evidence"] |= {"source_locator", "valid_from", "valid_to", "content_excerpt"}
ARTIFACT_ALLOWED["context"] |= {"parent_context_ref"}
ARTIFACT_ALLOWED["test_request"] |= {"input_refs"}
ARTIFACT_ALLOWED["closure"] |= {"divergence_flags"}


def make_artifact(kind: str, values: Mapping[str, Any]) -> ImmutableArtifact:
    if kind not in ARTIFACT_FIELDS: raise ContractError("UNKNOWN_ARTIFACT", kind)
    required, field = ARTIFACT_FIELDS[kind]
    value = dict(values); value.setdefault("schema_version", SCHEMA_VERSION)
    if "recorded_at" in required: value.setdefault("recorded_at", EPOCH)
    value.pop(field, None); value = _finalize(value, field)
    _strict_payload(value, required, ARTIFACT_ALLOWED[kind], field)
    _validate_schema(kind, value)
    return ImmutableArtifact(value, field)


class FindingV1(ImmutableArtifact):
    @classmethod
    def create(cls, values: Mapping[str, Any]) -> "FindingV1":
        item = make_artifact("finding", values)
        _validate_schema("finding", item.payload)
        return cls(item.payload, item.digest_field)
    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "FindingV1":
        required, field = ARTIFACT_FIELDS["finding"]
        value = _strict_payload(raw, required, ARTIFACT_ALLOWED["finding"], field)
        _validate_schema("finding", raw)
        return cls({**value, field: raw[field]}, field)


class EvidenceItemV1(ImmutableArtifact):
    @classmethod
    def create(cls, values: Mapping[str, Any]) -> "EvidenceItemV1":
        item = make_artifact("evidence", values)
        _validate_schema("evidence", item.payload)
        return cls(item.payload, item.digest_field)
    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "EvidenceItemV1":
        required, field = ARTIFACT_FIELDS["evidence"]
        value = _strict_payload(raw, required, ARTIFACT_ALLOWED["evidence"], field)
        _validate_schema("evidence", raw)
        return cls({**value, field: raw[field]}, field)


class HandoffPacketV1(ImmutableArtifact):
    @classmethod
    def create(cls, values: Mapping[str, Any]) -> "HandoffPacketV1":
        item = make_artifact("handoff", values)
        _validate_schema("handoff", item.payload)
        return cls(item.payload, item.digest_field)
    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "HandoffPacketV1":
        required, field = ARTIFACT_FIELDS["handoff"]
        value = _strict_payload(raw, required, ARTIFACT_ALLOWED["handoff"], field)
        _validate_schema("handoff", raw)
        return cls({**value, field: raw[field]}, field)


class ContextPacketV1(ImmutableArtifact):
    """Immutable, rebuildable projection; it never embeds source material."""
    @classmethod
    def create(cls, values: Mapping[str, Any]) -> "ContextPacketV1":
        item = make_artifact("context", values)
        return cls(item.payload, item.digest_field)

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "ContextPacketV1":
        required, field = ARTIFACT_FIELDS["context"]
        value = _strict_payload(raw, required, ARTIFACT_ALLOWED["context"], field)
        _validate_schema("context", raw)
        return cls({**value, field: raw[field]}, field)


class TestRequestV1(ImmutableArtifact):
    __test__ = False

    @classmethod
    def create(cls, values: Mapping[str, Any]) -> "TestRequestV1":
        required, _ = ARTIFACT_FIELDS["test_request"]; value = dict(values); value.setdefault("schema_version", SCHEMA_VERSION); value.setdefault("recorded_at", EPOCH)
        unknown = set(value) - ARTIFACT_ALLOWED["test_request"]
        if unknown: raise ContractError("UNKNOWN_FIELD", sorted(unknown)[0])
        if required - set(value): raise ContractError("MISSING_FIELD", sorted(required - set(value))[0])
        _validate_schema("test_request", value)
        return cls(value, "")
    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "TestRequestV1":
        required, _ = ARTIFACT_FIELDS["test_request"]; unknown = set(raw) - ARTIFACT_ALLOWED["test_request"]
        if unknown: raise ContractError("UNKNOWN_FIELD", sorted(unknown)[0])
        if required - set(raw): raise ContractError("MISSING_FIELD", sorted(required - set(raw))[0])
        _validate_schema("test_request", raw)
        return cls(dict(raw), "")

    @property
    def artifact_digest(self) -> str:
        return digest(self.payload)


def _normalize_context_ref(item: Mapping[str, Any]) -> tuple[dict[str, str], str | None]:
    """Strip compiler-only classification metadata before a context is emitted."""
    if not isinstance(item, Mapping):
        raise ContractError("INVALID_CONTEXT_REF")
    allowed = {"ref", "digest", "purpose", "classification", "secret_class", "withhold"}
    if set(item) - allowed or not {"ref", "digest", "purpose"} <= set(item):
        raise ContractError("INVALID_CONTEXT_REF")
    ref = {key: item[key] for key in ("ref", "digest", "purpose")}
    _check_ref(ref["ref"], "ref")
    if not isinstance(ref["digest"], str) or not ref["digest"].startswith("sha256:"):
        raise ContractError("INVALID_DIGEST", "digest")
    if not isinstance(ref["purpose"], str) or not ref["purpose"]:
        raise ContractError("INVALID_CONTEXT_REF", "purpose")
    classification = item.get("classification") or item.get("secret_class")
    if classification is not None and (not isinstance(classification, str) or not classification):
        raise ContractError("INVALID_CONTEXT_CLASSIFICATION")
    if item.get("withhold") is not None and type(item["withhold"]) is not bool:
        raise ContractError("INVALID_CONTEXT_CLASSIFICATION")
    return ref, classification if item.get("withhold") or classification not in (None, "none", "public") else None


def _resolution_withheld(resolution: Any) -> str | None:
    """Read classification metadata only; source payloads remain with their owners."""
    if not isinstance(resolution, Mapping):
        return None
    taint = resolution.get("taint")
    secret_class = taint.get("secret_class") if isinstance(taint, Mapping) else None
    if secret_class not in (None, "none"):
        return str(secret_class)
    return None


class ContextCompiler:
    """Pure compiler. Resolver hooks make exact refs and source freeze checkable."""
    def compile(self, mission_ref: str, role_ref: str, refs: Sequence[Mapping[str, Any]], *, withheld: Sequence[str] | None = None, omitted: Sequence[str] | None = None, resolve_ref: Callable[[str, str], bool | Mapping[str, Any]] | None = None, source_revision: str | None = None, frozen_source_revision: str | None = None, max_refs: int | None = None) -> ContextPacketV1:
        _check_ref(mission_ref, "mission_ref"); _check_ref(role_ref, "role_contract_ref")
        if frozen_source_revision is not None and source_revision != frozen_source_revision: raise ContractError("STALE_SOURCE_FREEZE")
        if max_refs is not None and (type(max_refs) is not int or max_refs < 1):
            raise ContractError("INVALID_CONTEXT_LIMIT", "max_refs")
        normalized: list[dict[str, str]] = []
        auto_withheld: list[str] = []
        for raw in refs:
            item, classification = _normalize_context_ref(raw)
            resolution = resolve_ref(item["ref"], item["digest"]) if resolve_ref is not None else True
            if resolution is False or resolution is None:
                raise ContractError("MISSING_EVIDENCE_REF", item["ref"])
            classification = classification or _resolution_withheld(resolution)
            if classification:
                auto_withheld.append(classification)
                continue
            normalized.append(item)
        ordered_refs = sorted({(item["ref"], item["digest"], item["purpose"]) for item in normalized})
        normalized = [{"ref": ref, "digest": item_digest, "purpose": purpose} for ref, item_digest, purpose in ordered_refs]
        omitted_classes = set(omitted or ())
        if max_refs is not None and len(normalized) > max_refs:
            normalized = normalized[:max_refs]
            omitted_classes.add("context_truncated")
        if not normalized:
            raise ContractError("EVIDENCE_DEFICIT", "included_refs")
        withheld_classes = set(withheld or ()) | set(auto_withheld)
        if not all(isinstance(value, str) and value for value in omitted_classes | withheld_classes):
            raise ContractError("INVALID_CONTEXT_CLASSIFICATION")
        identity = {"mission_ref": mission_ref, "role_contract_ref": role_ref, "included_refs": normalized, "omitted_classes": sorted(omitted_classes), "withheld_until_commit": sorted(withheld_classes)}
        value = {"schema_version": SCHEMA_VERSION, "context_packet_id": "ctx:" + digest(identity)[7:], "mission_ref": mission_ref, "role_contract_ref": role_ref, "included_refs": normalized, "omitted_classes": sorted(omitted_classes), "withheld_until_commit": sorted(withheld_classes), "compiler_version": COMPILER_VERSION, "compiled_at": EPOCH}
        return ContextPacketV1.create(value)


class PermitReceiptAdapter(Protocol):
    def validate_and_consume(self, *, mission_ref: str, tool_name: str, args_digest: str, target_ref: str) -> Mapping[str, Any]: ...
    def record_receipt(self, receipt: Mapping[str, Any]) -> None: ...


class PermitBridgeState(str, Enum):
    CONSUMED = "consumed"
    UNAVAILABLE = "unavailable"
    DENIED = "denied"
    MALFORMED = "malformed"


@dataclass(frozen=True)
class PermitBridgeOutcome:
    """Typed result from the canonical daemon; never a local permit projection."""
    state: PermitBridgeState
    code: str
    facts: Mapping[str, Any] | None = None


class OperatorApprovalWitnessProvider(Protocol):
    """Surface-owned source for a durable approval witness.

    Ares does not interpret, mint, cache, or upgrade an approval.  A UI or
    gateway integration must supply the canonical durable witness for this
    exact request; without one the effect is denied before daemon contact.
    """

    def issue_witness(
        self,
        *,
        mission_ref: str,
        target_ref: str,
        call: Mapping[str, Any],
    ) -> Mapping[str, Any] | None: ...


@dataclass(frozen=True)
class DesktopProductionApprovalEnvelope:
    """Exact, one-shot production approval request for the Desktop controller.

    This is deliberately an unsigned request, not a permit, binding, digest, or
    approval decision.  Electron owns the private signing key and must display
    this exact payload before returning its daemon-verifiable witness.  Keeping
    the request typed here lets the Ares-side provider fail closed without
    treating the existing gateway ``approval.respond`` choice as evidence.
    """

    approval_id: str
    schema: str
    mission_ref: str
    target_ref: str
    tool_name: str
    args: Mapping[str, Any]
    worktree_root: str
    validity_ms: int = 300_000
    one_use: bool = True
    retry_allowed: bool = False
    network_allowed: bool = False
    delegation_allowed: bool = False
    ambiguous_outcome: str = "terminal_quarantine"

    SCHEMA = "recursive-agent.desktop-production-approval-request/v1"

    @classmethod
    def for_call(
        cls,
        *,
        mission_ref: str,
        target_ref: str,
        call: Mapping[str, Any],
        worktree_root: str | Path,
        approval_id: str | None = None,
    ) -> "DesktopProductionApprovalEnvelope":
        if not isinstance(call, Mapping) or set(call) != {"tool", "args", "frozen_clock"}:
            raise ContractError("DESKTOP_APPROVAL_CALL_MALFORMED")
        if call.get("tool") != "write_file" or call.get("frozen_clock") is not None:
            raise ContractError("DESKTOP_APPROVAL_SCOPE_DENIED")
        args = call.get("args")
        if type(args) is not dict or set(args) != {"path", "content"}:
            raise ContractError("DESKTOP_APPROVAL_SCOPE_DENIED")
        if not isinstance(args["path"], str) or not isinstance(args["content"], str):
            raise ContractError("DESKTOP_APPROVAL_SCOPE_DENIED")
        _check_ref(mission_ref, "mission_ref")
        _check_ref(target_ref, "target_ref")
        selected_approval_id = approval_id or "approval:" + digest({"mission_ref": mission_ref, "target_ref": target_ref, "call": call})[7:]
        _check_ref(selected_approval_id, "approval_id")
        try:
            root = Path(worktree_root).resolve(strict=True)
            candidate = Path(args["path"]).resolve(strict=False)
            candidate.relative_to(root)
        except (OSError, ValueError):
            raise ContractError("DESKTOP_APPROVAL_SCOPE_DENIED") from None
        return cls(
            approval_id=selected_approval_id,
            schema=cls.SCHEMA,
            mission_ref=mission_ref,
            target_ref=target_ref,
            tool_name="write_file",
            # Preserve the original, un-normalized tool payload for display and
            # eventual daemon binding; path resolution above is validation only.
            args=_freeze(dict(args)),
            worktree_root=str(root),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the exact payload Electron must display and sign over."""
        return {
            "approval_id": self.approval_id,
            "schema": self.schema,
            "mission_ref": self.mission_ref,
            "target_ref": self.target_ref,
            "call": {"tool": self.tool_name, "args": _thaw(self.args), "frozen_clock": None},
            "constraints": {
                "validity_ms": self.validity_ms,
                "one_use": self.one_use,
                "retry_allowed": self.retry_allowed,
                "network_allowed": self.network_allowed,
                "delegation_allowed": self.delegation_allowed,
                "allowed_write_root": self.worktree_root,
                "ambiguous_outcome": self.ambiguous_outcome,
            },
        }


class DesktopProductionApprovalController(Protocol):
    """Electron-owned signing boundary; no gateway choice is accepted here."""

    def request_signed_witness(
        self, *, envelope: DesktopProductionApprovalEnvelope
    ) -> Mapping[str, Any] | None: ...


class DesktopProductionApprovalWitnessProvider:
    """Ask the Desktop controller for a witness for the one admitted call.

    The provider stores neither approval choices nor witnesses.  In particular,
    ``approval.respond`` is intentionally not an input: a controller must
    display the typed envelope and have Electron sign it before a future daemon
    protocol can consume the resulting opaque witness.
    """

    def __init__(self, *, controller: DesktopProductionApprovalController | None, worktree_root: str | Path) -> None:
        self._controller = controller
        self._worktree_root = Path(worktree_root)

    def issue_witness(
        self,
        *,
        mission_ref: str,
        target_ref: str,
        call: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        controller = self._controller
        if controller is None:
            return None
        envelope = DesktopProductionApprovalEnvelope.for_call(
            mission_ref=mission_ref,
            target_ref=target_ref,
            call=call,
            worktree_root=self._worktree_root,
        )
        witness = controller.request_signed_witness(envelope=envelope)
        # A bare choice (or any other non-witness scalar) is never evidence.
        return dict(witness) if isinstance(witness, Mapping) else None


class GatewayProductionApprovalWitnessProvider:
    """Bridge the daemon witness wait through the canonical gateway queue.

    The agent thread blocks here while the gateway owns the pending prompt. The
    renderer receives the typed ``production_permit`` envelope, Electron signs
    it, and the separate ``production_permit.respond`` method returns the opaque
    witness to this exact queue entry. No choice-only approval is upgraded.
    """

    def __init__(self, *, worktree_root: str | Path) -> None:
        self._worktree_root = Path(worktree_root)

    def issue_witness(
        self,
        *,
        mission_ref: str,
        target_ref: str,
        call: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        envelope = DesktopProductionApprovalEnvelope.for_call(
            mission_ref=mission_ref,
            target_ref=target_ref,
            call=call,
            worktree_root=self._worktree_root,
        )
        try:
            import tools.approval as approval

            session_key = approval.get_current_session_key(default="")
            with approval._lock:
                notify_cb = approval._gateway_notify_cbs.get(session_key)
            if not session_key or notify_cb is None:
                return None
            result = approval._await_gateway_decision(
                session_key,
                notify_cb,
                {
                    "command": json.dumps(envelope.to_dict(), sort_keys=True, separators=(",", ":")),
                    "description": "One-time approval for the exact bounded production write.",
                    "pattern_key": "production_per_call_write_file",
                    "pattern_keys": ["production_per_call_write_file"],
                    "allow_permanent": False,
                    "allow_session": False,
                    "choices": ["once", "deny"],
                    "production_permit": envelope.to_dict(),
                },
                surface="production_permit",
            )
        except Exception:
            return None
        if result.get("resolved") and result.get("choice") == "once":
            witness = result.get("witness")
            return dict(witness) if isinstance(witness, Mapping) and witness else None
        return None


class DaemonPermitReceiptAdapter:
    """Thin client for the canonical daemon-owned permit and receipt lanes.

    Configuration is read-only from ``ares.permit_daemon`` in Hermes config and
    supplies only transport settings.  For every effect, a surface-owned
    ``OperatorApprovalWitnessProvider`` must supply a durable witness for the
    exact call.  The daemon alone then issues the permit and binding, consumes
    that same call, and records the outcome.  This adapter never mints permits,
    creates approval witnesses, or persists receipts locally.
    """
    REQUEST_SCHEMA = "recursive-agent.ipc/request/v1"
    PROTOCOL_VERSION = 1
    MAX_FRAME_BYTES = 1024 * 1024 + 64 * 1024

    def __init__(self, config: Mapping[str, Any], *, approval_witness_provider: OperatorApprovalWitnessProvider | None = None) -> None:
        self._config = dict(config)
        self._approval_witness_provider = approval_witness_provider

    @classmethod
    def from_ares_config(cls, config: Mapping[str, Any]) -> "DaemonPermitReceiptAdapter | None":
        ares = config.get("ares") if isinstance(config, Mapping) else None
        bridge = ares.get("permit_daemon") if isinstance(ares, Mapping) else None
        if not isinstance(bridge, Mapping):
            return None
        provider = None
        if bridge.get("mode") == cls._PRODUCTION_MODE:
            provider = GatewayProductionApprovalWitnessProvider(
                worktree_root=bridge.get("worktree_root", "/home/sikmindz/work/ares-production-permit-20260830")
            )
        return cls(bridge, approval_witness_provider=provider)

    _TEST_ONLY_MODE = "test_only_echo"
    _TEST_ONLY_WITNESS_SCHEMA = "recursive-agent.operator-test-permit-issuance-approval/v1"
    # This is the daemon's closed wire witness, not an Ares approval schema.
    # The provider must obtain it from the trusted controller after displaying
    # the exact call. Ares validates only shape and forwards it unchanged; the
    # daemon recomputes and authenticates the canonical request binding.
    _TEST_ONLY_WITNESS_FIELDS = frozenset({"operator_case", "request_digest", "authenticator"})

    def _test_only_enabled(self) -> bool:
        """Production/default configuration never enables this test fixture."""
        return self._config.get("mode") == self._TEST_ONLY_MODE

    def _validate_test_only_witness(self, witness: Mapping[str, Any]) -> dict[str, Any]:
        if set(witness) != self._TEST_ONLY_WITNESS_FIELDS:
            raise ContractError("OPERATOR_APPROVAL_WITNESS_MALFORMED")
        value = dict(witness)
        if (
            not isinstance(value["operator_case"], str)
            or not value["operator_case"].startswith("approval:test:")
            or not isinstance(value["request_digest"], str)
            or len(value["request_digest"]) != 64
            or any(char not in "0123456789abcdef" for char in value["request_digest"])
            or not isinstance(value["authenticator"], str)
            or len(value["authenticator"]) != 64
            or any(char not in "0123456789abcdef" for char in value["authenticator"])
        ):
            raise ContractError("OPERATOR_APPROVAL_WITNESS_MALFORMED")
        return value

    def _operator_approval_witness(self, *, mission_ref: str, target_ref: str, call: Mapping[str, Any]) -> dict[str, Any]:
        provider = self._approval_witness_provider
        if provider is None:
            raise ContractError("OPERATOR_APPROVAL_WITNESS_UNAVAILABLE")
        try:
            witness = provider.issue_witness(mission_ref=mission_ref, target_ref=target_ref, call=call)
        except ContractError:
            raise
        except Exception:
            raise ContractError("OPERATOR_APPROVAL_WITNESS_UNAVAILABLE") from None
        if witness is None:
            raise ContractError("OPERATOR_APPROVAL_WITNESS_MISSING")
        if not isinstance(witness, Mapping):
            raise ContractError("OPERATOR_APPROVAL_WITNESS_MALFORMED")
        return self._validate_test_only_witness(witness)

    def canonical_args_digest(self, args: Any) -> str:
        """Ask the daemon owner to canonically digest an exact tool payload."""
        command = self._config.get("canonical_digest_command")
        timeout = self._config.get("timeout_seconds", 5.0)
        if (
            not isinstance(command, str)
            or not os.path.isabs(command)
            or not isinstance(timeout, (int, float))
            or timeout <= 0
        ):
            raise ContractError("CANONICAL_DIGEST_UNAVAILABLE")
        input_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(mode="wb", prefix="ares-permit-", suffix=".json", delete=False) as handle:
                input_path = handle.name
                os.fchmod(handle.fileno(), 0o600)
                handle.write(canonical_json(args))
            result = subprocess.run(
                [command, "canonical-digest", "--json", input_path],
                capture_output=True,
                text=True,
                timeout=float(timeout),
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            raise ContractError("CANONICAL_DIGEST_UNAVAILABLE") from None
        finally:
            if input_path is not None:
                try:
                    os.unlink(input_path)
                except OSError:
                    pass
        if result.returncode != 0 or not isinstance(result.stdout, str):
            raise ContractError("CANONICAL_DIGEST_UNAVAILABLE")
        raw = result.stdout
        if raw.endswith("\n"):
            raw = raw[:-1]
        if len(raw) != 64 or any(character not in "0123456789abcdef" for character in raw):
            raise ContractError("CANONICAL_DIGEST_MALFORMED")
        return raw

    @staticmethod
    def _send_frame(stream: socket.socket, value: Mapping[str, Any]) -> None:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
        if len(payload) > DaemonPermitReceiptAdapter.MAX_FRAME_BYTES:
            raise ContractError("PERMIT_BRIDGE_MALFORMED")
        stream.sendall(struct.pack(">I", len(payload)) + payload)

    @staticmethod
    def _recv_exact(stream: socket.socket, size: int) -> bytes:
        result = bytearray()
        while len(result) < size:
            chunk = stream.recv(size - len(result))
            if not chunk:
                raise ContractError("PERMIT_BRIDGE_UNAVAILABLE")
            result.extend(chunk)
        return bytes(result)

    def _connect(self) -> socket.socket:
        path = self._config.get("socket_path")
        timeout = self._config.get("timeout_seconds", 5.0)
        if not isinstance(path, str) or not path or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ContractError("PERMIT_BRIDGE_UNAVAILABLE")
        try:
            node = Path(path).stat()
            if not stat.S_ISSOCK(node.st_mode) or node.st_uid != os.geteuid() or node.st_mode & 0o022:
                raise ContractError("PERMIT_BRIDGE_UNAVAILABLE")
            stream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            stream.settimeout(float(timeout))
            stream.connect(path)
            if not hasattr(socket, "SO_PEERCRED"):
                stream.close()
                raise ContractError("PERMIT_BRIDGE_UNAVAILABLE")
            peer_pid, peer_uid, _peer_gid = struct.unpack("3i", stream.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
            if peer_pid <= 0 or peer_uid != os.geteuid():
                stream.close()
                raise ContractError("PERMIT_BRIDGE_UNAVAILABLE")
            return stream
        except ContractError:
            raise
        except (OSError, ValueError):
            raise ContractError("PERMIT_BRIDGE_UNAVAILABLE") from None

    _PRODUCTION_MODE = "production_per_call"

    def _production_enabled(self) -> bool:
        return self._config.get("mode") == self._PRODUCTION_MODE

    def _production_approval_witness(
        self, *, mission_ref: str, target_ref: str, call: Mapping[str, Any]
    ) -> dict[str, Any]:
        provider = self._approval_witness_provider
        if provider is None:
            raise ContractError("OPERATOR_APPROVAL_WITNESS_UNAVAILABLE")
        try:
            witness = provider.issue_witness(
                mission_ref=mission_ref, target_ref=target_ref, call=call
            )
        except ContractError:
            raise
        except Exception:
            raise ContractError("OPERATOR_APPROVAL_WITNESS_UNAVAILABLE") from None
        if witness is None:
            raise ContractError("OPERATOR_APPROVAL_WITNESS_MISSING")
        if not isinstance(witness, Mapping) or not witness:
            raise ContractError("OPERATOR_APPROVAL_WITNESS_MALFORMED")
        # Production witness fields and signature remain opaque here: only the
        # daemon's injected public verifier owns their canonical validation.
        return dict(witness)

    def consume(self, *, mission_ref: str, tool_name: str, args: Mapping[str, Any], target_ref: str) -> PermitBridgeOutcome:
        production = self._production_enabled()
        if not self._test_only_enabled() and not production:
            return PermitBridgeOutcome(PermitBridgeState.DENIED, "TEST_ONLY_ECHO_DISABLED")
        if production:
            if tool_name != "write_file" or type(args) is not dict or set(args) != {"path", "content"} or not all(isinstance(args.get(key), str) for key in ("path", "content")):
                return PermitBridgeOutcome(PermitBridgeState.DENIED, "PRODUCTION_WRITE_FILE_REQUIRED")
            call = {"tool": "write_file", "args": {"path": args["path"], "content": args["content"]}, "frozen_clock": None}
            try:
                witness = self._production_approval_witness(mission_ref=mission_ref, target_ref=target_ref, call=call)
            except ContractError as exc:
                state = PermitBridgeState.UNAVAILABLE if exc.code == "OPERATOR_APPROVAL_WITNESS_UNAVAILABLE" else PermitBridgeState.DENIED
                return PermitBridgeOutcome(state, exc.code)
            issuance_body = {"kind": "permit_issue_production", "witness": witness}
        else:
            if tool_name != "echo" or type(args) is not dict or set(args) != {"text"} or not isinstance(args.get("text"), str) or target_ref != "tool:echo":
                return PermitBridgeOutcome(PermitBridgeState.DENIED, "TEST_ONLY_ECHO_REQUIRED")
            call = {"tool": "echo", "args": {"text": args["text"]}, "frozen_clock": None}
            try:
                witness = self._operator_approval_witness(mission_ref=mission_ref, target_ref=target_ref, call=call)
            except ContractError as exc:
                state = PermitBridgeState.UNAVAILABLE if exc.code == "OPERATOR_APPROVAL_WITNESS_UNAVAILABLE" else PermitBridgeState.DENIED
                return PermitBridgeOutcome(state, exc.code)
            issuance_body = {"kind": "permit_issue", "request": {"call": call, "requested_validity_ms": 300000}, "approval": witness}
        issuance_request_id = "ares:" + secrets.token_hex(16)
        issuance_request = {
            "schema": self.REQUEST_SCHEMA,
            "protocol_version": self.PROTOCOL_VERSION,
            "request_id": issuance_request_id,
            "request": issuance_body,
        }
        try:
            with self._connect() as stream:
                self._send_frame(stream, issuance_request)
                length = struct.unpack(">I", self._recv_exact(stream, 4))[0]
                if length > self.MAX_FRAME_BYTES:
                    return PermitBridgeOutcome(PermitBridgeState.MALFORMED, "PERMIT_BRIDGE_MALFORMED")
                issuance_response = json.loads(self._recv_exact(stream, length).decode("utf-8"))
        except ContractError as exc:
            return PermitBridgeOutcome(PermitBridgeState.UNAVAILABLE, exc.code)
        except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
            return PermitBridgeOutcome(PermitBridgeState.UNAVAILABLE, "PERMIT_BRIDGE_UNAVAILABLE")
        if not isinstance(issuance_response, Mapping) or issuance_response.get("request_id") != issuance_request_id:
            return PermitBridgeOutcome(PermitBridgeState.MALFORMED, "PERMIT_BRIDGE_MALFORMED")
        error = issuance_response.get("error")
        if error is not None:
            if not isinstance(error, Mapping) or error.get("code") != "runtime_error" or not isinstance(error.get("message"), str):
                return PermitBridgeOutcome(PermitBridgeState.MALFORMED, "PERMIT_BRIDGE_MALFORMED")
            return PermitBridgeOutcome(PermitBridgeState.DENIED, "PERMIT_DENIED", dict(issuance_response))
        permit_id = issuance_response.get("permit_id")
        binding = issuance_response.get("binding")
        if not isinstance(permit_id, str) or not permit_id or not isinstance(binding, Mapping) or not binding:
            return PermitBridgeOutcome(PermitBridgeState.MALFORMED, "PERMIT_BRIDGE_MALFORMED")
        request_id = "ares:" + secrets.token_hex(16)
        request = {
            "schema": self.REQUEST_SCHEMA,
            "protocol_version": self.PROTOCOL_VERSION,
            "request_id": request_id,
            "request": {
                "kind": "permit_consume",
                "permit_id": permit_id,
                "binding": dict(binding),
                "call": call,
            },
        }
        try:
            with self._connect() as stream:
                self._send_frame(stream, request)
                length = struct.unpack(">I", self._recv_exact(stream, 4))[0]
                if length > self.MAX_FRAME_BYTES:
                    return PermitBridgeOutcome(PermitBridgeState.MALFORMED, "PERMIT_BRIDGE_MALFORMED")
                response = json.loads(self._recv_exact(stream, length).decode("utf-8"))
        except ContractError as exc:
            return PermitBridgeOutcome(PermitBridgeState.UNAVAILABLE, exc.code)
        except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
            return PermitBridgeOutcome(PermitBridgeState.UNAVAILABLE, "PERMIT_BRIDGE_UNAVAILABLE")
        if not isinstance(response, Mapping) or response.get("request_id") != request_id:
            return PermitBridgeOutcome(PermitBridgeState.MALFORMED, "PERMIT_BRIDGE_MALFORMED")
        error = response.get("error")
        if error is not None:
            if not isinstance(error, Mapping) or error.get("code") != "runtime_error" or not isinstance(error.get("message"), str):
                return PermitBridgeOutcome(PermitBridgeState.MALFORMED, "PERMIT_BRIDGE_MALFORMED")
            return PermitBridgeOutcome(PermitBridgeState.DENIED, "PERMIT_DENIED", dict(response))
        required = {"permit_id", "evidence", "preflight_artifact", "receipt_artifact"}
        if not required <= set(response) or response.get("permit_id") != permit_id or not all(isinstance(response.get(key), Mapping) for key in ("evidence", "preflight_artifact", "receipt_artifact")):
            return PermitBridgeOutcome(PermitBridgeState.MALFORMED, "PERMIT_BRIDGE_MALFORMED")
        facts = dict(response)
        facts["canonical_permit_ref"] = permit_id
        return PermitBridgeOutcome(PermitBridgeState.CONSUMED, "PERMIT_CONSUMED", facts)

    def validate_and_consume_call(self, *, mission_ref: str, tool_name: str, args: Mapping[str, Any], target_ref: str) -> Mapping[str, Any]:
        outcome = self.consume(mission_ref=mission_ref, tool_name=tool_name, args=args, target_ref=target_ref)
        if outcome.state is not PermitBridgeState.CONSUMED or outcome.facts is None:
            raise ContractError(outcome.code)
        return outcome.facts

    def record_receipt(self, receipt: Mapping[str, Any]) -> None:
        if not self._test_only_enabled() and not self._production_enabled():
            raise ContractError("TEST_ONLY_ECHO_DISABLED")
        permit_id = receipt.get("permit_ref")
        preflight = receipt.get("preflight_receipt")
        state = receipt.get("state")
        duration_ms = receipt.get("duration_ms")
        error_type = receipt.get("error_type")
        if (
            not isinstance(permit_id, str)
            or not isinstance(preflight, Mapping)
            or not isinstance(preflight.get("receipt_digest"), str)
            or state not in {"ok", "error", "ambiguous"}
            or not isinstance(duration_ms, int)
            or duration_ms < 0
            or error_type is not None and not isinstance(error_type, str)
        ):
            raise ContractError("PERMIT_BRIDGE_MALFORMED")
        safe_error_type = (
            error_type
            if isinstance(error_type, str) and error_type and len(error_type) <= 256 and not any(character.isspace() and character != " " for character in error_type)
            else "tool_error" if state == "error" else None
        )
        reported_state = "succeeded" if state == "ok" else "failed" if state == "error" else "outcome_ambiguous"
        request_id = "ares:" + secrets.token_hex(16)
        request = {
            "schema": self.REQUEST_SCHEMA,
            "protocol_version": self.PROTOCOL_VERSION,
            "request_id": request_id,
            "request": {
                "kind": "permit_outcome_record",
                "permit_id": permit_id,
                "preflight_receipt_digest": preflight["receipt_digest"],
                "reported": {
                    "state": reported_state,
                    "duration_ms": duration_ms,
                    "error_type": safe_error_type,
                },
            },
        }
        try:
            with self._connect() as stream:
                self._send_frame(stream, request)
                length = struct.unpack(">I", self._recv_exact(stream, 4))[0]
                if length > self.MAX_FRAME_BYTES:
                    raise ContractError("PERMIT_BRIDGE_MALFORMED")
                response = json.loads(self._recv_exact(stream, length).decode("utf-8"))
        except ContractError:
            raise
        except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
            raise ContractError("PERMIT_BRIDGE_UNAVAILABLE") from None
        if not isinstance(response, Mapping) or response.get("request_id") != request_id:
            raise ContractError("PERMIT_BRIDGE_MALFORMED")
        if response.get("error") is not None:
            raise ContractError("PERMIT_OUTCOME_DENIED")
        outcome_artifact = response.get("outcome_artifact")
        if response.get("permit_id") != permit_id or not isinstance(outcome_artifact, Mapping):
            raise ContractError("PERMIT_BRIDGE_MALFORMED")
        if state == "ambiguous" and outcome_artifact.get("state") != "terminal_quarantine":
            raise ContractError("PERMIT_BRIDGE_MALFORMED")


def configured_permit_adapter() -> PermitReceiptAdapter | None:
    try:
        from hermes_cli.config import load_config_readonly
        return DaemonPermitReceiptAdapter.from_ares_config(load_config_readonly())
    except Exception:
        return None


_ADAPTER: contextvars.ContextVar[PermitReceiptAdapter | None] = contextvars.ContextVar("ares_permit_adapter", default=None)
_EFFECTFUL_TOOLS = frozenset({"write_file", "patch", "terminal", "execute_code", "browser_exec", "browser_click", "browser_type", "browser_navigate", "browser_press", "browser_scroll", "browser_back", "browser_dialog", "browser_cdp", "browser_console", "send_message", "computer_use", "image_generate", "bfl_flux3_image_to_video", "bfl_flux3_keyframes_to_video", "bfl_flux3_text_to_video", "bfl_flux3_video_continuation"})
_DEFAULT_EFFECT_SCHEMAS: dict[str, Mapping[str, Any]] = {
    "write_file": {"type": "object", "required": ["path", "content"], "properties": {"path": {"type": "string"}, "content": {"type": "string"}}},
    "patch": {"type": "object", "required": ["path", "old_string", "new_string"], "properties": {"path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}, "replace_all": {"type": "boolean"}}},
    "terminal": {"type": "object", "required": ["command"], "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}, "workdir": {"type": "string"}, "pty": {"type": "boolean"}, "background": {"type": "boolean"}, "notify_on_complete": {"type": "boolean"}}},
    "execute_code": {"type": "object", "required": ["code"], "properties": {"code": {"type": "string"}}},
    "browser_exec": {"type": "object", "required": ["code"], "properties": {"code": {"type": "string"}}},
    "browser_navigate": {"type": "object", "required": ["url"], "properties": {"url": {"type": "string"}}},
    "browser_type": {"type": "object", "required": ["text"], "properties": {"text": {"type": "string"}}},
    "browser_click": {"type": "object", "properties": {"element": {"type": "integer"}, "coordinate": {"type": "array", "items": {"type": "integer"}}, "button": {"type": "string"}, "modifiers": {"type": "array", "items": {"type": "string"}}}},
    "browser_press": {"type": "object", "required": ["keys"], "properties": {"keys": {"type": "string"}}},
    "browser_scroll": {"type": "object", "properties": {"direction": {"type": "string"}, "amount": {"type": "integer"}}},
    "browser_back": {"type": "object", "properties": {}},
    "browser_dialog": {"type": "object", "required": ["action"], "properties": {"action": {"type": "string"}, "prompt_text": {"type": "string"}}},
    "browser_cdp": {"type": "object", "required": ["code"], "properties": {"code": {"type": "string"}}},
    "browser_console": {"type": "object", "properties": {"level": {"type": "string"}}},
    "send_message": {"type": "object", "required": ["recipient", "content"], "properties": {"recipient": {"type": "string"}, "content": {"type": "string"}}},
    "computer_use": {"type": "object", "required": ["action"], "properties": {"action": {"type": "string"}, "app": {"type": "string"}, "pid": {"type": "integer"}, "window_id": {"type": "integer"}}},
    "image_generate": {"type": "object", "required": ["prompt"], "properties": {"prompt": {"type": "string"}, "aspect_ratio": {"type": "string"}}},
    "bfl_flux3_image_to_video": {"type": "object", "required": ["prompt", "input_image"], "properties": {"prompt": {"type": "string"}, "input_image": {"type": "string"}, "aspect_ratio": {"type": "string"}, "duration": {"type": "integer"}, "resolution": {"type": "string"}, "generate_audio": {"type": "boolean"}}},
    "bfl_flux3_keyframes_to_video": {"type": "object", "required": ["prompt", "input_images"], "properties": {"prompt": {"type": "string"}, "input_images": {"type": "array", "items": {"type": "string"}}, "keyframe_indices": {"type": "array", "items": {"type": "integer"}}, "aspect_ratio": {"type": "string"}, "duration": {"type": "integer"}}},
    "bfl_flux3_text_to_video": {"type": "object", "required": ["prompt"], "properties": {"prompt": {"type": "string"}, "aspect_ratio": {"type": "string"}, "duration": {"type": "integer"}, "resolution": {"type": "string"}, "generate_audio": {"type": "boolean"}}},
    "bfl_flux3_video_continuation": {"type": "object", "required": ["prompt", "input_video"], "properties": {"prompt": {"type": "string"}, "input_video": {"type": "string"}, "aspect_ratio": {"type": "string"}, "duration": {"type": "integer"}}},
}


def _exact_type(value: Any, expected: Any) -> bool:
    if isinstance(expected, list): return any(_exact_type(value, x) for x in expected)
    return {"object": type(value) is dict, "array": type(value) is list, "string": type(value) is str, "integer": type(value) is int, "number": type(value) in (int, float) and type(value) is not bool, "boolean": type(value) is bool, "null": value is None}.get(expected, False)


def validate_effect_args(args: Any, schema: Mapping[str, Any]) -> None:
    if type(args) is not dict or schema.get("type") != "object": raise ContractError("INVALID_EFFECT_ARGS")
    required, allowed = set(schema.get("required", [])), set(schema.get("properties", {}))
    if set(args) - allowed: raise ContractError("UNKNOWN_FIELD", sorted(set(args) - allowed)[0])
    if required - set(args): raise ContractError("MISSING_FIELD", sorted(required - set(args))[0])
    for key, spec in schema.get("properties", {}).items():
        if key in args and not _exact_type(args[key], spec.get("type")): raise ContractError("COERCION_REQUIRED", key)


def target_for(tool_name: str, args: Mapping[str, Any]) -> str:
    if tool_name in {"write_file", "patch"}:
        if not isinstance(args.get("path"), str): raise ContractError("INVALID_TARGET", "path")
        return "path:" + digest(args["path"])[7:]
    if tool_name in {"send_message"}:
        if not isinstance(args.get("recipient"), str): raise ContractError("INVALID_TARGET", "recipient")
        return "recipient:" + digest(args["recipient"])[7:]
    return "tool:" + tool_name


def dispatcher_boundary(tool_name: str, args: Any, *, mission_ref: str | None, schema: Mapping[str, Any] | None = None, target_ref: str | None = None, authorize_permit: bool = True, consume_permit: bool = True) -> tuple[bool, str | None, Mapping[str, Any] | None]:
    strict = os.getenv("ARES_STRICT_EFFECT_TOOL_ARGS_V1", "0") == "1"
    permits = os.getenv("ARES_RUNTIME_PERMITS_V1", "0") == "1"
    if tool_name in _EFFECTFUL_TOOLS and strict:
        strict_schema = schema or _DEFAULT_EFFECT_SCHEMAS.get(tool_name)
        if strict_schema is None: return False, "EFFECT_SCHEMA_MISSING", None
        try: validate_effect_args(args, strict_schema)
        except ContractError as exc: return False, exc.code, None
    if not permits or tool_name not in _EFFECTFUL_TOOLS or not authorize_permit: return True, None, None
    adapter = _ADAPTER.get() or configured_permit_adapter()
    if adapter is None: return False, "PERMIT_BRIDGE_UNAVAILABLE", None
    if not mission_ref: return False, "PERMIT_MISSING", None
    if not consume_permit: return True, None, None
    try:
        target = target_ref or target_for(tool_name, args)
        if isinstance(adapter, DaemonPermitReceiptAdapter):
            permit = adapter.validate_and_consume_call(
                mission_ref=mission_ref,
                tool_name=tool_name,
                args=args,
                target_ref=target,
            )
        else:
            permit = adapter.validate_and_consume(
                mission_ref=mission_ref,
                tool_name=tool_name,
                args_digest=digest(args),
                target_ref=target,
            )
    except ContractError as exc: return False, exc.code, None
    except Exception: return False, "PERMIT_DENIED", None
    return True, None, permit


def record_receipt(receipt: Mapping[str, Any]) -> None:
    adapter = _ADAPTER.get()
    if adapter is not None: adapter.record_receipt(dict(receipt))


def permit_adapter(adapter: PermitReceiptAdapter): return _ADAPTER.set(adapter)
def reset_permit_adapter(token: contextvars.Token) -> None: _ADAPTER.reset(token)


class BlindWitness:
    """One-shot witness receipt: commit before reveal on a distinct route."""

    def __init__(self) -> None:
        self._committed_context: str | None = None
        self._commitment: str | None = None
        self._executor_route_ref: str | None = None
        self._witness_route_ref: str | None = None
        self._revealed = False

    def commit(self, context_digest: str, *, executor_route_ref: str, witness_route_ref: str) -> str:
        if self._commitment is not None or self._revealed:
            raise ContractError("COMMITMENT_ALREADY_RECORDED")
        if not isinstance(context_digest, str) or not context_digest.startswith("sha256:"):
            raise ContractError("INVALID_CONTEXT_DIGEST")
        _check_ref(executor_route_ref, "executor_route_ref")
        _check_ref(witness_route_ref, "witness_route_ref")
        if executor_route_ref == witness_route_ref:
            raise ContractError("WITNESS_ROUTE_NOT_INDEPENDENT")
        self._committed_context = context_digest
        self._executor_route_ref = executor_route_ref
        self._witness_route_ref = witness_route_ref
        self._commitment = digest({"context_digest": context_digest, "executor_route_ref": executor_route_ref, "witness_route_ref": witness_route_ref, "nonce": secrets.token_hex(16)})
        return self._commitment

    def reveal_and_record(self, result: Mapping[str, Any], *, mission_ref: str, test_request_ref: str, role_contract_ref: str, context_digest: str, executor_route_ref: str, witness_route_ref: str) -> ImmutableArtifact:
        if self._commitment is None or self._revealed:
            raise ContractError("COMMITMENT_REQUIRED")
        if executor_route_ref != self._executor_route_ref or witness_route_ref != self._witness_route_ref or context_digest != self._committed_context:
            raise ContractError("COMMITMENT_MISMATCH")
        if executor_route_ref == witness_route_ref:
            raise ContractError("WITNESS_ROUTE_NOT_INDEPENDENT")
        if not isinstance(result, Mapping) or result.get("verdict") not in {"pass", "fail", "blocked", "inconclusive", "not_applicable"}:
            raise ContractError("INVALID_WITNESS_RESULT")
        self._revealed = True
        return make_artifact("witness", {"mission_ref": mission_ref, "test_request_ref": test_request_ref, "role_contract_ref": role_contract_ref, "witness_id": "witness:" + digest(result)[7:], "independence": {"executor_output_withheld_until_commit": True, "context_digest": context_digest, "commitment": self._commitment, "executor_route_ref": executor_route_ref, "witness_route_ref": witness_route_ref, "shared_evidence_refs": []}, "verdict": result["verdict"], "evidence_refs": list(result.get("evidence_refs", [])), "coverage": list(result.get("coverage", ["declared test request"])), "limitations": list(result.get("limitations", []))})


class ClosureProjector:
    """Pure, rebuildable projection over owner events; it never closes a mission."""

    PROFILE_REQUIRED_GATES = {"engineering": frozenset({"test"}), "research": frozenset({"evidence_review"}), "public_artifact": frozenset({"artifact_verification"}), "effectful_operation": frozenset({"effect_receipt"}), "ordinary_response": frozenset({"response_evidence"}), "mixed": frozenset({"test", "evidence_review"})}
    ALLOWED_FLAGS = {"KANBAN_DONE_WITHOUT_CLOSURE", "GOAL_DONE_WITHOUT_CLOSURE", "NARRATIVE_AHEAD_OF_LEDGER", "LEDGER_AHEAD_OF_UI", "STALE_SOURCE_FREEZE", "AMBIGUOUS_EFFECT", "MISSING_WITNESS", "RUNTIME_CANDIDATE_NOT_ACTIVE", "UI_TASK_COUNT_MISMATCH"}

    def project(self, mission_ref: str, closure_profile: str, gates: Mapping[str, bool], *, source_event_refs: Sequence[str], source_event_exists: Callable[[str], bool], flags: Sequence[str] = (), previous_projection: Mapping[str, Any] | ImmutableArtifact | None = None) -> ImmutableArtifact:
        _check_ref(mission_ref, "mission_ref")
        if closure_profile not in self.PROFILE_REQUIRED_GATES:
            raise ContractError("UNKNOWN_CLOSURE_PROFILE", closure_profile)
        if not isinstance(gates, Mapping) or any(type(value) is not bool for value in gates.values()):
            raise ContractError("INVALID_GATE_STATE")
        if not source_event_refs:
            raise ContractError("MISSING_SOURCE_EVENTS")
        normalized_events = sorted(set(source_event_refs))
        for event_ref in normalized_events:
            _check_ref(event_ref, "source_event_refs")
            if not source_event_exists(event_ref):
                raise ContractError("MISSING_SOURCE_EVENT", event_ref)
        if previous_projection is not None:
            prior = previous_projection.to_dict() if isinstance(previous_projection, ImmutableArtifact) else dict(previous_projection)
            if prior.get("mission_ref") != mission_ref or prior.get("closure_profile") != closure_profile:
                raise ContractError("PROJECTION_LINEAGE_MISMATCH")
            if prior.get("state") == "closed" and not set(normalized_events).difference(prior.get("source_event_refs", [])):
                raise ContractError("REOPEN_REQUIRES_NEW_EVIDENCE")
        unknown = set(flags) - self.ALLOWED_FLAGS
        if unknown:
            raise ContractError("UNKNOWN_DIVERGENCE_FLAG", sorted(unknown)[0])
        effective_gates = dict(gates)
        for required_gate in self.PROFILE_REQUIRED_GATES[closure_profile]:
            effective_gates.setdefault(required_gate, False)
        satisfied = sorted(key for key, value in effective_gates.items() if value)
        unsatisfied = sorted(key for key, value in effective_gates.items() if not value)
        state = "closed" if not unsatisfied and not flags else ("quarantined" if "AMBIGUOUS_EFFECT" in flags else "evidence_pending")
        return make_artifact("closure", {"mission_ref": mission_ref, "closure_profile": closure_profile, "state": state, "satisfied_gate_ids": satisfied, "unsatisfied_gate_ids": unsatisfied, "blocking_refs": unsatisfied, "source_event_refs": normalized_events, "projected_at": EPOCH, "divergence_flags": sorted(set(flags))})


BASELINE_DEFINITIONS: Mapping[str, Mapping[str, str]] = {
    "B0": {"label": "incumbent", "topology": "incumbent"},
    "B1": {"label": "single_executor", "topology": "single_executor"},
    "B2": {"label": "homogeneous_blind_pair", "topology": "homogeneous_blind_pair"},
    "B3": {"label": "heterogeneous_blind_pair", "topology": "heterogeneous_blind_pair"},
}

# Canonical names are intentionally narrow.  Legacy spellings remain accepted
# only so historical replay fixtures can be evaluated, never silently omitted.
_MUTATION_ALIASES = {
    "skip_required_test": "skipped_test",
    "source_revision_mismatch": "stale_source",
    "missing_evidence_ref": "missing_evidence",
}
_REQUIRED_MUTATIONS = frozenset({
    "skipped_test", "ambiguous_effect", "missing_witness", "stale_source", "missing_evidence",
})


@dataclass(frozen=True)
class FrozenReplayCorpusV1:
    """Immutable, canonical replay input; it carries no runtime or UI truth."""
    corpus_version: str
    corpus_digest: str
    _canonical_cases: bytes

    @property
    def cases(self) -> tuple[dict[str, Any], ...]:
        return tuple(json.loads(self._canonical_cases.decode("utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus_version": self.corpus_version,
            "cases": list(self.cases),
            "corpus_digest": self.corpus_digest,
        }


def freeze_replay_corpus(cases: Sequence[Mapping[str, Any]], *, corpus_version: str = "held-out-v1") -> FrozenReplayCorpusV1:
    """Validate, sort, and freeze a replay corpus before deterministic grading."""
    if not isinstance(corpus_version, str) or not corpus_version:
        raise ContractError("INVALID_CORPUS_VERSION")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in cases:
        if not isinstance(raw, Mapping):
            raise ContractError("INVALID_REPLAY_CASE")
        case = dict(raw)
        case_id = case.get("case_id")
        if not isinstance(case_id, str):
            raise ContractError("INVALID_REFERENCE", "case_id")
        _check_ref(case_id, "case_id")
        if case_id in seen:
            raise ContractError("DUPLICATE_REPLAY_CASE", case_id)
        seen.add(case_id)
        normalized.append(case)
    if not normalized:
        raise ContractError("EMPTY_REPLAY_CORPUS")
    normalized.sort(key=lambda case: case["case_id"])
    try:
        frozen_bytes = canonical_json(normalized)
    except (TypeError, ValueError) as exc:
        raise ContractError("INVALID_REPLAY_CASE") from exc
    corpus_digest = digest({"corpus_version": corpus_version, "cases": normalized})
    return FrozenReplayCorpusV1(corpus_version, corpus_digest, frozen_bytes)


@dataclass(frozen=True)
class BaselineResultV1:
    """Immutable per-baseline evaluation projection, suitable for comparison only."""
    baseline_id: str
    corpus_digest: str
    result_digest: str
    _canonical_outcomes: bytes

    @property
    def cases(self) -> tuple[dict[str, Any], ...]:
        return tuple(json.loads(self._canonical_outcomes.decode("utf-8")))

    def to_dict(self) -> dict[str, Any]:
        cases = list(self.cases)
        return {
            "baseline_id": self.baseline_id,
            "definition": dict(BASELINE_DEFINITIONS[self.baseline_id]),
            "corpus_digest": self.corpus_digest,
            "cases": cases,
            "verified_closure_count": sum(item["verified_closure"] for item in cases),
            "false_closure_count": sum(item["false_closure"] for item in cases),
            "authority_violation_count": sum(item["authority_violation"] for item in cases),
            "quarantined": any(item["quarantined"] for item in cases),
            "result_digest": self.result_digest,
        }


def _normalized_mutation(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ContractError("INVALID_MUTATION")
    return _MUTATION_ALIASES.get(value, value)


def _case_outcome(case: Mapping[str, Any], baseline_id: str) -> dict[str, Any]:
    baseline_overrides = case.get("baselines", {})
    if not isinstance(baseline_overrides, Mapping):
        raise ContractError("INVALID_BASELINE_OVERRIDES")
    override = baseline_overrides.get(baseline_id, {})
    if not isinstance(override, Mapping):
        raise ContractError("INVALID_BASELINE_OVERRIDE", baseline_id)
    gates = dict(case.get("gates", {}))
    if not all(isinstance(name, str) and type(value) is bool for name, value in gates.items()):
        raise ContractError("INVALID_REPLAY_GATES")
    override_gates = override.get("gates", {})
    if not isinstance(override_gates, Mapping) or not all(isinstance(name, str) and type(value) is bool for name, value in override_gates.items()):
        raise ContractError("INVALID_REPLAY_GATES")
    gates.update(override_gates)
    mutation = _normalized_mutation(override.get("mutation", case.get("mutation")))
    if mutation in _REQUIRED_MUTATIONS:
        gates[mutation] = False
    divergence_flags = override.get("divergence_flags", case.get("divergence_flags", ()))
    if not isinstance(divergence_flags, (list, tuple, set)) or not all(isinstance(flag, str) for flag in divergence_flags):
        raise ContractError("INVALID_DIVERGENCE_FLAGS")
    closed = bool(gates) and all(gates.values()) and not divergence_flags
    expected_verified = bool(override.get("expected_verified_closure", case.get("expected_verified_closure", True)))
    authority_violation = bool(override.get("authority_violation", case.get("authority_violation", False)))
    critical_authority_violation = bool(override.get("critical_authority_violation", case.get("critical_authority_violation", authority_violation)))
    if mutation == "critical_authority_violation":
        authority_violation = critical_authority_violation = True
    quarantined = critical_authority_violation
    if quarantined:
        closed = False
    verified_closure = closed and expected_verified and not authority_violation
    false_closure = closed and not expected_verified
    closure_state = "quarantined" if quarantined else ("closed" if verified_closure else "evidence_pending")
    outcome = {
        "case_id": case["case_id"], "mutation": mutation, "closed": closed,
        "verified_closure": verified_closure, "false_closure": false_closure,
        "authority_violation": authority_violation,
        "critical_authority_violation": critical_authority_violation,
        "injected_mutation": mutation is not None,
        "quarantined": quarantined, "closure_state": closure_state,
        "unsatisfied_gate_ids": sorted(name for name, passed in gates.items() if not passed),
        "divergence_flags": sorted(set(divergence_flags)),
    }
    outcome["digest"] = digest(outcome)
    return outcome


def _baseline_result(corpus: FrozenReplayCorpusV1, baseline_id: str) -> BaselineResultV1:
    outcomes = [_case_outcome(case, baseline_id) for case in corpus.cases]
    canonical_outcomes = canonical_json(outcomes)
    result_digest = digest({"baseline_id": baseline_id, "corpus_digest": corpus.corpus_digest, "cases": outcomes})
    return BaselineResultV1(baseline_id, corpus.corpus_digest, result_digest, canonical_outcomes)


def replay_mutations(cases: Sequence[Mapping[str, Any]] | FrozenReplayCorpusV1) -> dict[str, Any]:
    """Evaluate a frozen held-out corpus across B0--B3 without promoting anything."""
    corpus = cases if isinstance(cases, FrozenReplayCorpusV1) else freeze_replay_corpus(cases)
    results = {baseline: _baseline_result(corpus, baseline) for baseline in BASELINE_DEFINITIONS}
    baseline_data = {baseline: result.to_dict() for baseline, result in results.items()}
    b3_cases = baseline_data["B3"]["cases"]
    mutations = sorted({item["mutation"] for item in b3_cases if item["mutation"]})
    critical_mutations = [item for item in b3_cases if item["mutation"] in _REQUIRED_MUTATIONS]
    critical_authority = any(
        case["critical_authority_violation"] and not case["injected_mutation"]
        for result in baseline_data.values() for case in result["cases"]
    )
    # Injected mutations are expected hostile fixtures. Their per-case state is
    # quarantined, but successful containment must not masquerade as a real
    # candidate authority violation.
    quarantined = critical_authority or any(
        case["quarantined"] and not case["injected_mutation"]
        for result in baseline_data.values() for case in result["cases"]
    )
    return {
        "corpus_version": corpus.corpus_version,
        "corpus_digest": corpus.corpus_digest,
        "cases": b3_cases,
        "baseline_results": baseline_data,
        "critical_mutations_blocked": bool(critical_mutations) and all(not item["closed"] for item in critical_mutations),
        "mutation_coverage": mutations,
        "quarantined": quarantined,
        "promotion_state": "QUARANTINED" if quarantined else "NOT_PROMOTED",
    }


def closure_ui_projection(closure: ImmutableArtifact | Mapping[str, Any]) -> dict[str, Any]:
    """Derived read model only; callers must not treat it as closure authority."""
    value = closure.to_dict() if isinstance(closure, ImmutableArtifact) else dict(closure)
    return {
        "projection_kind": "closure_ui_v1", "authoritative": False,
        "mission_ref": value.get("mission_ref"), "state": value.get("state"),
        "source_event_refs": list(value.get("source_event_refs", ())),
        "divergence_flags": list(value.get("divergence_flags", ())),
        "closure_projection_digest": value.get("projection_digest"),
    }


def evaluation_ui_projection(result: Mapping[str, Any], *, source_refs: Sequence[str] = ()) -> dict[str, Any]:
    """Derived evaluation display data; it cannot create truth or alter promotion."""
    for ref in source_refs:
        _check_ref(ref, "source_refs")
    baselines = result.get("baseline_results", {})
    if not isinstance(baselines, Mapping):
        raise ContractError("INVALID_EVALUATION_RESULT")
    summary = {
        baseline: {
            key: value.get(key) for key in (
                "verified_closure_count", "false_closure_count", "authority_violation_count", "quarantined",
            )
        }
        for baseline, value in sorted(baselines.items()) if isinstance(value, Mapping)
    }
    return {
        "projection_kind": "evaluation_ui_v1", "authoritative": False,
        "corpus_version": result.get("corpus_version"), "corpus_digest": result.get("corpus_digest"),
        "promotion_state": result.get("promotion_state"), "quarantined": bool(result.get("quarantined")),
        "source_refs": sorted(set(source_refs)), "baseline_summary": summary,
    }
