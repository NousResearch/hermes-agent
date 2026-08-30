"""Small, JSON-safe contracts for evidence-first worker collaboration.

The contracts are deliberately passive.  They validate worker-produced
metadata, but do not grant authority, dispatch work, or decide whether a
task is complete.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Mapping


class ContractValidationError(ValueError):
    """Raised when a worker contract is incomplete or internally unsafe."""


_CONFIDENCES = {"unknown", "low", "medium", "high"}
_EVIDENCE_CLASSES = {
    "unknown",
    "observation",
    "research",
    "diagnostic",
    "targeted",
    "governed",
    "acceptance",
}
_CAPABILITY_STATUSES = {"proposed", "tested", "reviewed", "active"}
_CAPABILITY_ORDER = {status: index for index, status in enumerate(("proposed", "tested", "reviewed", "active"))}
_CONSENSUS_STATUSES = {"pending", "partial", "needs_review", "accepted", "rejected"}
_VERBOSITIES = {"concise", "normal", "detailed"}
_DIRECTNESS = {"low", "normal", "high"}


def _require_text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{name} must be a non-empty string")
    return value.strip()


def _validate_texts(name: str, values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)):
        raise ContractValidationError(f"{name} must be a sequence of strings")
    try:
        normalized = tuple(_require_text(f"{name}[{index}]", value) for index, value in enumerate(values))
    except TypeError as exc:
        raise ContractValidationError(f"{name} must be a sequence of strings") from exc
    return normalized


def _validate_choice(name: str, value: Any, choices: set[str]) -> str:
    value = _require_text(name, value)
    if value not in choices:
        allowed = ", ".join(sorted(choices))
        raise ContractValidationError(f"{name} must be one of: {allowed}")
    return value


def _parse_timestamp(name: str, value: Any) -> datetime:
    value = _require_text(name, value)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractValidationError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ContractValidationError(f"{name} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _validate_bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ContractValidationError(f"{name} must be boolean")
    return value


def _validate_positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ContractValidationError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class EvidencePacket:
    """Evidence with observations kept separate from interpretation."""

    observations: tuple[str, ...]
    sources: tuple[str, ...]
    hypotheses: tuple[str, ...] = ()
    conclusions: tuple[str, ...] = ()
    unknowns: tuple[str, ...] = ()
    confidence: str = "unknown"
    evidence_class: str = "unknown"
    artifacts: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    def validate(self) -> "EvidencePacket":
        observations = _validate_texts("observations", self.observations)
        sources = _validate_texts("sources", self.sources)
        _validate_texts("hypotheses", self.hypotheses)
        conclusions = _validate_texts("conclusions", self.conclusions)
        _validate_texts("unknowns", self.unknowns)
        _validate_texts("artifacts", self.artifacts)
        _validate_texts("limitations", self.limitations)
        _validate_choice("confidence", self.confidence, _CONFIDENCES)
        _validate_choice("evidence_class", self.evidence_class, _EVIDENCE_CLASSES)

        if conclusions and not observations:
            raise ContractValidationError("conclusions require observations")
        if observations and not sources:
            raise ContractValidationError("observations require sources")
        if conclusions and not sources:
            raise ContractValidationError("conclusions require sources")
        return self

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "observations": list(self.observations),
            "sources": list(self.sources),
            "hypotheses": list(self.hypotheses),
            "conclusions": list(self.conclusions),
            "unknowns": list(self.unknowns),
            "confidence": self.confidence,
            "evidence_class": self.evidence_class,
            "artifacts": list(self.artifacts),
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True)
class ObjectiveStack:
    """Operator-visible mission and constraints for one worker invocation."""

    profile: str
    authority: str
    mission: str
    constraints: tuple[str, ...] = ()
    forbidden_actions: tuple[str, ...] = ()
    hidden_objectives: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()

    def validate(self) -> "ObjectiveStack":
        _require_text("profile", self.profile)
        _require_text("authority", self.authority)
        _require_text("mission", self.mission)
        _validate_texts("constraints", self.constraints)
        _validate_texts("forbidden_actions", self.forbidden_actions)
        hidden = _validate_texts("hidden_objectives", self.hidden_objectives)
        conflicts = _validate_texts("conflicts", self.conflicts)
        if hidden:
            raise ContractValidationError("hidden objectives are forbidden")
        if conflicts:
            raise ContractValidationError("objective conflict requires review")
        return self

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "profile": self.profile,
            "authority": self.authority,
            "mission": self.mission,
            "constraints": list(self.constraints),
            "forbidden_actions": list(self.forbidden_actions),
            "hidden_objectives": [],
            "conflicts": [],
        }


@dataclass(frozen=True)
class CapabilityRecord:
    """A capability claim tied to a tested source and explicit limitations."""

    name: str
    owner_profile: str
    authority: str
    evidence_class: str = "unknown"
    status: str = "proposed"
    tested_at: str | None = None
    source_sha: str | None = None
    limitations: tuple[str, ...] = ()

    def validate(self) -> "CapabilityRecord":
        _require_text("name", self.name)
        _require_text("owner_profile", self.owner_profile)
        _require_text("authority", self.authority)
        _validate_choice("evidence_class", self.evidence_class, _EVIDENCE_CLASSES)
        status = _validate_choice("status", self.status, _CAPABILITY_STATUSES)
        _validate_texts("limitations", self.limitations)
        if status in {"tested", "reviewed", "active"}:
            _require_text("tested_at", self.tested_at)
            _require_text("source_sha", self.source_sha)
        return self

    def advance_to(
        self,
        status: str,
        *,
        tested_at: str | None = None,
        source_sha: str | None = None,
    ) -> "CapabilityRecord":
        """Advance one lifecycle stage while preserving tested provenance."""

        self.validate()
        target = _validate_choice("status", status, _CAPABILITY_STATUSES)
        current_index = _CAPABILITY_ORDER[self.status]
        target_index = _CAPABILITY_ORDER[target]
        if target_index <= current_index:
            raise ContractValidationError("capability lifecycle must move forward")
        if target_index != current_index + 1:
            raise ContractValidationError("active capability promotion requires reviewed status")
        promoted = replace(
            self,
            status=target,
            tested_at=tested_at if tested_at is not None else self.tested_at,
            source_sha=source_sha if source_sha is not None else self.source_sha,
        )
        return promoted.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "name": self.name,
            "owner_profile": self.owner_profile,
            "authority": self.authority,
            "evidence_class": self.evidence_class,
            "status": self.status,
            "tested_at": self.tested_at,
            "source_sha": self.source_sha,
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True)
class ConsensusRecord:
    """Independent worker reports with disagreement retained explicitly."""

    worker_reports: tuple[Mapping[str, Any], ...]
    agreement: tuple[str, ...] = ()
    dissent: tuple[str, ...] = ()
    status: str = "pending"
    quorum: int = 1

    def validate(self) -> "ConsensusRecord":
        if not self.worker_reports:
            raise ContractValidationError("worker_reports must not be empty")
        for index, report in enumerate(self.worker_reports):
            if not isinstance(report, Mapping):
                raise ContractValidationError(f"worker_reports[{index}] must be an object")
            _require_text(f"worker_reports[{index}].worker", report.get("worker"))
        _validate_texts("agreement", self.agreement)
        _validate_texts("dissent", self.dissent)
        _validate_choice("status", self.status, _CONSENSUS_STATUSES)
        quorum = _validate_positive_int("quorum", self.quorum)
        if quorum > len(self.worker_reports):
            raise ContractValidationError("quorum cannot exceed worker_reports")
        return self

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "worker_reports": [dict(report) for report in self.worker_reports],
            "agreement": list(self.agreement),
            "dissent": list(self.dissent),
            "status": self.status,
            "quorum": self.quorum,
        }


@dataclass(frozen=True)
class WorkerConstitution:
    """Immutable profile rules for values, authority, and escalation."""

    profile: str
    values: tuple[str, ...]
    authority: str
    forbidden_actions: tuple[str, ...] = ()
    required_evidence: tuple[str, ...] = ()
    escalation_path: str = "operator-review"

    def validate(self) -> "WorkerConstitution":
        _require_text("profile", self.profile)
        values = _validate_texts("values", self.values)
        _require_text("authority", self.authority)
        _validate_texts("forbidden_actions", self.forbidden_actions)
        required_evidence = _validate_texts("required_evidence", self.required_evidence)
        for index, evidence_class in enumerate(required_evidence):
            _validate_choice(f"required_evidence[{index}]", evidence_class, _EVIDENCE_CLASSES)
        _require_text("escalation_path", self.escalation_path)
        if not values:
            raise ContractValidationError("values must not be empty")
        return self

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "profile": self.profile,
            "values": list(self.values),
            "authority": self.authority,
            "forbidden_actions": list(self.forbidden_actions),
            "required_evidence": list(self.required_evidence),
            "escalation_path": self.escalation_path,
        }


@dataclass(frozen=True)
class JobContract:
    """A scoped, expiring assignment for one specialist worker."""

    name: str
    worker_profile: str
    job: str
    scope: tuple[str, ...]
    authority: str
    obligations: tuple[str, ...] = ()
    granted_at: str = ""
    expires_at: str = ""
    requires_review: bool = True

    def validate(self) -> "JobContract":
        _require_text("name", self.name)
        _require_text("worker_profile", self.worker_profile)
        _require_text("job", self.job)
        scope = _validate_texts("scope", self.scope)
        _require_text("authority", self.authority)
        _validate_texts("obligations", self.obligations)
        granted_at = _parse_timestamp("granted_at", self.granted_at)
        expires_at = _parse_timestamp("expires_at", self.expires_at)
        _validate_bool("requires_review", self.requires_review)
        if not scope:
            raise ContractValidationError("scope must not be empty")
        if expires_at <= granted_at:
            raise ContractValidationError("expires_at must be after granted_at")
        return self

    def is_active(self, at: datetime | None = None) -> bool:
        self.validate()
        instant = at or datetime.now(timezone.utc)
        if instant.tzinfo is None or instant.utcoffset() is None:
            raise ContractValidationError("at must include a timezone")
        granted_at = _parse_timestamp("granted_at", self.granted_at)
        expires_at = _parse_timestamp("expires_at", self.expires_at)
        instant = instant.astimezone(timezone.utc)
        return granted_at <= instant < expires_at

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "name": self.name,
            "worker_profile": self.worker_profile,
            "job": self.job,
            "scope": list(self.scope),
            "authority": self.authority,
            "obligations": list(self.obligations),
            "granted_at": self.granted_at,
            "expires_at": self.expires_at,
            "requires_review": self.requires_review,
        }


@dataclass(frozen=True)
class EmergencyAuthority:
    """A temporary coordination grant that must be revocable and expiring."""

    name: str
    granted_to: str
    issuer: str
    reason: str
    scope: tuple[str, ...]
    granted_at: str
    expires_at: str
    revocable: bool = True

    def validate(self) -> "EmergencyAuthority":
        _require_text("name", self.name)
        _require_text("granted_to", self.granted_to)
        _require_text("issuer", self.issuer)
        _require_text("reason", self.reason)
        scope = _validate_texts("scope", self.scope)
        granted_at = _parse_timestamp("granted_at", self.granted_at)
        expires_at = _parse_timestamp("expires_at", self.expires_at)
        revocable = _validate_bool("revocable", self.revocable)
        if not scope:
            raise ContractValidationError("scope must not be empty")
        if expires_at <= granted_at:
            raise ContractValidationError("expires_at must be after granted_at")
        if not revocable:
            raise ContractValidationError("emergency authority must be revocable")
        return self

    def is_active(self, at: datetime | None = None) -> bool:
        self.validate()
        instant = at or datetime.now(timezone.utc)
        if instant.tzinfo is None or instant.utcoffset() is None:
            raise ContractValidationError("at must include a timezone")
        granted_at = _parse_timestamp("granted_at", self.granted_at)
        expires_at = _parse_timestamp("expires_at", self.expires_at)
        instant = instant.astimezone(timezone.utc)
        return granted_at <= instant < expires_at

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "name": self.name,
            "granted_to": self.granted_to,
            "issuer": self.issuer,
            "reason": self.reason,
            "scope": list(self.scope),
            "granted_at": self.granted_at,
            "expires_at": self.expires_at,
            "revocable": True,
        }


@dataclass(frozen=True)
class WorkerMode:
    """Communication settings whose safety requirements cannot be disabled."""

    name: str
    verbosity: str = "normal"
    directness: str = "normal"
    requires_citations: bool = True
    requires_uncertainty: bool = True
    humor_enabled: bool = False

    def validate(self) -> "WorkerMode":
        _require_text("name", self.name)
        _validate_choice("verbosity", self.verbosity, _VERBOSITIES)
        _validate_choice("directness", self.directness, _DIRECTNESS)
        if self.requires_citations is not True:
            raise ContractValidationError("citations are mandatory")
        if self.requires_uncertainty is not True:
            raise ContractValidationError("uncertainty reporting is mandatory")
        if not isinstance(self.humor_enabled, bool):
            raise ContractValidationError("humor_enabled must be boolean")
        return self

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "name": self.name,
            "verbosity": self.verbosity,
            "directness": self.directness,
            "requires_citations": True,
            "requires_uncertainty": True,
            "humor_enabled": self.humor_enabled,
        }


_CONTRACT_FIELDS = {
    "evidence_packet": {
        "kind",
        "observations",
        "sources",
        "hypotheses",
        "conclusions",
        "unknowns",
        "confidence",
        "evidence_class",
        "artifacts",
        "limitations",
    },
    "objective_stack": {
        "kind",
        "profile",
        "authority",
        "mission",
        "constraints",
        "forbidden_actions",
        "hidden_objectives",
        "conflicts",
    },
    "capability": {
        "kind",
        "name",
        "owner_profile",
        "authority",
        "evidence_class",
        "status",
        "tested_at",
        "source_sha",
        "limitations",
    },
    "consensus": {"kind", "worker_reports", "agreement", "dissent", "status"},
    "worker_constitution": {
        "kind",
        "profile",
        "values",
        "authority",
        "forbidden_actions",
        "required_evidence",
        "escalation_path",
    },
    "job_contract": {
        "kind",
        "name",
        "worker_profile",
        "job",
        "scope",
        "authority",
        "obligations",
        "granted_at",
        "expires_at",
        "requires_review",
    },
    "emergency_authority": {
        "kind",
        "name",
        "granted_to",
        "issuer",
        "reason",
        "scope",
        "granted_at",
        "expires_at",
        "revocable",
    },
    "worker_mode": {
        "kind",
        "name",
        "verbosity",
        "directness",
        "requires_citations",
        "requires_uncertainty",
        "humor_enabled",
    },
}


def validate_contract_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Reject unknown fields before a serialized contract is interpreted."""

    if not isinstance(value, Mapping):
        raise ContractValidationError("contract must be an object")
    kind = value.get("kind")
    if not isinstance(kind, str) or kind not in _CONTRACT_FIELDS:
        raise ContractValidationError("kind must identify a known contract")
    unknown = set(value) - _CONTRACT_FIELDS[kind]
    if unknown:
        field_names = ", ".join(sorted(str(field) for field in unknown))
        raise ContractValidationError(f"unknown field(s): {field_names}")
    return value
