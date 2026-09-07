"""Small, JSON-safe contracts for evidence-first worker collaboration.

The contracts are deliberately passive.  They validate worker-produced
metadata, but do not grant authority, dispatch work, or decide whether a
task is complete.
"""

from __future__ import annotations

import dataclasses
import math
import re
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
_CAPABILITY_ORDER = {
    status: index
    for index, status in enumerate(("proposed", "tested", "reviewed", "active"))
}
_CONSENSUS_STATUSES = {"pending", "partial", "needs_review", "accepted", "rejected"}
_VERBOSITIES = {"concise", "normal", "detailed"}
_DIRECTNESS = {"low", "normal", "high"}
_MEMORY_RETENTIONS = {"ephemeral", "session", "bounded"}
_MEMORY_SENSITIVITIES = {"public", "internal", "sensitive", "restricted"}
_EXECUTION_LANES = {"simulation", "canary", "production"}
_DEGRADATION_STATES = {"full", "degraded", "advisory", "unavailable"}
_LINEAGE_STATES = {"active", "quarantined", "retired"}


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
        normalized = tuple(
            _require_text(f"{name}[{index}]", value)
            for index, value in enumerate(values)
        )
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
            raise ContractValidationError(
                "active capability promotion requires reviewed status"
            )
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
                raise ContractValidationError(
                    f"worker_reports[{index}] must be an object"
                )
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
            _validate_choice(
                f"required_evidence[{index}]", evidence_class, _EVIDENCE_CLASSES
            )
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


@dataclass(frozen=True)
class MemoryPolicy:
    """Purpose-bound, expiring memory rules for a worker."""

    purpose: str
    retention: str = "session"
    retention_seconds: int = 3_600
    sensitivity: str = "internal"
    allow_cross_session: bool = False
    allow_export: bool = False
    redactions: tuple[str, ...] = ()

    def validate(self) -> "MemoryPolicy":
        _require_text("purpose", self.purpose)
        _validate_choice("retention", self.retention, _MEMORY_RETENTIONS)
        if isinstance(self.retention_seconds, bool) or not isinstance(
            self.retention_seconds, int
        ):
            raise ContractValidationError("retention_seconds must be an integer")
        if self.retention_seconds < 1:
            raise ContractValidationError(
                "memory retention must be bounded and positive"
            )
        _validate_choice("sensitivity", self.sensitivity, _MEMORY_SENSITIVITIES)
        _validate_bool("allow_cross_session", self.allow_cross_session)
        _validate_bool("allow_export", self.allow_export)
        _validate_texts("redactions", self.redactions)
        if self.retention == "ephemeral" and self.allow_cross_session:
            raise ContractValidationError("ephemeral memory cannot cross sessions")
        if (
            self.sensitivity in {"sensitive", "restricted"}
            and self.allow_export
            and not self.redactions
        ):
            raise ContractValidationError(
                "sensitive memory export requires explicit redaction"
            )
        return self

    def is_retained(self, created_at: datetime, at: datetime | None = None) -> bool:
        self.validate()
        if created_at.tzinfo is None or created_at.utcoffset() is None:
            raise ContractValidationError("created_at must include a timezone")
        instant = at or datetime.now(timezone.utc)
        if instant.tzinfo is None or instant.utcoffset() is None:
            raise ContractValidationError("at must include a timezone")
        age = (
            instant.astimezone(timezone.utc) - created_at.astimezone(timezone.utc)
        ).total_seconds()
        return 0 <= age <= self.retention_seconds

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "purpose": self.purpose,
            "retention": self.retention,
            "retention_seconds": self.retention_seconds,
            "sensitivity": self.sensitivity,
            "allow_cross_session": self.allow_cross_session,
            "allow_export": self.allow_export,
            "redactions": list(self.redactions),
        }


@dataclass(frozen=True)
class TrustVector:
    """Dimensioned trust evidence; intentionally has no aggregate score."""

    dimensions: Mapping[str, float]
    updated_at: str
    decay_half_life_seconds: int = 86_400

    def validate(self) -> "TrustVector":
        if not isinstance(self.dimensions, Mapping) or not self.dimensions:
            raise ContractValidationError("trust dimensions must not be empty")
        for name, value in self.dimensions.items():
            _require_text("trust dimension", name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ContractValidationError(
                    "trust dimension values must be finite numbers"
                )
            if not 0 <= value <= 1:
                raise ContractValidationError(
                    "trust dimension values must be between 0 and 1"
                )
        _parse_timestamp("updated_at", self.updated_at)
        if isinstance(self.decay_half_life_seconds, bool) or not isinstance(
            self.decay_half_life_seconds, int
        ):
            raise ContractValidationError("decay_half_life_seconds must be an integer")
        if self.decay_half_life_seconds < 1:
            raise ContractValidationError("decay_half_life_seconds must be positive")
        return self

    def decayed(self, at: datetime | None = None) -> "TrustVector":
        self.validate()
        instant = at or datetime.now(timezone.utc)
        if instant.tzinfo is None or instant.utcoffset() is None:
            raise ContractValidationError("at must include a timezone")
        age = max(
            0.0,
            (
                instant.astimezone(timezone.utc)
                - _parse_timestamp("updated_at", self.updated_at)
            ).total_seconds(),
        )
        factor = 0.5 ** (age / self.decay_half_life_seconds)
        return replace(
            self,
            dimensions={
                name: value * factor for name, value in self.dimensions.items()
            },
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "dimensions": dict(self.dimensions),
            "updated_at": self.updated_at,
            "decay_half_life_seconds": self.decay_half_life_seconds,
        }


@dataclass(frozen=True)
class WorkerLineage:
    """Explicit parentage and lifecycle for every worker identity."""

    worker_id: str
    fork_reason: str
    source_sha: str
    created_at: str
    parent_worker_id: str | None = None
    root_worker_id: str | None = None
    status: str = "active"

    def validate(self) -> "WorkerLineage":
        worker_id = _require_text("worker_id", self.worker_id)
        _require_text("fork_reason", self.fork_reason)
        _require_text("source_sha", self.source_sha)
        _parse_timestamp("created_at", self.created_at)
        _validate_choice("status", self.status, _LINEAGE_STATES)
        if self.parent_worker_id is not None:
            _require_text("parent_worker_id", self.parent_worker_id)
            if self.parent_worker_id == worker_id:
                raise ContractValidationError("worker cannot be its own parent")
        if self.root_worker_id is not None:
            _require_text("root_worker_id", self.root_worker_id)
        return self

    def quarantine(self, reason: str) -> "WorkerLineage":
        _require_text("reason", reason)
        self.validate()
        return replace(
            self,
            status="quarantined",
            fork_reason=f"{self.fork_reason}; quarantine: {reason}",
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "worker_id": self.worker_id,
            "parent_worker_id": self.parent_worker_id,
            "root_worker_id": self.root_worker_id,
            "fork_reason": self.fork_reason,
            "source_sha": self.source_sha,
            "created_at": self.created_at,
            "status": self.status,
        }


@dataclass(frozen=True)
class ExecutionEnvelope:
    """Named execution lane with explicit side-effect and approval gates."""

    lane: str = "simulation"
    side_effects_allowed: bool = False
    approval_ref: str | None = None

    def validate(self) -> "ExecutionEnvelope":
        _validate_choice("lane", self.lane, _EXECUTION_LANES)
        _validate_bool("side_effects_allowed", self.side_effects_allowed)
        if self.lane in {"simulation", "canary"} and self.side_effects_allowed:
            raise ContractValidationError(
                f"{self.lane} lane cannot permit side effects"
            )
        if self.side_effects_allowed:
            _require_text("approval_ref", self.approval_ref)
        return self

    def allows_side_effects(self) -> bool:
        return self.validate().side_effects_allowed

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "lane": self.lane,
            "side_effects_allowed": self.side_effects_allowed,
            "approval_ref": self.approval_ref,
        }


@dataclass(frozen=True)
class CapabilityDegradation:
    """Visible capability state; degradation always carries a reason."""

    state: str = "full"
    reason: str | None = None
    changed_at: str = ""

    def validate(self) -> "CapabilityDegradation":
        _validate_choice("state", self.state, _DEGRADATION_STATES)
        if self.state != "full":
            _require_text("reason", self.reason)
        _parse_timestamp("changed_at", self.changed_at)
        return self

    def transition(
        self, state: str, *, reason: str, changed_at: str
    ) -> "CapabilityDegradation":
        self.validate()
        return replace(
            self, state=state, reason=reason, changed_at=changed_at
        ).validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "state": self.state,
            "reason": self.reason,
            "changed_at": self.changed_at,
        }


@dataclass(frozen=True)
class RoleSeparation:
    """Planner, critic, and executor identities with a review gate."""

    planner_id: str
    critic_id: str | None = None
    executor_id: str | None = None
    critic_accepted: bool = False
    execution_approved: bool = False

    def validate(self) -> "RoleSeparation":
        planner = _require_text("planner_id", self.planner_id)
        ids = [planner]
        for name, value in (
            ("critic_id", self.critic_id),
            ("executor_id", self.executor_id),
        ):
            if value is not None:
                normalized = _require_text(name, value)
                if normalized in ids:
                    raise ContractValidationError(
                        "planner, critic, and executor must be distinct"
                    )
                ids.append(normalized)
        _validate_bool("critic_accepted", self.critic_accepted)
        _validate_bool("execution_approved", self.execution_approved)
        if self.execution_approved and (
            self.critic_id is None or not self.critic_accepted
        ):
            raise ContractValidationError(
                "execution approval requires an accepted critic"
            )
        return self

    def can_execute(self, envelope: ExecutionEnvelope) -> bool:
        self.validate()
        envelope.validate()
        return bool(
            self.executor_id and self.critic_accepted and self.execution_approved
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "planner_id": self.planner_id,
            "critic_id": self.critic_id,
            "executor_id": self.executor_id,
            "critic_accepted": self.critic_accepted,
            "execution_approved": self.execution_approved,
        }


@dataclass(frozen=True)
class ContextProfile:
    """Context contract that surfaces missing assumptions as warnings."""

    profile: str
    assumptions: tuple[str, ...] = ()
    required_context: tuple[str, ...] = ()
    assumption_warnings: tuple[str, ...] = ()

    def validate(self) -> "ContextProfile":
        _require_text("profile", self.profile)
        _validate_texts("assumptions", self.assumptions)
        _validate_texts("required_context", self.required_context)
        _validate_texts("assumption_warnings", self.assumption_warnings)
        if self.assumptions and len(self.assumption_warnings) < len(self.assumptions):
            raise ContractValidationError(
                "each assumption requires an explicit warning"
            )
        return self

    def warnings_for(self, provided_context: str | None) -> tuple[str, ...]:
        self.validate()
        context = provided_context or ""
        missing = tuple(
            f"missing required context: {item}"
            for item in self.required_context
            if item not in context
        )
        return tuple(self.assumption_warnings) + missing

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "profile": self.profile,
            "assumptions": list(self.assumptions),
            "required_context": list(self.required_context),
            "assumption_warnings": list(self.assumption_warnings),
        }


@dataclass(frozen=True)
class PrivacyPolicy:
    """Disclosure policy for memory, reports, and external destinations."""

    allowed_scopes: tuple[str, ...] = ()
    denied_scopes: tuple[str, ...] = ()
    export_consent: bool = False

    def validate(self) -> "PrivacyPolicy":
        allowed = set(_validate_texts("allowed_scopes", self.allowed_scopes))
        denied = set(_validate_texts("denied_scopes", self.denied_scopes))
        _validate_bool("export_consent", self.export_consent)
        if allowed & denied:
            raise ContractValidationError(
                "privacy scope cannot be both allowed and denied"
            )
        return self

    def can_disclose(self, scope: str, *, export: bool = False) -> bool:
        self.validate()
        _require_text("scope", scope)
        if scope in self.denied_scopes or scope not in self.allowed_scopes:
            return False
        return not export or self.export_consent

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "allowed_scopes": list(self.allowed_scopes),
            "denied_scopes": list(self.denied_scopes),
            "export_consent": self.export_consent,
        }


@dataclass(frozen=True)
class HostileInputAssessment:
    """Conservative assessment of untrusted content; quarantine is explicit."""

    source: str
    indicators: tuple[str, ...] = ()
    quarantined: bool = False

    def validate(self) -> "HostileInputAssessment":
        _require_text("source", self.source)
        _validate_texts("indicators", self.indicators)
        _validate_bool("quarantined", self.quarantined)
        if self.quarantined and not self.indicators:
            raise ContractValidationError("quarantined input requires indicators")
        return self

    @classmethod
    def assess(cls, source: str, content: str) -> "HostileInputAssessment":
        _require_text("source", source)
        if not isinstance(content, str):
            raise ContractValidationError("content must be a string")
        patterns = {
            "instruction_override": r"ignore\s+(all\s+)?(previous|prior|earlier)\s+instructions",
            "secret_exfiltration": r"(reveal|print|send|upload).{0,40}(secret|token|credential|password)",
            "authority_spoofing": r"(system|admin|operator)\s+(message|override|approval)",
        }
        indicators = tuple(
            name
            for name, pattern in patterns.items()
            if re.search(pattern, content, re.IGNORECASE | re.DOTALL)
        )
        return cls(source, indicators, bool(indicators)).validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "source": self.source,
            "indicators": list(self.indicators),
            "quarantined": self.quarantined,
        }


@dataclass(frozen=True)
class ScenarioEnsemble:
    """Probabilistic alternatives with uncertainty kept visible."""

    scenarios: tuple[Mapping[str, Any], ...]
    confidence: str = "unknown"
    unknowns: tuple[str, ...] = ()

    def validate(self) -> "ScenarioEnsemble":
        if not self.scenarios:
            raise ContractValidationError("scenarios must not be empty")
        total = 0.0
        for index, scenario in enumerate(self.scenarios):
            if not isinstance(scenario, Mapping):
                raise ContractValidationError(f"scenarios[{index}] must be an object")
            _require_text(f"scenarios[{index}].id", scenario.get("id"))
            probability = scenario.get("probability")
            if (
                isinstance(probability, bool)
                or not isinstance(probability, (int, float))
                or not math.isfinite(probability)
                or not 0 <= probability <= 1
            ):
                raise ContractValidationError(
                    "scenario probability must be between 0 and 1"
                )
            total += probability
        if abs(total - 1.0) > 1e-6:
            raise ContractValidationError("scenario probabilities must sum to 1")
        _validate_choice("confidence", self.confidence, _CONFIDENCES)
        _validate_texts("unknowns", self.unknowns)
        return self

    def requires_review(self) -> bool:
        self.validate()
        return self.confidence in {"unknown", "low"} or bool(self.unknowns)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "scenarios": [dict(item) for item in self.scenarios],
            "confidence": self.confidence,
            "unknowns": list(self.unknowns),
        }


@dataclass(frozen=True)
class WorkforceGovernance:
    """Single policy envelope combining the reusable workforce controls."""

    memory: MemoryPolicy | None = None
    trust: TrustVector | None = None
    lineage: WorkerLineage | None = None
    execution: ExecutionEnvelope | None = None
    degradation: CapabilityDegradation | None = None
    roles: RoleSeparation | None = None
    context: ContextProfile | None = None
    privacy: PrivacyPolicy | None = None
    hostile_input: HostileInputAssessment | None = None
    scenarios: ScenarioEnsemble | None = None

    def validate(self) -> "WorkforceGovernance":
        for field in dataclasses.fields(self):
            policy = getattr(self, field.name)
            if policy is not None:
                policy.validate()
        if self.execution is not None and self.roles is not None:
            if self.execution.lane == "production" and not self.roles.can_execute(
                self.execution
            ):
                raise ContractValidationError(
                    "production execution requires critic acceptance and approval"
                )
        return self

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "memory": self.memory.to_dict() if self.memory else None,
            "trust": self.trust.to_dict() if self.trust else None,
            "lineage": self.lineage.to_dict() if self.lineage else None,
            "execution": self.execution.to_dict() if self.execution else None,
            "degradation": self.degradation.to_dict() if self.degradation else None,
            "roles": self.roles.to_dict() if self.roles else None,
            "context": self.context.to_dict() if self.context else None,
            "privacy": self.privacy.to_dict() if self.privacy else None,
            "hostile_input": self.hostile_input.to_dict()
            if self.hostile_input
            else None,
            "scenarios": self.scenarios.to_dict() if self.scenarios else None,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WorkforceGovernance":
        """Restore a serialized policy envelope without accepting extra fields."""

        if not isinstance(value, Mapping):
            raise ContractValidationError("workforce governance must be an object")
        allowed = {
            "memory",
            "trust",
            "lineage",
            "execution",
            "degradation",
            "roles",
            "context",
            "privacy",
            "hostile_input",
            "scenarios",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ContractValidationError(
                f"unknown workforce governance field(s): {', '.join(sorted(unknown))}"
            )

        def nested(name: str) -> Mapping[str, Any] | None:
            item = value.get(name)
            if item is None:
                return None
            if not isinstance(item, Mapping):
                raise ContractValidationError(f"{name} must be an object")
            return item

        memory = nested("memory")
        trust = nested("trust")
        lineage = nested("lineage")
        execution = nested("execution")
        degradation = nested("degradation")
        roles = nested("roles")
        context = nested("context")
        privacy = nested("privacy")
        hostile = nested("hostile_input")
        scenarios = nested("scenarios")
        result = cls(
            memory=(
                MemoryPolicy(**{
                    **memory,
                    "redactions": tuple(memory.get("redactions", ())),
                })
                if memory
                else None
            ),
            trust=TrustVector(**trust) if trust else None,
            lineage=WorkerLineage(**lineage) if lineage else None,
            execution=ExecutionEnvelope(**execution) if execution else None,
            degradation=CapabilityDegradation(**degradation) if degradation else None,
            roles=RoleSeparation(**roles) if roles else None,
            context=(
                ContextProfile(**{
                    **context,
                    "assumptions": tuple(context.get("assumptions", ())),
                    "required_context": tuple(context.get("required_context", ())),
                    "assumption_warnings": tuple(
                        context.get("assumption_warnings", ())
                    ),
                })
                if context
                else None
            ),
            privacy=(
                PrivacyPolicy(**{
                    **privacy,
                    "allowed_scopes": tuple(privacy.get("allowed_scopes", ())),
                    "denied_scopes": tuple(privacy.get("denied_scopes", ())),
                })
                if privacy
                else None
            ),
            hostile_input=(
                HostileInputAssessment(**{
                    **hostile,
                    "indicators": tuple(hostile.get("indicators", ())),
                })
                if hostile
                else None
            ),
            scenarios=(
                ScenarioEnsemble(**{
                    **scenarios,
                    "scenarios": tuple(scenarios.get("scenarios", ())),
                    "unknowns": tuple(scenarios.get("unknowns", ())),
                })
                if scenarios
                else None
            ),
        )
        return result.validate()


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
    "consensus": {
        "kind",
        "worker_reports",
        "agreement",
        "dissent",
        "status",
        "quorum",
    },
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
    "memory_policy": {
        "kind",
        "purpose",
        "retention",
        "retention_seconds",
        "sensitivity",
        "allow_cross_session",
        "allow_export",
        "redactions",
    },
    "trust_vector": {"kind", "dimensions", "updated_at", "decay_half_life_seconds"},
    "worker_lineage": {
        "kind",
        "worker_id",
        "parent_worker_id",
        "root_worker_id",
        "fork_reason",
        "source_sha",
        "created_at",
        "status",
    },
    "execution_envelope": {"kind", "lane", "side_effects_allowed", "approval_ref"},
    "capability_degradation": {"kind", "state", "reason", "changed_at"},
    "role_separation": {
        "kind",
        "planner_id",
        "critic_id",
        "executor_id",
        "critic_accepted",
        "execution_approved",
    },
    "context_profile": {
        "kind",
        "profile",
        "assumptions",
        "required_context",
        "assumption_warnings",
    },
    "privacy_policy": {"kind", "allowed_scopes", "denied_scopes", "export_consent"},
    "hostile_input_assessment": {"kind", "source", "indicators", "quarantined"},
    "scenario_ensemble": {"kind", "scenarios", "confidence", "unknowns"},
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
