"""Small, JSON-safe contracts for evidence-first worker collaboration.

The contracts are deliberately passive.  They validate worker-produced
metadata, but do not grant authority, dispatch work, or decide whether a
task is complete.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
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
_CONSENSUS_STATUSES = {"pending", "partial", "needs_review", "accepted", "rejected"}
_VERBOSITIES = {"concise", "normal", "detailed"}
_DIRECTNESS = {"low", "normal", "high"}
_OUTCOMES = {"completed", "partial", "blocked", "failed"}


def _require_text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{name} must be a non-empty string")
    return value.strip()


def _validate_texts(name: str, values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if not isinstance(values, (list, tuple)):
        raise ContractValidationError(f"{name} must be a sequence of strings")
    return tuple(_require_text(f"{name}[{index}]", value) for index, value in enumerate(values))


def _validate_choice(name: str, value: Any, choices: set[str]) -> str:
    value = _require_text(name, value)
    if value not in choices:
        allowed = ", ".join(sorted(choices))
        raise ContractValidationError(f"{name} must be one of: {allowed}")
    return value


def _validate_timestamp(name: str, value: Any) -> str:
    timestamp = _require_text(name, value)
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError as exc:
        raise ContractValidationError(f"{name} must be a timezone-aware ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ContractValidationError(f"{name} must be a timezone-aware ISO-8601 timestamp")
    return timestamp


def _validate_json_value(name: str, value: Any) -> None:
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if math.isfinite(value):
            return
        raise ContractValidationError(f"{name} must contain only JSON-compatible values")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(f"{name}[{index}]", item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractValidationError(f"{name} must contain only JSON-compatible values")
            _validate_json_value(f"{name}.{key}", item)
        return
    raise ContractValidationError(f"{name} must contain only JSON-compatible values")


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
    freshness: str = ""
    reversible_next_actions: tuple[str, ...] = ()
    outcome: str = ""

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
        _validate_timestamp("freshness", self.freshness)
        next_actions = _validate_texts("reversible_next_actions", self.reversible_next_actions)
        if not next_actions:
            raise ContractValidationError("reversible_next_actions must not be empty")
        outcome = _validate_choice("outcome", self.outcome, _OUTCOMES)
        if outcome == "completed" and not observations:
            raise ContractValidationError("completed outcome requires source-backed observations")
        return self

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "kind": "evidence_packet",
            "observations": list(_validate_texts("observations", self.observations)),
            "sources": list(_validate_texts("sources", self.sources)),
            "hypotheses": list(_validate_texts("hypotheses", self.hypotheses)),
            "conclusions": list(_validate_texts("conclusions", self.conclusions)),
            "unknowns": list(_validate_texts("unknowns", self.unknowns)),
            "confidence": _validate_choice("confidence", self.confidence, _CONFIDENCES),
            "evidence_class": _validate_choice(
                "evidence_class", self.evidence_class, _EVIDENCE_CLASSES
            ),
            "artifacts": list(_validate_texts("artifacts", self.artifacts)),
            "limitations": list(_validate_texts("limitations", self.limitations)),
            "freshness": _validate_timestamp("freshness", self.freshness),
            "reversible_next_actions": list(
                _validate_texts("reversible_next_actions", self.reversible_next_actions)
            ),
            "outcome": _validate_choice("outcome", self.outcome, _OUTCOMES),
        }


@dataclass(frozen=True)
class ObjectiveStack:
    """Operator-visible mission and constraints for one worker invocation."""

    profile: str
    authority: str
    mission: str
    task_id: str = ""
    owner_id: str = ""
    repository: str = ""
    constraints: tuple[str, ...] = ()
    forbidden_actions: tuple[str, ...] = ()
    hidden_objectives: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()

    def validate(self) -> "ObjectiveStack":
        _require_text("profile", self.profile)
        _require_text("authority", self.authority)
        _require_text("mission", self.mission)
        _require_text("task_id", self.task_id)
        _require_text("owner_id", self.owner_id)
        _require_text("repository", self.repository)
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
            "kind": "objective_stack",
            "profile": _require_text("profile", self.profile),
            "authority": _require_text("authority", self.authority),
            "mission": _require_text("mission", self.mission),
            "task_id": _require_text("task_id", self.task_id),
            "owner_id": _require_text("owner_id", self.owner_id),
            "repository": _require_text("repository", self.repository),
            "constraints": list(_validate_texts("constraints", self.constraints)),
            "forbidden_actions": list(
                _validate_texts("forbidden_actions", self.forbidden_actions)
            ),
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

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "kind": "capability",
            "name": _require_text("name", self.name),
            "owner_profile": _require_text("owner_profile", self.owner_profile),
            "authority": _require_text("authority", self.authority),
            "evidence_class": _validate_choice(
                "evidence_class", self.evidence_class, _EVIDENCE_CLASSES
            ),
            "status": _validate_choice("status", self.status, _CAPABILITY_STATUSES),
            "tested_at": self.tested_at,
            "source_sha": self.source_sha,
            "limitations": list(_validate_texts("limitations", self.limitations)),
        }


@dataclass(frozen=True)
class ConsensusRecord:
    """Independent worker reports with disagreement retained explicitly."""

    worker_reports: tuple[Mapping[str, Any], ...]
    agreement: tuple[str, ...] = ()
    dissent: tuple[str, ...] = ()
    status: str = "pending"

    def validate(self) -> "ConsensusRecord":
        if not isinstance(self.worker_reports, (list, tuple)):
            raise ContractValidationError("worker_reports must be a stable sequence of objects")
        if not self.worker_reports:
            raise ContractValidationError("worker_reports must not be empty")
        workers: set[str] = set()
        for index, report in enumerate(self.worker_reports):
            if not isinstance(report, Mapping):
                raise ContractValidationError(f"worker_reports[{index}] must be an object")
            _validate_json_value(f"worker_reports[{index}]", report)
            worker = _require_text(f"worker_reports[{index}].worker", report.get("worker"))
            if worker in workers:
                raise ContractValidationError("worker_reports must contain independent workers; duplicate worker")
            workers.add(worker)
        _validate_texts("agreement", self.agreement)
        _validate_texts("dissent", self.dissent)
        status = _validate_choice("status", self.status, _CONSENSUS_STATUSES)
        if status == "accepted" and len(workers) < 2:
            raise ContractValidationError("accepted consensus requires independent worker reports")
        return self

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "kind": "consensus",
            "worker_reports": [dict(report) for report in self.worker_reports],
            "agreement": list(_validate_texts("agreement", self.agreement)),
            "dissent": list(_validate_texts("dissent", self.dissent)),
            "status": _validate_choice("status", self.status, _CONSENSUS_STATUSES),
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
            "kind": "worker_mode",
            "name": _require_text("name", self.name),
            "verbosity": _validate_choice("verbosity", self.verbosity, _VERBOSITIES),
            "directness": _validate_choice("directness", self.directness, _DIRECTNESS),
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
        "freshness",
        "reversible_next_actions",
        "outcome",
    },
    "objective_stack": {
        "kind",
        "profile",
        "authority",
        "mission",
        "task_id",
        "owner_id",
        "repository",
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
    """Validate a complete serialized contract before it is interpreted."""

    if not isinstance(value, Mapping):
        raise ContractValidationError("contract must be an object")
    kind = value.get("kind")
    if not isinstance(kind, str) or kind not in _CONTRACT_FIELDS:
        raise ContractValidationError("kind must identify a known contract")
    unknown = set(value) - _CONTRACT_FIELDS[kind]
    if unknown:
        field_names = ", ".join(sorted(str(field) for field in unknown))
        raise ContractValidationError(f"unknown field(s): {field_names}")
    missing = _CONTRACT_FIELDS[kind] - set(value)
    if missing:
        field_names = ", ".join(sorted(missing))
        raise ContractValidationError(f"missing field(s): {field_names}")
    payload = dict(value)
    payload.pop("kind")
    constructors = {
        "evidence_packet": EvidencePacket,
        "objective_stack": ObjectiveStack,
        "capability": CapabilityRecord,
        "consensus": ConsensusRecord,
        "worker_mode": WorkerMode,
    }
    try:
        constructors[kind](**payload).validate()
    except ContractValidationError:
        raise
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(f"invalid {kind} payload: {exc}") from exc
    return value
