"""Deterministic authority gate for independent project checker verdicts.

This module is a pure value-state boundary.  Callers own durable task/run
storage and pass its current project generation plus integrated candidate
snapshot into this gate.  The gate never creates tasks, runs repairs, writes
final reports, or mutates a database; it only returns the next authoritative
checker-gate state and a machine-readable routing decision.

State owner: the caller persists :class:`CheckerGate` transactionally.
Inputs: project generation, candidate snapshot, implementation profiles,
checker identity, machine verdict, and caller-owned optimistic version.
Outputs: immutable gate state and deterministic decision.
Invariants: exactly zero or one active checker; only a current independent
checker can authorize a verdict; verdict evidence is append-only; and a
candidate snapshot change clears authority until a fresh checker is created.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

PASS = "PASS"
FAIL_REPAIRABLE_STATUS = "FAIL_REPAIRABLE"
BLOCKED_STATUS = "BLOCKED"
MACHINE_VERDICTS = frozenset((PASS, FAIL_REPAIRABLE_STATUS, BLOCKED_STATUS))

PASS_ACCEPTED = "PASS_ACCEPTED"
CHECKER_REQUIRED = "CHECKER_REQUIRED"
CHECKER_ALREADY_EXISTS = "CHECKER_ALREADY_EXISTS"
FAIL_REPAIRABLE = "FAIL_REPAIRABLE"
BLOCKED = "BLOCKED"
STALE_VERDICT = "STALE_VERDICT"
UNAUTHORIZED_CHECKER = "UNAUTHORIZED_CHECKER"
MALFORMED_VERDICT = "MALFORMED_VERDICT"
STALE_SNAPSHOT = "STALE_SNAPSHOT"
CHECKER_OF_CHECKER_DENIED = "CHECKER_OF_CHECKER_DENIED"
STALE_LOCK_VERSION = "STALE_LOCK_VERSION"

_VALID_OUTCOMES = frozenset(
    (
        PASS_ACCEPTED,
        CHECKER_REQUIRED,
        CHECKER_ALREADY_EXISTS,
        FAIL_REPAIRABLE,
        BLOCKED,
        STALE_VERDICT,
        UNAUTHORIZED_CHECKER,
        MALFORMED_VERDICT,
        STALE_SNAPSHOT,
        CHECKER_OF_CHECKER_DENIED,
        STALE_LOCK_VERSION,
    )
)


@dataclass(frozen=True)
class ProjectIdentity:
    """Stable project identity for one generation."""

    project_id: str
    generation: int


@dataclass(frozen=True)
class CandidateBinding:
    """Integrated candidate identity and the exact evaluated snapshot."""

    snapshot_version: str
    candidate_id: str


@dataclass(frozen=True)
class CheckerIdentity:
    """One checker task/run/profile identity."""

    task_id: str
    run_id: str
    profile: str


@dataclass(frozen=True)
class CheckerRequest:
    """Request one active checker; checker targets are always forbidden."""

    task_id: str
    run_id: str
    profile: str
    target_is_checker: bool = False

    def identity(self) -> CheckerIdentity:
        return CheckerIdentity(self.task_id, self.run_id, self.profile)


@dataclass(frozen=True)
class MachineVerdict:
    """The complete machine-readable proof submitted by the checker."""

    project: ProjectIdentity
    candidate: CandidateBinding
    checker: CheckerIdentity
    status: str


@dataclass(frozen=True)
class VerdictHistoryEntry:
    """Append-only non-secret checker evidence and rejection audit."""

    outcome: str
    verdict: MachineVerdict | None


@dataclass(frozen=True)
class CheckerGate:
    """Durable checker authority state owned and versioned by the caller."""

    project: ProjectIdentity
    candidate: CandidateBinding
    implementation_profiles: tuple[str, ...]
    active_checker: CheckerIdentity | None = None
    active_verdict: MachineVerdict | None = None
    history: tuple[VerdictHistoryEntry, ...] = ()
    version: int = 0


@dataclass(frozen=True)
class GateDecision:
    """Machine-readable routing output; no side effect is performed here."""

    outcome: str
    lifecycle: str
    reason: str


@dataclass(frozen=True)
class _ParsedVerdict:
    value: MachineVerdict | None
    error: str | None


def create_checker_gate(
    *,
    project: ProjectIdentity,
    candidate: CandidateBinding,
    implementation_profiles: tuple[str, ...] | list[str],
) -> CheckerGate:
    """Create an empty gate for exactly one project generation and snapshot."""

    _validate_project(project)
    _validate_candidate(candidate)
    profiles = _normalize_profiles(implementation_profiles)
    return CheckerGate(project=project, candidate=candidate, implementation_profiles=profiles)


def reconcile_checker(
    gate: CheckerGate,
    *,
    request: CheckerRequest,
    expected_version: int,
    candidate: CandidateBinding | None = None,
) -> tuple[CheckerGate, GateDecision]:
    """Create or reconcile the single active checker identity.

    The exact same candidate always returns the pre-existing active checker.
    A changed candidate snapshot invalidates active authority, preserves all
    evidence, and requires a different fresh checker task/run identity.
    """

    conflict = _version_conflict(gate, expected_version)
    if conflict is not None:
        return gate, conflict
    try:
        _validate_request(request)
        requested_candidate = candidate or gate.candidate
        _validate_candidate(requested_candidate)
    except (TypeError, ValueError) as error:
        return gate, _decision(MALFORMED_VERDICT, "WAITING", str(error))

    if request.target_is_checker:
        return gate, _decision(
            CHECKER_OF_CHECKER_DENIED,
            "WAITING",
            "checker authority cannot create or reconcile a checker for a checker",
        )

    identity = request.identity()
    if identity.profile in gate.implementation_profiles:
        return gate, _decision(
            UNAUTHORIZED_CHECKER,
            "WAITING",
            "checker profile must be independent from every implementation profile",
        )

    if requested_candidate == gate.candidate and gate.active_checker is not None:
        return gate, _decision(
            CHECKER_ALREADY_EXISTS,
            "WAITING",
            "the current candidate already has one active checker identity",
        )

    if requested_candidate != gate.candidate:
        stale_history = gate.history
        if gate.active_verdict is not None:
            stale_history += (VerdictHistoryEntry(STALE_VERDICT, gate.active_verdict),)
        reset = replace(
            gate,
            candidate=requested_candidate,
            active_checker=None,
            active_verdict=None,
            history=stale_history,
            version=gate.version + 1,
        )
        if gate.active_checker == identity:
            return reset, _decision(
                CHECKER_REQUIRED,
                "WAITING",
                "a changed candidate snapshot requires a fresh checker identity",
            )
        gate = reset

    created = replace(gate, active_checker=identity, active_verdict=None, version=gate.version + 1)
    return created, _decision(
        CHECKER_REQUIRED,
        "WAITING",
        "current candidate requires a machine-readable verdict from its active independent checker",
    )


def submit_checker_verdict(
    gate: CheckerGate,
    *,
    verdict: MachineVerdict | Mapping[str, Any] | object,
    expected_version: int,
) -> tuple[CheckerGate, GateDecision]:
    """Validate and record one checker verdict without overwriting history."""

    conflict = _version_conflict(gate, expected_version)
    if conflict is not None:
        return gate, conflict
    parsed = _parse_machine_verdict(verdict)
    if parsed.value is None:
        return gate, _decision(MALFORMED_VERDICT, "WAITING", parsed.error or "verdict is malformed")
    value = parsed.value

    if gate.active_checker is None:
        return _record_rejection(gate, STALE_VERDICT, value, "no active checker exists for the current candidate")
    if value.project != gate.project:
        return _record_rejection(gate, STALE_SNAPSHOT, value, "verdict project identity or generation is stale")
    if value.candidate != gate.candidate:
        return _record_rejection(gate, STALE_SNAPSHOT, value, "verdict snapshot or candidate identity is stale")
    if value.checker != gate.active_checker or value.checker.profile in gate.implementation_profiles:
        return _record_rejection(gate, UNAUTHORIZED_CHECKER, value, "verdict checker is not the authorized independent checker")

    if gate.active_verdict is not None:
        if gate.active_verdict == value:
            return gate, _accepted_decision(value.status)
        return gate, _decision(MALFORMED_VERDICT, "WAITING", "active checker verdict is immutable and cannot be replaced")

    outcome = _outcome_for_status(value.status)
    updated = replace(
        gate,
        active_verdict=value,
        history=gate.history + (VerdictHistoryEntry(outcome, value),),
        version=gate.version + 1,
    )
    return updated, _accepted_decision(value.status)


def evaluate_checker_gate(
    gate: CheckerGate,
    *,
    project: ProjectIdentity,
    candidate: CandidateBinding,
) -> GateDecision:
    """Return the current completion/repair/block route without mutating state."""

    try:
        _validate_project(project)
        _validate_candidate(candidate)
    except (TypeError, ValueError) as error:
        return _decision(MALFORMED_VERDICT, "WAITING", str(error))
    if project != gate.project or candidate != gate.candidate:
        return _decision(STALE_VERDICT, "WAITING", "gate authority is bound to a different project generation or candidate snapshot")
    if gate.active_checker is None or gate.active_verdict is None:
        return _decision(CHECKER_REQUIRED, "WAITING", "a current authorized checker PASS is required before completion")
    if gate.active_verdict.project != project or gate.active_verdict.candidate != candidate:
        return _decision(STALE_VERDICT, "WAITING", "stored checker verdict is not bound to the current snapshot")
    if gate.active_verdict.checker != gate.active_checker:
        return _decision(UNAUTHORIZED_CHECKER, "WAITING", "stored checker verdict does not match the active checker identity")
    if gate.active_verdict.checker.profile in gate.implementation_profiles:
        return _decision(UNAUTHORIZED_CHECKER, "WAITING", "stored checker profile is not independent")
    return _accepted_decision(gate.active_verdict.status)


def _record_rejection(
    gate: CheckerGate,
    outcome: str,
    verdict: MachineVerdict,
    reason: str,
) -> tuple[CheckerGate, GateDecision]:
    updated = replace(
        gate,
        history=gate.history + (VerdictHistoryEntry(outcome, verdict),),
        version=gate.version + 1,
    )
    return updated, _decision(outcome, "WAITING", reason)


def _accepted_decision(status: str) -> GateDecision:
    if status == PASS:
        return _decision(PASS_ACCEPTED, "COMPLETE_ELIGIBLE", "current authorized independent checker passed")
    if status == FAIL_REPAIRABLE_STATUS:
        return _decision(FAIL_REPAIRABLE, "REPAIR_POLICY", "current checker requested repair; no repair was performed")
    if status == BLOCKED_STATUS:
        return _decision(BLOCKED, "TERMINAL_MANUAL_BLOCKED", "current checker is blocked; manual action is required")
    return _decision(MALFORMED_VERDICT, "WAITING", "verdict status is outside the machine-readable vocabulary")


def _outcome_for_status(status: str) -> str:
    return _accepted_decision(status).outcome


def _decision(outcome: str, lifecycle: str, reason: str) -> GateDecision:
    if outcome not in _VALID_OUTCOMES:
        raise AssertionError(f"unsupported checker gate outcome: {outcome}")
    return GateDecision(outcome=outcome, lifecycle=lifecycle, reason=reason)


def _version_conflict(gate: CheckerGate, expected_version: int) -> GateDecision | None:
    if isinstance(expected_version, bool) or not isinstance(expected_version, int):
        return _decision(STALE_LOCK_VERSION, "WAITING", "expected lock version must be an integer")
    if expected_version != gate.version:
        return _decision(STALE_LOCK_VERSION, "WAITING", "checker gate version does not match the caller snapshot")
    return None


def _parse_machine_verdict(value: object) -> _ParsedVerdict:
    if isinstance(value, MachineVerdict):
        candidate = value
    elif isinstance(value, Mapping):
        expected = {
            "project_id",
            "generation",
            "snapshot_version",
            "candidate_id",
            "checker_task_id",
            "checker_run_id",
            "checker_profile",
            "status",
        }
        if set(value) != expected:
            return _ParsedVerdict(None, "machine-readable verdict fields are missing, extra, or ambiguous")
        candidate = MachineVerdict(
            project=ProjectIdentity(value["project_id"], value["generation"]),
            candidate=CandidateBinding(value["snapshot_version"], value["candidate_id"]),
            checker=CheckerIdentity(value["checker_task_id"], value["checker_run_id"], value["checker_profile"]),
            status=value["status"],
        )
    else:
        return _ParsedVerdict(None, "machine-readable verdict is missing")
    try:
        _validate_machine_verdict(candidate)
    except (TypeError, ValueError) as error:
        return _ParsedVerdict(None, str(error))
    return _ParsedVerdict(candidate, None)


def _validate_machine_verdict(verdict: MachineVerdict) -> None:
    if not isinstance(verdict, MachineVerdict):
        raise TypeError("verdict must be a MachineVerdict")
    _validate_project(verdict.project)
    _validate_candidate(verdict.candidate)
    _validate_identity(verdict.checker)
    if verdict.status not in MACHINE_VERDICTS:
        raise ValueError("verdict status must be PASS, FAIL_REPAIRABLE, or BLOCKED")


def _validate_request(request: CheckerRequest) -> None:
    if not isinstance(request, CheckerRequest):
        raise TypeError("checker request must be a CheckerRequest")
    if not isinstance(request.target_is_checker, bool):
        raise TypeError("target_is_checker must be boolean")
    _validate_identity(request.identity())


def _validate_project(project: ProjectIdentity) -> None:
    if not isinstance(project, ProjectIdentity):
        raise TypeError("project must be a ProjectIdentity")
    _require_text("project_id", project.project_id)
    if isinstance(project.generation, bool) or not isinstance(project.generation, int) or project.generation < 1:
        raise ValueError("generation must be an integer at least 1")


def _validate_candidate(candidate: CandidateBinding) -> None:
    if not isinstance(candidate, CandidateBinding):
        raise TypeError("candidate must be a CandidateBinding")
    _require_text("snapshot_version", candidate.snapshot_version)
    _require_text("candidate_id", candidate.candidate_id)


def _validate_identity(identity: CheckerIdentity) -> None:
    if not isinstance(identity, CheckerIdentity):
        raise TypeError("checker identity must be a CheckerIdentity")
    _require_text("checker task_id", identity.task_id)
    _require_text("checker run_id", identity.run_id)
    _require_text("checker profile", identity.profile)


def _normalize_profiles(profiles: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if not isinstance(profiles, (tuple, list)):
        raise TypeError("implementation_profiles must be a list or tuple")
    normalized = tuple(_require_text("implementation profile", profile) for profile in profiles)
    if not normalized:
        raise ValueError("at least one implementation profile is required")
    if len(set(normalized)) != len(normalized):
        raise ValueError("implementation profiles must be unique")
    return tuple(sorted(normalized))


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


__all__ = [
    "BLOCKED",
    "CHECKER_ALREADY_EXISTS",
    "CHECKER_OF_CHECKER_DENIED",
    "CHECKER_REQUIRED",
    "FAIL_REPAIRABLE",
    "MALFORMED_VERDICT",
    "PASS_ACCEPTED",
    "STALE_LOCK_VERSION",
    "STALE_SNAPSHOT",
    "STALE_VERDICT",
    "UNAUTHORIZED_CHECKER",
    "CandidateBinding",
    "CheckerGate",
    "CheckerIdentity",
    "CheckerRequest",
    "GateDecision",
    "MachineVerdict",
    "ProjectIdentity",
    "VerdictHistoryEntry",
    "create_checker_gate",
    "evaluate_checker_gate",
    "reconcile_checker",
    "submit_checker_verdict",
]
