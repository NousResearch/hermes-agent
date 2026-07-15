"""Versioned StudyOS project and schedule validators."""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

from plugins.study_os.contract_models import (
    DATETIME_WITH_OFFSET_PATTERN,
    DATE_PATTERN,
    EVIDENCE_DIMENSIONS,
    PROJECT_ID_PATTERN,
    SCHEDULE_ID_PATTERN,
    SOURCE_ANCHOR_KINDS,
    validate_project_contract,
    validate_schedule_contract,
)


PROJECT_SCHEMA_VERSION_V1 = "study_project.v1"
PROJECT_SCHEMA_VERSION = "study_project.v2"
SCHEDULE_SCHEMA_VERSION = "study_schedule.v1"
ATTEMPT_SCHEMA_VERSION = "study_attempt.v1"
PATTERN_PROPOSAL_SCHEMA_VERSION = "study_pattern_proposal.v1"
LEARNING_CONTRACT_SCHEMA_VERSION = "learning_contract.v1"
INTERVENTION_QUEUE_SCHEMA_VERSION = "study_intervention_queue.v1"
PLAN_PROPOSAL_SCHEMA_VERSION = "study_plan_proposal.v1"
INTERVENTION_POLICY_VERSION = "study_intervention_policy.v1"

PROJECT_ID_RE = re.compile(PROJECT_ID_PATTERN)
SCHEDULE_ID_RE = re.compile(SCHEDULE_ID_PATTERN)
DATE_RE = re.compile(DATE_PATTERN)
DATETIME_WITH_OFFSET_RE = re.compile(DATETIME_WITH_OFFSET_PATTERN)

ATTEMPT_RESULTS = {"correct", "partial", "incorrect", "abandoned"}
TRANSFER_LEVELS = set(EVIDENCE_DIMENSIONS)
LEARNING_MODES = {"execute", "learn", "assess", "research"}
ASSISTANCE_LEVELS = {"direct", "guided", "hints_only", "independent"}
EVALUATOR_KINDS = {"self", "agent", "program", "human"}
DIAGNOSIS_REQUIRED_FIELDS = ("kind", "evidence")
DIAGNOSIS_OBJECT_EXAMPLE = (
    '{"kind":"condition_missed","evidence":"The required condition was not checked."}'
)
INTERVENTION_KINDS = {
    "evidence_probe",
    "guided_repair",
    "independence_probe",
    "misconception_probe",
    "prerequisite_repair",
    "near_transfer_probe",
    "far_transfer_probe",
    "retention_probe",
}
PLAN_PROPOSAL_STATUSES = {"proposed", "accepted", "rejected"}
VERIFICATION_STATUSES = {"unobserved", "developing", "supported", "independent"}
EVIDENCE_AGE_BANDS = {"unobserved", "fresh", "aging", "stale"}
DEADLINE_BANDS = {"none", "distant", "approaching", "near", "critical", "overdue"}

DEFAULT_PROMPT_POLICY: dict[str, Any] = {
    "base_max_chars": 2000,
    "intent_max_chars": 2500,
    "domain_max_chars": 2000,
    "project_summary_max_chars": 1200,
    "total_max_chars": 6000,
    "updates_apply": "next_session",
}


def _type_name(value: Any) -> str:
    return type(value).__name__


def _require_mapping(value: Any, path: str, errors: list[str]) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        errors.append(f"{path} must be an object, got {_type_name(value)}")
        return None
    return value


def _require_string(data: dict[str, Any], key: str, errors: list[str], *, non_empty: bool = True) -> str | None:
    value = data.get(key)
    if not isinstance(value, str):
        errors.append(f"{key} must be a string")
        return None
    if non_empty and not value.strip():
        errors.append(f"{key} must not be empty")
        return None
    return value


def _parse_date(value: Any, path: str, errors: list[str]) -> date | None:
    if not isinstance(value, str) or not DATE_RE.match(value):
        errors.append(f"{path} must be ISO date YYYY-MM-DD")
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        errors.append(f"{path} must be a valid ISO date")
        return None


def _parse_datetime(value: Any, path: str, errors: list[str]) -> datetime | None:
    if not isinstance(value, str):
        errors.append(f"{path} must be a string")
        return None
    if not DATETIME_WITH_OFFSET_RE.match(value):
        errors.append(f"{path} must include timezone offset")
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        errors.append(f"{path} must be a valid ISO datetime")
        return None


def _validate_string_array(value: Any, path: str, errors: list[str], *, non_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (non_empty and not value) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        qualifier = "non-empty " if non_empty else ""
        errors.append(f"{path} must be a {qualifier}string array")
        return []
    return value


def _validate_evidence_targets(value: Any, path: str, errors: list[str], *, non_empty: bool = True) -> list[str]:
    targets = _validate_string_array(value, path, errors, non_empty=non_empty)
    unknown = sorted(set(targets) - TRANSFER_LEVELS)
    if unknown:
        errors.append(f"{path} contains unsupported evidence dimensions: {', '.join(unknown)}")
    return targets


def _validate_source_anchors(value: Any, path: str, errors: list[str]) -> None:
    if not isinstance(value, list):
        errors.append(f"{path} must be an array")
        return
    for index, anchor in enumerate(value):
        item_path = f"{path}[{index}]"
        if not isinstance(anchor, dict):
            errors.append(f"{item_path} must be an object")
            continue
        if anchor.get("kind") not in SOURCE_ANCHOR_KINDS:
            errors.append(f"{item_path}.kind must be one of: {', '.join(sorted(SOURCE_ANCHOR_KINDS))}")
        if not isinstance(anchor.get("ref"), str) or not anchor["ref"].strip():
            errors.append(f"{item_path}.ref must be a non-empty string")
        for key in ("version", "locator"):
            if anchor.get(key) is not None and (not isinstance(anchor[key], str) or not anchor[key].strip()):
                errors.append(f"{item_path}.{key} must be a non-empty string when provided")


def validate_study_project(data: Any) -> tuple[bool, dict[str, Any] | list[str]]:
    """Validate against the canonical generated Project contract."""

    return validate_project_contract(data)


def validate_learning_contract(
    data: Any,
    *,
    project: dict[str, Any] | None = None,
) -> tuple[bool, dict[str, Any] | list[str]]:
    """Validate one explicit, turn-independent agreement for a learning Session."""

    errors: list[str] = []
    contract = _require_mapping(data, "contract", errors)
    if contract is None:
        return False, errors
    if contract.get("schema_version") != LEARNING_CONTRACT_SCHEMA_VERSION:
        errors.append(f"schema_version must be {LEARNING_CONTRACT_SCHEMA_VERSION}")
    for key in ("contract_id", "project_id", "objective"):
        _require_string(contract, key, errors)
    if isinstance(contract.get("contract_id"), str) and not SCHEDULE_ID_RE.match(contract["contract_id"]):
        errors.append("contract_id must match ^[a-z0-9][a-z0-9-]{2,79}$")
    project_id = contract.get("project_id")
    if isinstance(project_id, str) and not PROJECT_ID_RE.match(project_id):
        errors.append("project_id must match ^[a-z0-9][a-z0-9-]{2,63}$")
    if project is not None and project_id != project.get("project_id"):
        errors.append("project_id must match project manifest")
    if contract.get("mode") not in LEARNING_MODES:
        errors.append(f"mode must be one of: {', '.join(sorted(LEARNING_MODES))}")
    if contract.get("assistance_level") not in ASSISTANCE_LEVELS:
        errors.append(f"assistance_level must be one of: {', '.join(sorted(ASSISTANCE_LEVELS))}")
    budget = contract.get("time_budget_minutes")
    if not isinstance(budget, int) or isinstance(budget, bool) or not 1 <= budget <= 720:
        errors.append("time_budget_minutes must be an integer from 1 to 720")
    objective_ids = _validate_string_array(contract.get("objective_ids", []), "objective_ids", errors)
    evidence_targets = _validate_evidence_targets(contract.get("evidence_targets"), "evidence_targets", errors)
    if project is not None and project.get("schema_version") == PROJECT_SCHEMA_VERSION:
        objectives = {
            str(item.get("objective_id")): item
            for item in project.get("objectives", [])
            if isinstance(item, dict) and item.get("objective_id")
        }
        known_ids = set(objectives)
        unknown_ids = sorted(set(objective_ids) - known_ids)
        if unknown_ids:
            errors.append(f"objective_ids must exist in project objectives: {', '.join(unknown_ids)}")
        elif objective_ids:
            supported_targets = {
                str(target)
                for objective_id in objective_ids
                for target in objectives[objective_id].get("evidence_targets", [])
            }
            unsupported_targets = sorted(set(evidence_targets) - supported_targets)
            if unsupported_targets:
                errors.append(
                    "evidence_targets must be declared by the referenced objectives: "
                    + ", ".join(unsupported_targets)
                )
    _parse_datetime(contract.get("created_at"), "created_at", errors)
    return (False, errors) if errors else (True, contract)


def validate_study_schedule(
    data: Any,
) -> tuple[bool, dict[str, Any] | list[str]]:
    """Validate against the canonical generated Schedule contract."""

    return validate_schedule_contract(data)


def validate_study_attempt(data: Any) -> tuple[bool, dict[str, Any] | list[str]]:
    """Validate one immutable learning attempt event."""

    errors: list[str] = []
    attempt = _require_mapping(data, "attempt", errors)
    if attempt is None:
        return False, errors

    if attempt.get("schema_version") != ATTEMPT_SCHEMA_VERSION:
        errors.append(f"schema_version must be {ATTEMPT_SCHEMA_VERSION}")
    for key in ("attempt_id", "project_id", "item_id", "response", "result"):
        _require_string(attempt, key, errors)
    project_id = attempt.get("project_id")
    if isinstance(project_id, str) and not PROJECT_ID_RE.match(project_id):
        errors.append("project_id must match ^[a-z0-9][a-z0-9-]{2,63}$")
    _parse_datetime(attempt.get("occurred_at"), "occurred_at", errors)

    if attempt.get("result") not in ATTEMPT_RESULTS:
        errors.append(f"result must be one of: {', '.join(sorted(ATTEMPT_RESULTS))}")
    score = attempt.get("score")
    if not isinstance(score, (int, float)) or isinstance(score, bool) or not 0 <= score <= 1:
        errors.append("score must be a number from 0 to 1")
    duration = attempt.get("duration_seconds")
    if duration is not None and (not isinstance(duration, int) or isinstance(duration, bool) or duration < 0):
        errors.append("duration_seconds must be a non-negative integer")
    hints = attempt.get("hints_used")
    if hints is not None and (not isinstance(hints, int) or isinstance(hints, bool) or hints < 0):
        errors.append("hints_used must be a non-negative integer")
    confidence = attempt.get("self_confidence")
    if confidence is not None and (
        not isinstance(confidence, int) or isinstance(confidence, bool) or not 1 <= confidence <= 5
    ):
        errors.append("self_confidence must be an integer from 1 to 5")
    evaluator_confidence = attempt.get("evaluator_confidence")
    if evaluator_confidence is not None and (
        not isinstance(evaluator_confidence, (int, float))
        or isinstance(evaluator_confidence, bool)
        or not 0 <= evaluator_confidence <= 1
    ):
        errors.append("evaluator_confidence must be a number from 0 to 1")
    transfer_level = attempt.get("transfer_level")
    if transfer_level is not None and transfer_level not in TRANSFER_LEVELS:
        errors.append(f"transfer_level must be one of: {', '.join(sorted(TRANSFER_LEVELS))}")

    for key in ("concepts", "patterns", "objective_ids"):
        value = attempt.get(key, [])
        if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
            errors.append(f"{key} must be an array of non-empty strings")
    evaluator = attempt.get("evaluator")
    if evaluator is not None:
        if not isinstance(evaluator, dict):
            errors.append("evaluator must be an object")
        else:
            if evaluator.get("kind") not in EVALUATOR_KINDS:
                errors.append(f"evaluator.kind must be one of: {', '.join(sorted(EVALUATOR_KINDS))}")
            evaluator_score = evaluator.get("confidence")
            if evaluator_score is not None and (
                not isinstance(evaluator_score, (int, float))
                or isinstance(evaluator_score, bool)
                or not 0 <= evaluator_score <= 1
            ):
                errors.append("evaluator.confidence must be a number from 0 to 1")
            if evaluator.get("id") is not None and (
                not isinstance(evaluator["id"], str) or not evaluator["id"].strip()
            ):
                errors.append("evaluator.id must be a non-empty string when provided")
    assistance = attempt.get("assistance")
    if assistance is not None:
        if not isinstance(assistance, dict):
            errors.append("assistance must be an object")
        else:
            if assistance.get("level") not in ASSISTANCE_LEVELS:
                errors.append(f"assistance.level must be one of: {', '.join(sorted(ASSISTANCE_LEVELS))}")
            assistance_hints = assistance.get("hints_used", 0)
            if not isinstance(assistance_hints, int) or isinstance(assistance_hints, bool) or assistance_hints < 0:
                errors.append("assistance.hints_used must be a non-negative integer")
    if "source_anchors" in attempt:
        _validate_source_anchors(attempt["source_anchors"], "source_anchors", errors)
    if "artifact_refs" in attempt:
        _validate_string_array(attempt["artifact_refs"], "artifact_refs", errors)
    diagnoses = attempt.get("diagnoses", [])
    if not isinstance(diagnoses, list):
        errors.append("diagnoses must be an array")
    else:
        for index, diagnosis in enumerate(diagnoses):
            path = f"diagnoses[{index}]"
            if not isinstance(diagnosis, dict):
                errors.append(
                    f'{path} must be an object with non-empty string fields "kind" and "evidence"; '
                    f"example: {DIAGNOSIS_OBJECT_EXAMPLE}"
                )
                continue
            for key in DIAGNOSIS_REQUIRED_FIELDS:
                if not isinstance(diagnosis.get(key), str) or not diagnosis[key].strip():
                    errors.append(f"{path}.{key} must be a non-empty string")

    return (False, errors) if errors else (True, attempt)


def validate_pattern_proposal(data: Any) -> tuple[bool, dict[str, Any] | list[str]]:
    """Validate a versioned, evidence-backed problem-pattern proposal."""

    errors: list[str] = []
    proposal = _require_mapping(data, "proposal", errors)
    if proposal is None:
        return False, errors
    if proposal.get("schema_version") != PATTERN_PROPOSAL_SCHEMA_VERSION:
        errors.append(f"schema_version must be {PATTERN_PROPOSAL_SCHEMA_VERSION}")
    for key in ("proposal_id", "project_id", "title", "change_type", "status", "rationale"):
        _require_string(proposal, key, errors)
    project_id = proposal.get("project_id")
    if isinstance(project_id, str) and not PROJECT_ID_RE.match(project_id):
        errors.append("project_id must match ^[a-z0-9][a-z0-9-]{2,63}$")
    if proposal.get("change_type") not in {"create", "supplement", "split", "merge", "demote"}:
        errors.append("change_type must be create, supplement, split, merge, or demote")
    if proposal.get("status") not in {"candidate", "accepted", "rejected"}:
        errors.append("status must be candidate, accepted, or rejected")
    evidence_ids = proposal.get("evidence_attempt_ids")
    if not isinstance(evidence_ids, list) or not evidence_ids or any(
        not isinstance(item, str) or not item.strip() for item in evidence_ids
    ):
        errors.append("evidence_attempt_ids must be a non-empty string array")
    _parse_datetime(proposal.get("created_at"), "created_at", errors)
    return (False, errors) if errors else (True, proposal)

def validate_plan_proposal(data: Any) -> tuple[bool, dict[str, Any] | list[str]]:
    """Validate a durable proposal derived from an Intervention Queue.

    A proposal may cite no attempts when its reason is precisely that an
    Objective has never produced evidence.  It never represents an applied
    Schedule change; acceptance is a recorded decision, not a hidden write to
    the Schedule aggregate.
    """

    errors: list[str] = []
    proposal = _require_mapping(data, "proposal", errors)
    if proposal is None:
        return False, errors

    if proposal.get("schema_version") != PLAN_PROPOSAL_SCHEMA_VERSION:
        errors.append(f"schema_version must be {PLAN_PROPOSAL_SCHEMA_VERSION}")
    if proposal.get("policy_version") != INTERVENTION_POLICY_VERSION:
        errors.append(f"policy_version must be {INTERVENTION_POLICY_VERSION}")
    for key in ("proposal_id", "project_id", "title", "status", "rationale", "generation_fingerprint"):
        _require_string(proposal, key, errors)

    proposal_id = proposal.get("proposal_id")
    if isinstance(proposal_id, str) and not SCHEDULE_ID_RE.match(proposal_id):
        errors.append("proposal_id must match ^[a-z0-9][a-z0-9-]{2,79}$")
    project_id = proposal.get("project_id")
    if isinstance(project_id, str) and not PROJECT_ID_RE.match(project_id):
        errors.append("project_id must match ^[a-z0-9][a-z0-9-]{2,63}$")
    fingerprint = proposal.get("generation_fingerprint")
    if isinstance(fingerprint, str) and not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        errors.append("generation_fingerprint must be a 64-character lowercase hex digest")
    elif (
        isinstance(fingerprint, str)
        and isinstance(proposal_id, str)
        and proposal_id != f"plan-{fingerprint[:20]}"
    ):
        errors.append("proposal_id must be derived from generation_fingerprint")
    status = proposal.get("status")
    if status not in PLAN_PROPOSAL_STATUSES:
        errors.append("status must be proposed, accepted, or rejected")

    _parse_datetime(proposal.get("created_at"), "created_at", errors)
    _parse_datetime(proposal.get("as_of"), "as_of", errors)
    proposal_evidence = _validate_string_array(
        proposal.get("evidence_attempt_ids"),
        "evidence_attempt_ids",
        errors,
    )
    if len(proposal_evidence) != len(set(proposal_evidence)):
        errors.append("evidence_attempt_ids must not contain duplicates")

    items = proposal.get("items")
    item_evidence: list[str] = []
    if not isinstance(items, list) or not items:
        errors.append("items must be a non-empty array")
    else:
        seen_intervention_ids: set[str] = set()
        for index, item in enumerate(items):
            path = f"items[{index}]"
            if not isinstance(item, dict):
                errors.append(f"{path} must be an object")
                continue
            for key in ("intervention_id", "objective_id", "capability", "kind", "evidence_dimension", "priority_band"):
                if not isinstance(item.get(key), str) or not item[key].strip():
                    errors.append(f"{path}.{key} must be a non-empty string")
            intervention_id = item.get("intervention_id")
            if isinstance(intervention_id, str):
                if not SCHEDULE_ID_RE.match(intervention_id):
                    errors.append(f"{path}.intervention_id must match ^[a-z0-9][a-z0-9-]{{2,79}}$")
                elif intervention_id in seen_intervention_ids:
                    errors.append(f"{path}.intervention_id must be unique")
                else:
                    seen_intervention_ids.add(intervention_id)
            objective_id = item.get("objective_id")
            if isinstance(objective_id, str) and not SCHEDULE_ID_RE.match(objective_id):
                errors.append(f"{path}.objective_id must match ^[a-z0-9][a-z0-9-]{{2,79}}$")
            if item.get("kind") not in INTERVENTION_KINDS:
                errors.append(f"{path}.kind must be a supported intervention kind")
            if item.get("evidence_dimension") not in TRANSFER_LEVELS:
                errors.append(f"{path}.evidence_dimension must be a supported evidence dimension")

            score = item.get("priority_score")
            if (
                not isinstance(score, (int, float))
                or isinstance(score, bool)
                or not 0 <= score <= 100
            ):
                errors.append(f"{path}.priority_score must be a number from 0 to 100")
            elif item.get("priority_band") != (
                "high" if score >= 80 else "medium" if score >= 55 else "low"
            ):
                errors.append(f"{path}.priority_band must match priority_score")

            _validate_string_array(item.get("reasons"), f"{path}.reasons", errors, non_empty=True)
            evidence_ids = _validate_string_array(
                item.get("evidence_attempt_ids"),
                f"{path}.evidence_attempt_ids",
                errors,
            )
            if len(evidence_ids) != len(set(evidence_ids)):
                errors.append(f"{path}.evidence_attempt_ids must not contain duplicates")
            item_evidence.extend(evidence_ids)

            factors = _require_mapping(item.get("reason_factors"), f"{path}.reason_factors", errors)
            if factors is not None:
                if factors.get("verification_status") not in VERIFICATION_STATUSES:
                    errors.append(f"{path}.reason_factors.verification_status is invalid")
                age = factors.get("evidence_age_days")
                if age is not None and (
                    not isinstance(age, int) or isinstance(age, bool) or age < 0
                ):
                    errors.append(f"{path}.reason_factors.evidence_age_days must be null or a non-negative integer")
                if factors.get("evidence_age_band") not in EVIDENCE_AGE_BANDS:
                    errors.append(f"{path}.reason_factors.evidence_age_band is invalid")
                freshness = factors.get("freshness_threshold_days")
                if not isinstance(freshness, int) or isinstance(freshness, bool) or freshness <= 0:
                    errors.append(f"{path}.reason_factors.freshness_threshold_days must be a positive integer")
                days_to_deadline = factors.get("days_to_deadline")
                if days_to_deadline is not None and (
                    not isinstance(days_to_deadline, int) or isinstance(days_to_deadline, bool)
                ):
                    errors.append(f"{path}.reason_factors.days_to_deadline must be null or an integer")
                if factors.get("deadline_band") not in DEADLINE_BANDS:
                    errors.append(f"{path}.reason_factors.deadline_band is invalid")

            activity = _require_mapping(item.get("recommended_activity"), f"{path}.recommended_activity", errors)
            if activity is not None:
                if not isinstance(activity.get("activity_kind"), str) or not activity["activity_kind"].strip():
                    errors.append(f"{path}.recommended_activity.activity_kind must be a non-empty string")
                if activity.get("evidence_target") not in TRANSFER_LEVELS:
                    errors.append(f"{path}.recommended_activity.evidence_target is invalid")
                if activity.get("assistance_level") not in ASSISTANCE_LEVELS:
                    errors.append(f"{path}.recommended_activity.assistance_level is invalid")
                duration = activity.get("duration_minutes")
                if not isinstance(duration, int) or isinstance(duration, bool) or not 1 <= duration <= 720:
                    errors.append(f"{path}.recommended_activity.duration_minutes must be an integer from 1 to 720")
                _validate_string_array(
                    activity.get("success_criteria"),
                    f"{path}.recommended_activity.success_criteria",
                    errors,
                    non_empty=True,
                )
                if "source_anchors" in activity:
                    _validate_source_anchors(
                        activity["source_anchors"],
                        f"{path}.recommended_activity.source_anchors",
                        errors,
                    )

    if set(item_evidence) != set(proposal_evidence):
        errors.append("evidence_attempt_ids must equal the union of item evidence_attempt_ids")

    schedule_change = _require_mapping(proposal.get("schedule_change"), "schedule_change", errors)
    if schedule_change is not None:
        if schedule_change.get("state") != "not_applied":
            errors.append("schedule_change.state must be not_applied")
        if schedule_change.get("requires_explicit_save") is not True:
            errors.append("schedule_change.requires_explicit_save must be true")

    decision = proposal.get("decision")
    if status == "proposed":
        if decision is not None:
            errors.append("decision must be absent while status is proposed")
    elif status in {"accepted", "rejected"}:
        decision_data = _require_mapping(decision, "decision", errors)
        if decision_data is not None:
            if decision_data.get("outcome") != status:
                errors.append("decision.outcome must match status")
            _parse_datetime(decision_data.get("decided_at"), "decision.decided_at", errors)
            if decision_data.get("note") is not None and (
                not isinstance(decision_data["note"], str) or not decision_data["note"].strip()
            ):
                errors.append("decision.note must be a non-empty string when provided")

    return (False, errors) if errors else (True, proposal)
