"""Versioned StudyOS project and schedule validators."""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any


PROJECT_SCHEMA_VERSION_V1 = "study_project.v1"
PROJECT_SCHEMA_VERSION = "study_project.v2"
SCHEDULE_SCHEMA_VERSION = "study_schedule.v1"
ATTEMPT_SCHEMA_VERSION = "study_attempt.v1"
PATTERN_PROPOSAL_SCHEMA_VERSION = "study_pattern_proposal.v1"
LEARNING_CONTRACT_SCHEMA_VERSION = "learning_contract.v1"

PROJECT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,63}$")
SCHEDULE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,79}$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
DATETIME_WITH_OFFSET_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:[+-]\d{2}:\d{2}|Z)$")

ATTEMPT_RESULTS = {"correct", "partial", "incorrect", "abandoned"}
EVIDENCE_DIMENSIONS = (
    "recall",
    "recognition",
    "execution",
    "explanation",
    "near_transfer",
    "far_transfer",
)
TRANSFER_LEVELS = set(EVIDENCE_DIMENSIONS)
LEARNING_MODES = {"execute", "learn", "assess", "research"}
ASSISTANCE_LEVELS = {"direct", "guided", "hints_only", "independent"}
EVALUATOR_KINDS = {"self", "agent", "program", "human"}
SOURCE_ANCHOR_KINDS = {"file", "paper", "book", "web", "dataset", "command", "commit", "note", "other"}

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


def _validate_subjects(value: Any, errors: list[str]) -> set[str]:
    subject_ids: set[str] = set()
    if not isinstance(value, list) or not value:
        errors.append("subjects must be a non-empty array")
        return subject_ids
    for index, subject in enumerate(value):
        path = f"subjects[{index}]"
        if not isinstance(subject, dict):
            errors.append(f"{path} must be an object")
            continue
        subject_id = subject.get("id")
        if not isinstance(subject_id, str) or not subject_id.strip():
            errors.append(f"{path}.id must be a non-empty string")
        elif subject_id in subject_ids:
            errors.append(f"{path}.id must be unique")
        else:
            subject_ids.add(subject_id)
        if not isinstance(subject.get("label"), str) or not subject["label"].strip():
            errors.append(f"{path}.label must be a non-empty string")
        target_score = subject.get("target_score")
        if target_score is not None and not isinstance(target_score, (int, float)):
            errors.append(f"{path}.target_score must be a number")
    return subject_ids


def _validate_tracks(value: Any, errors: list[str]) -> set[str]:
    track_ids: set[str] = set()
    if not isinstance(value, list) or not value:
        errors.append("tracks must be a non-empty array")
        return track_ids
    for index, track in enumerate(value):
        path = f"tracks[{index}]"
        if not isinstance(track, dict):
            errors.append(f"{path} must be an object")
            continue
        track_id = track.get("id")
        if not isinstance(track_id, str) or not track_id.strip():
            errors.append(f"{path}.id must be a non-empty string")
        elif track_id in track_ids:
            errors.append(f"{path}.id must be unique")
        else:
            track_ids.add(track_id)
        if not isinstance(track.get("label"), str) or not track["label"].strip():
            errors.append(f"{path}.label must be a non-empty string")
    return track_ids


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


def _validate_objectives(value: Any, errors: list[str]) -> set[str]:
    objective_ids: set[str] = set()
    if not isinstance(value, list) or not value:
        errors.append("objectives must be a non-empty array")
        return objective_ids
    for index, objective in enumerate(value):
        path = f"objectives[{index}]"
        if not isinstance(objective, dict):
            errors.append(f"{path} must be an object")
            continue
        objective_id = objective.get("objective_id")
        if not isinstance(objective_id, str) or not SCHEDULE_ID_RE.match(objective_id):
            errors.append(f"{path}.objective_id must match ^[a-z0-9][a-z0-9-]{{2,79}}$")
        elif objective_id in objective_ids:
            errors.append(f"{path}.objective_id must be unique")
        else:
            objective_ids.add(objective_id)
        if not isinstance(objective.get("capability"), str) or not objective["capability"].strip():
            errors.append(f"{path}.capability must be a non-empty string")
        _validate_string_array(objective.get("success_criteria"), f"{path}.success_criteria", errors, non_empty=True)
        _validate_evidence_targets(objective.get("evidence_targets"), f"{path}.evidence_targets", errors)
        if "source_anchors" in objective:
            _validate_source_anchors(objective["source_anchors"], f"{path}.source_anchors", errors)
    return objective_ids


def _validate_prompt_policy(value: Any, errors: list[str]) -> None:
    policy = _require_mapping(value, "prompt_policy", errors)
    if policy is None:
        return
    for key in (
        "base_max_chars",
        "intent_max_chars",
        "domain_max_chars",
        "project_summary_max_chars",
        "total_max_chars",
    ):
        if not isinstance(policy.get(key), int) or policy[key] <= 0:
            errors.append(f"prompt_policy.{key} must be a positive integer")
    if policy.get("updates_apply") != "next_session":
        errors.append("prompt_policy.updates_apply must be next_session")


def validate_study_project(data: Any) -> tuple[bool, dict[str, Any] | list[str]]:
    """Validate a backward-compatible StudyOS project manifest.

    Unknown fields are intentionally preserved by returning the original mapping
    when validation succeeds.
    """

    errors: list[str] = []
    project = _require_mapping(data, "project", errors)
    if project is None:
        return False, errors

    schema_version = project.get("schema_version")
    if schema_version not in {PROJECT_SCHEMA_VERSION_V1, PROJECT_SCHEMA_VERSION}:
        errors.append(f"schema_version must be {PROJECT_SCHEMA_VERSION_V1} or {PROJECT_SCHEMA_VERSION}")

    project_id = _require_string(project, "project_id", errors)
    if project_id and not PROJECT_ID_RE.match(project_id):
        errors.append("project_id must match ^[a-z0-9][a-z0-9-]{2,63}$")

    for key in ("title", "domain", "timezone", "phase", "domain_pack", "created_at", "updated_at"):
        _require_string(project, key, errors)
    if schema_version == PROJECT_SCHEMA_VERSION_V1:
        _require_string(project, "exam_type", errors)
        _parse_date(project.get("exam_date"), "exam_date", errors)
        _validate_subjects(project.get("subjects"), errors)
    elif schema_version == PROJECT_SCHEMA_VERSION:
        for key in ("workspace_type", "artifact_policy"):
            _require_string(project, key, errors)
        if project.get("deadline") is not None:
            _parse_date(project.get("deadline"), "deadline", errors)
        _validate_tracks(project.get("tracks"), errors)
        _validate_objectives(project.get("objectives"), errors)
    _validate_prompt_policy(project.get("prompt_policy"), errors)
    _parse_datetime(project.get("created_at"), "created_at", errors)
    _parse_datetime(project.get("updated_at"), "updated_at", errors)

    return (False, errors) if errors else (True, project)


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
    *,
    project: dict[str, Any] | None = None,
) -> tuple[bool, dict[str, Any] | list[str]]:
    """Validate a ``study_schedule.v1`` artifact."""

    errors: list[str] = []
    schedule = _require_mapping(data, "schedule", errors)
    if schedule is None:
        return False, errors

    if schedule.get("schema_version") != SCHEDULE_SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEDULE_SCHEMA_VERSION}")

    schedule_id = _require_string(schedule, "schedule_id", errors)
    if schedule_id and not SCHEDULE_ID_RE.match(schedule_id):
        errors.append("schedule_id must match ^[a-z0-9][a-z0-9-]{2,79}$")

    project_id = _require_string(schedule, "project_id", errors)
    if project is not None and project_id and project.get("project_id") != project_id:
        errors.append("project_id must match project manifest")

    for key in ("title", "timezone"):
        _require_string(schedule, key, errors)

    range_data = _require_mapping(schedule.get("range"), "range", errors)
    range_start = range_end = None
    if range_data is not None:
        range_start = _parse_date(range_data.get("start"), "range.start", errors)
        range_end = _parse_date(range_data.get("end"), "range.end", errors)
        if range_start and range_end and range_end < range_start:
            errors.append("range.end must be on or after range.start")

    _validate_phases(schedule.get("phases"), errors)
    _validate_events(schedule.get("events"), errors, range_start=range_start, range_end=range_end, project=project)

    return (False, errors) if errors else (True, schedule)


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
                errors.append(f"{path} must be an object")
                continue
            for key in ("kind", "evidence"):
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


def _validate_phases(value: Any, errors: list[str]) -> None:
    if not isinstance(value, list):
        errors.append("phases must be an array")
        return
    for index, phase in enumerate(value):
        path = f"phases[{index}]"
        if not isinstance(phase, dict):
            errors.append(f"{path} must be an object")
            continue
        for key in ("id", "title", "goal"):
            if not isinstance(phase.get(key), str) or not phase[key].strip():
                errors.append(f"{path}.{key} must be a non-empty string")
        start = _parse_date(phase.get("start"), f"{path}.start", errors)
        end = _parse_date(phase.get("end"), f"{path}.end", errors)
        if start and end and end < start:
            errors.append(f"{path}.end must be on or after start")


def _validate_events(
    value: Any,
    errors: list[str],
    *,
    range_start: date | None,
    range_end: date | None,
    project: dict[str, Any] | None,
) -> None:
    if not isinstance(value, list):
        errors.append("events must be an array")
        return
    subject_ids = _project_subject_ids(project)
    seen_ids: set[str] = set()
    for index, event in enumerate(value):
        path = f"events[{index}]"
        if not isinstance(event, dict):
            errors.append(f"{path} must be an object")
            continue
        event_id = event.get("id")
        if not isinstance(event_id, str) or not event_id.strip():
            errors.append(f"{path}.id must be a non-empty string")
        elif event_id in seen_ids:
            errors.append(f"{path}.id must be unique")
        else:
            seen_ids.add(event_id)
        for key in ("title", "subject_id", "type", "status"):
            if not isinstance(event.get(key), str) or not event[key].strip():
                errors.append(f"{path}.{key} must be a non-empty string")
        if subject_ids is not None and isinstance(event.get("subject_id"), str) and event["subject_id"] not in subject_ids:
            errors.append(f"{path}.subject_id must exist in project subjects")
        start = _parse_datetime(event.get("start"), f"{path}.start", errors)
        end = _parse_datetime(event.get("end"), f"{path}.end", errors)
        duration = event.get("duration_minutes")
        if not isinstance(duration, int) or not 1 <= duration <= 720:
            errors.append(f"{path}.duration_minutes must be an integer from 1 to 720")
        if start and end:
            if end <= start:
                errors.append(f"{path}.end must be after start")
            if isinstance(duration, int):
                actual_minutes = int((end - start).total_seconds() // 60)
                if actual_minutes != duration:
                    errors.append(f"{path}.duration_minutes does not match start/end")
            if range_start and range_end and not (range_start <= start.date() <= range_end and range_start <= end.date() <= range_end):
                errors.append(f"{path} must fall inside range")
        goals = event.get("goals")
        if not isinstance(goals, list) or not all(isinstance(goal, str) and goal.strip() for goal in goals):
            errors.append(f"{path}.goals must be an array of non-empty strings")


def _project_subject_ids(project: dict[str, Any] | None) -> set[str] | None:
    if project is None:
        return None
    subjects = project.get("tracks") if project.get("schema_version") == PROJECT_SCHEMA_VERSION else project.get("subjects")
    if not isinstance(subjects, list):
        return set()
    return {subject["id"] for subject in subjects if isinstance(subject, dict) and isinstance(subject.get("id"), str)}
