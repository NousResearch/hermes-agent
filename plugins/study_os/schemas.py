"""Versioned StudyOS project and schedule validators."""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any


PROJECT_SCHEMA_VERSION = "study_project.v1"
SCHEDULE_SCHEMA_VERSION = "study_schedule.v1"
ATTEMPT_SCHEMA_VERSION = "study_attempt.v1"
PATTERN_PROPOSAL_SCHEMA_VERSION = "study_pattern_proposal.v1"

PROJECT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,63}$")
SCHEDULE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,79}$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
DATETIME_WITH_OFFSET_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:[+-]\d{2}:\d{2}|Z)$")

ATTEMPT_RESULTS = {"correct", "partial", "incorrect", "abandoned"}
TRANSFER_LEVELS = {
    "recall",
    "recognition",
    "execution",
    "explanation",
    "near_transfer",
    "far_transfer",
}

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
    """Validate a ``study_project.v1`` manifest.

    Unknown fields are intentionally preserved by returning the original mapping
    when validation succeeds.
    """

    errors: list[str] = []
    project = _require_mapping(data, "project", errors)
    if project is None:
        return False, errors

    if project.get("schema_version") != PROJECT_SCHEMA_VERSION:
        errors.append(f"schema_version must be {PROJECT_SCHEMA_VERSION}")

    project_id = _require_string(project, "project_id", errors)
    if project_id and not PROJECT_ID_RE.match(project_id):
        errors.append("project_id must match ^[a-z0-9][a-z0-9-]{2,63}$")

    for key in ("title", "domain", "exam_type", "timezone", "phase", "domain_pack", "created_at", "updated_at"):
        _require_string(project, key, errors)
    _parse_date(project.get("exam_date"), "exam_date", errors)
    _validate_subjects(project.get("subjects"), errors)
    _validate_prompt_policy(project.get("prompt_policy"), errors)
    _parse_datetime(project.get("created_at"), "created_at", errors)
    _parse_datetime(project.get("updated_at"), "updated_at", errors)

    return (False, errors) if errors else (True, project)


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

    for key in ("concepts", "patterns"):
        value = attempt.get(key, [])
        if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
            errors.append(f"{key} must be an array of non-empty strings")
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
    subjects = project.get("subjects")
    if not isinstance(subjects, list):
        return set()
    return {subject["id"] for subject in subjects if isinstance(subject, dict) and isinstance(subject.get("id"), str)}
