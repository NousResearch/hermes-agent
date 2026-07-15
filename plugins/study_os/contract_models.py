"""Canonical StudyOS project and Schedule contracts.

Pydantic models are the single structural source.  Their normalized JSON
Schema generates the desktop types and runtime guards; Python callers validate
through the same models.  Aggregate relationships between a Project and a
Schedule deliberately remain in :mod:`plugins.study_os.application`.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime
from typing import Annotated, Any, Literal, Union

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    TypeAdapter,
    ValidationError,
    WithJsonSchema,
)


PROJECT_ID_PATTERN = r"^[a-z0-9][a-z0-9-]{2,63}$"
SCHEDULE_ID_PATTERN = r"^[a-z0-9][a-z0-9-]{2,79}$"
DATE_PATTERN = r"^\d{4}-\d{2}-\d{2}$"
DATETIME_WITH_OFFSET_PATTERN = (
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:[+-]\d{2}:\d{2}|Z)$"
)
EVIDENCE_DIMENSIONS = (
    "recall",
    "recognition",
    "execution",
    "explanation",
    "near_transfer",
    "far_transfer",
)
SOURCE_ANCHOR_KINDS = (
    "file",
    "paper",
    "book",
    "web",
    "dataset",
    "command",
    "commit",
    "note",
    "other",
)


def _valid_iso_date(value: str) -> str:
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("must be a valid ISO date") from exc
    return value


def _valid_offset_datetime(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("must be a valid ISO datetime") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("must include timezone offset")
    return value


NonEmptyString = Annotated[str, StringConstraints(pattern=r"\S")]
ProjectId = Annotated[str, StringConstraints(pattern=PROJECT_ID_PATTERN)]
ScheduleId = Annotated[str, StringConstraints(pattern=SCHEDULE_ID_PATTERN)]
IsoDate = Annotated[
    str,
    StringConstraints(pattern=DATE_PATTERN),
    AfterValidator(_valid_iso_date),
    WithJsonSchema(
        {"type": "string", "pattern": DATE_PATTERN, "format": "date"}
    ),
]
OffsetDateTime = Annotated[
    str,
    StringConstraints(pattern=DATETIME_WITH_OFFSET_PATTERN),
    AfterValidator(_valid_offset_datetime),
    WithJsonSchema(
        {
            "type": "string",
            "pattern": DATETIME_WITH_OFFSET_PATTERN,
            "format": "date-time",
        }
    ),
]
PositiveInt = Annotated[int, Field(ge=1)]


class _ContractModel(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)


class StudySubject(_ContractModel):
    id: NonEmptyString
    label: NonEmptyString
    target_score: int | float | None = None


class StudySourceAnchor(_ContractModel):
    kind: Literal[
        "file",
        "paper",
        "book",
        "web",
        "dataset",
        "command",
        "commit",
        "note",
        "other",
    ]
    ref: NonEmptyString
    version: NonEmptyString | None = None
    locator: NonEmptyString | None = None


class StudyObjective(_ContractModel):
    objective_id: ScheduleId
    capability: NonEmptyString
    success_criteria: list[NonEmptyString] = Field(min_length=1)
    evidence_targets: list[
        Literal[
            "recall",
            "recognition",
            "execution",
            "explanation",
            "near_transfer",
            "far_transfer",
        ]
    ] = Field(min_length=1)
    source_anchors: list[StudySourceAnchor] | None = None


class StudyPromptPolicy(_ContractModel):
    base_max_chars: PositiveInt
    intent_max_chars: PositiveInt
    domain_max_chars: PositiveInt
    project_summary_max_chars: PositiveInt
    total_max_chars: PositiveInt
    updates_apply: Literal["next_session"]


class StudyProjectV1(_ContractModel):
    model_config = ConfigDict(
        extra="allow",
        strict=True,
        json_schema_extra={"x-study-rules": ["unique:subjects:id"]},
    )

    schema_version: Literal["study_project.v1"]
    project_id: ProjectId
    title: NonEmptyString
    domain: NonEmptyString
    exam_type: NonEmptyString
    exam_date: IsoDate
    timezone: NonEmptyString
    phase: NonEmptyString
    domain_pack: NonEmptyString
    subjects: list[StudySubject] = Field(min_length=1)
    prompt_policy: StudyPromptPolicy
    created_at: OffsetDateTime
    updated_at: OffsetDateTime


class StudyProjectV2(_ContractModel):
    model_config = ConfigDict(
        extra="allow",
        strict=True,
        json_schema_extra={
            "x-study-rules": [
                "unique:tracks:id",
                "unique:objectives:objective_id",
            ]
        },
    )

    schema_version: Literal["study_project.v2"]
    project_id: ProjectId
    title: NonEmptyString
    domain: NonEmptyString
    timezone: NonEmptyString
    phase: NonEmptyString
    domain_pack: NonEmptyString
    workspace_type: NonEmptyString
    artifact_policy: NonEmptyString
    deadline: IsoDate | None = None
    tracks: list[StudySubject] = Field(min_length=1)
    objectives: list[StudyObjective] = Field(min_length=1)
    prompt_policy: StudyPromptPolicy
    created_at: OffsetDateTime
    updated_at: OffsetDateTime


StudyProject = Annotated[
    Union[StudyProjectV1, StudyProjectV2],
    Field(discriminator="schema_version"),
]


class StudyScheduleRange(_ContractModel):
    start: IsoDate
    end: IsoDate


class StudySchedulePhase(_ContractModel):
    id: NonEmptyString
    title: NonEmptyString
    start: IsoDate
    end: IsoDate
    goal: NonEmptyString
    effort_minutes: PositiveInt | None = None
    goals: list[NonEmptyString] | None = None
    source_curricula: list[NonEmptyString] | None = None
    status: NonEmptyString | None = None


class StudyScheduleEvent(_ContractModel):
    id: NonEmptyString
    title: NonEmptyString
    subject_id: NonEmptyString
    type: NonEmptyString
    start: OffsetDateTime
    end: OffsetDateTime
    duration_minutes: Annotated[int, Field(ge=1, le=720)]
    goals: list[NonEmptyString]
    source_curriculum: NonEmptyString | None = None
    status: NonEmptyString


class StudySchedule(_ContractModel):
    model_config = ConfigDict(
        extra="allow",
        strict=True,
        json_schema_extra={"x-study-rules": ["schedule-invariants"]},
    )

    schema_version: Literal["study_schedule.v1"]
    schedule_id: ScheduleId
    project_id: ProjectId
    title: NonEmptyString
    timezone: NonEmptyString
    range: StudyScheduleRange
    phases: list[StudySchedulePhase]
    events: list[StudyScheduleEvent]


class StudyScheduleSummary(_ContractModel):
    schedule_id: ScheduleId
    project_id: ProjectId
    title: NonEmptyString
    timezone: NonEmptyString
    range: StudyScheduleRange
    phase_count: int | None = None
    event_count: int


class InvalidStudySchedule(_ContractModel):
    schedule_id: str
    path: str
    errors: list[str]


class StudyProjectsResponse(_ContractModel):
    configured: bool
    projects: list[StudyProject]
    active_project_id: str | None = None
    message: str | None = None
    vault_path: str | None = None


class StudyActiveProjectResponse(_ContractModel):
    active_project_id: ProjectId
    project: StudyProject


class StudySchedulesResponse(_ContractModel):
    project_id: ProjectId
    schedules: list[StudyScheduleSummary]
    invalid_schedules: list[InvalidStudySchedule] = Field(default_factory=list)


class StudyOverviewResponse(_ContractModel):
    project: StudyProject


class _StudyContractDocument(_ContractModel):
    project: StudyProject
    schedule: StudySchedule


_PROJECT_ADAPTER = TypeAdapter(StudyProject)


def study_contract_json_schema() -> dict[str, Any]:
    """Return the normalized schema consumed by code generation."""

    document = _StudyContractDocument.model_json_schema(
        ref_template="#/$defs/{model}"
    )
    definitions = deepcopy(document.get("$defs", {}))
    definitions["StudyProject"] = deepcopy(document["properties"]["project"])
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://hermes.local/study-os/contracts.schema.json",
        "title": "StudyOS Contracts",
        "$defs": definitions,
    }


def _path(parts: tuple[Any, ...]) -> str:
    result = ""
    for part in parts:
        if isinstance(part, int):
            result += f"[{part}]"
        else:
            result += ("." if result else "") + str(part)
    return result


def _structural_errors(exc: ValidationError) -> list[str]:
    errors: list[str] = []
    for issue in exc.errors(include_url=False):
        location = tuple(
            part
            for part in issue.get("loc", ())
            if part not in {"study_project.v1", "study_project.v2"}
        )
        path = _path(location)
        issue_type = str(issue.get("type") or "")
        context = issue.get("ctx") or {}
        pattern = str(context.get("pattern") or "")
        if issue_type == "missing":
            message = f"{path} is required"
        elif issue_type == "string_type":
            message = f"{path} must be a string"
        elif issue_type == "string_pattern_mismatch":
            if pattern == PROJECT_ID_PATTERN:
                message = f"{path} must match {PROJECT_ID_PATTERN}"
            elif pattern == SCHEDULE_ID_PATTERN:
                message = f"{path} must match {SCHEDULE_ID_PATTERN}"
            elif pattern == DATETIME_WITH_OFFSET_PATTERN:
                message = f"{path} must include timezone offset"
            elif pattern == DATE_PATTERN:
                message = f"{path} must be ISO date YYYY-MM-DD"
            else:
                message = f"{path} must be a non-empty string"
        elif issue_type == "too_short" and path.endswith(
            ("success_criteria", "evidence_targets")
        ):
            message = f"{path} must be a non-empty string array"
        elif issue_type == "too_short":
            message = f"{path} must be a non-empty array"
        elif issue_type in {"greater_than_equal", "greater_than"} and path.endswith(
            "effort_minutes"
        ):
            message = f"{path} must be a positive integer"
        elif issue_type in {
            "greater_than_equal",
            "less_than_equal",
            "int_type",
        } and path.endswith("duration_minutes"):
            message = f"{path} must be an integer from 1 to 720"
        elif issue_type == "value_error":
            raw = str(context.get("error") or issue.get("msg") or "invalid value")
            message = f"{path} {raw.removeprefix('Value error, ').strip()}"
        else:
            message = f"{path} {issue.get('msg', 'is invalid')}".strip()
        if message not in errors:
            errors.append(message)
    return errors


def _duplicate_errors(
    values: Any,
    *,
    collection: str,
    key: str,
) -> list[str]:
    if not isinstance(values, list):
        return []
    seen: set[str] = set()
    errors: list[str] = []
    for index, item in enumerate(values):
        if not isinstance(item, dict) or not isinstance(item.get(key), str):
            continue
        value = item[key]
        if value in seen:
            errors.append(f"{collection}[{index}].{key} must be unique")
        seen.add(value)
    return errors


def _project_semantic_errors(project: dict[str, Any]) -> list[str]:
    if project.get("schema_version") == "study_project.v1":
        return _duplicate_errors(
            project.get("subjects"), collection="subjects", key="id"
        )
    return [
        *_duplicate_errors(project.get("tracks"), collection="tracks", key="id"),
        *_duplicate_errors(
            project.get("objectives"),
            collection="objectives",
            key="objective_id",
        ),
    ]


def _schedule_semantic_errors(schedule: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    range_data = schedule["range"]
    range_start = date.fromisoformat(range_data["start"])
    range_end = date.fromisoformat(range_data["end"])
    if range_end < range_start:
        errors.append("range.end must be on or after range.start")

    for index, phase in enumerate(schedule["phases"]):
        if date.fromisoformat(phase["end"]) < date.fromisoformat(phase["start"]):
            errors.append(f"phases[{index}].end must be on or after start")

    errors.extend(
        _duplicate_errors(schedule["events"], collection="events", key="id")
    )
    for index, event in enumerate(schedule["events"]):
        path = f"events[{index}]"
        start = datetime.fromisoformat(event["start"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(event["end"].replace("Z", "+00:00"))
        if end <= start:
            errors.append(f"{path}.end must be after start")
            continue
        actual_minutes = int((end - start).total_seconds() // 60)
        if actual_minutes > 720:
            errors.append(
                f"{path} spans more than 720 minutes; use phases for long-term "
                "ranges and events only for concrete study sessions"
            )
        elif actual_minutes != event["duration_minutes"]:
            errors.append(f"{path}.duration_minutes does not match start/end")
        if not (
            range_start <= start.date() <= range_end
            and range_start <= end.date() <= range_end
        ):
            errors.append(f"{path} must fall inside range")
    return errors


def validate_project_contract(
    data: Any,
) -> tuple[bool, dict[str, Any] | list[str]]:
    if not isinstance(data, dict):
        return False, [f"project must be an object, got {type(data).__name__}"]
    try:
        _PROJECT_ADAPTER.validate_python(data)
    except ValidationError as exc:
        return False, _structural_errors(exc)
    errors = _project_semantic_errors(data)
    return (False, errors) if errors else (True, data)


def validate_schedule_contract(
    data: Any,
) -> tuple[bool, dict[str, Any] | list[str]]:
    if not isinstance(data, dict):
        return False, [f"schedule must be an object, got {type(data).__name__}"]
    errors: list[str] = []
    try:
        StudySchedule.model_validate(data)
    except ValidationError as exc:
        errors.extend(_structural_errors(exc))
    try:
        semantic_errors = _schedule_semantic_errors(data)
    except (KeyError, TypeError, ValueError, OverflowError):
        semantic_errors = []
    errors.extend(error for error in semantic_errors if error not in errors)
    return (False, errors) if errors else (True, data)


def _inline_schema(value: Any, definitions: dict[str, Any]) -> Any:
    if isinstance(value, list):
        return [_inline_schema(item, definitions) for item in value]
    if not isinstance(value, dict):
        return value
    reference = value.get("$ref")
    if isinstance(reference, str) and reference.startswith("#/$defs/"):
        return _inline_schema(
            deepcopy(definitions[reference.removeprefix("#/$defs/")]),
            definitions,
        )
    return {
        key: _inline_schema(item, definitions)
        for key, item in value.items()
        if key not in {"$defs", "title"}
    }


def study_project_tool_properties() -> dict[str, Any]:
    """Return reusable Project fields for legacy/model tool schemas."""

    schema = study_contract_json_schema()
    definitions = schema["$defs"]
    merged: dict[str, Any] = {}
    for model_name in ("StudyProjectV1", "StudyProjectV2"):
        properties = definitions[model_name]["properties"]
        for name, value in properties.items():
            resolved = _inline_schema(value, definitions)
            if name not in merged:
                merged[name] = resolved
            elif merged[name] != resolved:
                alternatives = merged[name].get("anyOf")
                if not isinstance(alternatives, list):
                    alternatives = [merged[name]]
                if resolved not in alternatives:
                    alternatives.append(resolved)
                merged[name] = {"anyOf": alternatives}
    return merged


def study_project_id_json_schema() -> dict[str, Any]:
    return deepcopy(study_project_tool_properties()["project_id"])


def study_schedule_json_schema() -> dict[str, Any]:
    schema = study_contract_json_schema()
    return _inline_schema(schema["$defs"]["StudySchedule"], schema["$defs"])


__all__ = [
    "InvalidStudySchedule",
    "StudyActiveProjectResponse",
    "StudyOverviewResponse",
    "StudyProject",
    "StudyProjectV1",
    "StudyProjectV2",
    "StudyProjectsResponse",
    "StudySchedule",
    "StudyScheduleEvent",
    "StudySchedulePhase",
    "StudyScheduleSummary",
    "StudySchedulesResponse",
    "study_contract_json_schema",
    "study_project_id_json_schema",
    "study_project_tool_properties",
    "study_schedule_json_schema",
    "validate_project_contract",
    "validate_schedule_contract",
]
