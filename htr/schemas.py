"""Minimal JSON schemas for HTR workspace status files."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

SchemaName = Literal[
    "run_manifest",
    "task_status",
    "attempt_status",
    "artifact_manifest",
    "artifact_entry",
    "event",
    "task_card",
    "attempt_result",
    "verification_result",
    "verification_check",
    "task_completion_record",
    "run_completion_record",
    "run_review_record",
    "run_followup_plan_record",
    "run_execution_request_record",
    "run_execution_result_record",
    "run_execution_verification_record",
    "run_post_verification_followup_plan_record",
    "run_post_verification_execution_request_record",
    "run_post_verification_execution_result_record",
    "run_post_verification_execution_verification_record",
    "run_final_closure_record",
]


class RunManifest(TypedDict):
    run_id: str
    created_at: str
    status: str


class TaskStatus(TypedDict):
    task_id: str
    run_id: str
    status: str
    attempts: list[str]


class AttemptStatus(TypedDict):
    attempt_id: str
    task_id: str
    run_id: str
    status: str


class ArtifactManifest(TypedDict, total=False):
    schema_version: str
    run_id: str
    task_id: str
    attempt_id: str
    artifacts: list[dict[str, Any]]


class EventRecord(TypedDict, total=False):
    event_id: str
    event_type: str
    run_id: str
    task_id: str
    attempt_id: str
    created_at: str
    actor: str
    payload: dict[str, Any]
    previous_status: str
    new_status: str


class TaskCard(TypedDict):
    schema_version: str
    run_id: str
    task_id: str
    title: str
    instruction: str
    created_at: str
    created_by: str
    inputs: dict[str, Any]
    constraints: dict[str, Any]
    acceptance: dict[str, Any]
    metadata: dict[str, Any]


class VerificationCheck(TypedDict, total=False):
    name: str
    status: str
    message: str | None
    metadata: dict[str, Any]


class VerificationResult(TypedDict):
    schema_version: str
    run_id: str
    task_id: str
    attempt_id: str
    outcome: str
    summary: str | None
    checks: list[dict[str, Any]]
    metadata: dict[str, Any]
    created_at: str


class TaskCompletionRecord(TypedDict):
    schema_version: str
    run_id: str
    task_id: str
    attempt_id: str
    reason: str | None
    metadata: dict[str, Any]
    created_at: str


class RunCompletionRecord(TypedDict):
    schema_version: str
    run_id: str
    completed_task_ids: list[str]
    reason: str | None
    metadata: dict[str, Any]
    created_at: str


class RunReviewRecord(TypedDict):
    schema_version: str
    run_id: str
    decision: str
    reviewer: str
    notes: str | None
    metadata: dict[str, Any]
    created_at: str


class RunFollowupPlanRecord(TypedDict):
    schema_version: int
    run_id: str
    source_review_decision: str
    planner: str
    plan_status: str
    summary: str
    followup_items: list[dict[str, Any]]
    notes: str | None
    metadata: dict[str, Any]
    created_at: str


class RunExecutionRequestRecord(TypedDict):
    schema_version: int
    run_id: str
    source_followup_plan_fingerprint: str
    requester: str
    request_status: str
    execution_items: list[dict[str, Any]]
    notes: str | None
    metadata: dict[str, Any]
    created_at: str


class AttemptResult(TypedDict):
    schema_version: str
    run_id: str
    task_id: str
    attempt_id: str
    created_at: str
    produced_by: str
    summary: str
    outputs: dict[str, Any]
    artifacts: list[Any]
    metrics: dict[str, Any]
    metadata: dict[str, Any]


_REQUIRED_FIELDS: dict[SchemaName, tuple[str, ...]] = {
    "run_manifest": ("run_id", "created_at", "status"),
    "task_status": ("task_id", "run_id", "status", "attempts"),
    "attempt_status": ("attempt_id", "task_id", "run_id", "status"),
    "artifact_manifest": ("attempt_id", "artifacts"),
    "artifact_entry": ("path", "kind", "created_at", "metadata"),
    "event": (
        "event_id",
        "event_type",
        "run_id",
        "created_at",
        "actor",
        "payload",
    ),
    "task_card": (
        "schema_version",
        "run_id",
        "task_id",
        "title",
        "instruction",
        "created_at",
        "created_by",
        "inputs",
        "constraints",
        "acceptance",
        "metadata",
    ),
    "attempt_result": (
        "schema_version",
        "run_id",
        "task_id",
        "attempt_id",
        "created_at",
        "produced_by",
        "summary",
        "outputs",
        "artifacts",
        "metrics",
        "metadata",
    ),
    "verification_result": (
        "schema_version",
        "run_id",
        "task_id",
        "attempt_id",
        "outcome",
        "summary",
        "checks",
        "metadata",
        "created_at",
    ),
    "verification_check": ("name", "status", "metadata"),
    "task_completion_record": (
        "schema_version",
        "run_id",
        "task_id",
        "attempt_id",
        "reason",
        "metadata",
        "created_at",
    ),
    "run_completion_record": (
        "schema_version",
        "run_id",
        "completed_task_ids",
        "reason",
        "metadata",
        "created_at",
    ),
    "run_review_record": (
        "schema_version",
        "run_id",
        "decision",
        "reviewer",
        "notes",
        "metadata",
        "created_at",
    ),
    "run_followup_plan_record": (
        "schema_version",
        "run_id",
        "source_review_decision",
        "planner",
        "plan_status",
        "summary",
        "followup_items",
        "notes",
        "metadata",
        "created_at",
    ),
    "run_execution_request_record": (
        "schema_version",
        "run_id",
        "source_followup_plan_fingerprint",
        "requester",
        "request_status",
        "execution_items",
        "notes",
        "metadata",
        "created_at",
    ),
    "run_execution_result_record": (
        "schema_version",
        "run_id",
        "source_execution_request_fingerprint",
        "executor",
        "result_status",
        "item_results",
        "notes",
        "metadata",
        "created_at",
    ),
    "run_execution_verification_record": (
        "schema_version",
        "run_id",
        "source_execution_result_fingerprint",
        "reviewer",
        "verification_decision",
        "item_verifications",
        "notes",
        "metadata",
        "created_at",
    ),
    "run_post_verification_followup_plan_record": (
        "schema_version",
        "run_id",
        "source_execution_result_fingerprint",
        "source_execution_verification_fingerprint",
        "planner",
        "plan_status",
        "followup_items",
        "notes",
        "metadata",
        "created_at",
    ),
    "run_post_verification_execution_request_record": (
        "schema_version",
        "run_id",
        "source_execution_result_fingerprint",
        "source_execution_verification_fingerprint",
        "source_post_verification_followup_plan_fingerprint",
        "requester",
        "request_status",
        "request_items",
        "notes",
        "metadata",
        "created_at",
    ),
    "run_post_verification_execution_result_record": (
        "schema_version",
        "run_id",
        "source_execution_result_fingerprint",
        "source_execution_verification_fingerprint",
        "source_post_verification_followup_plan_fingerprint",
        "source_post_verification_execution_request_fingerprint",
        "executor",
        "result_status",
        "result_items",
        "notes",
        "metadata",
        "created_at",
    ),
    "run_post_verification_execution_verification_record": (
        "schema_version",
        "run_id",
        "source_execution_result_fingerprint",
        "source_execution_verification_fingerprint",
        "source_post_verification_followup_plan_fingerprint",
        "source_post_verification_execution_request_fingerprint",
        "source_post_verification_execution_result_fingerprint",
        "verifier",
        "verification_status",
        "verification_items",
        "notes",
        "metadata",
        "created_at",
    ),
    "run_final_closure_record": (
        "schema_version",
        "run_id",
        "source_run_completion_fingerprint",
        "source_run_review_fingerprint",
        "source_run_followup_plan_fingerprint",
        "source_run_execution_request_fingerprint",
        "source_run_execution_result_fingerprint",
        "source_run_execution_verification_fingerprint",
        "source_post_verification_followup_plan_fingerprint",
        "source_post_verification_execution_request_fingerprint",
        "source_post_verification_execution_result_fingerprint",
        "source_post_verification_execution_verification_fingerprint",
        "closer",
        "final_closure_status",
        "closure_reason",
        "closure_items",
        "notes",
        "metadata",
        "created_at",
    ),
}


def validate(data: Any, schema_name: SchemaName) -> None:
    """Validate *data* against a named HTR workspace schema.

    Raises:
        TypeError: when *data* is not a mapping.
        ValueError: when required fields are missing or have wrong types.
    """
    if not isinstance(data, dict):
        raise TypeError(f"{schema_name}: expected dict, got {type(data).__name__}")

    missing = [field for field in _REQUIRED_FIELDS[schema_name] if field not in data]
    if missing:
        raise ValueError(f"{schema_name}: missing fields: {', '.join(missing)}")

    if schema_name == "run_manifest":
        _validate_run_manifest(data)
    elif schema_name == "task_status":
        _validate_task_status(data)
    elif schema_name == "attempt_status":
        _validate_attempt_status(data)
    elif schema_name == "artifact_manifest":
        _validate_artifact_manifest(data)
    elif schema_name == "artifact_entry":
        _validate_artifact_entry(data)
    elif schema_name == "event":
        _validate_event(data)
    elif schema_name == "task_card":
        _validate_task_card(data)
    elif schema_name == "attempt_result":
        _validate_attempt_result(data)
    elif schema_name == "verification_result":
        _validate_verification_result(data)
    elif schema_name == "verification_check":
        _validate_verification_check(data)
    elif schema_name == "task_completion_record":
        _validate_task_completion_record(data)
    elif schema_name == "run_completion_record":
        _validate_run_completion_record(data)
    elif schema_name == "run_review_record":
        _validate_run_review_record(data)
    elif schema_name == "run_followup_plan_record":
        _validate_run_followup_plan_record(data)
    elif schema_name == "run_execution_request_record":
        _validate_run_execution_request_record(data)
    elif schema_name == "run_execution_result_record":
        _validate_run_execution_result_record(data)
    elif schema_name == "run_execution_verification_record":
        _validate_run_execution_verification_record(data)
    elif schema_name == "run_post_verification_followup_plan_record":
        _validate_run_post_verification_followup_plan_record(data)
    elif schema_name == "run_post_verification_execution_request_record":
        _validate_run_post_verification_execution_request_record(data)
    elif schema_name == "run_post_verification_execution_result_record":
        _validate_run_post_verification_execution_result_record(data)
    elif schema_name == "run_post_verification_execution_verification_record":
        _validate_run_post_verification_execution_verification_record(data)
    elif schema_name == "run_final_closure_record":
        _validate_run_final_closure_record(data)


def _require_str(data: dict[str, Any], field: str, schema_name: str) -> None:
    if not isinstance(data[field], str) or not data[field]:
        raise ValueError(f"{schema_name}: {field} must be a non-empty string")


def _validate_run_manifest(data: dict[str, Any]) -> None:
    for field in ("run_id", "created_at", "status"):
        _require_str(data, field, "run_manifest")


def _validate_task_status(data: dict[str, Any]) -> None:
    for field in ("task_id", "run_id", "status"):
        _require_str(data, field, "task_status")
    if not isinstance(data["attempts"], list):
        raise ValueError("task_status: attempts must be a list")


def _validate_attempt_status(data: dict[str, Any]) -> None:
    for field in ("attempt_id", "task_id", "run_id", "status"):
        _require_str(data, field, "attempt_status")


def _validate_artifact_manifest(data: dict[str, Any]) -> None:
    _require_str(data, "attempt_id", "artifact_manifest")
    if not isinstance(data["artifacts"], list):
        raise ValueError("artifact_manifest: artifacts must be a list")
    for entry in data["artifacts"]:
        if not isinstance(entry, dict):
            raise ValueError("artifact_manifest: each artifact must be a dict")
        _validate_artifact_entry(entry)
    for optional_field in ("schema_version", "run_id", "task_id"):
        if optional_field in data:
            _require_str(data, optional_field, "artifact_manifest")


def _validate_artifact_entry(data: dict[str, Any]) -> None:
    for field in ("path", "kind", "created_at"):
        _require_str(data, field, "artifact_entry")
    if not isinstance(data["metadata"], dict):
        raise ValueError("artifact_entry: metadata must be a dict")
    if "sha256" in data and data["sha256"] is not None:
        if not isinstance(data["sha256"], str) or not data["sha256"]:
            raise ValueError("artifact_entry: sha256 must be a non-empty string or null")
    if "size_bytes" in data and data["size_bytes"] is not None:
        if not isinstance(data["size_bytes"], int) or data["size_bytes"] < 0:
            raise ValueError("artifact_entry: size_bytes must be a non-negative int or null")


def _validate_event(data: dict[str, Any]) -> None:
    for field in (
        "event_id",
        "event_type",
        "run_id",
        "created_at",
        "actor",
    ):
        _require_str(data, field, "event")
    if "task_id" in data:
        _require_str(data, "task_id", "event")
    if not isinstance(data["payload"], dict):
        raise ValueError("event: payload must be a dict")
    for optional_field in ("attempt_id", "previous_status", "new_status"):
        if optional_field in data and (
            not isinstance(data[optional_field], str) or not data[optional_field]
        ):
            raise ValueError(f"event: {optional_field} must be a non-empty string")


def _validate_task_card(data: dict[str, Any]) -> None:
    for field in (
        "schema_version",
        "run_id",
        "task_id",
        "title",
        "instruction",
        "created_at",
        "created_by",
    ):
        _require_str(data, field, "task_card")
    for field in ("inputs", "constraints", "acceptance", "metadata"):
        if not isinstance(data[field], dict):
            raise ValueError(f"task_card: {field} must be a dict")


def _validate_attempt_result(data: dict[str, Any]) -> None:
    for field in (
        "schema_version",
        "run_id",
        "task_id",
        "attempt_id",
        "created_at",
        "produced_by",
        "summary",
    ):
        _require_str(data, field, "attempt_result")
    if not isinstance(data["outputs"], dict):
        raise ValueError("attempt_result: outputs must be a dict")
    if not isinstance(data["artifacts"], list):
        raise ValueError("attempt_result: artifacts must be a list")
    if not isinstance(data["metrics"], dict):
        raise ValueError("attempt_result: metrics must be a dict")
    if not isinstance(data["metadata"], dict):
        raise ValueError("attempt_result: metadata must be a dict")


_VERIFICATION_OUTCOMES = frozenset({"passed", "failed", "heal_required"})
_VERIFICATION_CHECK_STATUSES = frozenset({"passed", "failed", "skipped"})


def _validate_verification_check(data: dict[str, Any]) -> None:
    _require_str(data, "name", "verification_check")
    status = data["status"]
    if status not in _VERIFICATION_CHECK_STATUSES:
        raise ValueError(
            "verification_check: status must be one of passed, failed, skipped"
        )
    if "message" in data and data["message"] is not None:
        if not isinstance(data["message"], str):
            raise ValueError("verification_check: message must be a string or null")
    if not isinstance(data["metadata"], dict):
        raise ValueError("verification_check: metadata must be a dict")


def _validate_verification_result(data: dict[str, Any]) -> None:
    for field in (
        "schema_version",
        "run_id",
        "task_id",
        "attempt_id",
        "outcome",
        "created_at",
    ):
        _require_str(data, field, "verification_result")
    if data["outcome"] not in _VERIFICATION_OUTCOMES:
        raise ValueError(
            "verification_result: outcome must be one of passed, failed, heal_required"
        )
    summary = data["summary"]
    if summary is not None and not isinstance(summary, str):
        raise ValueError("verification_result: summary must be a string or null")
    if not isinstance(data["checks"], list):
        raise ValueError("verification_result: checks must be a list")
    for check in data["checks"]:
        if not isinstance(check, dict):
            raise ValueError("verification_result: each check must be a dict")
        _validate_verification_check(check)
    if not isinstance(data["metadata"], dict):
        raise ValueError("verification_result: metadata must be a dict")


def _validate_task_completion_record(data: dict[str, Any]) -> None:
    for field in (
        "schema_version",
        "run_id",
        "task_id",
        "attempt_id",
        "created_at",
    ):
        _require_str(data, field, "task_completion_record")
    if "reason" not in data:
        raise ValueError("task_completion_record: missing fields: reason")
    reason = data["reason"]
    if reason is not None and not isinstance(reason, str):
        raise ValueError("task_completion_record: reason must be a string or null")
    if not isinstance(data["metadata"], dict):
        raise ValueError("task_completion_record: metadata must be a dict")


def _validate_run_completion_record(data: dict[str, Any]) -> None:
    for field in ("schema_version", "run_id", "created_at"):
        _require_str(data, field, "run_completion_record")
    if "reason" not in data:
        raise ValueError("run_completion_record: missing fields: reason")
    reason = data["reason"]
    if reason is not None and not isinstance(reason, str):
        raise ValueError("run_completion_record: reason must be a string or null")
    completed_task_ids = data.get("completed_task_ids")
    if not isinstance(completed_task_ids, list) or not completed_task_ids:
        raise ValueError("run_completion_record: completed_task_ids must be a non-empty list")
    seen: set[str] = set()
    for task_id in completed_task_ids:
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(
                "run_completion_record: each completed_task_id must be a non-empty string"
            )
        if task_id in seen:
            raise ValueError("run_completion_record: completed_task_ids must be unique")
        seen.add(task_id)
    if not isinstance(data["metadata"], dict):
        raise ValueError("run_completion_record: metadata must be a dict")


def _validate_run_review_record(data: dict[str, Any]) -> None:
    for field in ("schema_version", "run_id", "decision", "reviewer", "created_at"):
        _require_str(data, field, "run_review_record")
    if data["decision"] not in {"accepted", "rejected", "needs_followup"}:
        raise ValueError(
            "run_review_record: decision must be one of "
            "accepted, rejected, needs_followup"
        )
    if "notes" not in data:
        raise ValueError("run_review_record: missing fields: notes")
    notes = data["notes"]
    if notes is not None and not isinstance(notes, str):
        raise ValueError("run_review_record: notes must be a string or null")
    if not isinstance(data["metadata"], dict):
        raise ValueError("run_review_record: metadata must be a dict")


def _validate_run_followup_plan_record(data: dict[str, Any]) -> None:
    from htr.ids import validate_id

    schema_version = data.get("schema_version")
    if not isinstance(schema_version, int):
        raise ValueError("run_followup_plan_record: schema_version must be an int")
    _require_str(data, "run_id", "run_followup_plan_record")
    if not validate_id(data["run_id"], "run"):
        raise ValueError("run_followup_plan_record: run_id must be a valid run id")
    _require_str(data, "source_review_decision", "run_followup_plan_record")
    if data["source_review_decision"] not in {
        "accepted",
        "rejected",
        "needs_followup",
    }:
        raise ValueError(
            "run_followup_plan_record: source_review_decision must be one of "
            "accepted, rejected, needs_followup"
        )
    _require_str(data, "planner", "run_followup_plan_record")
    _require_str(data, "plan_status", "run_followup_plan_record")
    if data["plan_status"] not in {"open", "cancelled"}:
        raise ValueError(
            "run_followup_plan_record: plan_status must be one of open, cancelled"
        )
    _require_str(data, "summary", "run_followup_plan_record")
    _require_str(data, "created_at", "run_followup_plan_record")
    if "notes" not in data:
        raise ValueError("run_followup_plan_record: missing fields: notes")
    notes = data["notes"]
    if notes is not None and not isinstance(notes, str):
        raise ValueError("run_followup_plan_record: notes must be a string or null")
    followup_items = data.get("followup_items")
    if not isinstance(followup_items, list):
        raise ValueError("run_followup_plan_record: followup_items must be a list")
    for item in followup_items:
        if not isinstance(item, dict):
            raise ValueError(
                "run_followup_plan_record: each followup item must be a dict"
            )
        for field in ("item_id", "title", "kind", "proposed_action"):
            if field not in item or not isinstance(item[field], str) or not item[field]:
                raise ValueError(
                    f"run_followup_plan_record: followup item {field} must be a "
                    "non-empty string"
                )
        if item["kind"] not in {
            "manual_check",
            "rerun_recommended",
            "documentation_update",
            "external_action",
            "other",
        }:
            raise ValueError("run_followup_plan_record: followup item kind is invalid")
        if "rationale" not in item:
            raise ValueError(
                "run_followup_plan_record: followup item missing fields: rationale"
            )
        rationale = item["rationale"]
        if rationale is not None and not isinstance(rationale, str):
            raise ValueError(
                "run_followup_plan_record: followup item rationale must be a string or null"
            )
        if "metadata" not in item or not isinstance(item["metadata"], dict):
            raise ValueError(
                "run_followup_plan_record: followup item metadata must be a dict"
            )
    if not isinstance(data["metadata"], dict):
        raise ValueError("run_followup_plan_record: metadata must be a dict")


def _validate_run_execution_request_record(data: dict[str, Any]) -> None:
    from htr.ids import validate_id

    schema_version = data.get("schema_version")
    if not isinstance(schema_version, int):
        raise ValueError("run_execution_request_record: schema_version must be an int")
    _require_str(data, "run_id", "run_execution_request_record")
    if not validate_id(data["run_id"], "run"):
        raise ValueError("run_execution_request_record: run_id must be a valid run id")
    _require_str(
        data, "source_followup_plan_fingerprint", "run_execution_request_record"
    )
    _require_str(data, "requester", "run_execution_request_record")
    _require_str(data, "request_status", "run_execution_request_record")
    if data["request_status"] not in {"pending", "cancelled"}:
        raise ValueError(
            "run_execution_request_record: request_status must be one of "
            "pending, cancelled"
        )
    _require_str(data, "created_at", "run_execution_request_record")
    if "notes" not in data:
        raise ValueError("run_execution_request_record: missing fields: notes")
    notes = data["notes"]
    if notes is not None and not isinstance(notes, str):
        raise ValueError("run_execution_request_record: notes must be a string or null")
    execution_items = data.get("execution_items")
    if not isinstance(execution_items, list):
        raise ValueError(
            "run_execution_request_record: execution_items must be a list"
        )
    for item in execution_items:
        if not isinstance(item, dict):
            raise ValueError(
                "run_execution_request_record: each execution item must be a dict"
            )
        for field in (
            "item_id",
            "source_followup_item_id",
            "title",
            "execution_kind",
        ):
            if field not in item or not isinstance(item[field], str) or not item[field]:
                raise ValueError(
                    f"run_execution_request_record: execution item {field} must be a "
                    "non-empty string"
                )
        if item["execution_kind"] not in {
            "manual_open_link",
            "rerun_task",
            "regenerate_output",
            "update_documentation",
            "external_action",
            "other",
        }:
            raise ValueError(
                "run_execution_request_record: execution item execution_kind is invalid"
            )
        if "command" not in item or not isinstance(item["command"], dict):
            raise ValueError(
                "run_execution_request_record: execution item command must be a dict"
            )
        if "approval_reason" not in item:
            raise ValueError(
                "run_execution_request_record: execution item missing fields: "
                "approval_reason"
            )
        approval_reason = item["approval_reason"]
        if approval_reason is not None and not isinstance(approval_reason, str):
            raise ValueError(
                "run_execution_request_record: execution item approval_reason must "
                "be a string or null"
            )
        if "metadata" not in item or not isinstance(item["metadata"], dict):
            raise ValueError(
                "run_execution_request_record: execution item metadata must be a dict"
            )
    if not isinstance(data["metadata"], dict):
        raise ValueError("run_execution_request_record: metadata must be a dict")


def _validate_run_execution_result_record(data: dict[str, Any]) -> None:
    from htr.ids import validate_id

    schema_version = data.get("schema_version")
    if not isinstance(schema_version, int):
        raise ValueError("run_execution_result_record: schema_version must be an int")
    _require_str(data, "run_id", "run_execution_result_record")
    if not validate_id(data["run_id"], "run"):
        raise ValueError("run_execution_result_record: run_id must be a valid run id")
    _require_str(
        data, "source_execution_request_fingerprint", "run_execution_result_record"
    )
    _require_str(data, "executor", "run_execution_result_record")
    _require_str(data, "result_status", "run_execution_result_record")
    if data["result_status"] not in {"completed", "partial", "failed"}:
        raise ValueError(
            "run_execution_result_record: result_status must be one of "
            "completed, partial, failed"
        )
    _require_str(data, "created_at", "run_execution_result_record")
    if "notes" not in data:
        raise ValueError("run_execution_result_record: missing fields: notes")
    notes = data["notes"]
    if notes is not None and not isinstance(notes, str):
        raise ValueError("run_execution_result_record: notes must be a string or null")
    item_results = data.get("item_results")
    if not isinstance(item_results, list):
        raise ValueError("run_execution_result_record: item_results must be a list")
    for item in item_results:
        if not isinstance(item, dict):
            raise ValueError(
                "run_execution_result_record: each item result must be a dict"
            )
        for field in (
            "item_id",
            "source_followup_item_id",
            "execution_kind",
            "item_status",
        ):
            if field not in item or not isinstance(item[field], str) or not item[field]:
                raise ValueError(
                    f"run_execution_result_record: item result {field} must be a "
                    "non-empty string"
                )
        if item["execution_kind"] not in {
            "manual_open_link",
            "rerun_task",
            "regenerate_output",
            "update_documentation",
            "external_action",
            "other",
        }:
            raise ValueError(
                "run_execution_result_record: item result execution_kind is invalid"
            )
        if item["item_status"] not in {"completed", "skipped", "failed", "unsupported"}:
            raise ValueError(
                "run_execution_result_record: item result item_status is invalid"
            )
        if "output" not in item or not isinstance(item["output"], dict):
            raise ValueError(
                "run_execution_result_record: item result output must be a dict"
            )
        if "error" not in item:
            raise ValueError(
                "run_execution_result_record: item result missing fields: error"
            )
        error = item["error"]
        if error is not None and not isinstance(error, str):
            raise ValueError(
                "run_execution_result_record: item result error must be a string or null"
            )
        if "metadata" not in item or not isinstance(item["metadata"], dict):
            raise ValueError(
                "run_execution_result_record: item result metadata must be a dict"
            )
    if not isinstance(data["metadata"], dict):
        raise ValueError("run_execution_result_record: metadata must be a dict")


def _validate_run_execution_verification_record(data: dict[str, Any]) -> None:
    from htr.contracts import _validate_execution_verification_decision_consistency
    from htr.ids import validate_id

    schema_version = data.get("schema_version")
    if not isinstance(schema_version, int):
        raise ValueError(
            "run_execution_verification_record: schema_version must be an int"
        )
    _require_str(data, "run_id", "run_execution_verification_record")
    if not validate_id(data["run_id"], "run"):
        raise ValueError(
            "run_execution_verification_record: run_id must be a valid run id"
        )
    _require_str(
        data, "source_execution_result_fingerprint", "run_execution_verification_record"
    )
    _require_str(data, "reviewer", "run_execution_verification_record")
    _require_str(data, "verification_decision", "run_execution_verification_record")
    if data["verification_decision"] not in {"accepted", "rejected", "needs_changes"}:
        raise ValueError(
            "run_execution_verification_record: verification_decision must be one of "
            "accepted, rejected, needs_changes"
        )
    _require_str(data, "created_at", "run_execution_verification_record")
    if "notes" not in data:
        raise ValueError("run_execution_verification_record: missing fields: notes")
    notes = data["notes"]
    if notes is not None and not isinstance(notes, str):
        raise ValueError(
            "run_execution_verification_record: notes must be a string or null"
        )
    item_verifications = data.get("item_verifications")
    if not isinstance(item_verifications, list):
        raise ValueError(
            "run_execution_verification_record: item_verifications must be a list"
        )
    for item in item_verifications:
        if not isinstance(item, dict):
            raise ValueError(
                "run_execution_verification_record: each item verification must be a dict"
            )
        for field in (
            "item_id",
            "source_followup_item_id",
            "execution_kind",
            "item_status",
            "verification_decision",
        ):
            if field not in item or not isinstance(item[field], str) or not item[field]:
                raise ValueError(
                    f"run_execution_verification_record: item verification {field} "
                    "must be a non-empty string"
                )
        if item["execution_kind"] not in {
            "manual_open_link",
            "rerun_task",
            "regenerate_output",
            "update_documentation",
            "external_action",
            "other",
        }:
            raise ValueError(
                "run_execution_verification_record: item verification execution_kind "
                "is invalid"
            )
        if item["item_status"] not in {"completed", "skipped", "failed", "unsupported"}:
            raise ValueError(
                "run_execution_verification_record: item verification item_status "
                "is invalid"
            )
        if item["verification_decision"] not in {
            "accepted",
            "rejected",
            "needs_changes",
            "not_reviewed",
        }:
            raise ValueError(
                "run_execution_verification_record: item verification "
                "verification_decision is invalid"
            )
        if "reviewer_notes" not in item:
            raise ValueError(
                "run_execution_verification_record: item verification missing fields: "
                "reviewer_notes"
            )
        reviewer_notes = item["reviewer_notes"]
        if reviewer_notes is not None and not isinstance(reviewer_notes, str):
            raise ValueError(
                "run_execution_verification_record: item verification reviewer_notes "
                "must be a string or null"
            )
        if "evidence" not in item or not isinstance(item["evidence"], dict):
            raise ValueError(
                "run_execution_verification_record: item verification evidence must "
                "be a dict"
            )
        if "metadata" not in item or not isinstance(item["metadata"], dict):
            raise ValueError(
                "run_execution_verification_record: item verification metadata must "
                "be a dict"
            )
    if not isinstance(data["metadata"], dict):
        raise ValueError("run_execution_verification_record: metadata must be a dict")
    _validate_execution_verification_decision_consistency(data)


def _require_str_or_null(
    data: dict[str, Any], field: str, schema_name: str
) -> None:
    if field not in data:
        raise ValueError(f"{schema_name}: missing fields: {field}")
    value = data[field]
    if value is not None and (not isinstance(value, str) or not value):
        raise ValueError(
            f"{schema_name}: {field} must be a non-empty string or null"
        )


def _validate_run_post_verification_followup_plan_record(data: dict[str, Any]) -> None:
    from htr.contracts import (
        POST_VERIFICATION_FOLLOWUP_KINDS,
        POST_VERIFICATION_FOLLOWUP_PLAN_STATUSES,
        _validate_post_verification_followup_plan_status_consistency,
    )
    from htr.ids import validate_id

    schema_version = data.get("schema_version")
    if not isinstance(schema_version, int):
        raise ValueError(
            "run_post_verification_followup_plan_record: schema_version must be an int"
        )
    _require_str(data, "run_id", "run_post_verification_followup_plan_record")
    if not validate_id(data["run_id"], "run"):
        raise ValueError(
            "run_post_verification_followup_plan_record: run_id must be a valid run id"
        )
    _require_str(
        data,
        "source_execution_result_fingerprint",
        "run_post_verification_followup_plan_record",
    )
    _require_str(
        data,
        "source_execution_verification_fingerprint",
        "run_post_verification_followup_plan_record",
    )
    _require_str(data, "planner", "run_post_verification_followup_plan_record")
    _require_str(data, "plan_status", "run_post_verification_followup_plan_record")
    if data["plan_status"] not in POST_VERIFICATION_FOLLOWUP_PLAN_STATUSES:
        raise ValueError(
            "run_post_verification_followup_plan_record: plan_status must be one of "
            "planned, empty"
        )
    _require_str(data, "created_at", "run_post_verification_followup_plan_record")
    if "notes" not in data:
        raise ValueError(
            "run_post_verification_followup_plan_record: missing fields: notes"
        )
    notes = data["notes"]
    if notes is not None and not isinstance(notes, str):
        raise ValueError(
            "run_post_verification_followup_plan_record: notes must be a string or null"
        )
    followup_items = data.get("followup_items")
    if not isinstance(followup_items, list):
        raise ValueError(
            "run_post_verification_followup_plan_record: followup_items must be a list"
        )
    for item in followup_items:
        if not isinstance(item, dict):
            raise ValueError(
                "run_post_verification_followup_plan_record: each followup item "
                "must be a dict"
            )
        _require_str(
            item, "followup_item_id", "run_post_verification_followup_plan_record"
        )
        for optional_field in (
            "source_execution_item_id",
            "source_followup_item_id",
            "execution_kind",
            "item_status",
        ):
            _require_str_or_null(
                item, optional_field, "run_post_verification_followup_plan_record"
            )
        _require_str(
            item, "verification_decision", "run_post_verification_followup_plan_record"
        )
        _require_str(item, "followup_kind", "run_post_verification_followup_plan_record")
        if item["followup_kind"] not in POST_VERIFICATION_FOLLOWUP_KINDS:
            raise ValueError(
                "run_post_verification_followup_plan_record: followup item "
                "followup_kind is invalid"
            )
        if "instructions" not in item:
            raise ValueError(
                "run_post_verification_followup_plan_record: followup item missing "
                "fields: instructions"
            )
        instructions = item["instructions"]
        if instructions is not None and not isinstance(instructions, str):
            raise ValueError(
                "run_post_verification_followup_plan_record: followup item "
                "instructions must be a string or null"
            )
        if "command" not in item or not isinstance(item["command"], dict):
            raise ValueError(
                "run_post_verification_followup_plan_record: followup item command "
                "must be a dict"
            )
        if "metadata" not in item or not isinstance(item["metadata"], dict):
            raise ValueError(
                "run_post_verification_followup_plan_record: followup item metadata "
                "must be a dict"
            )
    if not isinstance(data["metadata"], dict):
        raise ValueError(
            "run_post_verification_followup_plan_record: metadata must be a dict"
        )
    _validate_post_verification_followup_plan_status_consistency(data)


def _validate_run_post_verification_execution_request_record(
    data: dict[str, Any],
) -> None:
    from htr.contracts import (
        POST_VERIFICATION_EXECUTION_REQUEST_KINDS,
        POST_VERIFICATION_EXECUTION_REQUEST_STATUSES,
        _validate_post_verification_execution_request_status_consistency,
    )
    from htr.ids import validate_id

    schema_version = data.get("schema_version")
    if not isinstance(schema_version, int):
        raise ValueError(
            "run_post_verification_execution_request_record: schema_version must be an int"
        )
    _require_str(data, "run_id", "run_post_verification_execution_request_record")
    if not validate_id(data["run_id"], "run"):
        raise ValueError(
            "run_post_verification_execution_request_record: run_id must be a valid run id"
        )
    _require_str(
        data,
        "source_execution_result_fingerprint",
        "run_post_verification_execution_request_record",
    )
    _require_str(
        data,
        "source_execution_verification_fingerprint",
        "run_post_verification_execution_request_record",
    )
    _require_str(
        data,
        "source_post_verification_followup_plan_fingerprint",
        "run_post_verification_execution_request_record",
    )
    _require_str(data, "requester", "run_post_verification_execution_request_record")
    _require_str(data, "request_status", "run_post_verification_execution_request_record")
    if data["request_status"] not in POST_VERIFICATION_EXECUTION_REQUEST_STATUSES:
        raise ValueError(
            "run_post_verification_execution_request_record: request_status must be one of "
            "requested, empty"
        )
    _require_str(
        data, "created_at", "run_post_verification_execution_request_record"
    )
    if "notes" not in data:
        raise ValueError(
            "run_post_verification_execution_request_record: missing fields: notes"
        )
    notes = data["notes"]
    if notes is not None and not isinstance(notes, str):
        raise ValueError(
            "run_post_verification_execution_request_record: notes must be a string or null"
        )
    request_items = data.get("request_items")
    if not isinstance(request_items, list):
        raise ValueError(
            "run_post_verification_execution_request_record: request_items must be a list"
        )
    for item in request_items:
        if not isinstance(item, dict):
            raise ValueError(
                "run_post_verification_execution_request_record: request_items "
                "entries must be dicts"
            )
        _require_str(
            item, "request_item_id", "run_post_verification_execution_request_record"
        )
        _require_str_or_null(
            item,
            "source_post_verification_followup_item_id",
            "run_post_verification_execution_request_record",
        )
        _require_str_or_null(
            item, "source_execution_item_id", "run_post_verification_execution_request_record"
        )
        _require_str_or_null(
            item, "source_followup_item_id", "run_post_verification_execution_request_record"
        )
        _require_str_or_null(
            item, "execution_kind", "run_post_verification_execution_request_record"
        )
        _require_str_or_null(
            item, "item_status", "run_post_verification_execution_request_record"
        )
        _require_str_or_null(
            item, "verification_decision", "run_post_verification_execution_request_record"
        )
        _require_str_or_null(
            item, "followup_kind", "run_post_verification_execution_request_record"
        )
        _require_str(
            item, "request_kind", "run_post_verification_execution_request_record"
        )
        if item["request_kind"] not in POST_VERIFICATION_EXECUTION_REQUEST_KINDS:
            raise ValueError(
                "run_post_verification_execution_request_record: request_kind must be one of "
                "reopen_link_manually, update_documentation_manually, "
                "prepare_new_execution_request, review_rejected_item, "
                "review_needs_changes_item, review_not_reviewed_item, other"
            )
        _require_str_or_null(
            item, "instructions", "run_post_verification_execution_request_record"
        )
        if "command" not in item or not isinstance(item["command"], dict):
            raise ValueError(
                "run_post_verification_execution_request_record: request item command "
                "must be a dict"
            )
        if "metadata" not in item or not isinstance(item["metadata"], dict):
            raise ValueError(
                "run_post_verification_execution_request_record: request item metadata "
                "must be a dict"
            )
    if not isinstance(data["metadata"], dict):
        raise ValueError(
            "run_post_verification_execution_request_record: metadata must be a dict"
        )
    _validate_post_verification_execution_request_status_consistency(data)


def _validate_run_post_verification_execution_result_record(
    data: dict[str, Any],
) -> None:
    from htr.contracts import (
        POST_VERIFICATION_EXECUTION_RESULT_ITEM_STATUSES,
        POST_VERIFICATION_EXECUTION_RESULT_STATUSES,
        _validate_post_verification_execution_result_status_consistency,
    )
    from htr.ids import validate_id

    schema_version = data.get("schema_version")
    if not isinstance(schema_version, int):
        raise ValueError(
            "run_post_verification_execution_result_record: schema_version must be an int"
        )
    _require_str(data, "run_id", "run_post_verification_execution_result_record")
    if not validate_id(data["run_id"], "run"):
        raise ValueError(
            "run_post_verification_execution_result_record: run_id must be a valid run id"
        )
    _require_str(
        data,
        "source_execution_result_fingerprint",
        "run_post_verification_execution_result_record",
    )
    _require_str(
        data,
        "source_execution_verification_fingerprint",
        "run_post_verification_execution_result_record",
    )
    _require_str(
        data,
        "source_post_verification_followup_plan_fingerprint",
        "run_post_verification_execution_result_record",
    )
    _require_str(
        data,
        "source_post_verification_execution_request_fingerprint",
        "run_post_verification_execution_result_record",
    )
    _require_str(data, "executor", "run_post_verification_execution_result_record")
    _require_str(data, "result_status", "run_post_verification_execution_result_record")
    if data["result_status"] not in POST_VERIFICATION_EXECUTION_RESULT_STATUSES:
        raise ValueError(
            "run_post_verification_execution_result_record: result_status must be one of "
            "completed, failed, partial, empty"
        )
    _require_str(
        data, "created_at", "run_post_verification_execution_result_record"
    )
    if "notes" not in data:
        raise ValueError(
            "run_post_verification_execution_result_record: missing fields: notes"
        )
    notes = data["notes"]
    if notes is not None and not isinstance(notes, str):
        raise ValueError(
            "run_post_verification_execution_result_record: notes must be a string or null"
        )
    result_items = data.get("result_items")
    if not isinstance(result_items, list):
        raise ValueError(
            "run_post_verification_execution_result_record: result_items must be a list"
        )
    for item in result_items:
        if not isinstance(item, dict):
            raise ValueError(
                "run_post_verification_execution_result_record: result_items "
                "entries must be dicts"
            )
        _require_str(
            item, "result_item_id", "run_post_verification_execution_result_record"
        )
        _require_str_or_null(
            item, "source_request_item_id", "run_post_verification_execution_result_record"
        )
        _require_str_or_null(
            item,
            "source_post_verification_followup_item_id",
            "run_post_verification_execution_result_record",
        )
        _require_str_or_null(
            item, "source_execution_item_id", "run_post_verification_execution_result_record"
        )
        _require_str_or_null(
            item, "source_followup_item_id", "run_post_verification_execution_result_record"
        )
        _require_str_or_null(
            item, "request_kind", "run_post_verification_execution_result_record"
        )
        _require_str_or_null(
            item, "execution_kind", "run_post_verification_execution_result_record"
        )
        _require_str_or_null(
            item, "item_status", "run_post_verification_execution_result_record"
        )
        _require_str_or_null(
            item, "verification_decision", "run_post_verification_execution_result_record"
        )
        _require_str_or_null(
            item, "followup_kind", "run_post_verification_execution_result_record"
        )
        _require_str(
            item, "result_item_status", "run_post_verification_execution_result_record"
        )
        if item["result_item_status"] not in POST_VERIFICATION_EXECUTION_RESULT_ITEM_STATUSES:
            raise ValueError(
                "run_post_verification_execution_result_record: result_item_status "
                "must be one of completed, failed, skipped, not_applicable"
            )
        _require_str_or_null(
            item, "outcome", "run_post_verification_execution_result_record"
        )
        if "evidence" not in item or not isinstance(item["evidence"], dict):
            raise ValueError(
                "run_post_verification_execution_result_record: result item evidence "
                "must be a dict"
            )
        if "metadata" not in item or not isinstance(item["metadata"], dict):
            raise ValueError(
                "run_post_verification_execution_result_record: result item metadata "
                "must be a dict"
            )
    if not isinstance(data["metadata"], dict):
        raise ValueError(
            "run_post_verification_execution_result_record: metadata must be a dict"
        )
    _validate_post_verification_execution_result_status_consistency(data)


def _validate_run_post_verification_execution_verification_record(
    data: dict[str, Any],
) -> None:
    from htr.contracts import (
        POST_VERIFICATION_EXECUTION_VERIFICATION_ITEM_DECISIONS,
        POST_VERIFICATION_EXECUTION_VERIFICATION_STATUSES,
        _validate_post_verification_execution_verification_status_consistency,
    )
    from htr.ids import validate_id

    schema_version = data.get("schema_version")
    if not isinstance(schema_version, int):
        raise ValueError(
            "run_post_verification_execution_verification_record: schema_version "
            "must be an int"
        )
    _require_str(
        data, "run_id", "run_post_verification_execution_verification_record"
    )
    if not validate_id(data["run_id"], "run"):
        raise ValueError(
            "run_post_verification_execution_verification_record: run_id must be a "
            "valid run id"
        )
    _require_str(
        data,
        "source_execution_result_fingerprint",
        "run_post_verification_execution_verification_record",
    )
    _require_str(
        data,
        "source_execution_verification_fingerprint",
        "run_post_verification_execution_verification_record",
    )
    _require_str(
        data,
        "source_post_verification_followup_plan_fingerprint",
        "run_post_verification_execution_verification_record",
    )
    _require_str(
        data,
        "source_post_verification_execution_request_fingerprint",
        "run_post_verification_execution_verification_record",
    )
    _require_str(
        data,
        "source_post_verification_execution_result_fingerprint",
        "run_post_verification_execution_verification_record",
    )
    _require_str(
        data, "verifier", "run_post_verification_execution_verification_record"
    )
    _require_str(
        data,
        "verification_status",
        "run_post_verification_execution_verification_record",
    )
    if data["verification_status"] not in POST_VERIFICATION_EXECUTION_VERIFICATION_STATUSES:
        raise ValueError(
            "run_post_verification_execution_verification_record: "
            "verification_status must be one of verified, rejected, needs_changes, "
            "empty"
        )
    _require_str(
        data, "created_at", "run_post_verification_execution_verification_record"
    )
    if "notes" not in data:
        raise ValueError(
            "run_post_verification_execution_verification_record: missing fields: "
            "notes"
        )
    notes = data["notes"]
    if notes is not None and not isinstance(notes, str):
        raise ValueError(
            "run_post_verification_execution_verification_record: notes must be a "
            "string or null"
        )
    verification_items = data.get("verification_items")
    if not isinstance(verification_items, list):
        raise ValueError(
            "run_post_verification_execution_verification_record: "
            "verification_items must be a list"
        )
    for item in verification_items:
        if not isinstance(item, dict):
            raise ValueError(
                "run_post_verification_execution_verification_record: "
                "verification_items entries must be dicts"
            )
        _require_str(
            item,
            "verification_item_id",
            "run_post_verification_execution_verification_record",
        )
        _require_str_or_null(
            item,
            "source_result_item_id",
            "run_post_verification_execution_verification_record",
        )
        _require_str_or_null(
            item,
            "source_request_item_id",
            "run_post_verification_execution_verification_record",
        )
        _require_str_or_null(
            item,
            "source_post_verification_followup_item_id",
            "run_post_verification_execution_verification_record",
        )
        _require_str_or_null(
            item,
            "source_execution_item_id",
            "run_post_verification_execution_verification_record",
        )
        _require_str_or_null(
            item,
            "source_followup_item_id",
            "run_post_verification_execution_verification_record",
        )
        _require_str_or_null(
            item, "request_kind", "run_post_verification_execution_verification_record"
        )
        _require_str_or_null(
            item,
            "execution_kind",
            "run_post_verification_execution_verification_record",
        )
        _require_str_or_null(
            item, "item_status", "run_post_verification_execution_verification_record"
        )
        _require_str_or_null(
            item,
            "verification_decision",
            "run_post_verification_execution_verification_record",
        )
        _require_str_or_null(
            item, "followup_kind", "run_post_verification_execution_verification_record"
        )
        _require_str_or_null(
            item,
            "result_item_status",
            "run_post_verification_execution_verification_record",
        )
        _require_str(
            item,
            "verification_decision_after_result",
            "run_post_verification_execution_verification_record",
        )
        if (
            item["verification_decision_after_result"]
            not in POST_VERIFICATION_EXECUTION_VERIFICATION_ITEM_DECISIONS
        ):
            raise ValueError(
                "run_post_verification_execution_verification_record: "
                "verification_decision_after_result must be one of verified, "
                "rejected, needs_changes, not_applicable"
            )
        _require_str_or_null(
            item, "reason", "run_post_verification_execution_verification_record"
        )
        if "evidence" not in item or not isinstance(item["evidence"], dict):
            raise ValueError(
                "run_post_verification_execution_verification_record: "
                "verification item evidence must be a dict"
            )
        if "metadata" not in item or not isinstance(item["metadata"], dict):
            raise ValueError(
                "run_post_verification_execution_verification_record: "
                "verification item metadata must be a dict"
            )
    if not isinstance(data["metadata"], dict):
        raise ValueError(
            "run_post_verification_execution_verification_record: metadata must be a "
            "dict"
        )
    _validate_post_verification_execution_verification_status_consistency(data)


def _validate_run_final_closure_record(data: dict[str, Any]) -> None:
    from htr.contracts import (
        RUN_FINAL_CLOSURE_ITEM_DECISIONS,
        RUN_FINAL_CLOSURE_STATUSES,
        _validate_run_final_closure_status_consistency,
    )
    from htr.ids import validate_id

    schema_version = data.get("schema_version")
    if not isinstance(schema_version, int):
        raise ValueError("run_final_closure_record: schema_version must be an int")
    _require_str(data, "run_id", "run_final_closure_record")
    if not validate_id(data["run_id"], "run"):
        raise ValueError("run_final_closure_record: run_id must be a valid run id")
    for field in (
        "source_run_completion_fingerprint",
        "source_run_review_fingerprint",
        "source_run_followup_plan_fingerprint",
        "source_run_execution_request_fingerprint",
        "source_run_execution_result_fingerprint",
        "source_run_execution_verification_fingerprint",
        "source_post_verification_followup_plan_fingerprint",
        "source_post_verification_execution_request_fingerprint",
        "source_post_verification_execution_result_fingerprint",
        "source_post_verification_execution_verification_fingerprint",
    ):
        _require_str(data, field, "run_final_closure_record")
    _require_str(data, "closer", "run_final_closure_record")
    _require_str(data, "final_closure_status", "run_final_closure_record")
    if data["final_closure_status"] not in RUN_FINAL_CLOSURE_STATUSES:
        raise ValueError(
            "run_final_closure_record: final_closure_status must be one of "
            "closed_verified, closed_rejected, closed_needs_more_work, closed_no_action"
        )
    _require_str(data, "closure_reason", "run_final_closure_record")
    _require_str(data, "created_at", "run_final_closure_record")
    if "notes" not in data:
        raise ValueError("run_final_closure_record: missing fields: notes")
    notes = data["notes"]
    if notes is not None and not isinstance(notes, str):
        raise ValueError("run_final_closure_record: notes must be a string or null")
    closure_items = data.get("closure_items")
    if not isinstance(closure_items, list):
        raise ValueError("run_final_closure_record: closure_items must be a list")
    for item in closure_items:
        if not isinstance(item, dict):
            raise ValueError(
                "run_final_closure_record: closure_items entries must be dicts"
            )
        _require_str(item, "closure_item_id", "run_final_closure_record")
        _require_str_or_null(
            item,
            "source_post_verification_execution_verification_item_id",
            "run_final_closure_record",
        )
        _require_str_or_null(
            item,
            "source_post_verification_execution_result_item_id",
            "run_final_closure_record",
        )
        _require_str_or_null(
            item,
            "source_post_verification_execution_request_item_id",
            "run_final_closure_record",
        )
        _require_str_or_null(
            item,
            "source_post_verification_followup_item_id",
            "run_final_closure_record",
        )
        _require_str_or_null(
            item, "source_execution_item_id", "run_final_closure_record"
        )
        _require_str_or_null(
            item, "source_followup_item_id", "run_final_closure_record"
        )
        _require_str_or_null(
            item,
            "verification_decision_after_result",
            "run_final_closure_record",
        )
        _require_str(item, "closure_decision", "run_final_closure_record")
        if item["closure_decision"] not in RUN_FINAL_CLOSURE_ITEM_DECISIONS:
            raise ValueError(
                "run_final_closure_record: closure_decision must be one of "
                "accepted, rejected, needs_more_work, no_action"
            )
        _require_str_or_null(item, "reason", "run_final_closure_record")
        if "evidence" not in item or not isinstance(item["evidence"], dict):
            raise ValueError(
                "run_final_closure_record: closure item evidence must be a dict"
            )
        if "metadata" not in item or not isinstance(item["metadata"], dict):
            raise ValueError(
                "run_final_closure_record: closure item metadata must be a dict"
            )
    if not isinstance(data["metadata"], dict):
        raise ValueError("run_final_closure_record: metadata must be a dict")
    _validate_run_final_closure_status_consistency(data)
