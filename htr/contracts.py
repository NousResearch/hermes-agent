"""HTR task card and attempt result contracts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from htr import paths
from htr.execution_lock import begin_run_write, run_mutation_boundary
from htr.io import atomic_write_json, ensure_dir, read_json, sha256_file
from htr.schemas import validate as validate_schema


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def task_card_json_path(
    run_id: str,
    task_id: str,
    base_dir: Path | None = None,
) -> Path:
    """Return the JSON task card path for *task_id*."""
    return paths.task_dir(run_id, task_id, base_dir) / "task_card.json"


def make_task_card(
    *,
    run_id: str,
    task_id: str,
    title: str,
    instruction: str,
    created_by: str,
    inputs: dict[str, Any] | None = None,
    constraints: dict[str, Any] | None = None,
    acceptance: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    schema_version: str = "1",
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a validated task card envelope."""
    task_card: dict[str, Any] = {
        "schema_version": schema_version,
        "run_id": run_id,
        "task_id": task_id,
        "title": title,
        "instruction": instruction,
        "created_at": created_at or _utc_now_iso(),
        "created_by": created_by,
        "inputs": inputs if inputs is not None else {},
        "constraints": constraints if constraints is not None else {},
        "acceptance": acceptance if acceptance is not None else {},
        "metadata": metadata if metadata is not None else {},
    }
    validate_schema(task_card, "task_card")
    return task_card


@run_mutation_boundary
def write_task_card(
    run_id: str,
    task_id: str,
    task_card: dict[str, Any],
    base_dir: Path | None = None,
) -> Path:
    """Atomically write *task_card* to the task workspace."""
    validate_schema(task_card, "task_card")
    if task_card["run_id"] != run_id or task_card["task_id"] != task_id:
        raise ValueError("task_card run_id/task_id do not match write target")
    target = task_card_json_path(run_id, task_id, base_dir)
    begin_run_write()
    ensure_dir(target.parent)
    atomic_write_json(target, task_card)
    return target


def read_task_card(
    run_id: str,
    task_id: str,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    """Read and validate the task card for *task_id*."""
    target = task_card_json_path(run_id, task_id, base_dir)
    task_card = read_json(target)
    validate_schema(task_card, "task_card")
    if task_card["run_id"] != run_id or task_card["task_id"] != task_id:
        raise ValueError("task_card run_id/task_id do not match read target")
    return task_card


def make_attempt_result(
    *,
    run_id: str,
    task_id: str,
    attempt_id: str,
    produced_by: str,
    summary: str,
    outputs: dict[str, Any] | None = None,
    artifacts: list[Any] | None = None,
    metrics: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    schema_version: str = "1",
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a validated attempt result envelope."""
    result: dict[str, Any] = {
        "schema_version": schema_version,
        "run_id": run_id,
        "task_id": task_id,
        "attempt_id": attempt_id,
        "created_at": created_at or _utc_now_iso(),
        "produced_by": produced_by,
        "summary": summary,
        "outputs": outputs if outputs is not None else {},
        "artifacts": artifacts if artifacts is not None else [],
        "metrics": metrics if metrics is not None else {},
        "metadata": metadata if metadata is not None else {},
    }
    validate_schema(result, "attempt_result")
    return result


def result_fingerprint(result: dict[str, Any]) -> str:
    """Return a stable semantic fingerprint for an attempt result envelope."""
    validate_schema(result, "attempt_result")
    return json.dumps(result, sort_keys=True, ensure_ascii=False)


def compute_sha256(path: Path | str) -> str:
    """Return the lowercase hex SHA-256 digest of *path*."""
    return sha256_file(path)


def verification_result_json_path(
    run_id: str,
    task_id: str,
    attempt_id: str,
    base_dir: Path | None = None,
) -> Path:
    """Return the JSON verification result path for *attempt_id*."""
    return (
        paths.verification_dir(run_id, task_id, attempt_id, base_dir)
        / "verification_result.json"
    )


def make_verification_result(
    *,
    run_id: str,
    task_id: str,
    attempt_id: str,
    outcome: str,
    summary: str | None = None,
    checks: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
    schema_version: str = "1",
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a validated manual verification result envelope."""
    verification_result: dict[str, Any] = {
        "schema_version": schema_version,
        "run_id": run_id,
        "task_id": task_id,
        "attempt_id": attempt_id,
        "outcome": outcome,
        "summary": summary,
        "checks": checks if checks is not None else [],
        "metadata": metadata if metadata is not None else {},
        "created_at": created_at or _utc_now_iso(),
    }
    validate_schema(verification_result, "verification_result")
    return verification_result


def verification_fingerprint(verification_result: dict[str, Any]) -> str:
    """Return a stable semantic fingerprint for a verification result envelope."""
    validate_schema(verification_result, "verification_result")
    return json.dumps(
        verification_result,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def task_completion_record_json_path(
    run_id: str,
    task_id: str,
    base_dir: Path | None = None,
) -> Path:
    """Return the JSON task completion record path for *task_id*."""
    return paths.task_dir(run_id, task_id, base_dir) / "task_completion_record.json"


def make_task_completion_record(
    *,
    run_id: str,
    task_id: str,
    attempt_id: str,
    reason: str | None = None,
    metadata: dict[str, Any] | None = None,
    schema_version: str = "1",
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a validated manual task completion record envelope."""
    completion_record: dict[str, Any] = {
        "schema_version": schema_version,
        "run_id": run_id,
        "task_id": task_id,
        "attempt_id": attempt_id,
        "reason": reason,
        "metadata": metadata if metadata is not None else {},
        "created_at": created_at or _utc_now_iso(),
    }
    validate_schema(completion_record, "task_completion_record")
    return completion_record


def task_completion_fingerprint(completion_record: dict[str, Any]) -> str:
    """Return a stable semantic fingerprint for a task completion record envelope."""
    validate_schema(completion_record, "task_completion_record")
    return json.dumps(
        completion_record,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def run_completion_record_json_path(
    run_id: str,
    base_dir: Path | None = None,
) -> Path:
    """Return the JSON run completion record path for *run_id*."""
    return paths.run_root(run_id, base_dir) / "run_completion_record.json"


def make_run_completion_record(
    *,
    run_id: str,
    completed_task_ids: list[str],
    reason: str | None = None,
    metadata: dict[str, Any] | None = None,
    schema_version: str = "1",
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a validated manual run completion record envelope."""
    completion_record: dict[str, Any] = {
        "schema_version": schema_version,
        "run_id": run_id,
        "completed_task_ids": list(completed_task_ids),
        "reason": reason,
        "metadata": metadata if metadata is not None else {},
        "created_at": created_at or _utc_now_iso(),
    }
    validate_schema(completion_record, "run_completion_record")
    return completion_record


def run_completion_fingerprint(completion_record: dict[str, Any]) -> str:
    """Return a stable semantic fingerprint for a run completion record envelope."""
    validate_schema(completion_record, "run_completion_record")
    return json.dumps(
        completion_record,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


RUN_REVIEW_ACCEPTED = "accepted"
RUN_REVIEW_REJECTED = "rejected"
RUN_REVIEW_NEEDS_FOLLOWUP = "needs_followup"

RUN_REVIEW_DECISIONS: frozenset[str] = frozenset(
    {
        RUN_REVIEW_ACCEPTED,
        RUN_REVIEW_REJECTED,
        RUN_REVIEW_NEEDS_FOLLOWUP,
    }
)


def run_review_record_json_path(
    run_id: str,
    base_dir: Path | None = None,
) -> Path:
    """Return the JSON run review record path for *run_id*."""
    return paths.run_root(run_id, base_dir) / "run_review_record.json"


def make_run_review_record(
    *,
    run_id: str,
    decision: str,
    reviewer: str = "human",
    notes: str | None = None,
    metadata: dict[str, Any] | None = None,
    schema_version: str = "1",
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a validated manual run review record envelope."""
    review_record: dict[str, Any] = {
        "schema_version": schema_version,
        "run_id": run_id,
        "decision": decision,
        "reviewer": reviewer,
        "notes": notes,
        "metadata": metadata if metadata is not None else {},
        "created_at": created_at or _utc_now_iso(),
    }
    validate_schema(review_record, "run_review_record")
    return review_record


def run_review_fingerprint(review_record: dict[str, Any]) -> str:
    """Return a stable semantic fingerprint for a run review record envelope."""
    validate_schema(review_record, "run_review_record")
    return json.dumps(
        review_record,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


FOLLOWUP_PLAN_OPEN = "open"
FOLLOWUP_PLAN_CANCELLED = "cancelled"

FOLLOWUP_PLAN_STATUSES: frozenset[str] = frozenset(
    {
        FOLLOWUP_PLAN_OPEN,
        FOLLOWUP_PLAN_CANCELLED,
    }
)

FOLLOWUP_ITEM_KINDS: frozenset[str] = frozenset(
    {
        "manual_check",
        "rerun_recommended",
        "documentation_update",
        "external_action",
        "other",
    }
)


def run_followup_plan_record_json_path(
    run_id: str,
    base_dir: Path | None = None,
) -> Path:
    """Return the JSON run follow-up plan record path for *run_id*."""
    return paths.run_root(run_id, base_dir) / "run_followup_plan_record.json"


def _normalize_followup_items(
    followup_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in followup_items:
        normalized.append(
            {
                "item_id": item["item_id"],
                "title": item["title"],
                "kind": item["kind"],
                "rationale": item.get("rationale"),
                "proposed_action": item["proposed_action"],
                "metadata": item["metadata"] if item.get("metadata") is not None else {},
            }
        )
    return normalized


def make_run_followup_plan_record(
    *,
    run_id: str,
    source_review_decision: str,
    summary: str,
    followup_items: list[dict[str, Any]],
    planner: str = "human",
    plan_status: str = FOLLOWUP_PLAN_OPEN,
    notes: str | None = None,
    metadata: dict[str, Any] | None = None,
    schema_version: int = 1,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a validated review-gated run follow-up plan record envelope."""
    followup_plan_record: dict[str, Any] = {
        "schema_version": schema_version,
        "run_id": run_id,
        "source_review_decision": source_review_decision,
        "planner": planner,
        "plan_status": plan_status,
        "summary": summary,
        "followup_items": _normalize_followup_items(followup_items),
        "notes": notes,
        "metadata": metadata if metadata is not None else {},
        "created_at": created_at or _utc_now_iso(),
    }
    validate_schema(followup_plan_record, "run_followup_plan_record")
    return followup_plan_record


def run_followup_plan_fingerprint(followup_plan_record: dict[str, Any]) -> str:
    """Return a stable semantic fingerprint for a run follow-up plan record."""
    validate_schema(followup_plan_record, "run_followup_plan_record")
    return json.dumps(
        followup_plan_record,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


EXECUTION_REQUEST_PENDING = "pending"
EXECUTION_REQUEST_CANCELLED = "cancelled"

EXECUTION_REQUEST_STATUSES: frozenset[str] = frozenset(
    {
        EXECUTION_REQUEST_PENDING,
        EXECUTION_REQUEST_CANCELLED,
    }
)

EXECUTION_KINDS: frozenset[str] = frozenset(
    {
        "manual_open_link",
        "rerun_task",
        "regenerate_output",
        "update_documentation",
        "external_action",
        "other",
    }
)


def run_execution_request_record_json_path(
    run_id: str,
    base_dir: Path | None = None,
) -> Path:
    """Return the JSON run execution request record path for *run_id*."""
    return paths.run_root(run_id, base_dir) / "run_execution_request_record.json"


def _normalize_execution_items(
    execution_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in execution_items:
        normalized.append(
            {
                "item_id": item["item_id"],
                "source_followup_item_id": item["source_followup_item_id"],
                "title": item["title"],
                "execution_kind": item["execution_kind"],
                "command": item["command"],
                "approval_reason": item.get("approval_reason"),
                "metadata": item["metadata"] if item.get("metadata") is not None else {},
            }
        )
    return normalized


def make_run_execution_request_record(
    *,
    run_id: str,
    source_followup_plan_fingerprint: str,
    execution_items: list[dict[str, Any]],
    requester: str = "human",
    request_status: str = EXECUTION_REQUEST_PENDING,
    notes: str | None = None,
    metadata: dict[str, Any] | None = None,
    schema_version: int = 1,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a validated review-gated run execution request record envelope."""
    execution_request_record: dict[str, Any] = {
        "schema_version": schema_version,
        "run_id": run_id,
        "source_followup_plan_fingerprint": source_followup_plan_fingerprint,
        "requester": requester,
        "request_status": request_status,
        "execution_items": _normalize_execution_items(execution_items),
        "notes": notes,
        "metadata": metadata if metadata is not None else {},
        "created_at": created_at or _utc_now_iso(),
    }
    validate_schema(execution_request_record, "run_execution_request_record")
    return execution_request_record


def run_execution_request_fingerprint(execution_request_record: dict[str, Any]) -> str:
    """Return a stable semantic fingerprint for a run execution request record."""
    validate_schema(execution_request_record, "run_execution_request_record")
    return json.dumps(
        execution_request_record,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


EXECUTION_RESULT_COMPLETED = "completed"
EXECUTION_RESULT_PARTIAL = "partial"
EXECUTION_RESULT_FAILED = "failed"

EXECUTION_RESULT_STATUSES: frozenset[str] = frozenset(
    {
        EXECUTION_RESULT_COMPLETED,
        EXECUTION_RESULT_PARTIAL,
        EXECUTION_RESULT_FAILED,
    }
)

EXECUTION_ITEM_COMPLETED = "completed"
EXECUTION_ITEM_SKIPPED = "skipped"
EXECUTION_ITEM_FAILED = "failed"
EXECUTION_ITEM_UNSUPPORTED = "unsupported"

EXECUTION_ITEM_STATUSES: frozenset[str] = frozenset(
    {
        EXECUTION_ITEM_COMPLETED,
        EXECUTION_ITEM_SKIPPED,
        EXECUTION_ITEM_FAILED,
        EXECUTION_ITEM_UNSUPPORTED,
    }
)


def run_execution_result_record_json_path(
    run_id: str,
    base_dir: Path | None = None,
) -> Path:
    """Return the JSON run execution result record path for *run_id*."""
    return paths.run_root(run_id, base_dir) / "run_execution_result_record.json"


def _normalize_item_results(
    item_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in item_results:
        normalized.append(
            {
                "item_id": item["item_id"],
                "source_followup_item_id": item["source_followup_item_id"],
                "execution_kind": item["execution_kind"],
                "item_status": item["item_status"],
                "output": item["output"],
                "error": item.get("error"),
                "metadata": item["metadata"] if item.get("metadata") is not None else {},
            }
        )
    return normalized


def process_execution_item(item: dict[str, Any]) -> dict[str, Any]:
    """Process one approved execution item without external side effects."""
    kind = item["execution_kind"]
    command = dict(item["command"])
    base = {
        "item_id": item["item_id"],
        "source_followup_item_id": item["source_followup_item_id"],
        "execution_kind": kind,
        "metadata": item["metadata"] if item.get("metadata") is not None else {},
    }

    if kind == "manual_open_link":
        return {
            **base,
            "item_status": EXECUTION_ITEM_SKIPPED,
            "output": {
                "human_action_required": True,
                "command": command,
            },
            "error": None,
        }

    if kind == "update_documentation":
        return {
            **base,
            "item_status": EXECUTION_ITEM_SKIPPED,
            "output": {
                "proposed_update": command,
                "command": command,
            },
            "error": None,
        }

    if kind == "other":
        if command.get("no_op") is True:
            return {
                **base,
                "item_status": EXECUTION_ITEM_COMPLETED,
                "output": {
                    "no_op_completed": True,
                    "command": command,
                },
                "error": None,
            }
        return {
            **base,
            "item_status": EXECUTION_ITEM_UNSUPPORTED,
            "output": {"command": command},
            "error": "unsupported execution kind handling for other",
        }

    if kind in {"rerun_task", "regenerate_output", "external_action"}:
        return {
            **base,
            "item_status": EXECUTION_ITEM_UNSUPPORTED,
            "output": {"command": command},
            "error": f"{kind} is not supported in Task 10",
        }

    return {
        **base,
        "item_status": EXECUTION_ITEM_UNSUPPORTED,
        "output": {"command": command},
        "error": f"unsupported execution kind {kind!r}",
    }


def process_execution_items(
    execution_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Process approved execution items and return per-item results."""
    return [process_execution_item(item) for item in execution_items]


def compute_execution_result_status(item_results: list[dict[str, Any]]) -> str:
    """Derive aggregate execution result status from item results."""
    if not item_results:
        return EXECUTION_RESULT_FAILED
    completed = sum(
        1
        for item in item_results
        if item["item_status"] == EXECUTION_ITEM_COMPLETED
    )
    if completed == len(item_results):
        return EXECUTION_RESULT_COMPLETED
    if completed > 0:
        return EXECUTION_RESULT_PARTIAL
    return EXECUTION_RESULT_FAILED


def make_run_execution_result_record(
    *,
    run_id: str,
    source_execution_request_fingerprint: str,
    item_results: list[dict[str, Any]],
    executor: str = "human",
    result_status: str | None = None,
    notes: str | None = None,
    metadata: dict[str, Any] | None = None,
    schema_version: int = 1,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a validated controlled run execution result record envelope."""
    normalized_item_results = _normalize_item_results(item_results)
    resolved_result_status = result_status or compute_execution_result_status(
        normalized_item_results
    )
    execution_result_record: dict[str, Any] = {
        "schema_version": schema_version,
        "run_id": run_id,
        "source_execution_request_fingerprint": source_execution_request_fingerprint,
        "executor": executor,
        "result_status": resolved_result_status,
        "item_results": normalized_item_results,
        "notes": notes,
        "metadata": metadata if metadata is not None else {},
        "created_at": created_at or _utc_now_iso(),
    }
    validate_schema(execution_result_record, "run_execution_result_record")
    return execution_result_record


def run_execution_result_fingerprint(execution_result_record: dict[str, Any]) -> str:
    """Return a stable semantic fingerprint for a run execution result record."""
    validate_schema(execution_result_record, "run_execution_result_record")
    return json.dumps(
        execution_result_record,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


EXECUTION_VERIFICATION_ACCEPTED = "accepted"
EXECUTION_VERIFICATION_REJECTED = "rejected"
EXECUTION_VERIFICATION_NEEDS_CHANGES = "needs_changes"

EXECUTION_VERIFICATION_DECISIONS: frozenset[str] = frozenset(
    {
        EXECUTION_VERIFICATION_ACCEPTED,
        EXECUTION_VERIFICATION_REJECTED,
        EXECUTION_VERIFICATION_NEEDS_CHANGES,
    }
)

EXECUTION_ITEM_VERIFICATION_ACCEPTED = "accepted"
EXECUTION_ITEM_VERIFICATION_REJECTED = "rejected"
EXECUTION_ITEM_VERIFICATION_NEEDS_CHANGES = "needs_changes"
EXECUTION_ITEM_VERIFICATION_NOT_REVIEWED = "not_reviewed"

EXECUTION_ITEM_VERIFICATION_DECISIONS: frozenset[str] = frozenset(
    {
        EXECUTION_ITEM_VERIFICATION_ACCEPTED,
        EXECUTION_ITEM_VERIFICATION_REJECTED,
        EXECUTION_ITEM_VERIFICATION_NEEDS_CHANGES,
        EXECUTION_ITEM_VERIFICATION_NOT_REVIEWED,
    }
)


def run_execution_verification_record_json_path(
    run_id: str,
    base_dir: Path | None = None,
) -> Path:
    """Return the JSON run execution verification record path for *run_id*."""
    return paths.run_root(run_id, base_dir) / "run_execution_verification_record.json"


def _normalize_item_verifications(
    item_verifications: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in item_verifications:
        normalized.append(
            {
                "item_id": item["item_id"],
                "source_followup_item_id": item["source_followup_item_id"],
                "execution_kind": item["execution_kind"],
                "item_status": item["item_status"],
                "verification_decision": item["verification_decision"],
                "reviewer_notes": item.get("reviewer_notes"),
                "evidence": item["evidence"] if item.get("evidence") is not None else {},
                "metadata": item["metadata"] if item.get("metadata") is not None else {},
            }
        )
    return normalized


def _validate_execution_verification_decision_consistency(
    verification_record: dict[str, Any],
) -> None:
    """Validate run-level and item-level verification decision consistency."""
    decision = verification_record["verification_decision"]
    item_decisions = [
        item["verification_decision"]
        for item in verification_record["item_verifications"]
    ]

    if decision == EXECUTION_VERIFICATION_ACCEPTED:
        if item_decisions and not all(
            d == EXECUTION_ITEM_VERIFICATION_ACCEPTED for d in item_decisions
        ):
            raise ValueError(
                "run_execution_verification_record: accepted decision requires all "
                "item_verifications to be accepted"
            )
    elif decision == EXECUTION_VERIFICATION_REJECTED:
        if EXECUTION_ITEM_VERIFICATION_REJECTED not in item_decisions:
            raise ValueError(
                "run_execution_verification_record: rejected decision requires at "
                "least one item_verification rejected"
            )
    elif decision == EXECUTION_VERIFICATION_NEEDS_CHANGES:
        if EXECUTION_ITEM_VERIFICATION_NEEDS_CHANGES not in item_decisions:
            raise ValueError(
                "run_execution_verification_record: needs_changes decision requires "
                "at least one item_verification needs_changes"
            )

    if any(
        d == EXECUTION_ITEM_VERIFICATION_NOT_REVIEWED for d in item_decisions
    ) and decision not in {
        EXECUTION_VERIFICATION_REJECTED,
        EXECUTION_VERIFICATION_NEEDS_CHANGES,
    }:
        raise ValueError(
            "run_execution_verification_record: not_reviewed items are allowed only "
            "when verification_decision is rejected or needs_changes"
        )


def validate_item_verifications_correspond_to_results(
    item_verifications: list[dict[str, Any]],
    item_results: list[dict[str, Any]],
    *,
    allow_partial: bool = False,
) -> None:
    """Ensure item verifications align with execution result item_results."""
    result_by_id = {item["item_id"]: item for item in item_results}
    verification_by_id = {item["item_id"]: item for item in item_verifications}

    if not allow_partial:
        if set(result_by_id) != set(verification_by_id):
            raise ValueError(
                "item_verifications item_id set does not match item_results"
            )

    for item_id, verification in verification_by_id.items():
        if item_id not in result_by_id:
            raise ValueError(
                f"item_verifications contains unknown item_id {item_id!r}"
            )
        result = result_by_id[item_id]
        for field in ("source_followup_item_id", "execution_kind", "item_status"):
            if verification[field] != result[field]:
                raise ValueError(
                    f"item_verifications {field} does not match item_results for "
                    f"{item_id!r}"
                )

    if not allow_partial:
        missing = set(result_by_id) - set(verification_by_id)
        if missing:
            raise ValueError(
                "item_verifications missing item_id(s): "
                + ", ".join(sorted(missing))
            )


def make_run_execution_verification_record(
    *,
    run_id: str,
    source_execution_result_fingerprint: str,
    item_verifications: list[dict[str, Any]],
    reviewer: str = "human",
    verification_decision: str = EXECUTION_VERIFICATION_ACCEPTED,
    notes: str | None = None,
    metadata: dict[str, Any] | None = None,
    schema_version: int = 1,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a validated manual run execution verification record envelope."""
    verification_record: dict[str, Any] = {
        "schema_version": schema_version,
        "run_id": run_id,
        "source_execution_result_fingerprint": source_execution_result_fingerprint,
        "reviewer": reviewer,
        "verification_decision": verification_decision,
        "item_verifications": _normalize_item_verifications(item_verifications),
        "notes": notes,
        "metadata": metadata if metadata is not None else {},
        "created_at": created_at or _utc_now_iso(),
    }
    validate_schema(verification_record, "run_execution_verification_record")
    return verification_record


def run_execution_verification_fingerprint(
    verification_record: dict[str, Any],
) -> str:
    """Return a stable semantic fingerprint for a run execution verification record."""
    validate_schema(verification_record, "run_execution_verification_record")
    return json.dumps(
        verification_record,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


POST_VERIFICATION_FOLLOWUP_PLAN_STATUS_PLANNED = "planned"
POST_VERIFICATION_FOLLOWUP_PLAN_STATUS_EMPTY = "empty"

POST_VERIFICATION_FOLLOWUP_PLAN_STATUSES: frozenset[str] = frozenset(
    {
        POST_VERIFICATION_FOLLOWUP_PLAN_STATUS_PLANNED,
        POST_VERIFICATION_FOLLOWUP_PLAN_STATUS_EMPTY,
    }
)

POST_VERIFICATION_FOLLOWUP_KIND_REVIEW_REJECTED_ITEM = "review_rejected_item"
POST_VERIFICATION_FOLLOWUP_KIND_REVIEW_NEEDS_CHANGES_ITEM = "review_needs_changes_item"
POST_VERIFICATION_FOLLOWUP_KIND_REVIEW_NOT_REVIEWED_ITEM = "review_not_reviewed_item"
POST_VERIFICATION_FOLLOWUP_KIND_REOPEN_LINK_MANUALLY = "reopen_link_manually"
POST_VERIFICATION_FOLLOWUP_KIND_UPDATE_DOCUMENTATION_MANUALLY = (
    "update_documentation_manually"
)
POST_VERIFICATION_FOLLOWUP_KIND_PREPARE_NEW_EXECUTION_REQUEST = (
    "prepare_new_execution_request"
)
POST_VERIFICATION_FOLLOWUP_KIND_OTHER = "other"

POST_VERIFICATION_FOLLOWUP_KINDS: frozenset[str] = frozenset(
    {
        POST_VERIFICATION_FOLLOWUP_KIND_REVIEW_REJECTED_ITEM,
        POST_VERIFICATION_FOLLOWUP_KIND_REVIEW_NEEDS_CHANGES_ITEM,
        POST_VERIFICATION_FOLLOWUP_KIND_REVIEW_NOT_REVIEWED_ITEM,
        POST_VERIFICATION_FOLLOWUP_KIND_REOPEN_LINK_MANUALLY,
        POST_VERIFICATION_FOLLOWUP_KIND_UPDATE_DOCUMENTATION_MANUALLY,
        POST_VERIFICATION_FOLLOWUP_KIND_PREPARE_NEW_EXECUTION_REQUEST,
        POST_VERIFICATION_FOLLOWUP_KIND_OTHER,
    }
)

_VERIFICATION_TO_FOLLOWUP_KIND: dict[str, str] = {
    EXECUTION_ITEM_VERIFICATION_REJECTED: POST_VERIFICATION_FOLLOWUP_KIND_REVIEW_REJECTED_ITEM,
    EXECUTION_ITEM_VERIFICATION_NEEDS_CHANGES: POST_VERIFICATION_FOLLOWUP_KIND_REVIEW_NEEDS_CHANGES_ITEM,
    EXECUTION_ITEM_VERIFICATION_NOT_REVIEWED: POST_VERIFICATION_FOLLOWUP_KIND_REVIEW_NOT_REVIEWED_ITEM,
}


def run_post_verification_followup_plan_record_json_path(
    run_id: str,
    base_dir: Path | None = None,
) -> Path:
    """Return the JSON post-verification follow-up plan record path for *run_id*."""
    return (
        paths.run_root(run_id, base_dir)
        / "run_post_verification_followup_plan_record.json"
    )


def _normalize_post_verification_followup_items(
    followup_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in followup_items:
        normalized.append(
            {
                "followup_item_id": item["followup_item_id"],
                "source_execution_item_id": item.get("source_execution_item_id"),
                "source_followup_item_id": item.get("source_followup_item_id"),
                "execution_kind": item.get("execution_kind"),
                "item_status": item.get("item_status"),
                "verification_decision": item["verification_decision"],
                "followup_kind": item["followup_kind"],
                "instructions": item.get("instructions"),
                "command": item["command"] if item.get("command") is not None else {},
                "metadata": item["metadata"] if item.get("metadata") is not None else {},
            }
        )
    return normalized


def _validate_post_verification_followup_plan_status_consistency(
    plan_record: dict[str, Any],
) -> None:
    plan_status = plan_record["plan_status"]
    followup_items = plan_record["followup_items"]
    if plan_status == POST_VERIFICATION_FOLLOWUP_PLAN_STATUS_EMPTY:
        if followup_items:
            raise ValueError(
                "run_post_verification_followup_plan_record: empty plan_status "
                "requires followup_items to be empty"
            )
    elif plan_status == POST_VERIFICATION_FOLLOWUP_PLAN_STATUS_PLANNED:
        if not followup_items:
            raise ValueError(
                "run_post_verification_followup_plan_record: planned plan_status "
                "requires non-empty followup_items"
            )


def derive_post_verification_followup_items(
    execution_result_record: dict[str, Any],
    verification_record: dict[str, Any],
) -> list[dict[str, Any]]:
    """Derive deterministic follow-up items from execution verification items."""
    validate_schema(execution_result_record, "run_execution_result_record")
    validate_schema(verification_record, "run_execution_verification_record")
    if verification_record["verification_decision"] == EXECUTION_VERIFICATION_ACCEPTED:
        return []

    followup_items: list[dict[str, Any]] = []
    for item_verification in verification_record["item_verifications"]:
        decision = item_verification["verification_decision"]
        followup_kind = _VERIFICATION_TO_FOLLOWUP_KIND.get(decision)
        if followup_kind is None:
            continue
        followup_items.append(
            {
                "followup_item_id": f"pvfp-{item_verification['item_id']}",
                "source_execution_item_id": item_verification["item_id"],
                "source_followup_item_id": item_verification["source_followup_item_id"],
                "execution_kind": item_verification["execution_kind"],
                "item_status": item_verification["item_status"],
                "verification_decision": decision,
                "followup_kind": followup_kind,
                "instructions": None,
                "command": {},
                "metadata": {},
            }
        )
    return followup_items


def validate_post_verification_followup_items_correspond(
    followup_items: list[dict[str, Any]],
    execution_result_record: dict[str, Any],
    verification_record: dict[str, Any],
) -> None:
    """Ensure follow-up items align with execution result and verification items."""
    result_by_id = {
        item["item_id"]: item for item in execution_result_record["item_results"]
    }
    verification_by_id = {
        item["item_id"]: item for item in verification_record["item_verifications"]
    }

    for followup_item in followup_items:
        source_execution_item_id = followup_item.get("source_execution_item_id")
        if source_execution_item_id is None:
            continue
        if source_execution_item_id not in result_by_id:
            raise ValueError(
                f"followup item references unknown source_execution_item_id "
                f"{source_execution_item_id!r}"
            )
        if source_execution_item_id not in verification_by_id:
            raise ValueError(
                f"followup item references unknown verification item_id "
                f"{source_execution_item_id!r}"
            )
        result_item = result_by_id[source_execution_item_id]
        verification_item = verification_by_id[source_execution_item_id]
        for field in ("source_followup_item_id", "execution_kind", "item_status"):
            expected = result_item[field]
            actual = followup_item.get(field)
            if actual is not None and actual != expected:
                raise ValueError(
                    f"followup item {field} does not match execution result for "
                    f"{source_execution_item_id!r}"
                )
        if (
            followup_item["verification_decision"]
            != verification_item["verification_decision"]
        ):
            raise ValueError(
                f"followup item verification_decision does not match verification "
                f"record for {source_execution_item_id!r}"
            )


def make_run_post_verification_followup_plan_record(
    *,
    run_id: str,
    source_execution_result_fingerprint: str,
    source_execution_verification_fingerprint: str,
    followup_items: list[dict[str, Any]],
    planner: str = "human",
    plan_status: str | None = None,
    notes: str | None = None,
    metadata: dict[str, Any] | None = None,
    schema_version: int = 1,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a validated post-verification follow-up plan record envelope."""
    normalized_items = _normalize_post_verification_followup_items(followup_items)
    resolved_plan_status = plan_status
    if resolved_plan_status is None:
        resolved_plan_status = (
            POST_VERIFICATION_FOLLOWUP_PLAN_STATUS_EMPTY
            if not normalized_items
            else POST_VERIFICATION_FOLLOWUP_PLAN_STATUS_PLANNED
        )
    plan_record: dict[str, Any] = {
        "schema_version": schema_version,
        "run_id": run_id,
        "source_execution_result_fingerprint": source_execution_result_fingerprint,
        "source_execution_verification_fingerprint": (
            source_execution_verification_fingerprint
        ),
        "planner": planner,
        "plan_status": resolved_plan_status,
        "followup_items": normalized_items,
        "notes": notes,
        "metadata": metadata if metadata is not None else {},
        "created_at": created_at or _utc_now_iso(),
    }
    validate_schema(plan_record, "run_post_verification_followup_plan_record")
    return plan_record


def run_post_verification_followup_plan_fingerprint(
    plan_record: dict[str, Any],
) -> str:
    """Return a stable semantic fingerprint for a post-verification follow-up plan."""
    validate_schema(plan_record, "run_post_verification_followup_plan_record")
    return json.dumps(
        plan_record,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


POST_VERIFICATION_EXECUTION_REQUEST_STATUS_REQUESTED = "requested"
POST_VERIFICATION_EXECUTION_REQUEST_STATUS_EMPTY = "empty"

POST_VERIFICATION_EXECUTION_REQUEST_STATUSES: frozenset[str] = frozenset(
    {
        POST_VERIFICATION_EXECUTION_REQUEST_STATUS_REQUESTED,
        POST_VERIFICATION_EXECUTION_REQUEST_STATUS_EMPTY,
    }
)

POST_VERIFICATION_EXECUTION_REQUEST_KIND_REOPEN_LINK_MANUALLY = "reopen_link_manually"
POST_VERIFICATION_EXECUTION_REQUEST_KIND_UPDATE_DOCUMENTATION_MANUALLY = (
    "update_documentation_manually"
)
POST_VERIFICATION_EXECUTION_REQUEST_KIND_PREPARE_NEW_EXECUTION_REQUEST = (
    "prepare_new_execution_request"
)
POST_VERIFICATION_EXECUTION_REQUEST_KIND_REVIEW_REJECTED_ITEM = "review_rejected_item"
POST_VERIFICATION_EXECUTION_REQUEST_KIND_REVIEW_NEEDS_CHANGES_ITEM = (
    "review_needs_changes_item"
)
POST_VERIFICATION_EXECUTION_REQUEST_KIND_REVIEW_NOT_REVIEWED_ITEM = (
    "review_not_reviewed_item"
)
POST_VERIFICATION_EXECUTION_REQUEST_KIND_OTHER = "other"

POST_VERIFICATION_EXECUTION_REQUEST_KINDS: frozenset[str] = frozenset(
    {
        POST_VERIFICATION_EXECUTION_REQUEST_KIND_REOPEN_LINK_MANUALLY,
        POST_VERIFICATION_EXECUTION_REQUEST_KIND_UPDATE_DOCUMENTATION_MANUALLY,
        POST_VERIFICATION_EXECUTION_REQUEST_KIND_PREPARE_NEW_EXECUTION_REQUEST,
        POST_VERIFICATION_EXECUTION_REQUEST_KIND_REVIEW_REJECTED_ITEM,
        POST_VERIFICATION_EXECUTION_REQUEST_KIND_REVIEW_NEEDS_CHANGES_ITEM,
        POST_VERIFICATION_EXECUTION_REQUEST_KIND_REVIEW_NOT_REVIEWED_ITEM,
        POST_VERIFICATION_EXECUTION_REQUEST_KIND_OTHER,
    }
)

_FOLLOWUP_TO_EXECUTION_REQUEST_KIND: dict[str, str] = {
    POST_VERIFICATION_FOLLOWUP_KIND_REOPEN_LINK_MANUALLY: (
        POST_VERIFICATION_EXECUTION_REQUEST_KIND_REOPEN_LINK_MANUALLY
    ),
    POST_VERIFICATION_FOLLOWUP_KIND_UPDATE_DOCUMENTATION_MANUALLY: (
        POST_VERIFICATION_EXECUTION_REQUEST_KIND_UPDATE_DOCUMENTATION_MANUALLY
    ),
    POST_VERIFICATION_FOLLOWUP_KIND_PREPARE_NEW_EXECUTION_REQUEST: (
        POST_VERIFICATION_EXECUTION_REQUEST_KIND_PREPARE_NEW_EXECUTION_REQUEST
    ),
    POST_VERIFICATION_FOLLOWUP_KIND_REVIEW_REJECTED_ITEM: (
        POST_VERIFICATION_EXECUTION_REQUEST_KIND_REVIEW_REJECTED_ITEM
    ),
    POST_VERIFICATION_FOLLOWUP_KIND_REVIEW_NEEDS_CHANGES_ITEM: (
        POST_VERIFICATION_EXECUTION_REQUEST_KIND_REVIEW_NEEDS_CHANGES_ITEM
    ),
    POST_VERIFICATION_FOLLOWUP_KIND_REVIEW_NOT_REVIEWED_ITEM: (
        POST_VERIFICATION_EXECUTION_REQUEST_KIND_REVIEW_NOT_REVIEWED_ITEM
    ),
    POST_VERIFICATION_FOLLOWUP_KIND_OTHER: POST_VERIFICATION_EXECUTION_REQUEST_KIND_OTHER,
}


def run_post_verification_execution_request_record_json_path(
    run_id: str,
    base_dir: Path | None = None,
) -> Path:
    """Return the JSON post-verification execution request record path for *run_id*."""
    return (
        paths.run_root(run_id, base_dir)
        / "run_post_verification_execution_request_record.json"
    )


def _normalize_post_verification_execution_request_items(
    request_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in request_items:
        normalized.append(
            {
                "request_item_id": item["request_item_id"],
                "source_post_verification_followup_item_id": item.get(
                    "source_post_verification_followup_item_id"
                ),
                "source_execution_item_id": item.get("source_execution_item_id"),
                "source_followup_item_id": item.get("source_followup_item_id"),
                "execution_kind": item.get("execution_kind"),
                "item_status": item.get("item_status"),
                "verification_decision": item.get("verification_decision"),
                "followup_kind": item.get("followup_kind"),
                "request_kind": item["request_kind"],
                "instructions": item.get("instructions"),
                "command": item["command"] if item.get("command") is not None else {},
                "metadata": item["metadata"] if item.get("metadata") is not None else {},
            }
        )
    return normalized


def _validate_post_verification_execution_request_status_consistency(
    request_record: dict[str, Any],
) -> None:
    request_status = request_record["request_status"]
    request_items = request_record["request_items"]
    if request_status == POST_VERIFICATION_EXECUTION_REQUEST_STATUS_EMPTY:
        if request_items:
            raise ValueError(
                "run_post_verification_execution_request_record: empty request_status "
                "requires request_items to be empty"
            )
    elif request_status == POST_VERIFICATION_EXECUTION_REQUEST_STATUS_REQUESTED:
        if not request_items:
            raise ValueError(
                "run_post_verification_execution_request_record: requested "
                "request_status requires non-empty request_items"
            )


def derive_post_verification_execution_request_items(
    followup_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Derive deterministic execution request items from follow-up plan items."""
    if not followup_items:
        return []

    request_items: list[dict[str, Any]] = []
    for followup_item in followup_items:
        followup_kind = followup_item["followup_kind"]
        request_kind = _FOLLOWUP_TO_EXECUTION_REQUEST_KIND.get(
            followup_kind, POST_VERIFICATION_EXECUTION_REQUEST_KIND_OTHER
        )
        request_items.append(
            {
                "request_item_id": f"pver-{followup_item['followup_item_id']}",
                "source_post_verification_followup_item_id": followup_item[
                    "followup_item_id"
                ],
                "source_execution_item_id": followup_item.get(
                    "source_execution_item_id"
                ),
                "source_followup_item_id": followup_item.get("source_followup_item_id"),
                "execution_kind": followup_item.get("execution_kind"),
                "item_status": followup_item.get("item_status"),
                "verification_decision": followup_item.get("verification_decision"),
                "followup_kind": followup_item.get("followup_kind"),
                "request_kind": request_kind,
                "instructions": followup_item.get("instructions"),
                "command": dict(followup_item.get("command") or {}),
                "metadata": dict(followup_item.get("metadata") or {}),
            }
        )
    return request_items


def validate_post_verification_execution_request_items_correspond(
    request_items: list[dict[str, Any]],
    followup_plan_record: dict[str, Any],
) -> None:
    """Ensure request items align with post-verification follow-up plan items."""
    followup_by_id = {
        item["followup_item_id"]: item
        for item in followup_plan_record["followup_items"]
    }

    for request_item in request_items:
        source_followup_item_id = request_item.get(
            "source_post_verification_followup_item_id"
        )
        if source_followup_item_id is None:
            continue
        if source_followup_item_id not in followup_by_id:
            raise ValueError(
                "request item references unknown "
                f"source_post_verification_followup_item_id "
                f"{source_followup_item_id!r}"
            )
        followup_item = followup_by_id[source_followup_item_id]
        for field in (
            "source_execution_item_id",
            "source_followup_item_id",
            "execution_kind",
            "item_status",
            "verification_decision",
            "followup_kind",
        ):
            if request_item.get(field) != followup_item.get(field):
                raise ValueError(
                    f"request item {field} does not match followup plan for "
                    f"{source_followup_item_id!r}"
                )


def make_run_post_verification_execution_request_record(
    *,
    run_id: str,
    source_execution_result_fingerprint: str,
    source_execution_verification_fingerprint: str,
    source_post_verification_followup_plan_fingerprint: str,
    request_items: list[dict[str, Any]],
    requester: str = "human",
    request_status: str | None = None,
    notes: str | None = None,
    metadata: dict[str, Any] | None = None,
    schema_version: int = 1,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a validated post-verification execution request record envelope."""
    normalized_items = _normalize_post_verification_execution_request_items(
        request_items
    )
    resolved_request_status = request_status
    if resolved_request_status is None:
        resolved_request_status = (
            POST_VERIFICATION_EXECUTION_REQUEST_STATUS_EMPTY
            if not normalized_items
            else POST_VERIFICATION_EXECUTION_REQUEST_STATUS_REQUESTED
        )
    request_record: dict[str, Any] = {
        "schema_version": schema_version,
        "run_id": run_id,
        "source_execution_result_fingerprint": source_execution_result_fingerprint,
        "source_execution_verification_fingerprint": (
            source_execution_verification_fingerprint
        ),
        "source_post_verification_followup_plan_fingerprint": (
            source_post_verification_followup_plan_fingerprint
        ),
        "requester": requester,
        "request_status": resolved_request_status,
        "request_items": normalized_items,
        "notes": notes,
        "metadata": metadata if metadata is not None else {},
        "created_at": created_at or _utc_now_iso(),
    }
    validate_schema(request_record, "run_post_verification_execution_request_record")
    return request_record


def run_post_verification_execution_request_fingerprint(
    request_record: dict[str, Any],
) -> str:
    """Return a stable semantic fingerprint for a post-verification execution request."""
    validate_schema(request_record, "run_post_verification_execution_request_record")
    return json.dumps(
        request_record,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


POST_VERIFICATION_EXECUTION_RESULT_STATUS_COMPLETED = "completed"
POST_VERIFICATION_EXECUTION_RESULT_STATUS_FAILED = "failed"
POST_VERIFICATION_EXECUTION_RESULT_STATUS_PARTIAL = "partial"
POST_VERIFICATION_EXECUTION_RESULT_STATUS_EMPTY = "empty"

POST_VERIFICATION_EXECUTION_RESULT_STATUSES: frozenset[str] = frozenset(
    {
        POST_VERIFICATION_EXECUTION_RESULT_STATUS_COMPLETED,
        POST_VERIFICATION_EXECUTION_RESULT_STATUS_FAILED,
        POST_VERIFICATION_EXECUTION_RESULT_STATUS_PARTIAL,
        POST_VERIFICATION_EXECUTION_RESULT_STATUS_EMPTY,
    }
)

POST_VERIFICATION_EXECUTION_RESULT_ITEM_STATUS_COMPLETED = "completed"
POST_VERIFICATION_EXECUTION_RESULT_ITEM_STATUS_FAILED = "failed"
POST_VERIFICATION_EXECUTION_RESULT_ITEM_STATUS_SKIPPED = "skipped"
POST_VERIFICATION_EXECUTION_RESULT_ITEM_STATUS_NOT_APPLICABLE = "not_applicable"

POST_VERIFICATION_EXECUTION_RESULT_ITEM_STATUSES: frozenset[str] = frozenset(
    {
        POST_VERIFICATION_EXECUTION_RESULT_ITEM_STATUS_COMPLETED,
        POST_VERIFICATION_EXECUTION_RESULT_ITEM_STATUS_FAILED,
        POST_VERIFICATION_EXECUTION_RESULT_ITEM_STATUS_SKIPPED,
        POST_VERIFICATION_EXECUTION_RESULT_ITEM_STATUS_NOT_APPLICABLE,
    }
)


def run_post_verification_execution_result_record_json_path(
    run_id: str,
    base_dir: Path | None = None,
) -> Path:
    """Return the JSON post-verification execution result record path for *run_id*."""
    return (
        paths.run_root(run_id, base_dir)
        / "run_post_verification_execution_result_record.json"
    )


def _normalize_post_verification_execution_result_items(
    result_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in result_items:
        normalized.append(
            {
                "result_item_id": item["result_item_id"],
                "source_request_item_id": item.get("source_request_item_id"),
                "source_post_verification_followup_item_id": item.get(
                    "source_post_verification_followup_item_id"
                ),
                "source_execution_item_id": item.get("source_execution_item_id"),
                "source_followup_item_id": item.get("source_followup_item_id"),
                "request_kind": item.get("request_kind"),
                "execution_kind": item.get("execution_kind"),
                "item_status": item.get("item_status"),
                "verification_decision": item.get("verification_decision"),
                "followup_kind": item.get("followup_kind"),
                "result_item_status": item["result_item_status"],
                "outcome": item.get("outcome"),
                "evidence": item["evidence"] if item.get("evidence") is not None else {},
                "metadata": item["metadata"] if item.get("metadata") is not None else {},
            }
        )
    return normalized


def compute_post_verification_execution_result_status(
    result_items: list[dict[str, Any]],
) -> str:
    """Derive aggregate post-verification execution result status from item results."""
    if not result_items:
        return POST_VERIFICATION_EXECUTION_RESULT_STATUS_EMPTY
    item_statuses = [item["result_item_status"] for item in result_items]
    if any(
        status == POST_VERIFICATION_EXECUTION_RESULT_ITEM_STATUS_SKIPPED
        for status in item_statuses
    ):
        return POST_VERIFICATION_EXECUTION_RESULT_STATUS_PARTIAL
    if all(
        status
        in (
            POST_VERIFICATION_EXECUTION_RESULT_ITEM_STATUS_COMPLETED,
            POST_VERIFICATION_EXECUTION_RESULT_ITEM_STATUS_NOT_APPLICABLE,
        )
        for status in item_statuses
    ):
        return POST_VERIFICATION_EXECUTION_RESULT_STATUS_COMPLETED
    if all(
        status
        in (
            POST_VERIFICATION_EXECUTION_RESULT_ITEM_STATUS_FAILED,
            POST_VERIFICATION_EXECUTION_RESULT_ITEM_STATUS_NOT_APPLICABLE,
        )
        for status in item_statuses
    ):
        return POST_VERIFICATION_EXECUTION_RESULT_STATUS_FAILED
    return POST_VERIFICATION_EXECUTION_RESULT_STATUS_PARTIAL


def _validate_post_verification_execution_result_status_consistency(
    result_record: dict[str, Any],
) -> None:
    result_status = result_record["result_status"]
    result_items = result_record["result_items"]
    if result_status == POST_VERIFICATION_EXECUTION_RESULT_STATUS_EMPTY:
        if result_items:
            raise ValueError(
                "run_post_verification_execution_result_record: empty result_status "
                "requires result_items to be empty"
            )
        return
    if not result_items:
        raise ValueError(
            "run_post_verification_execution_result_record: "
            f"{result_status} result_status requires non-empty result_items"
        )
    item_statuses = [item["result_item_status"] for item in result_items]
    if result_status == POST_VERIFICATION_EXECUTION_RESULT_STATUS_COMPLETED:
        if not all(
            status
            in (
                POST_VERIFICATION_EXECUTION_RESULT_ITEM_STATUS_COMPLETED,
                POST_VERIFICATION_EXECUTION_RESULT_ITEM_STATUS_NOT_APPLICABLE,
            )
            for status in item_statuses
        ):
            raise ValueError(
                "run_post_verification_execution_result_record: completed "
                "result_status requires all result_item_status values to be "
                "completed or not_applicable"
            )
    elif result_status == POST_VERIFICATION_EXECUTION_RESULT_STATUS_FAILED:
        if not all(
            status
            in (
                POST_VERIFICATION_EXECUTION_RESULT_ITEM_STATUS_FAILED,
                POST_VERIFICATION_EXECUTION_RESULT_ITEM_STATUS_NOT_APPLICABLE,
            )
            for status in item_statuses
        ):
            raise ValueError(
                "run_post_verification_execution_result_record: failed "
                "result_status requires all result_item_status values to be "
                "failed or not_applicable"
            )
    elif result_status == POST_VERIFICATION_EXECUTION_RESULT_STATUS_PARTIAL:
        has_skipped = any(
            status == POST_VERIFICATION_EXECUTION_RESULT_ITEM_STATUS_SKIPPED
            for status in item_statuses
        )
        all_completed_or_na = all(
            status
            in (
                POST_VERIFICATION_EXECUTION_RESULT_ITEM_STATUS_COMPLETED,
                POST_VERIFICATION_EXECUTION_RESULT_ITEM_STATUS_NOT_APPLICABLE,
            )
            for status in item_statuses
        )
        all_failed_or_na = all(
            status
            in (
                POST_VERIFICATION_EXECUTION_RESULT_ITEM_STATUS_FAILED,
                POST_VERIFICATION_EXECUTION_RESULT_ITEM_STATUS_NOT_APPLICABLE,
            )
            for status in item_statuses
        )
        if not has_skipped and (all_completed_or_na or all_failed_or_na):
            raise ValueError(
                "run_post_verification_execution_result_record: partial "
                "result_status requires a mix of item outcomes or a skipped item"
            )


def validate_post_verification_execution_result_items_correspond(
    result_items: list[dict[str, Any]],
    execution_request_record: dict[str, Any],
) -> None:
    """Ensure result items align with post-verification execution request items."""
    request_by_id = {
        item["request_item_id"]: item
        for item in execution_request_record["request_items"]
    }

    for result_item in result_items:
        source_request_item_id = result_item.get("source_request_item_id")
        if source_request_item_id is None:
            continue
        if source_request_item_id not in request_by_id:
            raise ValueError(
                "result item references unknown source_request_item_id "
                f"{source_request_item_id!r}"
            )
        request_item = request_by_id[source_request_item_id]
        for field in (
            "source_post_verification_followup_item_id",
            "source_execution_item_id",
            "source_followup_item_id",
            "request_kind",
            "execution_kind",
            "item_status",
            "verification_decision",
            "followup_kind",
        ):
            if result_item.get(field) != request_item.get(field):
                raise ValueError(
                    f"result item {field} does not match execution request for "
                    f"{source_request_item_id!r}"
                )


def make_run_post_verification_execution_result_record(
    *,
    run_id: str,
    source_execution_result_fingerprint: str,
    source_execution_verification_fingerprint: str,
    source_post_verification_followup_plan_fingerprint: str,
    source_post_verification_execution_request_fingerprint: str,
    result_items: list[dict[str, Any]],
    executor: str = "human",
    result_status: str | None = None,
    notes: str | None = None,
    metadata: dict[str, Any] | None = None,
    schema_version: int = 1,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a validated post-verification execution result record envelope."""
    normalized_items = _normalize_post_verification_execution_result_items(
        result_items
    )
    resolved_result_status = result_status
    if resolved_result_status is None:
        resolved_result_status = compute_post_verification_execution_result_status(
            normalized_items
        )
    result_record: dict[str, Any] = {
        "schema_version": schema_version,
        "run_id": run_id,
        "source_execution_result_fingerprint": source_execution_result_fingerprint,
        "source_execution_verification_fingerprint": (
            source_execution_verification_fingerprint
        ),
        "source_post_verification_followup_plan_fingerprint": (
            source_post_verification_followup_plan_fingerprint
        ),
        "source_post_verification_execution_request_fingerprint": (
            source_post_verification_execution_request_fingerprint
        ),
        "executor": executor,
        "result_status": resolved_result_status,
        "result_items": normalized_items,
        "notes": notes,
        "metadata": metadata if metadata is not None else {},
        "created_at": created_at or _utc_now_iso(),
    }
    validate_schema(result_record, "run_post_verification_execution_result_record")
    return result_record


def run_post_verification_execution_result_fingerprint(
    result_record: dict[str, Any],
) -> str:
    """Return a stable semantic fingerprint for a post-verification execution result."""
    validate_schema(result_record, "run_post_verification_execution_result_record")
    return json.dumps(
        result_record,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


POST_VERIFICATION_EXECUTION_VERIFICATION_STATUS_VERIFIED = "verified"
POST_VERIFICATION_EXECUTION_VERIFICATION_STATUS_REJECTED = "rejected"
POST_VERIFICATION_EXECUTION_VERIFICATION_STATUS_NEEDS_CHANGES = "needs_changes"
POST_VERIFICATION_EXECUTION_VERIFICATION_STATUS_EMPTY = "empty"

POST_VERIFICATION_EXECUTION_VERIFICATION_STATUSES: frozenset[str] = frozenset(
    {
        POST_VERIFICATION_EXECUTION_VERIFICATION_STATUS_VERIFIED,
        POST_VERIFICATION_EXECUTION_VERIFICATION_STATUS_REJECTED,
        POST_VERIFICATION_EXECUTION_VERIFICATION_STATUS_NEEDS_CHANGES,
        POST_VERIFICATION_EXECUTION_VERIFICATION_STATUS_EMPTY,
    }
)

POST_VERIFICATION_EXECUTION_VERIFICATION_ITEM_DECISION_VERIFIED = "verified"
POST_VERIFICATION_EXECUTION_VERIFICATION_ITEM_DECISION_REJECTED = "rejected"
POST_VERIFICATION_EXECUTION_VERIFICATION_ITEM_DECISION_NEEDS_CHANGES = "needs_changes"
POST_VERIFICATION_EXECUTION_VERIFICATION_ITEM_DECISION_NOT_APPLICABLE = (
    "not_applicable"
)

POST_VERIFICATION_EXECUTION_VERIFICATION_ITEM_DECISIONS: frozenset[str] = frozenset(
    {
        POST_VERIFICATION_EXECUTION_VERIFICATION_ITEM_DECISION_VERIFIED,
        POST_VERIFICATION_EXECUTION_VERIFICATION_ITEM_DECISION_REJECTED,
        POST_VERIFICATION_EXECUTION_VERIFICATION_ITEM_DECISION_NEEDS_CHANGES,
        POST_VERIFICATION_EXECUTION_VERIFICATION_ITEM_DECISION_NOT_APPLICABLE,
    }
)


def run_post_verification_execution_verification_record_json_path(
    run_id: str,
    base_dir: Path | None = None,
) -> Path:
    """Return the JSON post-verification execution verification record path."""
    return (
        paths.run_root(run_id, base_dir)
        / "run_post_verification_execution_verification_record.json"
    )


def _normalize_post_verification_execution_verification_items(
    verification_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in verification_items:
        normalized.append(
            {
                "verification_item_id": item["verification_item_id"],
                "source_result_item_id": item.get("source_result_item_id"),
                "source_request_item_id": item.get("source_request_item_id"),
                "source_post_verification_followup_item_id": item.get(
                    "source_post_verification_followup_item_id"
                ),
                "source_execution_item_id": item.get("source_execution_item_id"),
                "source_followup_item_id": item.get("source_followup_item_id"),
                "request_kind": item.get("request_kind"),
                "execution_kind": item.get("execution_kind"),
                "item_status": item.get("item_status"),
                "verification_decision": item.get("verification_decision"),
                "followup_kind": item.get("followup_kind"),
                "result_item_status": item.get("result_item_status"),
                "verification_decision_after_result": item[
                    "verification_decision_after_result"
                ],
                "reason": item.get("reason"),
                "evidence": item["evidence"] if item.get("evidence") is not None else {},
                "metadata": item["metadata"] if item.get("metadata") is not None else {},
            }
        )
    return normalized


def compute_post_verification_execution_verification_status(
    verification_items: list[dict[str, Any]],
) -> str:
    """Derive aggregate verification status from explicit item decisions."""
    if not verification_items:
        return POST_VERIFICATION_EXECUTION_VERIFICATION_STATUS_EMPTY
    decisions = [
        item["verification_decision_after_result"] for item in verification_items
    ]
    if any(
        decision
        == POST_VERIFICATION_EXECUTION_VERIFICATION_ITEM_DECISION_NEEDS_CHANGES
        for decision in decisions
    ):
        return POST_VERIFICATION_EXECUTION_VERIFICATION_STATUS_NEEDS_CHANGES
    if all(
        decision
        in (
            POST_VERIFICATION_EXECUTION_VERIFICATION_ITEM_DECISION_VERIFIED,
            POST_VERIFICATION_EXECUTION_VERIFICATION_ITEM_DECISION_NOT_APPLICABLE,
        )
        for decision in decisions
    ) and any(
        decision == POST_VERIFICATION_EXECUTION_VERIFICATION_ITEM_DECISION_VERIFIED
        for decision in decisions
    ):
        return POST_VERIFICATION_EXECUTION_VERIFICATION_STATUS_VERIFIED
    if all(
        decision
        in (
            POST_VERIFICATION_EXECUTION_VERIFICATION_ITEM_DECISION_REJECTED,
            POST_VERIFICATION_EXECUTION_VERIFICATION_ITEM_DECISION_NOT_APPLICABLE,
        )
        for decision in decisions
    ) and any(
        decision == POST_VERIFICATION_EXECUTION_VERIFICATION_ITEM_DECISION_REJECTED
        for decision in decisions
    ):
        return POST_VERIFICATION_EXECUTION_VERIFICATION_STATUS_REJECTED
    return POST_VERIFICATION_EXECUTION_VERIFICATION_STATUS_NEEDS_CHANGES


def _validate_post_verification_execution_verification_status_consistency(
    verification_record: dict[str, Any],
) -> None:
    verification_status = verification_record["verification_status"]
    verification_items = verification_record["verification_items"]
    if verification_status == POST_VERIFICATION_EXECUTION_VERIFICATION_STATUS_EMPTY:
        if verification_items:
            raise ValueError(
                "run_post_verification_execution_verification_record: empty "
                "verification_status requires verification_items to be empty"
            )
        return
    if not verification_items:
        raise ValueError(
            "run_post_verification_execution_verification_record: "
            f"{verification_status} verification_status requires non-empty "
            "verification_items"
        )
    decisions = [
        item["verification_decision_after_result"] for item in verification_items
    ]
    if all(
        decision
        == POST_VERIFICATION_EXECUTION_VERIFICATION_ITEM_DECISION_NOT_APPLICABLE
        for decision in decisions
    ):
        raise ValueError(
            "run_post_verification_execution_verification_record: "
            "verification_items cannot all be not_applicable"
        )
    if verification_status == POST_VERIFICATION_EXECUTION_VERIFICATION_STATUS_VERIFIED:
        if not all(
            decision
            in (
                POST_VERIFICATION_EXECUTION_VERIFICATION_ITEM_DECISION_VERIFIED,
                POST_VERIFICATION_EXECUTION_VERIFICATION_ITEM_DECISION_NOT_APPLICABLE,
            )
            for decision in decisions
        ):
            raise ValueError(
                "run_post_verification_execution_verification_record: verified "
                "verification_status requires all "
                "verification_decision_after_result values to be verified or "
                "not_applicable"
            )
        if not any(
            decision
            == POST_VERIFICATION_EXECUTION_VERIFICATION_ITEM_DECISION_VERIFIED
            for decision in decisions
        ):
            raise ValueError(
                "run_post_verification_execution_verification_record: verified "
                "verification_status requires at least one verified item"
            )
    elif (
        verification_status
        == POST_VERIFICATION_EXECUTION_VERIFICATION_STATUS_REJECTED
    ):
        if not all(
            decision
            in (
                POST_VERIFICATION_EXECUTION_VERIFICATION_ITEM_DECISION_REJECTED,
                POST_VERIFICATION_EXECUTION_VERIFICATION_ITEM_DECISION_NOT_APPLICABLE,
            )
            for decision in decisions
        ):
            raise ValueError(
                "run_post_verification_execution_verification_record: rejected "
                "verification_status requires all "
                "verification_decision_after_result values to be rejected or "
                "not_applicable"
            )
        if not any(
            decision
            == POST_VERIFICATION_EXECUTION_VERIFICATION_ITEM_DECISION_REJECTED
            for decision in decisions
        ):
            raise ValueError(
                "run_post_verification_execution_verification_record: rejected "
                "verification_status requires at least one rejected item"
            )
    elif (
        verification_status
        == POST_VERIFICATION_EXECUTION_VERIFICATION_STATUS_NEEDS_CHANGES
    ):
        has_needs_changes = any(
            decision
            == POST_VERIFICATION_EXECUTION_VERIFICATION_ITEM_DECISION_NEEDS_CHANGES
            for decision in decisions
        )
        all_verified_or_na = all(
            decision
            in (
                POST_VERIFICATION_EXECUTION_VERIFICATION_ITEM_DECISION_VERIFIED,
                POST_VERIFICATION_EXECUTION_VERIFICATION_ITEM_DECISION_NOT_APPLICABLE,
            )
            for decision in decisions
        ) and any(
            decision
            == POST_VERIFICATION_EXECUTION_VERIFICATION_ITEM_DECISION_VERIFIED
            for decision in decisions
        )
        all_rejected_or_na = all(
            decision
            in (
                POST_VERIFICATION_EXECUTION_VERIFICATION_ITEM_DECISION_REJECTED,
                POST_VERIFICATION_EXECUTION_VERIFICATION_ITEM_DECISION_NOT_APPLICABLE,
            )
            for decision in decisions
        ) and any(
            decision
            == POST_VERIFICATION_EXECUTION_VERIFICATION_ITEM_DECISION_REJECTED
            for decision in decisions
        )
        if not has_needs_changes and (all_verified_or_na or all_rejected_or_na):
            raise ValueError(
                "run_post_verification_execution_verification_record: "
                "needs_changes verification_status requires a mix of item "
                "decisions or a needs_changes item"
            )


def validate_post_verification_execution_verification_items_correspond(
    verification_items: list[dict[str, Any]],
    execution_result_record: dict[str, Any],
) -> None:
    """Ensure verification items align with post-verification execution result items."""
    result_by_id = {
        item["result_item_id"]: item
        for item in execution_result_record["result_items"]
    }

    for verification_item in verification_items:
        source_result_item_id = verification_item.get("source_result_item_id")
        if source_result_item_id is None:
            continue
        if source_result_item_id not in result_by_id:
            raise ValueError(
                "verification item references unknown source_result_item_id "
                f"{source_result_item_id!r}"
            )
        result_item = result_by_id[source_result_item_id]
        for field in (
            "source_request_item_id",
            "source_post_verification_followup_item_id",
            "source_execution_item_id",
            "source_followup_item_id",
            "request_kind",
            "execution_kind",
            "item_status",
            "verification_decision",
            "followup_kind",
            "result_item_status",
        ):
            if verification_item.get(field) != result_item.get(field):
                raise ValueError(
                    f"verification item {field} does not match execution result for "
                    f"{source_result_item_id!r}"
                )


def make_run_post_verification_execution_verification_record(
    *,
    run_id: str,
    source_execution_result_fingerprint: str,
    source_execution_verification_fingerprint: str,
    source_post_verification_followup_plan_fingerprint: str,
    source_post_verification_execution_request_fingerprint: str,
    source_post_verification_execution_result_fingerprint: str,
    verification_items: list[dict[str, Any]],
    verifier: str = "human",
    verification_status: str | None = None,
    notes: str | None = None,
    metadata: dict[str, Any] | None = None,
    schema_version: int = 1,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a validated post-verification execution verification record envelope."""
    normalized_items = _normalize_post_verification_execution_verification_items(
        verification_items
    )
    resolved_verification_status = verification_status
    if resolved_verification_status is None:
        resolved_verification_status = (
            compute_post_verification_execution_verification_status(normalized_items)
        )
    verification_record: dict[str, Any] = {
        "schema_version": schema_version,
        "run_id": run_id,
        "source_execution_result_fingerprint": source_execution_result_fingerprint,
        "source_execution_verification_fingerprint": (
            source_execution_verification_fingerprint
        ),
        "source_post_verification_followup_plan_fingerprint": (
            source_post_verification_followup_plan_fingerprint
        ),
        "source_post_verification_execution_request_fingerprint": (
            source_post_verification_execution_request_fingerprint
        ),
        "source_post_verification_execution_result_fingerprint": (
            source_post_verification_execution_result_fingerprint
        ),
        "verifier": verifier,
        "verification_status": resolved_verification_status,
        "verification_items": normalized_items,
        "notes": notes,
        "metadata": metadata if metadata is not None else {},
        "created_at": created_at or _utc_now_iso(),
    }
    validate_schema(
        verification_record, "run_post_verification_execution_verification_record"
    )
    return verification_record


def run_post_verification_execution_verification_fingerprint(
    verification_record: dict[str, Any],
) -> str:
    """Return a stable semantic fingerprint for post-verification execution verification."""
    validate_schema(
        verification_record, "run_post_verification_execution_verification_record"
    )
    return json.dumps(
        verification_record,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


RUN_FINAL_CLOSURE_STATUS_CLOSED_VERIFIED = "closed_verified"
RUN_FINAL_CLOSURE_STATUS_CLOSED_REJECTED = "closed_rejected"
RUN_FINAL_CLOSURE_STATUS_CLOSED_NEEDS_MORE_WORK = "closed_needs_more_work"
RUN_FINAL_CLOSURE_STATUS_CLOSED_NO_ACTION = "closed_no_action"

RUN_FINAL_CLOSURE_STATUSES: frozenset[str] = frozenset(
    {
        RUN_FINAL_CLOSURE_STATUS_CLOSED_VERIFIED,
        RUN_FINAL_CLOSURE_STATUS_CLOSED_REJECTED,
        RUN_FINAL_CLOSURE_STATUS_CLOSED_NEEDS_MORE_WORK,
        RUN_FINAL_CLOSURE_STATUS_CLOSED_NO_ACTION,
    }
)

RUN_FINAL_CLOSURE_ITEM_DECISION_ACCEPTED = "accepted"
RUN_FINAL_CLOSURE_ITEM_DECISION_REJECTED = "rejected"
RUN_FINAL_CLOSURE_ITEM_DECISION_NEEDS_MORE_WORK = "needs_more_work"
RUN_FINAL_CLOSURE_ITEM_DECISION_NO_ACTION = "no_action"

RUN_FINAL_CLOSURE_ITEM_DECISIONS: frozenset[str] = frozenset(
    {
        RUN_FINAL_CLOSURE_ITEM_DECISION_ACCEPTED,
        RUN_FINAL_CLOSURE_ITEM_DECISION_REJECTED,
        RUN_FINAL_CLOSURE_ITEM_DECISION_NEEDS_MORE_WORK,
        RUN_FINAL_CLOSURE_ITEM_DECISION_NO_ACTION,
    }
)


def run_final_closure_record_json_path(
    run_id: str,
    base_dir: Path | None = None,
) -> Path:
    """Return the JSON run final closure record path for *run_id*."""
    return paths.run_root(run_id, base_dir) / "run_final_closure_record.json"


def _normalize_run_final_closure_items(
    closure_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in closure_items:
        normalized.append(
            {
                "closure_item_id": item["closure_item_id"],
                "source_post_verification_execution_verification_item_id": item.get(
                    "source_post_verification_execution_verification_item_id"
                ),
                "source_post_verification_execution_result_item_id": item.get(
                    "source_post_verification_execution_result_item_id"
                ),
                "source_post_verification_execution_request_item_id": item.get(
                    "source_post_verification_execution_request_item_id"
                ),
                "source_post_verification_followup_item_id": item.get(
                    "source_post_verification_followup_item_id"
                ),
                "source_execution_item_id": item.get("source_execution_item_id"),
                "source_followup_item_id": item.get("source_followup_item_id"),
                "verification_decision_after_result": item.get(
                    "verification_decision_after_result"
                ),
                "closure_decision": item["closure_decision"],
                "reason": item.get("reason"),
                "evidence": item["evidence"] if item.get("evidence") is not None else {},
                "metadata": item["metadata"] if item.get("metadata") is not None else {},
            }
        )
    return normalized


def compute_run_final_closure_status(
    closure_items: list[dict[str, Any]],
) -> str:
    """Derive aggregate final closure status from explicit closure item decisions."""
    if not closure_items:
        return RUN_FINAL_CLOSURE_STATUS_CLOSED_NO_ACTION
    decisions = [item["closure_decision"] for item in closure_items]
    if any(
        decision == RUN_FINAL_CLOSURE_ITEM_DECISION_NEEDS_MORE_WORK
        for decision in decisions
    ):
        return RUN_FINAL_CLOSURE_STATUS_CLOSED_NEEDS_MORE_WORK
    if all(
        decision
        in (
            RUN_FINAL_CLOSURE_ITEM_DECISION_ACCEPTED,
            RUN_FINAL_CLOSURE_ITEM_DECISION_NO_ACTION,
        )
        for decision in decisions
    ) and any(
        decision == RUN_FINAL_CLOSURE_ITEM_DECISION_ACCEPTED for decision in decisions
    ):
        return RUN_FINAL_CLOSURE_STATUS_CLOSED_VERIFIED
    if all(
        decision
        in (
            RUN_FINAL_CLOSURE_ITEM_DECISION_REJECTED,
            RUN_FINAL_CLOSURE_ITEM_DECISION_NO_ACTION,
        )
        for decision in decisions
    ) and any(
        decision == RUN_FINAL_CLOSURE_ITEM_DECISION_REJECTED for decision in decisions
    ):
        return RUN_FINAL_CLOSURE_STATUS_CLOSED_REJECTED
    return RUN_FINAL_CLOSURE_STATUS_CLOSED_NEEDS_MORE_WORK


def _validate_run_final_closure_status_consistency(
    closure_record: dict[str, Any],
) -> None:
    final_closure_status = closure_record["final_closure_status"]
    closure_items = closure_record["closure_items"]
    if final_closure_status == RUN_FINAL_CLOSURE_STATUS_CLOSED_NO_ACTION:
        if closure_items:
            raise ValueError(
                "run_final_closure_record: closed_no_action final_closure_status "
                "requires closure_items to be empty"
            )
        return
    if not closure_items:
        raise ValueError(
            "run_final_closure_record: "
            f"{final_closure_status} final_closure_status requires non-empty "
            "closure_items"
        )
    decisions = [item["closure_decision"] for item in closure_items]
    if all(
        decision == RUN_FINAL_CLOSURE_ITEM_DECISION_NO_ACTION for decision in decisions
    ):
        raise ValueError(
            "run_final_closure_record: closure_items cannot all be no_action"
        )
    if final_closure_status == RUN_FINAL_CLOSURE_STATUS_CLOSED_VERIFIED:
        if not all(
            decision
            in (
                RUN_FINAL_CLOSURE_ITEM_DECISION_ACCEPTED,
                RUN_FINAL_CLOSURE_ITEM_DECISION_NO_ACTION,
            )
            for decision in decisions
        ):
            raise ValueError(
                "run_final_closure_record: closed_verified final_closure_status "
                "requires all closure_decision values to be accepted or no_action"
            )
        if not any(
            decision == RUN_FINAL_CLOSURE_ITEM_DECISION_ACCEPTED for decision in decisions
        ):
            raise ValueError(
                "run_final_closure_record: closed_verified final_closure_status "
                "requires at least one accepted item"
            )
    elif final_closure_status == RUN_FINAL_CLOSURE_STATUS_CLOSED_REJECTED:
        if not all(
            decision
            in (
                RUN_FINAL_CLOSURE_ITEM_DECISION_REJECTED,
                RUN_FINAL_CLOSURE_ITEM_DECISION_NO_ACTION,
            )
            for decision in decisions
        ):
            raise ValueError(
                "run_final_closure_record: closed_rejected final_closure_status "
                "requires all closure_decision values to be rejected or no_action"
            )
        if not any(
            decision == RUN_FINAL_CLOSURE_ITEM_DECISION_REJECTED for decision in decisions
        ):
            raise ValueError(
                "run_final_closure_record: closed_rejected final_closure_status "
                "requires at least one rejected item"
            )
    elif final_closure_status == RUN_FINAL_CLOSURE_STATUS_CLOSED_NEEDS_MORE_WORK:
        has_needs_more_work = any(
            decision == RUN_FINAL_CLOSURE_ITEM_DECISION_NEEDS_MORE_WORK
            for decision in decisions
        )
        all_accepted_or_na = all(
            decision
            in (
                RUN_FINAL_CLOSURE_ITEM_DECISION_ACCEPTED,
                RUN_FINAL_CLOSURE_ITEM_DECISION_NO_ACTION,
            )
            for decision in decisions
        ) and any(
            decision == RUN_FINAL_CLOSURE_ITEM_DECISION_ACCEPTED for decision in decisions
        )
        all_rejected_or_na = all(
            decision
            in (
                RUN_FINAL_CLOSURE_ITEM_DECISION_REJECTED,
                RUN_FINAL_CLOSURE_ITEM_DECISION_NO_ACTION,
            )
            for decision in decisions
        ) and any(
            decision == RUN_FINAL_CLOSURE_ITEM_DECISION_REJECTED for decision in decisions
        )
        if not has_needs_more_work and (all_accepted_or_na or all_rejected_or_na):
            raise ValueError(
                "run_final_closure_record: closed_needs_more_work "
                "final_closure_status requires a mix of item decisions or a "
                "needs_more_work item"
            )


def validate_run_final_closure_sources_correspond(
    closure_items: list[dict[str, Any]],
    verification_record: dict[str, Any],
) -> None:
    """Ensure closure items align with post-verification execution verification items."""
    verification_by_id = {
        item["verification_item_id"]: item
        for item in verification_record["verification_items"]
    }

    for closure_item in closure_items:
        source_verification_item_id = closure_item.get(
            "source_post_verification_execution_verification_item_id"
        )
        if source_verification_item_id is None:
            continue
        if source_verification_item_id not in verification_by_id:
            raise ValueError(
                "closure item references unknown "
                "source_post_verification_execution_verification_item_id "
                f"{source_verification_item_id!r}"
            )
        verification_item = verification_by_id[source_verification_item_id]
        field_map = {
            "source_post_verification_execution_result_item_id": "source_result_item_id",
            "source_post_verification_execution_request_item_id": "source_request_item_id",
            "source_post_verification_followup_item_id": (
                "source_post_verification_followup_item_id"
            ),
            "source_execution_item_id": "source_execution_item_id",
            "source_followup_item_id": "source_followup_item_id",
            "verification_decision_after_result": "verification_decision_after_result",
        }
        for closure_field, verification_field in field_map.items():
            if closure_item.get(closure_field) != verification_item.get(
                verification_field
            ):
                raise ValueError(
                    f"closure item {closure_field} does not match verification item "
                    f"for {source_verification_item_id!r}"
                )


def make_run_final_closure_record(
    *,
    run_id: str,
    source_run_completion_fingerprint: str,
    source_run_review_fingerprint: str,
    source_run_followup_plan_fingerprint: str,
    source_run_execution_request_fingerprint: str,
    source_run_execution_result_fingerprint: str,
    source_run_execution_verification_fingerprint: str,
    source_post_verification_followup_plan_fingerprint: str,
    source_post_verification_execution_request_fingerprint: str,
    source_post_verification_execution_result_fingerprint: str,
    source_post_verification_execution_verification_fingerprint: str,
    closure_reason: str,
    closure_items: list[dict[str, Any]],
    closer: str = "human",
    final_closure_status: str | None = None,
    notes: str | None = None,
    metadata: dict[str, Any] | None = None,
    schema_version: int = 1,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a validated run final closure record envelope."""
    normalized_items = _normalize_run_final_closure_items(closure_items)
    resolved_final_closure_status = final_closure_status
    if resolved_final_closure_status is None:
        resolved_final_closure_status = compute_run_final_closure_status(
            normalized_items
        )
    closure_record: dict[str, Any] = {
        "schema_version": schema_version,
        "run_id": run_id,
        "source_run_completion_fingerprint": source_run_completion_fingerprint,
        "source_run_review_fingerprint": source_run_review_fingerprint,
        "source_run_followup_plan_fingerprint": source_run_followup_plan_fingerprint,
        "source_run_execution_request_fingerprint": (
            source_run_execution_request_fingerprint
        ),
        "source_run_execution_result_fingerprint": (
            source_run_execution_result_fingerprint
        ),
        "source_run_execution_verification_fingerprint": (
            source_run_execution_verification_fingerprint
        ),
        "source_post_verification_followup_plan_fingerprint": (
            source_post_verification_followup_plan_fingerprint
        ),
        "source_post_verification_execution_request_fingerprint": (
            source_post_verification_execution_request_fingerprint
        ),
        "source_post_verification_execution_result_fingerprint": (
            source_post_verification_execution_result_fingerprint
        ),
        "source_post_verification_execution_verification_fingerprint": (
            source_post_verification_execution_verification_fingerprint
        ),
        "closer": closer,
        "final_closure_status": resolved_final_closure_status,
        "closure_reason": closure_reason,
        "closure_items": normalized_items,
        "notes": notes,
        "metadata": metadata if metadata is not None else {},
        "created_at": created_at or _utc_now_iso(),
    }
    validate_schema(closure_record, "run_final_closure_record")
    return closure_record


def run_final_closure_fingerprint(closure_record: dict[str, Any]) -> str:
    """Return a stable semantic fingerprint for a run final closure record."""
    validate_schema(closure_record, "run_final_closure_record")
    return json.dumps(
        closure_record,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


PHASE1_MANUAL_WORKFLOW_RECORD_CHAIN: tuple[str, ...] = (
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
)

PHASE1_TERMINAL_RECORD_TYPE = "run_final_closure_record"
PHASE1_TERMINAL_EVENT_TYPE = "run_final_closure_recorded"
PHASE1_BOUNDARY_STATUS = "phase1_manual_workflow_frozen"
