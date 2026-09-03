"""Read-only HTR run observation and integrity reporting (Phase 2 Task 19)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from htr import contracts, events
from htr.schemas import validate as validate_schema

OBSERVER_NAME = "htr.observe"
OBSERVER_VERSION = "1.0.0"
SNAPSHOT_SCHEMA_VERSION = "1"

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"

EXIT_OK = 0
EXIT_INTEGRITY = 1
EXIT_INVOCATION = 2

FINDING_MALFORMED_AUTHORITATIVE_JSON = "malformed_authoritative_json"
FINDING_SCHEMA_VALIDATION_FAILED = "schema_validation_failed"
FINDING_EVENT_WITHOUT_JSON_SOT = "event_without_json_sot"
FINDING_JSON_WITHOUT_MATCHING_EVENT = "json_without_matching_event"
FINDING_RECORD_FINGERPRINT_MISMATCH = "record_fingerprint_mismatch"
FINDING_PHASE1_CHAIN_GAP = "phase1_chain_gap"
FINDING_SOURCE_FINGERPRINT_MISMATCH = "source_fingerprint_mismatch"
FINDING_SOURCE_CORRESPONDENCE_FAILED = "source_correspondence_failed"
FINDING_DUPLICATE_EVENT_ID = "duplicate_event_id"
FINDING_TASK_RUN_IDENTITY_MISMATCH = "task_run_identity_mismatch"
FINDING_ATTEMPT_TASK_IDENTITY_MISMATCH = "attempt_task_identity_mismatch"
FINDING_POST_CLOSURE_ACTIVITY = "post_closure_activity"

_EXECUTION_VERIFICATION_EVENT_TYPES: dict[str, str] = {
    contracts.EXECUTION_VERIFICATION_ACCEPTED: events.EVENT_TYPE_RUN_EXECUTION_VERIFIED,
    contracts.EXECUTION_VERIFICATION_REJECTED: events.EVENT_TYPE_RUN_EXECUTION_REJECTED,
    contracts.EXECUTION_VERIFICATION_NEEDS_CHANGES: (
        events.EVENT_TYPE_RUN_EXECUTION_NEEDS_CHANGES
    ),
}

_PHASE1_WORKFLOW_EVENT_TYPES: frozenset[str] = frozenset(
    {
        events.EVENT_TYPE_MANUAL_RUN_COMPLETED,
        events.EVENT_TYPE_MANUAL_RUN_REVIEWED,
        events.EVENT_TYPE_MANUAL_RUN_FOLLOWUP_PLANNED,
        events.EVENT_TYPE_RUN_EXECUTION_REQUESTED,
        events.EVENT_TYPE_RUN_EXECUTION_COMPLETED,
        events.EVENT_TYPE_RUN_EXECUTION_VERIFIED,
        events.EVENT_TYPE_RUN_EXECUTION_REJECTED,
        events.EVENT_TYPE_RUN_EXECUTION_NEEDS_CHANGES,
        events.EVENT_TYPE_RUN_POST_VERIFICATION_FOLLOWUP_PLANNED,
        events.EVENT_TYPE_RUN_POST_VERIFICATION_EXECUTION_REQUESTED,
        events.EVENT_TYPE_RUN_POST_VERIFICATION_EXECUTION_RESULT_RECORDED,
        events.EVENT_TYPE_RUN_POST_VERIFICATION_EXECUTION_VERIFICATION_RECORDED,
        events.EVENT_TYPE_RUN_FINAL_CLOSURE_RECORDED,
    }
)


class ObserveInvocationError(Exception):
    """Raised when observation cannot start (invalid input or inaccessible run)."""


@dataclass(frozen=True)
class _RecordSpec:
    record_type: str
    path_fn: Callable[..., Path]
    fingerprint_fn: Callable[[dict[str, Any]], str]
    schema_name: str
    payload_fingerprint_key: str
    workflow_event_types: frozenset[str]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_root(run_id: str, base_dir: Path | None) -> Path:
    return contracts.run_completion_record_json_path(run_id, base_dir).parent


def _runs_root_for_run(run_root_path: Path, base_dir: Path | None) -> Path:
    if base_dir is not None:
        return Path(base_dir)
    return run_root_path.parent


def _ensure_run_workspace_contained(
    run_id: str,
    run_root_path: Path,
    base_dir: Path | None,
) -> None:
    """Fail closed when *run_root_path* resolves outside the configured runs root."""
    runs_root_path = _runs_root_for_run(run_root_path, base_dir)
    runs_resolved = runs_root_path.resolve()
    run_resolved = run_root_path.resolve()
    try:
        common = os.path.commonpath([str(runs_resolved), str(run_resolved)])
    except ValueError as exc:
        raise ObserveInvocationError(
            f"run workspace path is incompatible with runs root for run_id {run_id!r}"
        ) from exc
    if common != str(runs_resolved):
        raise ObserveInvocationError(
            f"run workspace for run_id {run_id!r} resolves outside configured runs root"
        )


def _rel_evidence(path: Path, run_root_path: Path) -> str:
    try:
        return str(path.relative_to(run_root_path))
    except ValueError:
        return path.name


def _event_timestamp(event: dict[str, Any]) -> str | None:
    """Return the canonical event time field when present."""
    for key in ("created_at", "timestamp"):
        value = event.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _read_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        row = json.loads(stripped)
        if not isinstance(row, dict):
            raise ValueError(f"expected JSON object lines in {path}")
        rows.append(row)
    return rows


def _record_specs() -> tuple[_RecordSpec, ...]:
    return (
        _RecordSpec(
            "run_completion_record",
            contracts.run_completion_record_json_path,
            contracts.run_completion_fingerprint,
            "run_completion_record",
            "run_completion_fingerprint",
            frozenset({events.EVENT_TYPE_MANUAL_RUN_COMPLETED}),
        ),
        _RecordSpec(
            "run_review_record",
            contracts.run_review_record_json_path,
            contracts.run_review_fingerprint,
            "run_review_record",
            "run_review_fingerprint",
            frozenset({events.EVENT_TYPE_MANUAL_RUN_REVIEWED}),
        ),
        _RecordSpec(
            "run_followup_plan_record",
            contracts.run_followup_plan_record_json_path,
            contracts.run_followup_plan_fingerprint,
            "run_followup_plan_record",
            "run_followup_plan_fingerprint",
            frozenset({events.EVENT_TYPE_MANUAL_RUN_FOLLOWUP_PLANNED}),
        ),
        _RecordSpec(
            "run_execution_request_record",
            contracts.run_execution_request_record_json_path,
            contracts.run_execution_request_fingerprint,
            "run_execution_request_record",
            "run_execution_request_fingerprint",
            frozenset({events.EVENT_TYPE_RUN_EXECUTION_REQUESTED}),
        ),
        _RecordSpec(
            "run_execution_result_record",
            contracts.run_execution_result_record_json_path,
            contracts.run_execution_result_fingerprint,
            "run_execution_result_record",
            "run_execution_result_fingerprint",
            frozenset({events.EVENT_TYPE_RUN_EXECUTION_COMPLETED}),
        ),
        _RecordSpec(
            "run_execution_verification_record",
            contracts.run_execution_verification_record_json_path,
            contracts.run_execution_verification_fingerprint,
            "run_execution_verification_record",
            "run_execution_verification_fingerprint",
            frozenset(set(_EXECUTION_VERIFICATION_EVENT_TYPES.values())),
        ),
        _RecordSpec(
            "run_post_verification_followup_plan_record",
            contracts.run_post_verification_followup_plan_record_json_path,
            contracts.run_post_verification_followup_plan_fingerprint,
            "run_post_verification_followup_plan_record",
            "run_post_verification_followup_plan_fingerprint",
            frozenset({events.EVENT_TYPE_RUN_POST_VERIFICATION_FOLLOWUP_PLANNED}),
        ),
        _RecordSpec(
            "run_post_verification_execution_request_record",
            contracts.run_post_verification_execution_request_record_json_path,
            contracts.run_post_verification_execution_request_fingerprint,
            "run_post_verification_execution_request_record",
            "run_post_verification_execution_request_fingerprint",
            frozenset({events.EVENT_TYPE_RUN_POST_VERIFICATION_EXECUTION_REQUESTED}),
        ),
        _RecordSpec(
            "run_post_verification_execution_result_record",
            contracts.run_post_verification_execution_result_record_json_path,
            contracts.run_post_verification_execution_result_fingerprint,
            "run_post_verification_execution_result_record",
            "run_post_verification_execution_result_fingerprint",
            frozenset({events.EVENT_TYPE_RUN_POST_VERIFICATION_EXECUTION_RESULT_RECORDED}),
        ),
        _RecordSpec(
            "run_post_verification_execution_verification_record",
            contracts.run_post_verification_execution_verification_record_json_path,
            contracts.run_post_verification_execution_verification_fingerprint,
            "run_post_verification_execution_verification_record",
            "run_post_verification_execution_verification_fingerprint",
            frozenset(
                {events.EVENT_TYPE_RUN_POST_VERIFICATION_EXECUTION_VERIFICATION_RECORDED}
            ),
        ),
        _RecordSpec(
            "run_final_closure_record",
            contracts.run_final_closure_record_json_path,
            contracts.run_final_closure_fingerprint,
            "run_final_closure_record",
            "run_final_closure_fingerprint",
            frozenset({events.EVENT_TYPE_RUN_FINAL_CLOSURE_RECORDED}),
        ),
    )


def _finding(
    *,
    code: str,
    severity: str,
    message: str,
    subject: dict[str, Any],
    evidence: list[str],
) -> dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "message": message,
        "subject": subject,
        "evidence": sorted(set(evidence)),
    }


def _event_matches_record(
    event: dict[str, Any],
    *,
    run_id: str,
    record: dict[str, Any],
    spec: _RecordSpec,
) -> bool:
    if event.get("run_id") != run_id:
        return False
    if event.get("event_type") not in spec.workflow_event_types:
        return False
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return False
    return payload.get(spec.payload_fingerprint_key) == spec.fingerprint_fn(record)


def _find_matching_events(
    all_events: list[dict[str, Any]],
    *,
    run_id: str,
    record: dict[str, Any],
    spec: _RecordSpec,
) -> list[dict[str, Any]]:
    return [
        event
        for event in all_events
        if _event_matches_record(event, run_id=run_id, record=record, spec=spec)
    ]


def _load_record(
    path: Path,
    spec: _RecordSpec,
    *,
    run_id: str,
    run_root_path: Path,
    findings: list[dict[str, Any]],
) -> dict[str, Any] | None:
    rel = _rel_evidence(path, run_root_path)
    try:
        record = _read_json_object(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        findings.append(
            _finding(
                code=FINDING_MALFORMED_AUTHORITATIVE_JSON,
                severity=SEVERITY_ERROR,
                message=str(exc),
                subject={"record_type": spec.record_type, "run_id": run_id},
                evidence=[rel],
            )
        )
        return None
    try:
        validate_schema(record, spec.schema_name)
    except Exception as exc:
        findings.append(
            _finding(
                code=FINDING_SCHEMA_VALIDATION_FAILED,
                severity=SEVERITY_ERROR,
                message=str(exc),
                subject={"record_type": spec.record_type, "run_id": run_id},
                evidence=[rel],
            )
        )
        return None
    try:
        spec.fingerprint_fn(record)
    except Exception as exc:
        findings.append(
            _finding(
                code=FINDING_RECORD_FINGERPRINT_MISMATCH,
                severity=SEVERITY_ERROR,
                message=f"fingerprint computation failed: {exc}",
                subject={"record_type": spec.record_type, "run_id": run_id},
                evidence=[rel],
            )
        )
    return record


def _check_event_payload_vs_record_fingerprints(
    all_events: list[dict[str, Any]],
    loaded: dict[str, dict[str, Any]],
    specs_by_type: dict[str, _RecordSpec],
    *,
    run_id: str,
    run_root_path: Path,
    base_dir: Path | None,
    findings: list[dict[str, Any]],
) -> None:
    events_rel = _rel_evidence(run_root_path / "task_events.jsonl", run_root_path)
    for spec in specs_by_type.values():
        path = spec.path_fn(run_id, base_dir)
        rel = _rel_evidence(path, run_root_path)
        record = loaded.get(spec.record_type)
        for event in all_events:
            if event.get("run_id") != run_id:
                continue
            if event.get("event_type") not in spec.workflow_event_types:
                continue
            payload = event.get("payload")
            if not isinstance(payload, dict):
                continue
            event_fp = payload.get(spec.payload_fingerprint_key)
            if event_fp is None:
                continue
            if record is None:
                findings.append(
                    _finding(
                        code=FINDING_EVENT_WITHOUT_JSON_SOT,
                        severity=SEVERITY_ERROR,
                        message=(
                            f"audit event references {spec.record_type} fingerprint "
                            "but JSON SoT is missing"
                        ),
                        subject={"run_id": run_id, "record_type": spec.record_type},
                        evidence=[events_rel, rel],
                    )
                )
                break
            record_fp = spec.fingerprint_fn(record)
            if event_fp != record_fp:
                findings.append(
                    _finding(
                        code=FINDING_RECORD_FINGERPRINT_MISMATCH,
                        severity=SEVERITY_ERROR,
                        message=(
                            f"audit event {spec.payload_fingerprint_key} does not match "
                            f"{spec.record_type} JSON SoT fingerprint"
                        ),
                        subject={"run_id": run_id, "record_type": spec.record_type},
                        evidence=[events_rel, rel],
                    )
                )


def _check_source_fingerprints(
    loaded: dict[str, dict[str, Any]],
    specs_by_type: dict[str, _RecordSpec],
    *,
    run_id: str,
    base_dir: Path | None,
    run_root_path: Path,
    findings: list[dict[str, Any]],
) -> None:
    links: tuple[tuple[str, str, str], ...] = (
        (
            "run_execution_request_record",
            "run_followup_plan_record",
            "source_followup_plan_fingerprint",
        ),
        (
            "run_execution_result_record",
            "run_execution_request_record",
            "source_execution_request_fingerprint",
        ),
        (
            "run_execution_verification_record",
            "run_execution_result_record",
            "source_execution_result_fingerprint",
        ),
        (
            "run_post_verification_followup_plan_record",
            "run_execution_result_record",
            "source_execution_result_fingerprint",
        ),
        (
            "run_post_verification_followup_plan_record",
            "run_execution_verification_record",
            "source_execution_verification_fingerprint",
        ),
        (
            "run_post_verification_execution_request_record",
            "run_post_verification_followup_plan_record",
            "source_post_verification_followup_plan_fingerprint",
        ),
        (
            "run_post_verification_execution_result_record",
            "run_execution_result_record",
            "source_execution_result_fingerprint",
        ),
        (
            "run_post_verification_execution_result_record",
            "run_execution_verification_record",
            "source_execution_verification_fingerprint",
        ),
        (
            "run_post_verification_execution_result_record",
            "run_post_verification_followup_plan_record",
            "source_post_verification_followup_plan_fingerprint",
        ),
        (
            "run_post_verification_execution_result_record",
            "run_post_verification_execution_request_record",
            "source_post_verification_execution_request_fingerprint",
        ),
        (
            "run_post_verification_execution_verification_record",
            "run_execution_result_record",
            "source_execution_result_fingerprint",
        ),
        (
            "run_post_verification_execution_verification_record",
            "run_execution_verification_record",
            "source_execution_verification_fingerprint",
        ),
        (
            "run_post_verification_execution_verification_record",
            "run_post_verification_followup_plan_record",
            "source_post_verification_followup_plan_fingerprint",
        ),
        (
            "run_post_verification_execution_verification_record",
            "run_post_verification_execution_request_record",
            "source_post_verification_execution_request_fingerprint",
        ),
        (
            "run_post_verification_execution_verification_record",
            "run_post_verification_execution_result_record",
            "source_post_verification_execution_result_fingerprint",
        ),
    )
    for current_type, prior_type, field_name in links:
        current = loaded.get(current_type)
        prior = loaded.get(prior_type)
        if current is None or prior is None:
            continue
        prior_spec = specs_by_type[prior_type]
        current_spec = specs_by_type[current_type]
        expected = prior_spec.fingerprint_fn(prior)
        if current.get(field_name) != expected:
            findings.append(
                _finding(
                    code=FINDING_SOURCE_FINGERPRINT_MISMATCH,
                    severity=SEVERITY_ERROR,
                    message=(
                        f"{field_name} on {current_type} does not match "
                        f"{prior_type} fingerprint"
                    ),
                    subject={
                        "run_id": run_id,
                        "record_type": current_type,
                        "prior_record_type": prior_type,
                        "field": field_name,
                    },
                    evidence=[
                        _rel_evidence(current_spec.path_fn(run_id, base_dir), run_root_path),
                        _rel_evidence(prior_spec.path_fn(run_id, base_dir), run_root_path),
                    ],
                )
            )

    closure = loaded.get("run_final_closure_record")
    if closure is None:
        return
    closure_spec = specs_by_type["run_final_closure_record"]
    closure_fields = (
        ("source_run_completion_fingerprint", "run_completion_record"),
        ("source_run_review_fingerprint", "run_review_record"),
        ("source_run_followup_plan_fingerprint", "run_followup_plan_record"),
        ("source_run_execution_request_fingerprint", "run_execution_request_record"),
        ("source_run_execution_result_fingerprint", "run_execution_result_record"),
        (
            "source_run_execution_verification_fingerprint",
            "run_execution_verification_record",
        ),
        (
            "source_post_verification_followup_plan_fingerprint",
            "run_post_verification_followup_plan_record",
        ),
        (
            "source_post_verification_execution_request_fingerprint",
            "run_post_verification_execution_request_record",
        ),
        (
            "source_post_verification_execution_result_fingerprint",
            "run_post_verification_execution_result_record",
        ),
        (
            "source_post_verification_execution_verification_fingerprint",
            "run_post_verification_execution_verification_record",
        ),
    )
    for field_name, prior_type in closure_fields:
        prior = loaded.get(prior_type)
        if prior is None:
            continue
        expected = specs_by_type[prior_type].fingerprint_fn(prior)
        if closure.get(field_name) != expected:
            findings.append(
                _finding(
                    code=FINDING_SOURCE_FINGERPRINT_MISMATCH,
                    severity=SEVERITY_ERROR,
                    message=(
                        f"{field_name} on run_final_closure_record does not match "
                        f"{prior_type} fingerprint"
                    ),
                    subject={
                        "run_id": run_id,
                        "record_type": "run_final_closure_record",
                        "prior_record_type": prior_type,
                        "field": field_name,
                    },
                    evidence=[
                        _rel_evidence(closure_spec.path_fn(run_id, base_dir), run_root_path),
                    ],
                )
            )


def _check_correspondence(
    loaded: dict[str, dict[str, Any]],
    *,
    run_id: str,
    base_dir: Path | None,
    run_root_path: Path,
    specs_by_type: dict[str, _RecordSpec],
    findings: list[dict[str, Any]],
) -> None:
    checks: list[tuple[str, str, Callable[[], None]]] = []

    exec_verification = loaded.get("run_execution_verification_record")
    exec_result = loaded.get("run_execution_result_record")
    if exec_verification is not None and exec_result is not None:
        checks.append(
            (
                "run_execution_verification_record",
                "run_execution_result_record",
                lambda: contracts.validate_item_verifications_correspond_to_results(
                    exec_verification["item_verifications"],
                    exec_result["item_results"],
                ),
            )
        )

    pv_followup = loaded.get("run_post_verification_followup_plan_record")
    if pv_followup is not None and exec_result is not None and exec_verification is not None:
        checks.append(
            (
                "run_post_verification_followup_plan_record",
                "run_execution_verification_record",
                lambda: contracts.validate_post_verification_followup_items_correspond(
                    pv_followup["followup_items"],
                    exec_result,
                    exec_verification,
                ),
            )
        )

    pv_request = loaded.get("run_post_verification_execution_request_record")
    if pv_request is not None and pv_followup is not None:
        checks.append(
            (
                "run_post_verification_execution_request_record",
                "run_post_verification_followup_plan_record",
                lambda: contracts.validate_post_verification_execution_request_items_correspond(
                    pv_request["request_items"],
                    pv_followup,
                ),
            )
        )

    pv_result = loaded.get("run_post_verification_execution_result_record")
    if pv_result is not None and pv_request is not None:
        checks.append(
            (
                "run_post_verification_execution_result_record",
                "run_post_verification_execution_request_record",
                lambda: contracts.validate_post_verification_execution_result_items_correspond(
                    pv_result["result_items"],
                    pv_request,
                ),
            )
        )

    pv_verification = loaded.get("run_post_verification_execution_verification_record")
    if pv_verification is not None and pv_result is not None:
        checks.append(
            (
                "run_post_verification_execution_verification_record",
                "run_post_verification_execution_result_record",
                lambda: contracts.validate_post_verification_execution_verification_items_correspond(
                    pv_verification["verification_items"],
                    pv_result,
                ),
            )
        )

    closure = loaded.get("run_final_closure_record")
    if closure is not None and pv_verification is not None:
        checks.append(
            (
                "run_final_closure_record",
                "run_post_verification_execution_verification_record",
                lambda: contracts.validate_run_final_closure_sources_correspond(
                    closure["closure_items"],
                    pv_verification,
                ),
            )
        )

    for record_type, related_type, check_fn in checks:
        try:
            check_fn()
        except (ValueError, KeyError, TypeError) as exc:
            findings.append(
                _finding(
                    code=FINDING_SOURCE_CORRESPONDENCE_FAILED,
                    severity=SEVERITY_ERROR,
                    message=str(exc),
                    subject={
                        "run_id": run_id,
                        "record_type": record_type,
                        "related_record_type": related_type,
                    },
                    evidence=[
                        _rel_evidence(
                            specs_by_type[record_type].path_fn(run_id, base_dir),
                            run_root_path,
                        ),
                    ],
                )
            )


def _discover_tasks(
    run_root_path: Path,
    *,
    run_id: str,
    findings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tasks_dir = run_root_path / "tasks"
    if not tasks_dir.is_dir():
        return []
    summaries: list[dict[str, Any]] = []
    for task_path in sorted(tasks_dir.iterdir(), key=lambda p: p.name):
        if not task_path.is_dir():
            continue
        task_id = task_path.name
        status_path = task_path / "task_status.json"
        task_summary: dict[str, Any] = {
            "task_id": task_id,
            "task_status_present": status_path.exists(),
            "status": None,
            "attempts": [],
        }
        if status_path.exists():
            rel = _rel_evidence(status_path, run_root_path)
            try:
                task_status = _read_json_object(status_path)
                validate_schema(task_status, "task_status")
                task_summary["status"] = task_status.get("status")
                if task_status.get("run_id") not in (None, run_id):
                    findings.append(
                        _finding(
                            code=FINDING_TASK_RUN_IDENTITY_MISMATCH,
                            severity=SEVERITY_ERROR,
                            message="task_status.run_id does not match observed run_id",
                            subject={"run_id": run_id, "task_id": task_id},
                            evidence=[rel],
                        )
                    )
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                findings.append(
                    _finding(
                        code=FINDING_MALFORMED_AUTHORITATIVE_JSON,
                        severity=SEVERITY_ERROR,
                        message=str(exc),
                        subject={"run_id": run_id, "task_id": task_id},
                        evidence=[rel],
                    )
                )
        attempts_dir = task_path / "attempts"
        if attempts_dir.is_dir():
            for attempt_path in sorted(attempts_dir.iterdir(), key=lambda p: p.name):
                if not attempt_path.is_dir():
                    continue
                attempt_id = attempt_path.name
                attempt_status_path = attempt_path / "attempt_status.json"
                attempt_summary: dict[str, Any] = {
                    "attempt_id": attempt_id,
                    "attempt_status_present": attempt_status_path.exists(),
                    "status": None,
                }
                if attempt_status_path.exists():
                    rel = _rel_evidence(attempt_status_path, run_root_path)
                    try:
                        attempt_status = _read_json_object(attempt_status_path)
                        validate_schema(attempt_status, "attempt_status")
                        attempt_summary["status"] = attempt_status.get("status")
                        if attempt_status.get("run_id") not in (None, run_id):
                            findings.append(
                                _finding(
                                    code=FINDING_TASK_RUN_IDENTITY_MISMATCH,
                                    severity=SEVERITY_ERROR,
                                    message=(
                                        "attempt_status.run_id does not match observed run_id"
                                    ),
                                    subject={
                                        "run_id": run_id,
                                        "task_id": task_id,
                                        "attempt_id": attempt_id,
                                    },
                                    evidence=[rel],
                                )
                            )
                        if attempt_status.get("task_id") not in (None, task_id):
                            findings.append(
                                _finding(
                                    code=FINDING_ATTEMPT_TASK_IDENTITY_MISMATCH,
                                    severity=SEVERITY_ERROR,
                                    message=(
                                        "attempt_status.task_id does not match parent task_id"
                                    ),
                                    subject={
                                        "run_id": run_id,
                                        "task_id": task_id,
                                        "attempt_id": attempt_id,
                                    },
                                    evidence=[rel],
                                )
                            )
                    except (OSError, json.JSONDecodeError, ValueError) as exc:
                        findings.append(
                            _finding(
                                code=FINDING_MALFORMED_AUTHORITATIVE_JSON,
                                severity=SEVERITY_ERROR,
                                message=str(exc),
                                subject={
                                    "run_id": run_id,
                                    "task_id": task_id,
                                    "attempt_id": attempt_id,
                                },
                                evidence=[rel],
                            )
                        )
                task_summary["attempts"].append(attempt_summary)
        summaries.append(task_summary)
    return summaries


def _check_phase1_chain(
    *,
    run_id: str,
    run_root_path: Path,
    base_dir: Path | None,
    all_events: list[dict[str, Any]],
    specs: tuple[_RecordSpec, ...],
    findings: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    specs_by_type = {spec.record_type: spec for spec in specs}
    loaded: dict[str, dict[str, Any]] = {}
    chain_entries: list[dict[str, Any]] = []
    presence: list[bool] = []
    events_rel = _rel_evidence(run_root_path / "task_events.jsonl", run_root_path)

    for spec in specs:
        path = spec.path_fn(run_id, base_dir)
        rel = _rel_evidence(path, run_root_path)
        present = path.exists()
        presence.append(present)
        entry: dict[str, Any] = {
            "record_type": spec.record_type,
            "json_path": rel,
            "present": present,
            "fingerprint": None,
            "matching_event_count": 0,
        }
        record: dict[str, Any] | None = None
        if present:
            record = _load_record(
                path, spec, run_id=run_id, run_root_path=run_root_path, findings=findings
            )
            if record is not None:
                loaded[spec.record_type] = record
                entry["fingerprint"] = spec.fingerprint_fn(record)
                matches = _find_matching_events(
                    all_events, run_id=run_id, record=record, spec=spec
                )
                entry["matching_event_count"] = len(matches)
                if not matches:
                    findings.append(
                        _finding(
                            code=FINDING_JSON_WITHOUT_MATCHING_EVENT,
                            severity=SEVERITY_ERROR,
                            message=(
                                f"{spec.record_type} JSON SoT exists without a matching "
                                "audit event"
                            ),
                            subject={"run_id": run_id, "record_type": spec.record_type},
                            evidence=[rel, events_rel],
                        )
                    )
        chain_entries.append(entry)

    max_present_index = max(
        (index for index, is_present in enumerate(presence) if is_present),
        default=-1,
    )
    for index, spec in enumerate(specs):
        if not presence[index] and index < max_present_index:
            findings.append(
                _finding(
                    code=FINDING_PHASE1_CHAIN_GAP,
                    severity=SEVERITY_ERROR,
                    message=(
                        f"missing {spec.record_type} while later Phase 1 chain "
                        "records exist"
                    ),
                    subject={"run_id": run_id, "record_type": spec.record_type},
                    evidence=[_rel_evidence(spec.path_fn(run_id, base_dir), run_root_path)],
                )
            )

    _check_event_payload_vs_record_fingerprints(
        all_events,
        loaded,
        specs_by_type,
        run_id=run_id,
        run_root_path=run_root_path,
        base_dir=base_dir,
        findings=findings,
    )
    _check_source_fingerprints(
        loaded,
        specs_by_type,
        run_id=run_id,
        base_dir=base_dir,
        run_root_path=run_root_path,
        findings=findings,
    )
    _check_correspondence(
        loaded,
        run_id=run_id,
        base_dir=base_dir,
        run_root_path=run_root_path,
        specs_by_type=specs_by_type,
        findings=findings,
    )
    return chain_entries, loaded


def _check_duplicate_event_ids(
    all_events: list[dict[str, Any]],
    *,
    run_id: str,
    run_root_path: Path,
    findings: list[dict[str, Any]],
) -> None:
    seen: dict[str, int] = {}
    events_rel = _rel_evidence(run_root_path / "task_events.jsonl", run_root_path)
    for event in all_events:
        if event.get("run_id") not in (None, run_id):
            continue
        event_id = event.get("event_id")
        if not isinstance(event_id, str):
            continue
        seen[event_id] = seen.get(event_id, 0) + 1
    for event_id, count in sorted(seen.items()):
        if count > 1:
            findings.append(
                _finding(
                    code=FINDING_DUPLICATE_EVENT_ID,
                    severity=SEVERITY_ERROR,
                    message=f"event_id {event_id!r} appears {count} times",
                    subject={"run_id": run_id, "event_id": event_id},
                    evidence=[events_rel],
                )
            )


def _check_post_closure_activity(
    all_events: list[dict[str, Any]],
    loaded: dict[str, dict[str, Any]],
    *,
    run_id: str,
    run_root_path: Path,
    findings: list[dict[str, Any]],
) -> bool:
    if "run_final_closure_record" not in loaded:
        return False
    closure_events = [
        event
        for event in all_events
        if event.get("run_id") == run_id
        and event.get("event_type") == events.EVENT_TYPE_RUN_FINAL_CLOSURE_RECORDED
    ]
    if not closure_events:
        return False
    closure_ts = _event_timestamp(closure_events[-1])
    post_event_ids: list[str] = []
    task_attempt_types = {
        events.EVENT_TYPE_TASK_STATUS_CHANGED,
        events.EVENT_TYPE_ATTEMPT_REGISTERED,
        events.EVENT_TYPE_ATTEMPT_STATUS_CHANGED,
        events.EVENT_TYPE_ATTEMPT_RESULT_SUBMITTED,
        events.EVENT_TYPE_MANUAL_VERIFICATION_SUBMITTED,
        events.EVENT_TYPE_MANUAL_TASK_COMPLETED,
    }
    for event in all_events:
        if event.get("run_id") != run_id:
            continue
        if event.get("event_type") not in task_attempt_types:
            continue
        if event.get("task_id") is None:
            continue
        ts = _event_timestamp(event)
        if closure_ts is not None and ts is not None and ts <= closure_ts:
            continue
        event_id = event.get("event_id")
        if isinstance(event_id, str):
            post_event_ids.append(event_id)
    if not post_event_ids:
        return False
    events_rel = _rel_evidence(run_root_path / "task_events.jsonl", run_root_path)
    findings.append(
        _finding(
            code=FINDING_POST_CLOSURE_ACTIVITY,
            severity=SEVERITY_WARNING,
            message=(
                "task/attempt lifecycle events occurred after final closure; "
                "Phase 1 manual run-record chain is terminal but no global hard lock "
                "is enforced"
            ),
            subject={"run_id": run_id, "post_closure_event_ids": sorted(post_event_ids)},
            evidence=[events_rel, "run_final_closure_record.json"],
        )
    )
    return True


def _summarize_manifest(
    run_root_path: Path,
    *,
    run_id: str,
    findings: list[dict[str, Any]],
) -> dict[str, Any] | None:
    manifest_path = run_root_path / "run_manifest.json"
    if not manifest_path.exists():
        return None
    rel = _rel_evidence(manifest_path, run_root_path)
    try:
        manifest = _read_json_object(manifest_path)
        validate_schema(manifest, "run_manifest")
        return {"status": manifest.get("status"), "run_id": manifest.get("run_id")}
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        findings.append(
            _finding(
                code=FINDING_MALFORMED_AUTHORITATIVE_JSON,
                severity=SEVERITY_ERROR,
                message=str(exc),
                subject={"run_id": run_id, "resource": "run_manifest"},
                evidence=[rel],
            )
        )
        return None


def _load_events(
    run_id: str,
    run_root_path: Path,
    findings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    events_path = run_root_path / "task_events.jsonl"
    rel = _rel_evidence(events_path, run_root_path)
    try:
        rows = _read_jsonl_objects(events_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        findings.append(
            _finding(
                code=FINDING_MALFORMED_AUTHORITATIVE_JSON,
                severity=SEVERITY_ERROR,
                message=str(exc),
                subject={"run_id": run_id, "resource": "task_events"},
                evidence=[rel],
            )
        )
        return []
    validated: list[dict[str, Any]] = []
    for index, event in enumerate(rows):
        try:
            validate_schema(event, "event")
        except Exception as exc:
            findings.append(
                _finding(
                    code=FINDING_SCHEMA_VALIDATION_FAILED,
                    severity=SEVERITY_ERROR,
                    message=f"line {index + 1}: {exc}",
                    subject={"run_id": run_id, "resource": "task_events"},
                    evidence=[rel],
                )
            )
            continue
        validated.append(event)
    return validated


def _decision_support(findings: list[dict[str, Any]]) -> dict[str, Any]:
    error_codes = sorted(
        {f["code"] for f in findings if f["severity"] == SEVERITY_ERROR}
    )
    warning_codes = sorted(
        {f["code"] for f in findings if f["severity"] == SEVERITY_WARNING}
    )
    has_errors = bool(error_codes)
    has_warnings = bool(warning_codes)
    return {
        "snapshot_trustworthy": not has_errors,
        "lifecycle_action_eligible": not has_errors,
        "human_checkpoint_recommended": has_errors or has_warnings,
        "integrity_fully_clean": not has_errors and not has_warnings,
        "blocking_finding_codes": error_codes,
        "warning_finding_codes": warning_codes,
    }


def compute_exit_code(snapshot: dict[str, Any], *, strict: bool = False) -> int:
    """Return process exit code for *snapshot*."""
    findings = snapshot.get("integrity", {}).get("findings", [])
    errors = [f for f in findings if f.get("severity") == SEVERITY_ERROR]
    if errors:
        return EXIT_INTEGRITY
    if strict:
        warnings = [f for f in findings if f.get("severity") == SEVERITY_WARNING]
        if warnings:
            return EXIT_INTEGRITY
    return EXIT_OK


def build_run_snapshot(
    run_id: str,
    *,
    base_dir: Path | None = None,
    observed_at: str | None = None,
) -> dict[str, Any]:
    """Build a read-only observation snapshot for *run_id*.

    Raises :class:`ObserveInvocationError` when the run cannot be resolved.
    """
    try:
        run_root_path = _run_root(run_id, base_dir)
    except ValueError as exc:
        raise ObserveInvocationError(str(exc)) from exc

    if not run_root_path.is_dir():
        raise ObserveInvocationError(f"run workspace not found for run_id {run_id!r}")

    _ensure_run_workspace_contained(run_id, run_root_path, base_dir)

    findings: list[dict[str, Any]] = []
    specs = _record_specs()
    all_events = _load_events(run_id, run_root_path, findings)
    _check_duplicate_event_ids(all_events, run_id=run_id, run_root_path=run_root_path, findings=findings)

    chain_entries, loaded = _check_phase1_chain(
        run_id=run_id,
        run_root_path=run_root_path,
        base_dir=base_dir,
        all_events=all_events,
        specs=specs,
        findings=findings,
    )

    post_closure = _check_post_closure_activity(
        all_events, loaded, run_id=run_id, run_root_path=run_root_path, findings=findings
    )

    manifest_summary = _summarize_manifest(run_root_path, run_id=run_id, findings=findings)
    tasks = _discover_tasks(run_root_path, run_id=run_id, findings=findings)

    present_count = sum(1 for entry in chain_entries if entry["present"])
    chain_complete = present_count == len(contracts.PHASE1_MANUAL_WORKFLOW_RECORD_CHAIN)
    terminal_reached = loaded.get(contracts.PHASE1_TERMINAL_RECORD_TYPE) is not None

    findings.sort(
        key=lambda f: (f["severity"], f["code"], json.dumps(f["subject"], sort_keys=True))
    )

    error_count = sum(1 for f in findings if f["severity"] == SEVERITY_ERROR)
    integrity_status = "pass" if error_count == 0 else "fail"

    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "observer": OBSERVER_NAME,
        "observer_version": OBSERVER_VERSION,
        "observed_at": observed_at or _utc_now_iso(),
        "run_id": run_id,
        "phase1_boundary_status": contracts.PHASE1_BOUNDARY_STATUS,
        "run_manifest": manifest_summary,
        "phase1_chain": {
            "terminal_record_type": contracts.PHASE1_TERMINAL_RECORD_TYPE,
            "terminal_event_type": contracts.PHASE1_TERMINAL_EVENT_TYPE,
            "records": chain_entries,
            "chain_complete": chain_complete,
            "terminal_reached": terminal_reached,
        },
        "events": {
            "count": len(all_events),
            "phase1_workflow_event_types_present": sorted(
                {
                    event.get("event_type")
                    for event in all_events
                    if event.get("event_type") in _PHASE1_WORKFLOW_EVENT_TYPES
                }
            ),
        },
        "tasks": tasks,
        "integrity": {
            "status": integrity_status,
            "finding_count": len(findings),
            "error_count": error_count,
            "findings": findings,
        },
        "decision_support": _decision_support(findings),
        "policy_hints": {
            "phase1_chain_terminal": terminal_reached,
            "post_closure_activity_detected": post_closure,
            "global_hard_lock_enforced": False,
        },
    }
