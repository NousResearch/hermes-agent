"""Derived read-only action planning on top of Task 19 observations (Phase 2 Task 21)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from htr import contracts
from htr.observe import (
    FINDING_PHASE1_CHAIN_GAP,
    FINDING_POST_CLOSURE_ACTIVITY,
    FINDING_RECORD_FINGERPRINT_MISMATCH,
    FINDING_SOURCE_CORRESPONDENCE_FAILED,
    FINDING_SOURCE_FINGERPRINT_MISMATCH,
    SEVERITY_ERROR,
)
from htr.schemas import validate as validate_schema

OBSERVATION_PROJECTION_VERSION = "htr.observe.semantic.v1"
PLAN_SCHEMA_VERSION = "1"
PLAN_KIND = "derived_action_plan"
PLANNER_NAME = "htr.action_plan"
PLANNER_VERSION = "1.0.0"
PLAN_DIGEST_PROJECTION_VERSION = "htr.action_plan.digest.v1"
POLICY_C_VERSION = "policy_c_v1"

# Committed Phase 1 path contract (htr/paths.py, htr/events.py):
# APIs name the first positional argument ``project_dir`` but immediately assign
# ``base_dir = Path(project_dir)`` and resolve records under ``runs_root(base_dir)/run_id``.
# That is the HTR runs-storage root — identical to the ``base_dir`` kwarg on earlier
# lifecycle APIs and to observe/plan ``--runs-root``. It is NOT a project repository root.
PROJECT_DIR_SEMANTIC_ROLE = "htr_runs_root"
PROJECT_DIR_BINDING_EXPLICIT = "explicit_input"
PROJECT_DIR_BINDING_OBSERVER = "observer_htr_runs_root"

EXIT_PROPOSABLE = 0
EXIT_PLAN_NOT_ELIGIBLE = 1
EXIT_INVOCATION = 2

STATE_PROPOSABLE = "proposable"
STATE_INPUTS_REQUIRED = "inputs_required"
STATE_BLOCKED_INTEGRITY = "blocked_integrity"
STATE_BLOCKED_FINALIZED = "blocked_finalized"
STATE_BLOCKED_PRECONDITION = "blocked_precondition"
STATE_UNSUPPORTED_ACTION = "unsupported_action"
STATE_RECOVERY_PROTOCOL_REQUIRED = "recovery_protocol_required"
STATE_INDETERMINATE = "indeterminate"

RISK_INFORMATIONAL = "informational"
RISK_LOW = "low"
RISK_MEDIUM = "medium"
RISK_HIGH = "high"
RISK_CRITICAL = "critical"
RISK_INDETERMINATE = "indeterminate"

CONF_HIGH = "high"
CONF_MEDIUM = "medium"
CONF_LOW = "low"
CONF_INDETERMINATE = "indeterminate"

_CHAIN = contracts.PHASE1_MANUAL_WORKFLOW_RECORD_CHAIN

_API_TO_INDEX: dict[str, int] = {
    "complete_run_manually": 0,
    "review_run_manually": 1,
    "plan_run_followup": 2,
    "request_run_execution": 3,
    "execute_run_execution_request": 4,
    "verify_run_execution_result": 5,
    "plan_post_verification_followup": 6,
    "request_post_verification_execution": 7,
    "record_post_verification_execution_result": 8,
    "record_post_verification_execution_verification": 9,
    "record_run_final_closure": 10,
}

_INDEX_TO_API: dict[int, str] = {v: k for k, v in _API_TO_INDEX.items()}

_PHASE1_PLAN_ALLOWLIST: frozenset[str] = frozenset(_API_TO_INDEX)

# (record_field, prior_record_type) derivable from observation fingerprints when prior present.
_SOURCE_FINGERPRINT_LINKS: dict[str, tuple[tuple[str, str], ...]] = {
    "run_execution_request_record": (
        ("source_followup_plan_fingerprint", "run_followup_plan_record"),
    ),
    "run_execution_result_record": (
        ("source_execution_request_fingerprint", "run_execution_request_record"),
    ),
    "run_execution_verification_record": (
        ("source_execution_result_fingerprint", "run_execution_result_record"),
    ),
    "run_post_verification_followup_plan_record": (
        ("source_execution_result_fingerprint", "run_execution_result_record"),
        ("source_execution_verification_fingerprint", "run_execution_verification_record"),
    ),
    "run_post_verification_execution_request_record": (
        (
            "source_post_verification_followup_plan_fingerprint",
            "run_post_verification_followup_plan_record",
        ),
    ),
    "run_post_verification_execution_result_record": (
        ("source_execution_result_fingerprint", "run_execution_result_record"),
        ("source_execution_verification_fingerprint", "run_execution_verification_record"),
        (
            "source_post_verification_followup_plan_fingerprint",
            "run_post_verification_followup_plan_record",
        ),
        (
            "source_post_verification_execution_request_fingerprint",
            "run_post_verification_execution_request_record",
        ),
    ),
    "run_post_verification_execution_verification_record": (
        ("source_execution_result_fingerprint", "run_execution_result_record"),
        ("source_execution_verification_fingerprint", "run_execution_verification_record"),
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
    ),
    "run_final_closure_record": (
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
    ),
}

_CATALOG: dict[str, dict[str, Any]] = {
    "complete_run_manually": {
        "module": "htr.events",
        "lifecycle_level": "run",
        "record_type": "run_completion_record",
        "schema_name": "run_completion_record",
        "expected_records": ["run_completion_record"],
        "expected_events": ["manual_run_completed"],
        "risk_class": RISK_MEDIUM,
        "risk_reason_codes": ["RISK_RUN_COMPLETION"],
        "reversibility_class": RISK_LOW,
        "reversibility_reason_codes": ["REV_AUDIT_TRAIL_APPEND"],
        "uses_project_dir": False,
        "requires_record": True,
        "requires_actor": True,
        "requires_executor": False,
    },
    "review_run_manually": {
        "module": "htr.events",
        "lifecycle_level": "run",
        "record_type": "run_review_record",
        "schema_name": "run_review_record",
        "expected_records": ["run_review_record"],
        "expected_events": ["manual_run_reviewed"],
        "risk_class": RISK_MEDIUM,
        "risk_reason_codes": ["RISK_HUMAN_REVIEW_RECORD"],
        "reversibility_class": RISK_LOW,
        "reversibility_reason_codes": ["REV_AUDIT_TRAIL_APPEND"],
        "uses_project_dir": False,
        "requires_record": True,
        "requires_actor": True,
        "requires_executor": False,
    },
    "plan_run_followup": {
        "module": "htr.events",
        "lifecycle_level": "run",
        "record_type": "run_followup_plan_record",
        "schema_name": "run_followup_plan_record",
        "expected_records": ["run_followup_plan_record"],
        "expected_events": ["manual_run_followup_planned"],
        "risk_class": RISK_MEDIUM,
        "risk_reason_codes": ["RISK_FOLLOWUP_PLAN"],
        "reversibility_class": RISK_LOW,
        "reversibility_reason_codes": ["REV_AUDIT_TRAIL_APPEND"],
        "uses_project_dir": False,
        "requires_record": True,
        "requires_actor": True,
        "requires_executor": False,
    },
    "request_run_execution": {
        "module": "htr.events",
        "lifecycle_level": "run",
        "record_type": "run_execution_request_record",
        "schema_name": "run_execution_request_record",
        "expected_records": ["run_execution_request_record"],
        "expected_events": ["run_execution_requested"],
        "risk_class": RISK_HIGH,
        "risk_reason_codes": ["RISK_EXECUTION_REQUEST"],
        "reversibility_class": RISK_LOW,
        "reversibility_reason_codes": ["REV_AUDIT_TRAIL_APPEND"],
        "uses_project_dir": False,
        "requires_record": True,
        "requires_actor": True,
        "requires_executor": False,
    },
    "execute_run_execution_request": {
        "module": "htr.events",
        "lifecycle_level": "run",
        "record_type": None,
        "schema_name": None,
        "expected_records": ["run_execution_result_record"],
        "expected_events": ["run_execution_completed"],
        "risk_class": RISK_HIGH,
        "risk_reason_codes": ["RISK_CONTROLLED_EXECUTION"],
        "reversibility_class": RISK_MEDIUM,
        "reversibility_reason_codes": ["REV_EXECUTION_SIDE_EFFECTS_POSSIBLE"],
        "uses_project_dir": True,
        "requires_record": False,
        "requires_actor": False,
        "requires_executor": True,
    },
    "verify_run_execution_result": {
        "module": "htr.events",
        "lifecycle_level": "run",
        "record_type": "run_execution_verification_record",
        "schema_name": "run_execution_verification_record",
        "expected_records": ["run_execution_verification_record"],
        "expected_events": [
            "run_execution_verified",
            "run_execution_rejected",
            "run_execution_needs_changes",
        ],
        "risk_class": RISK_HIGH,
        "risk_reason_codes": ["RISK_EXECUTION_VERIFICATION"],
        "reversibility_class": RISK_LOW,
        "reversibility_reason_codes": ["REV_AUDIT_TRAIL_APPEND"],
        "uses_project_dir": True,
        "requires_record": True,
        "requires_actor": True,
        "requires_executor": False,
    },
    "plan_post_verification_followup": {
        "module": "htr.events",
        "lifecycle_level": "run",
        "record_type": "run_post_verification_followup_plan_record",
        "schema_name": "run_post_verification_followup_plan_record",
        "expected_records": ["run_post_verification_followup_plan_record"],
        "expected_events": ["run_post_verification_followup_planned"],
        "risk_class": RISK_MEDIUM,
        "risk_reason_codes": ["RISK_PV_FOLLOWUP_PLAN"],
        "reversibility_class": RISK_LOW,
        "reversibility_reason_codes": ["REV_AUDIT_TRAIL_APPEND"],
        "uses_project_dir": True,
        "requires_record": True,
        "requires_actor": True,
        "requires_executor": False,
    },
    "request_post_verification_execution": {
        "module": "htr.events",
        "lifecycle_level": "run",
        "record_type": "run_post_verification_execution_request_record",
        "schema_name": "run_post_verification_execution_request_record",
        "expected_records": ["run_post_verification_execution_request_record"],
        "expected_events": ["run_post_verification_execution_requested"],
        "risk_class": RISK_HIGH,
        "risk_reason_codes": ["RISK_PV_EXECUTION_REQUEST"],
        "reversibility_class": RISK_LOW,
        "reversibility_reason_codes": ["REV_AUDIT_TRAIL_APPEND"],
        "uses_project_dir": True,
        "requires_record": True,
        "requires_actor": True,
        "requires_executor": False,
    },
    "record_post_verification_execution_result": {
        "module": "htr.events",
        "lifecycle_level": "run",
        "record_type": "run_post_verification_execution_result_record",
        "schema_name": "run_post_verification_execution_result_record",
        "expected_records": ["run_post_verification_execution_result_record"],
        "expected_events": ["run_post_verification_execution_result_recorded"],
        "risk_class": RISK_HIGH,
        "risk_reason_codes": ["RISK_PV_EXECUTION_RESULT"],
        "reversibility_class": RISK_LOW,
        "reversibility_reason_codes": ["REV_AUDIT_TRAIL_APPEND"],
        "uses_project_dir": True,
        "requires_record": True,
        "requires_actor": True,
        "requires_executor": False,
    },
    "record_post_verification_execution_verification": {
        "module": "htr.events",
        "lifecycle_level": "run",
        "record_type": "run_post_verification_execution_verification_record",
        "schema_name": "run_post_verification_execution_verification_record",
        "expected_records": ["run_post_verification_execution_verification_record"],
        "expected_events": ["run_post_verification_execution_verification_recorded"],
        "risk_class": RISK_HIGH,
        "risk_reason_codes": ["RISK_PV_EXECUTION_VERIFICATION"],
        "reversibility_class": RISK_LOW,
        "reversibility_reason_codes": ["REV_AUDIT_TRAIL_APPEND"],
        "uses_project_dir": True,
        "requires_record": True,
        "requires_actor": True,
        "requires_executor": False,
    },
    "record_run_final_closure": {
        "module": "htr.events",
        "lifecycle_level": "run",
        "record_type": "run_final_closure_record",
        "schema_name": "run_final_closure_record",
        "expected_records": ["run_final_closure_record"],
        "expected_events": ["run_final_closure_recorded"],
        "risk_class": RISK_CRITICAL,
        "risk_reason_codes": ["RISK_FINAL_CLOSURE"],
        "reversibility_class": RISK_LOW,
        "reversibility_reason_codes": ["REV_IRREVERSIBLE_TERMINAL"],
        "uses_project_dir": True,
        "requires_record": True,
        "requires_actor": True,
        "requires_executor": False,
    },
}

_CLOSURE_ERROR_CODES = frozenset(
    {
        FINDING_PHASE1_CHAIN_GAP,
        FINDING_RECORD_FINGERPRINT_MISMATCH,
        FINDING_SOURCE_FINGERPRINT_MISMATCH,
        FINDING_SOURCE_CORRESPONDENCE_FAILED,
    }
)


@dataclass(frozen=True)
class PlanningIntent:
    """Explicit caller planning intent for Task 21."""

    requested_action: str | None = None
    action_inputs: dict[str, Any] | None = None
    project_repository_checkpoint: str | None = None
    htr_runs_root: str | None = None
    remediation_oriented: bool = False


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_digest(obj: Any) -> str:
    payload = _canonical_json(obj).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _normalize_finding(finding: dict[str, Any]) -> dict[str, Any]:
    subject = finding.get("subject")
    if not isinstance(subject, dict):
        subject = {}
    return {
        "code": finding.get("code"),
        "severity": finding.get("severity"),
        "subject": dict(sorted(subject.items())),
    }


def _chain_records_map(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = snapshot.get("phase1_chain", {}).get("records", [])
    return {entry["record_type"]: entry for entry in records if isinstance(entry, dict)}


def _integrity_findings(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    return list(snapshot.get("integrity", {}).get("findings", []))


def _snapshot_trustworthy(snapshot: dict[str, Any]) -> bool:
    return bool(snapshot.get("decision_support", {}).get("snapshot_trustworthy"))


def _terminal_reached(snapshot: dict[str, Any]) -> bool:
    return bool(snapshot.get("phase1_chain", {}).get("terminal_reached"))


def _has_chain_gap(snapshot: dict[str, Any]) -> bool:
    return any(f.get("code") == FINDING_PHASE1_CHAIN_GAP for f in _integrity_findings(snapshot))


def _closure_trustworthy(snapshot: dict[str, Any]) -> bool:
    if not _terminal_reached(snapshot):
        return False
    if not _snapshot_trustworthy(snapshot):
        return False
    for finding in _integrity_findings(snapshot):
        if finding.get("severity") != SEVERITY_ERROR:
            continue
        subject = finding.get("subject") or {}
        record_type = subject.get("record_type")
        if record_type == "run_final_closure_record":
            return False
        if finding.get("code") in _CLOSURE_ERROR_CODES:
            if record_type in (None, "run_final_closure_record") or _has_chain_gap(snapshot):
                return False
    return True


def _post_closure_activity_detected(snapshot: dict[str, Any]) -> bool:
    return bool(snapshot.get("policy_hints", {}).get("post_closure_activity_detected"))


def _first_absent_chain_index(snapshot: dict[str, Any]) -> int | None:
    chain_map = _chain_records_map(snapshot)
    for index, record_type in enumerate(_CHAIN):
        entry = chain_map.get(record_type, {})
        if not entry.get("present"):
            return index
    return None


def _derive_fingerprints_from_chain(
    snapshot: dict[str, Any], record_type: str
) -> dict[str, str]:
    chain_map = _chain_records_map(snapshot)
    derived: dict[str, str] = {}
    for field_name, prior_type in _SOURCE_FINGERPRINT_LINKS.get(record_type, ()):
        prior = chain_map.get(prior_type, {})
        if prior.get("present") and prior.get("fingerprint"):
            derived[field_name] = prior["fingerprint"]
    return derived


def _normalize_path_for_digest(path: str) -> str:
    """Normalize a filesystem path string for digest binding (no I/O)."""
    return str(Path(path).as_posix())


def _project_dir_path_digest(path: str) -> str:
    return _sha256_digest({"normalized_path": _normalize_path_for_digest(path)})


def resolve_project_dir_binding(
    *,
    explicit_project_dir: str | None,
    htr_runs_root: str | None,
) -> tuple[dict[str, Any] | None, list[dict[str, str]], list[str]]:
    """Resolve canonical ``project_dir`` binding without filesystem access."""
    missing: list[dict[str, str]] = []
    errors: list[str] = []

    explicit = explicit_project_dir.strip() if isinstance(explicit_project_dir, str) else None
    if explicit == "":
        explicit = None

    if explicit is not None and htr_runs_root is not None:
        if _normalize_path_for_digest(explicit) != _normalize_path_for_digest(htr_runs_root):
            errors.append(
                "explicit project_dir does not match observer htr_runs_root path contract"
            )
            return None, missing, errors

    if explicit is not None:
        return (
            {
                "canonical_api_parameter": "project_dir",
                "semantic_role": PROJECT_DIR_SEMANTIC_ROLE,
                "binding": PROJECT_DIR_BINDING_EXPLICIT,
                "path_digest": _project_dir_path_digest(explicit),
            },
            missing,
            errors,
        )

    if htr_runs_root:
        return (
            {
                "canonical_api_parameter": "project_dir",
                "semantic_role": PROJECT_DIR_SEMANTIC_ROLE,
                "binding": PROJECT_DIR_BINDING_OBSERVER,
            },
            missing,
            errors,
        )

    missing.append({"path": "project_dir", "kind": "invocation_context"})
    return None, missing, errors


def _build_idempotency_summary(supplied_args: dict[str, Any]) -> dict[str, Any]:
    event_id = supplied_args.get("event_id")
    supplied = isinstance(event_id, str) and bool(event_id.strip())
    prereqs: list[str] = []
    if not supplied:
        prereqs.append("EVENT_ID_ALLOCATED_AT_INVOKE_IF_OMITTED")
    return {
        "event_id_supplied": supplied,
        "exact_event_identity_bound": supplied,
        "invoke_time_allocation_required": not supplied,
        "prerequisite_codes": prereqs,
    }


def project_semantic_observation(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Build the versioned semantic projection used for planning digests."""
    run_manifest = snapshot.get("run_manifest")
    manifest_summary = None
    if isinstance(run_manifest, dict):
        manifest_summary = {
            "run_id": run_manifest.get("run_id"),
            "status": run_manifest.get("status"),
        }

    chain = snapshot.get("phase1_chain", {})
    records = []
    for entry in chain.get("records", []):
        if not isinstance(entry, dict):
            continue
        records.append(
            {
                "record_type": entry.get("record_type"),
                "present": entry.get("present"),
                "fingerprint": entry.get("fingerprint"),
                "matching_event_count": entry.get("matching_event_count"),
            }
        )
    records.sort(key=lambda item: item.get("record_type") or "")

    findings = [_normalize_finding(f) for f in _integrity_findings(snapshot)]
    findings.sort(
        key=lambda f: (f.get("severity", ""), f.get("code", ""), _canonical_json(f.get("subject", {})))
    )

    decision_support = snapshot.get("decision_support", {})
    policy_hints = snapshot.get("policy_hints", {})

    tasks = []
    for task in snapshot.get("tasks", []):
        if not isinstance(task, dict):
            continue
        attempts = []
        for attempt in task.get("attempts", []):
            if not isinstance(attempt, dict):
                continue
            attempts.append(
                {
                    "attempt_id": attempt.get("attempt_id"),
                    "attempt_status_present": attempt.get("attempt_status_present"),
                    "status": attempt.get("status"),
                }
            )
        tasks.append(
            {
                "task_id": task.get("task_id"),
                "task_status_present": task.get("task_status_present"),
                "status": task.get("status"),
                "attempts": attempts,
            }
        )

    return {
        "projection_version": OBSERVATION_PROJECTION_VERSION,
        "run_id": snapshot.get("run_id"),
        "phase1_boundary_status": snapshot.get("phase1_boundary_status"),
        "run_manifest": manifest_summary,
        "phase1_chain": {
            "terminal_record_type": chain.get("terminal_record_type"),
            "terminal_event_type": chain.get("terminal_event_type"),
            "chain_complete": chain.get("chain_complete"),
            "terminal_reached": chain.get("terminal_reached"),
            "records": records,
        },
        "integrity": {
            "status": snapshot.get("integrity", {}).get("status"),
            "error_count": snapshot.get("integrity", {}).get("error_count"),
            "findings": findings,
        },
        "decision_support": {
            "snapshot_trustworthy": decision_support.get("snapshot_trustworthy"),
            "lifecycle_action_eligible": decision_support.get("lifecycle_action_eligible"),
            "human_checkpoint_recommended": decision_support.get(
                "human_checkpoint_recommended"
            ),
            "integrity_fully_clean": decision_support.get("integrity_fully_clean"),
            "blocking_finding_codes": sorted(
                decision_support.get("blocking_finding_codes", [])
            ),
            "warning_finding_codes": sorted(
                decision_support.get("warning_finding_codes", [])
            ),
        },
        "policy_hints": {
            "phase1_chain_terminal": policy_hints.get("phase1_chain_terminal"),
            "post_closure_activity_detected": policy_hints.get(
                "post_closure_activity_detected"
            ),
            "global_hard_lock_enforced": policy_hints.get("global_hard_lock_enforced"),
        },
        "tasks": tasks,
    }


def compute_source_observation_digest(snapshot: dict[str, Any]) -> str:
    """Return the derived planning digest for *snapshot*."""
    return _sha256_digest(project_semantic_observation(snapshot))


def infer_structural_next_action(snapshot: dict[str, Any]) -> dict[str, Any] | None:
    """Return a structural next-action hint when chain state is trustworthy."""
    if not _snapshot_trustworthy(snapshot):
        return None
    if _has_chain_gap(snapshot):
        return None
    if _terminal_reached(snapshot) and _closure_trustworthy(snapshot):
        return None
    index = _first_absent_chain_index(snapshot)
    if index is None:
        return None
    api = _INDEX_TO_API[index]
    return {
        "api": api,
        "record_type": _CHAIN[index],
        "chain_index": index,
    }


def _completed_task_ids(snapshot: dict[str, Any]) -> list[str]:
    completed: list[str] = []
    for task in snapshot.get("tasks", []):
        if task.get("status") == "completed":
            completed.append(task["task_id"])
    return sorted(completed)


def _preconditions_for_api(api: str, snapshot: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return (preconditions, failed_preconditions)."""
    passed: list[str] = []
    failed: list[str] = []
    chain_map = _chain_records_map(snapshot)
    manifest = snapshot.get("run_manifest") or {}

    def _req(name: str, ok: bool) -> None:
        if ok:
            passed.append(name)
        else:
            failed.append(name)

    if api == "complete_run_manually":
        _req("run_manifest_present", manifest.get("status") is not None)
        _req("completed_tasks_available", bool(_completed_task_ids(snapshot)))
        _req("run_completion_record_absent", not chain_map.get("run_completion_record", {}).get("present"))
        return passed, failed

    _req("run_manifest_completed", manifest.get("status") == "completed")

    index = _API_TO_INDEX[api]
    for prior_type in _CHAIN[:index]:
        _req(f"{prior_type}_present", chain_map.get(prior_type, {}).get("present", False))

    record_type = _CATALOG[api]["record_type"]
    if record_type:
        _req(f"{record_type}_absent", not chain_map.get(record_type, {}).get("present"))

    if api == "execute_run_execution_request":
        _req(
            "run_execution_request_record_present",
            chain_map.get("run_execution_request_record", {}).get("present", False),
        )

    return passed, failed


def _structural_index_for_action(api: str) -> int:
    return _API_TO_INDEX[api]


def _chain_position_ok(api: str, snapshot: dict[str, Any]) -> bool:
    if _has_chain_gap(snapshot):
        return False
    expected = _structural_index_for_action(api)
    chain_map = _chain_records_map(snapshot)
    if chain_map.get(_CHAIN[expected], {}).get("present"):
        return False
    first_absent = _first_absent_chain_index(snapshot)
    if first_absent is None:
        return False
    return first_absent == expected


def _validate_record_inputs(
    api: str,
    record: dict[str, Any],
    run_id: str,
    snapshot: dict[str, Any],
) -> tuple[list[str], list[str], dict[str, str], list[dict[str, str]]]:
    """Return missing_required, validation_errors, derived_args, mismatch fields."""
    catalog = _CATALOG[api]
    schema_name = catalog["schema_name"]
    missing: list[dict[str, str]] = []
    errors: list[str] = []
    derived: dict[str, str] = {"run_id": run_id}

    if record.get("run_id") != run_id:
        errors.append("record.run_id does not match observed run_id")

    try:
        validate_schema(record, schema_name)
    except Exception as exc:
        errors.append(f"schema validation failed: {exc}")
        return missing, errors, derived, []

    record_type = catalog["record_type"]
    assert record_type is not None
    expected_derived = _derive_fingerprints_from_chain(snapshot, record_type)
    for field_name, expected_fp in expected_derived.items():
        derived[field_name] = expected_fp
        actual = record.get(field_name)
        if actual is None:
            missing.append({"path": field_name, "kind": "derived_fingerprint"})
        elif actual != expected_fp:
            errors.append(
                f"{field_name} does not match trusted prior fingerprint from observation"
            )

    return missing, errors, derived, []


def _parse_action_inputs(
    api: str, action_inputs: dict[str, Any] | None
) -> tuple[dict[str, Any] | None, str | None, str | None, list[dict[str, str]], list[str]]:
    """Extract record, actor, executor, missing fields, and validation errors."""
    missing: list[dict[str, str]] = []
    errors: list[str] = []
    if action_inputs is None:
        action_inputs = {}

    if not isinstance(action_inputs, dict):
        return None, None, None, missing, ["action_inputs must be a JSON object"]

    catalog = _CATALOG[api]
    record = action_inputs.get("record")
    actor = action_inputs.get("actor")
    executor = action_inputs.get("executor")

    if catalog["requires_record"]:
        if record is None:
            missing.append({"path": "record", "kind": "semantic_input"})
        elif not isinstance(record, dict):
            errors.append("record must be a JSON object")
            record = None

    if catalog["requires_actor"]:
        if not actor:
            missing.append({"path": "actor", "kind": "control_input"})

    if catalog["requires_executor"]:
        if not executor:
            missing.append({"path": "executor", "kind": "control_input"})

    optional_keys = {"record", "actor", "executor", "event_id", "notes", "metadata"}
    if catalog.get("uses_project_dir"):
        optional_keys = optional_keys | {"project_dir"}
    extra = set(action_inputs) - optional_keys
    if extra:
        errors.append(f"unsupported action_inputs keys: {sorted(extra)}")

    return record, actor, executor, missing, errors, action_inputs.get("project_dir")


def _confidence_for_state(
    state: str,
    *,
    has_missing: bool,
    has_warnings: bool,
) -> tuple[str, list[str]]:
    if state in (STATE_BLOCKED_INTEGRITY, STATE_INDETERMINATE):
        return CONF_INDETERMINATE, ["CONF_INTEGRITY_OR_TRUST_FAILURE"]
    if state in (
        STATE_BLOCKED_FINALIZED,
        STATE_RECOVERY_PROTOCOL_REQUIRED,
        STATE_UNSUPPORTED_ACTION,
        STATE_BLOCKED_PRECONDITION,
    ):
        return CONF_LOW, ["CONF_POLICY_OR_PRECONDITION_BLOCK"]
    if state == STATE_INPUTS_REQUIRED or has_missing:
        return CONF_MEDIUM, ["CONF_MISSING_EXPLICIT_INPUTS"]
    if has_warnings:
        return CONF_MEDIUM, ["CONF_ADVISORY_WARNINGS"]
    if state == STATE_PROPOSABLE:
        return CONF_HIGH, ["CONF_INTEGRITY_CLEAN", "CONF_INPUTS_COMPLETE"]
    return CONF_INDETERMINATE, ["CONF_INDETERMINATE"]


def _risk_for_api(api: str | None, *, integrity_unknown: bool) -> tuple[str, list[str]]:
    if integrity_unknown or api is None:
        return RISK_INDETERMINATE, ["RISK_INTEGRITY_UNKNOWN"]
    catalog = _CATALOG[api]
    return catalog["risk_class"], list(catalog["risk_reason_codes"])


def _execution_eligible(state: str, snapshot: dict[str, Any]) -> bool:
    return (
        state == STATE_PROPOSABLE
        and _snapshot_trustworthy(snapshot)
        and not _terminal_reached(snapshot)
    )


def _plan_digest_projection(plan: dict[str, Any]) -> dict[str, Any]:
    source = plan.get("source", {})
    candidate = plan.get("candidate_action") or {}
    arguments = plan.get("arguments", {})
    automation = plan.get("automation_eligibility") or {}
    derived = dict(arguments.get("derived") or {})
    project_dir_binding = derived.pop("project_dir_binding", None)
    supplied = dict(arguments.get("supplied") or {})
    return {
        "plan_digest_projection_version": PLAN_DIGEST_PROJECTION_VERSION,
        "plan_schema_version": plan.get("plan_schema_version"),
        "plan_kind": plan.get("plan_kind"),
        "policy_c_version": POLICY_C_VERSION,
        "plan_state": plan.get("plan_state"),
        "plan_state_reason_codes": sorted(plan.get("plan_state_reason_codes") or []),
        "run_id": plan.get("run_id"),
        "source_observation_digest": source.get("source_observation_digest"),
        "project_repository_checkpoint": source.get("project_repository_checkpoint"),
        "htr_runs_root_observed": source.get("htr_runs_root_observed"),
        "requested_action": (plan.get("requested_intent") or {}).get("requested_action"),
        "candidate_api": candidate.get("api"),
        "supplied_arguments": supplied,
        "derived_arguments": derived,
        "project_dir_binding": project_dir_binding,
        "missing_required_inputs": sorted(
            arguments.get("missing_required") or [],
            key=lambda item: (item.get("path", ""), item.get("kind", "")),
        ),
        "preconditions": sorted(plan.get("preconditions") or []),
        "failed_preconditions": sorted(plan.get("failed_preconditions") or []),
        "expected_records": sorted(plan.get("expected_records") or []),
        "expected_events": sorted(plan.get("expected_events") or []),
        "expected_postconditions": sorted(plan.get("expected_postconditions") or []),
        "verification_requirements": sorted(plan.get("verification_requirements") or []),
        "risk_class": (plan.get("risk") or {}).get("class"),
        "risk_reason_codes": sorted((plan.get("risk") or {}).get("reason_codes") or []),
        "confidence_class": (plan.get("confidence") or {}).get("class"),
        "confidence_reason_codes": sorted(
            (plan.get("confidence") or {}).get("reason_codes") or []
        ),
        "approval_required": plan.get("approval_required"),
        "reversibility_class": (plan.get("reversibility") or {}).get("class"),
        "reversibility_reason_codes": sorted(
            (plan.get("reversibility") or {}).get("reason_codes") or []
        ),
        "recovery_limitation": plan.get("recovery_limitation"),
        "execution_prerequisites": sorted(plan.get("execution_prerequisites") or []),
        "execution_eligibility_reason_codes": sorted(automation.get("reason_codes") or []),
        "escalation_reason_codes": sorted(plan.get("escalation_reason_codes") or []),
        "idempotency": plan.get("idempotency"),
    }


def compute_plan_digest(plan: dict[str, Any]) -> str:
    """Return the derived digest for a normalized plan."""
    return _sha256_digest(_plan_digest_projection(plan))


def _resolve_canonical_state(
    snapshot: dict[str, Any],
    intent: PlanningIntent,
    *,
    structural_hint: dict[str, Any] | None,
    unsupported: bool,
    position_ok: bool,
    missing_inputs: list[dict[str, str]],
    validation_errors: list[str],
    mutation_requested: bool,
) -> tuple[str, list[str]]:
    reason_codes: list[str] = []

    if not _snapshot_trustworthy(snapshot):
        reason_codes.append("BLOCK_INTEGRITY_ERRORS")
        return STATE_BLOCKED_INTEGRITY, reason_codes

    if _terminal_reached(snapshot) and not _closure_trustworthy(snapshot):
        reason_codes.append("CLOSURE_UNTRUSTWORTHY")
        return STATE_INDETERMINATE, reason_codes

    if mutation_requested and _terminal_reached(snapshot) and _closure_trustworthy(snapshot):
        if intent.remediation_oriented:
            reason_codes.extend(
                ["POLICY_C_FINALIZED", "REMEDIATION_REQUIRES_SUCCESSOR_PROTOCOL"]
            )
            return STATE_RECOVERY_PROTOCOL_REQUIRED, reason_codes
        reason_codes.append("POLICY_C_FINALIZED_ORIGINAL_IMMUTABLE")
        return STATE_BLOCKED_FINALIZED, reason_codes

    if intent.requested_action and unsupported:
        reason_codes.append("ACTION_NOT_IN_CATALOG")
        return STATE_UNSUPPORTED_ACTION, reason_codes

    if intent.requested_action and not position_ok:
        reason_codes.append("ACTION_NOT_AT_STRUCTURAL_CHAIN_POSITION")
        return STATE_BLOCKED_PRECONDITION, reason_codes

    if validation_errors:
        if intent.requested_action:
            reason_codes.append("INPUT_VALIDATION_FAILED")
            return STATE_BLOCKED_PRECONDITION, reason_codes

    if intent.requested_action is None or missing_inputs:
        if intent.requested_action is None:
            reason_codes.append("REQUESTED_ACTION_MISSING")
        else:
            reason_codes.append("REQUIRED_SEMANTIC_INPUTS_MISSING")
        return STATE_INPUTS_REQUIRED, reason_codes

    reason_codes.append("PLAN_COMPLETE")
    return STATE_PROPOSABLE, reason_codes


def build_action_plan(snapshot: dict[str, Any], intent: PlanningIntent) -> dict[str, Any]:
    """Build a normalized derived action plan from *snapshot* and *intent*."""
    run_id = snapshot.get("run_id", "")
    source_digest = compute_source_observation_digest(snapshot)
    structural_hint = infer_structural_next_action(snapshot)
    findings = _integrity_findings(snapshot)
    blocking_codes = sorted(
        {f["code"] for f in findings if f.get("severity") == SEVERITY_ERROR}
    )
    warning_codes = sorted(
        {f["code"] for f in findings if f.get("severity") == "warning"}
    )

    api = intent.requested_action
    unsupported = bool(api and api not in _PHASE1_PLAN_ALLOWLIST)
    catalog = _CATALOG.get(api or "", {})

    missing_inputs: list[dict[str, str]] = []
    validation_errors: list[str] = []
    derived_args: dict[str, Any] = {}
    supplied_args: dict[str, Any] = {}

    position_ok = False
    preconditions: list[str] = []
    failed_preconditions: list[str] = []
    expected_records: list[str] = []
    expected_events: list[str] = []

    mutation_requested = bool(api)

    if api and not unsupported:
        preconditions, failed_preconditions = _preconditions_for_api(api, snapshot)
        position_ok = _chain_position_ok(api, snapshot) and not failed_preconditions
        expected_records = list(catalog.get("expected_records", []))
        expected_events = list(catalog.get("expected_events", []))

        record, actor, executor, parse_missing, parse_errors, explicit_project_dir = (
            _parse_action_inputs(api, intent.action_inputs)
        )
        missing_inputs.extend(parse_missing)
        validation_errors.extend(parse_errors)

        if record is not None:
            supplied_args["record"] = record
            rec_missing, rec_errors, rec_derived, _ = _validate_record_inputs(
                api, record, run_id, snapshot
            )
            missing_inputs.extend(rec_missing)
            validation_errors.extend(rec_errors)
            derived_args.update(rec_derived)
        if actor:
            supplied_args["actor"] = actor
        if executor:
            supplied_args["executor"] = executor
        if intent.action_inputs:
            for key in ("event_id", "notes", "metadata"):
                if key in intent.action_inputs:
                    supplied_args[key] = intent.action_inputs[key]

        if catalog.get("uses_project_dir"):
            binding, pd_missing, pd_errors = resolve_project_dir_binding(
                explicit_project_dir=explicit_project_dir,
                htr_runs_root=intent.htr_runs_root,
            )
            missing_inputs.extend(pd_missing)
            validation_errors.extend(pd_errors)
            if binding is not None:
                derived_args["project_dir_binding"] = binding

        derived_args["run_id"] = run_id

    state, state_reason_codes = _resolve_canonical_state(
        snapshot,
        intent,
        structural_hint=structural_hint,
        unsupported=unsupported,
        position_ok=position_ok,
        missing_inputs=missing_inputs,
        validation_errors=validation_errors,
        mutation_requested=mutation_requested,
    )

    has_warnings = bool(warning_codes)
    conf_class, conf_codes = _confidence_for_state(
        state,
        has_missing=bool(missing_inputs),
        has_warnings=has_warnings,
    )
    risk_class, risk_codes = _risk_for_api(
        api if api and not unsupported else None,
        integrity_unknown=not _snapshot_trustworthy(snapshot),
    )

    if state == STATE_BLOCKED_INTEGRITY and _post_closure_activity_detected(snapshot):
        state_reason_codes.append("FUTURE_SUCCESSOR_RECOVERY_MAY_BE_REQUIRED")

    idempotency = _build_idempotency_summary(supplied_args)
    execution_prerequisites = [
        "task_22_immutable_finalized_run_seal",
        "task_23_execution_lock",
        "task_24_authoritative_approval",
        "task_25_human_gated_invoke",
    ]
    execution_prerequisites.extend(idempotency["prerequisite_codes"])

    plan: dict[str, Any] = {
        "plan_schema_version": PLAN_SCHEMA_VERSION,
        "plan_kind": PLAN_KIND,
        "planner": {"name": PLANNER_NAME, "version": PLANNER_VERSION},
        "plan_state": state,
        "plan_state_reason_codes": state_reason_codes,
        "run_id": run_id,
        "task_id": None,
        "attempt_id": None,
        "source": {
            "observation_projection_version": OBSERVATION_PROJECTION_VERSION,
            "source_observation_digest": source_digest,
            "planner_software_checkpoint": "unknown",
            "project_repository_checkpoint": intent.project_repository_checkpoint,
            "htr_runs_root_observed": intent.htr_runs_root is not None,
        },
        "integrity_summary": {
            "status": snapshot.get("integrity", {}).get("status"),
            "error_count": snapshot.get("integrity", {}).get("error_count"),
            "blocking_codes": blocking_codes,
            "warning_codes": warning_codes,
        },
        "finalized_run_policy": {
            "applied": POLICY_C_VERSION,
            "terminal_reached": _terminal_reached(snapshot),
            "closure_trustworthy": _closure_trustworthy(snapshot),
            "seal_implemented": False,
            "recovery_protocol_implemented": False,
        },
        "requested_intent": {
            "requested_action": intent.requested_action,
            "structural_hint_only": intent.requested_action is None,
        },
        "structural_next_action": structural_hint,
        "candidate_action": (
            {
                "api": api,
                "module": catalog.get("module"),
                "lifecycle_level": catalog.get("lifecycle_level"),
            }
            if api and not unsupported
            else None
        ),
        "arguments": {
            "supplied": supplied_args,
            "derived": derived_args,
            "missing_required": missing_inputs,
        },
        "expected_records": expected_records,
        "expected_events": expected_events,
        "preconditions": preconditions,
        "failed_preconditions": failed_preconditions,
        "expected_postconditions": expected_records,
        "verification_requirements": [
            "re_observe_integrity_pass",
            "matching_audit_event",
            "record_fingerprint_match",
        ],
        "risk": {"class": risk_class, "reason_codes": risk_codes},
        "confidence": {"class": conf_class, "reason_codes": conf_codes},
        "reversibility": {
            "class": (
                catalog.get("reversibility_class", RISK_INDETERMINATE)
                if api and not unsupported
                else RISK_INDETERMINATE
            ),
            "reason_codes": (
                list(catalog.get("reversibility_reason_codes", []))
                if api and not unsupported
                else ["REV_UNKNOWN"]
            ),
        },
        "recovery_limitation": (
            "No in-place finalized-run recovery; Recovery/Successor Run protocol "
            "not implemented (Task 27)."
        ),
        "approval_required": True,
        "execution_prerequisites": execution_prerequisites,
        "idempotency": idempotency,
        "proposable_completeness": {
            "semantic_record_inputs_complete": state == STATE_PROPOSABLE,
            "canonical_api_arguments_bound": state == STATE_PROPOSABLE,
            "exact_event_identity_bound": idempotency["exact_event_identity_bound"],
            "single_machine_path_binding": True,
        },
        "automation_eligibility": {
            "planning_only": True,
            "execution_eligible": _execution_eligible(state, snapshot),
            "reason_codes": (
                ["EXEC_PREREQUISITES_NOT_MET"]
                if not _execution_eligible(state, snapshot)
                else ["EXEC_STRUCTURALLY_COMPLETE_BUT_NOT_AUTHORIZED"]
            ),
        },
        "escalation_reason_codes": (
            ["recovery_protocol_not_implemented"]
            if state == STATE_RECOVERY_PROTOCOL_REQUIRED
            else []
        ),
        "expiration_policy": None,
    }

    if intent.project_repository_checkpoint is None:
        plan.setdefault("confidence", {})["reason_codes"] = list(
            dict.fromkeys(
                plan["confidence"]["reason_codes"] + ["CONF_PROJECT_CHECKPOINT_UNKNOWN"]
            )
        )

    plan["plan_digest"] = compute_plan_digest(plan)
    return plan


def compute_plan_exit_code(plan: dict[str, Any]) -> int:
    """Return CLI exit code for a normalized *plan*."""
    if plan.get("plan_state") == STATE_PROPOSABLE:
        return EXIT_PROPOSABLE
    return EXIT_PLAN_NOT_ELIGIBLE


def plan_run(
    run_id: str,
    intent: PlanningIntent,
    *,
    base_dir: Any = None,
) -> dict[str, Any]:
    """Convenience wrapper: observe then plan."""
    from htr.observe import build_run_snapshot

    snapshot = build_run_snapshot(run_id, base_dir=base_dir)
    runs_root = str(base_dir) if base_dir is not None else intent.htr_runs_root
    enriched = PlanningIntent(
        requested_action=intent.requested_action,
        action_inputs=intent.action_inputs,
        project_repository_checkpoint=intent.project_repository_checkpoint,
        htr_runs_root=runs_root,
        remediation_oriented=intent.remediation_oriented,
    )
    return build_action_plan(snapshot, enriched)


def make_invocation_error(code: str, message: str) -> dict[str, Any]:
    """Build a minimal invocation-failure envelope without plan digest."""
    return {
        "plan_schema_version": PLAN_SCHEMA_VERSION,
        "plan_kind": PLAN_KIND,
        "error": {"code": code, "message": message},
    }
