"""Focused read-only finalized-run seal evaluation for mutation guards."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from htr import contracts, paths
from htr.ids import validate_id
from htr.io import read_json, read_jsonl
from htr.schemas import validate as validate_schema
from htr.state import (
    ERROR_CODE_RUN_FINALIZED,
    ERROR_CODE_RUN_SEAL_BLOCKED,
    RunFinalizedError,
    RunSealBlockedError,
)

RecordPathFn = Callable[[str, Path | None], Path]
RecordFingerprintFn = Callable[[dict[str, Any]], str]


class SealState(str, Enum):
    NOT_FINALIZED = "not_finalized"
    FINALIZED_VALID = "finalized_valid"
    CLOSURE_PRESENT_UNTRUSTED = "closure_present_untrusted"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True)
class SealEvaluation:
    state: SealState
    reason_codes: tuple[str, ...]
    run_id: str


_RECORD_SPECS: dict[str, tuple[RecordPathFn, RecordFingerprintFn, str]] = {
    "run_completion_record": (
        contracts.run_completion_record_json_path,
        contracts.run_completion_fingerprint,
        "run_completion_record",
    ),
    "run_review_record": (
        contracts.run_review_record_json_path,
        contracts.run_review_fingerprint,
        "run_review_record",
    ),
    "run_followup_plan_record": (
        contracts.run_followup_plan_record_json_path,
        contracts.run_followup_plan_fingerprint,
        "run_followup_plan_record",
    ),
    "run_execution_request_record": (
        contracts.run_execution_request_record_json_path,
        contracts.run_execution_request_fingerprint,
        "run_execution_request_record",
    ),
    "run_execution_result_record": (
        contracts.run_execution_result_record_json_path,
        contracts.run_execution_result_fingerprint,
        "run_execution_result_record",
    ),
    "run_execution_verification_record": (
        contracts.run_execution_verification_record_json_path,
        contracts.run_execution_verification_fingerprint,
        "run_execution_verification_record",
    ),
    "run_post_verification_followup_plan_record": (
        contracts.run_post_verification_followup_plan_record_json_path,
        contracts.run_post_verification_followup_plan_fingerprint,
        "run_post_verification_followup_plan_record",
    ),
    "run_post_verification_execution_request_record": (
        contracts.run_post_verification_execution_request_record_json_path,
        contracts.run_post_verification_execution_request_fingerprint,
        "run_post_verification_execution_request_record",
    ),
    "run_post_verification_execution_result_record": (
        contracts.run_post_verification_execution_result_record_json_path,
        contracts.run_post_verification_execution_result_fingerprint,
        "run_post_verification_execution_result_record",
    ),
    "run_post_verification_execution_verification_record": (
        contracts.run_post_verification_execution_verification_record_json_path,
        contracts.run_post_verification_execution_verification_fingerprint,
        "run_post_verification_execution_verification_record",
    ),
    "run_final_closure_record": (
        contracts.run_final_closure_record_json_path,
        contracts.run_final_closure_fingerprint,
        "run_final_closure_record",
    ),
}

_SOURCE_FINGERPRINT_LINKS: tuple[tuple[str, str, str], ...] = (
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

_CLOSURE_SOURCE_FIELDS: tuple[tuple[str, str], ...] = (
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


def _runs_root_for_run(run_root_path: Path, base_dir: Path | None) -> Path:
    if base_dir is not None:
        return Path(base_dir)
    return run_root_path.parent


def _ensure_run_workspace_contained(
    run_id: str,
    run_root_path: Path,
    base_dir: Path | None,
) -> None:
    runs_root_path = _runs_root_for_run(run_root_path, base_dir)
    runs_resolved = runs_root_path.resolve()
    run_resolved = run_root_path.resolve()
    try:
        common = os.path.commonpath([str(runs_resolved), str(run_resolved)])
    except ValueError as exc:
        raise RunSealBlockedError(
            f"run workspace path is incompatible with runs root for run_id {run_id!r}",
            run_id=run_id,
            reason_codes=("RUN_PATH_UNSAFE",),
        ) from exc
    if common != str(runs_resolved):
        raise RunSealBlockedError(
            f"run workspace for run_id {run_id!r} resolves outside configured runs root",
            run_id=run_id,
            reason_codes=("RUN_PATH_UNSAFE",),
        )


def matches_run_final_closure_recorded_event(
    existing: dict[str, Any],
    *,
    run_id: str,
    actor: str,
    closure_record: dict[str, Any],
) -> bool:
    """Return True when *existing* matches a successful final closure replay."""
    payload = existing.get("payload")
    if not isinstance(payload, dict):
        return False
    return (
        existing.get("event_type") == contracts.PHASE1_TERMINAL_EVENT_TYPE
        and existing.get("run_id") == run_id
        and existing.get("actor") == actor
        and payload.get("run_id") == run_id
        and payload.get("closer") == closure_record["closer"]
        and payload.get("final_closure_status") == closure_record["final_closure_status"]
        and payload.get("source_run_completion_fingerprint")
        == closure_record["source_run_completion_fingerprint"]
        and payload.get("source_run_review_fingerprint")
        == closure_record["source_run_review_fingerprint"]
        and payload.get("source_run_followup_plan_fingerprint")
        == closure_record["source_run_followup_plan_fingerprint"]
        and payload.get("source_run_execution_request_fingerprint")
        == closure_record["source_run_execution_request_fingerprint"]
        and payload.get("source_run_execution_result_fingerprint")
        == closure_record["source_run_execution_result_fingerprint"]
        and payload.get("source_run_execution_verification_fingerprint")
        == closure_record["source_run_execution_verification_fingerprint"]
        and payload.get("source_post_verification_followup_plan_fingerprint")
        == closure_record["source_post_verification_followup_plan_fingerprint"]
        and payload.get("source_post_verification_execution_request_fingerprint")
        == closure_record["source_post_verification_execution_request_fingerprint"]
        and payload.get("source_post_verification_execution_result_fingerprint")
        == closure_record["source_post_verification_execution_result_fingerprint"]
        and payload.get("source_post_verification_execution_verification_fingerprint")
        == closure_record["source_post_verification_execution_verification_fingerprint"]
        and payload.get("run_final_closure_fingerprint")
        == contracts.run_final_closure_fingerprint(closure_record)
    )


def _event_matches_closure_record(
    event: dict[str, Any],
    *,
    run_id: str,
    closure_record: dict[str, Any],
) -> bool:
    closer = closure_record.get("closer")
    if not isinstance(closer, str):
        return False
    actor = event.get("actor")
    if not isinstance(actor, str):
        return False
    return matches_run_final_closure_recorded_event(
        event,
        run_id=run_id,
        actor=actor,
        closure_record=closure_record,
    ) and actor == closer


def _load_chain_record(
    record_type: str,
    *,
    run_id: str,
    base_dir: Path | None,
) -> tuple[dict[str, Any] | None, str | None]:
    path_fn, fingerprint_fn, schema_name = _RECORD_SPECS[record_type]
    path = path_fn(run_id, base_dir)
    if not path.exists():
        return None, "CHAIN_RECORD_MISSING"
    try:
        record = read_json(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return None, "CHAIN_JSON_MALFORMED"
    try:
        validate_schema(record, schema_name)
    except Exception:
        return None, "CHAIN_SCHEMA_INVALID"
    try:
        fingerprint_fn(record)
    except Exception:
        return None, "CHAIN_FINGERPRINT_INVALID"
    if record.get("run_id") != run_id:
        return None, "CHAIN_RUN_ID_MISMATCH"
    return record, None


def _source_fingerprints_valid(loaded: dict[str, dict[str, Any]]) -> str | None:
    for current_type, prior_type, field_name in _SOURCE_FINGERPRINT_LINKS:
        current = loaded.get(current_type)
        prior = loaded.get(prior_type)
        if current is None or prior is None:
            continue
        _, prior_fp_fn, _ = _RECORD_SPECS[prior_type]
        expected = prior_fp_fn(prior)
        if current.get(field_name) != expected:
            return "SOURCE_FINGERPRINT_MISMATCH"
    closure = loaded.get("run_final_closure_record")
    if closure is None:
        return "CLOSURE_RECORD_MISSING"
    for field_name, prior_type in _CLOSURE_SOURCE_FIELDS:
        prior = loaded.get(prior_type)
        if prior is None:
            return "CHAIN_RECORD_MISSING"
        _, prior_fp_fn, _ = _RECORD_SPECS[prior_type]
        expected = prior_fp_fn(prior)
        if closure.get(field_name) != expected:
            return "SOURCE_FINGERPRINT_MISMATCH"
    return None


def evaluate_run_seal(run_id: str, base_dir: Path | None = None) -> SealEvaluation:
    """Return read-only seal state for *run_id* without mutating storage."""
    try:
        validate_id(run_id, "run")
    except ValueError:
        return SealEvaluation(
            SealState.INDETERMINATE,
            ("INVALID_RUN_ID",),
            run_id,
        )

    run_root = paths.run_root(run_id, base_dir)
    try:
        _ensure_run_workspace_contained(run_id, run_root, base_dir)
    except RunSealBlockedError:
        return SealEvaluation(
            SealState.INDETERMINATE,
            ("RUN_PATH_UNSAFE",),
            run_id,
        )

    closure_path = contracts.run_final_closure_record_json_path(run_id, base_dir)
    events_path = paths.task_events_path(run_id, base_dir)

    try:
        all_events = read_jsonl(events_path) if events_path.exists() else []
    except (OSError, json.JSONDecodeError, ValueError):
        return SealEvaluation(
            SealState.INDETERMINATE,
            ("EVENTS_UNREADABLE",),
            run_id,
        )

    closure_events = [
        event
        for event in all_events
        if event.get("run_id") == run_id
        and event.get("event_type") == contracts.PHASE1_TERMINAL_EVENT_TYPE
    ]

    if not closure_path.exists():
        if closure_events:
            return SealEvaluation(
                SealState.CLOSURE_PRESENT_UNTRUSTED,
                ("CLOSURE_EVENT_WITHOUT_JSON",),
                run_id,
            )
        return SealEvaluation(SealState.NOT_FINALIZED, (), run_id)

    closure_record, closure_reason = _load_chain_record(
        "run_final_closure_record",
        run_id=run_id,
        base_dir=base_dir,
    )
    if closure_record is None:
        return SealEvaluation(
            SealState.CLOSURE_PRESENT_UNTRUSTED,
            (closure_reason or "CLOSURE_UNTRUSTED",),
            run_id,
        )

    if not closure_events:
        return SealEvaluation(
            SealState.CLOSURE_PRESENT_UNTRUSTED,
            ("CLOSURE_PENDING_EVENT",),
            run_id,
        )

    if len(closure_events) > 1:
        return SealEvaluation(
            SealState.CLOSURE_PRESENT_UNTRUSTED,
            ("DUPLICATE_CLOSURE_EVENTS",),
            run_id,
        )

    closure_event = closure_events[0]
    if not _event_matches_closure_record(
        closure_event,
        run_id=run_id,
        closure_record=closure_record,
    ):
        return SealEvaluation(
            SealState.CLOSURE_PRESENT_UNTRUSTED,
            ("CLOSURE_EVENT_MISMATCH",),
            run_id,
        )

    loaded: dict[str, dict[str, Any]] = {
        "run_final_closure_record": closure_record,
    }
    for record_type in contracts.PHASE1_MANUAL_WORKFLOW_RECORD_CHAIN[:-1]:
        record, reason = _load_chain_record(
            record_type,
            run_id=run_id,
            base_dir=base_dir,
        )
        if record is None:
            return SealEvaluation(
                SealState.CLOSURE_PRESENT_UNTRUSTED,
                (reason or "CHAIN_RECORD_MISSING",),
                run_id,
            )
        loaded[record_type] = record

    source_reason = _source_fingerprints_valid(loaded)
    if source_reason is not None:
        return SealEvaluation(
            SealState.CLOSURE_PRESENT_UNTRUSTED,
            (source_reason,),
            run_id,
        )

    pv_verification = loaded["run_post_verification_execution_verification_record"]
    try:
        contracts.validate_run_final_closure_sources_correspond(
            closure_record["closure_items"],
            pv_verification,
        )
    except (ValueError, KeyError, TypeError):
        return SealEvaluation(
            SealState.CLOSURE_PRESENT_UNTRUSTED,
            ("CLOSURE_CORRESPONDENCE_FAILED",),
            run_id,
        )

    return SealEvaluation(SealState.FINALIZED_VALID, (), run_id)


def assert_run_mutation_allowed(run_id: str, base_dir: Path | None = None) -> None:
    """Raise when *run_id* is sealed or closure state blocks mutation."""
    evaluation = evaluate_run_seal(run_id, base_dir)
    if evaluation.state == SealState.FINALIZED_VALID:
        raise RunFinalizedError(run_id=run_id, error_code=ERROR_CODE_RUN_FINALIZED)
    if evaluation.state in (
        SealState.CLOSURE_PRESENT_UNTRUSTED,
        SealState.INDETERMINATE,
    ):
        raise RunSealBlockedError(
            run_id=run_id,
            error_code=ERROR_CODE_RUN_SEAL_BLOCKED,
            reason_codes=evaluation.reason_codes,
        )
