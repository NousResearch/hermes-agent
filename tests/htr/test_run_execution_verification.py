"""Tests for Task 11 — Manual Verification Gate for Execution Results."""

import ast
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from htr import contracts, events, io, paths, schemas
from htr.ids import new_attempt_id, new_event_id, new_run_id, new_task_id
from htr.state import (
    ATTEMPT_RUNNING,
    RUN_COMPLETED,
    TASK_RUNNING,
    EventConflict,
    InvalidTransition,
)


def _sample_followup_item(**overrides):
    item = {
        "item_id": "followup-1",
        "title": "Check output",
        "kind": "manual_check",
        "rationale": None,
        "proposed_action": "Verify output manually",
        "metadata": {},
    }
    item.update(overrides)
    return item


def _plan_record(run_id, **kwargs):
    return contracts.make_run_followup_plan_record(
        run_id=run_id,
        source_review_decision=contracts.RUN_REVIEW_NEEDS_FOLLOWUP,
        summary=kwargs.pop("summary", "Follow up after review"),
        followup_items=kwargs.pop("followup_items", [_sample_followup_item()]),
        **kwargs,
    )


def _sample_execution_item(**overrides):
    item = {
        "item_id": "exec-1",
        "source_followup_item_id": "followup-1",
        "title": "Open dashboard",
        "execution_kind": "manual_open_link",
        "command": {"url": "https://example.com"},
        "approval_reason": None,
        "metadata": {},
    }
    item.update(overrides)
    return item


def _complete_task(tmp_path, run_id, task_id):
    attempt_id = new_attempt_id()
    events.apply_task_transition(
        run_id, task_id, TASK_RUNNING, actor="test", base_dir=tmp_path
    )
    events.register_attempt(run_id, task_id, attempt_id, actor="test", base_dir=tmp_path)
    events.apply_attempt_transition(
        run_id, task_id, attempt_id, ATTEMPT_RUNNING, actor="test", base_dir=tmp_path
    )
    result = contracts.make_attempt_result(
        run_id=run_id,
        task_id=task_id,
        attempt_id=attempt_id,
        produced_by="worker",
        summary="done",
    )
    events.submit_attempt_result(
        run_id, task_id, attempt_id, result, base_dir=tmp_path
    )
    verification = contracts.make_verification_result(
        run_id=run_id,
        task_id=task_id,
        attempt_id=attempt_id,
        outcome="passed",
    )
    events.submit_manual_verification(
        run_id, task_id, attempt_id, verification, base_dir=tmp_path
    )
    events.complete_task_manually(
        run_id,
        task_id,
        attempt_id,
        contracts.make_task_completion_record(
            run_id=run_id, task_id=task_id, attempt_id=attempt_id
        ),
        base_dir=tmp_path,
    )
    return attempt_id


def _run_with_reviewed_run(tmp_path, task_count=1):
    run_id = new_run_id()
    io.create_run_workspace(run_id, base_dir=tmp_path)
    task_ids = []
    attempt_ids = []
    for _ in range(task_count):
        task_id = new_task_id()
        io.create_task_workspace(run_id, task_id, base_dir=tmp_path)
        attempt_id = _complete_task(tmp_path, run_id, task_id)
        task_ids.append(task_id)
        attempt_ids.append(attempt_id)
    completion_record = contracts.make_run_completion_record(
        run_id=run_id, completed_task_ids=task_ids
    )
    events.complete_run_manually(run_id, completion_record, base_dir=tmp_path)
    review_record = contracts.make_run_review_record(
        run_id=run_id, decision=contracts.RUN_REVIEW_NEEDS_FOLLOWUP
    )
    events.review_run_manually(run_id, review_record, base_dir=tmp_path)
    return run_id, task_ids, attempt_ids, completion_record, review_record


def _run_with_planned_run(tmp_path):
    run_id, task_ids, attempt_ids, completion, review = _run_with_reviewed_run(tmp_path)
    plan = _plan_record(run_id)
    events.plan_run_followup(run_id, plan, base_dir=tmp_path)
    return run_id, task_ids, attempt_ids, completion, review, plan


def _run_with_execution_request(tmp_path, execution_items=None):
    run_id, task_ids, attempt_ids, completion, review, plan = _run_with_planned_run(
        tmp_path
    )
    items = execution_items if execution_items is not None else [_sample_execution_item()]
    request = contracts.make_run_execution_request_record(
        run_id=run_id,
        source_followup_plan_fingerprint=contracts.run_followup_plan_fingerprint(plan),
        execution_items=items,
    )
    events.request_run_execution(run_id, request, base_dir=tmp_path)
    return run_id, task_ids, attempt_ids, completion, review, plan, request


def _run_with_execution_result(tmp_path, execution_items=None):
    run_id, task_ids, attempt_ids, completion, review, plan, request = (
        _run_with_execution_request(tmp_path, execution_items)
    )
    result = events.execute_run_execution_request(tmp_path, run_id, "human")
    return run_id, task_ids, attempt_ids, completion, review, plan, request, result


def _item_verification_from_result(item_result, decision=None, **overrides):
    verification = {
        "item_id": item_result["item_id"],
        "source_followup_item_id": item_result["source_followup_item_id"],
        "execution_kind": item_result["execution_kind"],
        "item_status": item_result["item_status"],
        "verification_decision": decision or contracts.EXECUTION_ITEM_VERIFICATION_ACCEPTED,
        "reviewer_notes": None,
        "evidence": {},
        "metadata": {},
    }
    verification.update(overrides)
    return verification


def _verification_record(
    run_id,
    result_record,
    verification_decision=contracts.EXECUTION_VERIFICATION_ACCEPTED,
    **kwargs,
):
    item_verifications = kwargs.pop("item_verifications", None)
    if item_verifications is None:
        if verification_decision == contracts.EXECUTION_VERIFICATION_ACCEPTED:
            item_verifications = [
                _item_verification_from_result(r) for r in result_record["item_results"]
            ]
        elif verification_decision == contracts.EXECUTION_VERIFICATION_REJECTED:
            item_verifications = [
                _item_verification_from_result(
                    r, decision=contracts.EXECUTION_ITEM_VERIFICATION_REJECTED
                )
                for r in result_record["item_results"]
            ]
        else:
            results = result_record["item_results"]
            item_verifications = [
                _item_verification_from_result(
                    results[0],
                    decision=contracts.EXECUTION_ITEM_VERIFICATION_NEEDS_CHANGES,
                )
            ]
            for item in results[1:]:
                item_verifications.append(
                    _item_verification_from_result(
                        item,
                        decision=contracts.EXECUTION_ITEM_VERIFICATION_NOT_REVIEWED,
                    )
                )
    source_fp = kwargs.pop(
        "source_execution_result_fingerprint",
        contracts.run_execution_result_fingerprint(result_record),
    )
    return contracts.make_run_execution_verification_record(
        run_id=run_id,
        source_execution_result_fingerprint=source_fp,
        verification_decision=verification_decision,
        item_verifications=item_verifications,
        **kwargs,
    )


def _snapshot(tmp_path, run_id, task_ids):
    result_path = contracts.run_execution_result_record_json_path(run_id, tmp_path)
    request_path = contracts.run_execution_request_record_json_path(run_id, tmp_path)
    verification_path = contracts.run_execution_verification_record_json_path(
        run_id, tmp_path
    )
    return {
        "events_len": len(events.read_task_events(run_id, base_dir=tmp_path)),
        "manifest": io.read_json(paths.run_manifest_path(run_id, tmp_path)),
        "result": result_path.read_bytes() if result_path.exists() else None,
        "request": request_path.read_bytes() if request_path.exists() else None,
        "verification_exists": verification_path.exists(),
        "followup": (
            contracts.run_followup_plan_record_json_path(run_id, tmp_path).read_bytes()
            if contracts.run_followup_plan_record_json_path(run_id, tmp_path).exists()
            else None
        ),
        "completion": (
            contracts.run_completion_record_json_path(run_id, tmp_path).read_bytes()
            if contracts.run_completion_record_json_path(run_id, tmp_path).exists()
            else None
        ),
        "review": (
            contracts.run_review_record_json_path(run_id, tmp_path).read_bytes()
            if contracts.run_review_record_json_path(run_id, tmp_path).exists()
            else None
        ),
        "task_statuses": {
            tid: io.read_json(paths.task_status_path(run_id, tid, tmp_path))
            for tid in task_ids
        },
    }


# --- A. Schema ---


def test_valid_run_execution_verification_record_passes():
    run_id = new_run_id()
    result = contracts.make_run_execution_result_record(
        run_id=run_id,
        source_execution_request_fingerprint="fp",
        item_results=[
            {
                "item_id": "e1",
                "source_followup_item_id": "f1",
                "execution_kind": "other",
                "item_status": contracts.EXECUTION_ITEM_COMPLETED,
                "output": {},
                "error": None,
                "metadata": {},
            }
        ],
    )
    schemas.validate(
        _verification_record(run_id, result), "run_execution_verification_record"
    )


@pytest.mark.parametrize(
    "missing",
    [
        "run_id",
        "source_execution_result_fingerprint",
        "reviewer",
        "verification_decision",
        "item_verifications",
        "notes",
        "metadata",
    ],
)
def test_missing_required_fields_fail(missing):
    run_id = new_run_id()
    result = contracts.make_run_execution_result_record(
        run_id=run_id,
        source_execution_request_fingerprint="fp",
        item_results=[
            {
                "item_id": "e1",
                "source_followup_item_id": "f1",
                "execution_kind": "other",
                "item_status": contracts.EXECUTION_ITEM_COMPLETED,
                "output": {},
                "error": None,
            }
        ],
    )
    record = _verification_record(run_id, result)
    del record[missing]
    with pytest.raises(ValueError, match="missing fields"):
        schemas.validate(record, "run_execution_verification_record")


def test_invalid_run_id_fails_schema():
    run_id = new_run_id()
    result = contracts.make_run_execution_result_record(
        run_id=run_id,
        source_execution_request_fingerprint="fp",
        item_results=[
            {
                "item_id": "e1",
                "source_followup_item_id": "f1",
                "execution_kind": "other",
                "item_status": contracts.EXECUTION_ITEM_COMPLETED,
                "output": {},
                "error": None,
            }
        ],
    )
    record = _verification_record(run_id, result)
    record["run_id"] = "bad"
    with pytest.raises(ValueError, match="run_id must be a valid run id"):
        schemas.validate(record, "run_execution_verification_record")


def test_invalid_source_execution_result_fingerprint_fails():
    run_id = new_run_id()
    result = contracts.make_run_execution_result_record(
        run_id=run_id,
        source_execution_request_fingerprint="fp",
        item_results=[
            {
                "item_id": "e1",
                "source_followup_item_id": "f1",
                "execution_kind": "other",
                "item_status": contracts.EXECUTION_ITEM_COMPLETED,
                "output": {},
                "error": None,
            }
        ],
    )
    with pytest.raises(ValueError, match="source_execution_result_fingerprint"):
        _verification_record(run_id, result, source_execution_result_fingerprint="")


def test_invalid_reviewer_fails():
    run_id = new_run_id()
    result = contracts.make_run_execution_result_record(
        run_id=run_id,
        source_execution_request_fingerprint="fp",
        item_results=[
            {
                "item_id": "e1",
                "source_followup_item_id": "f1",
                "execution_kind": "other",
                "item_status": contracts.EXECUTION_ITEM_COMPLETED,
                "output": {},
                "error": None,
            }
        ],
    )
    with pytest.raises(ValueError, match="reviewer must be a non-empty string"):
        _verification_record(run_id, result, reviewer="")


def test_invalid_verification_decision_fails():
    run_id = new_run_id()
    result = contracts.make_run_execution_result_record(
        run_id=run_id,
        source_execution_request_fingerprint="fp",
        item_results=[
            {
                "item_id": "e1",
                "source_followup_item_id": "f1",
                "execution_kind": "other",
                "item_status": contracts.EXECUTION_ITEM_COMPLETED,
                "output": {},
                "error": None,
            }
        ],
    )
    with pytest.raises(ValueError, match="verification_decision"):
        _verification_record(run_id, result, verification_decision="maybe")


def test_item_verifications_must_be_list():
    run_id = new_run_id()
    result = contracts.make_run_execution_result_record(
        run_id=run_id,
        source_execution_request_fingerprint="fp",
        item_results=[
            {
                "item_id": "e1",
                "source_followup_item_id": "f1",
                "execution_kind": "other",
                "item_status": contracts.EXECUTION_ITEM_COMPLETED,
                "output": {},
                "error": None,
            }
        ],
    )
    record = _verification_record(run_id, result)
    record["item_verifications"] = {}
    with pytest.raises(ValueError, match="item_verifications must be a list"):
        schemas.validate(record, "run_execution_verification_record")


def test_item_verifications_entries_must_be_dicts():
    run_id = new_run_id()
    result = contracts.make_run_execution_result_record(
        run_id=run_id,
        source_execution_request_fingerprint="fp",
        item_results=[
            {
                "item_id": "e1",
                "source_followup_item_id": "f1",
                "execution_kind": "other",
                "item_status": contracts.EXECUTION_ITEM_COMPLETED,
                "output": {},
                "error": None,
            }
        ],
    )
    record = _verification_record(run_id, result)
    record["item_verifications"] = ["bad"]
    with pytest.raises(ValueError, match="each item verification must be a dict"):
        schemas.validate(record, "run_execution_verification_record")


@pytest.mark.parametrize(
    "field",
    [
        "item_id",
        "source_followup_item_id",
        "execution_kind",
        "item_status",
        "verification_decision",
        "evidence",
    ],
)
def test_item_verification_missing_required_field_fails(field):
    run_id = new_run_id()
    result = contracts.make_run_execution_result_record(
        run_id=run_id,
        source_execution_request_fingerprint="fp",
        item_results=[
            {
                "item_id": "e1",
                "source_followup_item_id": "f1",
                "execution_kind": "other",
                "item_status": contracts.EXECUTION_ITEM_COMPLETED,
                "output": {},
                "error": None,
            }
        ],
    )
    record = _verification_record(run_id, result)
    item = dict(record["item_verifications"][0])
    del item[field]
    record["item_verifications"] = [item]
    with pytest.raises(ValueError, match=f"item verification {field}"):
        schemas.validate(record, "run_execution_verification_record")


def test_invalid_item_verification_decision_fails():
    run_id = new_run_id()
    result = contracts.make_run_execution_result_record(
        run_id=run_id,
        source_execution_request_fingerprint="fp",
        item_results=[
            {
                "item_id": "e1",
                "source_followup_item_id": "f1",
                "execution_kind": "other",
                "item_status": contracts.EXECUTION_ITEM_COMPLETED,
                "output": {},
                "error": None,
            }
        ],
    )
    with pytest.raises(ValueError, match="verification_decision is invalid"):
        _verification_record(
            run_id,
            result,
            item_verifications=[
                _item_verification_from_result(
                    result["item_results"][0], decision="maybe"
                )
            ],
        )


def test_reviewer_notes_must_be_string_or_none():
    run_id = new_run_id()
    result = contracts.make_run_execution_result_record(
        run_id=run_id,
        source_execution_request_fingerprint="fp",
        item_results=[
            {
                "item_id": "e1",
                "source_followup_item_id": "f1",
                "execution_kind": "other",
                "item_status": contracts.EXECUTION_ITEM_COMPLETED,
                "output": {},
                "error": None,
            }
        ],
    )
    with pytest.raises(ValueError, match="reviewer_notes must be a string or null"):
        _verification_record(
            run_id,
            result,
            item_verifications=[
                _item_verification_from_result(
                    result["item_results"][0], reviewer_notes=123
                )
            ],
        )


def test_evidence_must_be_dict():
    run_id = new_run_id()
    result = contracts.make_run_execution_result_record(
        run_id=run_id,
        source_execution_request_fingerprint="fp",
        item_results=[
            {
                "item_id": "e1",
                "source_followup_item_id": "f1",
                "execution_kind": "other",
                "item_status": contracts.EXECUTION_ITEM_COMPLETED,
                "output": {},
                "error": None,
            }
        ],
    )
    with pytest.raises(ValueError, match="evidence must be a dict"):
        _verification_record(
            run_id,
            result,
            item_verifications=[
                _item_verification_from_result(
                    result["item_results"][0], evidence="bad"
                )
            ],
        )


def test_notes_must_be_string_or_none():
    run_id = new_run_id()
    result = contracts.make_run_execution_result_record(
        run_id=run_id,
        source_execution_request_fingerprint="fp",
        item_results=[
            {
                "item_id": "e1",
                "source_followup_item_id": "f1",
                "execution_kind": "other",
                "item_status": contracts.EXECUTION_ITEM_COMPLETED,
                "output": {},
                "error": None,
            }
        ],
    )
    record = _verification_record(run_id, result)
    record["notes"] = 1
    with pytest.raises(ValueError, match="notes must be a string or null"):
        schemas.validate(record, "run_execution_verification_record")


def test_metadata_must_be_dict():
    run_id = new_run_id()
    result = contracts.make_run_execution_result_record(
        run_id=run_id,
        source_execution_request_fingerprint="fp",
        item_results=[
            {
                "item_id": "e1",
                "source_followup_item_id": "f1",
                "execution_kind": "other",
                "item_status": contracts.EXECUTION_ITEM_COMPLETED,
                "output": {},
                "error": None,
            }
        ],
    )
    with pytest.raises(ValueError, match="metadata must be a dict"):
        _verification_record(run_id, result, metadata=[])


def test_accepted_requires_all_items_accepted():
    run_id = new_run_id()
    result = contracts.make_run_execution_result_record(
        run_id=run_id,
        source_execution_request_fingerprint="fp",
        item_results=[
            {
                "item_id": "e1",
                "source_followup_item_id": "f1",
                "execution_kind": "other",
                "item_status": contracts.EXECUTION_ITEM_COMPLETED,
                "output": {},
                "error": None,
            }
        ],
    )
    with pytest.raises(ValueError, match="accepted decision requires all"):
        _verification_record(
            run_id,
            result,
            verification_decision=contracts.EXECUTION_VERIFICATION_ACCEPTED,
            item_verifications=[
                _item_verification_from_result(
                    result["item_results"][0],
                    decision=contracts.EXECUTION_ITEM_VERIFICATION_REJECTED,
                )
            ],
        )


def test_rejected_requires_at_least_one_rejected_item():
    run_id = new_run_id()
    result = contracts.make_run_execution_result_record(
        run_id=run_id,
        source_execution_request_fingerprint="fp",
        item_results=[
            {
                "item_id": "e1",
                "source_followup_item_id": "f1",
                "execution_kind": "other",
                "item_status": contracts.EXECUTION_ITEM_COMPLETED,
                "output": {},
                "error": None,
            }
        ],
    )
    with pytest.raises(ValueError, match="rejected decision requires"):
        _verification_record(
            run_id,
            result,
            verification_decision=contracts.EXECUTION_VERIFICATION_REJECTED,
            item_verifications=[
                _item_verification_from_result(result["item_results"][0])
            ],
        )


def test_needs_changes_requires_at_least_one_needs_changes_item():
    run_id = new_run_id()
    result = contracts.make_run_execution_result_record(
        run_id=run_id,
        source_execution_request_fingerprint="fp",
        item_results=[
            {
                "item_id": "e1",
                "source_followup_item_id": "f1",
                "execution_kind": "other",
                "item_status": contracts.EXECUTION_ITEM_COMPLETED,
                "output": {},
                "error": None,
            }
        ],
    )
    with pytest.raises(ValueError, match="needs_changes decision requires"):
        _verification_record(
            run_id,
            result,
            verification_decision=contracts.EXECUTION_VERIFICATION_NEEDS_CHANGES,
            item_verifications=[
                _item_verification_from_result(result["item_results"][0])
            ],
        )


def test_not_reviewed_only_allowed_for_rejected_or_needs_changes():
    run_id = new_run_id()
    result = contracts.make_run_execution_result_record(
        run_id=run_id,
        source_execution_request_fingerprint="fp",
        item_results=[
            {
                "item_id": "e1",
                "source_followup_item_id": "f1",
                "execution_kind": "other",
                "item_status": contracts.EXECUTION_ITEM_COMPLETED,
                "output": {},
                "error": None,
            }
        ],
    )
    with pytest.raises(ValueError, match="accepted decision requires all"):
        _verification_record(
            run_id,
            result,
            verification_decision=contracts.EXECUTION_VERIFICATION_ACCEPTED,
            item_verifications=[
                _item_verification_from_result(
                    result["item_results"][0],
                    decision=contracts.EXECUTION_ITEM_VERIFICATION_NOT_REVIEWED,
                )
            ],
        )


# --- B. Factory ---


def test_make_run_execution_verification_record_returns_valid_schema(tmp_path):
    run_id, _, _, _, _, _, _, result = _run_with_execution_result(tmp_path)
    schemas.validate(
        _verification_record(run_id, result), "run_execution_verification_record"
    )


def test_factory_metadata_defaults_to_empty_dict(tmp_path):
    run_id, _, _, _, _, _, _, result = _run_with_execution_result(tmp_path)
    assert _verification_record(run_id, result)["metadata"] == {}


def test_factory_notes_remain_none(tmp_path):
    run_id, _, _, _, _, _, _, result = _run_with_execution_result(tmp_path)
    assert _verification_record(run_id, result)["notes"] is None


def test_factory_normalizes_evidence_and_item_metadata(tmp_path):
    run_id, _, _, _, _, _, _, result = _run_with_execution_result(tmp_path)
    record = contracts.make_run_execution_verification_record(
        run_id=run_id,
        source_execution_result_fingerprint=contracts.run_execution_result_fingerprint(
            result
        ),
        verification_decision=contracts.EXECUTION_VERIFICATION_ACCEPTED,
        item_verifications=[
            {
                "item_id": result["item_results"][0]["item_id"],
                "source_followup_item_id": result["item_results"][0][
                    "source_followup_item_id"
                ],
                "execution_kind": result["item_results"][0]["execution_kind"],
                "item_status": result["item_results"][0]["item_status"],
                "verification_decision": contracts.EXECUTION_ITEM_VERIFICATION_ACCEPTED,
            }
        ],
    )
    item = record["item_verifications"][0]
    assert item["evidence"] == {}
    assert item["metadata"] == {}
    assert item["reviewer_notes"] is None


def test_factory_does_not_write_files(tmp_path):
    run_id, _, _, _, _, _, _, result = _run_with_execution_result(tmp_path)
    with patch("htr.contracts.atomic_write_json") as mock_write:
        _verification_record(run_id, result)
    mock_write.assert_not_called()


def test_factory_does_not_create_events(tmp_path):
    run_id, _, _, _, _, _, _, result = _run_with_execution_result(tmp_path)
    before = len(events.read_task_events(run_id, base_dir=tmp_path))
    _verification_record(run_id, result)
    assert len(events.read_task_events(run_id, base_dir=tmp_path)) == before


# --- C. Fingerprint ---


def test_fingerprint_validates_schema_first():
    run_id = new_run_id()
    result = contracts.make_run_execution_result_record(
        run_id=run_id,
        source_execution_request_fingerprint="fp",
        item_results=[
            {
                "item_id": "e1",
                "source_followup_item_id": "f1",
                "execution_kind": "other",
                "item_status": contracts.EXECUTION_ITEM_COMPLETED,
                "output": {},
                "error": None,
            }
        ],
    )
    record = _verification_record(run_id, result)
    record["metadata"] = []
    with pytest.raises(ValueError, match="metadata must be a dict"):
        contracts.run_execution_verification_fingerprint(record)


def test_equivalent_records_same_fingerprint():
    run_id = new_run_id()
    result = contracts.make_run_execution_result_record(
        run_id=run_id,
        source_execution_request_fingerprint="fp",
        item_results=[
            {
                "item_id": "e1",
                "source_followup_item_id": "f1",
                "execution_kind": "other",
                "item_status": contracts.EXECUTION_ITEM_COMPLETED,
                "output": {},
                "error": None,
            }
        ],
    )
    a = _verification_record(run_id, result, created_at="2026-07-18T08:00:00+00:00")
    b = dict(a)
    assert contracts.run_execution_verification_fingerprint(
        a
    ) == contracts.run_execution_verification_fingerprint(b)


def test_changed_item_verifications_changes_fingerprint():
    run_id = new_run_id()
    result = contracts.make_run_execution_result_record(
        run_id=run_id,
        source_execution_request_fingerprint="fp",
        item_results=[
            {
                "item_id": "e1",
                "source_followup_item_id": "f1",
                "execution_kind": "other",
                "item_status": contracts.EXECUTION_ITEM_COMPLETED,
                "output": {},
                "error": None,
            }
        ],
    )
    first = _verification_record(run_id, result)
    second = _verification_record(
        run_id,
        result,
        item_verifications=[
            _item_verification_from_result(
                result["item_results"][0],
                reviewer_notes="updated",
            )
        ],
    )
    assert contracts.run_execution_verification_fingerprint(
        first
    ) != contracts.run_execution_verification_fingerprint(second)


def test_fingerprint_uses_canonical_json():
    run_id = new_run_id()
    result = contracts.make_run_execution_result_record(
        run_id=run_id,
        source_execution_request_fingerprint="fp",
        item_results=[
            {
                "item_id": "e1",
                "source_followup_item_id": "f1",
                "execution_kind": "other",
                "item_status": contracts.EXECUTION_ITEM_COMPLETED,
                "output": {},
                "error": None,
            }
        ],
    )
    record = _verification_record(run_id, result, metadata={"b": 2, "a": 1})
    expected = json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    assert contracts.run_execution_verification_fingerprint(record) == expected


# --- D. Lifecycle success ---


@pytest.mark.parametrize(
    "decision,event_type",
    [
        (contracts.EXECUTION_VERIFICATION_ACCEPTED, events.EVENT_TYPE_RUN_EXECUTION_VERIFIED),
        (contracts.EXECUTION_VERIFICATION_REJECTED, events.EVENT_TYPE_RUN_EXECUTION_REJECTED),
        (
            contracts.EXECUTION_VERIFICATION_NEEDS_CHANGES,
            events.EVENT_TYPE_RUN_EXECUTION_NEEDS_CHANGES,
        ),
    ],
)
def test_verify_succeeds_and_appends_decision_event(tmp_path, decision, event_type):
    run_id, _, _, _, _, _, _, result = _run_with_execution_result(tmp_path)
    record = _verification_record(run_id, result, verification_decision=decision)
    before = len(events.read_task_events(run_id, base_dir=tmp_path))
    returned = events.verify_run_execution_result(
        tmp_path, run_id, record, actor="reviewer"
    )
    assert returned == record
    rows = events.read_task_events(run_id, base_dir=tmp_path)
    assert len(rows) == before + 1
    event = rows[-1]
    assert event["event_type"] == event_type
    assert "task_id" not in event
    assert "attempt_id" not in event
    payload = event["payload"]
    assert payload["run_id"] == run_id
    assert payload["reviewer"] == record["reviewer"]
    assert payload["verification_decision"] == decision
    assert payload["source_execution_result_fingerprint"] == record[
        "source_execution_result_fingerprint"
    ]
    assert payload["run_execution_verification_fingerprint"] == contracts.run_execution_verification_fingerprint(
        record
    )
    assert payload["run_execution_verification_record_path"].endswith(
        "run_execution_verification_record.json"
    )


def test_verify_writes_verification_record_file(tmp_path):
    run_id, _, _, _, _, _, _, result = _run_with_execution_result(tmp_path)
    record = _verification_record(run_id, result)
    events.verify_run_execution_result(tmp_path, run_id, record, actor="human")
    assert io.read_json(
        contracts.run_execution_verification_record_json_path(run_id, tmp_path)
    ) == record


# --- E. Preconditions ---


def test_invalid_run_id_fails_no_side_effects(tmp_path):
    run_id, task_ids, _, _, _, _, _, result = _run_with_execution_result(tmp_path)
    before = _snapshot(tmp_path, run_id, task_ids)
    with pytest.raises(ValueError):
        events.verify_run_execution_result(
            tmp_path, "bad", _verification_record(run_id, result), actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids) == before


def test_invalid_schema_fails_no_side_effects(tmp_path):
    run_id, task_ids, _, _, _, _, _, result = _run_with_execution_result(tmp_path)
    record = _verification_record(run_id, result)
    record["metadata"] = "bad"
    before = _snapshot(tmp_path, run_id, task_ids)
    with pytest.raises(ValueError):
        events.verify_run_execution_result(tmp_path, run_id, record, actor="human")
    assert _snapshot(tmp_path, run_id, task_ids) == before


def test_run_id_mismatch_fails_no_side_effects(tmp_path):
    run_id, task_ids, _, _, _, _, _, result = _run_with_execution_result(tmp_path)
    before = _snapshot(tmp_path, run_id, task_ids)
    with pytest.raises(ValueError, match="run_id does not match"):
        events.verify_run_execution_result(
            tmp_path,
            run_id,
            _verification_record(new_run_id(), result),
            actor="human",
        )
    assert _snapshot(tmp_path, run_id, task_ids) == before


def test_missing_run_manifest_fails_no_side_effects(tmp_path):
    run_id = new_run_id()
    with pytest.raises(InvalidTransition):
        events.verify_run_execution_result(
            tmp_path,
            run_id,
            _verification_record(
                run_id,
                contracts.make_run_execution_result_record(
                    run_id=run_id,
                    source_execution_request_fingerprint="fp",
                    item_results=[],
                ),
            ),
            actor="human",
        )


def test_run_not_completed_fails_no_side_effects(tmp_path):
    run_id = new_run_id()
    io.create_run_workspace(run_id, base_dir=tmp_path)
    before = _snapshot(tmp_path, run_id, [])
    with pytest.raises(InvalidTransition, match="is not completed"):
        events.verify_run_execution_result(
            tmp_path,
            run_id,
            _verification_record(
                run_id,
                contracts.make_run_execution_result_record(
                    run_id=run_id,
                    source_execution_request_fingerprint="fp",
                    item_results=[],
                ),
            ),
            actor="human",
        )
    assert _snapshot(tmp_path, run_id, []) == before


def test_missing_execution_result_fails_no_side_effects(tmp_path):
    run_id, task_ids, _, _, _, _, _ = _run_with_execution_request(tmp_path)
    result = contracts.make_run_execution_result_record(
        run_id=run_id,
        source_execution_request_fingerprint="fp",
        item_results=[],
    )
    before = _snapshot(tmp_path, run_id, task_ids)
    with pytest.raises(InvalidTransition, match="run_execution_result_record.json is missing"):
        events.verify_run_execution_result(
            tmp_path, run_id, _verification_record(run_id, result), actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids) == before


def test_source_execution_result_fingerprint_mismatch_fails_no_side_effects(tmp_path):
    run_id, task_ids, _, _, _, _, _, result = _run_with_execution_result(tmp_path)
    record = _verification_record(
        run_id, result, source_execution_result_fingerprint="wrong"
    )
    before = _snapshot(tmp_path, run_id, task_ids)
    with pytest.raises(InvalidTransition, match="source_execution_result_fingerprint"):
        events.verify_run_execution_result(tmp_path, run_id, record, actor="human")
    assert _snapshot(tmp_path, run_id, task_ids) == before


def test_missing_item_verification_fails_no_side_effects(tmp_path):
    run_id, task_ids, _, _, _, _, _, result = _run_with_execution_result(tmp_path)
    record = _verification_record(run_id, result, item_verifications=[])
    before = _snapshot(tmp_path, run_id, task_ids)
    with pytest.raises(InvalidTransition, match="item_verifications item_id set"):
        events.verify_run_execution_result(tmp_path, run_id, record, actor="human")
    assert _snapshot(tmp_path, run_id, task_ids) == before


def test_extra_item_verification_fails_no_side_effects(tmp_path):
    run_id, task_ids, _, _, _, _, _, result = _run_with_execution_result(tmp_path)
    extra = _item_verification_from_result(result["item_results"][0], item_id="extra")
    record = _verification_record(
        run_id,
        result,
        item_verifications=[_item_verification_from_result(result["item_results"][0]), extra],
    )
    before = _snapshot(tmp_path, run_id, task_ids)
    with pytest.raises(InvalidTransition, match="item_verifications item_id set"):
        events.verify_run_execution_result(tmp_path, run_id, record, actor="human")
    assert _snapshot(tmp_path, run_id, task_ids) == before


def test_item_field_mismatch_fails_no_side_effects(tmp_path):
    run_id, task_ids, _, _, _, _, _, result = _run_with_execution_result(tmp_path)
    bad = _item_verification_from_result(
        result["item_results"][0], execution_kind="rerun_task"
    )
    record = _verification_record(run_id, result, item_verifications=[bad])
    before = _snapshot(tmp_path, run_id, task_ids)
    with pytest.raises(InvalidTransition, match="execution_kind does not match"):
        events.verify_run_execution_result(tmp_path, run_id, record, actor="human")
    assert _snapshot(tmp_path, run_id, task_ids) == before


# --- F. Write order ---


def test_write_order_record_before_event(tmp_path):
    run_id, _, _, _, _, _, _, result = _run_with_execution_result(tmp_path)
    record = _verification_record(run_id, result)
    ops = []
    real_append = events.append_run_event

    def track_write(path, data):
        if str(path).endswith("run_execution_verification_record.json"):
            ops.append("write_record")
        return io.atomic_write_json(path, data)

    def track_append(r, e, base_dir=None):
        ops.append("append_event")
        return real_append(r, e, base_dir)

    with patch("htr.events.atomic_write_json", side_effect=track_write), patch(
        "htr.events.append_run_event", side_effect=track_append
    ):
        events.verify_run_execution_result(tmp_path, run_id, record, actor="human")
    assert ops == ["write_record", "append_event"]


# --- G. Replay-only ---


def test_existing_verification_event_id_none_raises(tmp_path):
    run_id, task_ids, _, _, _, _, _, result = _run_with_execution_result(tmp_path)
    record = _verification_record(run_id, result)
    events.verify_run_execution_result(
        tmp_path, run_id, record, actor="human", event_id=new_event_id()
    )
    before = _snapshot(tmp_path, run_id, task_ids)
    with pytest.raises(InvalidTransition):
        events.verify_run_execution_result(tmp_path, run_id, record, actor="human")
    assert _snapshot(tmp_path, run_id, task_ids) == before


def test_existing_verification_missing_event_raises(tmp_path):
    run_id, task_ids, _, _, _, _, _, result = _run_with_execution_result(tmp_path)
    record = _verification_record(run_id, result)
    io.atomic_write_json(
        contracts.run_execution_verification_record_json_path(run_id, tmp_path), record
    )
    before = _snapshot(tmp_path, run_id, task_ids)
    with pytest.raises(InvalidTransition):
        events.verify_run_execution_result(
            tmp_path, run_id, record, actor="human", event_id=new_event_id()
        )
    assert _snapshot(tmp_path, run_id, task_ids) == before


@pytest.mark.parametrize(
    "mutator,expected",
    [
        (lambda e: {**e, "event_type": "task_status_changed"}, InvalidTransition),
        (lambda e: {**e, "run_id": new_run_id()}, EventConflict),
        (lambda e: {**e, "actor": "other"}, EventConflict),
        (
            lambda e: {
                **e,
                "payload": {
                    **e["payload"],
                    "run_execution_verification_fingerprint": "different",
                },
            },
            EventConflict,
        ),
        (
            lambda e: {
                **e,
                "payload": {
                    **e["payload"],
                    "source_execution_result_fingerprint": "different",
                },
            },
            EventConflict,
        ),
        (
            lambda e: {
                **e,
                "payload": {
                    **e["payload"],
                    "verification_decision": "rejected",
                },
            },
            EventConflict,
        ),
        (
            lambda e: {**e, "payload": {**e["payload"], "reviewer": "other"}},
            EventConflict,
        ),
    ],
)
def test_existing_verification_replay_semantic_mismatch(tmp_path, mutator, expected):
    run_id, task_ids, _, _, _, _, _, result = _run_with_execution_result(tmp_path)
    record = _verification_record(run_id, result)
    event_id = new_event_id()
    events.verify_run_execution_result(
        tmp_path, run_id, record, actor="human", event_id=event_id
    )
    event = events.read_task_events(run_id, base_dir=tmp_path)[-1]
    bad_event = mutator(event)
    before = _snapshot(tmp_path, run_id, task_ids)
    with patch("htr.events._find_run_event_by_id", return_value=bad_event):
        with pytest.raises(expected):
            events.verify_run_execution_result(
                tmp_path, run_id, record, actor="human", event_id=event_id
            )
    assert _snapshot(tmp_path, run_id, task_ids) == before


def test_existing_verification_same_event_id_same_semantic_returns_existing(tmp_path):
    run_id, task_ids, _, _, _, _, _, result = _run_with_execution_result(tmp_path)
    record = _verification_record(run_id, result)
    event_id = new_event_id()
    events.verify_run_execution_result(
        tmp_path, run_id, record, actor="human", event_id=event_id
    )
    before = _snapshot(tmp_path, run_id, task_ids)
    returned = events.verify_run_execution_result(
        tmp_path, run_id, record, actor="human", event_id=event_id
    )
    assert returned == record
    assert _snapshot(tmp_path, run_id, task_ids) == before


def test_replay_only_no_writes_or_previous_record_mutation(tmp_path):
    run_id, task_ids, _, _, _, _, _, result = _run_with_execution_result(tmp_path)
    record = _verification_record(run_id, result)
    event_id = new_event_id()
    events.verify_run_execution_result(
        tmp_path, run_id, record, actor="human", event_id=event_id
    )
    result_before = contracts.run_execution_result_record_json_path(
        run_id, tmp_path
    ).read_bytes()
    request_before = contracts.run_execution_request_record_json_path(
        run_id, tmp_path
    ).read_bytes()
    events.verify_run_execution_result(
        tmp_path, run_id, record, actor="human", event_id=event_id
    )
    assert (
        contracts.run_execution_result_record_json_path(run_id, tmp_path).read_bytes()
        == result_before
    )
    assert (
        contracts.run_execution_request_record_json_path(run_id, tmp_path).read_bytes()
        == request_before
    )


# --- H. Idempotency ---


def test_idempotent_same_event_id_same_semantic(tmp_path):
    run_id, _, _, _, _, _, _, result = _run_with_execution_result(tmp_path)
    record = _verification_record(run_id, result)
    event_id = new_event_id()
    first = events.verify_run_execution_result(
        tmp_path, run_id, record, actor="human", event_id=event_id
    )
    second = events.verify_run_execution_result(
        tmp_path, run_id, record, actor="human", event_id=event_id
    )
    assert second == first


def test_same_event_id_different_semantic_raises_conflict(tmp_path):
    run_id, _, _, _, _, _, _, result = _run_with_execution_result(tmp_path)
    event_id = new_event_id()
    accepted = _verification_record(run_id, result)
    events.verify_run_execution_result(
        tmp_path, run_id, accepted, actor="human", event_id=event_id
    )
    rejected = _verification_record(
        run_id, result, verification_decision=contracts.EXECUTION_VERIFICATION_REJECTED
    )
    with pytest.raises(EventConflict):
        events.verify_run_execution_result(
            tmp_path, run_id, rejected, actor="human", event_id=event_id
        )


def test_idempotency_requires_execution_result(tmp_path):
    run_id = new_run_id()
    io.create_run_workspace(run_id, base_dir=tmp_path)
    manifest = io.read_json(paths.run_manifest_path(run_id, tmp_path))
    manifest["status"] = RUN_COMPLETED
    io.atomic_write_json(paths.run_manifest_path(run_id, tmp_path), manifest)
    io.atomic_write_json(
        contracts.run_completion_record_json_path(run_id, tmp_path),
        contracts.make_run_completion_record(
            run_id=run_id, completed_task_ids=[new_task_id()]
        ),
    )
    io.atomic_write_json(
        contracts.run_review_record_json_path(run_id, tmp_path),
        contracts.make_run_review_record(
            run_id=run_id, decision=contracts.RUN_REVIEW_NEEDS_FOLLOWUP
        ),
    )
    io.atomic_write_json(
        contracts.run_followup_plan_record_json_path(run_id, tmp_path),
        _plan_record(run_id),
    )
    io.atomic_write_json(
        contracts.run_execution_request_record_json_path(run_id, tmp_path),
        contracts.make_run_execution_request_record(
            run_id=run_id,
            source_followup_plan_fingerprint=contracts.run_followup_plan_fingerprint(
                _plan_record(run_id)
            ),
            execution_items=[_sample_execution_item()],
        ),
    )
    stub_result = contracts.make_run_execution_result_record(
        run_id=run_id,
        source_execution_request_fingerprint="fp",
        item_results=[],
    )
    with pytest.raises(InvalidTransition, match="run_execution_result_record.json is missing"):
        events.verify_run_execution_result(
            tmp_path,
            run_id,
            _verification_record(run_id, stub_result),
            actor="human",
            event_id=new_event_id(),
        )


# --- I. Boundary preservation ---


def test_boundaries_preserved_on_success(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result = _run_with_execution_result(
        tmp_path
    )
    before = _snapshot(tmp_path, run_id, task_ids)
    record = _verification_record(run_id, result)
    events.verify_run_execution_result(tmp_path, run_id, record, actor="human")
    after = _snapshot(tmp_path, run_id, task_ids)
    assert after["manifest"] == before["manifest"]
    assert after["completion"] == before["completion"]
    assert after["review"] == before["review"]
    assert after["followup"] == before["followup"]
    assert after["request"] == before["request"]
    assert after["result"] == before["result"]
    assert after["task_statuses"] == before["task_statuses"]
    assert after["verification_exists"] is True
    assert len(list(paths.tasks_dir(run_id, tmp_path).iterdir())) == len(task_ids)
    task_id = task_ids[0]
    attempt_id = attempt_ids[0]
    assert paths.result_json_path(run_id, task_id, attempt_id, tmp_path).exists()
    assert contracts.verification_result_json_path(
        run_id, task_id, attempt_id, tmp_path
    ).exists()


# --- J. Import boundary ---


def test_task11_modules_do_not_import_forbidden_modules():
    repo_root = Path(__file__).resolve().parents[2]
    forbidden = {
        "runtime",
        "delegate_task",
        "deco",
        "heal",
        "scheduler",
        "queue",
        "database",
        "sqlite",
        "subprocess",
        "requests",
        "httpx",
        "urllib",
        "webbrowser",
        "os",
    }
    for relative in (
        "htr/contracts.py",
        "htr/events.py",
        "htr/schemas.py",
        "htr/__init__.py",
    ):
        tree = ast.parse((repo_root / relative).read_text(encoding="utf-8"))
        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported |= {
            node.module.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        assert forbidden.isdisjoint(imported), relative


def test_package_exports_task11_apis():
    import htr

    for name in (
        "make_run_execution_verification_record",
        "run_execution_verification_fingerprint",
        "verify_run_execution_result",
        "validate_item_verifications_correspond_to_results",
        "EVENT_TYPE_RUN_EXECUTION_VERIFIED",
        "EVENT_TYPE_RUN_EXECUTION_REJECTED",
        "EVENT_TYPE_RUN_EXECUTION_NEEDS_CHANGES",
        "EXECUTION_VERIFICATION_ACCEPTED",
        "EXECUTION_VERIFICATION_REJECTED",
        "EXECUTION_VERIFICATION_NEEDS_CHANGES",
        "EXECUTION_ITEM_VERIFICATION_ACCEPTED",
        "EXECUTION_ITEM_VERIFICATION_REJECTED",
        "EXECUTION_ITEM_VERIFICATION_NEEDS_CHANGES",
        "EXECUTION_ITEM_VERIFICATION_NOT_REVIEWED",
    ):
        assert hasattr(htr, name)
