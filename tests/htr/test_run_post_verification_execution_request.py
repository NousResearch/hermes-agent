"""Tests for Task 13 — Manual Post-Verification Execution Request Planning."""

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


def _sample_post_verification_followup_item(**overrides):
    item = {
        "followup_item_id": "pvfp-exec-1",
        "source_execution_item_id": "exec-1",
        "source_followup_item_id": "followup-1",
        "execution_kind": "manual_open_link",
        "item_status": contracts.EXECUTION_ITEM_COMPLETED,
        "verification_decision": contracts.EXECUTION_ITEM_VERIFICATION_REJECTED,
        "followup_kind": contracts.POST_VERIFICATION_FOLLOWUP_KIND_REVIEW_REJECTED_ITEM,
        "instructions": None,
        "command": {},
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


def _run_with_verified_execution(
    tmp_path,
    execution_items=None,
    verification_decision=contracts.EXECUTION_VERIFICATION_ACCEPTED,
):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result = (
        _run_with_execution_result(tmp_path, execution_items)
    )
    verification = _verification_record(
        run_id, result, verification_decision=verification_decision
    )
    events.verify_run_execution_result(tmp_path, run_id, verification, actor="human")
    return (
        run_id,
        task_ids,
        attempt_ids,
        completion,
        review,
        plan,
        request,
        result,
        verification,
    )


def _post_verification_plan_record(
    run_id,
    result_record,
    verification_record,
    **kwargs,
):
    followup_items = kwargs.pop("followup_items", None)
    if followup_items is None:
        followup_items = contracts.derive_post_verification_followup_items(
            result_record, verification_record
        )
    return contracts.make_run_post_verification_followup_plan_record(
        run_id=run_id,
        source_execution_result_fingerprint=contracts.run_execution_result_fingerprint(
            result_record
        ),
        source_execution_verification_fingerprint=contracts.run_execution_verification_fingerprint(
            verification_record
        ),
        followup_items=followup_items,
        **kwargs,
    )


    if attempt_ids is None:
        attempt_ids = []
    pvfp_path = contracts.run_post_verification_followup_plan_record_json_path(
        run_id, tmp_path
    )
    result_path = contracts.run_execution_result_record_json_path(run_id, tmp_path)
    request_path = contracts.run_execution_request_record_json_path(run_id, tmp_path)
    verification_path = contracts.run_execution_verification_record_json_path(
        run_id, tmp_path
    )
    followup_path = contracts.run_followup_plan_record_json_path(run_id, tmp_path)
    completion_path = contracts.run_completion_record_json_path(run_id, tmp_path)
    review_path = contracts.run_review_record_json_path(run_id, tmp_path)
    return {
        "events_len": len(events.read_task_events(run_id, base_dir=tmp_path)),
        "manifest": io.read_json(paths.run_manifest_path(run_id, tmp_path)),
        "completion": (
            completion_path.read_bytes() if completion_path.exists() else None
        ),
        "review": review_path.read_bytes() if review_path.exists() else None,
        "followup": followup_path.read_bytes() if followup_path.exists() else None,
        "request": request_path.read_bytes() if request_path.exists() else None,
        "result": result_path.read_bytes() if result_path.exists() else None,
        "verification": (
            verification_path.read_bytes() if verification_path.exists() else None
        ),
        "post_verification_plan_exists": pvfp_path.exists(),
        "task_statuses": {
            tid: io.read_json(paths.task_status_path(run_id, tid, tmp_path))
            for tid in task_ids
        },
        "attempt_statuses": {
            (tid, aid): io.read_json(
                paths.attempt_status_path(run_id, tid, aid, tmp_path)
            )
            for tid, aid in zip(task_ids, attempt_ids)
        }
        if attempt_ids
        else {},
    }




def _run_with_post_verification_plan(
    tmp_path,
    execution_items=None,
    verification_decision=contracts.EXECUTION_VERIFICATION_ACCEPTED,
    followup_items=None,
):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification = (
        _run_with_verified_execution(
            tmp_path,
            execution_items=execution_items,
            verification_decision=verification_decision,
        )
    )
    post_verification_plan = _post_verification_plan_record(
        run_id, result, verification, followup_items=followup_items
    )
    events.plan_post_verification_followup(
        tmp_path, run_id, post_verification_plan, actor="human"
    )
    return (
        run_id,
        task_ids,
        attempt_ids,
        completion,
        review,
        plan,
        request,
        result,
        verification,
        post_verification_plan,
    )


def _post_verification_execution_request_record(
    run_id,
    result_record,
    verification_record,
    post_verification_plan_record,
    **kwargs,
):
    request_items = kwargs.pop("request_items", None)
    if request_items is None:
        request_items = contracts.derive_post_verification_execution_request_items(
            post_verification_plan_record["followup_items"]
        )
    return contracts.make_run_post_verification_execution_request_record(
        run_id=run_id,
        source_execution_result_fingerprint=contracts.run_execution_result_fingerprint(
            result_record
        ),
        source_execution_verification_fingerprint=contracts.run_execution_verification_fingerprint(
            verification_record
        ),
        source_post_verification_followup_plan_fingerprint=contracts.run_post_verification_followup_plan_fingerprint(
            post_verification_plan_record
        ),
        request_items=request_items,
        **kwargs,
    )


def _sample_post_verification_request_item(**overrides):
    item = {
        "request_item_id": "pver-pvfp-exec-1",
        "source_post_verification_followup_item_id": "pvfp-exec-1",
        "source_execution_item_id": "exec-1",
        "source_followup_item_id": "followup-1",
        "execution_kind": "manual_open_link",
        "item_status": contracts.EXECUTION_ITEM_COMPLETED,
        "verification_decision": contracts.EXECUTION_ITEM_VERIFICATION_REJECTED,
        "followup_kind": contracts.POST_VERIFICATION_FOLLOWUP_KIND_REVIEW_REJECTED_ITEM,
        "request_kind": contracts.POST_VERIFICATION_EXECUTION_REQUEST_KIND_REVIEW_REJECTED_ITEM,
        "instructions": None,
        "command": {},
        "metadata": {},
    }
    item.update(overrides)
    return item


def _minimal_post_verification_execution_request(**overrides):
    run_id = overrides.pop("run_id", new_run_id())
    request_items = overrides.pop("request_items", [])
    request_status = overrides.pop("request_status", None)
    if request_status is None:
        request_status = (
            contracts.POST_VERIFICATION_EXECUTION_REQUEST_STATUS_EMPTY
            if not request_items
            else contracts.POST_VERIFICATION_EXECUTION_REQUEST_STATUS_REQUESTED
        )
    record = {
        "schema_version": 1,
        "run_id": run_id,
        "source_execution_result_fingerprint": "fp-result",
        "source_execution_verification_fingerprint": "fp-verification",
        "source_post_verification_followup_plan_fingerprint": "fp-pvfp",
        "requester": "human",
        "request_status": request_status,
        "request_items": request_items,
        "notes": None,
        "metadata": {},
        "created_at": "2026-07-18T08:00:00+00:00",
    }
    record.update(overrides)
    return record


def _snapshot(tmp_path, run_id, task_ids, attempt_ids=None):
    if attempt_ids is None:
        attempt_ids = []
    pvfp_path = contracts.run_post_verification_followup_plan_record_json_path(
        run_id, tmp_path
    )
    pver_path = contracts.run_post_verification_execution_request_record_json_path(
        run_id, tmp_path
    )
    result_path = contracts.run_execution_result_record_json_path(run_id, tmp_path)
    request_path = contracts.run_execution_request_record_json_path(run_id, tmp_path)
    verification_path = contracts.run_execution_verification_record_json_path(
        run_id, tmp_path
    )
    followup_path = contracts.run_followup_plan_record_json_path(run_id, tmp_path)
    completion_path = contracts.run_completion_record_json_path(run_id, tmp_path)
    review_path = contracts.run_review_record_json_path(run_id, tmp_path)
    return {
        "events_len": len(events.read_task_events(run_id, base_dir=tmp_path)),
        "manifest": io.read_json(paths.run_manifest_path(run_id, tmp_path)),
        "completion": (
            completion_path.read_bytes() if completion_path.exists() else None
        ),
        "review": review_path.read_bytes() if review_path.exists() else None,
        "followup": followup_path.read_bytes() if followup_path.exists() else None,
        "request": request_path.read_bytes() if request_path.exists() else None,
        "result": result_path.read_bytes() if result_path.exists() else None,
        "verification": (
            verification_path.read_bytes() if verification_path.exists() else None
        ),
        "post_verification_plan": (
            pvfp_path.read_bytes() if pvfp_path.exists() else None
        ),
        "post_verification_plan_exists": pvfp_path.exists(),
        "post_verification_execution_request": (
            pver_path.read_bytes() if pver_path.exists() else None
        ),
        "post_verification_execution_request_exists": pver_path.exists(),
        "task_statuses": {
            tid: io.read_json(paths.task_status_path(run_id, tid, tmp_path))
            for tid in task_ids
        },
        "attempt_statuses": {
            (tid, aid): io.read_json(
                paths.attempt_status_path(run_id, tid, aid, tmp_path)
            )
            for tid, aid in zip(task_ids, attempt_ids)
        }
        if attempt_ids
        else {},
    }



# --- A. Schema ---


def test_valid_run_post_verification_execution_request_record_passes():
    record = _minimal_post_verification_execution_request(
        request_items=[_sample_post_verification_request_item()],
        request_status=contracts.POST_VERIFICATION_EXECUTION_REQUEST_STATUS_REQUESTED,
    )
    schemas.validate(record, "run_post_verification_execution_request_record")


@pytest.mark.parametrize(
    "missing",
    [
        "run_id",
        "source_execution_result_fingerprint",
        "source_execution_verification_fingerprint",
        "source_post_verification_followup_plan_fingerprint",
        "requester",
        "request_status",
        "request_items",
        "notes",
        "metadata",
    ],
)
def test_missing_required_fields_fail(missing):
    record = _minimal_post_verification_execution_request()
    del record[missing]
    with pytest.raises(ValueError, match="missing fields"):
        schemas.validate(record, "run_post_verification_execution_request_record")


def test_invalid_run_id_fails_schema():
    record = _minimal_post_verification_execution_request(run_id="bad")
    with pytest.raises(ValueError, match="run_id must be a valid run id"):
        schemas.validate(record, "run_post_verification_execution_request_record")


def test_invalid_source_execution_result_fingerprint_fails():
    with pytest.raises(ValueError, match="source_execution_result_fingerprint"):
        contracts.make_run_post_verification_execution_request_record(
            run_id=new_run_id(),
            source_execution_result_fingerprint="",
            source_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            request_items=[],
        )


def test_invalid_source_execution_verification_fingerprint_fails():
    with pytest.raises(ValueError, match="source_execution_verification_fingerprint"):
        contracts.make_run_post_verification_execution_request_record(
            run_id=new_run_id(),
            source_execution_result_fingerprint="fp-result",
            source_execution_verification_fingerprint="",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            request_items=[],
        )


def test_invalid_source_post_verification_followup_plan_fingerprint_fails():
    with pytest.raises(
        ValueError, match="source_post_verification_followup_plan_fingerprint"
    ):
        contracts.make_run_post_verification_execution_request_record(
            run_id=new_run_id(),
            source_execution_result_fingerprint="fp-result",
            source_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="",
            request_items=[],
        )


def test_invalid_requester_fails():
    with pytest.raises(ValueError, match="requester must be a non-empty string"):
        contracts.make_run_post_verification_execution_request_record(
            run_id=new_run_id(),
            source_execution_result_fingerprint="fp-result",
            source_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            request_items=[],
            requester="",
        )


def test_invalid_request_status_fails():
    with pytest.raises(ValueError, match="request_status must be one of"):
        contracts.make_run_post_verification_execution_request_record(
            run_id=new_run_id(),
            source_execution_result_fingerprint="fp-result",
            source_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            request_items=[],
            request_status="draft",
        )


def test_request_items_must_be_list():
    record = _minimal_post_verification_execution_request()
    record["request_items"] = {}
    with pytest.raises(ValueError, match="request_items must be a list"):
        schemas.validate(record, "run_post_verification_execution_request_record")


def test_request_items_entries_must_be_dicts():
    record = _minimal_post_verification_execution_request()
    record["request_items"] = ["bad"]
    with pytest.raises(ValueError, match="request_items entries must be dicts"):
        schemas.validate(record, "run_post_verification_execution_request_record")


def test_missing_request_item_id_fails():
    with pytest.raises(KeyError, match="request_item_id"):
        contracts.make_run_post_verification_execution_request_record(
            run_id=new_run_id(),
            source_execution_result_fingerprint="fp-result",
            source_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            request_items=[
                {
                    "source_post_verification_followup_item_id": "pvfp-exec-1",
                    "request_kind": (
                        contracts.POST_VERIFICATION_EXECUTION_REQUEST_KIND_OTHER
                    ),
                    "instructions": None,
                    "command": {},
                    "metadata": {},
                }
            ],
            request_status=contracts.POST_VERIFICATION_EXECUTION_REQUEST_STATUS_REQUESTED,
        )


def test_invalid_request_kind_fails():
    with pytest.raises(ValueError, match="request_kind"):
        contracts.make_run_post_verification_execution_request_record(
            run_id=new_run_id(),
            source_execution_result_fingerprint="fp-result",
            source_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            request_items=[
                _sample_post_verification_request_item(request_kind="auto_execute")
            ],
            request_status=contracts.POST_VERIFICATION_EXECUTION_REQUEST_STATUS_REQUESTED,
        )


def test_verification_decision_must_be_string_or_none():
    with pytest.raises(ValueError, match="verification_decision"):
        contracts.make_run_post_verification_execution_request_record(
            run_id=new_run_id(),
            source_execution_result_fingerprint="fp-result",
            source_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            request_items=[
                _sample_post_verification_request_item(verification_decision=123)
            ],
            request_status=contracts.POST_VERIFICATION_EXECUTION_REQUEST_STATUS_REQUESTED,
        )


def test_followup_kind_must_be_string_or_none():
    with pytest.raises(ValueError, match="followup_kind"):
        contracts.make_run_post_verification_execution_request_record(
            run_id=new_run_id(),
            source_execution_result_fingerprint="fp-result",
            source_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            request_items=[_sample_post_verification_request_item(followup_kind=123)],
            request_status=contracts.POST_VERIFICATION_EXECUTION_REQUEST_STATUS_REQUESTED,
        )


def test_command_must_be_dict():
    with pytest.raises(ValueError, match="command must be a dict"):
        contracts.make_run_post_verification_execution_request_record(
            run_id=new_run_id(),
            source_execution_result_fingerprint="fp-result",
            source_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            request_items=[_sample_post_verification_request_item(command="bad")],
            request_status=contracts.POST_VERIFICATION_EXECUTION_REQUEST_STATUS_REQUESTED,
        )


def test_item_metadata_must_be_dict():
    with pytest.raises(ValueError, match="metadata must be a dict"):
        contracts.make_run_post_verification_execution_request_record(
            run_id=new_run_id(),
            source_execution_result_fingerprint="fp-result",
            source_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            request_items=[_sample_post_verification_request_item(metadata="bad")],
            request_status=contracts.POST_VERIFICATION_EXECUTION_REQUEST_STATUS_REQUESTED,
        )


def test_instructions_must_be_string_or_none():
    with pytest.raises(ValueError, match="instructions must be a non-empty string or null"):
        contracts.make_run_post_verification_execution_request_record(
            run_id=new_run_id(),
            source_execution_result_fingerprint="fp-result",
            source_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            request_items=[_sample_post_verification_request_item(instructions=123)],
            request_status=contracts.POST_VERIFICATION_EXECUTION_REQUEST_STATUS_REQUESTED,
        )


def test_notes_must_be_string_or_none():
    record = _minimal_post_verification_execution_request()
    record["notes"] = 1
    with pytest.raises(ValueError, match="notes must be a string or null"):
        schemas.validate(record, "run_post_verification_execution_request_record")


def test_metadata_must_be_dict():
    with pytest.raises(ValueError, match="metadata must be a dict"):
        contracts.make_run_post_verification_execution_request_record(
            run_id=new_run_id(),
            source_execution_result_fingerprint="fp-result",
            source_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            request_items=[],
            metadata=[],
        )


def test_empty_request_status_requires_empty_request_items():
    with pytest.raises(ValueError, match="empty request_status requires request_items"):
        contracts.make_run_post_verification_execution_request_record(
            run_id=new_run_id(),
            source_execution_result_fingerprint="fp-result",
            source_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            request_items=[_sample_post_verification_request_item()],
            request_status=contracts.POST_VERIFICATION_EXECUTION_REQUEST_STATUS_EMPTY,
        )


def test_requested_request_status_requires_non_empty_request_items():
    with pytest.raises(ValueError, match="requested request_status requires non-empty"):
        contracts.make_run_post_verification_execution_request_record(
            run_id=new_run_id(),
            source_execution_result_fingerprint="fp-result",
            source_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            request_items=[],
            request_status=contracts.POST_VERIFICATION_EXECUTION_REQUEST_STATUS_REQUESTED,
        )


# --- B. Factory ---


def test_make_run_post_verification_execution_request_record_returns_valid_schema(
    tmp_path,
):
    run_id, _, _, _, _, _, _, result, verification, pvfp = _run_with_post_verification_plan(
        tmp_path
    )
    schemas.validate(
        _post_verification_execution_request_record(run_id, result, verification, pvfp),
        "run_post_verification_execution_request_record",
    )


def test_factory_metadata_defaults_to_empty_dict(tmp_path):
    run_id, _, _, _, _, _, _, result, verification, pvfp = _run_with_post_verification_plan(
        tmp_path
    )
    assert (
        _post_verification_execution_request_record(run_id, result, verification, pvfp)[
            "metadata"
        ]
        == {}
    )


def test_factory_notes_remain_none(tmp_path):
    run_id, _, _, _, _, _, _, result, verification, pvfp = _run_with_post_verification_plan(
        tmp_path
    )
    assert (
        _post_verification_execution_request_record(run_id, result, verification, pvfp)[
            "notes"
        ]
        is None
    )


def test_factory_normalizes_command_and_item_metadata(tmp_path):
    run_id, _, _, _, _, _, _, result, verification, pvfp = _run_with_post_verification_plan(
        tmp_path,
        verification_decision=contracts.EXECUTION_VERIFICATION_REJECTED,
    )
    record = _post_verification_execution_request_record(
        run_id, result, verification, pvfp
    )
    item = record["request_items"][0]
    assert item["command"] == {}
    assert item["metadata"] == {}
    assert item["instructions"] is None


def test_factory_instructions_remain_none(tmp_path):
    run_id, _, _, _, _, _, _, result, verification, pvfp = _run_with_post_verification_plan(
        tmp_path,
        verification_decision=contracts.EXECUTION_VERIFICATION_REJECTED,
    )
    record = _post_verification_execution_request_record(
        run_id, result, verification, pvfp
    )
    assert record["request_items"][0]["instructions"] is None


def test_factory_does_not_write_files(tmp_path):
    run_id, _, _, _, _, _, _, result, verification, pvfp = _run_with_post_verification_plan(
        tmp_path
    )
    with patch("htr.contracts.atomic_write_json") as mock_write:
        _post_verification_execution_request_record(run_id, result, verification, pvfp)
    mock_write.assert_not_called()


def test_factory_does_not_create_events(tmp_path):
    run_id, _, _, _, _, _, _, result, verification, pvfp = _run_with_post_verification_plan(
        tmp_path
    )
    before = len(events.read_task_events(run_id, base_dir=tmp_path))
    _post_verification_execution_request_record(run_id, result, verification, pvfp)
    assert len(events.read_task_events(run_id, base_dir=tmp_path)) == before


def test_factory_does_not_execute_external_systems(tmp_path):
    run_id, _, _, _, _, _, _, result, verification, pvfp = _run_with_post_verification_plan(
        tmp_path
    )
    with patch("subprocess.run") as mock_subprocess, patch(
        "subprocess.Popen"
    ) as mock_popen, patch("htr.contracts.atomic_write_json") as mock_write, patch(
        "htr.events.append_run_event"
    ) as mock_append:
        _post_verification_execution_request_record(run_id, result, verification, pvfp)
    mock_subprocess.assert_not_called()
    mock_popen.assert_not_called()
    mock_write.assert_not_called()
    mock_append.assert_not_called()


def test_derive_empty_followup_plan_produces_empty_request(tmp_path):
    run_id, _, _, _, _, _, _, result, verification, pvfp = _run_with_post_verification_plan(
        tmp_path,
        verification_decision=contracts.EXECUTION_VERIFICATION_ACCEPTED,
    )
    items = contracts.derive_post_verification_execution_request_items(
        pvfp["followup_items"]
    )
    assert items == []
    record = _post_verification_execution_request_record(
        run_id, result, verification, pvfp
    )
    assert (
        record["request_status"]
        == contracts.POST_VERIFICATION_EXECUTION_REQUEST_STATUS_EMPTY
    )


def test_derive_rejected_verification_produces_requested_items(tmp_path):
    run_id, _, _, _, _, _, _, result, verification, pvfp = _run_with_post_verification_plan(
        tmp_path,
        verification_decision=contracts.EXECUTION_VERIFICATION_REJECTED,
    )
    items = contracts.derive_post_verification_execution_request_items(
        pvfp["followup_items"]
    )
    assert len(items) == 1
    record = _post_verification_execution_request_record(
        run_id, result, verification, pvfp
    )
    assert (
        record["request_status"]
        == contracts.POST_VERIFICATION_EXECUTION_REQUEST_STATUS_REQUESTED
    )
    assert len(record["request_items"]) == 1


# --- C. Fingerprint ---


def test_fingerprint_validates_schema_first():
    record = _minimal_post_verification_execution_request()
    record["metadata"] = []
    with pytest.raises(ValueError, match="metadata must be a dict"):
        contracts.run_post_verification_execution_request_fingerprint(record)


def test_equivalent_records_same_fingerprint():
    run_id = new_run_id()
    a = _minimal_post_verification_execution_request(
        run_id=run_id, created_at="2026-07-18T08:00:00+00:00"
    )
    b = dict(a)
    assert contracts.run_post_verification_execution_request_fingerprint(
        a
    ) == contracts.run_post_verification_execution_request_fingerprint(b)


def test_changed_request_items_changes_fingerprint():
    run_id = new_run_id()
    first = _minimal_post_verification_execution_request(
        run_id=run_id,
        request_items=[_sample_post_verification_request_item()],
        request_status=contracts.POST_VERIFICATION_EXECUTION_REQUEST_STATUS_REQUESTED,
    )
    second = _minimal_post_verification_execution_request(
        run_id=run_id,
        request_items=[
            _sample_post_verification_request_item(request_item_id="pver-pvfp-exec-2")
        ],
        request_status=contracts.POST_VERIFICATION_EXECUTION_REQUEST_STATUS_REQUESTED,
    )
    assert contracts.run_post_verification_execution_request_fingerprint(
        first
    ) != contracts.run_post_verification_execution_request_fingerprint(second)


def test_changed_metadata_changes_fingerprint():
    run_id = new_run_id()
    first = _minimal_post_verification_execution_request(run_id=run_id, metadata={"a": 1})
    second = _minimal_post_verification_execution_request(run_id=run_id, metadata={"a": 2})
    assert contracts.run_post_verification_execution_request_fingerprint(
        first
    ) != contracts.run_post_verification_execution_request_fingerprint(second)


def test_fingerprint_uses_canonical_json():
    run_id = new_run_id()
    record = _minimal_post_verification_execution_request(run_id=run_id, metadata={"b": 2, "a": 1})
    expected = json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    assert contracts.run_post_verification_execution_request_fingerprint(record) == expected


def test_fingerprint_does_not_read_files():
    record = _minimal_post_verification_execution_request()
    with patch("htr.io.read_json") as mock_read:
        contracts.run_post_verification_execution_request_fingerprint(record)
    mock_read.assert_not_called()


# --- D. Derivation helper ---


def test_derive_empty_followup_items_returns_empty_list():
    assert contracts.derive_post_verification_execution_request_items([]) == []


def test_derive_one_followup_item_returns_one_request_item(tmp_path):
    run_id, _, _, _, _, _, _, result, verification, pvfp = _run_with_post_verification_plan(
        tmp_path,
        verification_decision=contracts.EXECUTION_VERIFICATION_REJECTED,
    )
    items = contracts.derive_post_verification_execution_request_items(
        pvfp["followup_items"]
    )
    assert len(items) == 1


def test_derive_preserves_source_post_verification_followup_item_id(tmp_path):
    run_id, _, _, _, _, _, _, result, verification, pvfp = _run_with_post_verification_plan(
        tmp_path,
        verification_decision=contracts.EXECUTION_VERIFICATION_REJECTED,
    )
    items = contracts.derive_post_verification_execution_request_items(
        pvfp["followup_items"]
    )
    assert items[0]["source_post_verification_followup_item_id"] == "pvfp-exec-1"


def test_derive_preserves_source_execution_item_id(tmp_path):
    run_id, _, _, _, _, _, _, result, verification, pvfp = _run_with_post_verification_plan(
        tmp_path,
        verification_decision=contracts.EXECUTION_VERIFICATION_REJECTED,
    )
    items = contracts.derive_post_verification_execution_request_items(
        pvfp["followup_items"]
    )
    assert items[0]["source_execution_item_id"] == "exec-1"


def test_derive_preserves_source_followup_item_id(tmp_path):
    run_id, _, _, _, _, _, _, result, verification, pvfp = _run_with_post_verification_plan(
        tmp_path,
        verification_decision=contracts.EXECUTION_VERIFICATION_REJECTED,
    )
    items = contracts.derive_post_verification_execution_request_items(
        pvfp["followup_items"]
    )
    assert items[0]["source_followup_item_id"] == "followup-1"


def test_derive_preserves_execution_kind(tmp_path):
    run_id, _, _, _, _, _, _, result, verification, pvfp = _run_with_post_verification_plan(
        tmp_path,
        verification_decision=contracts.EXECUTION_VERIFICATION_REJECTED,
    )
    items = contracts.derive_post_verification_execution_request_items(
        pvfp["followup_items"]
    )
    assert items[0]["execution_kind"] == "manual_open_link"


def test_derive_preserves_item_status(tmp_path):
    run_id, _, _, _, _, _, _, result, verification, pvfp = _run_with_post_verification_plan(
        tmp_path,
        verification_decision=contracts.EXECUTION_VERIFICATION_REJECTED,
    )
    items = contracts.derive_post_verification_execution_request_items(
        pvfp["followup_items"]
    )
    assert items[0]["item_status"] == pvfp["followup_items"][0]["item_status"]


def test_derive_preserves_verification_decision(tmp_path):
    run_id, _, _, _, _, _, _, result, verification, pvfp = _run_with_post_verification_plan(
        tmp_path,
        verification_decision=contracts.EXECUTION_VERIFICATION_REJECTED,
    )
    items = contracts.derive_post_verification_execution_request_items(
        pvfp["followup_items"]
    )
    assert (
        items[0]["verification_decision"]
        == contracts.EXECUTION_ITEM_VERIFICATION_REJECTED
    )


def test_derive_preserves_followup_kind(tmp_path):
    run_id, _, _, _, _, _, _, result, verification, pvfp = _run_with_post_verification_plan(
        tmp_path,
        verification_decision=contracts.EXECUTION_VERIFICATION_REJECTED,
    )
    items = contracts.derive_post_verification_execution_request_items(
        pvfp["followup_items"]
    )
    assert (
        items[0]["followup_kind"]
        == contracts.POST_VERIFICATION_FOLLOWUP_KIND_REVIEW_REJECTED_ITEM
    )


def test_derive_preserves_instructions(tmp_path):
    followup = _sample_post_verification_followup_item(instructions="do thing")
    items = contracts.derive_post_verification_execution_request_items([followup])
    assert items[0]["instructions"] == "do thing"


def test_derive_preserves_command(tmp_path):
    followup = _sample_post_verification_followup_item(command={"url": "https://x.test"})
    items = contracts.derive_post_verification_execution_request_items([followup])
    assert items[0]["command"] == {"url": "https://x.test"}


def test_derive_preserves_metadata(tmp_path):
    followup = _sample_post_verification_followup_item(metadata={"k": "v"})
    items = contracts.derive_post_verification_execution_request_items([followup])
    assert items[0]["metadata"] == {"k": "v"}


@pytest.mark.parametrize(
    "followup_kind,request_kind",
    [
        (
            contracts.POST_VERIFICATION_FOLLOWUP_KIND_REOPEN_LINK_MANUALLY,
            contracts.POST_VERIFICATION_EXECUTION_REQUEST_KIND_REOPEN_LINK_MANUALLY,
        ),
        (
            contracts.POST_VERIFICATION_FOLLOWUP_KIND_UPDATE_DOCUMENTATION_MANUALLY,
            contracts.POST_VERIFICATION_EXECUTION_REQUEST_KIND_UPDATE_DOCUMENTATION_MANUALLY,
        ),
        (
            contracts.POST_VERIFICATION_FOLLOWUP_KIND_PREPARE_NEW_EXECUTION_REQUEST,
            contracts.POST_VERIFICATION_EXECUTION_REQUEST_KIND_PREPARE_NEW_EXECUTION_REQUEST,
        ),
        (
            contracts.POST_VERIFICATION_FOLLOWUP_KIND_REVIEW_REJECTED_ITEM,
            contracts.POST_VERIFICATION_EXECUTION_REQUEST_KIND_REVIEW_REJECTED_ITEM,
        ),
        (
            contracts.POST_VERIFICATION_FOLLOWUP_KIND_REVIEW_NEEDS_CHANGES_ITEM,
            contracts.POST_VERIFICATION_EXECUTION_REQUEST_KIND_REVIEW_NEEDS_CHANGES_ITEM,
        ),
        (
            contracts.POST_VERIFICATION_FOLLOWUP_KIND_REVIEW_NOT_REVIEWED_ITEM,
            contracts.POST_VERIFICATION_EXECUTION_REQUEST_KIND_REVIEW_NOT_REVIEWED_ITEM,
        ),
        (
            contracts.POST_VERIFICATION_FOLLOWUP_KIND_OTHER,
            contracts.POST_VERIFICATION_EXECUTION_REQUEST_KIND_OTHER,
        ),
    ],
)
def test_derive_maps_known_followup_kind_to_request_kind(followup_kind, request_kind):
    followup = _sample_post_verification_followup_item(followup_kind=followup_kind)
    items = contracts.derive_post_verification_execution_request_items([followup])
    assert items[0]["request_kind"] == request_kind


def test_derive_maps_unknown_followup_kind_to_other():
    followup = _sample_post_verification_followup_item(followup_kind="unknown_kind")
    items = contracts.derive_post_verification_execution_request_items([followup])
    assert items[0]["request_kind"] == contracts.POST_VERIFICATION_EXECUTION_REQUEST_KIND_OTHER


def test_derive_does_not_read_files():
    followup = _sample_post_verification_followup_item()
    with patch("htr.io.read_json") as mock_read:
        contracts.derive_post_verification_execution_request_items([followup])
    mock_read.assert_not_called()


def test_derive_does_not_call_external_systems():
    followup = _sample_post_verification_followup_item()
    with patch("subprocess.run") as mock_subprocess:
        contracts.derive_post_verification_execution_request_items([followup])
    mock_subprocess.assert_not_called()


# --- E. Request lifecycle success ---


def test_request_post_verification_execution_succeeds_after_full_chain(tmp_path):
    run_id, _, _, _, _, _, _, result, verification, pvfp = _run_with_post_verification_plan(
        tmp_path
    )
    record = _post_verification_execution_request_record(
        run_id, result, verification, pvfp
    )
    returned = events.request_post_verification_execution(
        tmp_path, run_id, record, actor="requester"
    )
    assert returned == record


def test_request_post_verification_execution_empty_request_writes_record_and_event(
    tmp_path,
):
    run_id, _, _, _, _, _, _, result, verification, pvfp = _run_with_post_verification_plan(
        tmp_path,
        verification_decision=contracts.EXECUTION_VERIFICATION_ACCEPTED,
    )
    record = _post_verification_execution_request_record(
        run_id, result, verification, pvfp
    )
    before = len(events.read_task_events(run_id, base_dir=tmp_path))
    events.request_post_verification_execution(
        tmp_path, run_id, record, actor="requester"
    )
    assert io.read_json(
        contracts.run_post_verification_execution_request_record_json_path(
            run_id, tmp_path
        )
    ) == record
    rows = events.read_task_events(run_id, base_dir=tmp_path)
    assert len(rows) == before + 1
    event = rows[-1]
    assert (
        event["event_type"]
        == events.EVENT_TYPE_RUN_POST_VERIFICATION_EXECUTION_REQUESTED
    )
    assert (
        record["request_status"]
        == contracts.POST_VERIFICATION_EXECUTION_REQUEST_STATUS_EMPTY
    )


def test_request_post_verification_execution_requested_request_writes_record_and_event(
    tmp_path,
):
    run_id, _, _, _, _, _, _, result, verification, pvfp = _run_with_post_verification_plan(
        tmp_path,
        verification_decision=contracts.EXECUTION_VERIFICATION_REJECTED,
    )
    record = _post_verification_execution_request_record(
        run_id, result, verification, pvfp
    )
    before = len(events.read_task_events(run_id, base_dir=tmp_path))
    events.request_post_verification_execution(
        tmp_path, run_id, record, actor="requester"
    )
    assert io.read_json(
        contracts.run_post_verification_execution_request_record_json_path(
            run_id, tmp_path
        )
    ) == record
    rows = events.read_task_events(run_id, base_dir=tmp_path)
    assert len(rows) == before + 1
    event = rows[-1]
    assert (
        event["event_type"]
        == events.EVENT_TYPE_RUN_POST_VERIFICATION_EXECUTION_REQUESTED
    )
    assert (
        record["request_status"]
        == contracts.POST_VERIFICATION_EXECUTION_REQUEST_STATUS_REQUESTED
    )
    assert record["request_items"]


def test_request_post_verification_execution_event_payload_fields(tmp_path):
    run_id, _, _, _, _, _, _, result, verification, pvfp = _run_with_post_verification_plan(
        tmp_path,
        verification_decision=contracts.EXECUTION_VERIFICATION_REJECTED,
    )
    record = _post_verification_execution_request_record(
        run_id, result, verification, pvfp, requester="assistant"
    )
    events.request_post_verification_execution(
        tmp_path, run_id, record, actor="operator"
    )
    event = events.read_task_events(run_id, base_dir=tmp_path)[-1]
    assert "task_id" not in event
    assert "attempt_id" not in event
    payload = event["payload"]
    assert payload["run_id"] == run_id
    assert payload["requester"] == "assistant"
    assert payload["request_status"] == record["request_status"]
    assert payload["source_execution_result_fingerprint"] == record[
        "source_execution_result_fingerprint"
    ]
    assert payload["source_execution_verification_fingerprint"] == record[
        "source_execution_verification_fingerprint"
    ]
    assert payload["source_post_verification_followup_plan_fingerprint"] == record[
        "source_post_verification_followup_plan_fingerprint"
    ]
    assert payload["run_post_verification_execution_request_fingerprint"] == (
        contracts.run_post_verification_execution_request_fingerprint(record)
    )
    assert payload["run_post_verification_execution_request_record_path"].endswith(
        "run_post_verification_execution_request_record.json"
    )


def test_item_linked_request_items_correspond_to_followup_plan(tmp_path):
    run_id, _, _, _, _, _, _, result, verification, pvfp = _run_with_post_verification_plan(
        tmp_path,
        verification_decision=contracts.EXECUTION_VERIFICATION_REJECTED,
    )
    record = _post_verification_execution_request_record(
        run_id, result, verification, pvfp
    )
    contracts.validate_post_verification_execution_request_items_correspond(
        record["request_items"], pvfp
    )
    events.request_post_verification_execution(
        tmp_path, run_id, record, actor="human"
    )
    assert contracts.run_post_verification_execution_request_record_json_path(
        run_id, tmp_path
    ).exists()


# --- F. Preconditions ---


def test_invalid_run_id_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp = (
        _run_with_post_verification_plan(tmp_path)
    )
    record = _post_verification_execution_request_record(
        run_id, result, verification, pvfp
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(ValueError):
        events.request_post_verification_execution(
            tmp_path, "bad", record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_invalid_schema_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp = (
        _run_with_post_verification_plan(tmp_path)
    )
    record = _post_verification_execution_request_record(
        run_id, result, verification, pvfp
    )
    record["metadata"] = "bad"
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(ValueError):
        events.request_post_verification_execution(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_run_id_mismatch_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp = (
        _run_with_post_verification_plan(tmp_path)
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(ValueError, match="run_id does not match"):
        events.request_post_verification_execution(
            tmp_path,
            run_id,
            _post_verification_execution_request_record(
                new_run_id(), result, verification, pvfp
            ),
            actor="human",
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_missing_run_manifest_fails_no_side_effects(tmp_path):
    run_id = new_run_id()
    with pytest.raises(InvalidTransition):
        events.request_post_verification_execution(
            tmp_path,
            run_id,
            _minimal_post_verification_execution_request(run_id=run_id),
            actor="human",
        )


def test_run_not_completed_fails_no_side_effects(tmp_path):
    run_id = new_run_id()
    io.create_run_workspace(run_id, base_dir=tmp_path)
    before = _snapshot(tmp_path, run_id, [])
    with pytest.raises(InvalidTransition, match="is not completed"):
        events.request_post_verification_execution(
            tmp_path,
            run_id,
            _minimal_post_verification_execution_request(run_id=run_id),
            actor="human",
        )
    assert _snapshot(tmp_path, run_id, []) == before


def test_missing_completion_record_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp = (
        _run_with_post_verification_plan(tmp_path)
    )
    contracts.run_completion_record_json_path(run_id, tmp_path).unlink()
    record = _post_verification_execution_request_record(
        run_id, result, verification, pvfp
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(InvalidTransition, match="run_completion_record.json is missing"):
        events.request_post_verification_execution(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_missing_review_record_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp = (
        _run_with_post_verification_plan(tmp_path)
    )
    contracts.run_review_record_json_path(run_id, tmp_path).unlink()
    record = _post_verification_execution_request_record(
        run_id, result, verification, pvfp
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(InvalidTransition, match="run_review_record.json is missing"):
        events.request_post_verification_execution(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_missing_followup_plan_record_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp = (
        _run_with_post_verification_plan(tmp_path)
    )
    contracts.run_followup_plan_record_json_path(run_id, tmp_path).unlink()
    record = _post_verification_execution_request_record(
        run_id, result, verification, pvfp
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(InvalidTransition, match="run_followup_plan_record.json is missing"):
        events.request_post_verification_execution(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_missing_execution_request_record_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp = (
        _run_with_post_verification_plan(tmp_path)
    )
    contracts.run_execution_request_record_json_path(run_id, tmp_path).unlink()
    record = _post_verification_execution_request_record(
        run_id, result, verification, pvfp
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(
        InvalidTransition, match="run_execution_request_record.json is missing"
    ):
        events.request_post_verification_execution(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_missing_execution_result_record_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp = (
        _run_with_post_verification_plan(tmp_path)
    )
    contracts.run_execution_result_record_json_path(run_id, tmp_path).unlink()
    record = _post_verification_execution_request_record(
        run_id, result, verification, pvfp
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(
        InvalidTransition, match="run_execution_result_record.json is missing"
    ):
        events.request_post_verification_execution(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_missing_execution_verification_record_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp = (
        _run_with_post_verification_plan(tmp_path)
    )
    contracts.run_execution_verification_record_json_path(run_id, tmp_path).unlink()
    record = _post_verification_execution_request_record(
        run_id, result, verification, pvfp
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(
        InvalidTransition, match="run_execution_verification_record.json is missing"
    ):
        events.request_post_verification_execution(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_missing_post_verification_followup_plan_record_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp = (
        _run_with_post_verification_plan(tmp_path)
    )
    contracts.run_post_verification_followup_plan_record_json_path(
        run_id, tmp_path
    ).unlink()
    record = _post_verification_execution_request_record(
        run_id, result, verification, pvfp
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(
        InvalidTransition,
        match="run_post_verification_followup_plan_record.json is missing",
    ):
        events.request_post_verification_execution(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_invalid_execution_result_schema_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp = (
        _run_with_post_verification_plan(tmp_path)
    )
    bad_result = dict(result)
    bad_result["metadata"] = "bad"
    io.atomic_write_json(
        contracts.run_execution_result_record_json_path(run_id, tmp_path), bad_result
    )
    record = _post_verification_execution_request_record(
        run_id, result, verification, pvfp
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(ValueError, match="metadata must be a dict"):
        events.request_post_verification_execution(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_invalid_execution_verification_schema_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp = (
        _run_with_post_verification_plan(tmp_path)
    )
    bad_verification = dict(verification)
    bad_verification["metadata"] = "bad"
    io.atomic_write_json(
        contracts.run_execution_verification_record_json_path(run_id, tmp_path),
        bad_verification,
    )
    record = _post_verification_execution_request_record(
        run_id, result, verification, pvfp
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(ValueError, match="metadata must be a dict"):
        events.request_post_verification_execution(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_invalid_post_verification_followup_plan_schema_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp = (
        _run_with_post_verification_plan(tmp_path)
    )
    bad_pvfp = dict(pvfp)
    bad_pvfp["metadata"] = "bad"
    io.atomic_write_json(
        contracts.run_post_verification_followup_plan_record_json_path(
            run_id, tmp_path
        ),
        bad_pvfp,
    )
    record = _post_verification_execution_request_record(
        run_id, result, verification, pvfp
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(ValueError, match="metadata must be a dict"):
        events.request_post_verification_execution(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_source_execution_result_fingerprint_mismatch_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp = (
        _run_with_post_verification_plan(tmp_path)
    )
    record = _post_verification_execution_request_record(
        run_id, result, verification, pvfp
    )
    bad = dict(record)
    bad["source_execution_result_fingerprint"] = "wrong"
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(InvalidTransition, match="source_execution_result_fingerprint"):
        events.request_post_verification_execution(
            tmp_path, run_id, bad, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_source_execution_verification_fingerprint_mismatch_fails_no_side_effects(
    tmp_path,
):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp = (
        _run_with_post_verification_plan(tmp_path)
    )
    record = _post_verification_execution_request_record(
        run_id, result, verification, pvfp
    )
    bad = dict(record)
    bad["source_execution_verification_fingerprint"] = "wrong"
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(
        InvalidTransition, match="source_execution_verification_fingerprint"
    ):
        events.request_post_verification_execution(
            tmp_path, run_id, bad, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_source_post_verification_followup_plan_fingerprint_mismatch_fails_no_side_effects(
    tmp_path,
):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp = (
        _run_with_post_verification_plan(tmp_path)
    )
    record = _post_verification_execution_request_record(
        run_id, result, verification, pvfp
    )
    bad = dict(record)
    bad["source_post_verification_followup_plan_fingerprint"] = "wrong"
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(
        InvalidTransition, match="source_post_verification_followup_plan_fingerprint"
    ):
        events.request_post_verification_execution(
            tmp_path, run_id, bad, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_verification_record_stale_result_fingerprint_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp = (
        _run_with_post_verification_plan(tmp_path)
    )
    stale = dict(verification)
    stale["source_execution_result_fingerprint"] = "stale"
    io.atomic_write_json(
        contracts.run_execution_verification_record_json_path(run_id, tmp_path), stale
    )
    record = _post_verification_execution_request_record(
        run_id, result, verification, pvfp
    )
    record["source_execution_verification_fingerprint"] = (
        contracts.run_execution_verification_fingerprint(stale)
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(
        InvalidTransition,
        match="run_execution_verification_record source_execution_result_fingerprint",
    ):
        events.request_post_verification_execution(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_followup_plan_stale_result_fingerprint_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp = (
        _run_with_post_verification_plan(tmp_path)
    )
    stale_pvfp = dict(pvfp)
    stale_pvfp["source_execution_result_fingerprint"] = "stale"
    io.atomic_write_json(
        contracts.run_post_verification_followup_plan_record_json_path(
            run_id, tmp_path
        ),
        stale_pvfp,
    )
    record = _post_verification_execution_request_record(
        run_id, result, verification, pvfp
    )
    record["source_post_verification_followup_plan_fingerprint"] = (
        contracts.run_post_verification_followup_plan_fingerprint(stale_pvfp)
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(
        InvalidTransition,
        match="run_post_verification_followup_plan_record source_execution_result_fingerprint",
    ):
        events.request_post_verification_execution(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_followup_plan_stale_verification_fingerprint_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp = (
        _run_with_post_verification_plan(tmp_path)
    )
    stale_pvfp = dict(pvfp)
    stale_pvfp["source_execution_verification_fingerprint"] = "stale"
    io.atomic_write_json(
        contracts.run_post_verification_followup_plan_record_json_path(
            run_id, tmp_path
        ),
        stale_pvfp,
    )
    record = _post_verification_execution_request_record(
        run_id, result, verification, pvfp
    )
    record["source_post_verification_followup_plan_fingerprint"] = (
        contracts.run_post_verification_followup_plan_fingerprint(stale_pvfp)
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(
        InvalidTransition,
        match="run_post_verification_followup_plan_record source_execution_verification_fingerprint",
    ):
        events.request_post_verification_execution(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_unknown_source_post_verification_followup_item_id_fails_no_side_effects(
    tmp_path,
):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp = (
        _run_with_post_verification_plan(
            tmp_path,
            verification_decision=contracts.EXECUTION_VERIFICATION_REJECTED,
        )
    )
    derived = contracts.derive_post_verification_execution_request_items(
        pvfp["followup_items"]
    )
    derived[0]["source_post_verification_followup_item_id"] = "unknown-pvfp"
    record = _post_verification_execution_request_record(
        run_id, result, verification, pvfp, request_items=derived
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(
        InvalidTransition,
        match="unknown source_post_verification_followup_item_id",
    ):
        events.request_post_verification_execution(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_source_execution_item_id_mismatch_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp = (
        _run_with_post_verification_plan(
            tmp_path,
            verification_decision=contracts.EXECUTION_VERIFICATION_REJECTED,
        )
    )
    derived = contracts.derive_post_verification_execution_request_items(
        pvfp["followup_items"]
    )
    derived[0]["source_execution_item_id"] = "wrong-exec"
    record = _post_verification_execution_request_record(
        run_id, result, verification, pvfp, request_items=derived
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(InvalidTransition, match="source_execution_item_id does not match"):
        events.request_post_verification_execution(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_source_followup_item_id_mismatch_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp = (
        _run_with_post_verification_plan(
            tmp_path,
            verification_decision=contracts.EXECUTION_VERIFICATION_REJECTED,
        )
    )
    derived = contracts.derive_post_verification_execution_request_items(
        pvfp["followup_items"]
    )
    derived[0]["source_followup_item_id"] = "wrong-followup"
    record = _post_verification_execution_request_record(
        run_id, result, verification, pvfp, request_items=derived
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(InvalidTransition, match="source_followup_item_id does not match"):
        events.request_post_verification_execution(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_execution_kind_mismatch_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp = (
        _run_with_post_verification_plan(
            tmp_path,
            verification_decision=contracts.EXECUTION_VERIFICATION_REJECTED,
        )
    )
    derived = contracts.derive_post_verification_execution_request_items(
        pvfp["followup_items"]
    )
    derived[0]["execution_kind"] = "rerun_task"
    record = _post_verification_execution_request_record(
        run_id, result, verification, pvfp, request_items=derived
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(InvalidTransition, match="execution_kind does not match"):
        events.request_post_verification_execution(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_item_status_mismatch_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp = (
        _run_with_post_verification_plan(
            tmp_path,
            verification_decision=contracts.EXECUTION_VERIFICATION_REJECTED,
        )
    )
    derived = contracts.derive_post_verification_execution_request_items(
        pvfp["followup_items"]
    )
    derived[0]["item_status"] = "failed"
    record = _post_verification_execution_request_record(
        run_id, result, verification, pvfp, request_items=derived
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(InvalidTransition, match="item_status does not match"):
        events.request_post_verification_execution(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_verification_decision_mismatch_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp = (
        _run_with_post_verification_plan(
            tmp_path,
            verification_decision=contracts.EXECUTION_VERIFICATION_REJECTED,
        )
    )
    derived = contracts.derive_post_verification_execution_request_items(
        pvfp["followup_items"]
    )
    derived[0]["verification_decision"] = contracts.EXECUTION_ITEM_VERIFICATION_ACCEPTED
    record = _post_verification_execution_request_record(
        run_id, result, verification, pvfp, request_items=derived
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(InvalidTransition, match="verification_decision does not match"):
        events.request_post_verification_execution(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_followup_kind_mismatch_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp = (
        _run_with_post_verification_plan(
            tmp_path,
            verification_decision=contracts.EXECUTION_VERIFICATION_REJECTED,
        )
    )
    derived = contracts.derive_post_verification_execution_request_items(
        pvfp["followup_items"]
    )
    derived[0]["followup_kind"] = contracts.POST_VERIFICATION_FOLLOWUP_KIND_OTHER
    record = _post_verification_execution_request_record(
        run_id, result, verification, pvfp, request_items=derived
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(InvalidTransition, match="followup_kind does not match"):
        events.request_post_verification_execution(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


# --- G. Write order ---


def test_write_order_record_before_event(tmp_path):
    run_id, _, _, _, _, _, _, result, verification, pvfp = _run_with_post_verification_plan(
        tmp_path
    )
    record = _post_verification_execution_request_record(
        run_id, result, verification, pvfp
    )
    ops = []
    real_append = events.append_run_event

    def track_write(path, data):
        if str(path).endswith("run_post_verification_execution_request_record.json"):
            ops.append("write_record")
        return io.atomic_write_json(path, data)

    def track_append(r, e, base_dir=None):
        ops.append("append_event")
        return real_append(r, e, base_dir)

    with patch("htr.events.atomic_write_json", side_effect=track_write), patch(
        "htr.events.append_run_event", side_effect=track_append
    ):
        events.request_post_verification_execution(
            tmp_path, run_id, record, actor="human"
        )
    assert ops == ["write_record", "append_event"]


# --- H. Replay-only ---


def test_existing_post_verification_execution_request_event_id_none_raises(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp = (
        _run_with_post_verification_plan(tmp_path)
    )
    record = _post_verification_execution_request_record(
        run_id, result, verification, pvfp
    )
    events.request_post_verification_execution(
        tmp_path, run_id, record, actor="human", event_id=new_event_id()
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(InvalidTransition):
        events.request_post_verification_execution(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_existing_post_verification_execution_request_missing_event_raises(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp = (
        _run_with_post_verification_plan(tmp_path)
    )
    record = _post_verification_execution_request_record(
        run_id, result, verification, pvfp
    )
    io.atomic_write_json(
        contracts.run_post_verification_execution_request_record_json_path(
            run_id, tmp_path
        ),
        record,
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(InvalidTransition):
        events.request_post_verification_execution(
            tmp_path, run_id, record, actor="human", event_id=new_event_id()
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


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
                    "run_post_verification_execution_request_fingerprint": "different",
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
                    "source_execution_verification_fingerprint": "different",
                },
            },
            EventConflict,
        ),
        (
            lambda e: {
                **e,
                "payload": {
                    **e["payload"],
                    "source_post_verification_followup_plan_fingerprint": "different",
                },
            },
            EventConflict,
        ),
        (
            lambda e: {
                **e,
                "payload": {
                    **e["payload"],
                    "request_status": (
                        contracts.POST_VERIFICATION_EXECUTION_REQUEST_STATUS_REQUESTED
                    ),
                },
            },
            EventConflict,
        ),
        (
            lambda e: {**e, "payload": {**e["payload"], "requester": "other"}},
            EventConflict,
        ),
    ],
)
def test_existing_post_verification_execution_request_replay_semantic_mismatch(
    tmp_path, mutator, expected
):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp = (
        _run_with_post_verification_plan(
            tmp_path,
            verification_decision=contracts.EXECUTION_VERIFICATION_ACCEPTED,
        )
    )
    record = _post_verification_execution_request_record(
        run_id, result, verification, pvfp
    )
    event_id = new_event_id()
    events.request_post_verification_execution(
        tmp_path, run_id, record, actor="human", event_id=event_id
    )
    event = events.read_task_events(run_id, base_dir=tmp_path)[-1]
    bad_event = mutator(event)
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with patch("htr.events._find_run_event_by_id", return_value=bad_event):
        with pytest.raises(expected):
            events.request_post_verification_execution(
                tmp_path, run_id, record, actor="human", event_id=event_id
            )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_existing_post_verification_execution_request_same_event_id_same_semantic_returns_existing(
    tmp_path,
):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp = (
        _run_with_post_verification_plan(tmp_path)
    )
    record = _post_verification_execution_request_record(
        run_id, result, verification, pvfp
    )
    event_id = new_event_id()
    events.request_post_verification_execution(
        tmp_path, run_id, record, actor="human", event_id=event_id
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    returned = events.request_post_verification_execution(
        tmp_path, run_id, record, actor="human", event_id=event_id
    )
    assert returned == record
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_replay_only_no_writes_or_previous_record_mutation(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp = (
        _run_with_post_verification_plan(tmp_path)
    )
    record = _post_verification_execution_request_record(
        run_id, result, verification, pvfp
    )
    event_id = new_event_id()
    events.request_post_verification_execution(
        tmp_path, run_id, record, actor="human", event_id=event_id
    )
    result_before = contracts.run_execution_result_record_json_path(
        run_id, tmp_path
    ).read_bytes()
    verification_before = contracts.run_execution_verification_record_json_path(
        run_id, tmp_path
    ).read_bytes()
    followup_before = contracts.run_followup_plan_record_json_path(
        run_id, tmp_path
    ).read_bytes()
    pvfp_before = contracts.run_post_verification_followup_plan_record_json_path(
        run_id, tmp_path
    ).read_bytes()
    events.request_post_verification_execution(
        tmp_path, run_id, record, actor="human", event_id=event_id
    )
    assert (
        contracts.run_execution_result_record_json_path(run_id, tmp_path).read_bytes()
        == result_before
    )
    assert (
        contracts.run_execution_verification_record_json_path(
            run_id, tmp_path
        ).read_bytes()
        == verification_before
    )
    assert (
        contracts.run_followup_plan_record_json_path(run_id, tmp_path).read_bytes()
        == followup_before
    )
    assert (
        contracts.run_post_verification_followup_plan_record_json_path(
            run_id, tmp_path
        ).read_bytes()
        == pvfp_before
    )
    assert len(list(paths.tasks_dir(run_id, tmp_path).iterdir())) == len(task_ids)


# --- I. Normal idempotency ---


def test_idempotent_same_event_id_same_semantic(tmp_path):
    run_id, _, _, _, _, _, _, result, verification, pvfp = _run_with_post_verification_plan(
        tmp_path
    )
    record = _post_verification_execution_request_record(
        run_id, result, verification, pvfp
    )
    event_id = new_event_id()
    first = events.request_post_verification_execution(
        tmp_path, run_id, record, actor="human", event_id=event_id
    )
    second = events.request_post_verification_execution(
        tmp_path, run_id, record, actor="human", event_id=event_id
    )
    assert second == first


def test_same_event_id_different_semantic_raises_conflict(tmp_path):
    run_id, _, _, _, _, _, _, result, verification, pvfp = _run_with_post_verification_plan(
        tmp_path,
        verification_decision=contracts.EXECUTION_VERIFICATION_ACCEPTED,
    )
    event_id = new_event_id()
    first = _post_verification_execution_request_record(
        run_id, result, verification, pvfp
    )
    events.request_post_verification_execution(
        tmp_path, run_id, first, actor="human", event_id=event_id
    )
    second = _post_verification_execution_request_record(
        run_id, result, verification, pvfp, notes="different semantic"
    )
    with pytest.raises(EventConflict):
        events.request_post_verification_execution(
            tmp_path, run_id, second, actor="human", event_id=event_id
        )


def test_idempotency_requires_post_verification_followup_plan(tmp_path):
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
    plan = _plan_record(run_id)
    io.atomic_write_json(
        contracts.run_followup_plan_record_json_path(run_id, tmp_path), plan
    )
    request = contracts.make_run_execution_request_record(
        run_id=run_id,
        source_followup_plan_fingerprint=contracts.run_followup_plan_fingerprint(plan),
        execution_items=[_sample_execution_item()],
    )
    io.atomic_write_json(
        contracts.run_execution_request_record_json_path(run_id, tmp_path), request
    )
    stub_result = contracts.make_run_execution_result_record(
        run_id=run_id,
        source_execution_request_fingerprint=contracts.run_execution_request_fingerprint(
            request
        ),
        item_results=[],
    )
    io.atomic_write_json(
        contracts.run_execution_result_record_json_path(run_id, tmp_path), stub_result
    )
    stub_verification = contracts.make_run_execution_verification_record(
        run_id=run_id,
        source_execution_result_fingerprint=contracts.run_execution_result_fingerprint(
            stub_result
        ),
        verification_decision=contracts.EXECUTION_VERIFICATION_ACCEPTED,
        item_verifications=[],
    )
    io.atomic_write_json(
        contracts.run_execution_verification_record_json_path(run_id, tmp_path),
        stub_verification,
    )
    stub_request = _minimal_post_verification_execution_request(run_id=run_id)
    with pytest.raises(
        InvalidTransition,
        match="run_post_verification_followup_plan_record.json is missing",
    ):
        events.request_post_verification_execution(
            tmp_path,
            run_id,
            stub_request,
            actor="human",
            event_id=new_event_id(),
        )


def test_idempotency_does_not_allow_non_completed_run(tmp_path):
    run_id = new_run_id()
    io.create_run_workspace(run_id, base_dir=tmp_path)
    with pytest.raises(InvalidTransition):
        events.request_post_verification_execution(
            tmp_path,
            run_id,
            _minimal_post_verification_execution_request(run_id=run_id),
            actor="human",
            event_id=new_event_id(),
        )


# --- J. Boundary preservation ---


def test_boundaries_preserved_on_success(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp = (
        _run_with_post_verification_plan(tmp_path)
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    record = _post_verification_execution_request_record(
        run_id, result, verification, pvfp
    )
    events.request_post_verification_execution(
        tmp_path, run_id, record, actor="human"
    )
    after = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    assert after["manifest"] == before["manifest"]
    assert after["completion"] == before["completion"]
    assert after["review"] == before["review"]
    assert after["followup"] == before["followup"]
    assert after["request"] == before["request"]
    assert after["result"] == before["result"]
    assert after["verification"] == before["verification"]
    assert after["post_verification_plan"] == before["post_verification_plan"]
    assert after["task_statuses"] == before["task_statuses"]
    assert after["attempt_statuses"] == before["attempt_statuses"]
    assert after["post_verification_execution_request_exists"] is True
    assert len(list(paths.tasks_dir(run_id, tmp_path).iterdir())) == len(task_ids)
    task_id = task_ids[0]
    attempt_id = attempt_ids[0]
    assert paths.result_json_path(run_id, task_id, attempt_id, tmp_path).exists()
    assert contracts.verification_result_json_path(
        run_id, task_id, attempt_id, tmp_path
    ).exists()
    result_before = paths.result_json_path(
        run_id, task_id, attempt_id, tmp_path
    ).read_bytes()
    verification_result_before = contracts.verification_result_json_path(
        run_id, task_id, attempt_id, tmp_path
    ).read_bytes()
    assert (
        paths.result_json_path(run_id, task_id, attempt_id, tmp_path).read_bytes()
        == result_before
    )
    assert (
        contracts.verification_result_json_path(
            run_id, task_id, attempt_id, tmp_path
        ).read_bytes()
        == verification_result_before
    )


# --- K. Import/call boundary ---


def test_task13_modules_do_not_import_forbidden_modules():
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


def test_package_exports_task13_apis():
    import htr

    for name in (
        "make_run_post_verification_execution_request_record",
        "run_post_verification_execution_request_fingerprint",
        "derive_post_verification_execution_request_items",
        "validate_post_verification_execution_request_items_correspond",
        "request_post_verification_execution",
        "EVENT_TYPE_RUN_POST_VERIFICATION_EXECUTION_REQUESTED",
        "POST_VERIFICATION_EXECUTION_REQUEST_STATUS_REQUESTED",
        "POST_VERIFICATION_EXECUTION_REQUEST_STATUS_EMPTY",
        "POST_VERIFICATION_EXECUTION_REQUEST_KIND_REOPEN_LINK_MANUALLY",
        "POST_VERIFICATION_EXECUTION_REQUEST_KIND_UPDATE_DOCUMENTATION_MANUALLY",
        "POST_VERIFICATION_EXECUTION_REQUEST_KIND_PREPARE_NEW_EXECUTION_REQUEST",
        "POST_VERIFICATION_EXECUTION_REQUEST_KIND_REVIEW_REJECTED_ITEM",
        "POST_VERIFICATION_EXECUTION_REQUEST_KIND_REVIEW_NEEDS_CHANGES_ITEM",
        "POST_VERIFICATION_EXECUTION_REQUEST_KIND_REVIEW_NOT_REVIEWED_ITEM",
        "POST_VERIFICATION_EXECUTION_REQUEST_KIND_OTHER",
    ):
        assert hasattr(htr, name)
