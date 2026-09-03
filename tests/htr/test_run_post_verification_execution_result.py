"""Tests for Task 14 — Manual Post-Verification Execution Result Recording."""

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


def _run_with_post_verification_execution_request(
    tmp_path,
    execution_items=None,
    verification_decision=contracts.EXECUTION_VERIFICATION_REJECTED,
    followup_items=None,
):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp = (
        _run_with_post_verification_plan(
            tmp_path,
            execution_items=execution_items,
            verification_decision=verification_decision,
            followup_items=followup_items,
        )
    )
    pver = _post_verification_execution_request_record(
        run_id, result, verification, pvfp
    )
    events.request_post_verification_execution(
        tmp_path, run_id, pver, actor="human"
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
        pvfp,
        pver,
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


def _sample_post_verification_result_item(**overrides):
    item = {
        "result_item_id": "pve-result-1",
        "source_request_item_id": "pver-pvfp-exec-1",
        "source_post_verification_followup_item_id": "pvfp-exec-1",
        "source_execution_item_id": "exec-1",
        "source_followup_item_id": "followup-1",
        "request_kind": contracts.POST_VERIFICATION_EXECUTION_REQUEST_KIND_REVIEW_REJECTED_ITEM,
        "execution_kind": "manual_open_link",
        "item_status": contracts.EXECUTION_ITEM_COMPLETED,
        "verification_decision": contracts.EXECUTION_ITEM_VERIFICATION_REJECTED,
        "followup_kind": contracts.POST_VERIFICATION_FOLLOWUP_KIND_REVIEW_REJECTED_ITEM,
        "result_item_status": contracts.POST_VERIFICATION_EXECUTION_RESULT_ITEM_STATUS_COMPLETED,
        "outcome": "Re-reviewed manually",
        "evidence": {},
        "metadata": {},
    }
    item.update(overrides)
    return item


def _result_item_from_request(request_item, **overrides):
    item = _sample_post_verification_result_item(
        result_item_id=f"result-{request_item['request_item_id']}",
        source_request_item_id=request_item["request_item_id"],
        source_post_verification_followup_item_id=request_item.get(
            "source_post_verification_followup_item_id"
        ),
        source_execution_item_id=request_item.get("source_execution_item_id"),
        source_followup_item_id=request_item.get("source_followup_item_id"),
        request_kind=request_item.get("request_kind"),
        execution_kind=request_item.get("execution_kind"),
        item_status=request_item.get("item_status"),
        verification_decision=request_item.get("verification_decision"),
        followup_kind=request_item.get("followup_kind"),
    )
    item.update(overrides)
    return item


def _post_verification_execution_result_record(
    run_id,
    result_record,
    verification_record,
    post_verification_plan_record,
    post_verification_execution_request_record,
    **kwargs,
):
    result_items = kwargs.pop("result_items", None)
    if result_items is None:
        request_items = post_verification_execution_request_record["request_items"]
        result_items = (
            [_result_item_from_request(request_items[0])] if request_items else []
        )
    return contracts.make_run_post_verification_execution_result_record(
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
        source_post_verification_execution_request_fingerprint=contracts.run_post_verification_execution_request_fingerprint(
            post_verification_execution_request_record
        ),
        result_items=result_items,
        **kwargs,
    )


def _minimal_post_verification_execution_result(**overrides):
    run_id = overrides.pop("run_id", new_run_id())
    result_items = overrides.pop("result_items", [])
    result_status = overrides.pop("result_status", None)
    if result_status is None:
        result_status = contracts.compute_post_verification_execution_result_status(
            result_items
        )
    record = {
        "schema_version": 1,
        "run_id": run_id,
        "source_execution_result_fingerprint": "fp-result",
        "source_execution_verification_fingerprint": "fp-verification",
        "source_post_verification_followup_plan_fingerprint": "fp-pvfp",
        "source_post_verification_execution_request_fingerprint": "fp-pver",
        "executor": "human",
        "result_status": result_status,
        "result_items": result_items,
        "notes": None,
        "metadata": {},
        "created_at": "2026-07-19T08:00:00+00:00",
    }
    record.update(overrides)
    return record


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
        "created_at": "2026-07-19T08:00:00+00:00",
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
    pve_result_path = contracts.run_post_verification_execution_result_record_json_path(
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
        "post_verification_execution_result": (
            pve_result_path.read_bytes() if pve_result_path.exists() else None
        ),
        "post_verification_execution_result_exists": pve_result_path.exists(),
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


def test_valid_run_post_verification_execution_result_record_passes():
    record = _minimal_post_verification_execution_result(
        result_items=[_sample_post_verification_result_item()],
        result_status=contracts.POST_VERIFICATION_EXECUTION_RESULT_STATUS_COMPLETED,
    )
    schemas.validate(record, "run_post_verification_execution_result_record")


@pytest.mark.parametrize(
    "missing",
    [
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
    ],
)
def test_missing_required_fields_fail(missing):
    record = _minimal_post_verification_execution_result()
    del record[missing]
    with pytest.raises(ValueError, match="missing fields"):
        schemas.validate(record, "run_post_verification_execution_result_record")


def test_invalid_run_id_fails_schema():
    record = _minimal_post_verification_execution_result(run_id="bad")
    with pytest.raises(ValueError, match="run_id must be a valid run id"):
        schemas.validate(record, "run_post_verification_execution_result_record")


def test_invalid_source_execution_result_fingerprint_fails():
    with pytest.raises(ValueError, match="source_execution_result_fingerprint"):
        contracts.make_run_post_verification_execution_result_record(
            run_id=new_run_id(),
            source_execution_result_fingerprint="",
            source_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            source_post_verification_execution_request_fingerprint="fp-pver",
            result_items=[],
        )


def test_invalid_source_execution_verification_fingerprint_fails():
    with pytest.raises(ValueError, match="source_execution_verification_fingerprint"):
        contracts.make_run_post_verification_execution_result_record(
            run_id=new_run_id(),
            source_execution_result_fingerprint="fp-result",
            source_execution_verification_fingerprint="",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            source_post_verification_execution_request_fingerprint="fp-pver",
            result_items=[],
        )


def test_invalid_source_post_verification_followup_plan_fingerprint_fails():
    with pytest.raises(
        ValueError, match="source_post_verification_followup_plan_fingerprint"
    ):
        contracts.make_run_post_verification_execution_result_record(
            run_id=new_run_id(),
            source_execution_result_fingerprint="fp-result",
            source_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="",
            source_post_verification_execution_request_fingerprint="fp-pver",
            result_items=[],
        )


def test_invalid_source_post_verification_execution_request_fingerprint_fails():
    with pytest.raises(
        ValueError, match="source_post_verification_execution_request_fingerprint"
    ):
        contracts.make_run_post_verification_execution_result_record(
            run_id=new_run_id(),
            source_execution_result_fingerprint="fp-result",
            source_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            source_post_verification_execution_request_fingerprint="",
            result_items=[],
        )


def test_invalid_executor_fails():
    with pytest.raises(ValueError, match="executor must be a non-empty string"):
        contracts.make_run_post_verification_execution_result_record(
            run_id=new_run_id(),
            source_execution_result_fingerprint="fp-result",
            source_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            source_post_verification_execution_request_fingerprint="fp-pver",
            result_items=[],
            executor="",
        )


def test_invalid_result_status_fails():
    with pytest.raises(ValueError, match="result_status must be one of"):
        contracts.make_run_post_verification_execution_result_record(
            run_id=new_run_id(),
            source_execution_result_fingerprint="fp-result",
            source_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            source_post_verification_execution_request_fingerprint="fp-pver",
            result_items=[],
            result_status="draft",
        )


def test_result_items_must_be_list():
    record = _minimal_post_verification_execution_result()
    record["result_items"] = {}
    with pytest.raises(ValueError, match="result_items must be a list"):
        schemas.validate(record, "run_post_verification_execution_result_record")


def test_result_items_entries_must_be_dicts():
    record = _minimal_post_verification_execution_result()
    record["result_items"] = ["bad"]
    with pytest.raises(ValueError, match="result_items entries must be dicts"):
        schemas.validate(record, "run_post_verification_execution_result_record")


def test_missing_result_item_id_fails():
    with pytest.raises(KeyError, match="result_item_id"):
        contracts.make_run_post_verification_execution_result_record(
            run_id=new_run_id(),
            source_execution_result_fingerprint="fp-result",
            source_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            source_post_verification_execution_request_fingerprint="fp-pver",
            result_items=[
                {
                    "source_request_item_id": None,
                    "result_item_status": (
                        contracts.POST_VERIFICATION_EXECUTION_RESULT_ITEM_STATUS_COMPLETED
                    ),
                    "outcome": None,
                    "evidence": {},
                    "metadata": {},
                }
            ],
        )


def test_invalid_result_item_status_fails():
    with pytest.raises(ValueError, match="result_item_status"):
        contracts.make_run_post_verification_execution_result_record(
            run_id=new_run_id(),
            source_execution_result_fingerprint="fp-result",
            source_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            source_post_verification_execution_request_fingerprint="fp-pver",
            result_items=[
                _sample_post_verification_result_item(result_item_status="draft")
            ],
        )


def test_source_request_item_id_must_be_string_or_none():
    with pytest.raises(ValueError, match="source_request_item_id"):
        contracts.make_run_post_verification_execution_result_record(
            run_id=new_run_id(),
            source_execution_result_fingerprint="fp-result",
            source_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            source_post_verification_execution_request_fingerprint="fp-pver",
            result_items=[_sample_post_verification_result_item(source_request_item_id=123)],
        )


def test_source_post_verification_followup_item_id_must_be_string_or_none():
    with pytest.raises(ValueError, match="source_post_verification_followup_item_id"):
        contracts.make_run_post_verification_execution_result_record(
            run_id=new_run_id(),
            source_execution_result_fingerprint="fp-result",
            source_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            source_post_verification_execution_request_fingerprint="fp-pver",
            result_items=[
                _sample_post_verification_result_item(
                    source_post_verification_followup_item_id=123
                )
            ],
        )


def test_source_execution_item_id_must_be_string_or_none():
    with pytest.raises(ValueError, match="source_execution_item_id"):
        contracts.make_run_post_verification_execution_result_record(
            run_id=new_run_id(),
            source_execution_result_fingerprint="fp-result",
            source_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            source_post_verification_execution_request_fingerprint="fp-pver",
            result_items=[_sample_post_verification_result_item(source_execution_item_id=123)],
        )


def test_source_followup_item_id_must_be_string_or_none():
    with pytest.raises(ValueError, match="source_followup_item_id"):
        contracts.make_run_post_verification_execution_result_record(
            run_id=new_run_id(),
            source_execution_result_fingerprint="fp-result",
            source_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            source_post_verification_execution_request_fingerprint="fp-pver",
            result_items=[_sample_post_verification_result_item(source_followup_item_id=123)],
        )


def test_request_kind_must_be_string_or_none():
    with pytest.raises(ValueError, match="request_kind"):
        contracts.make_run_post_verification_execution_result_record(
            run_id=new_run_id(),
            source_execution_result_fingerprint="fp-result",
            source_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            source_post_verification_execution_request_fingerprint="fp-pver",
            result_items=[_sample_post_verification_result_item(request_kind=123)],
        )


def test_execution_kind_must_be_string_or_none():
    with pytest.raises(ValueError, match="execution_kind"):
        contracts.make_run_post_verification_execution_result_record(
            run_id=new_run_id(),
            source_execution_result_fingerprint="fp-result",
            source_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            source_post_verification_execution_request_fingerprint="fp-pver",
            result_items=[_sample_post_verification_result_item(execution_kind=123)],
        )


def test_item_status_must_be_string_or_none():
    with pytest.raises(ValueError, match="item_status"):
        contracts.make_run_post_verification_execution_result_record(
            run_id=new_run_id(),
            source_execution_result_fingerprint="fp-result",
            source_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            source_post_verification_execution_request_fingerprint="fp-pver",
            result_items=[_sample_post_verification_result_item(item_status=123)],
        )


def test_verification_decision_must_be_string_or_none():
    with pytest.raises(ValueError, match="verification_decision"):
        contracts.make_run_post_verification_execution_result_record(
            run_id=new_run_id(),
            source_execution_result_fingerprint="fp-result",
            source_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            source_post_verification_execution_request_fingerprint="fp-pver",
            result_items=[_sample_post_verification_result_item(verification_decision=123)],
        )


def test_followup_kind_must_be_string_or_none():
    with pytest.raises(ValueError, match="followup_kind"):
        contracts.make_run_post_verification_execution_result_record(
            run_id=new_run_id(),
            source_execution_result_fingerprint="fp-result",
            source_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            source_post_verification_execution_request_fingerprint="fp-pver",
            result_items=[_sample_post_verification_result_item(followup_kind=123)],
        )


def test_outcome_must_be_string_or_none():
    with pytest.raises(ValueError, match="outcome"):
        contracts.make_run_post_verification_execution_result_record(
            run_id=new_run_id(),
            source_execution_result_fingerprint="fp-result",
            source_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            source_post_verification_execution_request_fingerprint="fp-pver",
            result_items=[_sample_post_verification_result_item(outcome=123)],
        )


def test_evidence_must_be_dict():
    with pytest.raises(ValueError, match="evidence"):
        contracts.make_run_post_verification_execution_result_record(
            run_id=new_run_id(),
            source_execution_result_fingerprint="fp-result",
            source_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            source_post_verification_execution_request_fingerprint="fp-pver",
            result_items=[_sample_post_verification_result_item(evidence="bad")],
        )


def test_item_metadata_must_be_dict():
    with pytest.raises(ValueError, match="metadata"):
        contracts.make_run_post_verification_execution_result_record(
            run_id=new_run_id(),
            source_execution_result_fingerprint="fp-result",
            source_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            source_post_verification_execution_request_fingerprint="fp-pver",
            result_items=[_sample_post_verification_result_item(metadata="bad")],
        )


def test_notes_must_be_string_or_none():
    record = _minimal_post_verification_execution_result()
    record["notes"] = 1
    with pytest.raises(ValueError, match="notes must be a string or null"):
        schemas.validate(record, "run_post_verification_execution_result_record")


def test_metadata_must_be_dict():
    with pytest.raises(ValueError, match="metadata must be a dict"):
        contracts.make_run_post_verification_execution_result_record(
            run_id=new_run_id(),
            source_execution_result_fingerprint="fp-result",
            source_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            source_post_verification_execution_request_fingerprint="fp-pver",
            result_items=[],
            metadata=[],
        )


def test_empty_result_status_requires_empty_result_items():
    with pytest.raises(ValueError, match="empty result_status requires result_items"):
        contracts.make_run_post_verification_execution_result_record(
            run_id=new_run_id(),
            source_execution_result_fingerprint="fp-result",
            source_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            source_post_verification_execution_request_fingerprint="fp-pver",
            result_items=[_sample_post_verification_result_item()],
            result_status=contracts.POST_VERIFICATION_EXECUTION_RESULT_STATUS_EMPTY,
        )


def test_completed_result_status_requires_completed_items():
    with pytest.raises(ValueError, match="completed result_status requires"):
        contracts.make_run_post_verification_execution_result_record(
            run_id=new_run_id(),
            source_execution_result_fingerprint="fp-result",
            source_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            source_post_verification_execution_request_fingerprint="fp-pver",
            result_items=[
                _sample_post_verification_result_item(
                    result_item_status=(
                        contracts.POST_VERIFICATION_EXECUTION_RESULT_ITEM_STATUS_FAILED
                    )
                )
            ],
            result_status=contracts.POST_VERIFICATION_EXECUTION_RESULT_STATUS_COMPLETED,
        )


def test_failed_result_status_requires_failed_items():
    with pytest.raises(ValueError, match="failed result_status requires"):
        contracts.make_run_post_verification_execution_result_record(
            run_id=new_run_id(),
            source_execution_result_fingerprint="fp-result",
            source_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            source_post_verification_execution_request_fingerprint="fp-pver",
            result_items=[
                _sample_post_verification_result_item(
                    result_item_status=(
                        contracts.POST_VERIFICATION_EXECUTION_RESULT_ITEM_STATUS_COMPLETED
                    )
                )
            ],
            result_status=contracts.POST_VERIFICATION_EXECUTION_RESULT_STATUS_FAILED,
        )


def test_partial_result_status_requires_mixed_or_skipped_items():
    with pytest.raises(ValueError, match="partial result_status requires"):
        contracts.make_run_post_verification_execution_result_record(
            run_id=new_run_id(),
            source_execution_result_fingerprint="fp-result",
            source_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            source_post_verification_execution_request_fingerprint="fp-pver",
            result_items=[
                _sample_post_verification_result_item(
                    result_item_status=(
                        contracts.POST_VERIFICATION_EXECUTION_RESULT_ITEM_STATUS_COMPLETED
                    )
                )
            ],
            result_status=contracts.POST_VERIFICATION_EXECUTION_RESULT_STATUS_PARTIAL,
        )


# --- B. Factory ---


def test_make_run_post_verification_execution_result_record_returns_valid_schema(
    tmp_path,
):
    run_id, _, _, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    record = _post_verification_execution_result_record(
        run_id, result, verification, pvfp, pver
    )
    schemas.validate(record, "run_post_verification_execution_result_record")


def test_factory_metadata_defaults_to_empty_dict(tmp_path):
    run_id, _, _, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    record = contracts.make_run_post_verification_execution_result_record(
        run_id=run_id,
        source_execution_result_fingerprint=contracts.run_execution_result_fingerprint(
            result
        ),
        source_execution_verification_fingerprint=contracts.run_execution_verification_fingerprint(
            verification
        ),
        source_post_verification_followup_plan_fingerprint=contracts.run_post_verification_followup_plan_fingerprint(
            pvfp
        ),
        source_post_verification_execution_request_fingerprint=contracts.run_post_verification_execution_request_fingerprint(
            pver
        ),
        result_items=[],
    )
    assert record["metadata"] == {}


def test_factory_notes_remain_none(tmp_path):
    run_id, _, _, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    record = contracts.make_run_post_verification_execution_result_record(
        run_id=run_id,
        source_execution_result_fingerprint=contracts.run_execution_result_fingerprint(
            result
        ),
        source_execution_verification_fingerprint=contracts.run_execution_verification_fingerprint(
            verification
        ),
        source_post_verification_followup_plan_fingerprint=contracts.run_post_verification_followup_plan_fingerprint(
            pvfp
        ),
        source_post_verification_execution_request_fingerprint=contracts.run_post_verification_execution_request_fingerprint(
            pver
        ),
        result_items=[],
        notes=None,
    )
    assert record["notes"] is None


def test_factory_normalizes_evidence_and_item_metadata(tmp_path):
    run_id, _, _, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    record = contracts.make_run_post_verification_execution_result_record(
        run_id=run_id,
        source_execution_result_fingerprint=contracts.run_execution_result_fingerprint(
            result
        ),
        source_execution_verification_fingerprint=contracts.run_execution_verification_fingerprint(
            verification
        ),
        source_post_verification_followup_plan_fingerprint=contracts.run_post_verification_followup_plan_fingerprint(
            pvfp
        ),
        source_post_verification_execution_request_fingerprint=contracts.run_post_verification_execution_request_fingerprint(
            pver
        ),
        result_items=[
            {
                "result_item_id": "ri-1",
                "source_request_item_id": None,
                "result_item_status": (
                    contracts.POST_VERIFICATION_EXECUTION_RESULT_ITEM_STATUS_COMPLETED
                ),
                "outcome": None,
            }
        ],
    )
    assert record["result_items"][0]["evidence"] == {}
    assert record["result_items"][0]["metadata"] == {}


def test_factory_outcome_remains_none(tmp_path):
    run_id, _, _, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    record = contracts.make_run_post_verification_execution_result_record(
        run_id=run_id,
        source_execution_result_fingerprint=contracts.run_execution_result_fingerprint(
            result
        ),
        source_execution_verification_fingerprint=contracts.run_execution_verification_fingerprint(
            verification
        ),
        source_post_verification_followup_plan_fingerprint=contracts.run_post_verification_followup_plan_fingerprint(
            pvfp
        ),
        source_post_verification_execution_request_fingerprint=contracts.run_post_verification_execution_request_fingerprint(
            pver
        ),
        result_items=[
            {
                "result_item_id": "ri-1",
                "source_request_item_id": None,
                "result_item_status": (
                    contracts.POST_VERIFICATION_EXECUTION_RESULT_ITEM_STATUS_COMPLETED
                ),
                "outcome": None,
            }
        ],
    )
    assert record["result_items"][0]["outcome"] is None


def test_factory_does_not_write_files(tmp_path):
    run_id, _, _, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    _post_verification_execution_result_record(run_id, result, verification, pvfp, pver)
    assert not contracts.run_post_verification_execution_result_record_json_path(
        run_id, tmp_path
    ).exists()


def test_factory_does_not_create_events(tmp_path):
    run_id, _, _, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    before = len(events.read_task_events(run_id, base_dir=tmp_path))
    _post_verification_execution_result_record(run_id, result, verification, pvfp, pver)
    assert len(events.read_task_events(run_id, base_dir=tmp_path)) == before


def test_factory_does_not_execute_external_systems(tmp_path):
    run_id, _, _, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    with patch("subprocess.run") as mock_subprocess, patch("webbrowser.open") as browser:
        _post_verification_execution_result_record(
            run_id, result, verification, pvfp, pver
        )
    mock_subprocess.assert_not_called()
    browser.assert_not_called()


# --- C. Fingerprint ---


def test_fingerprint_validates_schema_first():
    bad = _minimal_post_verification_execution_result()
    bad["metadata"] = "bad"
    with pytest.raises(ValueError, match="metadata must be a dict"):
        contracts.run_post_verification_execution_result_fingerprint(bad)


def test_equivalent_records_same_fingerprint():
    first = _minimal_post_verification_execution_result(
        result_items=[_sample_post_verification_result_item()],
        result_status=contracts.POST_VERIFICATION_EXECUTION_RESULT_STATUS_COMPLETED,
    )
    second = dict(first)
    assert contracts.run_post_verification_execution_result_fingerprint(
        first
    ) == contracts.run_post_verification_execution_result_fingerprint(second)


def test_changed_result_items_changes_fingerprint():
    base = _minimal_post_verification_execution_result(
        result_items=[_sample_post_verification_result_item()],
        result_status=contracts.POST_VERIFICATION_EXECUTION_RESULT_STATUS_COMPLETED,
    )
    changed = dict(base)
    changed["result_items"] = [
        _sample_post_verification_result_item(outcome="different outcome")
    ]
    assert contracts.run_post_verification_execution_result_fingerprint(
        base
    ) != contracts.run_post_verification_execution_result_fingerprint(changed)


def test_changed_metadata_changes_fingerprint():
    base = _minimal_post_verification_execution_result()
    changed = dict(base)
    changed["metadata"] = {"note": "changed"}
    assert contracts.run_post_verification_execution_result_fingerprint(
        base
    ) != contracts.run_post_verification_execution_result_fingerprint(changed)


def test_fingerprint_uses_canonical_json():
    record = _minimal_post_verification_execution_result(metadata={"z": 1, "a": 2})
    expected = json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    assert contracts.run_post_verification_execution_result_fingerprint(record) == expected


def test_fingerprint_does_not_read_files():
    record = _minimal_post_verification_execution_result()
    with patch("htr.io.read_json") as mock_read:
        contracts.run_post_verification_execution_result_fingerprint(record)
    mock_read.assert_not_called()


# --- D. Result item correspondence ---


def test_global_manual_result_items_with_null_source_request_item_id_allowed():
    request = _minimal_post_verification_execution_request(
        request_items=[_sample_post_verification_request_item()]
    )
    result_items = [
        _sample_post_verification_result_item(source_request_item_id=None)
    ]
    contracts.validate_post_verification_execution_result_items_correspond(
        result_items, request
    )


def test_request_linked_result_item_must_reference_existing_source_request_item_id():
    request = _minimal_post_verification_execution_request(
        request_items=[_sample_post_verification_request_item()]
    )
    result_items = [
        _sample_post_verification_result_item(source_request_item_id="missing")
    ]
    with pytest.raises(ValueError, match="unknown source_request_item_id"):
        contracts.validate_post_verification_execution_result_items_correspond(
            result_items, request
        )


def test_request_linked_result_item_preserves_source_post_verification_followup_item_id(
    tmp_path,
):
    run_id, _, _, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    result_item = _result_item_from_request(pver["request_items"][0])
    contracts.validate_post_verification_execution_result_items_correspond(
        [result_item], pver
    )
    assert result_item["source_post_verification_followup_item_id"] == pver[
        "request_items"
    ][0]["source_post_verification_followup_item_id"]


def test_request_linked_result_item_preserves_source_execution_item_id(tmp_path):
    run_id, _, _, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    result_item = _result_item_from_request(pver["request_items"][0])
    contracts.validate_post_verification_execution_result_items_correspond(
        [result_item], pver
    )
    assert result_item["source_execution_item_id"] == pver["request_items"][0][
        "source_execution_item_id"
    ]


def test_request_linked_result_item_preserves_source_followup_item_id(tmp_path):
    run_id, _, _, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    result_item = _result_item_from_request(pver["request_items"][0])
    contracts.validate_post_verification_execution_result_items_correspond(
        [result_item], pver
    )
    assert result_item["source_followup_item_id"] == pver["request_items"][0][
        "source_followup_item_id"
    ]


def test_request_linked_result_item_preserves_request_kind(tmp_path):
    run_id, _, _, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    result_item = _result_item_from_request(pver["request_items"][0])
    contracts.validate_post_verification_execution_result_items_correspond(
        [result_item], pver
    )
    assert result_item["request_kind"] == pver["request_items"][0]["request_kind"]


def test_request_linked_result_item_preserves_execution_kind(tmp_path):
    run_id, _, _, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    result_item = _result_item_from_request(pver["request_items"][0])
    contracts.validate_post_verification_execution_result_items_correspond(
        [result_item], pver
    )
    assert result_item["execution_kind"] == pver["request_items"][0]["execution_kind"]


def test_request_linked_result_item_preserves_item_status(tmp_path):
    run_id, _, _, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    result_item = _result_item_from_request(pver["request_items"][0])
    contracts.validate_post_verification_execution_result_items_correspond(
        [result_item], pver
    )
    assert result_item["item_status"] == pver["request_items"][0]["item_status"]


def test_request_linked_result_item_preserves_verification_decision(tmp_path):
    run_id, _, _, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    result_item = _result_item_from_request(pver["request_items"][0])
    contracts.validate_post_verification_execution_result_items_correspond(
        [result_item], pver
    )
    assert result_item["verification_decision"] == pver["request_items"][0][
        "verification_decision"
    ]


def test_request_linked_result_item_preserves_followup_kind(tmp_path):
    run_id, _, _, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    result_item = _result_item_from_request(pver["request_items"][0])
    contracts.validate_post_verification_execution_result_items_correspond(
        [result_item], pver
    )
    assert result_item["followup_kind"] == pver["request_items"][0]["followup_kind"]


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("source_post_verification_followup_item_id", "wrong-pvfp", "source_post_verification_followup_item_id"),
        ("source_execution_item_id", "wrong-exec", "source_execution_item_id"),
        ("source_followup_item_id", "wrong-followup", "source_followup_item_id"),
        ("request_kind", "other_kind", "request_kind"),
        ("execution_kind", "rerun_task", "execution_kind"),
        ("item_status", "failed", "item_status"),
        ("verification_decision", "accepted", "verification_decision"),
        ("followup_kind", contracts.POST_VERIFICATION_FOLLOWUP_KIND_OTHER, "followup_kind"),
    ],
)
def test_mismatch_for_preserved_source_field_fails(field, value, match):
    request = _minimal_post_verification_execution_request(
        request_items=[_sample_post_verification_request_item()]
    )
    result_item = _result_item_from_request(request["request_items"][0])
    result_item[field] = value
    with pytest.raises(ValueError, match=f"{match} does not match"):
        contracts.validate_post_verification_execution_result_items_correspond(
            [result_item], request
        )


# --- E. Record lifecycle success ---


def test_record_post_verification_execution_result_succeeds_after_full_chain(tmp_path):
    run_id, _, _, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    record = _post_verification_execution_result_record(
        run_id, result, verification, pvfp, pver
    )
    returned = events.record_post_verification_execution_result(
        tmp_path, run_id, record, actor="recorder"
    )
    assert returned == record


def test_source_execution_result_fingerprint_must_match_current(tmp_path):
    run_id, _, _, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    record = _post_verification_execution_result_record(
        run_id, result, verification, pvfp, pver
    )
    assert record["source_execution_result_fingerprint"] == (
        contracts.run_execution_result_fingerprint(result)
    )
    events.record_post_verification_execution_result(
        tmp_path, run_id, record, actor="recorder"
    )
    assert io.read_json(
        contracts.run_post_verification_execution_result_record_json_path(
            run_id, tmp_path
        )
    )["source_execution_result_fingerprint"] == contracts.run_execution_result_fingerprint(
        result
    )


def test_source_execution_verification_fingerprint_must_match_current(tmp_path):
    run_id, _, _, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    record = _post_verification_execution_result_record(
        run_id, result, verification, pvfp, pver
    )
    assert record["source_execution_verification_fingerprint"] == (
        contracts.run_execution_verification_fingerprint(verification)
    )


def test_source_post_verification_followup_plan_fingerprint_must_match_current(tmp_path):
    run_id, _, _, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    record = _post_verification_execution_result_record(
        run_id, result, verification, pvfp, pver
    )
    assert record["source_post_verification_followup_plan_fingerprint"] == (
        contracts.run_post_verification_followup_plan_fingerprint(pvfp)
    )


def test_source_post_verification_execution_request_fingerprint_must_match_current(
    tmp_path,
):
    run_id, _, _, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    record = _post_verification_execution_result_record(
        run_id, result, verification, pvfp, pver
    )
    assert record["source_post_verification_execution_request_fingerprint"] == (
        contracts.run_post_verification_execution_request_fingerprint(pver)
    )


def test_verification_record_points_to_current_execution_result_fingerprint(tmp_path):
    run_id, _, _, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    assert verification["source_execution_result_fingerprint"] == (
        contracts.run_execution_result_fingerprint(result)
    )


def test_post_verification_followup_plan_points_to_current_result_and_verification(
    tmp_path,
):
    run_id, _, _, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    assert pvfp["source_execution_result_fingerprint"] == (
        contracts.run_execution_result_fingerprint(result)
    )
    assert pvfp["source_execution_verification_fingerprint"] == (
        contracts.run_execution_verification_fingerprint(verification)
    )


def test_post_verification_execution_request_points_to_current_chain_fingerprints(
    tmp_path,
):
    run_id, _, _, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    assert pver["source_execution_result_fingerprint"] == (
        contracts.run_execution_result_fingerprint(result)
    )
    assert pver["source_execution_verification_fingerprint"] == (
        contracts.run_execution_verification_fingerprint(verification)
    )
    assert pver["source_post_verification_followup_plan_fingerprint"] == (
        contracts.run_post_verification_followup_plan_fingerprint(pvfp)
    )


def test_request_linked_result_items_correspond_to_execution_request(tmp_path):
    run_id, _, _, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    record = _post_verification_execution_result_record(
        run_id, result, verification, pvfp, pver
    )
    contracts.validate_post_verification_execution_result_items_correspond(
        record["result_items"], pver
    )
    events.record_post_verification_execution_result(
        tmp_path, run_id, record, actor="human"
    )


@pytest.mark.parametrize(
    "result_status,result_items",
    [
        (
            contracts.POST_VERIFICATION_EXECUTION_RESULT_STATUS_EMPTY,
            [],
        ),
        (
            contracts.POST_VERIFICATION_EXECUTION_RESULT_STATUS_COMPLETED,
            [
                _sample_post_verification_result_item(
                    result_item_status=(
                        contracts.POST_VERIFICATION_EXECUTION_RESULT_ITEM_STATUS_COMPLETED
                    )
                )
            ],
        ),
        (
            contracts.POST_VERIFICATION_EXECUTION_RESULT_STATUS_FAILED,
            [
                _sample_post_verification_result_item(
                    result_item_status=(
                        contracts.POST_VERIFICATION_EXECUTION_RESULT_ITEM_STATUS_FAILED
                    )
                )
            ],
        ),
        (
            contracts.POST_VERIFICATION_EXECUTION_RESULT_STATUS_PARTIAL,
            [
                _sample_post_verification_result_item(
                    result_item_status=(
                        contracts.POST_VERIFICATION_EXECUTION_RESULT_ITEM_STATUS_COMPLETED
                    )
                ),
                _sample_post_verification_result_item(
                    result_item_id="pve-result-2",
                    source_request_item_id=None,
                    result_item_status=(
                        contracts.POST_VERIFICATION_EXECUTION_RESULT_ITEM_STATUS_SKIPPED
                    ),
                ),
            ],
        ),
    ],
)
def test_result_status_writes_record_and_event(tmp_path, result_status, result_items):
    run_id, _, _, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(
            tmp_path,
            verification_decision=contracts.EXECUTION_VERIFICATION_ACCEPTED,
        )
    )
    if result_status != contracts.POST_VERIFICATION_EXECUTION_RESULT_STATUS_EMPTY:
        request_item = _sample_post_verification_request_item()
        pver = _post_verification_execution_request_record(
            run_id,
            result,
            verification,
            pvfp,
            request_items=[request_item],
        )
        io.atomic_write_json(
            contracts.run_post_verification_execution_request_record_json_path(
                run_id, tmp_path
            ),
            pver,
        )
        if result_items and result_items[0].get("source_request_item_id"):
            result_items = [
                _result_item_from_request(request_item, **item)
                for item in result_items
                if item.get("source_request_item_id") is not None
            ] + [
                item
                for item in result_items
                if item.get("source_request_item_id") is None
            ]
    record = _post_verification_execution_result_record(
        run_id,
        result,
        verification,
        pvfp,
        pver,
        result_items=result_items,
        result_status=result_status,
    )
    before = len(events.read_task_events(run_id, base_dir=tmp_path))
    events.record_post_verification_execution_result(
        tmp_path, run_id, record, actor="recorder"
    )
    assert io.read_json(
        contracts.run_post_verification_execution_result_record_json_path(
            run_id, tmp_path
        )
    ) == record
    rows = events.read_task_events(run_id, base_dir=tmp_path)
    assert len(rows) == before + 1
    assert (
        rows[-1]["event_type"]
        == events.EVENT_TYPE_RUN_POST_VERIFICATION_EXECUTION_RESULT_RECORDED
    )
    assert record["result_status"] == result_status


def test_record_post_verification_execution_result_event_payload_fields(tmp_path):
    run_id, _, _, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    record = _post_verification_execution_result_record(
        run_id, result, verification, pvfp, pver, executor="assistant"
    )
    events.record_post_verification_execution_result(
        tmp_path, run_id, record, actor="operator"
    )
    event = events.read_task_events(run_id, base_dir=tmp_path)[-1]
    assert "task_id" not in event
    assert "attempt_id" not in event
    payload = event["payload"]
    assert payload["run_id"] == run_id
    assert payload["executor"] == "assistant"
    assert payload["result_status"] == record["result_status"]
    assert payload["source_execution_result_fingerprint"] == record[
        "source_execution_result_fingerprint"
    ]
    assert payload["source_execution_verification_fingerprint"] == record[
        "source_execution_verification_fingerprint"
    ]
    assert payload["source_post_verification_followup_plan_fingerprint"] == record[
        "source_post_verification_followup_plan_fingerprint"
    ]
    assert payload["source_post_verification_execution_request_fingerprint"] == record[
        "source_post_verification_execution_request_fingerprint"
    ]
    assert payload["run_post_verification_execution_result_fingerprint"] == (
        contracts.run_post_verification_execution_result_fingerprint(record)
    )
    assert payload["run_post_verification_execution_result_record_path"].endswith(
        "run_post_verification_execution_result_record.json"
    )


def test_record_post_verification_execution_result_returns_written_record(tmp_path):
    run_id, _, _, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    record = _post_verification_execution_result_record(
        run_id, result, verification, pvfp, pver
    )
    returned = events.record_post_verification_execution_result(
        tmp_path, run_id, record, actor="human"
    )
    assert returned == io.read_json(
        contracts.run_post_verification_execution_result_record_json_path(
            run_id, tmp_path
        )
    )


# --- F. Preconditions ---


def test_invalid_run_id_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    record = _post_verification_execution_result_record(
        run_id, result, verification, pvfp, pver
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(ValueError):
        events.record_post_verification_execution_result(
            tmp_path, "bad", record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_invalid_schema_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    record = _post_verification_execution_result_record(
        run_id, result, verification, pvfp, pver
    )
    record["metadata"] = "bad"
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(ValueError):
        events.record_post_verification_execution_result(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_run_id_mismatch_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(ValueError, match="run_id does not match"):
        events.record_post_verification_execution_result(
            tmp_path,
            run_id,
            _post_verification_execution_result_record(
                new_run_id(), result, verification, pvfp, pver
            ),
            actor="human",
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_missing_run_manifest_fails_no_side_effects(tmp_path):
    run_id = new_run_id()
    with pytest.raises(InvalidTransition):
        events.record_post_verification_execution_result(
            tmp_path,
            run_id,
            _minimal_post_verification_execution_result(run_id=run_id),
            actor="human",
        )


def test_run_not_completed_fails_no_side_effects(tmp_path):
    run_id = new_run_id()
    io.create_run_workspace(run_id, base_dir=tmp_path)
    before = _snapshot(tmp_path, run_id, [])
    with pytest.raises(InvalidTransition, match="is not completed"):
        events.record_post_verification_execution_result(
            tmp_path,
            run_id,
            _minimal_post_verification_execution_result(run_id=run_id),
            actor="human",
        )
    assert _snapshot(tmp_path, run_id, []) == before


def test_missing_completion_record_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    contracts.run_completion_record_json_path(run_id, tmp_path).unlink()
    record = _post_verification_execution_result_record(
        run_id, result, verification, pvfp, pver
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(InvalidTransition, match="run_completion_record.json is missing"):
        events.record_post_verification_execution_result(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_missing_review_record_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    contracts.run_review_record_json_path(run_id, tmp_path).unlink()
    record = _post_verification_execution_result_record(
        run_id, result, verification, pvfp, pver
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(InvalidTransition, match="run_review_record.json is missing"):
        events.record_post_verification_execution_result(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_missing_followup_plan_record_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    contracts.run_followup_plan_record_json_path(run_id, tmp_path).unlink()
    record = _post_verification_execution_result_record(
        run_id, result, verification, pvfp, pver
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(InvalidTransition, match="run_followup_plan_record.json is missing"):
        events.record_post_verification_execution_result(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_missing_execution_request_record_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    contracts.run_execution_request_record_json_path(run_id, tmp_path).unlink()
    record = _post_verification_execution_result_record(
        run_id, result, verification, pvfp, pver
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(
        InvalidTransition, match="run_execution_request_record.json is missing"
    ):
        events.record_post_verification_execution_result(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_missing_execution_result_record_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    contracts.run_execution_result_record_json_path(run_id, tmp_path).unlink()
    record = _post_verification_execution_result_record(
        run_id, result, verification, pvfp, pver
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(
        InvalidTransition, match="run_execution_result_record.json is missing"
    ):
        events.record_post_verification_execution_result(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_missing_execution_verification_record_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    contracts.run_execution_verification_record_json_path(run_id, tmp_path).unlink()
    record = _post_verification_execution_result_record(
        run_id, result, verification, pvfp, pver
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(
        InvalidTransition, match="run_execution_verification_record.json is missing"
    ):
        events.record_post_verification_execution_result(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_missing_post_verification_followup_plan_record_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    contracts.run_post_verification_followup_plan_record_json_path(
        run_id, tmp_path
    ).unlink()
    record = _post_verification_execution_result_record(
        run_id, result, verification, pvfp, pver
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(
        InvalidTransition,
        match="run_post_verification_followup_plan_record.json is missing",
    ):
        events.record_post_verification_execution_result(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_missing_post_verification_execution_request_record_fails_no_side_effects(
    tmp_path,
):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    contracts.run_post_verification_execution_request_record_json_path(
        run_id, tmp_path
    ).unlink()
    record = _post_verification_execution_result_record(
        run_id, result, verification, pvfp, pver
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(
        InvalidTransition,
        match="run_post_verification_execution_request_record.json is missing",
    ):
        events.record_post_verification_execution_result(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_invalid_execution_result_schema_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    bad_result = dict(result)
    bad_result["metadata"] = "bad"
    io.atomic_write_json(
        contracts.run_execution_result_record_json_path(run_id, tmp_path), bad_result
    )
    record = _post_verification_execution_result_record(
        run_id, result, verification, pvfp, pver
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(ValueError, match="metadata must be a dict"):
        events.record_post_verification_execution_result(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_invalid_execution_verification_schema_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    bad_verification = dict(verification)
    bad_verification["metadata"] = "bad"
    io.atomic_write_json(
        contracts.run_execution_verification_record_json_path(run_id, tmp_path),
        bad_verification,
    )
    record = _post_verification_execution_result_record(
        run_id, result, verification, pvfp, pver
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(ValueError, match="metadata must be a dict"):
        events.record_post_verification_execution_result(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_invalid_post_verification_followup_plan_schema_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    bad_pvfp = dict(pvfp)
    bad_pvfp["metadata"] = "bad"
    io.atomic_write_json(
        contracts.run_post_verification_followup_plan_record_json_path(
            run_id, tmp_path
        ),
        bad_pvfp,
    )
    record = _post_verification_execution_result_record(
        run_id, result, verification, pvfp, pver
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(ValueError, match="metadata must be a dict"):
        events.record_post_verification_execution_result(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_invalid_post_verification_execution_request_schema_fails_no_side_effects(
    tmp_path,
):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    bad_pver = dict(pver)
    bad_pver["metadata"] = "bad"
    io.atomic_write_json(
        contracts.run_post_verification_execution_request_record_json_path(
            run_id, tmp_path
        ),
        bad_pver,
    )
    record = _post_verification_execution_result_record(
        run_id, result, verification, pvfp, pver
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(ValueError, match="metadata must be a dict"):
        events.record_post_verification_execution_result(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_source_execution_result_fingerprint_mismatch_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    record = _post_verification_execution_result_record(
        run_id, result, verification, pvfp, pver
    )
    bad = dict(record)
    bad["source_execution_result_fingerprint"] = "wrong"
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(InvalidTransition, match="source_execution_result_fingerprint"):
        events.record_post_verification_execution_result(
            tmp_path, run_id, bad, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_source_execution_verification_fingerprint_mismatch_fails_no_side_effects(
    tmp_path,
):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    record = _post_verification_execution_result_record(
        run_id, result, verification, pvfp, pver
    )
    bad = dict(record)
    bad["source_execution_verification_fingerprint"] = "wrong"
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(
        InvalidTransition, match="source_execution_verification_fingerprint"
    ):
        events.record_post_verification_execution_result(
            tmp_path, run_id, bad, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_source_post_verification_followup_plan_fingerprint_mismatch_fails_no_side_effects(
    tmp_path,
):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    record = _post_verification_execution_result_record(
        run_id, result, verification, pvfp, pver
    )
    bad = dict(record)
    bad["source_post_verification_followup_plan_fingerprint"] = "wrong"
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(
        InvalidTransition, match="source_post_verification_followup_plan_fingerprint"
    ):
        events.record_post_verification_execution_result(
            tmp_path, run_id, bad, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_source_post_verification_execution_request_fingerprint_mismatch_fails_no_side_effects(
    tmp_path,
):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    record = _post_verification_execution_result_record(
        run_id, result, verification, pvfp, pver
    )
    bad = dict(record)
    bad["source_post_verification_execution_request_fingerprint"] = "wrong"
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(
        InvalidTransition,
        match="source_post_verification_execution_request_fingerprint",
    ):
        events.record_post_verification_execution_result(
            tmp_path, run_id, bad, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_verification_record_stale_result_fingerprint_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    stale = dict(verification)
    stale["source_execution_result_fingerprint"] = "stale"
    io.atomic_write_json(
        contracts.run_execution_verification_record_json_path(run_id, tmp_path), stale
    )
    record = _post_verification_execution_result_record(
        run_id, result, verification, pvfp, pver
    )
    record["source_execution_verification_fingerprint"] = (
        contracts.run_execution_verification_fingerprint(stale)
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(
        InvalidTransition,
        match="run_execution_verification_record source_execution_result_fingerprint",
    ):
        events.record_post_verification_execution_result(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_followup_plan_stale_result_fingerprint_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    stale_pvfp = dict(pvfp)
    stale_pvfp["source_execution_result_fingerprint"] = "stale"
    io.atomic_write_json(
        contracts.run_post_verification_followup_plan_record_json_path(
            run_id, tmp_path
        ),
        stale_pvfp,
    )
    record = _post_verification_execution_result_record(
        run_id, result, verification, pvfp, pver
    )
    record["source_post_verification_followup_plan_fingerprint"] = (
        contracts.run_post_verification_followup_plan_fingerprint(stale_pvfp)
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(
        InvalidTransition,
        match="run_post_verification_followup_plan_record source_execution_result_fingerprint",
    ):
        events.record_post_verification_execution_result(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_followup_plan_stale_verification_fingerprint_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    stale_pvfp = dict(pvfp)
    stale_pvfp["source_execution_verification_fingerprint"] = "stale"
    io.atomic_write_json(
        contracts.run_post_verification_followup_plan_record_json_path(
            run_id, tmp_path
        ),
        stale_pvfp,
    )
    record = _post_verification_execution_result_record(
        run_id, result, verification, pvfp, pver
    )
    record["source_post_verification_followup_plan_fingerprint"] = (
        contracts.run_post_verification_followup_plan_fingerprint(stale_pvfp)
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(
        InvalidTransition,
        match="run_post_verification_followup_plan_record source_execution_verification_fingerprint",
    ):
        events.record_post_verification_execution_result(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_request_stale_result_fingerprint_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    stale_pver = dict(pver)
    stale_pver["source_execution_result_fingerprint"] = "stale"
    io.atomic_write_json(
        contracts.run_post_verification_execution_request_record_json_path(
            run_id, tmp_path
        ),
        stale_pver,
    )
    record = _post_verification_execution_result_record(
        run_id, result, verification, pvfp, pver
    )
    record["source_post_verification_execution_request_fingerprint"] = (
        contracts.run_post_verification_execution_request_fingerprint(stale_pver)
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(
        InvalidTransition,
        match="run_post_verification_execution_request_record source_execution_result_fingerprint",
    ):
        events.record_post_verification_execution_result(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_request_stale_verification_fingerprint_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    stale_pver = dict(pver)
    stale_pver["source_execution_verification_fingerprint"] = "stale"
    io.atomic_write_json(
        contracts.run_post_verification_execution_request_record_json_path(
            run_id, tmp_path
        ),
        stale_pver,
    )
    record = _post_verification_execution_result_record(
        run_id, result, verification, pvfp, pver
    )
    record["source_post_verification_execution_request_fingerprint"] = (
        contracts.run_post_verification_execution_request_fingerprint(stale_pver)
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(
        InvalidTransition,
        match="run_post_verification_execution_request_record source_execution_verification_fingerprint",
    ):
        events.record_post_verification_execution_result(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_request_stale_followup_plan_fingerprint_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    stale_pver = dict(pver)
    stale_pver["source_post_verification_followup_plan_fingerprint"] = "stale"
    io.atomic_write_json(
        contracts.run_post_verification_execution_request_record_json_path(
            run_id, tmp_path
        ),
        stale_pver,
    )
    record = _post_verification_execution_result_record(
        run_id, result, verification, pvfp, pver
    )
    record["source_post_verification_execution_request_fingerprint"] = (
        contracts.run_post_verification_execution_request_fingerprint(stale_pver)
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(
        InvalidTransition,
        match="run_post_verification_execution_request_record source_post_verification_followup_plan_fingerprint",
    ):
        events.record_post_verification_execution_result(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_unknown_source_request_item_id_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    result_items = [
        _result_item_from_request(
            pver["request_items"][0],
            source_request_item_id="unknown-request",
        )
    ]
    record = _post_verification_execution_result_record(
        run_id, result, verification, pvfp, pver, result_items=result_items
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(InvalidTransition, match="unknown source_request_item_id"):
        events.record_post_verification_execution_result(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("source_post_verification_followup_item_id", "wrong-pvfp", "source_post_verification_followup_item_id"),
        ("source_execution_item_id", "wrong-exec", "source_execution_item_id"),
        ("source_followup_item_id", "wrong-followup", "source_followup_item_id"),
        ("request_kind", "other_kind", "request_kind"),
        ("execution_kind", "rerun_task", "execution_kind"),
        ("item_status", "failed", "item_status"),
        ("verification_decision", "accepted", "verification_decision"),
        ("followup_kind", contracts.POST_VERIFICATION_FOLLOWUP_KIND_OTHER, "followup_kind"),
    ],
)
def test_result_item_field_mismatch_fails_no_side_effects(
    tmp_path, field, value, match
):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    result_items = [_result_item_from_request(pver["request_items"][0])]
    result_items[0][field] = value
    record = _post_verification_execution_result_record(
        run_id, result, verification, pvfp, pver, result_items=result_items
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(InvalidTransition, match=f"{match} does not match"):
        events.record_post_verification_execution_result(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


# --- G. Write order ---


def test_write_order_record_before_event(tmp_path):
    run_id, _, _, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    record = _post_verification_execution_result_record(
        run_id, result, verification, pvfp, pver
    )
    ops = []
    real_append = events.append_run_event

    def track_write(path, data):
        if str(path).endswith("run_post_verification_execution_result_record.json"):
            ops.append("write_record")
        return io.atomic_write_json(path, data)

    def track_append(r, e, base_dir=None):
        ops.append("append_event")
        return real_append(r, e, base_dir)

    with patch("htr.events.atomic_write_json", side_effect=track_write), patch(
        "htr.events.append_run_event", side_effect=track_append
    ):
        events.record_post_verification_execution_result(
            tmp_path, run_id, record, actor="human"
        )
    assert ops == ["write_record", "append_event"]


# --- H. Replay-only ---


def test_existing_post_verification_execution_result_event_id_none_raises(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    record = _post_verification_execution_result_record(
        run_id, result, verification, pvfp, pver
    )
    events.record_post_verification_execution_result(
        tmp_path, run_id, record, actor="human", event_id=new_event_id()
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(InvalidTransition):
        events.record_post_verification_execution_result(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_existing_post_verification_execution_result_missing_event_raises(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    record = _post_verification_execution_result_record(
        run_id, result, verification, pvfp, pver
    )
    io.atomic_write_json(
        contracts.run_post_verification_execution_result_record_json_path(
            run_id, tmp_path
        ),
        record,
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(InvalidTransition):
        events.record_post_verification_execution_result(
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
                    "run_post_verification_execution_result_fingerprint": "different",
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
                    "source_post_verification_execution_request_fingerprint": "different",
                },
            },
            EventConflict,
        ),
        (
            lambda e: {
                **e,
                "payload": {
                    **e["payload"],
                    "result_status": (
                        contracts.POST_VERIFICATION_EXECUTION_RESULT_STATUS_PARTIAL
                    ),
                },
            },
            EventConflict,
        ),
        (
            lambda e: {**e, "payload": {**e["payload"], "executor": "other"}},
            EventConflict,
        ),
    ],
)
def test_existing_post_verification_execution_result_replay_semantic_mismatch(
    tmp_path, mutator, expected
):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    record = _post_verification_execution_result_record(
        run_id, result, verification, pvfp, pver
    )
    event_id = new_event_id()
    events.record_post_verification_execution_result(
        tmp_path, run_id, record, actor="human", event_id=event_id
    )
    event = events.read_task_events(run_id, base_dir=tmp_path)[-1]
    bad_event = mutator(event)
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with patch("htr.events._find_run_event_by_id", return_value=bad_event):
        with pytest.raises(expected):
            events.record_post_verification_execution_result(
                tmp_path, run_id, record, actor="human", event_id=event_id
            )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_existing_post_verification_execution_result_same_event_id_same_semantic_returns_existing(
    tmp_path,
):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    record = _post_verification_execution_result_record(
        run_id, result, verification, pvfp, pver
    )
    event_id = new_event_id()
    events.record_post_verification_execution_result(
        tmp_path, run_id, record, actor="human", event_id=event_id
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    returned = events.record_post_verification_execution_result(
        tmp_path, run_id, record, actor="human", event_id=event_id
    )
    assert returned == record
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_replay_only_no_writes_or_previous_record_mutation(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    record = _post_verification_execution_result_record(
        run_id, result, verification, pvfp, pver
    )
    event_id = new_event_id()
    events.record_post_verification_execution_result(
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
    pver_before = contracts.run_post_verification_execution_request_record_json_path(
        run_id, tmp_path
    ).read_bytes()
    events.record_post_verification_execution_result(
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
    assert (
        contracts.run_post_verification_execution_request_record_json_path(
            run_id, tmp_path
        ).read_bytes()
        == pver_before
    )
    assert len(list(paths.tasks_dir(run_id, tmp_path).iterdir())) == len(task_ids)


# --- I. Normal idempotency ---


def test_idempotent_same_event_id_same_semantic(tmp_path):
    run_id, _, _, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    record = _post_verification_execution_result_record(
        run_id, result, verification, pvfp, pver
    )
    event_id = new_event_id()
    first = events.record_post_verification_execution_result(
        tmp_path, run_id, record, actor="human", event_id=event_id
    )
    second = events.record_post_verification_execution_result(
        tmp_path, run_id, record, actor="human", event_id=event_id
    )
    assert second == first


def test_same_event_id_different_semantic_raises_conflict(tmp_path):
    run_id, _, _, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    event_id = new_event_id()
    first = _post_verification_execution_result_record(
        run_id, result, verification, pvfp, pver
    )
    events.record_post_verification_execution_result(
        tmp_path, run_id, first, actor="human", event_id=event_id
    )
    second = _post_verification_execution_result_record(
        run_id, result, verification, pvfp, pver, notes="different semantic"
    )
    with pytest.raises(EventConflict):
        events.record_post_verification_execution_result(
            tmp_path, run_id, second, actor="human", event_id=event_id
        )


def test_idempotency_requires_post_verification_execution_request(tmp_path):
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
    stub_pvfp = contracts.make_run_post_verification_followup_plan_record(
        run_id=run_id,
        source_execution_result_fingerprint=contracts.run_execution_result_fingerprint(
            stub_result
        ),
        source_execution_verification_fingerprint=contracts.run_execution_verification_fingerprint(
            stub_verification
        ),
        followup_items=[],
    )
    io.atomic_write_json(
        contracts.run_post_verification_followup_plan_record_json_path(
            run_id, tmp_path
        ),
        stub_pvfp,
    )
    stub_result_record = _minimal_post_verification_execution_result(run_id=run_id)
    with pytest.raises(
        InvalidTransition,
        match="run_post_verification_execution_request_record.json is missing",
    ):
        events.record_post_verification_execution_result(
            tmp_path,
            run_id,
            stub_result_record,
            actor="human",
            event_id=new_event_id(),
        )


def test_idempotency_does_not_allow_non_completed_run(tmp_path):
    run_id = new_run_id()
    io.create_run_workspace(run_id, base_dir=tmp_path)
    with pytest.raises(InvalidTransition):
        events.record_post_verification_execution_result(
            tmp_path,
            run_id,
            _minimal_post_verification_execution_result(run_id=run_id),
            actor="human",
            event_id=new_event_id(),
        )


# --- J. Boundary preservation ---


def test_boundaries_preserved_on_success(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path)
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    record = _post_verification_execution_result_record(
        run_id, result, verification, pvfp, pver
    )
    doc_path = tmp_path / "docs" / "readme.md"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("unchanged", encoding="utf-8")
    events.record_post_verification_execution_result(
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
    assert after["post_verification_execution_request"] == before[
        "post_verification_execution_request"
    ]
    assert after["task_statuses"] == before["task_statuses"]
    assert after["attempt_statuses"] == before["attempt_statuses"]
    assert after["post_verification_execution_result_exists"] is True
    assert len(list(paths.tasks_dir(run_id, tmp_path).iterdir())) == len(task_ids)
    assert doc_path.read_text(encoding="utf-8") == "unchanged"
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


def test_task14_modules_do_not_import_forbidden_modules():
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


def test_package_exports_task14_apis():
    import htr

    for name in (
        "make_run_post_verification_execution_result_record",
        "run_post_verification_execution_result_fingerprint",
        "validate_post_verification_execution_result_items_correspond",
        "compute_post_verification_execution_result_status",
        "record_post_verification_execution_result",
        "EVENT_TYPE_RUN_POST_VERIFICATION_EXECUTION_RESULT_RECORDED",
        "POST_VERIFICATION_EXECUTION_RESULT_STATUS_COMPLETED",
        "POST_VERIFICATION_EXECUTION_RESULT_STATUS_FAILED",
        "POST_VERIFICATION_EXECUTION_RESULT_STATUS_PARTIAL",
        "POST_VERIFICATION_EXECUTION_RESULT_STATUS_EMPTY",
        "POST_VERIFICATION_EXECUTION_RESULT_ITEM_STATUS_COMPLETED",
        "POST_VERIFICATION_EXECUTION_RESULT_ITEM_STATUS_FAILED",
        "POST_VERIFICATION_EXECUTION_RESULT_ITEM_STATUS_SKIPPED",
        "POST_VERIFICATION_EXECUTION_RESULT_ITEM_STATUS_NOT_APPLICABLE",
    ):
        assert hasattr(htr, name)
