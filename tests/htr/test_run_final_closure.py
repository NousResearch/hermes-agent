"""Tests for Task 16 — Run Final Closure Record."""

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
    RunFinalizedError,
    RunSealBlockedError,
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
        "source_run_execution_result_fingerprint",
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


def _run_with_post_verification_execution_result(tmp_path, execution_items=None):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver = (
        _run_with_post_verification_execution_request(tmp_path, execution_items)
    )
    pve_result = _post_verification_execution_result_record(
        run_id, result, verification, pvfp, pver
    )
    events.record_post_verification_execution_result(
        tmp_path, run_id, pve_result, actor="human"
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
        pve_result,
    )


def _sample_post_verification_verification_item(**overrides):
    item = {
        "verification_item_id": "pve-verify-1",
        "source_result_item_id": "pve-result-1",
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
        "verification_decision_after_result": (
            contracts.POST_VERIFICATION_EXECUTION_VERIFICATION_ITEM_DECISION_VERIFIED
        ),
        "reason": None,
        "evidence": {},
        "metadata": {},
    }
    item.update(overrides)
    return item


def _verification_item_from_result(result_item, decision=None, **overrides):
    item = _sample_post_verification_verification_item(
        verification_item_id=f"verify-{result_item['result_item_id']}",
        source_result_item_id=result_item["result_item_id"],
        source_request_item_id=result_item.get("source_request_item_id"),
        source_post_verification_followup_item_id=result_item.get(
            "source_post_verification_followup_item_id"
        ),
        source_execution_item_id=result_item.get("source_execution_item_id"),
        source_followup_item_id=result_item.get("source_followup_item_id"),
        request_kind=result_item.get("request_kind"),
        execution_kind=result_item.get("execution_kind"),
        item_status=result_item.get("item_status"),
        verification_decision=result_item.get("verification_decision"),
        followup_kind=result_item.get("followup_kind"),
        result_item_status=result_item.get("result_item_status"),
        verification_decision_after_result=decision
        or contracts.POST_VERIFICATION_EXECUTION_VERIFICATION_ITEM_DECISION_VERIFIED,
    )
    item.update(overrides)
    return item


def _post_verification_execution_verification_record(
    run_id,
    result_record,
    verification_record,
    post_verification_plan_record,
    post_verification_execution_request_record,
    post_verification_execution_result_record,
    **kwargs,
):
    verification_items = kwargs.pop("verification_items", None)
    if verification_items is None:
        result_items = post_verification_execution_result_record["result_items"]
        verification_items = (
            [_verification_item_from_result(result_items[0])] if result_items else []
        )
    return contracts.make_run_post_verification_execution_verification_record(
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
        source_post_verification_execution_result_fingerprint=contracts.run_post_verification_execution_result_fingerprint(
            post_verification_execution_result_record
        ),
        verification_items=verification_items,
        **kwargs,
    )


def _minimal_post_verification_execution_verification(**overrides):
    run_id = overrides.pop("run_id", new_run_id())
    verification_items = overrides.pop("verification_items", [])
    verification_status = overrides.pop("verification_status", None)
    if verification_status is None:
        verification_status = (
            contracts.compute_post_verification_execution_verification_status(
                verification_items
            )
        )
    record = {
        "schema_version": 1,
        "run_id": run_id,
        "source_run_execution_result_fingerprint": "fp-result",
        "source_run_execution_verification_fingerprint": "fp-verification",
        "source_post_verification_followup_plan_fingerprint": "fp-pvfp",
        "source_post_verification_execution_request_fingerprint": "fp-pver",
        "source_post_verification_execution_result_fingerprint": "fp-pve-result",
        "verifier": "human",
        "verification_status": verification_status,
        "verification_items": verification_items,
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
        "post_verification_execution_verification": (
            contracts.run_post_verification_execution_verification_record_json_path(
                run_id, tmp_path
            ).read_bytes()
            if contracts.run_post_verification_execution_verification_record_json_path(
                run_id, tmp_path
            ).exists()
            else None
        ),
        "post_verification_execution_verification_exists": (
            contracts.run_post_verification_execution_verification_record_json_path(
                run_id, tmp_path
            ).exists()
        ),
        "final_closure": (
            contracts.run_final_closure_record_json_path(run_id, tmp_path).read_bytes()
            if contracts.run_final_closure_record_json_path(run_id, tmp_path).exists()
            else None
        ),
        "final_closure_exists": contracts.run_final_closure_record_json_path(
            run_id, tmp_path
        ).exists(),
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

def _run_with_post_verification_execution_verification(tmp_path, execution_items=None):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result = (
        _run_with_post_verification_execution_result(tmp_path, execution_items)
    )
    pve_verification = _post_verification_execution_verification_record(
        run_id, result, verification, pvfp, pver, pve_result
    )
    events.record_post_verification_execution_verification(
        tmp_path, run_id, pve_verification, actor="human"
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
        pve_result,
        pve_verification,
    )


def _sample_closure_item(**overrides):
    item = {
        "closure_item_id": "closure-1",
        "source_post_verification_execution_verification_item_id": "pve-verify-1",
        "source_post_verification_execution_result_item_id": "pve-result-1",
        "source_post_verification_execution_request_item_id": "pver-pvfp-exec-1",
        "source_post_verification_followup_item_id": "pvfp-exec-1",
        "source_execution_item_id": "exec-1",
        "source_followup_item_id": "followup-1",
        "verification_decision_after_result": (
            contracts.POST_VERIFICATION_EXECUTION_VERIFICATION_ITEM_DECISION_VERIFIED
        ),
        "closure_decision": contracts.RUN_FINAL_CLOSURE_ITEM_DECISION_ACCEPTED,
        "reason": None,
        "evidence": {},
        "metadata": {},
    }
    item.update(overrides)
    return item


def _closure_item_from_verification_item(verification_item, decision=None, **overrides):
    item = _sample_closure_item(
        closure_item_id=f"closure-{verification_item['verification_item_id']}",
        source_post_verification_execution_verification_item_id=verification_item[
            "verification_item_id"
        ],
        source_post_verification_execution_result_item_id=verification_item.get(
            "source_result_item_id"
        ),
        source_post_verification_execution_request_item_id=verification_item.get(
            "source_request_item_id"
        ),
        source_post_verification_followup_item_id=verification_item.get(
            "source_post_verification_followup_item_id"
        ),
        source_execution_item_id=verification_item.get("source_execution_item_id"),
        source_followup_item_id=verification_item.get("source_followup_item_id"),
        verification_decision_after_result=verification_item.get(
            "verification_decision_after_result"
        ),
        closure_decision=decision or contracts.RUN_FINAL_CLOSURE_ITEM_DECISION_ACCEPTED,
    )
    item.update(overrides)
    return item


def _run_final_closure_record(
    run_id,
    completion,
    review,
    plan,
    request,
    result,
    verification,
    pvfp,
    pver,
    pve_result,
    pve_verification,
    **kwargs,
):
    closure_items = kwargs.pop("closure_items", None)
    if closure_items is None:
        verification_items = pve_verification["verification_items"]
        closure_items = (
            [_closure_item_from_verification_item(verification_items[0])]
            if verification_items
            else []
        )
    return contracts.make_run_final_closure_record(
        run_id=run_id,
        source_run_completion_fingerprint=contracts.run_completion_fingerprint(completion),
        source_run_review_fingerprint=contracts.run_review_fingerprint(review),
        source_run_followup_plan_fingerprint=contracts.run_followup_plan_fingerprint(plan),
        source_run_execution_request_fingerprint=contracts.run_execution_request_fingerprint(
            request
        ),
        source_run_execution_result_fingerprint=contracts.run_execution_result_fingerprint(
            result
        ),
        source_run_execution_verification_fingerprint=contracts.run_execution_verification_fingerprint(
            verification
        ),
        source_post_verification_followup_plan_fingerprint=contracts.run_post_verification_followup_plan_fingerprint(
            pvfp
        ),
        source_post_verification_execution_request_fingerprint=contracts.run_post_verification_execution_request_fingerprint(
            pver
        ),
        source_post_verification_execution_result_fingerprint=contracts.run_post_verification_execution_result_fingerprint(
            pve_result
        ),
        source_post_verification_execution_verification_fingerprint=contracts.run_post_verification_execution_verification_fingerprint(
            pve_verification
        ),
        closure_reason=kwargs.pop("closure_reason", "Manual final closure"),
        closure_items=closure_items,
        **kwargs,
    )


def _minimal_run_final_closure(**overrides):
    run_id = overrides.pop("run_id", new_run_id())
    closure_items = overrides.pop("closure_items", [])
    final_closure_status = overrides.pop("final_closure_status", None)
    if final_closure_status is None:
        final_closure_status = contracts.compute_run_final_closure_status(closure_items)
    record = {
        "schema_version": 1,
        "run_id": run_id,
        "source_run_completion_fingerprint": "fp-completion",
        "source_run_review_fingerprint": "fp-review",
        "source_run_followup_plan_fingerprint": "fp-followup-plan",
        "source_run_execution_request_fingerprint": "fp-exec-request",
        "source_run_execution_result_fingerprint": "fp-exec-result",
        "source_run_execution_verification_fingerprint": "fp-exec-verification",
        "source_post_verification_followup_plan_fingerprint": "fp-pvfp",
        "source_post_verification_execution_request_fingerprint": "fp-pver",
        "source_post_verification_execution_result_fingerprint": "fp-pve-result",
        "source_post_verification_execution_verification_fingerprint": "fp-pve-verification",
        "closer": "human",
        "final_closure_status": final_closure_status,
        "closure_reason": "done",
        "closure_items": closure_items,
        "notes": None,
        "metadata": {},
        "created_at": "2026-07-19T08:00:00+00:00",
    }
    record.update(overrides)
    return record

# --- A. Schema ---


def test_valid_run_final_closure_record_passes():
    record = _minimal_run_final_closure(
        closure_items=[_sample_closure_item()],
        final_closure_status=contracts.RUN_FINAL_CLOSURE_STATUS_CLOSED_VERIFIED,
    )
    schemas.validate(record, "run_final_closure_record")


@pytest.mark.parametrize(
    "missing",
    [
        "run_id",
        "source_run_execution_result_fingerprint",
        "source_run_execution_verification_fingerprint",
        "source_post_verification_followup_plan_fingerprint",
        "source_post_verification_execution_request_fingerprint",
        "source_post_verification_execution_result_fingerprint",
        "closer",
        "final_closure_status",
        "closure_items",
        "notes",
        "metadata",
    ],
)
def test_missing_required_fields_fail(missing):
    record = _minimal_run_final_closure()
    del record[missing]
    with pytest.raises(ValueError, match="missing fields"):
        schemas.validate(record, "run_final_closure_record")


def test_invalid_run_id_fails_schema():
    record = _minimal_run_final_closure(run_id="bad")
    with pytest.raises(ValueError, match="run_id must be a valid run id"):
        schemas.validate(record, "run_final_closure_record")


def test_invalid_source_execution_result_fingerprint_fails():
    with pytest.raises(ValueError, match="source_run_execution_result_fingerprint"):
        contracts.make_run_final_closure_record(
            run_id=new_run_id(),
            source_run_completion_fingerprint="fp-completion",
            source_run_review_fingerprint="fp-review",
            source_run_followup_plan_fingerprint="fp-followup-plan",
            source_run_execution_request_fingerprint="fp-exec-request",
            source_run_execution_result_fingerprint="",
            source_run_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            source_post_verification_execution_request_fingerprint="fp-pver",
            source_post_verification_execution_result_fingerprint="fp-pve-result",
            source_post_verification_execution_verification_fingerprint="fp-pve-verification",
            closure_reason="done",
            closure_items=[],
        )


def test_invalid_source_execution_verification_fingerprint_fails():
    with pytest.raises(ValueError, match="source_run_execution_verification_fingerprint"):
        contracts.make_run_final_closure_record(
            run_id=new_run_id(),
            source_run_completion_fingerprint="fp-completion",
            source_run_review_fingerprint="fp-review",
            source_run_followup_plan_fingerprint="fp-followup-plan",
            source_run_execution_request_fingerprint="fp-exec-request",
            source_run_execution_result_fingerprint="fp-result",
            source_run_execution_verification_fingerprint="",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            source_post_verification_execution_request_fingerprint="fp-pver",
            source_post_verification_execution_result_fingerprint="fp-pve-result",
            source_post_verification_execution_verification_fingerprint="fp-pve-verification",
            closure_reason="done",
            closure_items=[],
        )


def test_invalid_source_post_verification_followup_plan_fingerprint_fails():
    with pytest.raises(
        ValueError, match="source_post_verification_followup_plan_fingerprint"
    ):
        contracts.make_run_final_closure_record(
            run_id=new_run_id(),
            source_run_completion_fingerprint="fp-completion",
            source_run_review_fingerprint="fp-review",
            source_run_followup_plan_fingerprint="fp-followup-plan",
            source_run_execution_request_fingerprint="fp-exec-request",
            source_run_execution_result_fingerprint="fp-result",
            source_run_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="",
            source_post_verification_execution_request_fingerprint="fp-pver",
            source_post_verification_execution_result_fingerprint="fp-pve-result",
            source_post_verification_execution_verification_fingerprint="fp-pve-verification",
            closure_reason="done",
            closure_items=[],
        )


def test_invalid_source_post_verification_execution_request_fingerprint_fails():
    with pytest.raises(
        ValueError, match="source_post_verification_execution_request_fingerprint"
    ):
        contracts.make_run_final_closure_record(
            run_id=new_run_id(),
            source_run_completion_fingerprint="fp-completion",
            source_run_review_fingerprint="fp-review",
            source_run_followup_plan_fingerprint="fp-followup-plan",
            source_run_execution_request_fingerprint="fp-exec-request",
            source_run_execution_result_fingerprint="fp-result",
            source_run_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            source_post_verification_execution_request_fingerprint="",
            source_post_verification_execution_result_fingerprint="fp-pve-result",
            source_post_verification_execution_verification_fingerprint="fp-pve-verification",
            closure_reason="done",
            closure_items=[],
        )


def test_invalid_source_post_verification_execution_result_fingerprint_fails():
    with pytest.raises(
        ValueError, match="source_post_verification_execution_result_fingerprint"
    ):
        contracts.make_run_final_closure_record(
            run_id=new_run_id(),
            source_run_completion_fingerprint="fp-completion",
            source_run_review_fingerprint="fp-review",
            source_run_followup_plan_fingerprint="fp-followup-plan",
            source_run_execution_request_fingerprint="fp-exec-request",
            source_run_execution_result_fingerprint="fp-result",
            source_run_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            source_post_verification_execution_request_fingerprint="fp-pver",
            source_post_verification_execution_result_fingerprint="",
            source_post_verification_execution_verification_fingerprint="fp-pve-verification",
            closure_reason="done",
            closure_items=[],
        )


def test_invalid_closer_fails():
    with pytest.raises(ValueError, match="closer must be a non-empty string"):
        contracts.make_run_final_closure_record(
            run_id=new_run_id(),
            source_run_completion_fingerprint="fp-completion",
            source_run_review_fingerprint="fp-review",
            source_run_followup_plan_fingerprint="fp-followup-plan",
            source_run_execution_request_fingerprint="fp-exec-request",
            source_run_execution_result_fingerprint="fp-result",
            source_run_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            source_post_verification_execution_request_fingerprint="fp-pver",
            source_post_verification_execution_result_fingerprint="fp-pve-result",
            source_post_verification_execution_verification_fingerprint="fp-pve-verification",
            closure_reason="done",
            closure_items=[],
            closer="",
        )


def test_invalid_final_closure_status_fails():
    with pytest.raises(ValueError, match="final_closure_status must be one of"):
        contracts.make_run_final_closure_record(
            run_id=new_run_id(),
            source_run_completion_fingerprint="fp-completion",
            source_run_review_fingerprint="fp-review",
            source_run_followup_plan_fingerprint="fp-followup-plan",
            source_run_execution_request_fingerprint="fp-exec-request",
            source_run_execution_result_fingerprint="fp-result",
            source_run_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            source_post_verification_execution_request_fingerprint="fp-pver",
            source_post_verification_execution_result_fingerprint="fp-pve-result",
            source_post_verification_execution_verification_fingerprint="fp-pve-verification",
            closure_reason="done",
            closure_items=[],
            final_closure_status="draft",
        )


def test_closure_items_must_be_list():
    record = _minimal_run_final_closure()
    record["closure_items"] = {}
    with pytest.raises(ValueError, match="closure_items must be a list"):
        schemas.validate(record, "run_final_closure_record")


def test_closure_items_entries_must_be_dicts():
    record = _minimal_run_final_closure()
    record["closure_items"] = ["bad"]
    with pytest.raises(ValueError, match="closure_items entries must be dicts"):
        schemas.validate(record, "run_final_closure_record")


def test_missing_closure_item_id_fails():
    with pytest.raises(KeyError, match="closure_item_id"):
        contracts.make_run_final_closure_record(
            run_id=new_run_id(),
            source_run_completion_fingerprint="fp-completion",
            source_run_review_fingerprint="fp-review",
            source_run_followup_plan_fingerprint="fp-followup-plan",
            source_run_execution_request_fingerprint="fp-exec-request",
            source_run_execution_result_fingerprint="fp-result",
            source_run_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            source_post_verification_execution_request_fingerprint="fp-pver",
            source_post_verification_execution_result_fingerprint="fp-pve-result",
            source_post_verification_execution_verification_fingerprint="fp-pve-verification",
            closure_reason="done",
            closure_items=[
                {
                    "source_request_item_id": None,
                    "verification_decision_after_result": (
                        contracts.RUN_FINAL_CLOSURE_ITEM_DECISION_ACCEPTED
                    ),
                    "evidence": {},
                    "metadata": {},
                }
            ],
        )


def test_invalid_verification_decision_after_result_fails():
    with pytest.raises(ValueError, match="verification_decision_after_result"):
        contracts.make_run_final_closure_record(
            run_id=new_run_id(),
            source_run_completion_fingerprint="fp-completion",
            source_run_review_fingerprint="fp-review",
            source_run_followup_plan_fingerprint="fp-followup-plan",
            source_run_execution_request_fingerprint="fp-exec-request",
            source_run_execution_result_fingerprint="fp-result",
            source_run_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            source_post_verification_execution_request_fingerprint="fp-pver",
            source_post_verification_execution_result_fingerprint="fp-pve-result",
            source_post_verification_execution_verification_fingerprint="fp-pve-verification",
            closure_reason="done",
            closure_items=[
                _sample_closure_item(verification_decision_after_result=123)
            ],
        )


def test_source_post_verification_execution_result_item_id_must_be_string_or_none():
    with pytest.raises(ValueError, match="source_post_verification_execution_result_item_id"):
        contracts.make_run_final_closure_record(
            run_id=new_run_id(),
            source_run_completion_fingerprint="fp-completion",
            source_run_review_fingerprint="fp-review",
            source_run_followup_plan_fingerprint="fp-followup-plan",
            source_run_execution_request_fingerprint="fp-exec-request",
            source_run_execution_result_fingerprint="fp-result",
            source_run_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            source_post_verification_execution_request_fingerprint="fp-pver",
            source_post_verification_execution_result_fingerprint="fp-pve-result",
            source_post_verification_execution_verification_fingerprint="fp-pve-verification",
            closure_reason="done",
            closure_items=[
                _sample_closure_item(source_post_verification_execution_result_item_id=123)
            ],
        )


def test_source_post_verification_followup_item_id_must_be_string_or_none():
    with pytest.raises(ValueError, match="source_post_verification_followup_item_id"):
        contracts.make_run_final_closure_record(
            run_id=new_run_id(),
            source_run_completion_fingerprint="fp-completion",
            source_run_review_fingerprint="fp-review",
            source_run_followup_plan_fingerprint="fp-followup-plan",
            source_run_execution_request_fingerprint="fp-exec-request",
            source_run_execution_result_fingerprint="fp-result",
            source_run_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            source_post_verification_execution_request_fingerprint="fp-pver",
            source_post_verification_execution_result_fingerprint="fp-pve-result",
            source_post_verification_execution_verification_fingerprint="fp-pve-verification",
            closure_reason="done",
            closure_items=[
                _sample_closure_item(
                    source_post_verification_followup_item_id=123
                )
            ],
        )


def test_source_execution_item_id_must_be_string_or_none():
    with pytest.raises(ValueError, match="source_execution_item_id"):
        contracts.make_run_final_closure_record(
            run_id=new_run_id(),
            source_run_completion_fingerprint="fp-completion",
            source_run_review_fingerprint="fp-review",
            source_run_followup_plan_fingerprint="fp-followup-plan",
            source_run_execution_request_fingerprint="fp-exec-request",
            source_run_execution_result_fingerprint="fp-result",
            source_run_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            source_post_verification_execution_request_fingerprint="fp-pver",
            source_post_verification_execution_result_fingerprint="fp-pve-result",
            source_post_verification_execution_verification_fingerprint="fp-pve-verification",
            closure_reason="done",
            closure_items=[_sample_closure_item(source_execution_item_id=123)],
        )


def test_source_followup_item_id_must_be_string_or_none():
    with pytest.raises(ValueError, match="source_followup_item_id"):
        contracts.make_run_final_closure_record(
            run_id=new_run_id(),
            source_run_completion_fingerprint="fp-completion",
            source_run_review_fingerprint="fp-review",
            source_run_followup_plan_fingerprint="fp-followup-plan",
            source_run_execution_request_fingerprint="fp-exec-request",
            source_run_execution_result_fingerprint="fp-result",
            source_run_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            source_post_verification_execution_request_fingerprint="fp-pver",
            source_post_verification_execution_result_fingerprint="fp-pve-result",
            source_post_verification_execution_verification_fingerprint="fp-pve-verification",
            closure_reason="done",
            closure_items=[_sample_closure_item(source_followup_item_id=123)],
        )





def test_invalid_closure_decision_fails():
    with pytest.raises(ValueError, match="closure_decision must be one of"):
        contracts.make_run_final_closure_record(
            run_id=new_run_id(),
            source_run_completion_fingerprint="fp-completion",
            source_run_review_fingerprint="fp-review",
            source_run_followup_plan_fingerprint="fp-followup-plan",
            source_run_execution_request_fingerprint="fp-exec-request",
            source_run_execution_result_fingerprint="fp-result",
            source_run_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            source_post_verification_execution_request_fingerprint="fp-pver",
            source_post_verification_execution_result_fingerprint="fp-pve-result",
            source_post_verification_execution_verification_fingerprint="fp-pve-verification",
            closure_reason="done",
            closure_items=[_sample_closure_item(closure_decision="draft")],
        )


def test_invalid_closure_reason_fails():
    with pytest.raises(ValueError, match="closure_reason"):
        contracts.make_run_final_closure_record(
            run_id=new_run_id(),
            source_run_completion_fingerprint="fp-completion",
            source_run_review_fingerprint="fp-review",
            source_run_followup_plan_fingerprint="fp-followup-plan",
            source_run_execution_request_fingerprint="fp-exec-request",
            source_run_execution_result_fingerprint="fp-result",
            source_run_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            source_post_verification_execution_request_fingerprint="fp-pver",
            source_post_verification_execution_result_fingerprint="fp-pve-result",
            source_post_verification_execution_verification_fingerprint="fp-pve-verification",
            closure_reason="",
            closure_items=[],
        )


def test_reason_must_be_string_or_none():
    with pytest.raises(ValueError, match="reason"):
        contracts.make_run_final_closure_record(
            run_id=new_run_id(),
            source_run_completion_fingerprint="fp-completion",
            source_run_review_fingerprint="fp-review",
            source_run_followup_plan_fingerprint="fp-followup-plan",
            source_run_execution_request_fingerprint="fp-exec-request",
            source_run_execution_result_fingerprint="fp-result",
            source_run_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            source_post_verification_execution_request_fingerprint="fp-pver",
            source_post_verification_execution_result_fingerprint="fp-pve-result",
            source_post_verification_execution_verification_fingerprint="fp-pve-verification",
            closure_reason="done",
            closure_items=[_sample_closure_item(reason=123)],
        )


def test_evidence_must_be_dict():
    with pytest.raises(ValueError, match="evidence"):
        contracts.make_run_final_closure_record(
            run_id=new_run_id(),
            source_run_completion_fingerprint="fp-completion",
            source_run_review_fingerprint="fp-review",
            source_run_followup_plan_fingerprint="fp-followup-plan",
            source_run_execution_request_fingerprint="fp-exec-request",
            source_run_execution_result_fingerprint="fp-result",
            source_run_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            source_post_verification_execution_request_fingerprint="fp-pver",
            source_post_verification_execution_result_fingerprint="fp-pve-result",
            source_post_verification_execution_verification_fingerprint="fp-pve-verification",
            closure_reason="done",
            closure_items=[_sample_closure_item(evidence="bad")],
        )


def test_item_metadata_must_be_dict():
    with pytest.raises(ValueError, match="metadata"):
        contracts.make_run_final_closure_record(
            run_id=new_run_id(),
            source_run_completion_fingerprint="fp-completion",
            source_run_review_fingerprint="fp-review",
            source_run_followup_plan_fingerprint="fp-followup-plan",
            source_run_execution_request_fingerprint="fp-exec-request",
            source_run_execution_result_fingerprint="fp-result",
            source_run_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            source_post_verification_execution_request_fingerprint="fp-pver",
            source_post_verification_execution_result_fingerprint="fp-pve-result",
            source_post_verification_execution_verification_fingerprint="fp-pve-verification",
            closure_reason="done",
            closure_items=[_sample_closure_item(metadata="bad")],
        )


def test_notes_must_be_string_or_none():
    record = _minimal_run_final_closure()
    record["notes"] = 1
    with pytest.raises(ValueError, match="notes must be a string or null"):
        schemas.validate(record, "run_final_closure_record")


def test_metadata_must_be_dict():
    with pytest.raises(ValueError, match="metadata must be a dict"):
        contracts.make_run_final_closure_record(
            run_id=new_run_id(),
            source_run_completion_fingerprint="fp-completion",
            source_run_review_fingerprint="fp-review",
            source_run_followup_plan_fingerprint="fp-followup-plan",
            source_run_execution_request_fingerprint="fp-exec-request",
            source_run_execution_result_fingerprint="fp-result",
            source_run_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            source_post_verification_execution_request_fingerprint="fp-pver",
            source_post_verification_execution_result_fingerprint="fp-pve-result",
            source_post_verification_execution_verification_fingerprint="fp-pve-verification",
            closure_reason="done",
            closure_items=[],
            metadata=[],
        )


def test_closed_no_action_requires_empty_closure_items():
    with pytest.raises(ValueError, match="closed_no_action final_closure_status requires closure_items to be empty"):
        contracts.make_run_final_closure_record(
            run_id=new_run_id(),
            source_run_completion_fingerprint="fp-completion",
            source_run_review_fingerprint="fp-review",
            source_run_followup_plan_fingerprint="fp-followup-plan",
            source_run_execution_request_fingerprint="fp-exec-request",
            source_run_execution_result_fingerprint="fp-result",
            source_run_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            source_post_verification_execution_request_fingerprint="fp-pver",
            source_post_verification_execution_result_fingerprint="fp-pve-result",
            source_post_verification_execution_verification_fingerprint="fp-pve-verification",
            closure_reason="done",
            closure_items=[_sample_closure_item()],
            final_closure_status=contracts.RUN_FINAL_CLOSURE_STATUS_CLOSED_NO_ACTION,
        )


def test_closed_verified_requires_accepted_items():
    with pytest.raises(ValueError, match="closed_verified final_closure_status requires"):
        contracts.make_run_final_closure_record(
            run_id=new_run_id(),
            source_run_completion_fingerprint="fp-completion",
            source_run_review_fingerprint="fp-review",
            source_run_followup_plan_fingerprint="fp-followup-plan",
            source_run_execution_request_fingerprint="fp-exec-request",
            source_run_execution_result_fingerprint="fp-result",
            source_run_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            source_post_verification_execution_request_fingerprint="fp-pver",
            source_post_verification_execution_result_fingerprint="fp-pve-result",
            source_post_verification_execution_verification_fingerprint="fp-pve-verification",
            closure_reason="done",
            closure_items=[
                _sample_closure_item(
                    closure_decision=contracts.RUN_FINAL_CLOSURE_ITEM_DECISION_REJECTED
                )
            ],
            final_closure_status=contracts.RUN_FINAL_CLOSURE_STATUS_CLOSED_VERIFIED,
        )


def test_closed_rejected_requires_rejected_items():
    with pytest.raises(ValueError, match="closed_rejected final_closure_status requires"):
        contracts.make_run_final_closure_record(
            run_id=new_run_id(),
            source_run_completion_fingerprint="fp-completion",
            source_run_review_fingerprint="fp-review",
            source_run_followup_plan_fingerprint="fp-followup-plan",
            source_run_execution_request_fingerprint="fp-exec-request",
            source_run_execution_result_fingerprint="fp-result",
            source_run_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            source_post_verification_execution_request_fingerprint="fp-pver",
            source_post_verification_execution_result_fingerprint="fp-pve-result",
            source_post_verification_execution_verification_fingerprint="fp-pve-verification",
            closure_reason="done",
            closure_items=[
                _sample_closure_item(
                    closure_decision=contracts.RUN_FINAL_CLOSURE_ITEM_DECISION_ACCEPTED
                )
            ],
            final_closure_status=contracts.RUN_FINAL_CLOSURE_STATUS_CLOSED_REJECTED,
        )


def test_closed_needs_more_work_requires_mixed_or_needs_more_work_items():
    with pytest.raises(ValueError, match="closed_needs_more_work"):
        contracts.make_run_final_closure_record(
            run_id=new_run_id(),
            source_run_completion_fingerprint="fp-completion",
            source_run_review_fingerprint="fp-review",
            source_run_followup_plan_fingerprint="fp-followup-plan",
            source_run_execution_request_fingerprint="fp-exec-request",
            source_run_execution_result_fingerprint="fp-result",
            source_run_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            source_post_verification_execution_request_fingerprint="fp-pver",
            source_post_verification_execution_result_fingerprint="fp-pve-result",
            source_post_verification_execution_verification_fingerprint="fp-pve-verification",
            closure_reason="done",
            closure_items=[
                _sample_closure_item(
                    closure_decision=contracts.RUN_FINAL_CLOSURE_ITEM_DECISION_ACCEPTED
                )
            ],
            final_closure_status=contracts.RUN_FINAL_CLOSURE_STATUS_CLOSED_NEEDS_MORE_WORK,
        )


def test_all_no_action_closure_items_invalid():
    with pytest.raises(ValueError, match="closure_items cannot all be no_action"):
        contracts.make_run_final_closure_record(
            run_id=new_run_id(),
            source_run_completion_fingerprint="fp-completion",
            source_run_review_fingerprint="fp-review",
            source_run_followup_plan_fingerprint="fp-followup-plan",
            source_run_execution_request_fingerprint="fp-exec-request",
            source_run_execution_result_fingerprint="fp-result",
            source_run_execution_verification_fingerprint="fp-verification",
            source_post_verification_followup_plan_fingerprint="fp-pvfp",
            source_post_verification_execution_request_fingerprint="fp-pver",
            source_post_verification_execution_result_fingerprint="fp-pve-result",
            source_post_verification_execution_verification_fingerprint="fp-pve-verification",
            closure_reason="done",
            closure_items=[
                _sample_closure_item(
                    closure_decision=contracts.RUN_FINAL_CLOSURE_ITEM_DECISION_NO_ACTION
                )
            ],
        )


# --- B. Factory ---


def test_make_run_final_closure_record_returns_valid_schema(
    tmp_path,
):
    run_id, _, _, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    schemas.validate(record, "run_final_closure_record")


def test_factory_metadata_defaults_to_empty_dict(tmp_path):
    run_id, _, _, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    record = _run_final_closure_record(
        run_id,
        completion,
        review,
        plan,
        request,
        result,
        verification,
        pvfp,
        pver,
        pve_result,
        pve_verification,
        closure_items=[],
    )
    assert record["metadata"] == {}


def test_factory_notes_remain_none(tmp_path):
    run_id, _, _, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    record = _run_final_closure_record(
        run_id,
        completion,
        review,
        plan,
        request,
        result,
        verification,
        pvfp,
        pver,
        pve_result,
        pve_verification,
        closure_items=[],
        notes=None,
    )
    assert record["notes"] is None


def test_factory_reason_remains_none(tmp_path):
    run_id, _, _, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    record = _run_final_closure_record(
        run_id,
        completion,
        review,
        plan,
        request,
        result,
        verification,
        pvfp,
        pver,
        pve_result,
        pve_verification,
        closure_items=[_sample_closure_item(reason=None)],
    )
    assert record["closure_items"][0]["reason"] is None


def test_factory_normalizes_evidence_and_item_metadata(tmp_path):
    run_id, _, _, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    record = _run_final_closure_record(
        run_id,
        completion,
        review,
        plan,
        request,
        result,
        verification,
        pvfp,
        pver,
        pve_result,
        pve_verification,
        closure_items=[
            {
                "closure_item_id": "ri-1",
                "source_post_verification_execution_verification_item_id": None,
                "source_post_verification_execution_result_item_id": None,
                "source_post_verification_execution_request_item_id": None,
                "source_post_verification_followup_item_id": None,
                "source_execution_item_id": None,
                "source_followup_item_id": None,
                "verification_decision_after_result": None,
                "closure_decision": contracts.RUN_FINAL_CLOSURE_ITEM_DECISION_ACCEPTED,
                "reason": None,
            }
        ],
    )
    assert record["closure_items"][0]["evidence"] == {}
    assert record["closure_items"][0]["metadata"] == {}


def test_factory_does_not_write_files(tmp_path):
    run_id, _, _, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    assert not contracts.run_final_closure_record_json_path(
        run_id, tmp_path
    ).exists()


def test_factory_does_not_create_events(tmp_path):
    run_id, _, _, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    before = len(events.read_task_events(run_id, base_dir=tmp_path))
    _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    assert len(events.read_task_events(run_id, base_dir=tmp_path)) == before


def test_factory_does_not_execute_external_systems(tmp_path):
    run_id, _, _, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    with patch("subprocess.run") as mock_subprocess, patch("webbrowser.open") as browser:
        _run_final_closure_record(
            run_id,
            completion,
            review,
            plan,
            request,
            result,
            verification,
            pvfp,
            pver,
            pve_result,
            pve_verification,
        )
    mock_subprocess.assert_not_called()
    browser.assert_not_called()


# --- C. Fingerprint ---


def test_fingerprint_validates_schema_first():
    bad = _minimal_run_final_closure()
    bad["metadata"] = "bad"
    with pytest.raises(ValueError, match="metadata must be a dict"):
        contracts.run_final_closure_fingerprint(bad)


def test_equivalent_records_same_fingerprint():
    first = _minimal_run_final_closure(
        closure_items=[_sample_closure_item()],
        final_closure_status=contracts.RUN_FINAL_CLOSURE_STATUS_CLOSED_VERIFIED,
    )
    second = dict(first)
    assert contracts.run_final_closure_fingerprint(
        first
    ) == contracts.run_final_closure_fingerprint(second)


def test_changed_closure_items_changes_fingerprint():
    base = _minimal_run_final_closure(
        closure_items=[_sample_closure_item()],
        final_closure_status=contracts.RUN_FINAL_CLOSURE_STATUS_CLOSED_VERIFIED,
    )
    changed = dict(base)
    changed["closure_items"] = [
        _sample_closure_item(reason="different reason")
    ]
    assert contracts.run_final_closure_fingerprint(
        base
    ) != contracts.run_final_closure_fingerprint(changed)


def test_changed_metadata_changes_fingerprint():
    base = _minimal_run_final_closure()
    changed = dict(base)
    changed["metadata"] = {"note": "changed"}
    assert contracts.run_final_closure_fingerprint(
        base
    ) != contracts.run_final_closure_fingerprint(changed)


def test_fingerprint_uses_canonical_json():
    record = _minimal_run_final_closure(metadata={"z": 1, "a": 2})
    expected = json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    assert contracts.run_final_closure_fingerprint(record) == expected


def test_fingerprint_does_not_read_files():
    record = _minimal_run_final_closure()
    with patch("htr.io.read_json") as mock_read:
        contracts.run_final_closure_fingerprint(record)
    mock_read.assert_not_called()


# --- D. Closure source/item correspondence ---


def test_global_manual_closure_items_with_null_source_verification_item_id_allowed():
    verification_record = _minimal_post_verification_execution_verification(
        verification_items=[_sample_post_verification_verification_item()]
    )
    closure_items = [
        _sample_closure_item(source_post_verification_execution_verification_item_id=None)
    ]
    contracts.validate_run_final_closure_sources_correspond(
        closure_items, verification_record
    )


def test_verification_linked_closure_item_must_reference_existing_source_verification_item_id():
    verification_record = _minimal_post_verification_execution_verification(
        verification_items=[_sample_post_verification_verification_item()]
    )
    closure_items = [
        _sample_closure_item(
            source_post_verification_execution_verification_item_id="missing"
        )
    ]
    with pytest.raises(
        ValueError,
        match="unknown source_post_verification_execution_verification_item_id",
    ):
        contracts.validate_run_final_closure_sources_correspond(
            closure_items, verification_record
        )


def test_verification_linked_closure_item_preserves_source_post_verification_execution_result_item_id(
    tmp_path,
):
    run_id, _, _, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    closure_item = _closure_item_from_verification_item(
        pve_verification["verification_items"][0]
    )
    contracts.validate_run_final_closure_sources_correspond(
        [closure_item], pve_verification
    )


def test_verification_linked_closure_item_preserves_source_post_verification_execution_request_item_id(
    tmp_path,
):
    run_id, _, _, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    closure_item = _closure_item_from_verification_item(
        pve_verification["verification_items"][0]
    )
    assert (
        closure_item["source_post_verification_execution_request_item_id"]
        == pve_verification["verification_items"][0]["source_request_item_id"]
    )


def test_verification_linked_closure_item_preserves_source_post_verification_followup_item_id(
    tmp_path,
):
    run_id, _, _, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    closure_item = _closure_item_from_verification_item(
        pve_verification["verification_items"][0]
    )
    assert (
        closure_item["source_post_verification_followup_item_id"]
        == pve_verification["verification_items"][0][
            "source_post_verification_followup_item_id"
        ]
    )


def test_verification_linked_closure_item_preserves_source_execution_item_id(tmp_path):
    run_id, _, _, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    closure_item = _closure_item_from_verification_item(
        pve_verification["verification_items"][0]
    )
    assert (
        closure_item["source_execution_item_id"]
        == pve_verification["verification_items"][0]["source_execution_item_id"]
    )


def test_verification_linked_closure_item_preserves_source_followup_item_id(tmp_path):
    run_id, _, _, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    closure_item = _closure_item_from_verification_item(
        pve_verification["verification_items"][0]
    )
    assert (
        closure_item["source_followup_item_id"]
        == pve_verification["verification_items"][0]["source_followup_item_id"]
    )


def test_verification_linked_closure_item_preserves_verification_decision_after_result(
    tmp_path,
):
    run_id, _, _, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    closure_item = _closure_item_from_verification_item(
        pve_verification["verification_items"][0]
    )
    assert (
        closure_item["verification_decision_after_result"]
        == pve_verification["verification_items"][0][
            "verification_decision_after_result"
        ]
    )


@pytest.mark.parametrize(
    "field,value,match",
    [
        (
            "source_post_verification_execution_result_item_id",
            "wrong-result",
            "source_post_verification_execution_result_item_id",
        ),
        (
            "source_post_verification_execution_request_item_id",
            "wrong-request",
            "source_post_verification_execution_request_item_id",
        ),
        (
            "source_post_verification_followup_item_id",
            "wrong-pvfp",
            "source_post_verification_followup_item_id",
        ),
        ("source_execution_item_id", "wrong-exec", "source_execution_item_id"),
        ("source_followup_item_id", "wrong-followup", "source_followup_item_id"),
        (
            "verification_decision_after_result",
            "rejected",
            "verification_decision_after_result",
        ),
    ],
)
def test_mismatch_for_preserved_source_field_fails(field, value, match):
    closure_item = _sample_closure_item()
    verification_record = {
        "verification_items": [
            {
                "verification_item_id": closure_item[
                    "source_post_verification_execution_verification_item_id"
                ],
                "source_result_item_id": closure_item[
                    "source_post_verification_execution_result_item_id"
                ],
                "source_request_item_id": closure_item[
                    "source_post_verification_execution_request_item_id"
                ],
                "source_post_verification_followup_item_id": closure_item[
                    "source_post_verification_followup_item_id"
                ],
                "source_execution_item_id": closure_item["source_execution_item_id"],
                "source_followup_item_id": closure_item["source_followup_item_id"],
                "verification_decision_after_result": closure_item[
                    "verification_decision_after_result"
                ],
            }
        ]
    }
    closure_item[field] = value
    with pytest.raises(ValueError, match=f"{match} does not match"):
        contracts.validate_run_final_closure_sources_correspond(
            [closure_item], verification_record
        )


# --- E. Record lifecycle success ---


def test_record_run_final_closure_succeeds_after_full_chain(tmp_path):
    run_id, _, _, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    returned = events.record_run_final_closure(
        tmp_path, run_id, record, actor="recorder"
    )
    assert returned == record


def test_source_execution_result_fingerprint_must_match_current(tmp_path):
    run_id, _, _, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    assert record["source_run_execution_result_fingerprint"] == (
        contracts.run_execution_result_fingerprint(result)
    )
    events.record_run_final_closure(
        tmp_path, run_id, record, actor="recorder"
    )
    assert io.read_json(
        contracts.run_final_closure_record_json_path(
            run_id, tmp_path
        )
    )["source_run_execution_result_fingerprint"] == contracts.run_execution_result_fingerprint(
        result
    )


def test_source_execution_verification_fingerprint_must_match_current(tmp_path):
    run_id, _, _, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    assert record["source_run_execution_verification_fingerprint"] == (
        contracts.run_execution_verification_fingerprint(verification)
    )


def test_source_post_verification_followup_plan_fingerprint_must_match_current(tmp_path):
    run_id, _, _, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    assert record["source_post_verification_followup_plan_fingerprint"] == (
        contracts.run_post_verification_followup_plan_fingerprint(pvfp)
    )


def test_source_post_verification_execution_result_fingerprint_must_match_current(
    tmp_path,
):
    run_id, _, _, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    assert record["source_post_verification_execution_result_fingerprint"] == (
        contracts.run_post_verification_execution_result_fingerprint(pve_result)
    )


def test_source_post_verification_execution_request_fingerprint_must_match_current(
    tmp_path,
):
    run_id, _, _, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    assert record["source_post_verification_execution_request_fingerprint"] == (
        contracts.run_post_verification_execution_request_fingerprint(pver)
    )


def test_verification_linked_closure_items_correspond_to_pve_verification(tmp_path):
    run_id, _, _, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    contracts.validate_run_final_closure_sources_correspond(
        record["closure_items"], pve_verification
    )
    events.record_run_final_closure(
        tmp_path, run_id, record, actor="human"
    )


def test_verification_record_points_to_current_execution_result_fingerprint(tmp_path):
    run_id, _, _, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    assert verification["source_execution_result_fingerprint"] == (
        contracts.run_execution_result_fingerprint(result)
    )


def test_post_verification_followup_plan_points_to_current_result_and_verification(
    tmp_path,
):
    run_id, _, _, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    assert pvfp["source_execution_result_fingerprint"] == (
        contracts.run_execution_result_fingerprint(result)
    )
    assert pvfp["source_execution_verification_fingerprint"] == (
        contracts.run_execution_verification_fingerprint(verification)
    )


def test_post_verification_execution_result_points_to_current_chain_fingerprints(
    tmp_path,
):
    run_id, _, _, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    assert pve_result["source_execution_result_fingerprint"] == (
        contracts.run_execution_result_fingerprint(result)
    )
    assert pve_result["source_execution_verification_fingerprint"] == (
        contracts.run_execution_verification_fingerprint(verification)
    )
    assert pve_result["source_post_verification_followup_plan_fingerprint"] == (
        contracts.run_post_verification_followup_plan_fingerprint(pvfp)
    )
    assert pve_result["source_post_verification_execution_request_fingerprint"] == (
        contracts.run_post_verification_execution_request_fingerprint(pver)
    )


def test_post_verification_execution_request_points_to_current_chain_fingerprints(
    tmp_path,
):
    run_id, _, _, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
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


@pytest.mark.parametrize(
    "final_closure_status,item_decisions",
    [
        (contracts.RUN_FINAL_CLOSURE_STATUS_CLOSED_NO_ACTION, []),
        (
            contracts.RUN_FINAL_CLOSURE_STATUS_CLOSED_VERIFIED,
            [contracts.RUN_FINAL_CLOSURE_ITEM_DECISION_ACCEPTED],
        ),
        (
            contracts.RUN_FINAL_CLOSURE_STATUS_CLOSED_REJECTED,
            [contracts.RUN_FINAL_CLOSURE_ITEM_DECISION_REJECTED],
        ),
        (
            contracts.RUN_FINAL_CLOSURE_STATUS_CLOSED_NEEDS_MORE_WORK,
            [contracts.RUN_FINAL_CLOSURE_ITEM_DECISION_NEEDS_MORE_WORK],
        ),
    ],
)
def test_final_closure_status_writes_record_and_event(tmp_path, final_closure_status, item_decisions):
    run_id, _, _, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    closure_items = []
    for idx, decision in enumerate(item_decisions):
        if idx < len(pve_verification["verification_items"]):
            closure_items.append(
                _closure_item_from_verification_item(
                    pve_verification["verification_items"][idx],
                    decision=decision,
                    closure_item_id=f"closure-{idx + 1}",
                )
            )
        else:
            closure_items.append(
                _sample_closure_item(
                    closure_item_id=f"closure-{idx + 1}",
                    source_post_verification_execution_verification_item_id=None,
                    closure_decision=decision,
                )
            )
    record = _run_final_closure_record(
        run_id,
        completion,
        review,
        plan,
        request,
        result,
        verification,
        pvfp,
        pver,
        pve_result,
        pve_verification,
        closure_items=closure_items,
        final_closure_status=final_closure_status,
    )
    before = len(events.read_task_events(run_id, base_dir=tmp_path))
    events.record_run_final_closure(
        tmp_path, run_id, record, actor="recorder"
    )
    assert io.read_json(
        contracts.run_final_closure_record_json_path(
            run_id, tmp_path
        )
    ) == record
    rows = events.read_task_events(run_id, base_dir=tmp_path)
    assert len(rows) == before + 1
    assert (
        rows[-1]["event_type"]
        == events.EVENT_TYPE_RUN_FINAL_CLOSURE_RECORDED
    )
    assert record["final_closure_status"] == final_closure_status


def test_record_run_final_closure_event_payload_fields(tmp_path):
    run_id, _, _, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification, closer="assistant"
    )
    events.record_run_final_closure(
        tmp_path, run_id, record, actor="operator"
    )
    event = events.read_task_events(run_id, base_dir=tmp_path)[-1]
    assert "task_id" not in event
    assert "attempt_id" not in event
    payload = event["payload"]
    assert payload["run_id"] == run_id
    assert payload["closer"] == record["closer"]
    assert payload["final_closure_status"] == record["final_closure_status"]
    assert payload["source_post_verification_followup_plan_fingerprint"] == record[
        "source_post_verification_followup_plan_fingerprint"
    ]
    assert payload["source_post_verification_execution_request_fingerprint"] == record[
        "source_post_verification_execution_request_fingerprint"
    ]
    assert payload["source_post_verification_execution_result_fingerprint"] == record[
        "source_post_verification_execution_result_fingerprint"
    ]
    assert payload["run_final_closure_fingerprint"] == (
        contracts.run_final_closure_fingerprint(record)
    )
    assert payload["run_final_closure_record_path"].endswith(
        "run_final_closure_record.json"
    )


def test_record_run_final_closure_returns_written_record(tmp_path):
    run_id, _, _, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    returned = events.record_run_final_closure(
        tmp_path, run_id, record, actor="human"
    )
    assert returned == io.read_json(
        contracts.run_final_closure_record_json_path(
            run_id, tmp_path
        )
    )


# --- F. Preconditions ---


def test_invalid_run_id_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(ValueError):
        events.record_run_final_closure(
            tmp_path, "bad", record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_invalid_schema_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    record["metadata"] = "bad"
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(ValueError):
        events.record_run_final_closure(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_run_id_mismatch_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(ValueError, match="closure record run_id does not match"):
        events.record_run_final_closure(
            tmp_path,
            run_id,
            _run_final_closure_record(
                new_run_id(),
                completion,
                review,
                plan,
                request,
                result,
                verification,
                pvfp,
                pver,
                pve_result,
                pve_verification,
            ),
            actor="human",
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_missing_run_manifest_fails_no_side_effects(tmp_path):
    run_id = new_run_id()
    with pytest.raises(InvalidTransition):
        events.record_run_final_closure(
            tmp_path,
            run_id,
            _minimal_run_final_closure(run_id=run_id),
            actor="human",
        )


def test_run_not_completed_fails_no_side_effects(tmp_path):
    run_id = new_run_id()
    io.create_run_workspace(run_id, base_dir=tmp_path)
    before = _snapshot(tmp_path, run_id, [])
    with pytest.raises(InvalidTransition, match="is not completed"):
        events.record_run_final_closure(
            tmp_path,
            run_id,
            _minimal_run_final_closure(run_id=run_id),
            actor="human",
        )
    assert _snapshot(tmp_path, run_id, []) == before


def test_missing_completion_record_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    contracts.run_completion_record_json_path(run_id, tmp_path).unlink()
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(InvalidTransition, match="run_completion_record.json is missing"):
        events.record_run_final_closure(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_missing_review_record_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    contracts.run_review_record_json_path(run_id, tmp_path).unlink()
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(InvalidTransition, match="run_review_record.json is missing"):
        events.record_run_final_closure(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_missing_followup_plan_record_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    contracts.run_followup_plan_record_json_path(run_id, tmp_path).unlink()
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(InvalidTransition, match="run_followup_plan_record.json is missing"):
        events.record_run_final_closure(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_missing_execution_request_record_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    contracts.run_execution_request_record_json_path(run_id, tmp_path).unlink()
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(
        InvalidTransition, match="run_execution_request_record.json is missing"
    ):
        events.record_run_final_closure(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_missing_execution_result_record_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    contracts.run_execution_result_record_json_path(run_id, tmp_path).unlink()
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(
        InvalidTransition, match="run_execution_result_record.json is missing"
    ):
        events.record_run_final_closure(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_missing_execution_verification_record_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    contracts.run_execution_verification_record_json_path(run_id, tmp_path).unlink()
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(
        InvalidTransition, match="run_execution_verification_record.json is missing"
    ):
        events.record_run_final_closure(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_missing_post_verification_followup_plan_record_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    contracts.run_post_verification_followup_plan_record_json_path(
        run_id, tmp_path
    ).unlink()
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(
        InvalidTransition,
        match="run_post_verification_followup_plan_record.json is missing",
    ):
        events.record_run_final_closure(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_missing_post_verification_execution_request_record_fails_no_side_effects(
    tmp_path,
):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    contracts.run_post_verification_execution_request_record_json_path(
        run_id, tmp_path
    ).unlink()
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(
        InvalidTransition,
        match="run_post_verification_execution_request_record.json is missing",
    ):
        events.record_run_final_closure(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_missing_post_verification_execution_result_record_fails_no_side_effects(
    tmp_path,
):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    contracts.run_post_verification_execution_result_record_json_path(
        run_id, tmp_path
    ).unlink()
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(
        InvalidTransition,
        match="run_post_verification_execution_result_record.json is missing",
    ):
        events.record_run_final_closure(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before




def test_missing_post_verification_execution_verification_record_fails_no_side_effects(
    tmp_path,
):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    contracts.run_post_verification_execution_verification_record_json_path(
        run_id, tmp_path
    ).unlink()
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(
        InvalidTransition,
        match="run_post_verification_execution_verification_record.json is missing",
    ):
        events.record_run_final_closure(tmp_path, run_id, record, actor="human")
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before




def test_invalid_completion_schema_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    bad = dict(completion)
    bad["metadata"] = "bad"
    io.atomic_write_json(
        contracts.run_completion_record_json_path(run_id, tmp_path), bad
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(ValueError, match="metadata must be a dict"):
        events.record_run_final_closure(tmp_path, run_id, record, actor="human")
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_invalid_review_schema_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    bad = dict(review)
    bad["metadata"] = "bad"
    io.atomic_write_json(contracts.run_review_record_json_path(run_id, tmp_path), bad)
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(ValueError, match="metadata must be a dict"):
        events.record_run_final_closure(tmp_path, run_id, record, actor="human")
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_invalid_followup_plan_schema_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    bad = dict(plan)
    bad["metadata"] = "bad"
    io.atomic_write_json(
        contracts.run_followup_plan_record_json_path(run_id, tmp_path), bad
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(ValueError, match="metadata must be a dict"):
        events.record_run_final_closure(tmp_path, run_id, record, actor="human")
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_invalid_execution_request_schema_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    bad = dict(request)
    bad["metadata"] = "bad"
    io.atomic_write_json(
        contracts.run_execution_request_record_json_path(run_id, tmp_path), bad
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(ValueError, match="metadata must be a dict"):
        events.record_run_final_closure(tmp_path, run_id, record, actor="human")
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_invalid_post_verification_execution_verification_schema_fails_no_side_effects(
    tmp_path,
):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    bad = dict(pve_verification)
    bad["metadata"] = "bad"
    io.atomic_write_json(
        contracts.run_post_verification_execution_verification_record_json_path(
            run_id, tmp_path
        ),
        bad,
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(ValueError, match="metadata must be a dict"):
        events.record_run_final_closure(tmp_path, run_id, record, actor="human")
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_invalid_execution_result_schema_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    bad_result = dict(result)
    bad_result["metadata"] = "bad"
    io.atomic_write_json(
        contracts.run_execution_result_record_json_path(run_id, tmp_path), bad_result
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(ValueError, match="metadata must be a dict"):
        events.record_run_final_closure(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_invalid_execution_verification_schema_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    bad_verification = dict(verification)
    bad_verification["metadata"] = "bad"
    io.atomic_write_json(
        contracts.run_execution_verification_record_json_path(run_id, tmp_path),
        bad_verification,
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(ValueError, match="metadata must be a dict"):
        events.record_run_final_closure(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_invalid_post_verification_followup_plan_schema_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    bad_pvfp = dict(pvfp)
    bad_pvfp["metadata"] = "bad"
    io.atomic_write_json(
        contracts.run_post_verification_followup_plan_record_json_path(
            run_id, tmp_path
        ),
        bad_pvfp,
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(ValueError, match="metadata must be a dict"):
        events.record_run_final_closure(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_invalid_post_verification_execution_request_schema_fails_no_side_effects(
    tmp_path,
):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    bad_pver = dict(pver)
    bad_pver["metadata"] = "bad"
    io.atomic_write_json(
        contracts.run_post_verification_execution_request_record_json_path(
            run_id, tmp_path
        ),
        bad_pver,
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(ValueError, match="metadata must be a dict"):
        events.record_run_final_closure(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_invalid_post_verification_execution_result_schema_fails_no_side_effects(
    tmp_path,
):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    bad_pve_result = dict(pve_result)
    bad_pve_result["metadata"] = "bad"
    io.atomic_write_json(
        contracts.run_post_verification_execution_result_record_json_path(
            run_id, tmp_path
        ),
        bad_pve_result,
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(ValueError, match="metadata must be a dict"):
        events.record_run_final_closure(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before




def test_source_run_completion_fingerprint_mismatch_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    bad = dict(record)
    bad["source_run_completion_fingerprint"] = "wrong"
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(InvalidTransition, match="source_run_completion_fingerprint"):
        events.record_run_final_closure(tmp_path, run_id, bad, actor="human")
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_source_run_review_fingerprint_mismatch_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    bad = dict(record)
    bad["source_run_review_fingerprint"] = "wrong"
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(InvalidTransition, match="source_run_review_fingerprint"):
        events.record_run_final_closure(tmp_path, run_id, bad, actor="human")
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_source_run_followup_plan_fingerprint_mismatch_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    bad = dict(record)
    bad["source_run_followup_plan_fingerprint"] = "wrong"
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(InvalidTransition, match="source_run_followup_plan_fingerprint"):
        events.record_run_final_closure(tmp_path, run_id, bad, actor="human")
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_source_run_execution_request_fingerprint_mismatch_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    bad = dict(record)
    bad["source_run_execution_request_fingerprint"] = "wrong"
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(InvalidTransition, match="source_run_execution_request_fingerprint"):
        events.record_run_final_closure(tmp_path, run_id, bad, actor="human")
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_source_post_verification_execution_verification_fingerprint_mismatch_fails_no_side_effects(
    tmp_path,
):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    bad = dict(record)
    bad["source_post_verification_execution_verification_fingerprint"] = "wrong"
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(
        InvalidTransition,
        match="source_post_verification_execution_verification_fingerprint",
    ):
        events.record_run_final_closure(tmp_path, run_id, bad, actor="human")
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_followup_plan_stale_review_decision_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    stale_plan = dict(plan)
    stale_plan["source_review_decision"] = contracts.RUN_REVIEW_ACCEPTED
    io.atomic_write_json(
        contracts.run_followup_plan_record_json_path(run_id, tmp_path), stale_plan
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    record["source_run_followup_plan_fingerprint"] = contracts.run_followup_plan_fingerprint(
        stale_plan
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(
        InvalidTransition,
        match="run_followup_plan_record source_review_decision",
    ):
        events.record_run_final_closure(tmp_path, run_id, record, actor="human")
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_execution_request_stale_followup_plan_fingerprint_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    stale_request = dict(request)
    stale_request["source_followup_plan_fingerprint"] = "stale"
    io.atomic_write_json(
        contracts.run_execution_request_record_json_path(run_id, tmp_path), stale_request
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    record["source_run_execution_request_fingerprint"] = (
        contracts.run_execution_request_fingerprint(stale_request)
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(
        InvalidTransition,
        match="run_execution_request_record source_followup_plan_fingerprint",
    ):
        events.record_run_final_closure(tmp_path, run_id, record, actor="human")
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_pve_verification_stale_result_fingerprint_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    stale_pve_ver = dict(pve_verification)
    stale_pve_ver["source_execution_result_fingerprint"] = "stale"
    io.atomic_write_json(
        contracts.run_post_verification_execution_verification_record_json_path(
            run_id, tmp_path
        ),
        stale_pve_ver,
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    record["source_post_verification_execution_verification_fingerprint"] = (
        contracts.run_post_verification_execution_verification_fingerprint(stale_pve_ver)
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(
        InvalidTransition,
        match="run_post_verification_execution_verification_record source_execution_result_fingerprint",
    ):
        events.record_run_final_closure(tmp_path, run_id, record, actor="human")
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_source_run_execution_result_fingerprint_mismatch_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    bad = dict(record)
    bad["source_run_execution_result_fingerprint"] = "wrong"
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(InvalidTransition, match="source_run_execution_result_fingerprint"):
        events.record_run_final_closure(
            tmp_path, run_id, bad, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_source_run_execution_verification_fingerprint_mismatch_fails_no_side_effects(
    tmp_path,
):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    bad = dict(record)
    bad["source_run_execution_verification_fingerprint"] = "wrong"
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(
        InvalidTransition, match="source_run_execution_verification_fingerprint"
    ):
        events.record_run_final_closure(
            tmp_path, run_id, bad, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_source_post_verification_followup_plan_fingerprint_mismatch_fails_no_side_effects(
    tmp_path,
):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    bad = dict(record)
    bad["source_post_verification_followup_plan_fingerprint"] = "wrong"
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(
        InvalidTransition, match="source_post_verification_followup_plan_fingerprint"
    ):
        events.record_run_final_closure(
            tmp_path, run_id, bad, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_source_post_verification_execution_request_fingerprint_mismatch_fails_no_side_effects(
    tmp_path,
):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    bad = dict(record)
    bad["source_post_verification_execution_request_fingerprint"] = "wrong"
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(
        InvalidTransition,
        match="source_post_verification_execution_request_fingerprint",
    ):
        events.record_run_final_closure(
            tmp_path, run_id, bad, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_source_post_verification_execution_result_fingerprint_mismatch_fails_no_side_effects(
    tmp_path,
):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    bad = dict(record)
    bad["source_post_verification_execution_result_fingerprint"] = "wrong"
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(
        InvalidTransition,
        match="source_post_verification_execution_result_fingerprint",
    ):
        events.record_run_final_closure(
            tmp_path, run_id, bad, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_verification_record_stale_result_fingerprint_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    stale = dict(verification)
    stale["source_execution_result_fingerprint"] = "stale"
    io.atomic_write_json(
        contracts.run_execution_verification_record_json_path(run_id, tmp_path), stale
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    record["source_run_execution_verification_fingerprint"] = (
        contracts.run_execution_verification_fingerprint(stale)
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(
        InvalidTransition,
        match="run_execution_verification_record source_execution_result_fingerprint",
    ):
        events.record_run_final_closure(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_followup_plan_stale_result_fingerprint_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    stale_pvfp = dict(pvfp)
    stale_pvfp["source_execution_result_fingerprint"] = "stale"
    io.atomic_write_json(
        contracts.run_post_verification_followup_plan_record_json_path(
            run_id, tmp_path
        ),
        stale_pvfp,
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    record["source_post_verification_followup_plan_fingerprint"] = (
        contracts.run_post_verification_followup_plan_fingerprint(stale_pvfp)
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(
        InvalidTransition,
        match="run_post_verification_followup_plan_record source_execution_result_fingerprint",
    ):
        events.record_run_final_closure(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_followup_plan_stale_verification_fingerprint_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    stale_pvfp = dict(pvfp)
    stale_pvfp["source_execution_verification_fingerprint"] = "stale"
    io.atomic_write_json(
        contracts.run_post_verification_followup_plan_record_json_path(
            run_id, tmp_path
        ),
        stale_pvfp,
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    record["source_post_verification_followup_plan_fingerprint"] = (
        contracts.run_post_verification_followup_plan_fingerprint(stale_pvfp)
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(
        InvalidTransition,
        match="run_post_verification_followup_plan_record source_execution_verification_fingerprint",
    ):
        events.record_run_final_closure(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_request_stale_result_fingerprint_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    stale_pver = dict(pver)
    stale_pver["source_execution_result_fingerprint"] = "stale"
    io.atomic_write_json(
        contracts.run_post_verification_execution_request_record_json_path(
            run_id, tmp_path
        ),
        stale_pver,
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    record["source_post_verification_execution_request_fingerprint"] = (
        contracts.run_post_verification_execution_request_fingerprint(stale_pver)
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(
        InvalidTransition,
        match="run_post_verification_execution_request_record source_execution_result_fingerprint",
    ):
        events.record_run_final_closure(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_request_stale_verification_fingerprint_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    stale_pver = dict(pver)
    stale_pver["source_execution_verification_fingerprint"] = "stale"
    io.atomic_write_json(
        contracts.run_post_verification_execution_request_record_json_path(
            run_id, tmp_path
        ),
        stale_pver,
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    record["source_post_verification_execution_request_fingerprint"] = (
        contracts.run_post_verification_execution_request_fingerprint(stale_pver)
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(
        InvalidTransition,
        match="run_post_verification_execution_request_record source_execution_verification_fingerprint",
    ):
        events.record_run_final_closure(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_request_stale_followup_plan_fingerprint_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    stale_pver = dict(pver)
    stale_pver["source_post_verification_followup_plan_fingerprint"] = "stale"
    io.atomic_write_json(
        contracts.run_post_verification_execution_request_record_json_path(
            run_id, tmp_path
        ),
        stale_pver,
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    record["source_post_verification_execution_request_fingerprint"] = (
        contracts.run_post_verification_execution_request_fingerprint(stale_pver)
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(
        InvalidTransition,
        match="run_post_verification_execution_request_record source_post_verification_followup_plan_fingerprint",
    ):
        events.record_run_final_closure(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_pve_result_stale_result_fingerprint_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    stale_pve_result = dict(pve_result)
    stale_pve_result["source_execution_result_fingerprint"] = "stale"
    io.atomic_write_json(
        contracts.run_post_verification_execution_result_record_json_path(
            run_id, tmp_path
        ),
        stale_pve_result,
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    record["source_post_verification_execution_result_fingerprint"] = (
        contracts.run_post_verification_execution_result_fingerprint(stale_pve_result)
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(
        InvalidTransition,
        match="run_post_verification_execution_result_record source_execution_result_fingerprint",
    ):
        events.record_run_final_closure(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_pve_result_stale_verification_fingerprint_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    stale_pve_result = dict(pve_result)
    stale_pve_result["source_execution_verification_fingerprint"] = "stale"
    io.atomic_write_json(
        contracts.run_post_verification_execution_result_record_json_path(
            run_id, tmp_path
        ),
        stale_pve_result,
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    record["source_post_verification_execution_result_fingerprint"] = (
        contracts.run_post_verification_execution_result_fingerprint(stale_pve_result)
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(
        InvalidTransition,
        match="run_post_verification_execution_result_record source_execution_verification_fingerprint",
    ):
        events.record_run_final_closure(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_pve_result_stale_followup_plan_fingerprint_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    stale_pve_result = dict(pve_result)
    stale_pve_result["source_post_verification_followup_plan_fingerprint"] = "stale"
    io.atomic_write_json(
        contracts.run_post_verification_execution_result_record_json_path(
            run_id, tmp_path
        ),
        stale_pve_result,
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    record["source_post_verification_execution_result_fingerprint"] = (
        contracts.run_post_verification_execution_result_fingerprint(stale_pve_result)
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(
        InvalidTransition,
        match="run_post_verification_execution_result_record source_post_verification_followup_plan_fingerprint",
    ):
        events.record_run_final_closure(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_pve_result_stale_request_fingerprint_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    stale_pve_result = dict(pve_result)
    stale_pve_result["source_post_verification_execution_request_fingerprint"] = "stale"
    io.atomic_write_json(
        contracts.run_post_verification_execution_result_record_json_path(
            run_id, tmp_path
        ),
        stale_pve_result,
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    record["source_post_verification_execution_result_fingerprint"] = (
        contracts.run_post_verification_execution_result_fingerprint(stale_pve_result)
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(
        InvalidTransition,
        match="run_post_verification_execution_result_record source_post_verification_execution_request_fingerprint",
    ):
        events.record_run_final_closure(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_unknown_source_verification_item_id_fails_no_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    closure_items = [
        _closure_item_from_verification_item(
            pve_verification["verification_items"][0],
            source_post_verification_execution_verification_item_id="unknown-verification",
        )
    ]
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification, closure_items=closure_items
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(InvalidTransition, match="unknown source_post_verification_execution_verification_item_id"):
        events.record_run_final_closure(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("source_post_verification_execution_result_item_id", "wrong-result", "source_post_verification_execution_result_item_id"),
        ("source_post_verification_execution_request_item_id", "wrong-request", "source_post_verification_execution_request_item_id"),
        ("source_post_verification_followup_item_id", "wrong-pvfp", "source_post_verification_followup_item_id"),
        ("source_execution_item_id", "wrong-exec", "source_execution_item_id"),
        ("source_followup_item_id", "wrong-followup", "source_followup_item_id"),
        ("verification_decision_after_result", "rejected", "verification_decision_after_result"),
    ],
)
def test_closure_item_field_mismatch_fails_no_side_effects(
    tmp_path, field, value, match
):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    closure_items = [_closure_item_from_verification_item(pve_verification["verification_items"][0])]
    closure_items[0][field] = value
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification, closure_items=closure_items
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(InvalidTransition, match=f"{match} does not match"):
        events.record_run_final_closure(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


# --- G. Write order ---


def test_write_order_record_before_event(tmp_path):
    run_id, _, _, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    ops = []
    real_append = events._append_run_event_internal

    def track_write(path, data):
        if str(path).endswith("run_final_closure_record.json"):
            ops.append("write_record")
        return io.atomic_write_json(path, data)

    def track_append(r, e, base_dir=None):
        ops.append("append_event")
        return real_append(r, e, base_dir)

    with patch("htr.events.atomic_write_json", side_effect=track_write), patch(
        "htr.events._append_run_event_internal", side_effect=track_append
    ):
        events.record_run_final_closure(
            tmp_path, run_id, record, actor="human"
        )
    assert ops == ["write_record", "append_event"]


# --- H. Replay-only ---


def test_existing_run_final_closure_event_id_none_raises(tmp_path):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    events.record_run_final_closure(
        tmp_path, run_id, record, actor="human", event_id=new_event_id()
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(RunFinalizedError):
        events.record_run_final_closure(
            tmp_path, run_id, record, actor="human"
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_existing_run_final_closure_missing_event_raises(tmp_path):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    io.atomic_write_json(
        contracts.run_final_closure_record_json_path(
            run_id, tmp_path
        ),
        record,
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(RunSealBlockedError):
        events.record_run_final_closure(
            tmp_path, run_id, record, actor="human", event_id=new_event_id()
        )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


@pytest.mark.parametrize(
    "mutator,expected",
    [
        (lambda e: {**e, "event_type": "task_status_changed"}, RunFinalizedError),
        (lambda e: {**e, "run_id": new_run_id()}, EventConflict),
        (lambda e: {**e, "actor": "other"}, EventConflict),
        (
            lambda e: {
                **e,
                "payload": {
                    **e["payload"],
                    "run_final_closure_fingerprint": "different",
                },
            },
            EventConflict,
        ),
        (
            lambda e: {
                **e,
                "payload": {
                    **e["payload"],
                    "source_run_execution_result_fingerprint": "different",
                },
            },
            EventConflict,
        ),
        (
            lambda e: {
                **e,
                "payload": {
                    **e["payload"],
                    "source_run_execution_verification_fingerprint": "different",
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
                    "source_post_verification_execution_result_fingerprint": "different",
                },
            },
            EventConflict,
        ),
        (
            lambda e: {
                **e,
                "payload": {
                    **e["payload"],
                    "final_closure_status": (
                        contracts.RUN_FINAL_CLOSURE_STATUS_CLOSED_NEEDS_MORE_WORK
                    ),
                },
            },
            EventConflict,
        ),
        (
            lambda e: {**e, "payload": {**e["payload"], "closer": "other"}},
            EventConflict,
        ),
    ],
)
def test_existing_run_final_closure_replay_semantic_mismatch(
    tmp_path, mutator, expected
):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    event_id = new_event_id()
    events.record_run_final_closure(
        tmp_path, run_id, record, actor="human", event_id=event_id
    )
    event = events.read_task_events(run_id, base_dir=tmp_path)[-1]
    bad_event = mutator(event)
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    with patch("htr.events._find_run_event_by_id", return_value=bad_event):
        with pytest.raises(expected):
            events.record_run_final_closure(
                tmp_path, run_id, record, actor="human", event_id=event_id
            )
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_existing_run_final_closure_same_event_id_same_semantic_returns_existing(
    tmp_path,
):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    event_id = new_event_id()
    events.record_run_final_closure(
        tmp_path, run_id, record, actor="human", event_id=event_id
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    returned = events.record_run_final_closure(
        tmp_path, run_id, record, actor="human", event_id=event_id
    )
    assert returned == record
    assert _snapshot(tmp_path, run_id, task_ids, attempt_ids) == before


def test_replay_only_no_writes_or_previous_record_mutation(tmp_path):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    event_id = new_event_id()
    events.record_run_final_closure(
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
    events.record_run_final_closure(
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
    run_id, _, _, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    event_id = new_event_id()
    first = events.record_run_final_closure(
        tmp_path, run_id, record, actor="human", event_id=event_id
    )
    second = events.record_run_final_closure(
        tmp_path, run_id, record, actor="human", event_id=event_id
    )
    assert second == first
    assert contracts.run_final_closure_record_json_path(run_id, tmp_path).exists()


def test_idempotent_replay_fails_when_event_exists_but_json_missing(tmp_path):
    run_id, _, _, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    event_id = new_event_id()
    events.record_run_final_closure(
        tmp_path, run_id, record, actor="human", event_id=event_id
    )
    closure_path = contracts.run_final_closure_record_json_path(run_id, tmp_path)
    assert closure_path.exists()
    closure_path.unlink()
    assert not closure_path.exists()
    with pytest.raises(RunSealBlockedError):
        events.record_run_final_closure(
            tmp_path, run_id, record, actor="human", event_id=event_id
        )
    assert not closure_path.exists()


def test_same_event_id_different_semantic_raises_conflict(tmp_path):
    run_id, _, _, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    event_id = new_event_id()
    first = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    events.record_run_final_closure(
        tmp_path, run_id, first, actor="human", event_id=event_id
    )
    second = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification, notes="different semantic"
    )
    with pytest.raises(EventConflict):
        events.record_run_final_closure(
            tmp_path, run_id, second, actor="human", event_id=event_id
        )


def test_idempotency_requires_post_verification_execution_verification(tmp_path):
    run_id = new_run_id()
    io.create_run_workspace(run_id, base_dir=tmp_path)
    manifest = io.read_json(paths.run_manifest_path(run_id, tmp_path))
    manifest["status"] = RUN_COMPLETED
    io.atomic_write_json(paths.run_manifest_path(run_id, tmp_path), manifest)
    completion_record = contracts.make_run_completion_record(
        run_id=run_id, completed_task_ids=[new_task_id()]
    )
    io.atomic_write_json(
        contracts.run_completion_record_json_path(run_id, tmp_path), completion_record
    )
    review_record = contracts.make_run_review_record(
        run_id=run_id, decision=contracts.RUN_REVIEW_NEEDS_FOLLOWUP
    )
    io.atomic_write_json(
        contracts.run_review_record_json_path(run_id, tmp_path), review_record
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
    stub_pver = contracts.make_run_post_verification_execution_request_record(
        run_id=run_id,
        source_execution_result_fingerprint=contracts.run_execution_result_fingerprint(
            stub_result
        ),
        source_execution_verification_fingerprint=contracts.run_execution_verification_fingerprint(
            stub_verification
        ),
        source_post_verification_followup_plan_fingerprint=contracts.run_post_verification_followup_plan_fingerprint(
            stub_pvfp
        ),
        request_items=[],
    )
    io.atomic_write_json(
        contracts.run_post_verification_execution_request_record_json_path(
            run_id, tmp_path
        ),
        stub_pver,
    )
    stub_pve_result = contracts.make_run_post_verification_execution_result_record(
        run_id=run_id,
        source_execution_result_fingerprint=contracts.run_execution_result_fingerprint(
            stub_result
        ),
        source_execution_verification_fingerprint=contracts.run_execution_verification_fingerprint(
            stub_verification
        ),
        source_post_verification_followup_plan_fingerprint=contracts.run_post_verification_followup_plan_fingerprint(
            stub_pvfp
        ),
        source_post_verification_execution_request_fingerprint=contracts.run_post_verification_execution_request_fingerprint(
            stub_pver
        ),
        result_items=[],
    )
    io.atomic_write_json(
        contracts.run_post_verification_execution_result_record_json_path(
            run_id, tmp_path
        ),
        stub_pve_result,
    )
    stub_closure = _minimal_run_final_closure(
        run_id=run_id,
        source_run_completion_fingerprint=contracts.run_completion_fingerprint(
            completion_record
        ),
        source_run_review_fingerprint=contracts.run_review_fingerprint(review_record),
        source_run_followup_plan_fingerprint=contracts.run_followup_plan_fingerprint(plan),
        source_run_execution_request_fingerprint=contracts.run_execution_request_fingerprint(
            request
        ),
        source_run_execution_result_fingerprint=contracts.run_execution_result_fingerprint(
            stub_result
        ),
        source_run_execution_verification_fingerprint=contracts.run_execution_verification_fingerprint(
            stub_verification
        ),
        source_post_verification_followup_plan_fingerprint=contracts.run_post_verification_followup_plan_fingerprint(
            stub_pvfp
        ),
        source_post_verification_execution_request_fingerprint=contracts.run_post_verification_execution_request_fingerprint(
            stub_pver
        ),
        source_post_verification_execution_result_fingerprint=contracts.run_post_verification_execution_result_fingerprint(
            stub_pve_result
        ),
    )
    with pytest.raises(
        InvalidTransition,
        match="run_post_verification_execution_verification_record.json is missing",
    ):
        events.record_run_final_closure(
            tmp_path,
            run_id,
            stub_closure,
            actor="human",
            event_id=new_event_id(),
        )


def test_idempotency_does_not_allow_non_completed_run(tmp_path):
    run_id = new_run_id()
    io.create_run_workspace(run_id, base_dir=tmp_path)
    with pytest.raises(InvalidTransition):
        events.record_run_final_closure(
            tmp_path,
            run_id,
            _minimal_run_final_closure(run_id=run_id),
            actor="human",
            event_id=new_event_id(),
        )


# --- J. Boundary preservation ---


def test_boundaries_preserved_on_success(tmp_path):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    before = _snapshot(tmp_path, run_id, task_ids, attempt_ids)
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    doc_path = tmp_path / "docs" / "readme.md"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("unchanged", encoding="utf-8")
    events.record_run_final_closure(
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
    assert after["post_verification_execution_result"] == before[
        "post_verification_execution_result"
    ]
    assert after["task_statuses"] == before["task_statuses"]
    assert after["attempt_statuses"] == before["attempt_statuses"]
    assert after["post_verification_execution_verification"] == before[
        "post_verification_execution_verification"
    ]
    assert after["final_closure_exists"] is True
    assert after["post_verification_execution_verification_exists"] is True
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


def test_task16_modules_do_not_import_forbidden_modules():
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


def test_package_exports_task16_apis():
    import htr

    for name in (
        "make_run_final_closure_record",
        "run_final_closure_fingerprint",
        "validate_run_final_closure_sources_correspond",
        "compute_run_final_closure_status",
        "record_run_final_closure",
        "EVENT_TYPE_RUN_FINAL_CLOSURE_RECORDED",
        "RUN_FINAL_CLOSURE_STATUS_CLOSED_VERIFIED",
        "RUN_FINAL_CLOSURE_STATUS_CLOSED_REJECTED",
        "RUN_FINAL_CLOSURE_STATUS_CLOSED_NEEDS_MORE_WORK",
        "RUN_FINAL_CLOSURE_STATUS_CLOSED_NO_ACTION",
        "RUN_FINAL_CLOSURE_ITEM_DECISION_ACCEPTED",
        "RUN_FINAL_CLOSURE_ITEM_DECISION_REJECTED",
        "RUN_FINAL_CLOSURE_ITEM_DECISION_NEEDS_MORE_WORK",
        "RUN_FINAL_CLOSURE_ITEM_DECISION_NO_ACTION",
    ):
        assert hasattr(htr, name)


# --- L. Phase 1 terminal semantics ---


def test_final_closure_is_terminal_for_phase_1_manual_workflow(tmp_path):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    events.record_run_final_closure(tmp_path, run_id, record, actor="human")
    assert contracts.run_final_closure_record_json_path(run_id, tmp_path).exists()
    event = events.read_task_events(run_id, base_dir=tmp_path)[-1]
    assert event["event_type"] == events.EVENT_TYPE_RUN_FINAL_CLOSURE_RECORDED


def test_record_run_final_closure_does_not_create_followup_plan(tmp_path):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    followup_before = contracts.run_followup_plan_record_json_path(
        run_id, tmp_path
    ).read_bytes()
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    events.record_run_final_closure(tmp_path, run_id, record, actor="human")
    assert (
        contracts.run_followup_plan_record_json_path(run_id, tmp_path).read_bytes()
        == followup_before
    )


def test_record_run_final_closure_does_not_create_execution_request(tmp_path):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    request_before = contracts.run_execution_request_record_json_path(
        run_id, tmp_path
    ).read_bytes()
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    events.record_run_final_closure(tmp_path, run_id, record, actor="human")
    assert (
        contracts.run_execution_request_record_json_path(run_id, tmp_path).read_bytes()
        == request_before
    )


def test_record_run_final_closure_does_not_create_execution_result(tmp_path):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    result_before = contracts.run_execution_result_record_json_path(
        run_id, tmp_path
    ).read_bytes()
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    events.record_run_final_closure(tmp_path, run_id, record, actor="human")
    assert (
        contracts.run_execution_result_record_json_path(run_id, tmp_path).read_bytes()
        == result_before
    )


def test_record_run_final_closure_does_not_create_verification_record(tmp_path):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    verification_before = contracts.run_execution_verification_record_json_path(
        run_id, tmp_path
    ).read_bytes()
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    events.record_run_final_closure(tmp_path, run_id, record, actor="human")
    assert (
        contracts.run_execution_verification_record_json_path(
            run_id, tmp_path
        ).read_bytes()
        == verification_before
    )


def test_record_run_final_closure_does_not_create_post_verification_records(tmp_path):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    pvfp_before = contracts.run_post_verification_followup_plan_record_json_path(
        run_id, tmp_path
    ).read_bytes()
    pver_before = contracts.run_post_verification_execution_request_record_json_path(
        run_id, tmp_path
    ).read_bytes()
    pve_result_before = contracts.run_post_verification_execution_result_record_json_path(
        run_id, tmp_path
    ).read_bytes()
    pve_ver_before = contracts.run_post_verification_execution_verification_record_json_path(
        run_id, tmp_path
    ).read_bytes()
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    events.record_run_final_closure(tmp_path, run_id, record, actor="human")
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
    assert (
        contracts.run_post_verification_execution_result_record_json_path(
            run_id, tmp_path
        ).read_bytes()
        == pve_result_before
    )
    assert (
        contracts.run_post_verification_execution_verification_record_json_path(
            run_id, tmp_path
        ).read_bytes()
        == pve_ver_before
    )


def test_record_run_final_closure_does_not_mutate_run_status(tmp_path):
    run_id, task_ids, attempt_ids, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    manifest_before = io.read_json(paths.run_manifest_path(run_id, tmp_path))
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    events.record_run_final_closure(tmp_path, run_id, record, actor="human")
    manifest_after = io.read_json(paths.run_manifest_path(run_id, tmp_path))
    assert manifest_after == manifest_before
    assert manifest_after["status"] == RUN_COMPLETED


def test_run_final_closure_record_is_source_of_truth_json(tmp_path):
    run_id, _, _, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    events.record_run_final_closure(tmp_path, run_id, record, actor="human")
    stored = io.read_json(contracts.run_final_closure_record_json_path(run_id, tmp_path))
    assert stored == record
    schemas.validate(stored, "run_final_closure_record")


def test_run_final_closure_recorded_event_is_audit_only(tmp_path):
    run_id, _, _, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification = (
        _run_with_post_verification_execution_verification(tmp_path)
    )
    record = _run_final_closure_record(
        run_id, completion, review, plan, request, result, verification, pvfp, pver, pve_result, pve_verification
    )
    before_events = len(events.read_task_events(run_id, base_dir=tmp_path))
    events.record_run_final_closure(tmp_path, run_id, record, actor="human")
    after_events = events.read_task_events(run_id, base_dir=tmp_path)
    assert len(after_events) == before_events + 1
    assert after_events[-1]["event_type"] == events.EVENT_TYPE_RUN_FINAL_CLOSURE_RECORDED
    assert "task_id" not in after_events[-1]
    assert "attempt_id" not in after_events[-1]
