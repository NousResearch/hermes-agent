"""Tests for Task 9 — Review-Gated Execution Request API."""

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
    TASK_COMPLETED,
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
    followup_items = kwargs.pop("followup_items", [_sample_followup_item()])
    summary = kwargs.pop("summary", "Follow up after review")
    return contracts.make_run_followup_plan_record(
        run_id=run_id,
        source_review_decision=contracts.RUN_REVIEW_NEEDS_FOLLOWUP,
        summary=summary,
        followup_items=followup_items,
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


def _execution_request_record(run_id, plan_record, **kwargs):
    execution_items = kwargs.pop("execution_items", [_sample_execution_item()])
    fp = kwargs.pop(
        "source_followup_plan_fingerprint",
        contracts.run_followup_plan_fingerprint(plan_record),
    )
    return contracts.make_run_execution_request_record(
        run_id=run_id,
        source_followup_plan_fingerprint=fp,
        execution_items=execution_items,
        **kwargs,
    )


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


def _run_with_reviewed_run(
    tmp_path,
    task_count=1,
    review_decision=contracts.RUN_REVIEW_NEEDS_FOLLOWUP,
):
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
        run_id=run_id, decision=review_decision
    )
    events.review_run_manually(run_id, review_record, base_dir=tmp_path)
    return run_id, task_ids, attempt_ids, completion_record, review_record


def _run_with_planned_run(tmp_path, task_count=1):
    run_id, task_ids, attempt_ids, completion_record, review_record = (
        _run_with_reviewed_run(tmp_path, task_count=task_count)
    )
    plan_record = _plan_record(run_id)
    events.plan_run_followup(run_id, plan_record, base_dir=tmp_path)
    return (
        run_id,
        task_ids,
        attempt_ids,
        completion_record,
        review_record,
        plan_record,
    )


def _snapshot(tmp_path, run_id, task_ids, attempt_ids=None):
    if attempt_ids is None:
        attempt_ids = []
    snap = {
        "events_len": len(events.read_task_events(run_id, base_dir=tmp_path)),
        "manifest": io.read_json(paths.run_manifest_path(run_id, tmp_path)),
        "execution_exists": contracts.run_execution_request_record_json_path(
            run_id, tmp_path
        ).exists(),
        "followup_exists": contracts.run_followup_plan_record_json_path(
            run_id, tmp_path
        ).exists(),
        "task_statuses": {
            tid: io.read_json(paths.task_status_path(run_id, tid, tmp_path))
            for tid in task_ids
        },
    }
    completion_path = contracts.run_completion_record_json_path(run_id, tmp_path)
    review_path = contracts.run_review_record_json_path(run_id, tmp_path)
    followup_path = contracts.run_followup_plan_record_json_path(run_id, tmp_path)
    snap["completion"] = (
        completion_path.read_bytes() if completion_path.exists() else None
    )
    snap["review"] = review_path.read_bytes() if review_path.exists() else None
    snap["followup"] = followup_path.read_bytes() if followup_path.exists() else None
    return snap


# --- A. Schema tests ---


def test_valid_run_execution_request_record_passes():
    run_id = new_run_id()
    plan = _plan_record(run_id)
    schemas.validate(
        _execution_request_record(run_id, plan), "run_execution_request_record"
    )


@pytest.mark.parametrize(
    "missing",
    [
        "run_id",
        "source_followup_plan_fingerprint",
        "requester",
        "request_status",
        "execution_items",
        "notes",
        "metadata",
    ],
)
def test_missing_required_fields_fail(missing):
    run_id = new_run_id()
    plan = _plan_record(run_id)
    record = _execution_request_record(run_id, plan)
    del record[missing]
    with pytest.raises(ValueError, match="missing fields"):
        schemas.validate(record, "run_execution_request_record")


def test_invalid_run_id_fails_schema():
    run_id = new_run_id()
    plan = _plan_record(run_id)
    record = _execution_request_record(run_id, plan)
    record["run_id"] = "not-a-run-id"
    with pytest.raises(ValueError, match="run_id must be a valid run id"):
        schemas.validate(record, "run_execution_request_record")


def test_invalid_source_followup_plan_fingerprint_fails():
    run_id = new_run_id()
    plan = _plan_record(run_id)
    record = _execution_request_record(run_id, plan)
    record["source_followup_plan_fingerprint"] = ""
    with pytest.raises(ValueError, match="source_followup_plan_fingerprint"):
        schemas.validate(record, "run_execution_request_record")


def test_invalid_requester_fails():
    with pytest.raises(ValueError, match="requester must be a non-empty string"):
        _execution_request_record(new_run_id(), _plan_record(new_run_id()), requester="")


def test_invalid_request_status_fails():
    with pytest.raises(ValueError, match="request_status"):
        _execution_request_record(
            new_run_id(),
            _plan_record(new_run_id()),
            request_status="running",
        )


def test_execution_items_must_be_list():
    run_id = new_run_id()
    plan = _plan_record(run_id)
    record = _execution_request_record(run_id, plan)
    record["execution_items"] = {}
    with pytest.raises(ValueError, match="execution_items must be a list"):
        schemas.validate(record, "run_execution_request_record")


def test_execution_items_entries_must_be_dicts():
    run_id = new_run_id()
    plan = _plan_record(run_id)
    record = _execution_request_record(run_id, plan)
    record["execution_items"] = ["bad"]
    with pytest.raises(ValueError, match="each execution item must be a dict"):
        schemas.validate(record, "run_execution_request_record")


@pytest.mark.parametrize(
    "field",
    ["item_id", "source_followup_item_id", "title", "execution_kind", "command"],
)
def test_execution_item_missing_required_field_fails(field):
    item = _sample_execution_item()
    del item[field]
    run_id = new_run_id()
    plan = _plan_record(run_id)
    record = _execution_request_record(run_id, plan)
    record["execution_items"] = [item]
    with pytest.raises(ValueError, match=f"execution item {field}"):
        schemas.validate(record, "run_execution_request_record")


def test_invalid_execution_kind_fails():
    with pytest.raises(ValueError, match="execution_kind is invalid"):
        _execution_request_record(
            new_run_id(),
            _plan_record(new_run_id()),
            execution_items=[_sample_execution_item(execution_kind="auto_run")],
        )


def test_command_must_be_dict():
    with pytest.raises(ValueError, match="command must be a dict"):
        _execution_request_record(
            new_run_id(),
            _plan_record(new_run_id()),
            execution_items=[_sample_execution_item(command="bad")],
        )


def test_approval_reason_must_be_string_or_none():
    with pytest.raises(ValueError, match="approval_reason must be a string or null"):
        _execution_request_record(
            new_run_id(),
            _plan_record(new_run_id()),
            execution_items=[_sample_execution_item(approval_reason=123)],
        )


def test_item_metadata_must_be_dict():
    with pytest.raises(ValueError, match="item metadata must be a dict"):
        _execution_request_record(
            new_run_id(),
            _plan_record(new_run_id()),
            execution_items=[_sample_execution_item(metadata="bad")],
        )


def test_notes_must_be_string_or_none():
    run_id = new_run_id()
    plan = _plan_record(run_id)
    record = _execution_request_record(run_id, plan)
    record["notes"] = 1
    with pytest.raises(ValueError, match="notes must be a string or null"):
        schemas.validate(record, "run_execution_request_record")


def test_metadata_must_be_dict():
    with pytest.raises(ValueError, match="metadata must be a dict"):
        _execution_request_record(
            new_run_id(), _plan_record(new_run_id()), metadata=[]
        )


# --- B. Factory tests ---


def test_make_run_execution_request_record_returns_valid_schema():
    run_id = new_run_id()
    plan = _plan_record(run_id)
    schemas.validate(
        _execution_request_record(run_id, plan), "run_execution_request_record"
    )


def test_make_run_execution_request_record_metadata_defaults_to_empty_dict():
    run_id = new_run_id()
    plan = _plan_record(run_id)
    assert _execution_request_record(run_id, plan)["metadata"] == {}


def test_make_run_execution_request_record_notes_remain_none():
    run_id = new_run_id()
    plan = _plan_record(run_id)
    assert _execution_request_record(run_id, plan)["notes"] is None


def test_make_run_execution_request_record_normalizes_item_metadata():
    run_id = new_run_id()
    plan = _plan_record(run_id)
    record = contracts.make_run_execution_request_record(
        run_id=run_id,
        source_followup_plan_fingerprint=contracts.run_followup_plan_fingerprint(plan),
        execution_items=[
            {
                "item_id": "e1",
                "source_followup_item_id": "f1",
                "title": "t",
                "execution_kind": "other",
                "command": {},
            }
        ],
    )
    assert record["execution_items"][0]["metadata"] == {}


def test_factory_does_not_write_files(tmp_path):
    run_id = new_run_id()
    plan = _plan_record(run_id)
    with patch("htr.contracts.atomic_write_json") as mock_write:
        _execution_request_record(run_id, plan)
    mock_write.assert_not_called()


def test_factory_does_not_create_events(tmp_path):
    run_id, _, _, _, _, plan_record = _run_with_planned_run(tmp_path)
    before = len(events.read_task_events(run_id, base_dir=tmp_path))
    _execution_request_record(run_id, plan_record)
    assert len(events.read_task_events(run_id, base_dir=tmp_path)) == before


# --- C. Fingerprint tests ---


def test_fingerprint_validates_schema_first():
    run_id = new_run_id()
    plan = _plan_record(run_id)
    record = _execution_request_record(run_id, plan)
    record["metadata"] = []
    with pytest.raises(ValueError, match="metadata must be a dict"):
        contracts.run_execution_request_fingerprint(record)


def test_equivalent_records_same_fingerprint():
    run_id = new_run_id()
    plan = _plan_record(run_id)
    a = _execution_request_record(run_id, plan, created_at="2026-07-18T08:00:00+00:00")
    b = dict(a)
    assert contracts.run_execution_request_fingerprint(
        a
    ) == contracts.run_execution_request_fingerprint(b)


def test_changed_execution_items_changes_fingerprint():
    run_id = new_run_id()
    plan = _plan_record(run_id)
    first = _execution_request_record(run_id, plan)
    second = _execution_request_record(
        run_id,
        plan,
        execution_items=[_sample_execution_item(item_id="exec-2")],
    )
    assert contracts.run_execution_request_fingerprint(
        first
    ) != contracts.run_execution_request_fingerprint(second)


def test_changed_metadata_changes_fingerprint():
    run_id = new_run_id()
    plan = _plan_record(run_id)
    first = _execution_request_record(run_id, plan, metadata={"a": 1})
    second = _execution_request_record(run_id, plan, metadata={"a": 2})
    assert contracts.run_execution_request_fingerprint(
        first
    ) != contracts.run_execution_request_fingerprint(second)


def test_fingerprint_uses_canonical_json():
    run_id = new_run_id()
    plan = _plan_record(run_id)
    record = _execution_request_record(run_id, plan, metadata={"b": 2, "a": 1})
    expected = json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    assert contracts.run_execution_request_fingerprint(record) == expected


def test_fingerprint_does_not_read_files():
    run_id = new_run_id()
    plan = _plan_record(run_id)
    record = _execution_request_record(run_id, plan)
    with patch("htr.io.read_json") as mock_read:
        contracts.run_execution_request_fingerprint(record)
    mock_read.assert_not_called()


# --- D. Lifecycle success ---


def test_request_run_execution_succeeds_after_plan_gate(tmp_path):
    run_id, _, _, _, _, plan_record = _run_with_planned_run(tmp_path)
    record = _execution_request_record(run_id, plan_record)
    returned = events.request_run_execution(run_id, record, base_dir=tmp_path)
    assert returned == record


def test_source_followup_plan_fingerprint_must_match(tmp_path):
    run_id, _, _, _, _, plan_record = _run_with_planned_run(tmp_path)
    record = _execution_request_record(
        run_id, plan_record, source_followup_plan_fingerprint="wrong"
    )
    with pytest.raises(InvalidTransition, match="source_followup_plan_fingerprint"):
        events.request_run_execution(run_id, record, base_dir=tmp_path)


def test_request_run_execution_writes_record_file(tmp_path):
    run_id, _, _, _, _, plan_record = _run_with_planned_run(tmp_path)
    record = _execution_request_record(run_id, plan_record)
    events.request_run_execution(run_id, record, base_dir=tmp_path)
    assert io.read_json(
        contracts.run_execution_request_record_json_path(run_id, tmp_path)
    ) == record


def test_request_run_execution_appends_event(tmp_path):
    run_id, _, _, _, _, plan_record = _run_with_planned_run(tmp_path)
    record = _execution_request_record(run_id, plan_record)
    before = len(events.read_task_events(run_id, base_dir=tmp_path))
    events.request_run_execution(run_id, record, base_dir=tmp_path)
    rows = events.read_task_events(run_id, base_dir=tmp_path)
    assert len(rows) == before + 1
    event = rows[-1]
    assert event["event_type"] == events.EVENT_TYPE_RUN_EXECUTION_REQUESTED
    assert "task_id" not in event
    assert "attempt_id" not in event


def test_request_run_execution_event_payload_fields(tmp_path):
    run_id, _, _, _, _, plan_record = _run_with_planned_run(tmp_path)
    record = _execution_request_record(run_id, plan_record, requester="assistant")
    events.request_run_execution(run_id, record, actor="operator", base_dir=tmp_path)
    event = events.read_task_events(run_id, base_dir=tmp_path)[-1]
    payload = event["payload"]
    assert payload["run_id"] == run_id
    assert payload["requester"] == "assistant"
    assert payload["request_status"] == contracts.EXECUTION_REQUEST_PENDING
    assert payload["source_followup_plan_fingerprint"] == contracts.run_followup_plan_fingerprint(
        plan_record
    )
    assert payload["run_execution_request_fingerprint"] == contracts.run_execution_request_fingerprint(
        record
    )
    assert payload["run_execution_request_record_path"].endswith(
        "run_execution_request_record.json"
    )


# --- E. Preconditions ---


def test_invalid_run_id_fails_no_side_effects(tmp_path):
    run_id, task_ids, _, _, _, plan_record = _run_with_planned_run(tmp_path)
    before = _snapshot(tmp_path, run_id, task_ids)
    with pytest.raises(ValueError):
        events.request_run_execution("bad", _execution_request_record(run_id, plan_record), base_dir=tmp_path)
    assert _snapshot(tmp_path, run_id, task_ids) == before


def test_invalid_schema_fails_no_side_effects(tmp_path):
    run_id, task_ids, _, _, _, plan_record = _run_with_planned_run(tmp_path)
    record = _execution_request_record(run_id, plan_record)
    record["metadata"] = "bad"
    before = _snapshot(tmp_path, run_id, task_ids)
    with pytest.raises(ValueError):
        events.request_run_execution(run_id, record, base_dir=tmp_path)
    assert _snapshot(tmp_path, run_id, task_ids) == before


def test_run_id_mismatch_fails_no_side_effects(tmp_path):
    run_id, task_ids, _, _, _, plan_record = _run_with_planned_run(tmp_path)
    before = _snapshot(tmp_path, run_id, task_ids)
    with pytest.raises(ValueError, match="run_id does not match"):
        events.request_run_execution(
            run_id,
            _execution_request_record(new_run_id(), plan_record),
            base_dir=tmp_path,
        )
    assert _snapshot(tmp_path, run_id, task_ids) == before


def test_missing_run_manifest_fails_no_side_effects(tmp_path):
    run_id = new_run_id()
    plan = _plan_record(run_id)
    before_events = []
    with pytest.raises(InvalidTransition):
        events.request_run_execution(
            run_id,
            _execution_request_record(run_id, plan),
            base_dir=tmp_path,
        )
    assert events.read_task_events(run_id, base_dir=tmp_path) == before_events


def test_run_not_completed_fails_no_side_effects(tmp_path):
    run_id = new_run_id()
    io.create_run_workspace(run_id, base_dir=tmp_path)
    plan = _plan_record(run_id)
    before = _snapshot(tmp_path, run_id, [])
    with pytest.raises(InvalidTransition, match="is not completed"):
        events.request_run_execution(
            run_id,
            _execution_request_record(run_id, plan),
            base_dir=tmp_path,
        )
    assert _snapshot(tmp_path, run_id, []) == before


def test_missing_completion_record_fails_no_side_effects(tmp_path):
    run_id, task_ids, _, _, _, plan_record = _run_with_planned_run(tmp_path)
    contracts.run_completion_record_json_path(run_id, tmp_path).unlink()
    before = _snapshot(tmp_path, run_id, task_ids)
    with pytest.raises(InvalidTransition, match="run_completion_record.json is missing"):
        events.request_run_execution(
            run_id,
            _execution_request_record(run_id, plan_record),
            base_dir=tmp_path,
        )
    assert _snapshot(tmp_path, run_id, task_ids) == before


def test_missing_review_record_fails_no_side_effects(tmp_path):
    run_id, task_ids, _, _, _, plan_record = _run_with_planned_run(tmp_path)
    contracts.run_review_record_json_path(run_id, tmp_path).unlink()
    before = _snapshot(tmp_path, run_id, task_ids)
    with pytest.raises(InvalidTransition, match="run_review_record.json is missing"):
        events.request_run_execution(
            run_id,
            _execution_request_record(run_id, plan_record),
            base_dir=tmp_path,
        )
    assert _snapshot(tmp_path, run_id, task_ids) == before


def test_missing_followup_plan_record_fails_no_side_effects(tmp_path):
    run_id, task_ids, _, _, _ = _run_with_reviewed_run(tmp_path)
    plan_record = _plan_record(run_id)
    before = _snapshot(tmp_path, run_id, task_ids)
    with pytest.raises(InvalidTransition, match="run_followup_plan_record.json is missing"):
        events.request_run_execution(
            run_id,
            _execution_request_record(run_id, plan_record),
            base_dir=tmp_path,
        )
    assert _snapshot(tmp_path, run_id, task_ids) == before


def test_source_followup_plan_fingerprint_mismatch_fails_no_side_effects(tmp_path):
    run_id, task_ids, _, _, _, plan_record = _run_with_planned_run(tmp_path)
    record = _execution_request_record(
        run_id, plan_record, source_followup_plan_fingerprint="mismatch"
    )
    before = _snapshot(tmp_path, run_id, task_ids)
    with pytest.raises(InvalidTransition, match="source_followup_plan_fingerprint"):
        events.request_run_execution(run_id, record, base_dir=tmp_path)
    assert _snapshot(tmp_path, run_id, task_ids) == before


# --- F. Write order ---


def test_write_order_record_before_event(tmp_path):
    run_id, _, _, _, _, plan_record = _run_with_planned_run(tmp_path)
    record = _execution_request_record(run_id, plan_record)
    ops = []
    real_append = events.append_run_event

    def track_write(path, data):
        if str(path).endswith("run_execution_request_record.json"):
            ops.append("write_record")
        return io.atomic_write_json(path, data)

    def track_append(r, e, base_dir=None):
        ops.append("append_event")
        return real_append(r, e, base_dir)

    with patch("htr.events.atomic_write_json", side_effect=track_write), patch(
        "htr.events.append_run_event", side_effect=track_append
    ):
        events.request_run_execution(run_id, record, base_dir=tmp_path)
    assert ops == ["write_record", "append_event"]


# --- G. Replay-only ---


def test_existing_execution_request_event_id_none_raises(tmp_path):
    run_id, task_ids, _, _, _, plan_record = _run_with_planned_run(tmp_path)
    record = _execution_request_record(run_id, plan_record)
    events.request_run_execution(
        run_id, record, event_id=new_event_id(), base_dir=tmp_path
    )
    before = _snapshot(tmp_path, run_id, task_ids)
    with pytest.raises(InvalidTransition):
        events.request_run_execution(run_id, record, base_dir=tmp_path)
    assert _snapshot(tmp_path, run_id, task_ids) == before


def test_existing_execution_request_missing_event_raises(tmp_path):
    run_id, task_ids, _, _, _, plan_record = _run_with_planned_run(tmp_path)
    record = _execution_request_record(run_id, plan_record)
    io.atomic_write_json(
        contracts.run_execution_request_record_json_path(run_id, tmp_path), record
    )
    before = _snapshot(tmp_path, run_id, task_ids)
    with pytest.raises(InvalidTransition):
        events.request_run_execution(
            run_id, record, event_id=new_event_id(), base_dir=tmp_path
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
                    "run_execution_request_fingerprint": "different",
                },
            },
            EventConflict,
        ),
        (
            lambda e: {
                **e,
                "payload": {
                    **e["payload"],
                    "source_followup_plan_fingerprint": "different",
                },
            },
            EventConflict,
        ),
        (
            lambda e: {
                **e,
                "payload": {
                    **e["payload"],
                    "request_status": contracts.EXECUTION_REQUEST_CANCELLED,
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
def test_existing_execution_request_replay_semantic_mismatch(tmp_path, mutator, expected):
    run_id, task_ids, _, _, _, plan_record = _run_with_planned_run(tmp_path)
    record = _execution_request_record(run_id, plan_record)
    event_id = new_event_id()
    events.request_run_execution(
        run_id, record, event_id=event_id, actor="human", base_dir=tmp_path
    )
    event = events.read_task_events(run_id, base_dir=tmp_path)[-1]
    bad_event = mutator(event)
    before = _snapshot(tmp_path, run_id, task_ids)
    with patch(
        "htr.events._find_run_event_by_id",
        return_value=bad_event,
    ):
        with pytest.raises(expected):
            events.request_run_execution(
                run_id,
                record,
                event_id=event_id,
                actor="human",
                base_dir=tmp_path,
            )
    assert _snapshot(tmp_path, run_id, task_ids) == before


def test_existing_execution_request_same_event_id_same_semantic_returns_existing(tmp_path):
    run_id, task_ids, _, _, _, plan_record = _run_with_planned_run(tmp_path)
    record = _execution_request_record(run_id, plan_record)
    event_id = new_event_id()
    events.request_run_execution(
        run_id, record, event_id=event_id, actor="human", base_dir=tmp_path
    )
    before = _snapshot(tmp_path, run_id, task_ids)
    returned = events.request_run_execution(
        run_id, record, event_id=event_id, actor="human", base_dir=tmp_path
    )
    assert returned == record
    assert _snapshot(tmp_path, run_id, task_ids) == before


def test_replay_only_no_writes_or_status_updates(tmp_path):
    run_id, task_ids, attempt_ids, _, _, plan_record = _run_with_planned_run(tmp_path)
    record = _execution_request_record(run_id, plan_record)
    event_id = new_event_id()
    events.request_run_execution(
        run_id, record, event_id=event_id, actor="human", base_dir=tmp_path
    )
    manifest_before = io.read_json(paths.run_manifest_path(run_id, tmp_path))
    task_before = io.read_json(paths.task_status_path(run_id, task_ids[0], tmp_path))
    events.request_run_execution(
        run_id, record, event_id=event_id, actor="human", base_dir=tmp_path
    )
    assert io.read_json(paths.run_manifest_path(run_id, tmp_path)) == manifest_before
    assert io.read_json(paths.task_status_path(run_id, task_ids[0], tmp_path)) == task_before
    assert len(list(paths.tasks_dir(run_id, tmp_path).iterdir())) == len(task_ids)


# --- H. Normal idempotency ---


def test_idempotent_same_event_id_same_semantic(tmp_path):
    run_id, _, _, _, _, plan_record = _run_with_planned_run(tmp_path)
    record = _execution_request_record(run_id, plan_record)
    event_id = new_event_id()
    first = events.request_run_execution(
        run_id, record, event_id=event_id, base_dir=tmp_path
    )
    second = events.request_run_execution(
        run_id, record, event_id=event_id, base_dir=tmp_path
    )
    assert second == first


def test_same_event_id_different_semantic_raises_conflict(tmp_path):
    run_id, _, _, _, _, plan_record = _run_with_planned_run(tmp_path)
    event_id = new_event_id()
    events.request_run_execution(
        run_id,
        _execution_request_record(run_id, plan_record, requester="first"),
        event_id=event_id,
        base_dir=tmp_path,
    )
    with pytest.raises(EventConflict):
        events.request_run_execution(
            run_id,
            _execution_request_record(run_id, plan_record, requester="second"),
            event_id=event_id,
            base_dir=tmp_path,
        )


def test_idempotency_requires_followup_plan_record(tmp_path):
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
    with pytest.raises(InvalidTransition, match="run_followup_plan_record.json is missing"):
        events.request_run_execution(
            run_id,
            _execution_request_record(run_id, plan),
            event_id=new_event_id(),
            base_dir=tmp_path,
        )


def test_idempotency_does_not_allow_non_completed_run(tmp_path):
    run_id = new_run_id()
    io.create_run_workspace(run_id, base_dir=tmp_path)
    plan = _plan_record(run_id)
    with pytest.raises(InvalidTransition):
        events.request_run_execution(
            run_id,
            _execution_request_record(run_id, plan),
            event_id=new_event_id(),
            base_dir=tmp_path,
        )


# --- I. Boundary preservation ---


def test_boundaries_preserved_on_success(tmp_path):
    run_id, task_ids, attempt_ids, _, _, plan_record = _run_with_planned_run(tmp_path)
    before = _snapshot(tmp_path, run_id, task_ids)
    record = _execution_request_record(run_id, plan_record)
    events.request_run_execution(run_id, record, base_dir=tmp_path)
    after = _snapshot(tmp_path, run_id, task_ids)
    assert after["manifest"] == before["manifest"]
    assert after["completion"] == before["completion"]
    assert after["review"] == before["review"]
    assert after["followup"] == before["followup"]
    assert after["task_statuses"] == before["task_statuses"]
    assert after["execution_exists"] is True
    assert len(list(paths.tasks_dir(run_id, tmp_path).iterdir())) == len(task_ids)
    task_id = task_ids[0]
    attempt_id = attempt_ids[0]
    result_path = paths.result_json_path(run_id, task_id, attempt_id, tmp_path)
    verification_path = contracts.verification_result_json_path(
        run_id, task_id, attempt_id, tmp_path
    )
    assert result_path.exists()
    assert verification_path.exists()


# --- J. Import boundary ---


def test_task9_modules_do_not_import_forbidden_modules():
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


def test_package_exports_task9_apis():
    import htr

    for name in (
        "make_run_execution_request_record",
        "run_execution_request_fingerprint",
        "request_run_execution",
        "EVENT_TYPE_RUN_EXECUTION_REQUESTED",
        "EXECUTION_REQUEST_PENDING",
        "EXECUTION_REQUEST_CANCELLED",
    ):
        assert hasattr(htr, name)
