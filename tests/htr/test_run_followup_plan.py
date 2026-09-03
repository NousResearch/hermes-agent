"""Tests for Task 8 — Review-Gated Follow-up Planning API."""

import ast
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from htr import contracts, events, io, paths, schemas
from htr.ids import new_attempt_id, new_event_id, new_run_id, new_task_id, validate_id
from htr.state import (
    ATTEMPT_RUNNING,
    RUN_COMPLETED,
    RUN_CREATED,
    TASK_COMPLETED,
    TASK_RUNNING,
    EventConflict,
    InvalidTransition,
)


def _sample_item(**overrides):
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


def _plan_record(run_id, decision=contracts.RUN_REVIEW_NEEDS_FOLLOWUP, **kwargs):
    followup_items = kwargs.pop("followup_items", [_sample_item()])
    summary = kwargs.pop("summary", "Follow up after review")
    return contracts.make_run_followup_plan_record(
        run_id=run_id,
        source_review_decision=decision,
        summary=summary,
        followup_items=followup_items,
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


def _snapshot(tmp_path, run_id, task_ids, attempt_ids=None):
    if attempt_ids is None:
        attempt_ids = []
    snap = {
        "events_len": len(events.read_task_events(run_id, base_dir=tmp_path)),
        "manifest": io.read_json(paths.run_manifest_path(run_id, tmp_path)),
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
    snap["completion"] = (
        completion_path.read_bytes() if completion_path.exists() else None
    )
    snap["review"] = review_path.read_bytes() if review_path.exists() else None
    return snap


# --- A. Schema tests ---


def test_valid_run_followup_plan_record_passes():
    run_id = new_run_id()
    schemas.validate(_plan_record(run_id), "run_followup_plan_record")


@pytest.mark.parametrize("missing", ["run_id", "summary", "followup_items", "planner"])
def test_missing_required_fields_fail(missing):
    record = _plan_record(new_run_id())
    del record[missing]
    with pytest.raises(ValueError, match="missing fields"):
        schemas.validate(record, "run_followup_plan_record")


def test_invalid_run_id_fails_schema():
    record = _plan_record(new_run_id())
    record["run_id"] = "not-a-run-id"
    with pytest.raises(ValueError, match="run_id must be a valid run id"):
        schemas.validate(record, "run_followup_plan_record")


def test_invalid_source_review_decision_fails():
    with pytest.raises(ValueError, match="source_review_decision"):
        _plan_record(new_run_id(), decision="maybe")


def test_invalid_planner_fails():
    with pytest.raises(ValueError, match="planner must be a non-empty string"):
        _plan_record(new_run_id(), planner="")


def test_invalid_plan_status_fails():
    with pytest.raises(ValueError, match="plan_status"):
        _plan_record(new_run_id(), plan_status="draft")


def test_invalid_summary_fails():
    with pytest.raises(ValueError, match="summary must be a non-empty string"):
        _plan_record(new_run_id(), summary="")


def test_followup_items_must_be_list():
    record = _plan_record(new_run_id())
    record["followup_items"] = {}
    with pytest.raises(ValueError, match="followup_items must be a list"):
        schemas.validate(record, "run_followup_plan_record")


def test_followup_items_entries_must_be_dicts():
    record = _plan_record(new_run_id())
    record["followup_items"] = ["bad"]
    with pytest.raises(ValueError, match="each followup item must be a dict"):
        schemas.validate(record, "run_followup_plan_record")


@pytest.mark.parametrize("field", ["item_id", "title", "kind", "proposed_action"])
def test_followup_item_missing_required_field_fails(field):
    item = _sample_item()
    del item[field]
    record = _plan_record(new_run_id())
    record["followup_items"] = [item]
    with pytest.raises(ValueError, match=f"followup item {field}"):
        schemas.validate(record, "run_followup_plan_record")


def test_invalid_item_kind_fails():
    with pytest.raises(ValueError, match="kind is invalid"):
        contracts.make_run_followup_plan_record(
            run_id=new_run_id(),
            source_review_decision=contracts.RUN_REVIEW_NEEDS_FOLLOWUP,
            summary="s",
            followup_items=[_sample_item(kind="auto_execute")],
        )


def test_item_rationale_must_be_string_or_none():
    with pytest.raises(ValueError, match="rationale must be a string or null"):
        contracts.make_run_followup_plan_record(
            run_id=new_run_id(),
            source_review_decision=contracts.RUN_REVIEW_NEEDS_FOLLOWUP,
            summary="s",
            followup_items=[_sample_item(rationale=123)],
        )


def test_item_metadata_must_be_dict():
    with pytest.raises(ValueError, match="item metadata must be a dict"):
        contracts.make_run_followup_plan_record(
            run_id=new_run_id(),
            source_review_decision=contracts.RUN_REVIEW_NEEDS_FOLLOWUP,
            summary="s",
            followup_items=[_sample_item(metadata="bad")],
        )


def test_notes_must_be_string_or_none():
    record = _plan_record(new_run_id())
    record["notes"] = 1
    with pytest.raises(ValueError, match="notes must be a string or null"):
        schemas.validate(record, "run_followup_plan_record")


def test_metadata_must_be_dict():
    with pytest.raises(ValueError, match="metadata must be a dict"):
        _plan_record(new_run_id(), metadata=[])


# --- B. Factory tests ---


def test_make_run_followup_plan_record_returns_valid_schema():
    schemas.validate(_plan_record(new_run_id()), "run_followup_plan_record")


def test_make_run_followup_plan_record_metadata_defaults_to_empty_dict():
    assert _plan_record(new_run_id())["metadata"] == {}


def test_make_run_followup_plan_record_notes_remain_none():
    assert _plan_record(new_run_id())["notes"] is None


def test_make_run_followup_plan_record_normalizes_item_metadata():
    record = contracts.make_run_followup_plan_record(
        run_id=new_run_id(),
        source_review_decision=contracts.RUN_REVIEW_NEEDS_FOLLOWUP,
        summary="s",
        followup_items=[{"item_id": "i1", "title": "t", "kind": "other", "proposed_action": "a"}],
    )
    assert record["followup_items"][0]["metadata"] == {}


def test_factory_does_not_write_files(tmp_path):
    run_id = new_run_id()
    with patch("htr.contracts.atomic_write_json") as mock_write:
        _plan_record(run_id)
    mock_write.assert_not_called()


def test_factory_does_not_create_events(tmp_path):
    run_id, task_ids, _, _, _ = _run_with_reviewed_run(tmp_path)
    before = len(events.read_task_events(run_id, base_dir=tmp_path))
    _plan_record(run_id)
    assert len(events.read_task_events(run_id, base_dir=tmp_path)) == before


def test_planner_may_be_assistant_or_tool():
    run_id = new_run_id()
    for planner in ("human", "assistant", "planning-tool-v1"):
        record = _plan_record(run_id, planner=planner)
        assert record["planner"] == planner


# --- C. Fingerprint tests ---


def test_fingerprint_validates_schema_first():
    record = _plan_record(new_run_id())
    record["metadata"] = []
    with pytest.raises(ValueError, match="metadata must be a dict"):
        contracts.run_followup_plan_fingerprint(record)


def test_equivalent_records_same_fingerprint():
    run_id = new_run_id()
    a = _plan_record(run_id, created_at="2026-07-18T08:00:00+00:00")
    b = dict(a)
    assert contracts.run_followup_plan_fingerprint(a) == contracts.run_followup_plan_fingerprint(
        b
    )


def test_changed_summary_changes_fingerprint():
    run_id = new_run_id()
    first = _plan_record(run_id, summary="one")
    second = _plan_record(run_id, summary="two")
    assert contracts.run_followup_plan_fingerprint(first) != contracts.run_followup_plan_fingerprint(
        second
    )


def test_changed_followup_items_changes_fingerprint():
    run_id = new_run_id()
    first = _plan_record(run_id)
    second = _plan_record(
        run_id,
        followup_items=[_sample_item(item_id="followup-2")],
    )
    assert contracts.run_followup_plan_fingerprint(first) != contracts.run_followup_plan_fingerprint(
        second
    )


def test_changed_metadata_changes_fingerprint():
    run_id = new_run_id()
    first = _plan_record(run_id, metadata={"a": 1})
    second = _plan_record(run_id, metadata={"a": 2})
    assert contracts.run_followup_plan_fingerprint(first) != contracts.run_followup_plan_fingerprint(
        second
    )


def test_fingerprint_uses_canonical_json():
    run_id = new_run_id()
    record = _plan_record(run_id, metadata={"b": 2, "a": 1})
    expected = json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    assert contracts.run_followup_plan_fingerprint(record) == expected


def test_fingerprint_does_not_read_files():
    record = _plan_record(new_run_id())
    with patch("htr.io.read_json") as mock_read:
        contracts.run_followup_plan_fingerprint(record)
    mock_read.assert_not_called()


# --- D. Lifecycle success ---


def test_plan_run_followup_succeeds_after_review_gate(tmp_path):
    run_id, _, _, _, _ = _run_with_reviewed_run(tmp_path)
    record = _plan_record(run_id)
    returned = events.plan_run_followup(run_id, record, base_dir=tmp_path)
    assert returned == record


def test_plan_run_followup_writes_record_file(tmp_path):
    run_id, _, _, _, _ = _run_with_reviewed_run(tmp_path)
    record = _plan_record(run_id)
    events.plan_run_followup(run_id, record, base_dir=tmp_path)
    assert io.read_json(
        contracts.run_followup_plan_record_json_path(run_id, tmp_path)
    ) == record


def test_plan_run_followup_appends_event(tmp_path):
    run_id, _, _, _, _ = _run_with_reviewed_run(tmp_path)
    record = _plan_record(run_id)
    before = len(events.read_task_events(run_id, base_dir=tmp_path))
    events.plan_run_followup(run_id, record, base_dir=tmp_path)
    rows = events.read_task_events(run_id, base_dir=tmp_path)
    assert len(rows) == before + 1
    event = rows[-1]
    assert event["event_type"] == events.EVENT_TYPE_MANUAL_RUN_FOLLOWUP_PLANNED
    assert "task_id" not in event
    assert "attempt_id" not in event


def test_plan_run_followup_event_payload_fields(tmp_path):
    run_id, _, _, _, _ = _run_with_reviewed_run(tmp_path)
    record = _plan_record(run_id, planner="assistant")
    events.plan_run_followup(run_id, record, actor="operator", base_dir=tmp_path)
    event = events.read_task_events(run_id, base_dir=tmp_path)[-1]
    payload = event["payload"]
    assert payload["run_id"] == run_id
    assert payload["source_review_decision"] == contracts.RUN_REVIEW_NEEDS_FOLLOWUP
    assert payload["plan_status"] == contracts.FOLLOWUP_PLAN_OPEN
    assert payload["planner"] == "assistant"
    assert payload["run_followup_plan_fingerprint"] == contracts.run_followup_plan_fingerprint(
        record
    )
    assert payload["run_followup_plan_record_path"].endswith("run_followup_plan_record.json")


# --- E. Preconditions ---


def test_invalid_run_id_fails_no_side_effects(tmp_path):
    run_id, task_ids, _, _, _ = _run_with_reviewed_run(tmp_path)
    before = _snapshot(tmp_path, run_id, task_ids)
    with pytest.raises(ValueError):
        events.plan_run_followup("bad", _plan_record(run_id), base_dir=tmp_path)
    assert _snapshot(tmp_path, run_id, task_ids) == before


def test_invalid_schema_fails_no_side_effects(tmp_path):
    run_id, task_ids, _, _, _ = _run_with_reviewed_run(tmp_path)
    record = _plan_record(run_id)
    record["metadata"] = "bad"
    before = _snapshot(tmp_path, run_id, task_ids)
    with pytest.raises(ValueError):
        events.plan_run_followup(run_id, record, base_dir=tmp_path)
    assert _snapshot(tmp_path, run_id, task_ids) == before


def test_run_id_mismatch_fails_no_side_effects(tmp_path):
    run_id, task_ids, _, _, _ = _run_with_reviewed_run(tmp_path)
    before = _snapshot(tmp_path, run_id, task_ids)
    with pytest.raises(ValueError, match="run_id does not match"):
        events.plan_run_followup(run_id, _plan_record(new_run_id()), base_dir=tmp_path)
    assert _snapshot(tmp_path, run_id, task_ids) == before


def test_missing_run_manifest_fails_no_side_effects(tmp_path):
    run_id = new_run_id()
    before_events = []
    with pytest.raises(InvalidTransition):
        events.plan_run_followup(run_id, _plan_record(run_id), base_dir=tmp_path)
    assert events.read_task_events(run_id, base_dir=tmp_path) == before_events


def test_run_not_completed_fails_no_side_effects(tmp_path):
    run_id = new_run_id()
    io.create_run_workspace(run_id, base_dir=tmp_path)
    before = _snapshot(tmp_path, run_id, [])
    with pytest.raises(InvalidTransition, match="is not completed"):
        events.plan_run_followup(run_id, _plan_record(run_id), base_dir=tmp_path)
    assert _snapshot(tmp_path, run_id, []) == before


def test_missing_completion_record_fails_no_side_effects(tmp_path):
    run_id, task_ids, _, _, _ = _run_with_reviewed_run(tmp_path)
    contracts.run_completion_record_json_path(run_id, tmp_path).unlink()
    before = _snapshot(tmp_path, run_id, task_ids)
    with pytest.raises(InvalidTransition, match="run_completion_record.json is missing"):
        events.plan_run_followup(run_id, _plan_record(run_id), base_dir=tmp_path)
    assert _snapshot(tmp_path, run_id, task_ids) == before


def test_missing_review_record_fails_no_side_effects(tmp_path):
    run_id, task_ids, _, _, _ = _run_with_reviewed_run(tmp_path)
    contracts.run_review_record_json_path(run_id, tmp_path).unlink()
    before = _snapshot(tmp_path, run_id, task_ids)
    with pytest.raises(InvalidTransition, match="run_review_record.json is missing"):
        events.plan_run_followup(run_id, _plan_record(run_id), base_dir=tmp_path)
    assert _snapshot(tmp_path, run_id, task_ids) == before


def test_source_review_decision_mismatch_fails_no_side_effects(tmp_path):
    run_id, task_ids, _, _, _ = _run_with_reviewed_run(
        tmp_path, review_decision=contracts.RUN_REVIEW_NEEDS_FOLLOWUP
    )
    before = _snapshot(tmp_path, run_id, task_ids)
    with pytest.raises(InvalidTransition, match="source_review_decision"):
        events.plan_run_followup(
            run_id,
            _plan_record(run_id, decision=contracts.RUN_REVIEW_ACCEPTED),
            base_dir=tmp_path,
        )
    assert _snapshot(tmp_path, run_id, task_ids) == before


# --- F. Write order ---


def test_write_order_record_before_event(tmp_path):
    run_id, _, _, _, _ = _run_with_reviewed_run(tmp_path)
    record = _plan_record(run_id)
    ops = []
    real_append = events.append_run_event

    def track_write(path, data):
        if str(path).endswith("run_followup_plan_record.json"):
            ops.append("write_record")
        return io.atomic_write_json(path, data)

    def track_append(r, e, base_dir=None):
        ops.append("append_event")
        return real_append(r, e, base_dir)

    with patch("htr.events.atomic_write_json", side_effect=track_write), patch(
        "htr.events.append_run_event", side_effect=track_append
    ):
        events.plan_run_followup(run_id, record, base_dir=tmp_path)
    assert ops == ["write_record", "append_event"]


# --- G. Replay-only ---


def test_existing_plan_event_id_none_raises(tmp_path):
    run_id, task_ids, _, _, _ = _run_with_reviewed_run(tmp_path)
    record = _plan_record(run_id)
    events.plan_run_followup(run_id, record, event_id=new_event_id(), base_dir=tmp_path)
    before = _snapshot(tmp_path, run_id, task_ids)
    with pytest.raises(InvalidTransition):
        events.plan_run_followup(run_id, record, base_dir=tmp_path)
    assert _snapshot(tmp_path, run_id, task_ids) == before


def test_existing_plan_missing_event_raises(tmp_path):
    run_id, task_ids, _, _, _ = _run_with_reviewed_run(tmp_path)
    record = _plan_record(run_id)
    io.atomic_write_json(
        contracts.run_followup_plan_record_json_path(run_id, tmp_path), record
    )
    before = _snapshot(tmp_path, run_id, task_ids)
    with pytest.raises(InvalidTransition):
        events.plan_run_followup(
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
                    "run_followup_plan_fingerprint": "different",
                },
            },
            EventConflict,
        ),
        (
            lambda e: {
                **e,
                "payload": {**e["payload"], "source_review_decision": "accepted"},
            },
            EventConflict,
        ),
        (
            lambda e: {
                **e,
                "payload": {**e["payload"], "plan_status": "cancelled"},
            },
            EventConflict,
        ),
        (
            lambda e: {**e, "payload": {**e["payload"], "planner": "other"}},
            EventConflict,
        ),
    ],
)
def test_existing_plan_replay_semantic_mismatch(tmp_path, mutator, expected):
    run_id, task_ids, _, _, _ = _run_with_reviewed_run(tmp_path)
    record = _plan_record(run_id)
    event_id = new_event_id()
    events.plan_run_followup(run_id, record, event_id=event_id, base_dir=tmp_path)
    event = events.read_task_events(run_id, base_dir=tmp_path)[-1]
    bad_event = mutator(event)
    before = _snapshot(tmp_path, run_id, task_ids)
    with patch(
        "htr.events._find_run_event_by_id",
        return_value=bad_event,
    ):
        with pytest.raises(expected):
            events.plan_run_followup(
                run_id, record, event_id=event_id, actor="human", base_dir=tmp_path
            )
    assert _snapshot(tmp_path, run_id, task_ids) == before


def test_existing_plan_same_event_id_same_semantic_returns_existing(tmp_path):
    run_id, task_ids, _, _, _ = _run_with_reviewed_run(tmp_path)
    record = _plan_record(run_id)
    event_id = new_event_id()
    events.plan_run_followup(
        run_id, record, event_id=event_id, actor="human", base_dir=tmp_path
    )
    before = _snapshot(tmp_path, run_id, task_ids)
    returned = events.plan_run_followup(
        run_id, record, event_id=event_id, actor="human", base_dir=tmp_path
    )
    assert returned == record
    assert _snapshot(tmp_path, run_id, task_ids) == before


def test_replay_only_no_writes_or_status_updates(tmp_path):
    run_id, task_ids, attempt_ids, _, _ = _run_with_reviewed_run(tmp_path)
    record = _plan_record(run_id)
    event_id = new_event_id()
    events.plan_run_followup(
        run_id, record, event_id=event_id, actor="human", base_dir=tmp_path
    )
    manifest_before = io.read_json(paths.run_manifest_path(run_id, tmp_path))
    task_before = io.read_json(paths.task_status_path(run_id, task_ids[0], tmp_path))
    events.plan_run_followup(
        run_id, record, event_id=event_id, actor="human", base_dir=tmp_path
    )
    assert io.read_json(paths.run_manifest_path(run_id, tmp_path)) == manifest_before
    assert io.read_json(paths.task_status_path(run_id, task_ids[0], tmp_path)) == task_before
    assert len(list(paths.tasks_dir(run_id, tmp_path).iterdir())) == len(task_ids)


# --- H. Normal idempotency ---


def test_idempotent_same_event_id_same_semantic(tmp_path):
    run_id, _, _, _, _ = _run_with_reviewed_run(tmp_path)
    record = _plan_record(run_id)
    event_id = new_event_id()
    first = events.plan_run_followup(
        run_id, record, event_id=event_id, base_dir=tmp_path
    )
    second = events.plan_run_followup(
        run_id, record, event_id=event_id, base_dir=tmp_path
    )
    assert second == first
    assert contracts.run_followup_plan_record_json_path(run_id, tmp_path).exists()


def test_idempotent_replay_fails_when_event_exists_but_json_missing(tmp_path):
    run_id, _, _, _, _ = _run_with_reviewed_run(tmp_path)
    record = _plan_record(run_id)
    event_id = new_event_id()
    events.plan_run_followup(run_id, record, event_id=event_id, base_dir=tmp_path)
    plan_path = contracts.run_followup_plan_record_json_path(run_id, tmp_path)
    assert plan_path.exists()
    plan_path.unlink()
    assert not plan_path.exists()
    with pytest.raises(
        InvalidTransition,
        match="run_followup_plan_record.json missing while matching audit event exists",
    ):
        events.plan_run_followup(run_id, record, event_id=event_id, base_dir=tmp_path)
    assert not plan_path.exists()


def test_same_event_id_different_semantic_raises_conflict(tmp_path):
    run_id, _, _, _, _ = _run_with_reviewed_run(tmp_path)
    event_id = new_event_id()
    events.plan_run_followup(
        run_id, _plan_record(run_id, summary="first"), event_id=event_id, base_dir=tmp_path
    )
    with pytest.raises(EventConflict):
        events.plan_run_followup(
            run_id,
            _plan_record(run_id, summary="second"),
            event_id=event_id,
            base_dir=tmp_path,
        )


def test_idempotency_requires_review_record(tmp_path):
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
    with pytest.raises(InvalidTransition, match="run_review_record.json is missing"):
        events.plan_run_followup(
            run_id, _plan_record(run_id), event_id=new_event_id(), base_dir=tmp_path
        )


def test_idempotency_does_not_allow_non_completed_run(tmp_path):
    run_id = new_run_id()
    io.create_run_workspace(run_id, base_dir=tmp_path)
    with pytest.raises(InvalidTransition):
        events.plan_run_followup(
            run_id, _plan_record(run_id), event_id=new_event_id(), base_dir=tmp_path
        )


# --- I. Boundary preservation ---


def test_boundaries_preserved_on_success(tmp_path):
    run_id, task_ids, attempt_ids, _, _ = _run_with_reviewed_run(tmp_path)
    before = _snapshot(tmp_path, run_id, task_ids)
    events.plan_run_followup(run_id, _plan_record(run_id), base_dir=tmp_path)
    after = _snapshot(tmp_path, run_id, task_ids)
    assert after["manifest"] == before["manifest"]
    assert after["completion"] == before["completion"]
    assert after["review"] == before["review"]
    assert after["task_statuses"] == before["task_statuses"]
    assert len(list(paths.tasks_dir(run_id, tmp_path).iterdir())) == len(task_ids)


# --- J. Import boundary ---


def test_task8_modules_do_not_import_forbidden_modules():
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


def test_package_exports_task8_apis():
    import htr

    for name in (
        "make_run_followup_plan_record",
        "run_followup_plan_fingerprint",
        "plan_run_followup",
        "EVENT_TYPE_MANUAL_RUN_FOLLOWUP_PLANNED",
        "FOLLOWUP_PLAN_OPEN",
        "FOLLOWUP_PLAN_CANCELLED",
    ):
        assert hasattr(htr, name)
