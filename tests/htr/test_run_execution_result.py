"""Tests for Task 10 — Controlled One-Shot Execution Adapter."""

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


def _sample_item_result(**overrides):
    result = {
        "item_id": "exec-1",
        "source_followup_item_id": "followup-1",
        "execution_kind": "manual_open_link",
        "item_status": contracts.EXECUTION_ITEM_SKIPPED,
        "output": {"human_action_required": True, "command": {"url": "https://x"}},
        "error": None,
        "metadata": {},
    }
    result.update(overrides)
    return result


def _result_record(run_id, request_fp, **kwargs):
    item_results = kwargs.pop("item_results", [_sample_item_result()])
    return contracts.make_run_execution_result_record(
        run_id=run_id,
        source_execution_request_fingerprint=request_fp,
        item_results=item_results,
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


def _run_with_planned_run(tmp_path, task_count=1):
    run_id, task_ids, attempt_ids, completion_record, review_record = (
        _run_with_reviewed_run(tmp_path, task_count=task_count)
    )
    plan_record = _plan_record(run_id)
    events.plan_run_followup(run_id, plan_record, base_dir=tmp_path)
    return run_id, task_ids, attempt_ids, completion_record, review_record, plan_record


def _run_with_execution_request(tmp_path, execution_items=None):
    run_id, task_ids, attempt_ids, completion, review, plan = _run_with_planned_run(
        tmp_path
    )
    items = execution_items if execution_items is not None else [_sample_execution_item()]
    request_record = _execution_request_record(run_id, plan, execution_items=items)
    events.request_run_execution(run_id, request_record, base_dir=tmp_path)
    return run_id, task_ids, attempt_ids, completion, review, plan, request_record


def _snapshot(tmp_path, run_id, task_ids):
    request_path = contracts.run_execution_request_record_json_path(run_id, tmp_path)
    result_path = contracts.run_execution_result_record_json_path(run_id, tmp_path)
    followup_path = contracts.run_followup_plan_record_json_path(run_id, tmp_path)
    completion_path = contracts.run_completion_record_json_path(run_id, tmp_path)
    review_path = contracts.run_review_record_json_path(run_id, tmp_path)
    return {
        "events_len": len(events.read_task_events(run_id, base_dir=tmp_path)),
        "manifest": io.read_json(paths.run_manifest_path(run_id, tmp_path)),
        "request": request_path.read_bytes() if request_path.exists() else None,
        "result_exists": result_path.exists(),
        "followup": followup_path.read_bytes() if followup_path.exists() else None,
        "completion": completion_path.read_bytes() if completion_path.exists() else None,
        "review": review_path.read_bytes() if review_path.exists() else None,
        "task_statuses": {
            tid: io.read_json(paths.task_status_path(run_id, tid, tmp_path))
            for tid in task_ids
        },
    }


# --- A. Schema ---


def test_valid_run_execution_result_record_passes():
    run_id = new_run_id()
    schemas.validate(
        _result_record(run_id, "fp"), "run_execution_result_record"
    )


@pytest.mark.parametrize(
    "missing",
    [
        "run_id",
        "source_execution_request_fingerprint",
        "executor",
        "result_status",
        "item_results",
        "notes",
        "metadata",
    ],
)
def test_missing_required_fields_fail(missing):
    record = _result_record(new_run_id(), "fp")
    del record[missing]
    with pytest.raises(ValueError, match="missing fields"):
        schemas.validate(record, "run_execution_result_record")


def test_invalid_run_id_fails_schema():
    record = _result_record(new_run_id(), "fp")
    record["run_id"] = "bad"
    with pytest.raises(ValueError, match="run_id must be a valid run id"):
        schemas.validate(record, "run_execution_result_record")


def test_invalid_source_execution_request_fingerprint_fails():
    record = _result_record(new_run_id(), "fp")
    record["source_execution_request_fingerprint"] = ""
    with pytest.raises(ValueError, match="source_execution_request_fingerprint"):
        schemas.validate(record, "run_execution_result_record")


def test_invalid_executor_fails():
    with pytest.raises(ValueError, match="executor must be a non-empty string"):
        _result_record(new_run_id(), "fp", executor="")


def test_invalid_result_status_fails():
    with pytest.raises(ValueError, match="result_status"):
        _result_record(new_run_id(), "fp", result_status="running")


def test_item_results_must_be_list():
    record = _result_record(new_run_id(), "fp")
    record["item_results"] = {}
    with pytest.raises(ValueError, match="item_results must be a list"):
        schemas.validate(record, "run_execution_result_record")


def test_item_results_entries_must_be_dicts():
    record = _result_record(new_run_id(), "fp")
    record["item_results"] = ["bad"]
    with pytest.raises(ValueError, match="each item result must be a dict"):
        schemas.validate(record, "run_execution_result_record")


@pytest.mark.parametrize(
    "field", ["item_id", "source_followup_item_id", "execution_kind", "item_status", "output"]
)
def test_item_result_missing_required_field_fails(field):
    item = _sample_item_result()
    del item[field]
    record = _result_record(new_run_id(), "fp")
    record["item_results"] = [item]
    with pytest.raises(ValueError, match=f"item result {field}"):
        schemas.validate(record, "run_execution_result_record")


def test_invalid_execution_kind_fails():
    with pytest.raises(ValueError, match="execution_kind is invalid"):
        _result_record(
            new_run_id(),
            "fp",
            item_results=[_sample_item_result(execution_kind="auto_run")],
        )


def test_invalid_item_status_fails():
    with pytest.raises(ValueError, match="item_status is invalid"):
        _result_record(
            new_run_id(),
            "fp",
            item_results=[_sample_item_result(item_status="running")],
        )


def test_output_must_be_dict():
    with pytest.raises(ValueError, match="output must be a dict"):
        _result_record(
            new_run_id(),
            "fp",
            item_results=[_sample_item_result(output="bad")],
        )


def test_error_must_be_string_or_none():
    with pytest.raises(ValueError, match="error must be a string or null"):
        _result_record(
            new_run_id(),
            "fp",
            item_results=[_sample_item_result(error=123)],
        )


def test_item_metadata_must_be_dict():
    with pytest.raises(ValueError, match="item result metadata must be a dict"):
        _result_record(
            new_run_id(),
            "fp",
            item_results=[_sample_item_result(metadata="bad")],
        )


def test_notes_must_be_string_or_none():
    record = _result_record(new_run_id(), "fp")
    record["notes"] = 1
    with pytest.raises(ValueError, match="notes must be a string or null"):
        schemas.validate(record, "run_execution_result_record")


def test_metadata_must_be_dict():
    with pytest.raises(ValueError, match="metadata must be a dict"):
        _result_record(new_run_id(), "fp", metadata=[])


# --- B. Factory ---


def test_make_run_execution_result_record_returns_valid_schema():
    schemas.validate(_result_record(new_run_id(), "fp"), "run_execution_result_record")


def test_make_run_execution_result_record_metadata_defaults_to_empty_dict():
    assert _result_record(new_run_id(), "fp")["metadata"] == {}


def test_make_run_execution_result_record_notes_remain_none():
    assert _result_record(new_run_id(), "fp")["notes"] is None


def test_make_run_execution_result_record_normalizes_item_metadata():
    record = contracts.make_run_execution_result_record(
        run_id=new_run_id(),
        source_execution_request_fingerprint="fp",
        item_results=[
            {
                "item_id": "i1",
                "source_followup_item_id": "f1",
                "execution_kind": "other",
                "item_status": contracts.EXECUTION_ITEM_UNSUPPORTED,
                "output": {},
                "error": None,
            }
        ],
    )
    assert record["item_results"][0]["metadata"] == {}


def test_factory_does_not_write_files(tmp_path):
    with patch("htr.contracts.atomic_write_json") as mock_write:
        _result_record(new_run_id(), "fp")
    mock_write.assert_not_called()


def test_factory_does_not_create_events(tmp_path):
    run_id, _, _, _, _, _, _ = _run_with_execution_request(tmp_path)
    before = len(events.read_task_events(run_id, base_dir=tmp_path))
    _result_record(run_id, "fp")
    assert len(events.read_task_events(run_id, base_dir=tmp_path)) == before


# --- C. Fingerprint ---


def test_fingerprint_validates_schema_first():
    record = _result_record(new_run_id(), "fp")
    record["metadata"] = []
    with pytest.raises(ValueError, match="metadata must be a dict"):
        contracts.run_execution_result_fingerprint(record)


def test_equivalent_records_same_fingerprint():
    run_id = new_run_id()
    a = _result_record(run_id, "fp", created_at="2026-07-18T08:00:00+00:00")
    b = dict(a)
    assert contracts.run_execution_result_fingerprint(
        a
    ) == contracts.run_execution_result_fingerprint(b)


def test_changed_item_results_changes_fingerprint():
    first = _result_record(new_run_id(), "fp")
    second = _result_record(
        new_run_id(),
        "fp",
        item_results=[_sample_item_result(item_id="exec-2")],
    )
    assert contracts.run_execution_result_fingerprint(
        first
    ) != contracts.run_execution_result_fingerprint(second)


def test_changed_metadata_changes_fingerprint():
    first = _result_record(new_run_id(), "fp", metadata={"a": 1})
    second = _result_record(new_run_id(), "fp", metadata={"a": 2})
    assert contracts.run_execution_result_fingerprint(
        first
    ) != contracts.run_execution_result_fingerprint(second)


def test_fingerprint_uses_canonical_json():
    record = _result_record(new_run_id(), "fp", metadata={"b": 2, "a": 1})
    expected = json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    assert contracts.run_execution_result_fingerprint(record) == expected


def test_fingerprint_does_not_read_files():
    with patch("htr.io.read_json") as mock_read:
        contracts.run_execution_result_fingerprint(_result_record(new_run_id(), "fp"))
    mock_read.assert_not_called()


# --- D. Execution processing ---


def test_manual_open_link_produces_human_action_required():
    results = contracts.process_execution_items(
        [_sample_execution_item(execution_kind="manual_open_link", command={"url": "u"})]
    )
    assert results[0]["item_status"] == contracts.EXECUTION_ITEM_SKIPPED
    assert results[0]["output"]["human_action_required"] is True
    assert results[0]["output"]["command"] == {"url": "u"}


def test_manual_open_link_does_not_open_browser_or_http(tmp_path):
    run_id, _, _, _, _, _, _ = _run_with_execution_request(tmp_path)
    with patch("webbrowser.open") as browser, patch("requests.get") as http:
        events.execute_run_execution_request(tmp_path, run_id, "human")
    browser.assert_not_called()
    http.assert_not_called()


def test_update_documentation_produces_proposed_update(tmp_path):
    run_id, _, _, _, _, plan = _run_with_planned_run(tmp_path)
    request = _execution_request_record(
        run_id,
        plan,
        execution_items=[
            _sample_execution_item(
                execution_kind="update_documentation",
                command={"path": "docs/readme.md", "text": "new"},
            )
        ],
    )
    events.request_run_execution(run_id, request, base_dir=tmp_path)
    doc_path = tmp_path / "docs" / "readme.md"
    doc_path.parent.mkdir(parents=True)
    doc_path.write_text("old", encoding="utf-8")
    result = events.execute_run_execution_request(tmp_path, run_id, "human")
    assert result["item_results"][0]["output"]["proposed_update"] == {
        "path": "docs/readme.md",
        "text": "new",
    }
    assert doc_path.read_text(encoding="utf-8") == "old"


def test_other_no_op_completes():
    results = contracts.process_execution_items(
        [_sample_execution_item(execution_kind="other", command={"no_op": True})]
    )
    assert results[0]["item_status"] == contracts.EXECUTION_ITEM_COMPLETED
    assert results[0]["output"]["no_op_completed"] is True


def test_other_without_no_op_is_unsupported():
    results = contracts.process_execution_items(
        [_sample_execution_item(execution_kind="other", command={"action": "run"})]
    )
    assert results[0]["item_status"] == contracts.EXECUTION_ITEM_UNSUPPORTED


@pytest.mark.parametrize("kind", ["rerun_task", "regenerate_output", "external_action"])
def test_unsupported_kinds_are_unsupported(kind):
    results = contracts.process_execution_items(
        [_sample_execution_item(execution_kind=kind, command={"x": 1})]
    )
    assert results[0]["item_status"] == contracts.EXECUTION_ITEM_UNSUPPORTED


def test_command_dict_is_copied_as_data_not_executed():
    command = {"url": "https://example.com", "shell": "rm -rf /"}
    results = contracts.process_execution_items(
        [_sample_execution_item(command=command)]
    )
    assert results[0]["output"]["command"] == command
    assert results[0]["output"]["command"] is not command


def test_aggregate_all_completed():
    items = [
        _sample_item_result(item_status=contracts.EXECUTION_ITEM_COMPLETED),
        _sample_item_result(item_id="exec-2", item_status=contracts.EXECUTION_ITEM_COMPLETED),
    ]
    assert contracts.compute_execution_result_status(items) == contracts.EXECUTION_RESULT_COMPLETED


def test_aggregate_mix_completed_and_skipped_is_partial():
    items = [
        _sample_item_result(item_status=contracts.EXECUTION_ITEM_COMPLETED),
        _sample_item_result(item_id="exec-2", item_status=contracts.EXECUTION_ITEM_SKIPPED),
    ]
    assert contracts.compute_execution_result_status(items) == contracts.EXECUTION_RESULT_PARTIAL


def test_aggregate_all_skipped_is_failed():
    items = [_sample_item_result(item_status=contracts.EXECUTION_ITEM_SKIPPED)]
    assert contracts.compute_execution_result_status(items) == contracts.EXECUTION_RESULT_FAILED


def test_aggregate_all_unsupported_is_failed():
    items = [_sample_item_result(item_status=contracts.EXECUTION_ITEM_UNSUPPORTED)]
    assert contracts.compute_execution_result_status(items) == contracts.EXECUTION_RESULT_FAILED


def test_aggregate_all_failed_is_failed():
    items = [_sample_item_result(item_status=contracts.EXECUTION_ITEM_FAILED)]
    assert contracts.compute_execution_result_status(items) == contracts.EXECUTION_RESULT_FAILED


# --- E. Lifecycle success ---


def test_execute_run_execution_request_succeeds(tmp_path):
    run_id, _, _, _, _, _, request = _run_with_execution_request(tmp_path)
    result = events.execute_run_execution_request(tmp_path, run_id, "human")
    assert result["run_id"] == run_id
    assert result["source_execution_request_fingerprint"] == contracts.run_execution_request_fingerprint(
        request
    )


def test_execute_writes_result_file(tmp_path):
    run_id, _, _, _, _, _, _ = _run_with_execution_request(tmp_path)
    result = events.execute_run_execution_request(tmp_path, run_id, "human")
    assert io.read_json(
        contracts.run_execution_result_record_json_path(run_id, tmp_path)
    ) == result


def test_execute_appends_event(tmp_path):
    run_id, _, _, _, _, _, _ = _run_with_execution_request(tmp_path)
    before = len(events.read_task_events(run_id, base_dir=tmp_path))
    events.execute_run_execution_request(tmp_path, run_id, "human")
    rows = events.read_task_events(run_id, base_dir=tmp_path)
    assert len(rows) == before + 1
    event = rows[-1]
    assert event["event_type"] == events.EVENT_TYPE_RUN_EXECUTION_COMPLETED
    assert "task_id" not in event
    assert "attempt_id" not in event


def test_execute_event_payload_fields(tmp_path):
    run_id, _, _, _, _, _, request = _run_with_execution_request(tmp_path)
    result = events.execute_run_execution_request(
        tmp_path, run_id, "assistant", event_id=new_event_id()
    )
    event = events.read_task_events(run_id, base_dir=tmp_path)[-1]
    payload = event["payload"]
    assert payload["run_id"] == run_id
    assert payload["executor"] == "assistant"
    assert payload["result_status"] == result["result_status"]
    assert payload["source_execution_request_fingerprint"] == contracts.run_execution_request_fingerprint(
        request
    )
    assert payload["run_execution_result_fingerprint"] == contracts.run_execution_result_fingerprint(
        result
    )
    assert payload["run_execution_result_record_path"].endswith(
        "run_execution_result_record.json"
    )


# --- F. Preconditions ---


def test_invalid_run_id_fails_no_side_effects(tmp_path):
    run_id, task_ids, _, _, _, _, _ = _run_with_execution_request(tmp_path)
    before = _snapshot(tmp_path, run_id, task_ids)
    with pytest.raises(ValueError):
        events.execute_run_execution_request(tmp_path, "bad", "human")
    assert _snapshot(tmp_path, run_id, task_ids) == before


def test_invalid_executor_fails_no_side_effects(tmp_path):
    run_id, task_ids, _, _, _, _, _ = _run_with_execution_request(tmp_path)
    before = _snapshot(tmp_path, run_id, task_ids)
    with pytest.raises(ValueError, match="executor must be a non-empty string"):
        events.execute_run_execution_request(tmp_path, run_id, "")
    assert _snapshot(tmp_path, run_id, task_ids) == before


def test_missing_run_manifest_fails_no_side_effects(tmp_path):
    run_id = new_run_id()
    with pytest.raises(InvalidTransition):
        events.execute_run_execution_request(tmp_path, run_id, "human")


def test_run_not_completed_fails_no_side_effects(tmp_path):
    run_id = new_run_id()
    io.create_run_workspace(run_id, base_dir=tmp_path)
    before = _snapshot(tmp_path, run_id, [])
    with pytest.raises(InvalidTransition, match="is not completed"):
        events.execute_run_execution_request(tmp_path, run_id, "human")
    assert _snapshot(tmp_path, run_id, []) == before


def test_missing_completion_record_fails_no_side_effects(tmp_path):
    run_id, task_ids, _, _, _, _, _ = _run_with_execution_request(tmp_path)
    contracts.run_completion_record_json_path(run_id, tmp_path).unlink()
    before = _snapshot(tmp_path, run_id, task_ids)
    with pytest.raises(InvalidTransition, match="run_completion_record.json is missing"):
        events.execute_run_execution_request(tmp_path, run_id, "human")
    assert _snapshot(tmp_path, run_id, task_ids) == before


def test_missing_review_record_fails_no_side_effects(tmp_path):
    run_id, task_ids, _, _, _, _, _ = _run_with_execution_request(tmp_path)
    contracts.run_review_record_json_path(run_id, tmp_path).unlink()
    before = _snapshot(tmp_path, run_id, task_ids)
    with pytest.raises(InvalidTransition, match="run_review_record.json is missing"):
        events.execute_run_execution_request(tmp_path, run_id, "human")
    assert _snapshot(tmp_path, run_id, task_ids) == before


def test_missing_followup_plan_record_fails_no_side_effects(tmp_path):
    run_id, task_ids, _, _, _, plan = _run_with_planned_run(tmp_path)
    request = _execution_request_record(run_id, plan)
    events.request_run_execution(run_id, request, base_dir=tmp_path)
    contracts.run_followup_plan_record_json_path(run_id, tmp_path).unlink()
    before = _snapshot(tmp_path, run_id, task_ids)
    with pytest.raises(InvalidTransition, match="run_followup_plan_record.json is missing"):
        events.execute_run_execution_request(tmp_path, run_id, "human")
    assert _snapshot(tmp_path, run_id, task_ids) == before


def test_missing_execution_request_record_fails_no_side_effects(tmp_path):
    run_id, task_ids, _, _, _, plan = _run_with_planned_run(tmp_path)
    before = _snapshot(tmp_path, run_id, task_ids)
    with pytest.raises(InvalidTransition, match="run_execution_request_record.json is missing"):
        events.execute_run_execution_request(tmp_path, run_id, "human")
    assert _snapshot(tmp_path, run_id, task_ids) == before


def test_invalid_execution_request_schema_fails_no_side_effects(tmp_path):
    run_id, task_ids, _, _, _, _, _ = _run_with_execution_request(tmp_path)
    path = contracts.run_execution_request_record_json_path(run_id, tmp_path)
    bad = io.read_json(path)
    bad["metadata"] = "bad"
    io.atomic_write_json(path, bad)
    before = _snapshot(tmp_path, run_id, task_ids)
    with pytest.raises(ValueError):
        events.execute_run_execution_request(tmp_path, run_id, "human")
    assert _snapshot(tmp_path, run_id, task_ids) == before


def test_source_followup_plan_fingerprint_mismatch_fails_no_side_effects(tmp_path):
    run_id, task_ids, _, _, _, plan = _run_with_planned_run(tmp_path)
    request = _execution_request_record(run_id, plan)
    bad_request = dict(request)
    bad_request["source_followup_plan_fingerprint"] = "wrong"
    io.atomic_write_json(
        contracts.run_execution_request_record_json_path(run_id, tmp_path),
        bad_request,
    )
    before = _snapshot(tmp_path, run_id, task_ids)
    with pytest.raises(InvalidTransition, match="source_followup_plan_fingerprint"):
        events.execute_run_execution_request(tmp_path, run_id, "human")
    assert _snapshot(tmp_path, run_id, task_ids) == before


def test_source_execution_request_fingerprint_mismatch_fails_no_side_effects(tmp_path):
    run_id, task_ids, _, _, _, _, _ = _run_with_execution_request(tmp_path)
    before = _snapshot(tmp_path, run_id, task_ids)
    original_read = io.read_json
    request_reads = 0

    def fake_read(path):
        nonlocal request_reads
        data = original_read(path)
        if str(path).endswith("run_execution_request_record.json"):
            request_reads += 1
            if request_reads >= 2:
                mutated = dict(data)
                mutated["metadata"] = {"changed": True}
                return mutated
        return data

    with patch("htr.events.read_json", side_effect=fake_read):
        with pytest.raises(InvalidTransition, match="source_execution_request_fingerprint"):
            events.execute_run_execution_request(tmp_path, run_id, "human")
    assert _snapshot(tmp_path, run_id, task_ids) == before


def test_request_status_cancelled_fails_no_side_effects(tmp_path):
    run_id, task_ids, _, _, _, plan = _run_with_planned_run(tmp_path)
    request = _execution_request_record(
        run_id,
        plan,
        request_status=contracts.EXECUTION_REQUEST_CANCELLED,
    )
    events.request_run_execution(run_id, request, base_dir=tmp_path)
    before = _snapshot(tmp_path, run_id, task_ids)
    with pytest.raises(InvalidTransition, match="expected pending"):
        events.execute_run_execution_request(tmp_path, run_id, "human")
    assert _snapshot(tmp_path, run_id, task_ids) == before


# --- G. Write order ---


def test_write_order_record_before_event(tmp_path):
    run_id, _, _, _, _, _, _ = _run_with_execution_request(tmp_path)
    ops = []
    real_append = events.append_run_event

    def track_write(path, data):
        if str(path).endswith("run_execution_result_record.json"):
            ops.append("write_record")
        return io.atomic_write_json(path, data)

    def track_append(r, e, base_dir=None):
        ops.append("append_event")
        return real_append(r, e, base_dir)

    with patch("htr.events.atomic_write_json", side_effect=track_write), patch(
        "htr.events.append_run_event", side_effect=track_append
    ):
        events.execute_run_execution_request(tmp_path, run_id, "human")
    assert ops == ["write_record", "append_event"]


# --- H. Replay-only ---


def test_existing_result_event_id_none_raises(tmp_path):
    run_id, task_ids, _, _, _, _, _ = _run_with_execution_request(tmp_path)
    events.execute_run_execution_request(
        tmp_path, run_id, "human", event_id=new_event_id()
    )
    before = _snapshot(tmp_path, run_id, task_ids)
    with pytest.raises(InvalidTransition):
        events.execute_run_execution_request(tmp_path, run_id, "human")
    assert _snapshot(tmp_path, run_id, task_ids) == before


def test_existing_result_missing_event_raises(tmp_path):
    run_id, task_ids, _, _, _, _, request = _run_with_execution_request(tmp_path)
    result = contracts.make_run_execution_result_record(
        run_id=run_id,
        source_execution_request_fingerprint=contracts.run_execution_request_fingerprint(
            request
        ),
        item_results=contracts.process_execution_items(request["execution_items"]),
        executor="human",
    )
    io.atomic_write_json(
        contracts.run_execution_result_record_json_path(run_id, tmp_path), result
    )
    before = _snapshot(tmp_path, run_id, task_ids)
    with pytest.raises(InvalidTransition):
        events.execute_run_execution_request(
            tmp_path, run_id, "human", event_id=new_event_id()
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
                    "run_execution_result_fingerprint": "different",
                },
            },
            EventConflict,
        ),
        (
            lambda e: {
                **e,
                "payload": {
                    **e["payload"],
                    "source_execution_request_fingerprint": "different",
                },
            },
            EventConflict,
        ),
        (
            lambda e: {
                **e,
                "payload": {**e["payload"], "result_status": "partial"},
            },
            EventConflict,
        ),
        (
            lambda e: {**e, "payload": {**e["payload"], "executor": "other"}},
            EventConflict,
        ),
    ],
)
def test_existing_result_replay_semantic_mismatch(tmp_path, mutator, expected):
    run_id, task_ids, _, _, _, _, _ = _run_with_execution_request(tmp_path)
    event_id = new_event_id()
    events.execute_run_execution_request(
        tmp_path, run_id, "human", event_id=event_id
    )
    event = events.read_task_events(run_id, base_dir=tmp_path)[-1]
    bad_event = mutator(event)
    before = _snapshot(tmp_path, run_id, task_ids)
    with patch("htr.events._find_run_event_by_id", return_value=bad_event):
        with pytest.raises(expected):
            events.execute_run_execution_request(
                tmp_path, run_id, "human", event_id=event_id
            )
    assert _snapshot(tmp_path, run_id, task_ids) == before


def test_existing_result_same_event_id_same_semantic_returns_existing(tmp_path):
    run_id, task_ids, _, _, _, _, _ = _run_with_execution_request(tmp_path)
    event_id = new_event_id()
    first = events.execute_run_execution_request(
        tmp_path, run_id, "human", event_id=event_id
    )
    before = _snapshot(tmp_path, run_id, task_ids)
    second = events.execute_run_execution_request(
        tmp_path, run_id, "human", event_id=event_id
    )
    assert second == first
    assert _snapshot(tmp_path, run_id, task_ids) == before


def test_replay_only_no_writes_or_external_execution(tmp_path):
    run_id, task_ids, _, _, _, _, _ = _run_with_execution_request(tmp_path)
    event_id = new_event_id()
    events.execute_run_execution_request(
        tmp_path, run_id, "human", event_id=event_id
    )
    manifest_before = io.read_json(paths.run_manifest_path(run_id, tmp_path))
    request_before = contracts.run_execution_request_record_json_path(
        run_id, tmp_path
    ).read_bytes()
    with patch("subprocess.run") as subprocess_run, patch("webbrowser.open") as browser:
        events.execute_run_execution_request(
            tmp_path, run_id, "human", event_id=event_id
        )
    subprocess_run.assert_not_called()
    browser.assert_not_called()
    assert io.read_json(paths.run_manifest_path(run_id, tmp_path)) == manifest_before
    assert (
        contracts.run_execution_request_record_json_path(run_id, tmp_path).read_bytes()
        == request_before
    )


# --- I. Idempotency ---


def test_idempotent_same_event_id_same_semantic(tmp_path):
    run_id, _, _, _, _, _, _ = _run_with_execution_request(tmp_path)
    event_id = new_event_id()
    first = events.execute_run_execution_request(
        tmp_path, run_id, "human", event_id=event_id
    )
    second = events.execute_run_execution_request(
        tmp_path, run_id, "human", event_id=event_id
    )
    assert second == first


def test_same_event_id_different_semantic_raises_conflict(tmp_path):
    run_id, _, _, _, _, _, _ = _run_with_execution_request(tmp_path)
    event_id = new_event_id()
    events.execute_run_execution_request(
        tmp_path, run_id, "first", event_id=event_id
    )
    with pytest.raises(EventConflict):
        events.execute_run_execution_request(
            tmp_path, run_id, "second", event_id=event_id
        )


def test_idempotency_requires_execution_request(tmp_path):
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
    with pytest.raises(InvalidTransition, match="run_execution_request_record.json is missing"):
        events.execute_run_execution_request(
            tmp_path, run_id, "human", event_id=new_event_id()
        )


def test_idempotency_does_not_allow_non_completed_run(tmp_path):
    run_id = new_run_id()
    io.create_run_workspace(run_id, base_dir=tmp_path)
    with pytest.raises(InvalidTransition):
        events.execute_run_execution_request(
            tmp_path, run_id, "human", event_id=new_event_id()
        )


# --- J. Boundary preservation ---


def test_boundaries_preserved_on_success(tmp_path):
    run_id, task_ids, attempt_ids, _, _, _, _ = _run_with_execution_request(tmp_path)
    before = _snapshot(tmp_path, run_id, task_ids)
    events.execute_run_execution_request(tmp_path, run_id, "human")
    after = _snapshot(tmp_path, run_id, task_ids)
    assert after["manifest"] == before["manifest"]
    assert after["completion"] == before["completion"]
    assert after["review"] == before["review"]
    assert after["followup"] == before["followup"]
    assert after["request"] == before["request"]
    assert after["task_statuses"] == before["task_statuses"]
    assert after["result_exists"] is True
    assert len(list(paths.tasks_dir(run_id, tmp_path).iterdir())) == len(task_ids)
    task_id = task_ids[0]
    attempt_id = attempt_ids[0]
    result_path = paths.result_json_path(run_id, task_id, attempt_id, tmp_path)
    verification_path = contracts.verification_result_json_path(
        run_id, task_id, attempt_id, tmp_path
    )
    assert result_path.exists()
    assert verification_path.exists()


# --- K. Import boundary ---


def test_task10_modules_do_not_import_forbidden_modules():
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


def test_package_exports_task10_apis():
    import htr

    for name in (
        "make_run_execution_result_record",
        "run_execution_result_fingerprint",
        "execute_run_execution_request",
        "process_execution_items",
        "compute_execution_result_status",
        "EVENT_TYPE_RUN_EXECUTION_COMPLETED",
        "EXECUTION_RESULT_COMPLETED",
        "EXECUTION_RESULT_PARTIAL",
        "EXECUTION_RESULT_FAILED",
        "EXECUTION_ITEM_COMPLETED",
        "EXECUTION_ITEM_SKIPPED",
        "EXECUTION_ITEM_FAILED",
        "EXECUTION_ITEM_UNSUPPORTED",
    ):
        assert hasattr(htr, name)
