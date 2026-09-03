import ast
from pathlib import Path
from unittest.mock import patch

import pytest

from htr import contracts, events, io, paths, schemas
from htr.ids import new_attempt_id, new_event_id, new_run_id, new_task_id
from htr.state import (
    ATTEMPT_RUNNING,
    RUN_COMPLETED,
    RUN_CREATED,
    TASK_COMPLETED,
    TASK_RUNNING,
    EventConflict,
    InvalidTransition,
)


def _review_record(run_id, decision=contracts.RUN_REVIEW_ACCEPTED, **kwargs):
    return contracts.make_run_review_record(
        run_id=run_id,
        decision=decision,
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


def _run_with_completed_run(tmp_path, task_count=1):
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
    return run_id, task_ids, attempt_ids, completion_record


def _snapshot_review_side_effects(
    tmp_path, run_id, task_ids, attempt_ids=None, completion_record=None
):
    if attempt_ids is None:
        attempt_ids = []
    snapshot = {
        "run_review_exists": contracts.run_review_record_json_path(
            run_id, tmp_path
        ).exists(),
        "events": list(events.read_task_events(run_id, base_dir=tmp_path)),
        "run_manifest": io.read_json(paths.run_manifest_path(run_id, tmp_path)),
        "task_statuses": {
            task_id: io.read_json(paths.task_status_path(run_id, task_id, tmp_path))
            for task_id in task_ids
        },
        "attempt_statuses": {},
    }
    if completion_record is not None:
        completion_path = contracts.run_completion_record_json_path(run_id, tmp_path)
        snapshot["run_completion_record"] = (
            completion_path.read_bytes() if completion_path.exists() else None
        )
    for i, attempt_id in enumerate(attempt_ids):
        if i >= len(task_ids):
            break
        task_id = task_ids[i]
        attempt_path = paths.attempt_status_path(run_id, task_id, attempt_id, tmp_path)
        if attempt_path.exists():
            snapshot["attempt_statuses"][attempt_id] = io.read_json(attempt_path)
        result_path = paths.result_json_path(run_id, task_id, attempt_id, tmp_path)
        verification_path = contracts.verification_result_json_path(
            run_id, task_id, attempt_id, tmp_path
        )
        snapshot[f"result:{attempt_id}"] = (
            result_path.read_bytes() if result_path.exists() else None
        )
        snapshot[f"verification:{attempt_id}"] = (
            verification_path.read_bytes() if verification_path.exists() else None
        )
    return snapshot


# --- Contract / schema / fingerprint ---


def test_make_run_review_record_valid_accepted():
    record = _review_record(new_run_id(), contracts.RUN_REVIEW_ACCEPTED)
    schemas.validate(record, "run_review_record")


def test_make_run_review_record_valid_rejected():
    record = _review_record(new_run_id(), contracts.RUN_REVIEW_REJECTED)
    schemas.validate(record, "run_review_record")


def test_make_run_review_record_valid_needs_followup():
    record = _review_record(new_run_id(), contracts.RUN_REVIEW_NEEDS_FOLLOWUP)
    schemas.validate(record, "run_review_record")


def test_make_run_review_record_preserves_notes_none():
    record = _review_record(new_run_id())
    assert record["notes"] is None


def test_make_run_review_record_metadata_defaults_to_empty_dict():
    record = _review_record(new_run_id())
    assert record["metadata"] == {}


def test_make_run_review_record_invalid_metadata_type_fails():
    with pytest.raises(ValueError, match="metadata must be a dict"):
        contracts.make_run_review_record(
            run_id=new_run_id(),
            decision=contracts.RUN_REVIEW_ACCEPTED,
            metadata=[],
        )


def test_make_run_review_record_invalid_decision_fails():
    with pytest.raises(ValueError, match="decision must be one of"):
        contracts.make_run_review_record(
            run_id=new_run_id(),
            decision="maybe",
        )


def test_make_run_review_record_empty_reviewer_fails():
    with pytest.raises(ValueError, match="reviewer must be a non-empty string"):
        contracts.make_run_review_record(
            run_id=new_run_id(),
            decision=contracts.RUN_REVIEW_ACCEPTED,
            reviewer="",
        )


def test_run_review_fingerprint_is_stable_for_key_order():
    run_id = new_run_id()
    first = {
        "schema_version": "1",
        "run_id": run_id,
        "decision": contracts.RUN_REVIEW_ACCEPTED,
        "reviewer": "human",
        "notes": None,
        "metadata": {"b": 2, "a": 1},
        "created_at": "2026-07-18T08:00:00+00:00",
    }
    second = {
        "schema_version": "1",
        "created_at": "2026-07-18T08:00:00+00:00",
        "metadata": {"a": 1, "b": 2},
        "decision": contracts.RUN_REVIEW_ACCEPTED,
        "reviewer": "human",
        "notes": None,
        "run_id": run_id,
    }
    schemas.validate(first, "run_review_record")
    schemas.validate(second, "run_review_record")
    assert contracts.run_review_fingerprint(first) == contracts.run_review_fingerprint(
        second
    )


def test_run_review_fingerprint_invalid_schema_fails():
    with pytest.raises(ValueError, match="metadata must be a dict"):
        contracts.run_review_fingerprint(
            {
                "schema_version": "1",
                "run_id": new_run_id(),
                "decision": contracts.RUN_REVIEW_ACCEPTED,
                "reviewer": "human",
                "notes": None,
                "metadata": [],
                "created_at": "2026-07-18T08:00:00+00:00",
            }
        )


# --- Success path ---


def test_review_run_manually_succeeds_when_run_completed_and_completion_record_exists(
    tmp_path,
):
    run_id, task_ids, _, completion_record = _run_with_completed_run(tmp_path)
    record = _review_record(run_id)
    event = events.review_run_manually(run_id, record, base_dir=tmp_path)
    assert event["event_type"] == events.EVENT_TYPE_MANUAL_RUN_REVIEWED
    assert contracts.run_review_record_json_path(run_id, tmp_path).exists()
    assert io.read_json(paths.run_manifest_path(run_id, tmp_path))["status"] == RUN_COMPLETED
    assert io.read_json(
        contracts.run_completion_record_json_path(run_id, tmp_path)
    ) == completion_record


def test_review_run_manually_writes_run_review_record(tmp_path):
    run_id, _, _, _ = _run_with_completed_run(tmp_path)
    record = _review_record(run_id, notes="looks good")
    events.review_run_manually(run_id, record, base_dir=tmp_path)
    assert io.read_json(contracts.run_review_record_json_path(run_id, tmp_path)) == record


def test_review_run_manually_appends_manual_run_reviewed_event(tmp_path):
    run_id, _, _, _ = _run_with_completed_run(tmp_path)
    record = _review_record(run_id)
    before = len(events.read_task_events(run_id, base_dir=tmp_path))
    event = events.review_run_manually(run_id, record, base_dir=tmp_path)
    rows = events.read_task_events(run_id, base_dir=tmp_path)
    assert len(rows) == before + 1
    assert rows[-1] == event
    assert event["event_type"] == events.EVENT_TYPE_MANUAL_RUN_REVIEWED
    assert "task_id" not in event


def test_review_run_manually_does_not_update_run_manifest(tmp_path):
    run_id, _, _, _ = _run_with_completed_run(tmp_path)
    before = io.read_json(paths.run_manifest_path(run_id, tmp_path))
    events.review_run_manually(
        run_id, _review_record(run_id), base_dir=tmp_path
    )
    assert io.read_json(paths.run_manifest_path(run_id, tmp_path)) == before


def test_review_run_manually_event_payload_and_fields(tmp_path):
    run_id, _, _, _ = _run_with_completed_run(tmp_path)
    record = _review_record(
        run_id, decision=contracts.RUN_REVIEW_REJECTED, reviewer="reviewer-a"
    )
    event = events.review_run_manually(
        run_id, record, actor="auditor", base_dir=tmp_path
    )
    assert event["run_id"] == run_id
    assert event["actor"] == "auditor"
    assert event["payload"]["decision"] == contracts.RUN_REVIEW_REJECTED
    assert event["payload"]["reviewer"] == "reviewer-a"
    assert event["payload"]["run_review_fingerprint"] == contracts.run_review_fingerprint(
        record
    )


def test_review_run_manually_write_order(tmp_path):
    run_id, _, _, _ = _run_with_completed_run(tmp_path)
    record = _review_record(run_id)
    operations = []
    real_append = events.append_run_event

    def track_write(path, data):
        if str(path).endswith("run_review_record.json"):
            operations.append("write_run_review_record")
        return io.atomic_write_json(path, data)

    def track_append(run, event, base_dir=None):
        operations.append("append_run_event")
        return real_append(run, event, base_dir)

    with patch("htr.events.atomic_write_json", side_effect=track_write), patch(
        "htr.events.append_run_event", side_effect=track_append
    ):
        events.review_run_manually(run_id, record, base_dir=tmp_path)

    assert operations == ["write_run_review_record", "append_run_event"]


# --- Boundaries ---


def test_review_run_manually_leaves_task_statuses_unchanged(tmp_path):
    run_id, task_ids, attempt_ids, completion_record = _run_with_completed_run(tmp_path)
    before = {
        task_id: io.read_json(paths.task_status_path(run_id, task_id, tmp_path))
        for task_id in task_ids
    }
    events.review_run_manually(run_id, _review_record(run_id), base_dir=tmp_path)
    for task_id in task_ids:
        assert io.read_json(paths.task_status_path(run_id, task_id, tmp_path)) == before[
            task_id
        ]


def test_review_run_manually_leaves_attempt_statuses_unchanged(tmp_path):
    run_id, task_ids, attempt_ids, _ = _run_with_completed_run(tmp_path)
    before = {
        attempt_id: io.read_json(
            paths.attempt_status_path(run_id, task_ids[i], attempt_id, tmp_path)
        )
        for i, attempt_id in enumerate(attempt_ids)
    }
    events.review_run_manually(run_id, _review_record(run_id), base_dir=tmp_path)
    for i, attempt_id in enumerate(attempt_ids):
        assert (
            io.read_json(
                paths.attempt_status_path(run_id, task_ids[i], attempt_id, tmp_path)
            )
            == before[attempt_id]
        )


def test_review_run_manually_leaves_result_and_verification_files_unchanged(tmp_path):
    run_id, task_ids, attempt_ids, completion_record = _run_with_completed_run(tmp_path)
    before = _snapshot_review_side_effects(
        tmp_path, run_id, task_ids, attempt_ids, completion_record
    )
    events.review_run_manually(run_id, _review_record(run_id), base_dir=tmp_path)
    after = _snapshot_review_side_effects(
        tmp_path, run_id, task_ids, attempt_ids, completion_record
    )
    for attempt_id in attempt_ids:
        assert before[f"result:{attempt_id}"] == after[f"result:{attempt_id}"]
        assert before[f"verification:{attempt_id}"] == after[f"verification:{attempt_id}"]


def test_review_run_manually_leaves_run_completion_record_unchanged(tmp_path):
    run_id, task_ids, attempt_ids, completion_record = _run_with_completed_run(tmp_path)
    before = _snapshot_review_side_effects(
        tmp_path, run_id, task_ids, attempt_ids, completion_record
    )
    events.review_run_manually(run_id, _review_record(run_id), base_dir=tmp_path)
    after = _snapshot_review_side_effects(
        tmp_path, run_id, task_ids, attempt_ids, completion_record
    )
    assert before["run_completion_record"] == after["run_completion_record"]


def test_review_run_manually_does_not_create_new_tasks_or_attempts(tmp_path):
    run_id, task_ids, _, _ = _run_with_completed_run(tmp_path, task_count=2)
    events.review_run_manually(run_id, _review_record(run_id), base_dir=tmp_path)
    for task_id in task_ids:
        status = io.read_json(paths.task_status_path(run_id, task_id, tmp_path))
        assert len(status["attempts"]) == 1


# --- Failure / no side effects ---


def test_review_run_manually_invalid_schema_blocks_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, completion_record = _run_with_completed_run(tmp_path)
    record = _review_record(run_id)
    record["metadata"] = "bad"
    before = _snapshot_review_side_effects(
        tmp_path, run_id, task_ids, attempt_ids, completion_record
    )
    with pytest.raises(ValueError, match="metadata must be a dict"):
        events.review_run_manually(run_id, record, base_dir=tmp_path)
    assert (
        _snapshot_review_side_effects(
            tmp_path, run_id, task_ids, attempt_ids, completion_record
        )
        == before
    )


def test_review_run_manually_identity_mismatch_blocks_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, completion_record = _run_with_completed_run(tmp_path)
    record = _review_record(new_run_id())
    before = _snapshot_review_side_effects(
        tmp_path, run_id, task_ids, attempt_ids, completion_record
    )
    with pytest.raises(ValueError, match="run_id does not match"):
        events.review_run_manually(run_id, record, base_dir=tmp_path)
    assert (
        _snapshot_review_side_effects(
            tmp_path, run_id, task_ids, attempt_ids, completion_record
        )
        == before
    )


def test_review_run_manually_run_not_completed_blocks_side_effects(tmp_path):
    run_id = new_run_id()
    io.create_run_workspace(run_id, base_dir=tmp_path)
    manifest_path = paths.run_manifest_path(run_id, tmp_path)
    manifest = io.read_json(manifest_path)
    assert manifest["status"] == RUN_CREATED
    io.atomic_write_json(
        contracts.run_completion_record_json_path(run_id, tmp_path),
        contracts.make_run_completion_record(run_id=run_id, completed_task_ids=[new_task_id()]),
    )
    before = _snapshot_review_side_effects(tmp_path, run_id, [], [])
    with pytest.raises(InvalidTransition, match="is not completed"):
        events.review_run_manually(run_id, _review_record(run_id), base_dir=tmp_path)
    assert _snapshot_review_side_effects(tmp_path, run_id, [], []) == before


def test_review_run_manually_missing_run_completion_record_blocks_side_effects(tmp_path):
    run_id, task_ids, attempt_ids, _ = _run_with_completed_run(tmp_path)
    completion_path = contracts.run_completion_record_json_path(run_id, tmp_path)
    completion_path.unlink()
    before = _snapshot_review_side_effects(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(InvalidTransition, match="run_completion_record.json is missing"):
        events.review_run_manually(run_id, _review_record(run_id), base_dir=tmp_path)
    assert (
        _snapshot_review_side_effects(tmp_path, run_id, task_ids, attempt_ids) == before
    )


def test_review_run_manually_existing_review_record_event_id_none_raises(tmp_path):
    run_id, task_ids, attempt_ids, _ = _run_with_completed_run(tmp_path)
    record = _review_record(run_id)
    events.review_run_manually(
        run_id, record, event_id=new_event_id(), base_dir=tmp_path
    )
    before = _snapshot_review_side_effects(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(InvalidTransition, match="run_review_record.json already exists"):
        events.review_run_manually(run_id, record, base_dir=tmp_path)
    assert (
        _snapshot_review_side_effects(tmp_path, run_id, task_ids, attempt_ids) == before
    )


def test_review_run_manually_existing_review_record_missing_event_raises(tmp_path):
    run_id, task_ids, attempt_ids, _ = _run_with_completed_run(tmp_path)
    record = _review_record(run_id)
    io.atomic_write_json(contracts.run_review_record_json_path(run_id, tmp_path), record)
    before = _snapshot_review_side_effects(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(InvalidTransition, match="run_review_record.json already exists"):
        events.review_run_manually(
            run_id, record, event_id=new_event_id(), base_dir=tmp_path
        )
    assert (
        _snapshot_review_side_effects(tmp_path, run_id, task_ids, attempt_ids) == before
    )


def test_review_run_manually_existing_review_record_different_semantic_raises_conflict(
    tmp_path,
):
    run_id, task_ids, attempt_ids, _ = _run_with_completed_run(tmp_path)
    record = _review_record(run_id)
    event_id = new_event_id()
    events.review_run_manually(
        run_id, record, event_id=event_id, base_dir=tmp_path
    )
    different = _review_record(run_id, notes="different")
    before = _snapshot_review_side_effects(tmp_path, run_id, task_ids, attempt_ids)
    with pytest.raises(EventConflict):
        events.review_run_manually(
            run_id, different, event_id=event_id, base_dir=tmp_path
        )
    assert (
        _snapshot_review_side_effects(tmp_path, run_id, task_ids, attempt_ids) == before
    )


# --- Idempotency / replay ---


def test_review_run_manually_idempotent_same_event_id(tmp_path):
    run_id, _, _, _ = _run_with_completed_run(tmp_path)
    record = _review_record(run_id)
    event_id = new_event_id()
    first = events.review_run_manually(
        run_id, record, event_id=event_id, base_dir=tmp_path
    )
    second = events.review_run_manually(
        run_id, record, event_id=event_id, base_dir=tmp_path
    )
    assert second == first


def test_review_run_manually_same_event_id_different_record_raises_conflict(tmp_path):
    run_id, _, _, _ = _run_with_completed_run(tmp_path)
    event_id = new_event_id()
    first = _review_record(run_id, notes="first")
    events.review_run_manually(run_id, first, event_id=event_id, base_dir=tmp_path)
    second = _review_record(run_id, notes="second")
    with pytest.raises(EventConflict):
        events.review_run_manually(run_id, second, event_id=event_id, base_dir=tmp_path)


def test_review_run_manually_replay_same_event_id_same_review_returns_existing(tmp_path):
    run_id, task_ids, attempt_ids, _ = _run_with_completed_run(tmp_path)
    record = _review_record(run_id)
    event_id = new_event_id()
    events.review_run_manually(
        run_id, record, event_id=event_id, actor="human", base_dir=tmp_path
    )
    returned = events.review_run_manually(
        run_id, record, event_id=event_id, actor="human", base_dir=tmp_path
    )
    assert returned["event_id"] == event_id


def test_review_run_manually_replay_same_event_id_different_review_raises_conflict(
    tmp_path,
):
    run_id, task_ids, attempt_ids, _ = _run_with_completed_run(tmp_path)
    record = _review_record(run_id)
    event_id = new_event_id()
    events.review_run_manually(
        run_id, record, event_id=event_id, base_dir=tmp_path
    )
    different = _review_record(run_id, decision=contracts.RUN_REVIEW_REJECTED)
    with pytest.raises(EventConflict):
        events.review_run_manually(
            run_id, different, event_id=event_id, base_dir=tmp_path
        )


def test_review_run_manually_existing_review_record_different_event_id_raises(tmp_path):
    run_id, _, _, _ = _run_with_completed_run(tmp_path)
    record = _review_record(run_id)
    events.review_run_manually(
        run_id, record, event_id=new_event_id(), base_dir=tmp_path
    )
    with pytest.raises(InvalidTransition):
        events.review_run_manually(
            run_id, record, event_id=new_event_id(), base_dir=tmp_path
        )


def test_review_run_manually_replay_does_not_rewrite_record_event_or_manifest(tmp_path):
    run_id, task_ids, attempt_ids, completion_record = _run_with_completed_run(tmp_path)
    record = _review_record(run_id)
    event_id = new_event_id()
    events.review_run_manually(
        run_id, record, event_id=event_id, actor="human", base_dir=tmp_path
    )
    review_path = contracts.run_review_record_json_path(run_id, tmp_path)
    review_bytes = review_path.read_bytes()
    manifest_before = io.read_json(paths.run_manifest_path(run_id, tmp_path))
    task_before = io.read_json(paths.task_status_path(run_id, task_ids[0], tmp_path))
    event_count = len(events.read_task_events(run_id, base_dir=tmp_path))

    events.review_run_manually(
        run_id, record, event_id=event_id, actor="human", base_dir=tmp_path
    )

    assert review_path.read_bytes() == review_bytes
    assert io.read_json(paths.run_manifest_path(run_id, tmp_path)) == manifest_before
    assert io.read_json(paths.task_status_path(run_id, task_ids[0], tmp_path)) == task_before
    assert len(events.read_task_events(run_id, base_dir=tmp_path)) == event_count


def test_review_run_manually_replay_event_lookup_scoped_to_run_id(tmp_path):
    run_a, _, _, _ = _run_with_completed_run(tmp_path)
    run_b, _, _, _ = _run_with_completed_run(tmp_path)
    record_a = _review_record(run_a)
    event_id_a = new_event_id()
    events.review_run_manually(
        run_a, record_a, event_id=event_id_a, base_dir=tmp_path
    )
    record_b = _review_record(run_b)
    events.review_run_manually(
        run_b, record_b, event_id=new_event_id(), base_dir=tmp_path
    )
    with pytest.raises(InvalidTransition):
        events.review_run_manually(
            run_b, record_b, event_id=event_id_a, base_dir=tmp_path
        )


# --- Import boundary ---


def test_htr_task7_modules_do_not_import_forbidden_modules():
    repo_root = Path(__file__).resolve().parents[2]
    forbidden = {
        "runtime",
        "delegate_task",
        "deco",
        "heal",
        "scheduler",
        "queue",
        "sqlite",
        "database",
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


def test_htr_package_exports_task7_apis():
    import htr

    for name in (
        "make_run_review_record",
        "run_review_fingerprint",
        "review_run_manually",
        "EVENT_TYPE_MANUAL_RUN_REVIEWED",
        "RUN_REVIEW_ACCEPTED",
        "RUN_REVIEW_REJECTED",
        "RUN_REVIEW_NEEDS_FOLLOWUP",
    ):
        assert hasattr(htr, name)
