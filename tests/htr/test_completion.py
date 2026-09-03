import ast
from pathlib import Path
from unittest.mock import patch

import pytest

from htr import contracts, events, io, paths, schemas
from htr.ids import new_attempt_id, new_event_id, new_run_id, new_task_id
from htr.state import (
    ATTEMPT_HEAL_REQUIRED,
    ATTEMPT_RESULT_SUBMITTED,
    ATTEMPT_RUNNING,
    ATTEMPT_VERIFICATION_FAILED,
    ATTEMPT_VERIFICATION_PASSED,
    TASK_COMPLETED,
    TASK_RUNNING,
    EventConflict,
    InvalidTransition,
)


def _bootstrap_task(tmp_path):
    run_id = new_run_id()
    task_id = new_task_id()
    io.create_run_workspace(run_id, base_dir=tmp_path)
    io.create_task_workspace(run_id, task_id, base_dir=tmp_path)
    return run_id, task_id


def _completion_record(run_id, task_id, attempt_id, **kwargs):
    return contracts.make_task_completion_record(
        run_id=run_id,
        task_id=task_id,
        attempt_id=attempt_id,
        **kwargs,
    )


def _verification_passed_setup(tmp_path, *, task_running=True):
    run_id, task_id = _bootstrap_task(tmp_path)
    attempt_id = new_attempt_id()
    if task_running:
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
    return run_id, task_id, attempt_id


def _snapshot_side_effects(tmp_path, run_id, task_id, attempt_id):
    completion_path = contracts.task_completion_record_json_path(
        run_id, task_id, tmp_path
    )
    return {
        "completion_exists": completion_path.exists(),
        "completion_bytes": completion_path.read_bytes() if completion_path.exists() else None,
        "events": list(events.read_task_events(run_id, base_dir=tmp_path)),
        "task_status": io.read_json(paths.task_status_path(run_id, task_id, tmp_path)),
        "attempt_status": io.read_json(
            paths.attempt_status_path(run_id, task_id, attempt_id, tmp_path)
        ),
        "run_manifest": io.read_json(paths.run_manifest_path(run_id, tmp_path)),
    }


def _assert_unchanged(before, after):
    assert before == after


# --- Contract / schema / fingerprint ---


def test_make_task_completion_record_valid():
    run_id = new_run_id()
    task_id = new_task_id()
    attempt_id = new_attempt_id()
    record = _completion_record(
        run_id, task_id, attempt_id, reason="accepted", metadata={"by": "human"}
    )
    schemas.validate(record, "task_completion_record")
    assert record["reason"] == "accepted"


def test_make_task_completion_record_preserves_reason_none():
    record = _completion_record(new_run_id(), new_task_id(), new_attempt_id())
    assert record["reason"] is None


def test_make_task_completion_record_metadata_defaults_to_empty_dict():
    record = _completion_record(new_run_id(), new_task_id(), new_attempt_id())
    assert record["metadata"] == {}


def test_make_task_completion_record_invalid_metadata_type_fails():
    with pytest.raises(ValueError, match="metadata must be a dict"):
        contracts.make_task_completion_record(
            run_id=new_run_id(),
            task_id=new_task_id(),
            attempt_id=new_attempt_id(),
            metadata=[],
        )


def test_task_completion_fingerprint_is_stable_for_key_order():
    run_id = new_run_id()
    task_id = new_task_id()
    attempt_id = new_attempt_id()
    first = {
        "schema_version": "1",
        "run_id": run_id,
        "task_id": task_id,
        "attempt_id": attempt_id,
        "reason": None,
        "metadata": {"b": 2, "a": 1},
        "created_at": "2026-07-18T08:00:00+00:00",
    }
    second = {
        "schema_version": "1",
        "created_at": "2026-07-18T08:00:00+00:00",
        "metadata": {"a": 1, "b": 2},
        "attempt_id": attempt_id,
        "task_id": task_id,
        "run_id": run_id,
        "reason": None,
    }
    schemas.validate(first, "task_completion_record")
    schemas.validate(second, "task_completion_record")
    assert contracts.task_completion_fingerprint(
        first
    ) == contracts.task_completion_fingerprint(second)


def test_task_completion_fingerprint_invalid_schema_fails():
    with pytest.raises(ValueError, match="metadata must be a dict"):
        contracts.task_completion_fingerprint(
            {
                "schema_version": "1",
                "run_id": new_run_id(),
                "task_id": new_task_id(),
                "attempt_id": new_attempt_id(),
                "reason": None,
                "metadata": [],
                "created_at": "2026-07-18T08:00:00+00:00",
            }
        )


# --- Success path ---


def test_complete_task_manually_succeeds_when_attempt_verification_passed(tmp_path):
    run_id, task_id, attempt_id = _verification_passed_setup(tmp_path)
    record = _completion_record(run_id, task_id, attempt_id)
    event = events.complete_task_manually(
        run_id, task_id, attempt_id, record, base_dir=tmp_path
    )
    assert event["new_status"] == TASK_COMPLETED
    assert io.read_json(paths.task_status_path(run_id, task_id, tmp_path))["status"] == TASK_COMPLETED


def test_complete_task_manually_writes_task_completion_record(tmp_path):
    run_id, task_id, attempt_id = _verification_passed_setup(tmp_path)
    record = _completion_record(run_id, task_id, attempt_id, reason="done")
    events.complete_task_manually(run_id, task_id, attempt_id, record, base_dir=tmp_path)
    assert io.read_json(
        contracts.task_completion_record_json_path(run_id, task_id, tmp_path)
    ) == record


def test_complete_task_manually_appends_manual_task_completed_event(tmp_path):
    run_id, task_id, attempt_id = _verification_passed_setup(tmp_path)
    record = _completion_record(run_id, task_id, attempt_id)
    before = len(events.read_task_events(run_id, base_dir=tmp_path))
    event = events.complete_task_manually(
        run_id, task_id, attempt_id, record, base_dir=tmp_path
    )
    rows = events.read_task_events(run_id, base_dir=tmp_path)
    assert len(rows) == before + 1
    assert rows[-1] == event
    assert event["event_type"] == events.EVENT_TYPE_MANUAL_TASK_COMPLETED


def test_complete_task_manually_event_payload_and_fields(tmp_path):
    run_id, task_id, attempt_id = _verification_passed_setup(tmp_path)
    record = _completion_record(run_id, task_id, attempt_id)
    event = events.complete_task_manually(
        run_id,
        task_id,
        attempt_id,
        record,
        actor="reviewer",
        base_dir=tmp_path,
    )
    assert event["previous_status"] == TASK_RUNNING
    assert event["new_status"] == TASK_COMPLETED
    assert event["actor"] == "reviewer"
    assert event["attempt_id"] == attempt_id
    assert event["payload"]["attempt_id"] == attempt_id
    assert event["payload"]["completion_fingerprint"] == contracts.task_completion_fingerprint(
        record
    )
    assert event["payload"]["completion_record_path"] == str(
        contracts.task_completion_record_json_path(run_id, task_id, tmp_path)
    )


def test_complete_task_manually_write_order(tmp_path):
    run_id, task_id, attempt_id = _verification_passed_setup(tmp_path)
    record = _completion_record(run_id, task_id, attempt_id)
    operations = []
    real_append = events.append_task_event

    def track_write(path, data):
        path_str = str(path)
        if path_str.endswith("task_completion_record.json"):
            operations.append("write_completion_record")
        elif path_str.endswith("task_status.json"):
            operations.append("write_task_status")
        return io.atomic_write_json(path, data)

    def track_append(run, event, base_dir=None):
        operations.append("append_task_event")
        return real_append(run, event, base_dir)

    with patch("htr.events.atomic_write_json", side_effect=track_write), patch(
        "htr.events.append_task_event", side_effect=track_append
    ):
        events.complete_task_manually(
            run_id, task_id, attempt_id, record, base_dir=tmp_path
        )

    assert operations == [
        "write_completion_record",
        "append_task_event",
        "write_task_status",
    ]


def test_complete_task_manually_validates_task_status_before_write(tmp_path):
    run_id, task_id, attempt_id = _verification_passed_setup(tmp_path)
    record = _completion_record(run_id, task_id, attempt_id)
    calls = []

    original_validate = schemas.validate

    def track_validate(data, schema_name):
        calls.append(schema_name)
        return original_validate(data, schema_name)

    with patch("htr.events.validate_schema", side_effect=track_validate):
        events.complete_task_manually(
            run_id, task_id, attempt_id, record, base_dir=tmp_path
        )

    task_status_indexes = [i for i, name in enumerate(calls) if name == "task_status"]
    assert task_status_indexes
    assert calls.index("task_completion_record") < task_status_indexes[-1]


# --- Boundaries ---


def test_complete_task_manually_leaves_attempt_status_unchanged(tmp_path):
    run_id, task_id, attempt_id = _verification_passed_setup(tmp_path)
    before = io.read_json(
        paths.attempt_status_path(run_id, task_id, attempt_id, tmp_path)
    )
    events.complete_task_manually(
        run_id,
        task_id,
        attempt_id,
        _completion_record(run_id, task_id, attempt_id),
        base_dir=tmp_path,
    )
    assert (
        io.read_json(
            paths.attempt_status_path(run_id, task_id, attempt_id, tmp_path)
        )
        == before
    )


def test_complete_task_manually_leaves_run_manifest_unchanged(tmp_path):
    run_id, task_id, attempt_id = _verification_passed_setup(tmp_path)
    before = io.read_json(paths.run_manifest_path(run_id, tmp_path))
    events.complete_task_manually(
        run_id,
        task_id,
        attempt_id,
        _completion_record(run_id, task_id, attempt_id),
        base_dir=tmp_path,
    )
    assert io.read_json(paths.run_manifest_path(run_id, tmp_path)) == before


def test_complete_task_manually_does_not_create_new_attempt(tmp_path):
    run_id, task_id, attempt_id = _verification_passed_setup(tmp_path)
    events.complete_task_manually(
        run_id,
        task_id,
        attempt_id,
        _completion_record(run_id, task_id, attempt_id),
        base_dir=tmp_path,
    )
    status = io.read_json(paths.task_status_path(run_id, task_id, tmp_path))
    assert status["attempts"] == [attempt_id]


# --- Failure / no side effects ---


def test_complete_task_manually_invalid_schema_blocks_side_effects(tmp_path):
    run_id, task_id, attempt_id = _verification_passed_setup(tmp_path)
    record = _completion_record(run_id, task_id, attempt_id)
    record["metadata"] = "bad"
    before = _snapshot_side_effects(tmp_path, run_id, task_id, attempt_id)
    with pytest.raises(ValueError, match="metadata must be a dict"):
        events.complete_task_manually(
            run_id, task_id, attempt_id, record, base_dir=tmp_path
        )
    _assert_unchanged(before, _snapshot_side_effects(tmp_path, run_id, task_id, attempt_id))


@pytest.mark.parametrize(
    "field,other_factory",
    [
        ("run_id", new_run_id),
        ("task_id", new_task_id),
        ("attempt_id", new_attempt_id),
    ],
)
def test_complete_task_manually_identity_mismatch_blocks_side_effects(
    tmp_path, field, other_factory
):
    run_id, task_id, attempt_id = _verification_passed_setup(tmp_path)
    record = _completion_record(run_id, task_id, attempt_id)
    record[field] = other_factory()
    before = _snapshot_side_effects(tmp_path, run_id, task_id, attempt_id)
    with pytest.raises(ValueError, match="completion_record ids do not match"):
        events.complete_task_manually(
            run_id, task_id, attempt_id, record, base_dir=tmp_path
        )
    _assert_unchanged(before, _snapshot_side_effects(tmp_path, run_id, task_id, attempt_id))


@pytest.mark.parametrize(
    "setup_status",
    [
        ATTEMPT_RESULT_SUBMITTED,
        ATTEMPT_VERIFICATION_FAILED,
        ATTEMPT_HEAL_REQUIRED,
        ATTEMPT_RUNNING,
    ],
)
def test_complete_task_manually_rejects_non_verification_passed_attempt(
    tmp_path, setup_status
):
    run_id, task_id = _bootstrap_task(tmp_path)
    events.apply_task_transition(
        run_id, task_id, TASK_RUNNING, actor="test", base_dir=tmp_path
    )
    attempt_id = new_attempt_id()
    events.register_attempt(run_id, task_id, attempt_id, actor="test", base_dir=tmp_path)
    if setup_status != ATTEMPT_RUNNING:
        events.apply_attempt_transition(
            run_id, task_id, attempt_id, ATTEMPT_RUNNING, actor="test", base_dir=tmp_path
        )
    if setup_status == ATTEMPT_RESULT_SUBMITTED:
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
    elif setup_status in {ATTEMPT_VERIFICATION_FAILED, ATTEMPT_HEAL_REQUIRED}:
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
        outcome = "failed" if setup_status == ATTEMPT_VERIFICATION_FAILED else "heal_required"
        events.submit_manual_verification(
            run_id,
            task_id,
            attempt_id,
            contracts.make_verification_result(
                run_id=run_id,
                task_id=task_id,
                attempt_id=attempt_id,
                outcome=outcome,
            ),
            base_dir=tmp_path,
        )
    elif setup_status == ATTEMPT_RUNNING:
        pass
    before = _snapshot_side_effects(tmp_path, run_id, task_id, attempt_id)
    with pytest.raises(InvalidTransition):
        events.complete_task_manually(
            run_id,
            task_id,
            attempt_id,
            _completion_record(run_id, task_id, attempt_id),
            base_dir=tmp_path,
        )
    _assert_unchanged(before, _snapshot_side_effects(tmp_path, run_id, task_id, attempt_id))


def test_complete_task_manually_invalid_task_transition_blocks_side_effects(tmp_path):
    run_id, task_id, attempt_id = _verification_passed_setup(
        tmp_path, task_running=False
    )
    before = _snapshot_side_effects(tmp_path, run_id, task_id, attempt_id)
    with pytest.raises(InvalidTransition):
        events.complete_task_manually(
            run_id,
            task_id,
            attempt_id,
            _completion_record(run_id, task_id, attempt_id),
            base_dir=tmp_path,
        )
    _assert_unchanged(before, _snapshot_side_effects(tmp_path, run_id, task_id, attempt_id))


def test_complete_task_manually_existing_record_while_not_completed_fails(tmp_path):
    run_id, task_id, attempt_id = _verification_passed_setup(tmp_path)
    record = _completion_record(run_id, task_id, attempt_id)
    path = contracts.task_completion_record_json_path(run_id, task_id, tmp_path)
    io.atomic_write_json(path, record)
    before = _snapshot_side_effects(tmp_path, run_id, task_id, attempt_id)
    with pytest.raises(InvalidTransition, match="task_completion_record.json exists"):
        events.complete_task_manually(
            run_id, task_id, attempt_id, record, base_dir=tmp_path
        )
    _assert_unchanged(before, _snapshot_side_effects(tmp_path, run_id, task_id, attempt_id))


# --- Idempotency / replay ---


def test_complete_task_manually_idempotent_same_event_id(tmp_path):
    run_id, task_id, attempt_id = _verification_passed_setup(tmp_path)
    record = _completion_record(run_id, task_id, attempt_id)
    event_id = new_event_id()
    first = events.complete_task_manually(
        run_id, task_id, attempt_id, record, event_id=event_id, base_dir=tmp_path
    )
    second = events.complete_task_manually(
        run_id, task_id, attempt_id, record, event_id=event_id, base_dir=tmp_path
    )
    assert second == first


def test_complete_task_manually_same_event_id_different_record_raises_conflict(tmp_path):
    run_id, task_id, attempt_id = _verification_passed_setup(tmp_path)
    event_id = new_event_id()
    first = _completion_record(run_id, task_id, attempt_id, reason="first")
    events.complete_task_manually(
        run_id, task_id, attempt_id, first, event_id=event_id, base_dir=tmp_path
    )
    second = _completion_record(run_id, task_id, attempt_id, reason="second")
    with pytest.raises(EventConflict):
        events.complete_task_manually(
            run_id, task_id, attempt_id, second, event_id=event_id, base_dir=tmp_path
        )


def test_complete_task_manually_replay_does_not_rewrite_record_event_or_status(tmp_path):
    run_id, task_id, attempt_id = _verification_passed_setup(tmp_path)
    record = _completion_record(run_id, task_id, attempt_id)
    event_id = new_event_id()
    events.complete_task_manually(
        run_id,
        task_id,
        attempt_id,
        record,
        event_id=event_id,
        actor="human",
        base_dir=tmp_path,
    )
    completion_path = contracts.task_completion_record_json_path(run_id, task_id, tmp_path)
    completion_bytes = completion_path.read_bytes()
    task_status_before = io.read_json(paths.task_status_path(run_id, task_id, tmp_path))
    attempt_status_before = io.read_json(
        paths.attempt_status_path(run_id, task_id, attempt_id, tmp_path)
    )
    event_count = len(events.read_task_events(run_id, base_dir=tmp_path))

    events.complete_task_manually(
        run_id,
        task_id,
        attempt_id,
        record,
        event_id=event_id,
        actor="human",
        base_dir=tmp_path,
    )

    assert completion_path.read_bytes() == completion_bytes
    assert io.read_json(paths.task_status_path(run_id, task_id, tmp_path)) == task_status_before
    assert (
        io.read_json(
            paths.attempt_status_path(run_id, task_id, attempt_id, tmp_path)
        )
        == attempt_status_before
    )
    assert len(events.read_task_events(run_id, base_dir=tmp_path)) == event_count


def test_complete_task_manually_different_event_id_after_completed_raises(tmp_path):
    run_id, task_id, attempt_id = _verification_passed_setup(tmp_path)
    record = _completion_record(run_id, task_id, attempt_id)
    events.complete_task_manually(
        run_id, task_id, attempt_id, record, event_id=new_event_id(), base_dir=tmp_path
    )
    with pytest.raises(InvalidTransition):
        events.complete_task_manually(
            run_id,
            task_id,
            attempt_id,
            record,
            event_id=new_event_id(),
            base_dir=tmp_path,
        )


def test_complete_task_manually_event_id_none_after_completed_raises(tmp_path):
    run_id, task_id, attempt_id = _verification_passed_setup(tmp_path)
    record = _completion_record(run_id, task_id, attempt_id)
    events.complete_task_manually(
        run_id, task_id, attempt_id, record, event_id=new_event_id(), base_dir=tmp_path
    )
    with pytest.raises(InvalidTransition):
        events.complete_task_manually(
            run_id, task_id, attempt_id, record, base_dir=tmp_path
        )


def test_complete_task_manually_replay_event_lookup_scoped_to_task_id(tmp_path):
    run_id = new_run_id()
    task_a = new_task_id()
    task_b = new_task_id()
    io.create_run_workspace(run_id, base_dir=tmp_path)
    io.create_task_workspace(run_id, task_a, base_dir=tmp_path)
    io.create_task_workspace(run_id, task_b, base_dir=tmp_path)
    attempt_a = _verification_passed_setup_for_task(tmp_path, run_id, task_a)
    attempt_b = _verification_passed_setup_for_task(tmp_path, run_id, task_b)
    record_a = _completion_record(run_id, task_a, attempt_a)
    event_id_a = new_event_id()
    events.complete_task_manually(
        run_id, task_a, attempt_a, record_a, event_id=event_id_a, base_dir=tmp_path
    )
    record_b = _completion_record(run_id, task_b, attempt_b)
    events.complete_task_manually(
        run_id,
        task_b,
        attempt_b,
        record_b,
        event_id=new_event_id(),
        base_dir=tmp_path,
    )
    with pytest.raises(InvalidTransition):
        events.complete_task_manually(
            run_id, task_b, attempt_b, record_b, event_id=event_id_a, base_dir=tmp_path
        )


def _verification_passed_setup_for_task(tmp_path, run_id, task_id):
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
    events.submit_manual_verification(
        run_id,
        task_id,
        attempt_id,
        contracts.make_verification_result(
            run_id=run_id, task_id=task_id, attempt_id=attempt_id, outcome="passed"
        ),
        base_dir=tmp_path,
    )
    return attempt_id


# --- Import boundary ---


def test_htr_task5_modules_do_not_import_forbidden_modules():
    repo_root = Path(__file__).resolve().parents[2]
    forbidden = {"runtime", "delegate_task", "deco", "heal", "scheduler"}
    for relative in (
        "htr/contracts.py",
        "htr/events.py",
        "htr/schemas.py",
        "htr/state.py",
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


def test_htr_package_exports_task5_apis():
    import htr

    for name in (
        "make_task_completion_record",
        "task_completion_fingerprint",
        "complete_task_manually",
        "EVENT_TYPE_MANUAL_TASK_COMPLETED",
    ):
        assert hasattr(htr, name)
