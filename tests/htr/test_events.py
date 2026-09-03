import pytest

from htr import events, io, paths
from htr.ids import new_attempt_id, new_event_id, new_run_id, new_task_id
from htr.state import (
    ATTEMPT_RUNNING,
    AttemptAlreadyRegistered,
    EventConflict,
    InvalidTransition,
    TASK_RUNNING,
)


def _bootstrap_task(tmp_path):
    run_id = new_run_id()
    task_id = new_task_id()
    io.create_run_workspace(run_id, base_dir=tmp_path)
    io.create_task_workspace(run_id, task_id, base_dir=tmp_path)
    return run_id, task_id


def test_append_and_read_event_round_trip(tmp_path):
    run_id, task_id = _bootstrap_task(tmp_path)
    event = events.make_event(
        event_type=events.EVENT_TYPE_TASK_STATUS_CHANGED,
        run_id=run_id,
        task_id=task_id,
        actor="test",
        previous_status="created",
        new_status=TASK_RUNNING,
    )
    events.append_task_event(run_id, event, base_dir=tmp_path)
    rows = events.read_task_events(run_id, base_dir=tmp_path)
    assert rows == [event]


def test_event_id_generated_if_absent(tmp_path):
    run_id, task_id = _bootstrap_task(tmp_path)
    event = events.make_event(
        event_type=events.EVENT_TYPE_TASK_STATUS_CHANGED,
        run_id=run_id,
        task_id=task_id,
        actor="test",
    )
    assert event["event_id"].startswith("evt_")


def test_duplicate_same_event_id_same_semantics_is_idempotent(tmp_path):
    run_id, task_id = _bootstrap_task(tmp_path)
    event_id = new_event_id()
    existing_event = events.make_event(
        event_type=events.EVENT_TYPE_TASK_STATUS_CHANGED,
        run_id=run_id,
        task_id=task_id,
        actor="test",
        event_id=event_id,
        previous_status="created",
        new_status=TASK_RUNNING,
    )
    events.append_task_event(run_id, existing_event, base_dir=tmp_path)
    assert io.read_json(paths.task_status_path(run_id, task_id, tmp_path))["status"] == "created"

    second = events.apply_task_transition(
        run_id,
        task_id,
        TASK_RUNNING,
        actor="test",
        event_id=event_id,
        base_dir=tmp_path,
    )
    assert second == existing_event
    assert len(events.read_task_events(run_id, base_dir=tmp_path)) == 1


def test_apply_task_transition_duplicate_event_id_invalid_current_transition_raises(
    tmp_path,
):
    run_id, task_id = _bootstrap_task(tmp_path)
    event_id = new_event_id()
    events.apply_task_transition(
        run_id,
        task_id,
        TASK_RUNNING,
        actor="test",
        event_id=event_id,
        base_dir=tmp_path,
    )

    with pytest.raises(InvalidTransition):
        events.apply_task_transition(
            run_id,
            task_id,
            TASK_RUNNING,
            actor="test",
            event_id=event_id,
            base_dir=tmp_path,
        )


def test_apply_attempt_transition_duplicate_event_id_invalid_current_transition_raises(
    tmp_path,
):
    run_id, task_id = _bootstrap_task(tmp_path)
    attempt_id = new_attempt_id()
    event_id = new_event_id()
    events.register_attempt(
        run_id,
        task_id,
        attempt_id,
        actor="test",
        base_dir=tmp_path,
    )
    events.apply_attempt_transition(
        run_id,
        task_id,
        attempt_id,
        ATTEMPT_RUNNING,
        actor="test",
        event_id=event_id,
        base_dir=tmp_path,
    )

    with pytest.raises(InvalidTransition):
        events.apply_attempt_transition(
            run_id,
            task_id,
            attempt_id,
            ATTEMPT_RUNNING,
            actor="test",
            event_id=event_id,
            base_dir=tmp_path,
        )


def test_semantic_fingerprint_different_previous_status_raises_event_conflict(tmp_path):
    run_id, task_id = _bootstrap_task(tmp_path)
    event_id = new_event_id()
    events.append_task_event(
        run_id,
        events.make_event(
            event_type=events.EVENT_TYPE_TASK_STATUS_CHANGED,
            run_id=run_id,
            task_id=task_id,
            actor="test",
            event_id=event_id,
            previous_status="created",
            new_status=TASK_RUNNING,
        ),
        base_dir=tmp_path,
    )
    io.atomic_write_json(
        paths.task_status_path(run_id, task_id, tmp_path),
        {
            "task_id": task_id,
            "run_id": run_id,
            "status": "blocked",
            "attempts": [],
        },
    )

    with pytest.raises(EventConflict):
        events.apply_task_transition(
            run_id,
            task_id,
            TASK_RUNNING,
            actor="test",
            event_id=event_id,
            base_dir=tmp_path,
        )


def test_duplicate_same_event_id_conflicting_semantics_raises_event_conflict(tmp_path):
    run_id, task_id = _bootstrap_task(tmp_path)
    event_id = new_event_id()
    events.apply_task_transition(
        run_id,
        task_id,
        TASK_RUNNING,
        actor="test",
        event_id=event_id,
        base_dir=tmp_path,
    )
    with pytest.raises(EventConflict):
        events.apply_task_transition(
            run_id,
            task_id,
            "blocked",
            actor="test",
            event_id=event_id,
            base_dir=tmp_path,
        )


def test_apply_task_transition_created_to_running_updates_status_and_appends_event(
    tmp_path,
):
    run_id, task_id = _bootstrap_task(tmp_path)
    event = events.apply_task_transition(
        run_id,
        task_id,
        TASK_RUNNING,
        actor="test",
        base_dir=tmp_path,
    )
    status = io.read_json(paths.task_status_path(run_id, task_id, tmp_path))
    assert status["status"] == TASK_RUNNING
    assert events.read_task_events(run_id, base_dir=tmp_path) == [event]
    assert event["event_type"] == events.EVENT_TYPE_TASK_STATUS_CHANGED


def test_illegal_task_transition_does_not_append_event_or_update_status(tmp_path):
    run_id, task_id = _bootstrap_task(tmp_path)
    status_path = paths.task_status_path(run_id, task_id, tmp_path)
    events_path = paths.task_events_path(run_id, tmp_path)

    with pytest.raises(InvalidTransition):
        events.apply_task_transition(
            run_id,
            task_id,
            "completed",
            actor="test",
            base_dir=tmp_path,
        )

    assert io.read_json(status_path)["status"] == "created"
    assert events.read_task_events(run_id, base_dir=tmp_path) == []
    assert events_path.read_text(encoding="utf-8") == ""


def test_register_attempt_creates_workspace_and_appends_attempt_id(tmp_path):
    run_id, task_id = _bootstrap_task(tmp_path)
    attempt_id = new_attempt_id()

    event = events.register_attempt(
        run_id,
        task_id,
        attempt_id,
        actor="test",
        base_dir=tmp_path,
    )

    status = io.read_json(paths.task_status_path(run_id, task_id, tmp_path))
    assert status["attempts"] == [attempt_id]
    assert paths.attempt_status_path(run_id, task_id, attempt_id, tmp_path).exists()
    assert event["event_type"] == events.EVENT_TYPE_ATTEMPT_REGISTERED


def test_register_attempt_same_event_id_same_semantics_retry_returns_existing(tmp_path):
    run_id, task_id = _bootstrap_task(tmp_path)
    attempt_id = new_attempt_id()
    event_id = new_event_id()

    first = events.register_attempt(
        run_id,
        task_id,
        attempt_id,
        actor="test",
        event_id=event_id,
        base_dir=tmp_path,
    )
    second = events.register_attempt(
        run_id,
        task_id,
        attempt_id,
        actor="test",
        event_id=event_id,
        base_dir=tmp_path,
    )

    assert second == first
    status = io.read_json(paths.task_status_path(run_id, task_id, tmp_path))
    assert status["attempts"] == [attempt_id]
    assert len(events.read_task_events(run_id, base_dir=tmp_path)) == 1


def test_register_attempt_does_not_duplicate_attempts(tmp_path):
    run_id, task_id = _bootstrap_task(tmp_path)
    attempt_id = new_attempt_id()
    event_id = new_event_id()

    events.register_attempt(
        run_id,
        task_id,
        attempt_id,
        actor="test",
        event_id=event_id,
        base_dir=tmp_path,
    )
    events.register_attempt(
        run_id,
        task_id,
        attempt_id,
        actor="test",
        event_id=event_id,
        base_dir=tmp_path,
    )

    status = io.read_json(paths.task_status_path(run_id, task_id, tmp_path))
    assert status["attempts"] == [attempt_id]
    assert len(events.read_task_events(run_id, base_dir=tmp_path)) == 1


def test_register_attempt_same_attempt_id_with_different_event_id_rejected(tmp_path):
    run_id, task_id = _bootstrap_task(tmp_path)
    attempt_id = new_attempt_id()

    events.register_attempt(
        run_id,
        task_id,
        attempt_id,
        actor="test",
        event_id=new_event_id(),
        base_dir=tmp_path,
    )

    with pytest.raises(AttemptAlreadyRegistered):
        events.register_attempt(
            run_id,
            task_id,
            attempt_id,
            actor="test",
            event_id=new_event_id(),
            base_dir=tmp_path,
        )


def test_apply_attempt_transition_created_to_running_updates_status_and_appends_event(
    tmp_path,
):
    run_id, task_id = _bootstrap_task(tmp_path)
    attempt_id = new_attempt_id()
    events.register_attempt(
        run_id,
        task_id,
        attempt_id,
        actor="test",
        base_dir=tmp_path,
    )

    event = events.apply_attempt_transition(
        run_id,
        task_id,
        attempt_id,
        ATTEMPT_RUNNING,
        actor="test",
        base_dir=tmp_path,
    )

    attempt_status = io.read_json(
        paths.attempt_status_path(run_id, task_id, attempt_id, tmp_path)
    )
    assert attempt_status["status"] == ATTEMPT_RUNNING
    assert len(events.read_task_events(run_id, base_dir=tmp_path)) == 2
    assert event["event_type"] == events.EVENT_TYPE_ATTEMPT_STATUS_CHANGED


def test_illegal_attempt_transition_does_not_append_event_or_update_status(tmp_path):
    run_id, task_id = _bootstrap_task(tmp_path)
    attempt_id = new_attempt_id()
    events.register_attempt(
        run_id,
        task_id,
        attempt_id,
        actor="test",
        base_dir=tmp_path,
    )
    attempt_status_path = paths.attempt_status_path(
        run_id, task_id, attempt_id, tmp_path
    )
    event_count_before = len(events.read_task_events(run_id, base_dir=tmp_path))

    with pytest.raises(InvalidTransition):
        events.apply_attempt_transition(
            run_id,
            task_id,
            attempt_id,
            "completed",
            actor="test",
            base_dir=tmp_path,
        )

    assert io.read_json(attempt_status_path)["status"] == "created"
    assert len(events.read_task_events(run_id, base_dir=tmp_path)) == event_count_before


def test_status_updates_preserve_unrelated_fields(tmp_path):
    run_id, task_id = _bootstrap_task(tmp_path)
    status_path = paths.task_status_path(run_id, task_id, tmp_path)
    io.atomic_write_json(
        status_path,
        {
            "task_id": task_id,
            "run_id": run_id,
            "status": "created",
            "attempts": [],
            "owner": "qa-team",
        },
    )

    events.apply_task_transition(
        run_id,
        task_id,
        TASK_RUNNING,
        actor="test",
        base_dir=tmp_path,
    )

    status = io.read_json(status_path)
    assert status["status"] == TASK_RUNNING
    assert status["owner"] == "qa-team"


def test_repeated_reads_tolerate_empty_jsonl(tmp_path):
    run_id, task_id = _bootstrap_task(tmp_path)
    assert events.read_task_events(run_id, base_dir=tmp_path) == []
    assert events.read_task_events(run_id, base_dir=tmp_path) == []
