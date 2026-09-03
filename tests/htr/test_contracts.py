import pytest

from htr import contracts, events, io, paths
from htr.ids import new_attempt_id, new_event_id, new_run_id, new_task_id
from htr.schemas import validate
from htr.state import (
    ATTEMPT_RESULT_SUBMITTED,
    ATTEMPT_RUNNING,
    EventConflict,
    InvalidTransition,
)


def _bootstrap_task(tmp_path):
    run_id = new_run_id()
    task_id = new_task_id()
    io.create_run_workspace(run_id, base_dir=tmp_path)
    io.create_task_workspace(run_id, task_id, base_dir=tmp_path)
    return run_id, task_id


def test_make_task_card_returns_valid_schema():
    run_id = new_run_id()
    task_id = new_task_id()
    card = contracts.make_task_card(
        run_id=run_id,
        task_id=task_id,
        title="Demo",
        instruction="Do the thing",
        created_by="architect",
    )
    validate(card, "task_card")
    assert card["schema_version"] == "1"


def test_write_task_card_writes_expected_file(tmp_path):
    run_id, task_id = _bootstrap_task(tmp_path)
    card = contracts.make_task_card(
        run_id=run_id,
        task_id=task_id,
        title="Demo",
        instruction="Do the thing",
        created_by="architect",
    )
    target = contracts.write_task_card(run_id, task_id, card, base_dir=tmp_path)
    assert target == contracts.task_card_json_path(run_id, task_id, tmp_path)
    assert target.exists()


def test_read_task_card_returns_same_validated_object(tmp_path):
    run_id, task_id = _bootstrap_task(tmp_path)
    card = contracts.make_task_card(
        run_id=run_id,
        task_id=task_id,
        title="Demo",
        instruction="Do the thing",
        created_by="architect",
        inputs={"source": "manual"},
    )
    contracts.write_task_card(run_id, task_id, card, base_dir=tmp_path)
    assert contracts.read_task_card(run_id, task_id, base_dir=tmp_path) == card


@pytest.mark.parametrize("field", ["inputs", "constraints", "acceptance", "metadata"])
def test_task_card_dict_fields_must_be_dicts(field):
    run_id = new_run_id()
    task_id = new_task_id()
    card = contracts.make_task_card(
        run_id=run_id,
        task_id=task_id,
        title="Demo",
        instruction="Do the thing",
        created_by="architect",
    )
    card[field] = "not-a-dict"
    with pytest.raises(ValueError):
        validate(card, "task_card")


@pytest.mark.parametrize("field", ["inputs", "constraints", "acceptance", "metadata"])
def test_make_task_card_list_fields_fail_validation_not_coerced(field):
    with pytest.raises(ValueError, match=f"{field} must be a dict"):
        contracts.make_task_card(
            run_id=new_run_id(),
            task_id=new_task_id(),
            title="Demo",
            instruction="Do the thing",
            created_by="architect",
            **{field: []},
        )


def test_write_task_card_rejects_mismatched_run_id(tmp_path):
    run_id, task_id = _bootstrap_task(tmp_path)
    other_run_id = new_run_id()
    card = contracts.make_task_card(
        run_id=other_run_id,
        task_id=task_id,
        title="Demo",
        instruction="Do the thing",
        created_by="architect",
    )
    with pytest.raises(ValueError, match="run_id/task_id do not match"):
        contracts.write_task_card(run_id, task_id, card, base_dir=tmp_path)


def test_write_task_card_rejects_mismatched_task_id(tmp_path):
    run_id, task_id = _bootstrap_task(tmp_path)
    other_task_id = new_task_id()
    card = contracts.make_task_card(
        run_id=run_id,
        task_id=other_task_id,
        title="Demo",
        instruction="Do the thing",
        created_by="architect",
    )
    with pytest.raises(ValueError, match="run_id/task_id do not match"):
        contracts.write_task_card(run_id, task_id, card, base_dir=tmp_path)


def test_write_task_card_does_not_append_lifecycle_event(tmp_path):
    run_id, task_id = _bootstrap_task(tmp_path)
    card = contracts.make_task_card(
        run_id=run_id,
        task_id=task_id,
        title="Demo",
        instruction="Do the thing",
        created_by="architect",
    )
    contracts.write_task_card(run_id, task_id, card, base_dir=tmp_path)
    assert events.read_task_events(run_id, base_dir=tmp_path) == []


def test_write_task_card_does_not_change_task_status(tmp_path):
    run_id, task_id = _bootstrap_task(tmp_path)
    status_path = paths.task_status_path(run_id, task_id, tmp_path)
    before = io.read_json(status_path)
    card = contracts.make_task_card(
        run_id=run_id,
        task_id=task_id,
        title="Demo",
        instruction="Do the thing",
        created_by="architect",
    )
    contracts.write_task_card(run_id, task_id, card, base_dir=tmp_path)
    assert io.read_json(status_path) == before


def test_write_task_card_does_not_create_attempts(tmp_path):
    run_id, task_id = _bootstrap_task(tmp_path)
    card = contracts.make_task_card(
        run_id=run_id,
        task_id=task_id,
        title="Demo",
        instruction="Do the thing",
        created_by="architect",
    )
    contracts.write_task_card(run_id, task_id, card, base_dir=tmp_path)
    status = io.read_json(paths.task_status_path(run_id, task_id, tmp_path))
    assert status["attempts"] == []
    assert list(paths.attempts_dir(run_id, task_id, tmp_path).iterdir()) == []


def test_make_attempt_result_returns_valid_schema():
    result = contracts.make_attempt_result(
        run_id=new_run_id(),
        task_id=new_task_id(),
        attempt_id=new_attempt_id(),
        produced_by="worker",
        summary="done",
    )
    validate(result, "attempt_result")


@pytest.mark.parametrize("field,value", [("outputs", "x"), ("artifacts", {}), ("metrics", []), ("metadata", [])])
def test_make_attempt_result_rejects_invalid_field_types(field, value):
    with pytest.raises(ValueError):
        contracts.make_attempt_result(
            run_id=new_run_id(),
            task_id=new_task_id(),
            attempt_id=new_attempt_id(),
            produced_by="worker",
            summary="done",
            **{field: value},
        )


def _running_attempt(tmp_path):
    run_id, task_id = _bootstrap_task(tmp_path)
    attempt_id = new_attempt_id()
    events.register_attempt(run_id, task_id, attempt_id, actor="test", base_dir=tmp_path)
    events.apply_attempt_transition(
        run_id,
        task_id,
        attempt_id,
        ATTEMPT_RUNNING,
        actor="test",
        base_dir=tmp_path,
    )
    return run_id, task_id, attempt_id


def test_submit_attempt_result_from_running_succeeds(tmp_path):
    run_id, task_id, attempt_id = _running_attempt(tmp_path)
    result = contracts.make_attempt_result(
        run_id=run_id,
        task_id=task_id,
        attempt_id=attempt_id,
        produced_by="worker",
        summary="done",
    )
    event = events.submit_attempt_result(
        run_id, task_id, attempt_id, result, base_dir=tmp_path
    )
    assert event["new_status"] == ATTEMPT_RESULT_SUBMITTED
    assert io.read_json(
        paths.attempt_status_path(run_id, task_id, attempt_id, tmp_path)
    )["status"] == ATTEMPT_RESULT_SUBMITTED


def test_submit_attempt_result_invalid_status_with_existing_event_id_raises(
    tmp_path,
):
    run_id, task_id = _bootstrap_task(tmp_path)
    attempt_id = new_attempt_id()
    events.register_attempt(run_id, task_id, attempt_id, actor="test", base_dir=tmp_path)
    result = contracts.make_attempt_result(
        run_id=run_id,
        task_id=task_id,
        attempt_id=attempt_id,
        produced_by="worker",
        summary="done",
    )
    event_id = new_event_id()
    result_path = paths.result_json_path(run_id, task_id, attempt_id, tmp_path)
    events.append_task_event(
        run_id,
        events.make_event(
            event_type=events.EVENT_TYPE_ATTEMPT_RESULT_SUBMITTED,
            run_id=run_id,
            task_id=task_id,
            actor="system",
            event_id=event_id,
            attempt_id=attempt_id,
            previous_status=ATTEMPT_RUNNING,
            new_status=ATTEMPT_RESULT_SUBMITTED,
            payload={
                "result_path": str(result_path),
                "result_fingerprint": contracts.result_fingerprint(result),
            },
        ),
        base_dir=tmp_path,
    )

    with pytest.raises(InvalidTransition):
        events.submit_attempt_result(
            run_id,
            task_id,
            attempt_id,
            result,
            event_id=event_id,
            base_dir=tmp_path,
        )

    assert not result_path.exists()
    assert (
        io.read_json(
            paths.attempt_status_path(run_id, task_id, attempt_id, tmp_path)
        )["status"]
        == "created"
    )


def test_submit_attempt_result_writes_result_json(tmp_path):
    run_id, task_id, attempt_id = _running_attempt(tmp_path)
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
    assert io.read_json(
        paths.result_json_path(run_id, task_id, attempt_id, tmp_path)
    ) == result


def test_submit_attempt_result_moves_attempt_to_result_submitted(tmp_path):
    run_id, task_id, attempt_id = _running_attempt(tmp_path)
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
    status = io.read_json(
        paths.attempt_status_path(run_id, task_id, attempt_id, tmp_path)
    )
    assert status["status"] == ATTEMPT_RESULT_SUBMITTED


def test_submit_attempt_result_appends_result_submitted_event(tmp_path):
    run_id, task_id, attempt_id = _running_attempt(tmp_path)
    result = contracts.make_attempt_result(
        run_id=run_id,
        task_id=task_id,
        attempt_id=attempt_id,
        produced_by="worker",
        summary="done",
    )
    event = events.submit_attempt_result(
        run_id, task_id, attempt_id, result, base_dir=tmp_path
    )
    rows = events.read_task_events(run_id, base_dir=tmp_path)
    assert rows[-1] == event
    assert event["event_type"] == events.EVENT_TYPE_ATTEMPT_RESULT_SUBMITTED


def test_submit_attempt_result_does_not_complete_task(tmp_path):
    run_id, task_id, attempt_id = _running_attempt(tmp_path)
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
    task_status = io.read_json(paths.task_status_path(run_id, task_id, tmp_path))
    assert task_status["status"] == "created"


def test_submit_attempt_result_does_not_invoke_verification_or_heal(tmp_path):
    run_id, task_id, attempt_id = _running_attempt(tmp_path)
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
    attempt_status = io.read_json(
        paths.attempt_status_path(run_id, task_id, attempt_id, tmp_path)
    )["status"]
    assert attempt_status == ATTEMPT_RESULT_SUBMITTED
    assert attempt_status not in {
        "verification_passed",
        "verification_failed",
        "heal_required",
        "completed",
        "failed",
    }


def test_submit_attempt_result_invalid_transition_blocks_submission(tmp_path):
    run_id, task_id = _bootstrap_task(tmp_path)
    attempt_id = new_attempt_id()
    events.register_attempt(run_id, task_id, attempt_id, actor="test", base_dir=tmp_path)
    result = contracts.make_attempt_result(
        run_id=run_id,
        task_id=task_id,
        attempt_id=attempt_id,
        produced_by="worker",
        summary="done",
    )
    with pytest.raises(InvalidTransition):
        events.submit_attempt_result(
            run_id, task_id, attempt_id, result, base_dir=tmp_path
        )
    assert not paths.result_json_path(run_id, task_id, attempt_id, tmp_path).exists()


def test_submit_attempt_result_invalid_result_blocks_side_effects(tmp_path):
    run_id, task_id, attempt_id = _running_attempt(tmp_path)
    result = contracts.make_attempt_result(
        run_id=run_id,
        task_id=task_id,
        attempt_id=attempt_id,
        produced_by="worker",
        summary="done",
    )
    result["outputs"] = "bad"
    event_count = len(events.read_task_events(run_id, base_dir=tmp_path))
    with pytest.raises(ValueError):
        events.submit_attempt_result(
            run_id, task_id, attempt_id, result, base_dir=tmp_path
        )
    assert (
        io.read_json(
            paths.attempt_status_path(run_id, task_id, attempt_id, tmp_path)
        )["status"]
        == ATTEMPT_RUNNING
    )
    assert len(events.read_task_events(run_id, base_dir=tmp_path)) == event_count


def test_submit_attempt_result_idempotent_same_event_id(tmp_path):
    run_id, task_id, attempt_id = _running_attempt(tmp_path)
    result = contracts.make_attempt_result(
        run_id=run_id,
        task_id=task_id,
        attempt_id=attempt_id,
        produced_by="worker",
        summary="done",
    )
    event_id = new_event_id()
    first = events.submit_attempt_result(
        run_id,
        task_id,
        attempt_id,
        result,
        event_id=event_id,
        base_dir=tmp_path,
    )
    second = events.submit_attempt_result(
        run_id,
        task_id,
        attempt_id,
        result,
        event_id=event_id,
        base_dir=tmp_path,
    )
    assert second == first
    assert len(events.read_task_events(run_id, base_dir=tmp_path)) == 3


def test_submit_attempt_result_same_event_id_different_result_raises_conflict(
    tmp_path,
):
    run_id, task_id, attempt_id = _running_attempt(tmp_path)
    event_id = new_event_id()
    first_result = contracts.make_attempt_result(
        run_id=run_id,
        task_id=task_id,
        attempt_id=attempt_id,
        produced_by="worker",
        summary="first",
    )
    events.submit_attempt_result(
        run_id,
        task_id,
        attempt_id,
        first_result,
        event_id=event_id,
        base_dir=tmp_path,
    )
    second_result = contracts.make_attempt_result(
        run_id=run_id,
        task_id=task_id,
        attempt_id=attempt_id,
        produced_by="worker",
        summary="second",
    )
    with pytest.raises(EventConflict):
        events.submit_attempt_result(
            run_id,
            task_id,
            attempt_id,
            second_result,
            event_id=event_id,
            base_dir=tmp_path,
        )


def test_submit_attempt_result_already_submitted_different_event_id_raises(
    tmp_path,
):
    run_id, task_id, attempt_id = _running_attempt(tmp_path)
    result = contracts.make_attempt_result(
        run_id=run_id,
        task_id=task_id,
        attempt_id=attempt_id,
        produced_by="worker",
        summary="done",
    )
    events.submit_attempt_result(
        run_id,
        task_id,
        attempt_id,
        result,
        event_id=new_event_id(),
        base_dir=tmp_path,
    )
    with pytest.raises(InvalidTransition):
        events.submit_attempt_result(
            run_id,
            task_id,
            attempt_id,
            result,
            event_id=new_event_id(),
            base_dir=tmp_path,
        )


def test_result_fingerprint_is_stable_for_same_result():
    run_id = new_run_id()
    task_id = new_task_id()
    attempt_id = new_attempt_id()
    result = contracts.make_attempt_result(
        run_id=run_id,
        task_id=task_id,
        attempt_id=attempt_id,
        produced_by="worker",
        summary="done",
        created_at="2026-07-18T08:00:00+00:00",
    )
    assert contracts.result_fingerprint(result) == contracts.result_fingerprint(result)


def test_result_fingerprint_changes_when_result_changes():
    run_id = new_run_id()
    task_id = new_task_id()
    attempt_id = new_attempt_id()
    first = contracts.make_attempt_result(
        run_id=run_id,
        task_id=task_id,
        attempt_id=attempt_id,
        produced_by="worker",
        summary="first",
        created_at="2026-07-18T08:00:00+00:00",
    )
    second = contracts.make_attempt_result(
        run_id=run_id,
        task_id=task_id,
        attempt_id=attempt_id,
        produced_by="worker",
        summary="second",
        created_at="2026-07-18T08:00:00+00:00",
    )
    assert contracts.result_fingerprint(first) != contracts.result_fingerprint(second)


def test_submit_attempt_result_event_payload_contains_result_fingerprint(tmp_path):
    run_id, task_id, attempt_id = _running_attempt(tmp_path)
    result = contracts.make_attempt_result(
        run_id=run_id,
        task_id=task_id,
        attempt_id=attempt_id,
        produced_by="worker",
        summary="done",
    )
    event = events.submit_attempt_result(
        run_id, task_id, attempt_id, result, base_dir=tmp_path
    )
    assert event["payload"]["result_fingerprint"] == contracts.result_fingerprint(result)


def test_submit_attempt_result_mismatched_result_identity_blocks_side_effects(tmp_path):
    run_id, task_id, attempt_id = _running_attempt(tmp_path)
    other_attempt_id = new_attempt_id()
    result = contracts.make_attempt_result(
        run_id=run_id,
        task_id=task_id,
        attempt_id=other_attempt_id,
        produced_by="worker",
        summary="done",
    )
    event_count = len(events.read_task_events(run_id, base_dir=tmp_path))
    with pytest.raises(ValueError, match="attempt_result ids do not match"):
        events.submit_attempt_result(
            run_id, task_id, attempt_id, result, base_dir=tmp_path
        )
    assert not paths.result_json_path(run_id, task_id, attempt_id, tmp_path).exists()
    assert (
        io.read_json(
            paths.attempt_status_path(run_id, task_id, attempt_id, tmp_path)
        )["status"]
        == ATTEMPT_RUNNING
    )
    assert len(events.read_task_events(run_id, base_dir=tmp_path)) == event_count


def test_submit_attempt_result_invalid_transition_does_not_append_event(tmp_path):
    run_id, task_id = _bootstrap_task(tmp_path)
    attempt_id = new_attempt_id()
    events.register_attempt(run_id, task_id, attempt_id, actor="test", base_dir=tmp_path)
    result = contracts.make_attempt_result(
        run_id=run_id,
        task_id=task_id,
        attempt_id=attempt_id,
        produced_by="worker",
        summary="done",
    )
    event_count = len(events.read_task_events(run_id, base_dir=tmp_path))
    with pytest.raises(InvalidTransition):
        events.submit_attempt_result(
            run_id, task_id, attempt_id, result, base_dir=tmp_path
        )
    assert len(events.read_task_events(run_id, base_dir=tmp_path)) == event_count
    assert (
        io.read_json(
            paths.attempt_status_path(run_id, task_id, attempt_id, tmp_path)
        )["status"]
        == "created"
    )


def test_submit_attempt_result_replay_does_not_rewrite_result_event_or_status(
    tmp_path,
):
    run_id, task_id, attempt_id = _running_attempt(tmp_path)
    result = contracts.make_attempt_result(
        run_id=run_id,
        task_id=task_id,
        attempt_id=attempt_id,
        produced_by="worker",
        summary="done",
    )
    event_id = new_event_id()
    events.submit_attempt_result(
        run_id,
        task_id,
        attempt_id,
        result,
        event_id=event_id,
        actor="system",
        base_dir=tmp_path,
    )
    result_path = paths.result_json_path(run_id, task_id, attempt_id, tmp_path)
    result_bytes = result_path.read_bytes()
    status_before = io.read_json(
        paths.attempt_status_path(run_id, task_id, attempt_id, tmp_path)
    )
    event_count = len(events.read_task_events(run_id, base_dir=tmp_path))

    events.submit_attempt_result(
        run_id,
        task_id,
        attempt_id,
        result,
        event_id=event_id,
        actor="system",
        base_dir=tmp_path,
    )

    assert result_path.read_bytes() == result_bytes
    assert (
        io.read_json(
            paths.attempt_status_path(run_id, task_id, attempt_id, tmp_path)
        )
        == status_before
    )
    assert len(events.read_task_events(run_id, base_dir=tmp_path)) == event_count


def test_submit_attempt_result_same_event_id_different_actor_raises_conflict(tmp_path):
    run_id, task_id, attempt_id = _running_attempt(tmp_path)
    result = contracts.make_attempt_result(
        run_id=run_id,
        task_id=task_id,
        attempt_id=attempt_id,
        produced_by="worker",
        summary="done",
    )
    event_id = new_event_id()
    events.submit_attempt_result(
        run_id,
        task_id,
        attempt_id,
        result,
        event_id=event_id,
        actor="system",
        base_dir=tmp_path,
    )
    with pytest.raises(EventConflict):
        events.submit_attempt_result(
            run_id,
            task_id,
            attempt_id,
            result,
            event_id=event_id,
            actor="other-actor",
            base_dir=tmp_path,
        )


def test_htr_package_exports_task3_apis():
    import htr

    for name in (
        "make_task_card",
        "write_task_card",
        "read_task_card",
        "make_attempt_result",
        "submit_attempt_result",
        "read_artifact_manifest",
        "write_artifact_manifest",
        "add_artifact",
        "list_artifacts",
        "compute_sha256",
        "ArtifactConflict",
    ):
        assert hasattr(htr, name)


def test_compute_sha256_known_hash(tmp_path):
    target = tmp_path / "hello.bin"
    target.write_bytes(b"abc")
    assert contracts.compute_sha256(target) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_compute_sha256_handles_binary_content(tmp_path):
    target = tmp_path / "binary.bin"
    target.write_bytes(bytes(range(256)))
    digest = contracts.compute_sha256(target)
    assert len(digest) == 64
    assert digest == digest.lower()


def test_compute_sha256_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        contracts.compute_sha256(tmp_path / "missing.bin")
