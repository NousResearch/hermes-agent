from __future__ import annotations

import pytest

from agent.personal_computer.tasks import PersonalComputerTaskStore, TaskStatus


def make_store(tmp_path):
    return PersonalComputerTaskStore(tmp_path / "hermes" / "personal-computer.sqlite3")


def create(store, **overrides):
    values = {
        "title": "Prepare a draft email",
        "requester_id": "user-1",
        "source": "whatsapp",
        "profile_id": "default",
        "idempotency_key": "message-123",
    }
    values.update(overrides)
    return store.create_task(**values)


def test_create_and_restore_task_after_reopen(tmp_path):
    path = tmp_path / "tasks.sqlite3"
    task = PersonalComputerTaskStore(path).create_task(
        title="Review invoice",
        requester_id="user-1",
        source="desktop",
        profile_id="default",
    )

    restored = PersonalComputerTaskStore(path).get_task(task.id)

    assert restored == task
    assert restored.status is TaskStatus.QUEUED


def test_duplicate_message_delivery_returns_same_task(tmp_path):
    store = make_store(tmp_path)

    first = create(store)
    duplicate = create(store)

    assert duplicate.id == first.id
    assert len(store.list_tasks(profile_id="default")) == 1


def test_task_lists_are_scoped_to_profile(tmp_path):
    store = make_store(tmp_path)
    create(store, idempotency_key="a", profile_id="personal")
    create(store, idempotency_key="b", profile_id="school")

    assert [task.profile_id for task in store.list_tasks(profile_id="personal")] == ["personal"]
    assert [task.profile_id for task in store.list_tasks(profile_id="school")] == ["school"]


def test_approval_is_explicit_and_does_not_execute_action(tmp_path):
    store = make_store(tmp_path)
    task = create(store)
    store.transition(task.id, TaskStatus.RUNNING)
    waiting = store.transition(task.id, TaskStatus.AWAITING_APPROVAL, summary="Draft ready")

    approved = store.approve(waiting.id, approved=True)

    assert approved.status is TaskStatus.RUNNING
    assert approved.approval_state == "approved"


def test_rejecting_approval_cancels_task(tmp_path):
    store = make_store(tmp_path)
    task = create(store)
    store.transition(task.id, TaskStatus.RUNNING)
    waiting = store.transition(task.id, TaskStatus.AWAITING_APPROVAL)

    rejected = store.approve(waiting.id, approved=False)

    assert rejected.status is TaskStatus.CANCELLED
    assert rejected.approval_state == "rejected"


def test_invalid_transitions_are_rejected(tmp_path):
    store = make_store(tmp_path)
    task = create(store)

    with pytest.raises(ValueError, match="Invalid task transition"):
        store.transition(task.id, TaskStatus.COMPLETED)


def test_expected_status_prevents_stale_updates(tmp_path):
    store = make_store(tmp_path)
    task = create(store)
    store.transition(task.id, TaskStatus.RUNNING)

    with pytest.raises(ValueError, match="Task state changed"):
        store.transition(task.id, TaskStatus.COMPLETED, expected_status=TaskStatus.QUEUED)


def test_invalid_limit_is_rejected(tmp_path):
    store = make_store(tmp_path)

    with pytest.raises(ValueError, match="limit must be between"):
        store.list_tasks(profile_id="default", limit=0)
