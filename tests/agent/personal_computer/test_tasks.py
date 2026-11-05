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


def test_idempotency_keys_are_scoped_to_profile(tmp_path):
    store = make_store(tmp_path)

    personal = create(store, profile_id="personal", idempotency_key="shared-key")
    school = create(store, profile_id="school", idempotency_key="shared-key")
    personal_duplicate = create(store, profile_id="personal", idempotency_key="shared-key")

    assert personal.id != school.id
    assert personal_duplicate.id == personal.id
    assert [task.profile_id for task in store.list_tasks(profile_id="personal")] == ["personal"]
    assert [task.profile_id for task in store.list_tasks(profile_id="school")] == ["school"]


def test_empty_status_filter_returns_no_tasks(tmp_path):
    store = make_store(tmp_path)
    create(store)

    assert store.list_tasks(profile_id="default", statuses=set()) == []


def test_direct_transition_cannot_bypass_approval(tmp_path):
    store = make_store(tmp_path)
    task = create(store)
    store.transition(task.id, TaskStatus.RUNNING)
    waiting = store.transition(task.id, TaskStatus.AWAITING_APPROVAL)

    with pytest.raises(ValueError, match="must be approved"):
        store.transition(waiting.id, TaskStatus.RUNNING)

    with pytest.raises(ValueError, match="only be changed by approve"):
        store.transition(waiting.id, TaskStatus.RUNNING, approval_state="approved")

    with pytest.raises(ValueError, match="only be changed by approve"):
        store.transition(waiting.id, TaskStatus.PAUSED, approval_state="approved")

    assert store.get_task(waiting.id).status is TaskStatus.AWAITING_APPROVAL


def test_paused_pending_approval_cannot_resume_without_decision(tmp_path):
    store = make_store(tmp_path)
    task = create(store)
    store.transition(task.id, TaskStatus.RUNNING)
    waiting = store.transition(task.id, TaskStatus.AWAITING_APPROVAL)
    paused = store.transition(waiting.id, TaskStatus.PAUSED)

    with pytest.raises(ValueError, match="must be approved"):
        store.transition(paused.id, TaskStatus.RUNNING)

    approved = store.approve(paused.id, approved=True)
    assert approved.status is TaskStatus.RUNNING
    assert approved.approval_state == "approved"


def test_legacy_idempotency_schema_migrates_without_losing_tasks(tmp_path):
    import sqlite3

    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE personal_computer_tasks (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                requester_id TEXT NOT NULL,
                source TEXT NOT NULL,
                profile_id TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                summary TEXT NOT NULL DEFAULT '',
                workspace_id TEXT NOT NULL DEFAULT 'default',
                idempotency_key TEXT,
                approval_required INTEGER NOT NULL DEFAULT 1,
                approval_state TEXT NOT NULL DEFAULT 'pending',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                UNIQUE(requester_id, source, idempotency_key)
            )
            """
        )
        connection.execute(
            """
            INSERT INTO personal_computer_tasks (
                id, title, requester_id, source, profile_id, status,
                created_at, updated_at, idempotency_key
            ) VALUES ('old-task', 'Old task', 'user-1', 'desktop', 'default',
                      'queued', '2026-01-01', '2026-01-01', 'old-key')
            """
        )

    store = PersonalComputerTaskStore(path)
    restored = store.get_task("old-task")
    other_profile = store.create_task(
        title="New profile task",
        requester_id="user-1",
        source="desktop",
        profile_id="school",
        idempotency_key="old-key",
    )

    assert restored is not None
    assert restored.title == "Old task"
    # Legacy pending state is preserved conservatively; approval cannot be bypassed.
    assert restored.approval_state == "pending"
    with pytest.raises(ValueError, match="must be approved"):
        store.transition(restored.id, TaskStatus.RUNNING)
    approved_legacy = store.approve(restored.id, approved=True)
    assert approved_legacy.status is TaskStatus.RUNNING
    assert other_profile.profile_id == "school"



def test_pending_approval_cannot_be_bypassed_through_queue(tmp_path):
    store = make_store(tmp_path)
    task = create(store)
    store.transition(task.id, TaskStatus.RUNNING)
    waiting = store.transition(task.id, TaskStatus.AWAITING_APPROVAL)
    paused = store.transition(waiting.id, TaskStatus.PAUSED)
    queued = store.transition(paused.id, TaskStatus.QUEUED)

    with pytest.raises(ValueError, match="must be approved"):
        store.transition(queued.id, TaskStatus.RUNNING)

    approved = store.approve(queued.id, approved=True)
    assert approved.status is TaskStatus.RUNNING
    assert approved.approval_state == "approved"
