from __future__ import annotations

import pytest

from hermes_cli.computer_mode import (
    ComputerTaskStore,
    InvalidTaskTransition,
    TaskStatus,
)


def test_task_persists_and_can_be_loaded_after_store_reopen(tmp_path):
    database = tmp_path / "computer-tasks.sqlite3"
    first_store = ComputerTaskStore(database)
    created = first_store.create_task("Open the customer portal", source="whatsapp")

    second_store = ComputerTaskStore(database)
    loaded = second_store.get_task(created.id)

    assert loaded == created
    assert loaded is not None
    assert loaded.source == "whatsapp"
    assert loaded.status is TaskStatus.QUEUED


def test_approval_is_required_before_task_continues(tmp_path):
    store = ComputerTaskStore(tmp_path / "tasks.sqlite3")
    task = store.create_task("Draft a reply, but do not send it")
    store.transition(task.id, TaskStatus.RUNNING)

    waiting = store.request_approval(task.id, "Sending an email affects an external recipient")
    assert waiting.status is TaskStatus.AWAITING_APPROVAL
    assert waiting.approval_state == "pending"

    denied = store.resolve_approval(task.id, approved=False, detail="User declined")
    assert denied.status is TaskStatus.PAUSED
    assert denied.approval_state == "denied"

    events = [event.event for event in store.list_events(task.id)]
    assert events == ["task_created", "status_changed", "approval_requested", "approval_denied"]


def test_approved_task_returns_to_running(tmp_path):
    store = ComputerTaskStore(tmp_path / "tasks.sqlite3")
    task = store.create_task("Prepare a draft")
    store.transition(task.id, TaskStatus.RUNNING)
    store.request_approval(task.id, "Review before external action")

    approved = store.resolve_approval(task.id, approved=True)
    assert approved.status is TaskStatus.RUNNING
    assert approved.approval_state == "approved"


def test_invalid_transition_is_rejected(tmp_path):
    store = ComputerTaskStore(tmp_path / "tasks.sqlite3")
    task = store.create_task("Example task")

    with pytest.raises(InvalidTaskTransition):
        store.transition(task.id, TaskStatus.COMPLETED)


def test_empty_prompt_and_unknown_task_are_rejected(tmp_path):
    store = ComputerTaskStore(tmp_path / "tasks.sqlite3")

    with pytest.raises(ValueError):
        store.create_task("   ")

    with pytest.raises(KeyError):
        store.transition("missing", TaskStatus.RUNNING)



def test_denied_approval_cannot_be_bypassed_by_status_transition(tmp_path):
    store = ComputerTaskStore(tmp_path / "tasks.sqlite3")
    task = store.create_task("Send a message")
    store.transition(task.id, TaskStatus.RUNNING)
    store.request_approval(task.id, "Sending is an external side effect")
    store.resolve_approval(task.id, approved=False)

    with pytest.raises(InvalidTaskTransition):
        store.transition(task.id, TaskStatus.RUNNING)

    retry = store.request_approval(task.id, "Ask the user again after revising the draft")
    assert retry.status is TaskStatus.AWAITING_APPROVAL



def test_pending_approval_cannot_be_bypassed_after_pause(tmp_path):
    store = ComputerTaskStore(tmp_path / "tasks.sqlite3")
    task = store.create_task("Submit a form")
    store.transition(task.id, TaskStatus.RUNNING)
    store.request_approval(task.id, "Submitting affects an external service")
    store.transition(task.id, TaskStatus.PAUSED)

    with pytest.raises(InvalidTaskTransition):
        store.transition(task.id, TaskStatus.RUNNING)

    approved = store.resolve_approval(task.id, approved=True, detail="Approved while paused")
    assert approved.status is TaskStatus.RUNNING
