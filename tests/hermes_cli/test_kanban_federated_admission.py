from types import SimpleNamespace

import pytest

from hermes_cli.fleet_protocol import FleetTask, TaskRequirement
from hermes_cli.kanban_ops import (
    federated_create_options,
    federated_enabled,
    fleet_task_from_kanban,
)


def _task(**overrides):
    values = {
        "id": "local-123",
        "title": "Build LunaBot",
        "body": "Run the tests",
        "project_id": "lunabot",
        "workspace_kind": "worktree",
        "model_override": "gpt-5",
        "provider_override": None,
        "assignee": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_federated_admission_is_explicitly_opt_in():
    assert federated_enabled({"kanban": {}}) is False
    assert federated_enabled({"kanban": {"federated": {"enabled": True}}}) is True


def test_kanban_task_becomes_a_wire_task_without_local_credentials():
    task = fleet_task_from_kanban(_task())

    assert isinstance(task, FleetTask)
    assert task.task_id == "local-123"
    assert task.requirement == TaskRequirement(
        models=("gpt-5",), project="lunabot", workspace_kind="worktree"
    )


def test_federated_create_clears_local_assignee_and_uses_ready_state():
    options = federated_create_options(SimpleNamespace(assignee=None, initial_status="running"))

    assert options == {"assignee": None, "initial_status": "ready", "created_by": "fleet"}


def test_federated_create_rejects_a_profile_assignee():
    with pytest.raises(ValueError, match="must not specify --assignee"):
        federated_create_options(SimpleNamespace(assignee="coding-expert", initial_status="ready"))
