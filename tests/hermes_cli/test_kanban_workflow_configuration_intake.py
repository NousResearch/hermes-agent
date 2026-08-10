"""Pre-dispatch routing guarantees for mutable workflow configuration work."""

from __future__ import annotations

import json

import pytest


@pytest.fixture
def workflow_configuration_payload():
    """Representative fresh task covering every mutable config surface."""
    return {
        "title": "refresh workflow configuration",
        "body": (
            "Update cron jobs, profile configuration, skills, and runtime files; "
            "run focused tests before dispatch."
        ),
        "metadata": {
            "task_type": "workflow_configuration",
            "scope": "cron/profile/skill/runtime",
            "human_gate": "preserve",
        },
    }


def test_workflow_configuration_routes_default_to_forge_or_dev(monkeypatch):
    from hermes_cli import kanban_intake
    monkeypatch.setattr(kanban_intake, "_profile_exists", lambda name: name in {"forge", "dev"})
    assignee, metadata = kanban_intake.preflight_workflow_configuration(
        task_type="workflow_configuration", assignee="default"
    )
    assert assignee == "forge"
    assert metadata["canonical"] is True
    assert metadata["lane"] == "FORGE"
    assert metadata["coding_agent"] == "codex"
    assert metadata["use_coding_router"] is True


@pytest.mark.parametrize("invalid_agent", ["direct", "bogus", "DEV"])
def test_workflow_configuration_rejects_invalid_coding_agent(monkeypatch, invalid_agent):
    from hermes_cli import kanban_intake
    monkeypatch.setattr(kanban_intake, "_profile_exists", lambda name: name == "dev")
    with pytest.raises(ValueError, match="coding_agent="):
        kanban_intake.preflight_workflow_configuration(
            task_type="workflow_configuration", assignee="default",
            metadata={"coding_agent": invalid_agent},
        )


def test_read_only_preflight_has_no_write_or_duplicate_counts(monkeypatch):
    from hermes_cli import kanban_intake
    monkeypatch.setattr(kanban_intake, "_profile_exists", lambda name: name == "dev")
    result = kanban_intake.preflight_output(
        task_type="workflow_configuration", assignee="halo",
        metadata={"route": "dev_direct", "use_coding_router": False},
    )
    assert result["accepted"] is True
    assert result["resolved_assignee"] == "dev"
    assert result["writes"] == 0
    assert result["duplicates_created"] == 0
    assert result["metadata"]["coding_agent"] is None
    assert json.dumps(result)


def test_create_task_normalizes_before_insert_and_rejects_invalid_metadata(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    (home / "profiles" / "forge").mkdir(parents=True)
    (home / "profiles" / "dev").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli import kanban_db as kb

    kb.init_db()
    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="workflow config fixture",
            assignee="default",
            metadata={"task_type": "workflow_configuration"},
        )
        task = kb.get_task(conn, task_id)
        assert task.assignee == "forge"
        assert task.metadata["canonical"] is True
        assert task.metadata["coding_agent"] == "codex"
        with pytest.raises(ValueError, match="coding_agent=.*direct"):
            kb.create_task(
                conn,
                title="invalid workflow config",
                assignee="default",
                metadata={"task_type": "workflow_configuration", "coding_agent": "direct"},
            )
        assert len(kb.list_tasks(conn)) == 1


def test_fresh_mutable_workflow_fixture_gets_canonical_metadata(monkeypatch, workflow_configuration_payload):
    from hermes_cli import kanban_intake

    monkeypatch.setattr(kanban_intake, "_profile_exists", lambda name: name == "dev")
    assignee, metadata = kanban_intake.preflight_workflow_configuration(
        task_type=workflow_configuration_payload["metadata"]["task_type"],
        assignee="default",
        metadata=workflow_configuration_payload["metadata"],
        requested_agent="cursor",
    )

    assert assignee == "dev"
    assert metadata["canonical"] is True
    assert metadata["implementation_lane"] is True
    assert metadata["lane"] == "DEV"
    assert metadata["coding_agent"] == "cursor"
    assert metadata["coding_agent_resolution"] == "explicit"
    assert metadata["human_gate"] == "preserve"


@pytest.mark.parametrize("assignee", ["default", "halo"])
def test_control_only_assignment_is_rejected_without_implementation_profile(monkeypatch, assignee):
    from hermes_cli import kanban_intake

    monkeypatch.setattr(kanban_intake, "_profile_exists", lambda _name: False)
    with pytest.raises(ValueError, match="requires an implementation profile"):
        kanban_intake.preflight_workflow_configuration(
            task_type="workflow_configuration", assignee=assignee,
        )


def test_direct_mode_is_explicit_and_router_is_disabled(monkeypatch):
    from hermes_cli import kanban_intake

    monkeypatch.setattr(kanban_intake, "_profile_exists", lambda name: name == "dev")
    assignee, metadata = kanban_intake.preflight_workflow_configuration(
        task_type="workflow_configuration",
        assignee="dev",
        metadata={"route": "dev_direct", "use_coding_router": False},
    )
    assert assignee == "dev"
    assert metadata["route"] == "dev_direct"
    assert metadata["use_coding_router"] is False
    assert metadata["coding_agent"] is None
    assert metadata["execution_fallback"] == "dev_direct"


def test_missing_implementation_lane_is_rejected_when_no_specialist_exists(monkeypatch):
    from hermes_cli import kanban_intake

    monkeypatch.setattr(kanban_intake, "_profile_exists", lambda _name: False)
    result = kanban_intake.preflight_output(
        task_type="workflow_configuration", assignee="halo", metadata={}
    )
    assert result["accepted"] is False
    assert "implementation profile" in result["error"]
    assert result["writes"] == 0
    assert result["duplicates_created"] == 0


def test_inconsistent_parent_child_links_are_rejected_without_duplicate_writes(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli import kanban_db as kb

    kb.init_db()
    with kb.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="dev")
        child = kb.create_task(conn, title="child", assignee="dev")
        kb.link_tasks(conn, parent, child)
        with pytest.raises(ValueError, match="would create a cycle"):
            kb.link_tasks(conn, child, parent)
        with pytest.raises(ValueError, match="cannot depend on itself"):
            kb.link_tasks(conn, parent, parent)
        with pytest.raises(ValueError, match="unknown task"):
            kb.link_tasks(conn, "missing-parent", child)
        assert len(kb.list_tasks(conn)) == 2