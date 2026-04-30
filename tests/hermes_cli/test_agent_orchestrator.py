"""Tests for AgentOrchestrator."""

import pytest

from hermes_cli.code.agent_orchestrator import (
    AgentOrchestrator,
    OrchestratorState,
    validate_transition,
)
from hermes_cli.code.artifact_ledger import ArtifactLedger


@pytest.fixture()
def orchestrator(tmp_path):
    return AgentOrchestrator(db_path=tmp_path / "state.db")


def test_create_run_defaults_to_intake(orchestrator):
    run = orchestrator.create_run(title="Task A")
    assert run["state"] == OrchestratorState.INTAKE


def test_valid_transition_function():
    ok, message = validate_transition(OrchestratorState.INTAKE, OrchestratorState.DISCOVERY)
    assert ok is True
    assert message == ""


def test_invalid_transition_function():
    ok, message = validate_transition(OrchestratorState.INTAKE, OrchestratorState.IMPLEMENTATION)
    assert ok is False
    assert "Invalid transition" in message


def test_transition_run_and_persist(orchestrator):
    run = orchestrator.create_run(title="Task B")
    updated = orchestrator.transition_run(run["id"], OrchestratorState.DISCOVERY, reason="start discovery")
    assert updated["state"] == OrchestratorState.DISCOVERY
    transitions = orchestrator.list_transitions(run["id"])
    assert len(transitions) == 1
    assert transitions[0]["to_state"] == OrchestratorState.DISCOVERY


def test_terminal_transition_sets_completed_at(orchestrator):
    run = orchestrator.create_run(title="Task C")
    run = orchestrator.transition_run(run["id"], OrchestratorState.DISCOVERY)
    run = orchestrator.transition_run(run["id"], OrchestratorState.PLANNING)
    run = orchestrator.transition_run(run["id"], OrchestratorState.APPROVAL)
    run = orchestrator.transition_run(run["id"], OrchestratorState.IMPLEMENTATION)
    run = orchestrator.transition_run(run["id"], OrchestratorState.VALIDATION)
    run = orchestrator.transition_run(run["id"], OrchestratorState.REVIEW)
    run = orchestrator.transition_run(run["id"], OrchestratorState.READY_FOR_PR)
    run = orchestrator.transition_run(run["id"], OrchestratorState.COMPLETED)
    assert run["completed_at"] is not None


def test_create_goal_creates_task_intake_artifact(orchestrator, tmp_path):
    run = orchestrator.create_run(
        title="Task D",
        goal="Add smoke tests",
        create_intake_artifact=True,
    )
    ledger = ArtifactLedger(db_path=tmp_path / "state.db")
    artifacts = ledger.list_artifacts(orchestrated_run_id=run["id"])
    assert len(artifacts) == 1
    assert artifacts[0]["artifact_type"] == "task_intake"
