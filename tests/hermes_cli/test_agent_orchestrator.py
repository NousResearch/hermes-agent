"""Tests for AgentOrchestrator state machine."""

import pytest
from hermes_cli.code.agent_orchestrator import (
    AgentOrchestrator,
    OrchestratorState,
    validate_transition,
    TRANSITIONS,
)


@pytest.fixture()
def orch(tmp_path):
    return AgentOrchestrator(db_path=tmp_path / "state.db")


class TestValidateTransition:
    def test_intake_to_discovery_valid(self):
        ok, reason = validate_transition(OrchestratorState.INTAKE, OrchestratorState.DISCOVERY)
        assert ok is True
        assert reason == ""

    def test_intake_to_implementation_invalid(self):
        ok, reason = validate_transition(OrchestratorState.INTAKE, OrchestratorState.IMPLEMENTATION)
        assert ok is False
        assert "intake" in reason

    def test_terminal_state_no_transitions(self):
        ok, reason = validate_transition(OrchestratorState.COMPLETED, OrchestratorState.DISCOVERY)
        assert ok is False
        assert "terminal" in reason.lower()

    def test_unknown_from_state(self):
        ok, reason = validate_transition("nonexistent", OrchestratorState.DISCOVERY)
        assert ok is False
        assert "nonexistent" in reason

    def test_unknown_to_state(self):
        ok, reason = validate_transition(OrchestratorState.INTAKE, "nonexistent")
        assert ok is False

    def test_cancelled_is_terminal(self):
        ok, _ = validate_transition(OrchestratorState.CANCELLED, OrchestratorState.INTAKE)
        assert ok is False

    def test_failed_is_terminal(self):
        ok, _ = validate_transition(OrchestratorState.FAILED, OrchestratorState.INTAKE)
        assert ok is False

    def test_all_non_terminal_have_cancel_path(self):
        non_terminal = OrchestratorState.ALL_STATES - OrchestratorState.TERMINAL_STATES
        for state in non_terminal:
            assert OrchestratorState.CANCELLED in TRANSITIONS.get(state, frozenset()), \
                f"State {state!r} has no cancel transition"

    def test_planning_to_approval_valid(self):
        ok, _ = validate_transition(OrchestratorState.PLANNING, OrchestratorState.APPROVAL)
        assert ok is True

    def test_validation_can_return_to_implementation(self):
        ok, _ = validate_transition(OrchestratorState.VALIDATION, OrchestratorState.IMPLEMENTATION)
        assert ok is True


class TestAgentOrchestrator:
    def test_create_run_starts_at_intake(self, orch):
        run = orch.create_run(title="Test task")
        assert run["id"]
        assert run["state"] == OrchestratorState.INTAKE
        assert run["title"] == "Test task"

    def test_create_run_with_description_creates_intake_artifact(self, orch, tmp_path):
        run = orch.create_run(
            title="Implement auth",
            task_description="Add OAuth2 login with Google",
        )
        # Verify intake artifact was created via ArtifactLedger
        from hermes_cli.code.artifact_ledger import ArtifactLedger
        ledger = ArtifactLedger(db_path=tmp_path / "state.db")
        arts = ledger.list_artifacts(orchestrated_run_id=run["id"])
        assert len(arts) == 1
        assert arts[0]["category"] == "task_intake"
        assert "OAuth2" in arts[0]["content"]

    def test_get_run(self, orch):
        run = orch.create_run(title="Get run test")
        fetched = orch.get_run(run["id"])
        assert fetched is not None
        assert fetched["id"] == run["id"]

    def test_get_missing_run_returns_none(self, orch):
        assert orch.get_run("nonexistent") is None

    def test_valid_transition(self, orch):
        run = orch.create_run(title="Transition test")
        updated = orch.transition(run["id"], OrchestratorState.DISCOVERY)
        assert updated["state"] == OrchestratorState.DISCOVERY

    def test_invalid_transition_raises(self, orch):
        run = orch.create_run(title="Invalid transition")
        with pytest.raises(ValueError, match="Invalid transition"):
            orch.transition(run["id"], OrchestratorState.IMPLEMENTATION)

    def test_cancel_run(self, orch):
        run = orch.create_run(title="Cancel test")
        cancelled = orch.cancel_run(run["id"], reason="User requested cancel")
        assert cancelled["state"] == OrchestratorState.CANCELLED
        assert cancelled["completed_at"] is not None

    def test_fail_run(self, orch):
        run = orch.create_run(title="Fail test")
        failed = orch.fail_run(run["id"], reason="Build failed")
        assert failed["state"] == OrchestratorState.FAILED

    def test_cancel_from_implementation(self, orch):
        run = orch.create_run()
        run = orch.transition(run["id"], OrchestratorState.DISCOVERY)
        run = orch.transition(run["id"], OrchestratorState.PLANNING)
        run = orch.transition(run["id"], OrchestratorState.IMPLEMENTATION)
        cancelled = orch.cancel_run(run["id"])
        assert cancelled["state"] == OrchestratorState.CANCELLED

    def test_list_events(self, orch):
        run = orch.create_run(title="Events test")
        orch.transition(run["id"], OrchestratorState.DISCOVERY)
        orch.transition(run["id"], OrchestratorState.PLANNING)
        events = orch.list_events(run["id"])
        # created + 2 transitions
        assert len(events) >= 3
        types = {e["type"] for e in events}
        assert "orchestrator.run.created" in types
        assert "orchestrator.run.transitioned" in types

    def test_list_runs(self, orch):
        orch.create_run(title="Run 1")
        orch.create_run(title="Run 2")
        runs = orch.list_runs()
        assert len(runs) >= 2

    def test_list_runs_filter_by_state(self, orch):
        r1 = orch.create_run(title="Active")
        orch.create_run(title="Also active")
        # Cancel r1
        orch.cancel_run(r1["id"])
        active = orch.list_runs(state=OrchestratorState.INTAKE)
        cancelled = orch.list_runs(state=OrchestratorState.CANCELLED)
        assert all(r["state"] == OrchestratorState.INTAKE for r in active)
        assert any(r["id"] == r1["id"] for r in cancelled)

    def test_transition_transition_missing_run_raises(self, orch):
        with pytest.raises(ValueError, match="not found"):
            orch.transition("no-such-run", OrchestratorState.DISCOVERY)

    def test_valid_states(self):
        states = AgentOrchestrator.valid_states()
        assert OrchestratorState.INTAKE in states
        assert OrchestratorState.COMPLETED in states
        assert len(states) == 13

    def test_valid_transitions_from_intake(self):
        transitions = AgentOrchestrator.valid_transitions(OrchestratorState.INTAKE)
        assert OrchestratorState.DISCOVERY in transitions
        assert OrchestratorState.CANCELLED in transitions

    def test_attach_artifact(self, orch, tmp_path):
        run = orch.create_run(title="Artifact attach test")
        art = orch.attach_artifact(
            run_id=run["id"],
            category="implementation_plan",
            content="## Plan\n1. Do thing\n2. Do other thing",
            title="My Plan",
        )
        assert art["category"] == "implementation_plan"
        assert art["orchestrated_run_id"] == run["id"]
