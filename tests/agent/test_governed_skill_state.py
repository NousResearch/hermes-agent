import json

from agent.governed_skill_state import (
    GovernancePhase,
    GovernedSkillState,
    classify_governed_mission,
)


def test_explicit_lah_workflow_turn_requires_router_then_decomposer():
    assert classify_governed_mission("/lah-workflow MISSION: audit repo") is True
    state = GovernedSkillState(governed=True)

    assert state.before_tool("skill_view", {"name": "lah-repo-router"}).allowed
    state.observe_skill_result("lah-repo-router", {"success": True})
    assert state.phase is GovernancePhase.DECOMPOSER_REQUIRED

    assert state.before_tool("skill_view", {"name": "mission-decomposer"}).allowed
    state.observe_skill_result("mission-decomposer", {"success": True})
    assert state.downstream_allowed is True


def test_malformed_router_cannot_satisfy_gate():
    state = GovernedSkillState(governed=True)
    decision = state.before_tool("skill_view", {"name": "lah-stack/lah-repo-router"})
    assert decision.allowed is False
    assert state.phase is GovernancePhase.ROUTER_REQUIRED
    assert "lah-repo-router" in decision.result


def test_failed_router_blocks_downstream_and_decomposer():
    state = GovernedSkillState(governed=True)
    state.observe_skill_result("lah-repo-router", {"success": False, "error": "missing"})

    for name in ("mission-decomposer", "terminal", "write_file"):
        decision = state.before_tool(
            "skill_view" if name == "mission-decomposer" else name,
            {"name": name} if name == "mission-decomposer" else {},
        )
        assert decision.allowed is False
        assert json.loads(decision.result)["governance"]["downstream_execution_allowed"] is False


def test_non_lah_turn_is_unaffected():
    assert classify_governed_mission("Explain Python decorators") is False
    state = GovernedSkillState(governed=False)
    assert state.before_tool("terminal", {"command": "true"}).allowed


def test_invalid_critical_authority_blocks_governed_turn():
    state = GovernedSkillState(
        governed=True,
        authority_valid=False,
        authority_errors=("lah-repo-router: content drift",),
    )
    assert state.downstream_allowed is False
    assert state.before_tool("terminal", {}).allowed is False
