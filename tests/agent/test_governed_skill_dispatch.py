import json

from agent.governed_skill_state import GovernedSkillState
from agent.tool_executor import _governance_observe, _governance_preflight


class _Agent:
    def __init__(self):
        self._governed_skill_state = GovernedSkillState(governed=True)


def test_dispatch_boundary_accepts_only_canonical_router_name():
    agent = _Agent()
    assert _governance_preflight(
        agent, "skill_view", {"name": "lah-stack/lah-repo-router"}
    )
    assert _governance_preflight(agent, "skill_view", {"name": "lah-repo-router"}) is None


def test_dispatch_boundary_denies_terminal_until_both_gates_pass():
    agent = _Agent()
    blocked = _governance_preflight(agent, "terminal", {"command": "true"})
    assert json.loads(blocked)["governance"]["downstream_execution_allowed"] is False

    _governance_observe(
        agent,
        "skill_view",
        {"name": "lah-repo-router"},
        json.dumps({"success": True, "skill_name": "lah-repo-router"}),
    )
    assert _governance_preflight(agent, "terminal", {"command": "true"})

    _governance_observe(
        agent,
        "skill_view",
        {"name": "mission-decomposer"},
        json.dumps({"success": True, "skill_name": "mission-decomposer"}),
    )
    assert _governance_preflight(agent, "terminal", {"command": "true"}) is None
