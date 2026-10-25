"""A lane turn cap reaches the owner-run worker without limiting other tenants/profiles."""
import pytest

from gateway.session_kanban import worker_turn_limit
from gateway.session_policy import build_policy
from hermes_state_runtime import RuntimeStoreError

LANE = 'routine:bcacfb960ceb'


def test_assigned_profile_cap_reaches_the_session_policy(tmp_path):
    config = {'kanban': {'worker_max_turns_by_tenant': {LANE: 12}}}
    limit = worker_turn_limit(config, LANE)
    assert limit == 12
    policy = build_policy({'source': 'cli', 'cwd': str(tmp_path), 'max_turns': limit}, config)
    assert policy.max_turns == 12
    from agent.iteration_budget import IterationBudget
    budget = IterationBudget(policy.max_turns)
    for _ in range(8):
        assert budget.consume()
    # Turn 8 leaves room for the lifecycle tool; the real model's cooperation
    # is verified after install, not inferred from this counter probe.
    assert budget.remaining == 4


def test_other_tenants_and_profiles_stay_uncapped():
    config = {'kanban': {'worker_max_turns_by_tenant': {LANE: 12}}}
    assert worker_turn_limit(config, 'routine:other') is None
    assert worker_turn_limit(config, None) is None
    assert worker_turn_limit({}, LANE) is None
    assert worker_turn_limit({'kanban': {}}, LANE) is None


@pytest.mark.parametrize('raw', [0, -1, True, '12', 1.5])
def test_invalid_limit_never_admits_an_unbounded_worker(raw):
    with pytest.raises(RuntimeStoreError, match='invalid_params'):
        worker_turn_limit({'kanban': {'worker_max_turns_by_tenant': {LANE: raw}}}, LANE)
