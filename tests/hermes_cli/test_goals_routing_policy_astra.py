"""Goal judging and drafting must not turn policy denial into continuation."""
import pytest
from hermes_cli.routing_policy import RoutingPolicyError


@pytest.mark.parametrize('draft', [False, True])
def test_goal_policy_denial_propagates(monkeypatch, draft):
    from hermes_cli import goals
    def denied(*a, **kw):
        raise RoutingPolicyError('denied')
    monkeypatch.setattr('agent.auxiliary_client.call_llm', denied)
    with pytest.raises(RoutingPolicyError):
        if draft:
            goals.draft_contract('objective', timeout=1)
        else:
            goals.judge_goal('objective', 'response', timeout=1)
