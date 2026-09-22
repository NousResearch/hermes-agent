"""Health evidence narrows eligibility without relaxing task qualifications."""
import pytest


def inputs():
    req = dict(schema_version=1, role="builder", execution_kind="delegate",
               execution_id="task", attempt_id="1", task_class="cross-component",
               required_capabilities=[], input_tokens=1000, reserve_tokens=2000,
               reasoning="high", target_profile="worker-a",
               provenance=dict(frozen_sha="", verified_by="host", complete=True, contributors=[]))
    routes = [dict(route_id=name, route_revision=1, provider="custom", model=name,
                   endpoint="http://127.0.0.1:1/v1", maker=name, model_family=name,
                   status="approved", allowed_roles=["builder"], capabilities=[],
                   verified_input_budget=100000, allowed_reasoning=["high"],
                   qualifications=["deep"], assessment="fixture", evidence=[])
              for name in ("preferred", "alternate")]
    policy = dict(schema_version=1, policy_id="fixture", revision=1, approval_ref="fixture",
                  routes=routes, rankings={"builder": {"deep": [r["route_id"] for r in routes]}})
    return req, policy


@pytest.mark.parametrize("status", ["missing_auth", "denied_model", "quota", "outage"])
def test_unavailable_routes_are_not_selected_or_alternates(status):
    from agent.model_selection import select, RoutingBlocked
    req, policy = inputs()
    health = {"preferred": dict(target_profile="worker-a", route_revision=1,
              endpoint=policy["routes"][0]["endpoint"], status=status,
              observed_at=100, retry_after=500)}
    decision = select(req, policy, health, now=110)
    assert decision["selected"]["route_id"] == "alternate"
    assert decision["rejections"]["preferred"] == ["provider_unavailable"]
    assert select(req, policy, health, now=501)["selected"]["route_id"] == "preferred"
    health["alternate"] = dict(health["preferred"])
    with pytest.raises(RoutingBlocked, match="provider_unavailable"):
        select(req, policy, health, now=110)


@pytest.mark.parametrize("change", [{"target_profile": "worker-b"}, {"route_revision": 2},
                                     {"endpoint": "https://other.invalid/v1"},
                                     {"observed_at": 200}])
def test_other_scopes_and_clock_ambiguous_health_remain_unknown(change):
    from agent.model_selection import select
    req, policy = inputs()
    observation = dict(target_profile="worker-a", route_revision=1,
                       endpoint=policy["routes"][0]["endpoint"], status="outage",
                       observed_at=100, retry_after=500)
    observation.update(change)
    decision = select(req, policy, {"preferred": observation}, now=110)
    assert decision["selected"]["route_id"] == "preferred"
    assert decision["availability"]["preferred"]["status"] == "unknown"
