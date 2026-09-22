"""Cohort diversity must never erase implementation provenance."""
from copy import deepcopy

import pytest


def test_cohort_preserves_each_slots_contributor_authority(tmp_path):
    from agent.model_selection_store import activate_policy, get_receipt, publish_policy
    from agent.moa_model_routing import resolve_moa_cohort

    routes = [dict(route_id=maker, route_revision=1, provider="custom", model=maker,
                   endpoint="http://127.0.0.1:1/v1", maker=maker, model_family=maker,
                   status="approved", allowed_roles=["reviewquality"], capabilities=[],
                   verified_input_budget=100000, allowed_reasoning=["medium"],
                   qualifications=["deep"], assessment="fixture", evidence=[])
              for maker in ("implementer", "reviewer-a", "reviewer-b")]
    publish_policy(tmp_path, dict(schema_version=1, policy_id="kanban-default", revision=1,
                   routes=routes, rankings={"reviewquality": {"deep": [r["route_id"] for r in routes]}}),
                   approval_ref="fixture")
    activate_policy(tmp_path, "kanban-default", 1)
    provenance = dict(frozen_sha="a" * 40, verified_by="parent", complete=True,
                      contributors=[dict(maker="implementer", evidence="run:builder")])
    slot = dict(routing_role="reviewquality", routing_requirements=dict(
        input_tokens=1000, reserve_tokens=2000, provenance=provenance))
    original = deepcopy(slot)
    cohort = resolve_moa_cohort([slot, slot], slot, execution_id="review", hermes_home=tmp_path)
    for resolution in cohort.values():
        receipt = get_receipt(tmp_path, resolution["receipt_id"])
        assert receipt["selected"]["maker"] != "implementer"
        assert receipt["requirements"]["provenance"] == provenance
    assert slot == original

    missing = deepcopy(slot)
    del missing["routing_requirements"]["provenance"]
    from agent.model_selection_types import RoutingBlocked
    with pytest.raises(RoutingBlocked, match="provenance_incomplete"):
        resolve_moa_cohort([slot, missing], slot, execution_id="missing", hermes_home=tmp_path)
