"""Quality and independence contracts for guided routing (not model snapshots)."""
from importlib.util import find_spec

import pytest


def route(route_id, maker, classes=("deep",), **overrides):
    return dict(route_id=route_id, route_revision=1, provider="openai", model=route_id,
                endpoint="https://api.openai.com/v1", maker=maker, model_family="fixture",
                status="approved", allowed_roles=["builder", "reviewquality"],
                capabilities=["tool_use", "text"], verified_input_budget=100000,
                allowed_reasoning=["high"], qualifications=list(classes),
                assessment="protocol fixture; provisional operator assessment",
                evidence=["fixture:tool-smoke"], **overrides)


def policy():
    return dict(schema_version=1, policy_id="test", revision=1,
                approval_ref="operator:conversation-1", routes=[route("a", "openai"), route("b", "anthropic")],
                rankings={"builder": {"deep": ["a", "b"]}, "reviewquality": {"deep": ["a", "b"]}})


def test_selection_is_reproducible_and_excludes_all_contributing_makers():
    assert find_spec("agent.model_selection") is not None, "guided selector is not implemented"
    from agent.model_selection import select
    req = dict(schema_version=1, role="reviewquality", execution_kind="delegate",
               execution_id="child", attempt_id="1", task_class="cross-component",
               required_capabilities=["tool_use"], input_tokens=1000, reserve_tokens=2000,
               reasoning="high", provenance={"frozen_sha": "a" * 40, "verified_by": "parent",
               "complete": True, "contributors": [{"maker": "openai", "evidence": "run:1"}]})
    first = select(req, policy(), {}, now=100)
    assert first == select(req, policy(), {}, now=100)
    assert first["selected"]["route_id"] == "b"
    assert first["rejections"]["a"] == ["contributing_maker"]
    assert first["requirements"]["quality"] == "deep"
    req["provenance"]["complete"] = False
    from agent.model_selection import RoutingBlocked
    with pytest.raises(RoutingBlocked, match="provenance_incomplete"):
        select(req, policy(), {}, now=100)
