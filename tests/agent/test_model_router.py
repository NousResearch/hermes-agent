"""Unit tests for the Phase03 deterministic delegation model router.

Everything is faked (inventory / catalog / pricing / credential health) so
these tests never touch live provider APIs. They assert *behaviour contracts*
(route inference, paid-before-free ordering, exhausted-provider skipping,
explicit-override precedence, provider-diverse fallback), not frozen model
lists — there are no hardcoded model names in the production path under test.
"""

import pytest

from agent.model_router import (
    build_candidates,
    candidate_is_free,
    infer_route_key,
    router_enabled,
    score_candidate,
    select_delegation_model,
    task_has_explicit_model,
)


# --- Fake inventory --------------------------------------------------------

FAKE_CATALOG = {
    "paidcoder": [
        ("acme-coder-pro", "strong coding and debugging model"),
        ("acme-chat", "general chat assistant"),
    ],
    "paidresearch": [
        ("beta-reason-max", "deep research and reasoning model"),
        ("beta-write", "long-form writing and summaries"),
    ],
    "localfree": [
        ("local-generalist", "runs locally, general instruct model"),
    ],
}

FAKE_PRICING = {
    "paidcoder": {
        "acme-coder-pro": {"prompt": 3.0, "completion": 9.0},
        "acme-chat": {"prompt": 1.0, "completion": 2.0},
    },
    "paidresearch": {
        "beta-reason-max": {"prompt": 2.0, "completion": 6.0},
        "beta-write": {"prompt": 1.0, "completion": 3.0},
    },
    "localfree": {
        "local-generalist": {"prompt": 0.0, "completion": 0.0},
    },
}


def _catalog_fn(provider):
    return FAKE_CATALOG.get(provider, [])


def _pricing_fn(provider):
    return FAKE_PRICING.get(provider, {})


def _inventory_fn():
    return ["paidcoder", "paidresearch", "localfree"]


def _router_cfg(**over):
    cfg = {"enabled": True, "free_providers": ["localfree"], "max_candidates": 12}
    cfg.update(over)
    return cfg


# --- Route inference -------------------------------------------------------

@pytest.mark.parametrize(
    "goal,expected",
    [
        ("Implement the function and fix the failing pytest", "coding"),
        ("Research and compare the latest agent frameworks", "research"),
        ("Draft a summary report of the findings", "writing"),
        ("Design the overall system architecture and pipeline", "architecture"),
        ("Say hello", "default"),
    ],
)
def test_infer_route_key(goal, expected):
    assert infer_route_key(goal) == expected


# --- Enable / override gating ---------------------------------------------

def test_router_disabled_by_default():
    assert router_enabled(None) is False
    assert router_enabled({}) is False
    assert router_enabled({"enabled": False}) is False


@pytest.mark.parametrize("val", [True, "true", "yes", "on", "1"])
def test_router_enabled_truthy(val):
    assert router_enabled({"enabled": val}) is True


def test_disabled_router_returns_none():
    out = select_delegation_model(
        {"goal": "write code"},
        {"enabled": False},
        provider_inventory_fn=_inventory_fn,
        catalog_fn=_catalog_fn,
        pricing_fn=_pricing_fn,
    )
    assert out is None


def test_explicit_model_override_wins():
    assert task_has_explicit_model({"model": "x/y"}) is True
    assert task_has_explicit_model({"provider": "x"}) is True
    out = select_delegation_model(
        {"goal": "write code", "model": {"provider": "x", "model": "y"}},
        _router_cfg(),
        provider_inventory_fn=_inventory_fn,
        catalog_fn=_catalog_fn,
        pricing_fn=_pricing_fn,
    )
    assert out is None


# --- Selection & scoring ---------------------------------------------------

def test_coding_task_prefers_coding_capable_model():
    out = select_delegation_model(
        {"goal": "Implement and debug the parser function"},
        _router_cfg(),
        provider_inventory_fn=_inventory_fn,
        catalog_fn=_catalog_fn,
        pricing_fn=_pricing_fn,
    )
    assert out is not None
    assert out["route"] == "coding"
    assert out["selected"] == {"provider": "paidcoder", "model": "acme-coder-pro"}


def test_research_task_prefers_reasoning_model():
    out = select_delegation_model(
        {"goal": "Research and analyze prior art, compare approaches"},
        _router_cfg(),
        provider_inventory_fn=_inventory_fn,
        catalog_fn=_catalog_fn,
        pricing_fn=_pricing_fn,
    )
    assert out is not None
    assert out["route"] == "research"
    assert out["selected"]["provider"] == "paidresearch"
    assert out["selected"]["model"] == "beta-reason-max"


def test_free_local_is_last_tier():
    cands = build_candidates(
        "default",
        _inventory_fn(),
        catalog_fn=_catalog_fn,
        pricing_fn=_pricing_fn,
        free_providers=["localfree"],
    )
    # Free/local candidate must not be first while any paid candidate exists.
    assert not (cands[0]["is_free"] or cands[0]["is_local"])
    assert any(c["is_free"] or c["is_local"] for c in cands)
    last = cands[-1]
    assert last["provider"] == "localfree"


def test_zero_priced_model_detected_free():
    assert candidate_is_free(
        "paidresearch", "beta-write",
        pricing={"beta-write": {"prompt": 0.0, "completion": 0.0}},
    ) is True


def test_local_provider_is_free_tier():
    assert candidate_is_free("ollama", "anything") is True
    assert candidate_is_free("lmstudio", "anything") is True


# --- Credential exhaustion -------------------------------------------------

def test_exhausted_provider_is_skipped():
    def credential_ok(provider):
        return provider != "paidcoder"  # paidcoder is exhausted

    out = select_delegation_model(
        {"goal": "Implement and debug the parser function"},  # coding task
        _router_cfg(),
        provider_inventory_fn=_inventory_fn,
        catalog_fn=_catalog_fn,
        pricing_fn=_pricing_fn,
        credential_ok_fn=credential_ok,
    )
    assert out is not None
    # paidcoder skipped -> selection must come from another provider
    assert out["selected"]["provider"] != "paidcoder"
    assert all(c["provider"] != "paidcoder" for c in out["candidates"])


def test_all_providers_exhausted_returns_none():
    out = select_delegation_model(
        {"goal": "write code"},
        _router_cfg(),
        provider_inventory_fn=_inventory_fn,
        catalog_fn=_catalog_fn,
        pricing_fn=_pricing_fn,
        credential_ok_fn=lambda p: False,
    )
    assert out is None


# --- Fallback chain --------------------------------------------------------

def test_fallback_chain_excludes_selected_and_is_provider_diverse():
    out = select_delegation_model(
        {"goal": "Implement and debug the parser function"},
        _router_cfg(),
        provider_inventory_fn=_inventory_fn,
        catalog_fn=_catalog_fn,
        pricing_fn=_pricing_fn,
    )
    assert out is not None
    selected = out["selected"]
    chain = out["fallback_chain"]
    # selected pair not repeated in the fallback chain
    assert {"provider": selected["provider"], "model": selected["model"]} not in chain
    # chain contains at least one different provider (provider diversity)
    providers_in_chain = {c["provider"] for c in chain}
    assert providers_in_chain - {selected["provider"]}
    # a free/local option is reachable somewhere in the full ordering
    assert any(c["is_free"] or c["is_local"] for c in out["candidates"])


def test_empty_inventory_returns_none():
    out = select_delegation_model(
        {"goal": "write code"},
        _router_cfg(),
        provider_inventory_fn=lambda: [],
        catalog_fn=_catalog_fn,
        pricing_fn=_pricing_fn,
    )
    assert out is None


def test_provider_priority_reorders_inventory():
    out = select_delegation_model(
        {"goal": "Say hello"},  # default route, weak signal
        _router_cfg(provider_priority=["paidresearch"]),
        provider_inventory_fn=_inventory_fn,
        catalog_fn=_catalog_fn,
        pricing_fn=_pricing_fn,
    )
    assert out is not None
    # With a weak route signal, priority should surface paidresearch first.
    assert out["candidates"][0]["provider"] == "paidresearch"


def test_score_is_deterministic():
    cand = {"provider": "paidcoder", "model": "acme-coder-pro", "description": "coding", "model_index": 0}
    assert score_candidate(cand, "coding") == score_candidate(cand, "coding")
    # coding route scores a coding model higher than the writing route does
    assert score_candidate(cand, "coding") > score_candidate(cand, "writing")
