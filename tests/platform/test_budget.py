"""Budget controls, and the line between what stops an agent and what merely watches it.

The tests that matter most here are the ones asserting NOTHING misrepresents a
measurement as a ceiling. Getting that wrong is not a bug a customer reports — it is a bug
they discover on an invoice.
"""

from __future__ import annotations

import sqlite3

import pytest

from nova.control import ControlAPI
from nova.errors import SpecError
from nova.policy import compile_policy, decide
from nova.policy.limits import (
    ENFORCING_CLASSES,
    HARD_BOUNDARY,
    HARD_PREEMPTIVE,
    OBSERVED_ONLY,
    SOFT_ADVISORY,
)
# Enforcement is a claim about a runtime, so the register lives with the adapter.
from nova.runtime.hermes.limits import LIMIT_FACTS, enforcement_of, limit_fact
from nova.runtime.hermes.materialize import build_config
from nova.runtime.hermes.paths import HermesPaths
from nova.runtime.hermes.usage import state_db_path
from nova.spec import AgentSpec


def _agent(**limits):
    return AgentSpec.parse({"id": "a", "name": "A", "limits": limits})


# -- the register is honest --------------------------------------------------


def test_no_token_or_cost_control_claims_to_enforce():
    """The audit found no way to veto a model call. Nothing may claim otherwise."""
    for key in ("tokens", "estimated_cost_usd"):
        fact = limit_fact(key)
        assert fact is not None
        assert fact.enforcement == OBSERVED_ONLY
        assert not fact.enforced


def test_run_budget_is_advisory_not_a_limit():
    fact = limit_fact("soft_wrapup_after_seconds")
    assert fact.enforcement == SOFT_ADVISORY
    assert not fact.enforced
    assert "NOT a limit" in fact.summary


def test_every_enforcing_control_cites_a_verified_call_site():
    """A control may not claim to enforce without a call site someone actually read."""
    for fact in LIMIT_FACTS:
        if fact.enforced:
            assert fact.verified_at, f"{fact.key} claims enforcement with no verified call site"


def test_unknown_keys_default_to_the_weakest_class():
    """A new control must be registered before it can claim to stop anything."""
    assert enforcement_of("something_invented") == OBSERVED_ONLY


def test_enforcing_classes_are_exactly_the_two_hard_kinds():
    assert ENFORCING_CLASSES == {HARD_PREEMPTIVE, HARD_BOUNDARY}


def test_the_runtime_declares_its_own_enforcement(runtime):
    """A future runtime that cannot veto a tool call must be able to say so."""
    declared = {fact.key for fact in runtime.limit_facts()}
    assert "max_tool_calls_per_run" in declared
    assert runtime.limit_facts() == LIMIT_FACTS


def test_a_runtime_that_declares_nothing_enforces_nothing(bundle):
    """The safe reading of silence."""
    from nova.runtime.base import AgentRuntime

    assert AgentRuntime.limit_facts(object()) == ()


# -- declaration -------------------------------------------------------------


def test_daily_token_budget_is_refused_with_an_explanation():
    """It promised a ceiling the runtime cannot enforce."""
    with pytest.raises(SpecError, match="cannot be enforced"):
        AgentSpec.parse({"id": "a", "limits": {"daily_token_budget": 1000}})


def test_child_timeout_below_the_runtime_floor_is_refused():
    """The runtime silently raises anything under 30s; better to refuse than mislead."""
    with pytest.raises(SpecError, match="at least 30"):
        _agent(delegation={"child_timeout_seconds": 10})


def test_child_timeout_of_zero_is_allowed_as_no_timeout():
    assert _agent(delegation={"child_timeout_seconds": 0}).limits.delegation.child_timeout_seconds == 0


def test_unknown_delegation_field_is_rejected():
    with pytest.raises(SpecError, match="unknown field"):
        _agent(delegation={"max_childrenn": 3})


# -- compilation to VERIFIED runtime keys ------------------------------------


def test_hard_limits_compile_to_the_exact_keys_the_runtime_reads():
    """Regression against writing a key the runtime ignores (see enabled_toolsets)."""
    config = build_config(
        _agent(
            max_turns=60,
            max_concurrent_tasks=3,
            delegation={
                "max_concurrent_children": 4,
                "max_depth": 2,
                "max_child_turns": 100,
                "child_timeout_seconds": 900,
                "orchestrator_enabled": False,
            },
        )
    )
    assert config["agent"]["max_turns"] == 60
    assert config["kanban"]["max_in_progress_per_profile"] == 3
    # The runtime's own names, not NOVA's.
    assert config["delegation"] == {
        "max_concurrent_children": 4,
        "max_spawn_depth": 2,
        "max_iterations": 100,
        "child_timeout_seconds": 900,
        "orchestrator_enabled": False,
    }


def test_soft_wrapup_compiles_to_run_budget_seconds():
    config = build_config(_agent(soft_wrapup_after_seconds=1800))
    assert config["agent"]["run_budget_seconds"] == 1800


def test_recorded_only_limits_stay_out_of_enforced_config():
    """They must not sit where a reader would take them for runtime settings."""
    config = build_config(_agent(max_task_runtime_seconds=900, max_retries=2))
    assert config.get("kanban", {}).get("max_runtime_seconds") is None
    assert config["nova"]["max_task_runtime_seconds"] == 900
    assert config["nova"]["max_retries"] == 2


def test_no_limit_compiles_to_a_key_outside_the_register():
    """Every emitted runtime key must trace to a registered, verified control."""
    config = build_config(
        _agent(max_turns=1, max_concurrent_tasks=1, soft_wrapup_after_seconds=60,
               delegation={"max_depth": 1})
    )
    emitted = set()
    for section in ("agent", "delegation", "kanban"):
        for key in config.get(section, {}):
            emitted.add(f"{section}.{key}")
    emitted.discard("agent.reasoning_effort")  # model config, not a limit
    registered = {fact.compiles_to for fact in LIMIT_FACTS if fact.compiles_to}
    assert emitted <= registered, f"unregistered runtime keys emitted: {emitted - registered}"


# -- the tool-call ceiling ---------------------------------------------------


def _policy_doc(**overrides):
    document = {
        "schema_version": 1,
        "agent_id": "a",
        "deny": [],
        "baseline": ["kanban_complete", "kanban_heartbeat"],
        "approval_actions": {},
        "allow": [],
        "unlisted_tool": "allow",
        "max_tool_calls_per_run": 0,
    }
    document.update(overrides)
    return document


def test_ceiling_blocks_once_reached():
    document = _policy_doc(max_tool_calls_per_run=3)
    assert decide(document, "crm_lookup", calls_used=2).effect == "allow"
    decision = decide(document, "crm_lookup", calls_used=3)
    assert decision.effect == "deny"
    assert decision.rule == "tool-call-ceiling"


def test_baseline_survives_the_ceiling():
    """An agent out of budget must still be able to close its own task."""
    document = _policy_doc(max_tool_calls_per_run=1)
    assert decide(document, "kanban_complete", calls_used=99).effect == "allow"
    assert decide(document, "kanban_heartbeat", calls_used=99).effect == "allow"


def test_explicit_deny_still_beats_the_ceiling():
    document = _policy_doc(max_tool_calls_per_run=100, deny=["terminal"])
    assert decide(document, "terminal", calls_used=0).rule == "explicit-deny"


def test_ceiling_beats_approval():
    """An exhausted agent must not queue work for a human it can no longer perform."""
    document = _policy_doc(max_tool_calls_per_run=1, approval_actions={"refund": ["crm_refund"]})
    assert decide(document, "crm_refund", calls_used=5).rule == "tool-call-ceiling"


def test_zero_or_absent_ceiling_means_no_ceiling():
    assert decide(_policy_doc(max_tool_calls_per_run=0), "x", calls_used=10_000).effect == "allow"
    document = _policy_doc()
    del document["max_tool_calls_per_run"]
    assert decide(document, "x", calls_used=10_000).effect == "allow"


def test_ceiling_reaches_the_compiled_document(bundle):
    compiled = compile_policy(bundle.agent("customer-support"), bundle.policy)
    assert compiled.document["max_tool_calls_per_run"] == 200


# -- the plugin's counter ----------------------------------------------------


def test_plugin_counts_only_budget_consuming_calls(bundle, runtime, audit, home):
    from nova.apply import apply_bundle

    from .test_policy_enforcement import load_installed_plugin

    apply_bundle(bundle, runtime, audit=audit)
    path = HermesPaths(home=home).policy_path("customer-support")
    import json

    document = json.loads(path.read_text(encoding="utf-8"))
    document["max_tool_calls_per_run"] = 2
    path.write_text(json.dumps(document), encoding="utf-8")

    plugin = load_installed_plugin(home, "customer-support", "nova_budget_probe")
    assert plugin.pre_tool_call(tool_name="crm_lookup", args={}) is None       # 1
    assert plugin.pre_tool_call(tool_name="kanban_heartbeat", args={}) is None  # baseline, free
    assert plugin.pre_tool_call(tool_name="crm_lookup", args={}) is None       # 2
    blocked = plugin.pre_tool_call(tool_name="crm_lookup", args={})            # over
    assert blocked["action"] == "block"
    assert "ceiling" in blocked["message"]
    # The agent can still report its outcome.
    assert plugin.pre_tool_call(tool_name="kanban_complete", args={}) is None


def test_refused_calls_do_not_consume_budget(bundle, runtime, audit, home):
    """A denied call costs nothing, so it must not eat the allowance."""
    from nova.apply import apply_bundle

    from .test_policy_enforcement import load_installed_plugin

    apply_bundle(bundle, runtime, audit=audit)
    plugin = load_installed_plugin(home, "customer-support", "nova_budget_probe2")
    for _ in range(5):
        plugin.pre_tool_call(tool_name="terminal", args={})
    assert plugin._CALLS_USED == 0


# -- reported usage ----------------------------------------------------------


def seed_usage(profile_dir, rows):
    path = state_db_path(profile_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE session_model_usage (session_id TEXT, model TEXT, billing_provider TEXT, "
        "billing_base_url TEXT, billing_mode TEXT, task TEXT, api_call_count INTEGER, "
        "input_tokens INTEGER, output_tokens INTEGER, cache_read_tokens INTEGER, "
        "cache_write_tokens INTEGER, reasoning_tokens INTEGER, estimated_cost_usd REAL, "
        "actual_cost_usd REAL, cost_status TEXT, cost_source TEXT, first_seen REAL, last_seen REAL)"
    )
    for session, model, provider, calls, inp, out, cost in rows:
        connection.execute(
            "INSERT INTO session_model_usage VALUES (?,?,?,'','','',?,?,?,0,0,0,?,0,'est','calc',0,0)",
            (session, model, provider, calls, inp, out, cost),
        )
    connection.commit()
    connection.close()


def test_usage_is_absent_before_an_agent_runs(runtime):
    summary = runtime.usage("customer-support")
    assert summary.available is False
    assert "has not run yet" in summary.detail


def test_usage_aggregates_across_sessions(runtime, home):
    seed_usage(
        HermesPaths(home=home).profile_dir("customer-support"),
        [
            ("s1", "m", "bedrock", 3, 100, 50, 0.01),
            ("s2", "m", "bedrock", 2, 200, 60, 0.02),
        ],
    )
    summary = runtime.usage("customer-support")
    assert summary.available is True
    assert summary.api_calls == 5
    assert summary.total_tokens == 410
    assert round(summary.estimated_cost_usd, 4) == 0.03


def test_usage_payload_carries_its_own_caveats(runtime, home):
    """A consumer cannot render these as a budget without contradicting its own data."""
    seed_usage(HermesPaths(home=home).profile_dir("customer-support"), [("s", "m", "p", 1, 1, 1, 0.1)])
    payload = runtime.usage("customer-support").to_dict()
    assert payload["enforcement"] == OBSERVED_ONLY
    assert any("not a spending limit" in c.lower() for c in payload["caveats"])
    assert any("lagging" in c.lower() for c in payload["caveats"])


def test_unreadable_usage_store_degrades(runtime, home):
    profile = HermesPaths(home=home).profile_dir("customer-support")
    profile.mkdir(parents=True, exist_ok=True)
    state_db_path(profile).write_text("not a database", encoding="utf-8")
    summary = runtime.usage("customer-support")
    assert summary.available is False


# -- the budget route --------------------------------------------------------


@pytest.fixture
def api(bundle, runtime):
    return ControlAPI(bundle, runtime)


def test_budget_route_separates_controls_from_observation(api):
    body = api.handle("/platform/v1/budget").body
    assert {"controls", "advisory", "recorded", "observed"} <= set(body)
    assert all(row["enforcement"] in ENFORCING_CLASSES for row in body["controls"])
    assert all(row["enforcement"] == SOFT_ADVISORY for row in body["advisory"])
    assert all(entry["enforcement"] == OBSERVED_ONLY for entry in body["observed"])


def test_budget_route_never_lists_usage_as_a_control(api):
    body = api.handle("/platform/v1/budget").body
    control_keys = {row["key"] for row in body["controls"]}
    assert "tokens" not in control_keys
    assert "estimated_cost_usd" not in control_keys
    assert "soft_wrapup_after_seconds" not in control_keys


def test_budget_route_states_the_observation_caveat(api):
    body = api.handle("/platform/v1/budget").body
    assert "not a limit" in body["observed_caveat"].lower()


def test_budget_route_names_the_runtime_key_for_each_control(api):
    """A security reviewer must be able to check the claim themselves."""
    for row in api.handle("/platform/v1/budget").body["controls"]:
        assert row["compiles_to"], f"{row['key']} claims enforcement without naming its key"
