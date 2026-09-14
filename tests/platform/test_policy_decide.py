"""The policy decision function.

This is a security control, so the tests are written around what must never happen:
failing open, an allow-list that strips a worker's ability to report, or a control that
looks present in a review and does nothing.
"""

from __future__ import annotations

import pytest

from nova.policy import ALLOW, DENY, REQUIRE_APPROVAL, decide
from nova.policy.decide import POLICY_SCHEMA_VERSION


def policy(**overrides):
    document = {
        "schema_version": POLICY_SCHEMA_VERSION,
        "agent_id": "a",
        "deny": [],
        "baseline": ["kanban_complete"],
        "approval_actions": {},
        "allow": [],
        "unlisted_tool": "allow",
    }
    document.update(overrides)
    return document


# -- failing closed ----------------------------------------------------------


def test_missing_policy_denies():
    """A control that fails open is not a control."""
    decision = decide(None, "anything")
    assert decision.effect == DENY
    assert decision.rule == "policy-missing"


def test_unsupported_schema_denies():
    """A document we cannot read must not become 'allow everything'."""
    decision = decide({"schema_version": 999}, "anything")
    assert decision.effect == DENY
    assert decision.rule == "policy-unsupported"


def test_empty_tool_name_denies():
    assert decide(policy(), "").effect == DENY
    assert decide(policy(), "   ").effect == DENY


# -- precedence --------------------------------------------------------------


def test_explicit_deny_wins_over_allow():
    decision = decide(policy(deny=["x"], allow=["x"]), "x")
    assert decision.effect == DENY
    assert decision.rule == "explicit-deny"


def test_explicit_deny_wins_over_baseline():
    """A customer who denies a tool means it; the compiler warns rather than the runtime
    silently overriding them."""
    assert decide(policy(deny=["kanban_complete"]), "kanban_complete").effect == DENY


def test_explicit_deny_wins_over_approval():
    document = policy(deny=["crm_refund"], approval_actions={"refund": ["crm_refund"]})
    assert decide(document, "crm_refund").effect == DENY


def test_baseline_wins_over_allowlist():
    """Otherwise an allow-list produces agents that run work and never close it."""
    decision = decide(policy(allow=["other"]), "kanban_complete")
    assert decision.effect == ALLOW
    assert decision.rule == "baseline"


def test_approval_wins_over_allowlist():
    document = policy(allow=["crm_refund"], approval_actions={"refund": ["crm_refund"]})
    decision = decide(document, "crm_refund")
    assert decision.effect == REQUIRE_APPROVAL
    assert decision.action == "refund"


# -- allow-list --------------------------------------------------------------


def test_allowlist_grants_listed_tools():
    assert decide(policy(allow=["crm_lookup"]), "crm_lookup").effect == ALLOW


def test_allowlist_denies_unlisted_even_when_default_is_allow():
    """An agent under an allow-list is closed, regardless of the tenant default."""
    document = policy(allow=["crm_lookup"], unlisted_tool="allow")
    decision = decide(document, "something_else")
    assert decision.effect == DENY
    assert decision.rule == "not-in-allowlist"


def test_without_an_allowlist_the_default_applies():
    assert decide(policy(unlisted_tool="allow"), "anything").effect == ALLOW
    assert decide(policy(unlisted_tool="deny"), "anything").effect == DENY


def test_default_deny_is_reported_as_such():
    assert decide(policy(unlisted_tool="deny"), "x").rule == "default-deny"


# -- explanations ------------------------------------------------------------


def test_every_decision_carries_a_reason_and_a_rule():
    """An audit trail with no reason is not a governance record."""
    documents = [
        (policy(deny=["x"]), "x"),
        (policy(), "kanban_complete"),
        (policy(approval_actions={"r": ["x"]}), "x"),
        (policy(allow=["x"]), "x"),
        (policy(allow=["y"]), "x"),
        (policy(unlisted_tool="deny"), "x"),
        (None, "x"),
    ]
    for document, tool in documents:
        decision = decide(document, tool)
        assert decision.reason, f"no reason for {tool} under {document}"
        assert decision.rule, f"no rule for {tool} under {document}"


def test_approval_decision_names_the_business_action():
    """The reason a human reads must be the business action, not the tool."""
    decision = decide(policy(approval_actions={"refund": ["crm_refund"]}), "crm_refund")
    assert "refund" in decision.reason
    assert decision.action == "refund"


@pytest.mark.parametrize("effect", [ALLOW, DENY, REQUIRE_APPROVAL])
def test_effects_are_distinct_strings(effect):
    assert isinstance(effect, str) and effect
