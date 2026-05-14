import json

from agent.execution_policy import (
    decide_tool_call,
    default_execution_policy,
    derive_child_policy,
    normalize_execution_policy,
)
from model_tools import handle_function_call


def test_cron_default_blocks_skill_mutation_and_delegation():
    policy = default_execution_policy("cron")

    skill = decide_tool_call(policy, "skill_manage", toolset="skills")
    delegate = decide_tool_call(policy, "delegate_task", toolset="delegation")

    assert skill.action == "block"
    assert skill.code == "tool_denied"
    assert delegate.action == "block"
    assert delegate.code in {"tool_denied", "recursive_delegation_disabled"}


def test_interactive_default_audits_high_risk_without_blocking():
    policy = default_execution_policy("interactive")

    decision = decide_tool_call(policy, "terminal", toolset="terminal")

    assert decision.action == "audit"
    assert decision.allows_execution is True
    assert decision.code == "high_risk_tool_observed"


def test_child_policy_denies_recursive_delegation_and_narrows_toolsets():
    parent = normalize_execution_policy({"mode": "enforce", "allow_toolsets": ["file", "terminal"]})

    child = derive_child_policy(parent, child_toolsets=["file"])

    assert child.source == "delegate"
    assert child.parent_policy_id == parent.policy_id
    assert child.disable_recursive_delegation is True
    assert child.allow_toolsets == frozenset({"file"})
    assert decide_tool_call(child, "delegate_task", toolset="delegation").action == "block"
    assert decide_tool_call(child, "terminal", toolset="terminal").action == "block"


def test_handle_function_call_blocks_by_execution_policy_before_dispatch():
    audit_events = []
    result = handle_function_call(
        "terminal",
        {"command": "echo should-not-run"},
        execution_policy={"mode": "enforce", "deny_tools": ["terminal"]},
        policy_audit_events=audit_events,
    )
    data = json.loads(result)

    assert "error" in data
    assert data["policy_audit_event"]["code"] == "tool_denied"
    assert audit_events and audit_events[0]["tool_name"] == "terminal"
