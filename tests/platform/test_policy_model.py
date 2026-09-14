"""Policy declaration and compilation."""

from __future__ import annotations

import shutil

import pytest

from nova.errors import SpecError
from nova.policy import PolicySpec, compile_policy
from nova.spec import AgentSpec, load_bundle

from .conftest import EXAMPLE_BUNDLE


def _policy(**overrides):
    data = {
        "actions": {"refund": {"tools": ["crm_refund"], "requires_approval": True}},
        "permissions": {"read_customers": {"tools": ["crm_lookup"]}},
        "defaults": {"unlisted_tool": "deny"},
    }
    data.update(overrides)
    return PolicySpec.parse(data)


def _agent(**overrides):
    data = {"id": "a", "name": "A"}
    data.update(overrides)
    return AgentSpec.parse(data)


# -- declaration -------------------------------------------------------------


def test_action_without_tools_is_rejected():
    """An action nothing can trigger is a control that looks present and is not."""
    with pytest.raises(SpecError, match="names no tools"):
        PolicySpec.parse({"actions": {"refund": {"description": "x"}}})


def test_permission_without_tools_is_rejected():
    with pytest.raises(SpecError, match="grants no tools"):
        PolicySpec.parse({"permissions": {"read": {"description": "x"}}})


def test_unknown_field_is_rejected():
    with pytest.raises(SpecError, match="unknown field"):
        PolicySpec.parse({"actions": {"refund": {"tools": ["x"], "requires_aproval": True}}})


def test_unlisted_tool_is_constrained():
    with pytest.raises(SpecError, match="must be one of"):
        PolicySpec.parse({"defaults": {"unlisted_tool": "maybe"}})


def test_baseline_defaults_cover_task_reporting():
    """The default baseline must let a worker close its own task."""
    spec = PolicySpec.parse({})
    assert "kanban_complete" in spec.baseline_tools
    assert "kanban_block" in spec.baseline_tools


def test_action_lookup_by_tool():
    spec = _policy()
    assert spec.action_for_tool("crm_refund").name == "refund"
    assert spec.action_for_tool("unrelated") is None


# -- compilation -------------------------------------------------------------


def test_permissions_become_allowed_tools():
    compiled = compile_policy(_agent(permissions=["read_customers"]), _policy())
    assert "crm_lookup" in compiled.document["allow"]


def test_agent_without_permissions_has_no_allowlist():
    """Phase 1 bundles must keep working when governance is introduced."""
    compiled = compile_policy(_agent(), _policy())
    assert compiled.has_allowlist is False


def test_tenant_default_approval_applies_to_every_agent():
    compiled = compile_policy(_agent(), _policy())
    assert "refund" in compiled.document["approval_actions"]


def test_agent_can_require_approval_the_tenant_does_not():
    policy = _policy(
        actions={
            "refund": {"tools": ["crm_refund"], "requires_approval": True},
            "email": {"tools": ["email_send"], "requires_approval": False},
        }
    )
    without = compile_policy(_agent(), policy)
    assert "email" not in without.document["approval_actions"]
    with_approval = compile_policy(_agent(approval={"required_for": ["email"]}), policy)
    assert "email" in with_approval.document["approval_actions"]


def test_denying_a_baseline_tool_warns():
    """The failure mode is work that runs and never closes — warn at build time."""
    compiled = compile_policy(_agent(tools={"deny": ["kanban_complete"]}), _policy())
    assert any("baseline" in warning for warning in compiled.warnings)


def test_denying_a_granted_tool_warns():
    compiled = compile_policy(
        _agent(permissions=["read_customers"], tools={"deny": ["crm_lookup"]}), _policy()
    )
    assert any("deny wins" in warning for warning in compiled.warnings)


def test_a_clean_agent_compiles_without_warnings():
    compiled = compile_policy(_agent(permissions=["read_customers"]), _policy())
    assert compiled.warnings == ()


def test_compiled_document_is_plain_data():
    """The enforcement point reads it with the standard library alone."""
    import json

    compiled = compile_policy(_agent(permissions=["read_customers"]), _policy())
    assert json.loads(json.dumps(compiled.document)) == compiled.document


# -- bundle integration ------------------------------------------------------


def test_example_bundle_declares_a_policy(bundle):
    assert bundle.policy is not None
    assert "refund" in bundle.policy.actions


def test_bundle_without_policy_still_loads(tmp_path):
    root = tmp_path / "b"
    shutil.copytree(EXAMPLE_BUNDLE, root)
    (root / "policy.yaml").unlink()
    # The agents reference policy names, so strip those too for a pre-governance bundle.
    support = root / "agents" / "customer-support.yaml"
    text = support.read_text(encoding="utf-8")
    text = text.replace("permissions:\n  - read_customers\n  - create_ticket\n", "")
    text = text.replace("  required_for: [send_external_email]\n", "  required_for: []\n")
    support.write_text(text, encoding="utf-8")
    ops = root / "agents" / "operations.yaml"
    ops.write_text(
        ops.read_text(encoding="utf-8").replace("permissions:\n  - read_inventory\n\n", ""),
        encoding="utf-8",
    )
    assert load_bundle(root).policy is None


def test_unknown_permission_fails_the_bundle(tmp_path):
    """A permission that grants nothing must not pass a security review unnoticed."""
    root = tmp_path / "b"
    shutil.copytree(EXAMPLE_BUNDLE, root)
    path = root / "agents" / "operations.yaml"
    path.write_text(
        path.read_text(encoding="utf-8").replace("read_inventory", "read_everything"),
        encoding="utf-8",
    )
    with pytest.raises(SpecError, match="does not define"):
        load_bundle(root)


def test_unknown_approval_action_fails_the_bundle(tmp_path):
    root = tmp_path / "b"
    shutil.copytree(EXAMPLE_BUNDLE, root)
    path = root / "agents" / "customer-support.yaml"
    path.write_text(
        path.read_text(encoding="utf-8").replace("send_external_email", "wire_transfer"),
        encoding="utf-8",
    )
    with pytest.raises(SpecError, match="does not define"):
        load_bundle(root)


def test_a_tool_may_perform_only_one_action(tmp_path):
    root = tmp_path / "b"
    shutil.copytree(EXAMPLE_BUNDLE, root)
    path = root / "policy.yaml"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "  send_external_email:\n    description: Send email to someone outside the company\n    tools: [email_send]",
            "  send_external_email:\n    description: Send email\n    tools: [crm_refund]",
        ),
        encoding="utf-8",
    )
    with pytest.raises(SpecError, match="one tool performs one business action"):
        load_bundle(root)


def test_runtime_plumbing_does_not_change_an_agents_identity():
    """Moving the audit log is not a policy change; it must not show as drift."""
    from nova.policy import agent_digest

    spec = _agent(permissions=["read_customers"])
    compiled = compile_policy(spec, _policy())
    before = agent_digest(spec, compiled)
    compiled.document["audit_log"] = "/somewhere/else/audit.jsonl"
    compiled.document["tenant_id"] = "acme"
    assert agent_digest(spec, compiled) == before


def test_a_policy_change_does_change_identity():
    """A widened permission must reach the runtime on the next apply."""
    from nova.policy import agent_digest

    spec = _agent(permissions=["read_customers"])
    narrow = agent_digest(spec, compile_policy(spec, _policy()))
    wide = agent_digest(
        spec,
        compile_policy(
            spec, _policy(permissions={"read_customers": {"tools": ["crm_lookup", "crm_search"]}})
        ),
    )
    assert narrow != wide
