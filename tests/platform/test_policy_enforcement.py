"""End-to-end policy enforcement: compile, install, and run the plugin as the runtime does.

The installed plugin is loaded from disk by file path, exactly the way the runtime's
plugin discovery loads it, and with NOVA absent from its import path. If these tests pass
only because ``nova`` happens to be importable, they are not testing what ships.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from nova.apply import apply_bundle
from nova.policy import compile_policy
from nova.runtime.hermes.paths import HermesPaths


def load_installed_plugin(home: Path, agent_id: str, name: str):
    """Import the plugin from the profile, the way plugin discovery would."""
    plugin_dir = HermesPaths(home=home).policy_plugin_dir(agent_id)
    spec = importlib.util.spec_from_file_location(
        name, plugin_dir / "__init__.py", submodule_search_locations=[str(plugin_dir)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


@pytest.fixture
def applied(bundle, runtime, audit, home):
    apply_bundle(bundle, runtime, audit=audit)
    return home


# -- installation ------------------------------------------------------------


def test_plugin_is_installed_into_each_agent(applied):
    for agent_id in ("customer-support", "operations"):
        plugin_dir = HermesPaths(home=applied).policy_plugin_dir(agent_id)
        assert (plugin_dir / "__init__.py").is_file()
        assert (plugin_dir / "_decide.py").is_file()
        assert (plugin_dir / "plugin.yaml").is_file()
        assert HermesPaths(home=applied).policy_path(agent_id).is_file()


def test_plugin_is_installed_where_discovery_will_find_it(applied):
    """Workers run with their profile as the runtime home, and discovery scans
    <home>/plugins — so the plugin must sit exactly there."""
    plugin_dir = HermesPaths(home=applied).policy_plugin_dir("customer-support")
    profile = HermesPaths(home=applied).profile_dir("customer-support")
    assert plugin_dir.parent == profile / "plugins"


def test_shipped_decision_module_is_identical_to_the_platform_one(applied):
    """Two implementations of a security decision will eventually disagree."""
    installed = HermesPaths(home=applied).policy_plugin_dir("customer-support") / "_decide.py"
    source = Path("nova/policy/decide.py")
    assert installed.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")


def test_no_plugin_is_installed_without_a_declared_policy(tmp_path, runtime, audit, home):
    """Introducing governance must not restrict agents that predate it."""
    import shutil

    from nova.spec import load_bundle

    from .conftest import EXAMPLE_BUNDLE

    root = tmp_path / "b"
    shutil.copytree(EXAMPLE_BUNDLE, root)
    (root / "policy.yaml").unlink()
    # The example's channel escalates a business action, which a bundle with no policy
    # cannot express. Removing the policy means removing that too — a channel asking for an
    # approval nothing defines is refused at load, which is its own correct error and is
    # covered in test_channels.py.
    (root / "channels.yaml").unlink()
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

    apply_bundle(load_bundle(root), runtime, audit=audit)
    assert not HermesPaths(home=home).policy_plugin_dir("customer-support").exists()
    assert not HermesPaths(home=home).policy_path("customer-support").exists()


# -- enforcement -------------------------------------------------------------


@pytest.mark.parametrize(
    ("tool", "expected"),
    [
        ("crm_lookup", None),            # granted by read_customers
        ("ticket_create", None),         # granted by create_ticket
        ("kanban_complete", None),       # baseline — a worker must report its outcome
        ("crm_refund", "approve"),       # tenant-wide approval
        ("email_send", "approve"),       # agent-specific approval
        ("terminal", "block"),           # explicitly denied
        ("erp_stock_query", "block"),    # another agent's permission
        ("totally_unknown", "block"),    # unlisted, default deny
    ],
)
def test_installed_plugin_enforces(applied, tool, expected):
    plugin = load_installed_plugin(applied, "customer-support", "nova_plugin_support")
    directive = plugin.pre_tool_call(tool_name=tool, args={})
    assert (directive.get("action") if directive else None) == expected


def test_policy_is_per_agent(applied):
    """The same tool gets opposite answers for different agents."""
    support = load_installed_plugin(applied, "customer-support", "nova_p1")
    operations = load_installed_plugin(applied, "operations", "nova_p2")

    assert support.pre_tool_call(tool_name="crm_lookup", args={}) is None
    assert operations.pre_tool_call(tool_name="crm_lookup", args={})["action"] == "block"
    assert operations.pre_tool_call(tool_name="erp_stock_query", args={}) is None
    assert support.pre_tool_call(tool_name="erp_stock_query", args={})["action"] == "block"


def test_escalation_carries_the_business_action_as_its_rule_key(applied):
    """Approving 'refund' once must not also approve every other escalated action."""
    plugin = load_installed_plugin(applied, "customer-support", "nova_p3")
    refund = plugin.pre_tool_call(tool_name="crm_refund", args={})
    email = plugin.pre_tool_call(tool_name="email_send", args={})
    assert refund["rule_key"] != email["rule_key"]
    assert refund["rule_key"] == "nova:refund"


def test_block_message_explains_itself(applied):
    plugin = load_installed_plugin(applied, "customer-support", "nova_p4")
    directive = plugin.pre_tool_call(tool_name="terminal", args={})
    assert "NOVA policy" in directive["message"]
    assert "explicitly denied" in directive["message"]


# -- failing closed ----------------------------------------------------------


def test_a_corrupt_policy_document_blocks_everything(applied):
    """A governance control that fails open is not a control."""
    HermesPaths(home=applied).policy_path("customer-support").write_text(
        "{ not json", encoding="utf-8"
    )
    plugin = load_installed_plugin(applied, "customer-support", "nova_p5")
    assert plugin.pre_tool_call(tool_name="crm_lookup", args={})["action"] == "block"


def test_a_deleted_policy_document_blocks_everything(applied):
    HermesPaths(home=applied).policy_path("customer-support").unlink()
    plugin = load_installed_plugin(applied, "customer-support", "nova_p6")
    assert plugin.pre_tool_call(tool_name="crm_lookup", args={})["action"] == "block"


def test_an_unreadable_schema_blocks_everything(applied):
    path = HermesPaths(home=applied).policy_path("customer-support")
    document = json.loads(path.read_text(encoding="utf-8"))
    document["schema_version"] = 9999
    path.write_text(json.dumps(document), encoding="utf-8")
    plugin = load_installed_plugin(applied, "customer-support", "nova_p7")
    assert plugin.pre_tool_call(tool_name="crm_lookup", args={})["action"] == "block"


# -- governance record -------------------------------------------------------


def test_refusals_and_escalations_are_recorded(applied, audit):
    plugin = load_installed_plugin(applied, "customer-support", "nova_p8")
    plugin.pre_tool_call(tool_name="terminal", args={})
    plugin.pre_tool_call(tool_name="crm_refund", args={})

    records = [event for event in audit.read() if event.kind == "policy.decision"]
    effects = {record.detail["tool"]: record.detail["effect"] for record in records}
    assert effects["terminal"] == "deny"
    assert effects["crm_refund"] == "require_approval"
    assert all(record.subject == "customer-support" for record in records)


def test_permitted_calls_are_not_recorded(applied, audit):
    """Otherwise the governance record buries what a reviewer is looking for."""
    plugin = load_installed_plugin(applied, "customer-support", "nova_p9")
    plugin.pre_tool_call(tool_name="crm_lookup", args={})
    assert [event for event in audit.read() if event.kind == "policy.decision"] == []


def test_every_record_carries_a_reason(applied, audit):
    plugin = load_installed_plugin(applied, "customer-support", "nova_p10")
    plugin.pre_tool_call(tool_name="terminal", args={})
    record = [event for event in audit.read() if event.kind == "policy.decision"][0]
    assert record.detail["reason"]
    assert record.detail["rule"] == "explicit-deny"


# -- re-apply ----------------------------------------------------------------


def test_policy_change_alone_triggers_a_rewrite(bundle, runtime, audit, home):
    """A policy edit with an unchanged agent spec must still reach the runtime."""
    apply_bundle(bundle, runtime, audit=audit)
    spec = bundle.agent("operations")
    before = compile_policy(spec, bundle.policy).document

    from dataclasses import replace

    from nova.policy.model import ActionSpec

    widened = replace(
        bundle.policy,
        actions={
            **bundle.policy.actions,
            "stock_write": ActionSpec(
                name="stock_write", tools=("erp_stock_write",), requires_approval=True
            ),
        },
    )
    after = compile_policy(spec, widened).document
    assert before != after

    report = apply_bundle(replace(bundle, policy=widened), runtime, audit=audit)
    assert "operations" in report.changed
