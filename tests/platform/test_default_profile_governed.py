"""The gateway's own default profile may do nothing.

It is not a NOVA agent — it is the home the gateway runs from — and it had no policy plugin
and the runtime's full tool set, terminal and code execution included. A live gateway test
reached it through the API server's unprefixed route and read /etc/hostname with no policy
at all. Apply now gives it a policy that refuses every tool.

The live half — a real ``hermes gateway run`` proving the refusal, and proving it does not
leak into the agents' own conversations — is ``test_gateway_governance_live.py``.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys

import yaml

from nova.apply import apply_bundle
from nova.policy.decide import DENY, decide


def _load_root_plugin(home):
    plugin_dir = home / "plugins" / "nova-policy"
    name = f"nova_root_policy_{len(sys.modules)}"
    spec = importlib.util.spec_from_file_location(
        name, plugin_dir / "__init__.py", submodule_search_locations=[str(plugin_dir)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def test_the_default_profile_refuses_every_tool(bundle, runtime, audit, home, monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    apply_bundle(bundle, runtime, audit=audit)
    document = json.loads((home / "nova-policy.json").read_text())
    assert document["refuse_all"]
    plugin = _load_root_plugin(home)
    for tool in ("terminal", "execute_code", "read_file", "web_search", "kanban_complete"):
        result = plugin.pre_tool_call(tool_name=tool, args={})
        assert result and result["action"] == "block", tool
        assert "not a NOVA agent" in result["message"]


def test_it_is_enabled_in_the_root_config_without_disturbing_anything_else(bundle, runtime, audit, home):
    (home / "config.yaml").write_text(yaml.safe_dump({
        "platforms": {"telegram": {"enabled": True}},
        "plugins": {"enabled": ["someone-elses-plugin"]},
    }))
    apply_bundle(bundle, runtime, audit=audit)
    config = yaml.safe_load((home / "config.yaml").read_text())
    assert config["platforms"] == {"telegram": {"enabled": True}}, "operator keys are left alone"
    assert config["plugins"]["enabled"] == ["someone-elses-plugin", "nova-policy"]
    assert config["plugins"]["entries"]["nova-policy"] == {"allow_tool_override": False}
    assert "model" in config, "and the runtime defaults are still written beside it"


def test_a_second_apply_changes_nothing(bundle, runtime, audit, home):
    apply_bundle(bundle, runtime, audit=audit)
    before = {p: p.read_bytes() for p in home.rglob("*") if p.is_file() and "profiles" not in p.parts}
    changed = runtime.govern_default_context(audit=audit, correlation_id="c")
    assert changed is False
    after = {p: p.read_bytes() for p in before}
    assert before == after


def test_not_installed_without_a_declared_policy(tmp_path, runtime, audit, home):
    from nova.spec import load_bundle

    from .conftest import EXAMPLE_BUNDLE

    root = tmp_path / "b"
    shutil.copytree(EXAMPLE_BUNDLE, root)
    (root / "policy.yaml").unlink()
    # The example's channel requires an approval, which a bundle with no policy refuses.
    (root / "channels.yaml").write_text("channels: []\n")
    apply_bundle(load_bundle(root), runtime, audit=audit)
    assert not (home / "nova-policy.json").exists()


def test_the_refusal_is_the_first_rule_and_names_why():
    decision = decide({"schema_version": 1, "refuse_all": "not an agent", "baseline": ["kanban_complete"],
                       "allow": ["kanban_complete"]}, "kanban_complete")
    assert (decision.effect, decision.rule) == (DENY, "not-an-agent")
