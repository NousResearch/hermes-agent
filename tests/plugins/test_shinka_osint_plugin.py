from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugins" / "shinka-osint"
SHINKA_ROOT = Path(
    r"C:\Users\downl\Desktop\ShinkaEvolve-OSINT-main\ShinkaEvolve-OSINT-main"
)


def load_plugin():
    package_name = "shinka_osint_test_plugin"
    for name in list(sys.modules):
        if name == package_name or name.startswith(f"{package_name}."):
            del sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        package_name,
        PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    return module


def load_core():
    spec = importlib.util.spec_from_file_location(
        "shinka_osint_core_test",
        PLUGIN_DIR / "core.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["shinka_osint_core_test"] = module
    spec.loader.exec_module(module)
    return module


def test_register_exposes_tools_and_cli_command():
    plugin = load_plugin()

    class Ctx:
        def __init__(self):
            self.tools = []
            self.commands = []
            self.cli_commands = []

        def register_tool(self, **kwargs):
            self.tools.append(kwargs)

        def register_command(self, *args, **kwargs):
            self.commands.append((args, kwargs))

        def register_cli_command(self, **kwargs):
            self.cli_commands.append(kwargs)

    ctx = Ctx()
    plugin.register(ctx)

    assert {tool["name"] for tool in ctx.tools} == {
        "shinka_osint_status",
        "shinka_osint_list_scenarios",
        "shinka_osint_analyze",
        "shinka_osint_briefing",
        "shinka_osint_verify",
        "shinka_osint_audit",
    }
    assert all(tool["toolset"] == "shinka_osint" for tool in ctx.tools)
    assert ctx.commands[0][0][0] == "shinka-osint"
    assert ctx.cli_commands[0]["name"] == "shinka-osint"


def test_match_scenarios_by_domain():
    core = load_core()
    scenarios = [
        {"scenario_id": "a", "domain": "middle_east", "query": "ホルムズ"},
        {"scenario_id": "b", "domain": "taiwan", "query": "台湾有事"},
    ]
    matched = core._match_scenarios(scenarios, domain="middle_east", max_scenarios=2)
    assert [s["scenario_id"] for s in matched] == ["a"]


def test_match_scenarios_by_topic_token():
    core = load_core()
    scenarios = [
        {"scenario_id": "a", "domain": "middle_east", "query": "ホルムズ海峡"},
        {"scenario_id": "b", "domain": "taiwan", "query": "台湾有事と南西諸島"},
    ]
    matched = core._match_scenarios(scenarios, topic="台湾", max_scenarios=2)
    assert matched[0]["scenario_id"] == "b"


def test_policy_ensure_ukraine_in_briefing():
    spec = importlib.util.spec_from_file_location(
        "shinka_osint_policy_test",
        PLUGIN_DIR / "policy.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    assert spec is not None and spec.loader is not None
    pol = importlib.util.module_from_spec(spec)
    sys.modules["shinka_osint_policy_test"] = pol
    spec.loader.exec_module(pol)

    pool = [
        {"scenario_id": "a", "domain": "middle_east", "query": "ホルムズ"},
        {"scenario_id": "japan_russia_military_overview", "domain": "japan_russia", "query": "ウクライナ侵略後のロシア極東軍事態勢"},
    ]
    selected = [pool[0]]
    policy = {"ensure_cyber_scenarios": False, "ensure_ukraine_scenarios": True}
    out = pol.ensure_priority_in_selection(selected, pool, max_scenarios=2, policy=policy)
    ids = [s["scenario_id"] for s in out]
    assert "japan_russia_military_overview" in ids


def test_normalize_domain_ukraine_topic():
    core = load_core()
    scenarios = [
        {"scenario_id": "x", "domain": "middle_east", "query": "ホルムズ"},
        {"scenario_id": "y", "domain": "japan_russia", "query": "ウクライナ情勢と極東"},
    ]
    matched = core._match_scenarios(scenarios, topic="ウクライナ情勢", max_scenarios=2)
    assert matched[0]["scenario_id"] == "y"


def test_handle_status_json():
    core = load_core()
    with patch.object(core.bridge, "root_status", return_value={"root_exists": False}):
        with patch.object(core.bridge, "check_available", return_value=False):
            payload = json.loads(core.handle_status({}))
    assert payload["available"] is False


@pytest.mark.skipif(not SHINKA_ROOT.is_dir(), reason="ShinkaEvolve-OSINT checkout not present")
def test_integration_status_against_local_checkout(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    core = load_core()
    core.bridge.save_root(SHINKA_ROOT, persist_env=False)
    payload = json.loads(core.handle_status({}))
    assert payload["root_exists"] is True
    assert payload.get("available") is True
    assert payload.get("example_count", 0) >= 1


@pytest.mark.skipif(not SHINKA_ROOT.is_dir(), reason="ShinkaEvolve-OSINT checkout not present")
def test_integration_list_scenarios(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    core = load_core()
    core.bridge.save_root(SHINKA_ROOT, persist_env=False)
    payload = json.loads(
        core.handle_list_scenarios({"example": "milspec_security_jp"})
    )
    assert payload["count"] >= 40
