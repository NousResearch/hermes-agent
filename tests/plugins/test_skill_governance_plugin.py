"""Tests for the skill-governance plugin.

Covers the bundled plugin at ``plugins/skill_governance/``:

* candidate catalog integrity for Vladimir-approved skill-to-plugin roadmap;
* tool handlers returning JSON-only results with filters and detailed plans;
* bundled standalone plugin discovery and opt-in tool registration.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    yield hermes_home


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_plugin_module(module_file: str):
    plugin_dir = _repo_root() / "plugins" / "skill_governance"
    module_name = "hermes_plugins.skill_governance_under_test"
    if "hermes_plugins" not in sys.modules:
        ns = types.ModuleType("hermes_plugins")
        ns.__path__ = []
        sys.modules["hermes_plugins"] = ns
    spec = importlib.util.spec_from_file_location(
        f"{module_name}.{module_file.removesuffix('.py')}",
        plugin_dir / module_file,
        submodule_search_locations=[str(plugin_dir)] if module_file == "__init__.py" else None,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    if module_file == "__init__.py":
        mod.__package__ = module_name
        mod.__path__ = [str(plugin_dir)]
        sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_catalog():
    return _load_plugin_module("catalog.py")


def _load_tools():
    plugin_dir = _repo_root() / "plugins" / "skill_governance"
    module_name = "hermes_plugins.skill_governance_under_test"
    if "hermes_plugins" not in sys.modules:
        ns = types.ModuleType("hermes_plugins")
        ns.__path__ = []
        sys.modules["hermes_plugins"] = ns
    if module_name not in sys.modules:
        pkg = types.ModuleType(module_name)
        pkg.__path__ = [str(plugin_dir)]
        pkg.__package__ = module_name
        sys.modules[module_name] = pkg
    spec = importlib.util.spec_from_file_location(
        f"{module_name}.tools",
        plugin_dir / "tools.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = module_name
    sys.modules[f"{module_name}.tools"] = mod
    spec.loader.exec_module(mod)
    return mod


class TestCandidateCatalog:
    def test_candidate_catalog_contract(self):
        catalog = _load_catalog()
        ids = [candidate["id"] for candidate in catalog.CANDIDATES]
        assert ids
        assert "skill_governance" in ids
        assert min(candidate["wave"] for candidate in catalog.CANDIDATES) == 1
        assert max(candidate["wave"] for candidate in catalog.CANDIDATES) <= 3

    def test_candidate_ids_and_tool_names_are_unique(self):
        catalog = _load_catalog()
        ids = [candidate["id"] for candidate in catalog.CANDIDATES]
        assert len(ids) == len(set(ids))
        all_tools = [tool for candidate in catalog.CANDIDATES for tool in candidate["tools"]]
        assert len(all_tools) == len(set(all_tools))

    def test_each_candidate_has_sources_guardrails_and_business_value(self):
        catalog = _load_catalog()
        required_fields = {
            "id",
            "title",
            "priority",
            "wave",
            "areas",
            "sources",
            "tools",
            "business_value",
            "guardrails",
            "requires_live_go",
        }
        for candidate in catalog.CANDIDATES:
            assert required_fields <= candidate.keys()
            assert candidate["title"]
            assert candidate["priority"] in {"very_high", "high", "medium_high", "medium"}
            assert candidate["wave"] in {1, 2, 3}
            assert candidate["sources"]
            assert candidate["tools"]
            assert candidate["business_value"]
            assert candidate["guardrails"]


class TestSkillGovernanceTools:
    def test_find_candidates_returns_top_priorities_by_default(self):
        tools = _load_tools()
        payload = json.loads(tools.skills_find_plugin_candidates({}))
        assert payload["success"] is True
        assert payload["count"] == 12
        assert [item["id"] for item in payload["candidates"][:3]] == [
            "bitrix_ops",
            "telegram_thread_router",
            "management_digest",
        ]

    def test_find_candidates_filters_by_area_and_limit(self):
        tools = _load_tools()
        payload = json.loads(
            tools.skills_find_plugin_candidates({"area": "saturn-business", "limit": 2})
        )
        assert payload["success"] is True
        assert payload["count"] == 2
        assert [item["id"] for item in payload["candidates"]] == [
            "procurement_tender_pipeline",
            "document_factory",
        ]

    def test_candidate_plan_contains_mvp_tools_and_gates(self):
        tools = _load_tools()
        payload = json.loads(tools.skills_to_plugin_plan({"candidate_id": "bitrix_ops"}))
        assert payload["success"] is True
        assert payload["candidate"]["id"] == "bitrix_ops"
        assert "bitrix_get_task" in payload["candidate"]["tools"]
        assert payload["phases"][0]["name"] == "contract"
        assert any("live" in gate.lower() for gate in payload["gates"])

    def test_unknown_candidate_returns_error_json(self):
        tools = _load_tools()
        payload = json.loads(tools.skills_to_plugin_plan({"candidate_id": "missing"}))
        assert payload["success"] is False
        assert "unknown candidate" in payload["error"]

    def test_malformed_args_are_handled_inside_plugin_json_contract(self):
        tools = _load_tools()
        candidates_payload = json.loads(tools.skills_find_plugin_candidates(None))
        assert candidates_payload["success"] is True
        assert candidates_payload["count"] == 12

        plan_payload = json.loads(tools.skills_to_plugin_plan(None))
        assert plan_payload["success"] is False
        assert plan_payload["error"] == "candidate_id is required"

    def test_roadmap_summary_summarizes_waves_and_guarded_live_candidates(self):
        tools = _load_tools()
        payload = json.loads(tools.skills_plugin_roadmap_summary({}))
        assert payload["success"] is True
        assert payload["summary_type"] == "static_roadmap_catalog"
        assert payload["total_candidates"] == 12
        assert payload["waves"] == {"1": 4, "2": 4, "3": 4}
        assert payload["registered_tools"] == [
            "skills_plugin_roadmap_summary",
            "skills_find_plugin_candidates",
            "skills_to_plugin_plan",
        ]
        assert "ozon_marketplace_import" in payload["requires_live_go"]


class TestBundledDiscovery:
    def _write_enabled_config(self, hermes_home, names):
        import yaml

        cfg_path = hermes_home / "config.yaml"
        cfg_path.write_text(yaml.safe_dump({"plugins": {"enabled": list(names)}}))

    def test_skill_governance_discovered_but_not_loaded_by_default(self, _isolate_env):
        from hermes_cli import plugins as pmod

        mgr = pmod.PluginManager()
        mgr.discover_and_load()
        assert "skill-governance" in mgr._plugins
        loaded = mgr._plugins["skill-governance"]
        assert loaded.manifest.source == "bundled"
        assert not loaded.enabled
        assert loaded.error and "not enabled" in loaded.error

    def test_skill_governance_loads_when_enabled(self, _isolate_env):
        self._write_enabled_config(_isolate_env, ["skill-governance"])

        from hermes_cli import plugins as pmod

        mgr = pmod.PluginManager()
        mgr.discover_and_load()
        loaded = mgr._plugins["skill-governance"]
        assert loaded.enabled
        assert sorted(loaded.tools_registered) == [
            "skills_find_plugin_candidates",
            "skills_plugin_roadmap_summary",
            "skills_to_plugin_plan",
        ]
