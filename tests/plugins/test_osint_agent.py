"""Tests for osint-agent unified OSINT plugin (no live network/browser)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugins" / "osint-agent"


def _load_module(name: str):
    pkg = "osint_agent_test_pkg"
    if pkg not in sys.modules:
        import types

        package = types.ModuleType(pkg)
        package.__path__ = [str(PLUGIN_DIR)]  # type: ignore[attr-defined]
        sys.modules[pkg] = package
        for stem in (
            "plugin_loader",
            "computer_use_playbooks",
            "orchestrator",
            "core",
            "stack",
            "cli",
            "cron_setup",
        ):
            mod_name = f"{pkg}.{stem}"
            spec = importlib.util.spec_from_file_location(mod_name, PLUGIN_DIR / f"{stem}.py")
            assert spec and spec.loader
            mod = importlib.util.module_from_spec(spec)
            mod.__package__ = pkg
            sys.modules[mod_name] = mod
            spec.loader.exec_module(mod)
    return sys.modules[f"{pkg}.{name}"]


def test_build_integrated_markdown_sections():
    orch = _load_module("orchestrator")
    md = orch.build_integrated_markdown(
        slot="morning",
        pdb={"success": True, "markdown": "## PDB\n\nbody"},
        sitdeck={"success": True, "digest": "## SitDeck\n\ncrawl"},
        gov_md="## Gov\n\nfeed",
        mhlw={"markdown": "## MHLW\n\nok"},
        wm_free={"markdown": "## World Monitor Free（ウェブ JSON）\n\n- headline"},
        multilayer_note="- **L5_web_search**: web_search — diversify",
    )
    assert "統合 OSINT ブリーフィング" in md
    assert "## PDB" in md
    assert "## SitDeck" in md
    assert "## Gov" in md
    assert "## MHLW" in md
    assert "World Monitor Free" in md
    assert "多層収集メモ" in md
    assert "Computer Use" in md
    assert "web_search" in md


@patch(
    "osint_agent_test_pkg.orchestrator._pdb_section",
    return_value={"success": True, "markdown": "## PDB"},
)
@patch(
    "osint_agent_test_pkg.orchestrator._sitdeck_section",
    return_value={"skipped": True},
)
@patch(
    "osint_agent_test_pkg.orchestrator._gov_feeds_markdown",
    return_value=("", {"skipped": True}),
)
@patch(
    "osint_agent_test_pkg.orchestrator._mhlw_section",
    return_value={"skipped": True},
)
@patch(
    "osint_agent_test_pkg.orchestrator._worldmonitor_free_section",
    return_value={"skipped": True, "reason": "mocked"},
)
def test_generate_integrated_brief_no_save(_wm, _mhlw, _gov, _sd, _pdb, tmp_path, monkeypatch):
    orch = _load_module("orchestrator")
    monkeypatch.setattr(orch, "get_hermes_home", lambda: tmp_path / ".hermes")
    result = orch.generate_integrated_brief(
        slot="evening",
        save=False,
        include_sitdeck=False,
        include_mhlw=False,
        include_gov_feeds=False,
        include_wm_free=False,
        include_multilayer_plan=True,
    )
    assert result["success"] is True
    assert result["slot"] == "evening"
    assert "統合 OSINT" in result["markdown"]
    assert "saved_markdown" not in result
    assert "multilayer_plan" in result["sections"]
    layers = (result["sections"]["multilayer_plan"] or {}).get("layers") or []
    assert any(layer.get("id") == "L5_web_search" for layer in layers)
    assert "多層収集メモ" in result["markdown"]


def test_handle_status_json():
    core = _load_module("core")
    with patch.object(
        core.plugin_loader,
        "load_plugin_modules",
        side_effect=RuntimeError("no sitdeck in test"),
    ):
        payload = json.loads(core.handle_status({}))
    assert payload["success"] is True
    assert "sitdeck-osint" in payload["stack"]
    assert "computer_use" in payload["stack"]
    assert "web" in payload["stack"]
    assert payload["computer_use"]["tool"] == "osint_agent_computer_use_plan"


def test_computer_use_playbooks_structure():
    playbooks = _load_module("computer_use_playbooks")
    wm = playbooks.worldmonitor_manual_playbook()
    sd = playbooks.sitdeck_computer_use_playbook()
    multi = playbooks.multilayer_search_plan(topic="台湾海峡")
    full = playbooks.build_full_osint_playbook(topic="台湾海峡")

    assert wm["url"] == "https://worldmonitor.app/"
    assert wm["mode"] == "computer_use_manual"
    assert len(wm["steps"]) >= 4
    assert sd["app_url"].startswith("https://app.sitdeck.com")
    assert any(step["id"] == "sd0" for step in sd["steps"])
    layer_ids = [layer["id"] for layer in multi["layers"]]
    assert layer_ids == [
        "L1_worldmonitor_free",
        "L2_gov_rss",
        "L3_sitdeck",
        "L4_computer_use_wm",
        "L5_web_search",
        "L6_shinka_fusion",
    ]
    l5 = next(layer for layer in multi["layers"] if layer["id"] == "L5_web_search")
    assert l5["tool"] == "web_search (toolset=web)"
    assert any("台湾海峡" in q for q in l5["queries"])
    assert full["success"] is True
    assert "computer_use" in full
    assert "multilayer" in full
    assert "hermes osint-agent stack enable" in full["enable"]


def test_handle_computer_use_plan_and_multilayer():
    core = _load_module("core")
    cu_all = json.loads(core.handle_computer_use_plan({"target": "all", "topic": "サイバー"}))
    assert cu_all["success"] is True
    assert "worldmonitor" in cu_all["computer_use"]
    assert "sitdeck" in cu_all["computer_use"]

    cu_wm = json.loads(core.handle_computer_use_plan({"target": "worldmonitor"}))
    assert cu_wm["playbook"]["target"] == "worldmonitor"

    cu_sd = json.loads(core.handle_computer_use_plan({"target": "sitdeck"}))
    assert cu_sd["playbook"]["target"] == "sitdeck"

    with patch.object(
        core.orchestrator,
        "_worldmonitor_free_section",
        return_value={"success": True, "headline_count": 0, "markdown": "## WM Free"},
    ):
        multi = json.loads(
            core.handle_multilayer_collect(
                {"topic": "サイバー", "fetch_wm_free": True, "queries": ["q1", "q2"]}
            )
        )
    assert multi["success"] is True
    assert multi["wm_free"]["success"] is True
    l5 = next(
        layer for layer in multi["plan"]["layers"] if layer["id"] == "L5_web_search"
    )
    assert l5["queries"] == ["q1", "q2"]
    assert any("web_search" in step for step in multi["next"])


def test_handle_brief_passes_wm_free_kwarg():
    core = _load_module("core")
    fake = {
        "success": True,
        "markdown": "# ok",
        "slot": "morning",
        "sections": {},
    }
    with patch.object(
        core.orchestrator,
        "generate_integrated_brief",
        return_value=fake,
    ) as brief_mock:
        payload = json.loads(core.handle_brief({"include_wm_free": False, "save": False}))
    assert payload["success"] is True
    kwargs = brief_mock.call_args.kwargs
    assert kwargs["include_wm_free"] is False
    assert kwargs["wm_tier"] == "free"


def test_cron_dry_run():
    cron = _load_module("cron_setup")
    with patch.object(cron, "_preflight_delivery", return_value={"ok": True, "deliver": "local"}):
        result = cron.install_integrated_cron(dry_run=True, deliver="local")
    assert result["success"] is True
    assert result["dry_run"] is True
    assert len(result["jobs"]) == 2
    assert result["jobs"][0]["name"] == cron.JOB_MORNING


@patch("hermes_cli.config.load_config", return_value={})
@patch("hermes_cli.tools_config._get_platform_tools", return_value=[])
@patch("hermes_cli.tools_config._save_platform_tools")
@patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set())
@patch("hermes_cli.plugins_cmd._resolve_plugin_key", side_effect=lambda n: n)
@patch("hermes_cli.plugins_cmd._save_enabled_set")
def test_setup_dry_run(_save_en, _resolve, _get_en, _save_tools, _get_tools, _load_cfg):
    stack = _load_module("stack")
    with patch.object(
        stack.plugin_loader,
        "load_plugin_modules",
        side_effect=RuntimeError("skip wm mcp"),
    ):
        result = stack.enable_osint_agent_stack(dry_run=True)
    assert result["success"] is True
    assert result["dry_run"] is True
    assert "osint-agent" in result["plugins"]
    assert "computer_use" in stack.TOOLSETS
    assert "web" in stack.TOOLSETS
    assert "search" in stack.TOOLSETS
    next_blob = " ".join(result["next_steps"])
    assert "computer-use" in next_blob
    assert "computer_use" in next_blob
