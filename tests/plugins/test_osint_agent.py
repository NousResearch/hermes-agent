"""Tests for osint-agent unified OSINT plugin (no live network/browser)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

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
    )
    assert "統合 OSINT ブリーフィング" in md
    assert "## PDB" in md
    assert "## SitDeck" in md
    assert "## Gov" in md
    assert "## MHLW" in md


@patch("osint_agent_test_pkg.orchestrator._pdb_section", return_value={"success": True, "markdown": "## PDB"})
@patch("osint_agent_test_pkg.orchestrator._sitdeck_section", return_value={"skipped": True})
@patch(
    "osint_agent_test_pkg.orchestrator._gov_feeds_markdown",
    return_value=("", {"skipped": True}),
)
@patch("osint_agent_test_pkg.orchestrator._mhlw_section", return_value={"skipped": True})
def test_generate_integrated_brief_no_save(_mhlw, _gov, _sd, _pdb, tmp_path, monkeypatch):
    orch = _load_module("orchestrator")
    monkeypatch.setattr(orch, "get_hermes_home", lambda: tmp_path / ".hermes")
    result = orch.generate_integrated_brief(
        slot="evening",
        save=False,
        include_sitdeck=False,
        include_mhlw=False,
        include_gov_feeds=False,
    )
    assert result["success"] is True
    assert result["slot"] == "evening"
    assert "統合 OSINT" in result["markdown"]
    assert "saved_markdown" not in result


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
