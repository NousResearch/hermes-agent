from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugins" / "worldmonitor-osint"


def load_plugin():
    package_name = "worldmonitor_osint_test_plugin"
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
        "worldmonitor_osint_core_test",
        PLUGIN_DIR / "core.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["worldmonitor_osint_core_test"] = module
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
        "worldmonitor_status",
        "worldmonitor_snapshot",
        "worldmonitor_free_crawl",
        "worldmonitor_country_brief",
        "worldmonitor_fusion_report",
        "worldmonitor_dev_status",
        "worldmonitor_dev_start",
        "worldmonitor_dev_stop",
    }
    assert all(tool["toolset"] == "worldmonitor_osint" for tool in ctx.tools)
    assert ctx.commands[0][0][0] == "worldmonitor-osint"
    assert ctx.cli_commands[0]["name"] == "worldmonitor-osint"


def test_handle_status_without_network():
    core = load_core()
    with patch.object(core.api, "connectivity_status", return_value={"api_base": "https://api.worldmonitor.app"}):
        with patch.object(core.auth_setup, "probe_sidecar", return_value={"running": False}):
            with patch.object(core.auth_setup, "_mcp_oauth_configured", return_value={"configured": False}):
                with patch.object(core, "_load_shinka_core", side_effect=ImportError("no shinka")):
                    with patch("hermes_cli.mcp_config._get_mcp_servers", return_value={}):
                        payload = json.loads(core.handle_status({}))
    assert payload["success"] is True
    assert payload["egov_law_mcp_configured"] is False


def test_fusion_report_merges_blocks():
    core = load_core()
    wm_snapshot = {"success": True, "sections": {"country_risk_jp": {}}}
    shinka_result = {"success": True, "scenario_count": 1, "runs": []}

    class FakeShinka:
        class bridge:
            @staticmethod
            def resolve_default_example():
                return "milspec_security_jp"

        @staticmethod
        def briefing(**_kwargs):
            return shinka_result

    with patch.object(core, "snapshot", return_value=wm_snapshot):
        with patch.object(core, "_load_shinka_core", return_value=FakeShinka()):
            payload = core.fusion_report(topic="テスト", source_mode="mock", save_report=False)

    assert payload["worldmonitor"] == wm_snapshot
    assert payload["shinka_milspec"] == shinka_result
    assert "egov_law_mcp" in payload["primary_sources"]


def load_auth_setup():
    spec = importlib.util.spec_from_file_location(
        "worldmonitor_osint_auth_test",
        PLUGIN_DIR / "auth_setup.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["worldmonitor_osint_auth_test"] = module
    spec.loader.exec_module(module)
    return module


def test_validate_wm_key():
    auth_mod = load_auth_setup()
    ok, _ = auth_mod.validate_wm_key("wm_" + "a" * 40)
    assert ok is True
    bad, _ = auth_mod.validate_wm_key("sk-or-v1-not-worldmonitor")
    assert bad is False


def test_auth_guidance_rejects_llm_key_reuse():
    core = load_core()
    payload = core.status()
    guidance = payload.get("auth_guidance") or {}
    assert guidance.get("llm_keys_cannot_be_reused") is True
    assert guidance.get("codex_oauth_cannot_be_reused") is True


def test_egov_law_manifest_exists():
    manifest = Path(__file__).resolve().parents[2] / "optional-mcps" / "egov-law" / "manifest.yaml"
    assert manifest.is_file()
    text = manifest.read_text(encoding="utf-8")
    assert "name: egov-law" in text
    assert "egov-law-mcp" in text


def test_worldmonitor_mcp_manifest_exists():
    manifest = Path(__file__).resolve().parents[2] / "optional-mcps" / "worldmonitor" / "manifest.yaml"
    assert manifest.is_file()
    text = manifest.read_text(encoding="utf-8")
    assert "name: worldmonitor" in text
    assert "worldmonitor.app/mcp" in text


def test_free_crawl_offline():
    core = load_core()
    fake = {
        "success": True,
        "tier": "free_web",
        "sections": {"news_digest": {"categories": {}}},
        "news_headlines": [{"title": "t"}],
    }
    with patch.object(core.free_web, "free_snapshot", return_value=fake):
        payload = core.free_crawl(news_limit=5)
    assert payload["tier"] == "free_web"
    assert payload["news_headlines"][0]["title"] == "t"


def test_resolve_repo_path_honors_env(tmp_path, monkeypatch):
    (tmp_path / "package.json").write_text("{}")
    monkeypatch.setenv("WORLDMONITOR_REPO", str(tmp_path))
    dev_spec = importlib.util.spec_from_file_location(
        "worldmonitor_osint_dev_repo_test",
        PLUGIN_DIR / "dev_server.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    assert dev_spec is not None and dev_spec.loader is not None
    module = importlib.util.module_from_spec(dev_spec)
    sys.modules["worldmonitor_osint_dev_repo_test"] = module
    dev_spec.loader.exec_module(module)
    assert module.resolve_repo_path() == tmp_path


def test_probe_dev_server_offline():
    dev_spec = importlib.util.spec_from_file_location(
        "worldmonitor_osint_dev_probe_test",
        PLUGIN_DIR / "dev_server.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    assert dev_spec is not None and dev_spec.loader is not None
    dev_mod = importlib.util.module_from_spec(dev_spec)
    sys.modules["worldmonitor_osint_dev_probe_test"] = dev_mod
    dev_spec.loader.exec_module(dev_mod)
    payload = dev_mod.probe_dev_server(port=31999)
    assert payload["running"] is False
    assert payload["port"] == 31999


def test_resolve_bind_host_tailscale(monkeypatch):
    dev_spec = importlib.util.spec_from_file_location(
        "worldmonitor_osint_dev_bind_test",
        PLUGIN_DIR / "dev_server.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    assert dev_spec is not None and dev_spec.loader is not None
    dev_mod = importlib.util.module_from_spec(dev_spec)
    sys.modules["worldmonitor_osint_dev_bind_test"] = dev_mod
    dev_spec.loader.exec_module(dev_mod)
    monkeypatch.setattr(dev_mod, "tailscale_ipv4", lambda: "100.91.183.75")
    bind, ts_ip = dev_mod._resolve_bind_host(tailscale=True)
    assert bind == "0.0.0.0"
    assert ts_ip == "100.91.183.75"


def test_vite_npm_command_includes_host_bind():
    dev_spec = importlib.util.spec_from_file_location(
        "worldmonitor_osint_dev_cmd_test",
        PLUGIN_DIR / "dev_server.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    assert dev_spec is not None and dev_spec.loader is not None
    dev_mod = importlib.util.module_from_spec(dev_spec)
    sys.modules["worldmonitor_osint_dev_cmd_test"] = dev_mod
    dev_spec.loader.exec_module(dev_mod)
    cmd = dev_mod._vite_npm_command("npm", "dev", port=3000, bind="0.0.0.0")
    assert cmd == ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "3000"]


def test_start_dev_tailscale_dry_run(monkeypatch):
    dev_spec = importlib.util.spec_from_file_location(
        "worldmonitor_osint_dev_start_ts_test",
        PLUGIN_DIR / "dev_server.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    assert dev_spec is not None and dev_spec.loader is not None
    dev_mod = importlib.util.module_from_spec(dev_spec)
    sys.modules["worldmonitor_osint_dev_start_ts_test"] = dev_mod
    dev_spec.loader.exec_module(dev_mod)
    repo = Path(__file__).resolve().parent / "_wm_dev_repo"
    repo.mkdir(exist_ok=True)
    (repo / "package.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(dev_mod, "_npm_exe", lambda: "npm")
    monkeypatch.setattr(dev_mod, "resolve_repo_path", lambda: repo)
    monkeypatch.setattr(dev_mod, "tailscale_ipv4", lambda: "100.91.183.75")
    monkeypatch.setattr(dev_mod, "tailscale_status", lambda: {"available": True, "ipv4": "100.91.183.75"})
    monkeypatch.setattr(dev_mod, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(dev_mod, "probe_dev_server", lambda _port=None: {"running": False})
    out = dev_mod.start_dev(dry_run=True, tailscale=True)
    assert out["success"] is True
    assert out["bind"] == "0.0.0.0"
    assert out["url"] == "http://100.91.183.75:3000"
    assert "--host" in out["command"]


def test_snapshot_auto_uses_free_without_paid():
    core = load_core()
    free = {"success": True, "tier": "free_web", "sections": {"news_digest": {}}}
    with patch.object(core.api, "connectivity_status", return_value={"api_key_configured": False, "local_sidecar": False}):
        with patch.object(core.free_web, "free_snapshot", return_value=free) as crawl:
            out = core.snapshot(country_code="US", region_id="europe", tier_mode="auto")
    crawl.assert_called_once()
    assert out["tier"] == "free_web"
