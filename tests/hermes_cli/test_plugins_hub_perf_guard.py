from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from hermes_cli import web_server
from hermes_cli import plugins_cmd
from tools import registry as tools_registry


_PLUGIN_ROW = [("demo", "1.0.0", "demo plugin", "user", "/tmp/demo-plugin", "demo")]


def _patch_minimal_hub_dependencies(monkeypatch, *, check_fn, discover_all_plugins=None):
    monkeypatch.setattr(web_server, "_get_dashboard_plugins", lambda force_rescan=False: [])
    monkeypatch.setattr(web_server, "_discover_memory_provider_statuses", lambda: [])
    monkeypatch.setattr(web_server, "get_hermes_home", lambda: Path("/tmp/hermes-home"))
    monkeypatch.setattr(web_server, "load_config", lambda: {"dashboard": {"hidden_plugins": []}})

    monkeypatch.setattr(
        plugins_cmd,
        "_discover_all_plugins",
        discover_all_plugins or (lambda: list(_PLUGIN_ROW)),
    )
    monkeypatch.setattr(plugins_cmd, "_get_current_context_engine", lambda: "compressor")
    monkeypatch.setattr(plugins_cmd, "_get_current_memory_provider", lambda: "")
    monkeypatch.setattr(plugins_cmd, "_discover_context_engines", lambda: [])
    monkeypatch.setattr(plugins_cmd, "_get_disabled_set", lambda: set())
    monkeypatch.setattr(plugins_cmd, "_get_enabled_set", lambda: {"demo"})
    monkeypatch.setattr(plugins_cmd, "_read_manifest", lambda _path: {"provides_tools": ["demo_tool"]})

    monkeypatch.setattr(
        tools_registry.registry,
        "get_entry",
        lambda _name: SimpleNamespace(check_fn=check_fn),
    )



def test_plugins_hub_does_not_probe_cold_check_fns(monkeypatch):
    tools_registry.invalidate_check_fn_cache()
    web_server._invalidate_plugins_hub_cache()

    calls = {"count": 0}

    def check_fn():
        calls["count"] += 1
        return False

    _patch_minimal_hub_dependencies(monkeypatch, check_fn=check_fn)

    payload = web_server._merged_plugins_hub(force_refresh=True)

    assert calls["count"] == 0
    assert payload["plugins"][0]["auth_required"] is False
    assert payload["plugins"][0]["auth_command"] == ""



def test_plugins_hub_uses_cached_failed_check_fn_verdict(monkeypatch):
    tools_registry.invalidate_check_fn_cache()
    web_server._invalidate_plugins_hub_cache()

    def check_fn():
        return False

    assert tools_registry._check_fn_cached(check_fn) is False
    _patch_minimal_hub_dependencies(monkeypatch, check_fn=check_fn)

    payload = web_server._merged_plugins_hub(force_refresh=True)

    assert payload["plugins"][0]["auth_required"] is True
    assert payload["plugins"][0]["auth_command"] == "hermes auth demo"



def test_plugins_hub_short_ttl_cache_collapses_duplicate_fetches(monkeypatch):
    tools_registry.invalidate_check_fn_cache()
    web_server._invalidate_plugins_hub_cache()

    calls = {"discover": 0}

    def discover_all_plugins():
        calls["discover"] += 1
        return list(_PLUGIN_ROW)

    _patch_minimal_hub_dependencies(
        monkeypatch,
        check_fn=lambda: True,
        discover_all_plugins=discover_all_plugins,
    )

    first = web_server._merged_plugins_hub(force_refresh=True)
    second = web_server._merged_plugins_hub()

    assert calls["discover"] == 1
    assert first is second
