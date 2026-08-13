from __future__ import annotations

import threading
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

    calls = {"count": 0, "threads": set()}

    def check_fn():
        calls["count"] += 1
        calls["threads"].add(threading.current_thread())
        return False

    _patch_minimal_hub_dependencies(monkeypatch, check_fn=check_fn)

    payload = web_server._merged_plugins_hub(force_refresh=True)

    # The request path itself must never execute the probe: the cold verdict
    # is unknown, so the payload reports no auth requirement. Any probing
    # happens on a background warmer thread, never inline.
    assert payload["plugins"][0]["auth_required"] is False
    assert payload["plugins"][0]["auth_command"] == ""
    assert threading.current_thread() not in calls["threads"]


def test_plugins_hub_cold_cache_schedules_background_probe(monkeypatch):
    tools_registry.invalidate_check_fn_cache()
    web_server._invalidate_plugins_hub_cache()

    probe_ran = threading.Event()

    def check_fn():
        probe_ran.set()
        return False

    _patch_minimal_hub_dependencies(monkeypatch, check_fn=check_fn)

    scheduled: list = []
    real_schedule = web_server._schedule_check_fn_probe

    def tracking_schedule(fn):
        thread = real_schedule(fn)
        scheduled.append(thread)
        return thread

    monkeypatch.setattr(web_server, "_schedule_check_fn_probe", tracking_schedule)

    # Cold cache → the fetch schedules a background probe and reports the
    # verdict as unknown (auth_required stays False for now).
    payload = web_server._merged_plugins_hub(force_refresh=True)
    assert payload["plugins"][0]["auth_required"] is False
    assert scheduled and scheduled[0] is not None

    scheduled[0].join(timeout=5)
    assert probe_ran.wait(timeout=5)

    # Once the TTL cache refreshes, the probed False verdict surfaces as an
    # auth requirement.
    refreshed = web_server._merged_plugins_hub(force_refresh=True)
    assert refreshed["plugins"][0]["auth_required"] is True
    assert refreshed["plugins"][0]["auth_command"] == "hermes auth demo"



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


def test_plugin_install_endpoint_invalidates_hub_cache(monkeypatch):
    import asyncio

    from hermes_cli.web_models import _AgentPluginInstallBody

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

    # Prime the TTL cache; a plain fetch must be served from it.
    web_server._merged_plugins_hub(force_refresh=True)
    web_server._merged_plugins_hub()
    assert calls["discover"] == 1

    # Simulate a successful install through the endpoint; its invalidation
    # hook must drop the memoized payload so the next fetch rebuilds.
    monkeypatch.setattr(web_server, "_require_token", lambda _request: None)
    monkeypatch.setattr(
        plugins_cmd, "dashboard_install_plugin", lambda *a, **k: {"ok": True}
    )

    asyncio.run(
        web_server.post_agent_plugin_install(
            object(), _AgentPluginInstallBody(identifier="demo")
        )
    )

    web_server._merged_plugins_hub()
    assert calls["discover"] == 2
