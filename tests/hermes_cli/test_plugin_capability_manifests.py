from __future__ import annotations

import pytest

from agent.session_capabilities import build_capability_plan
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


def _context(manager: PluginManager, name: str = "router") -> PluginContext:
    manifest = PluginManifest(name=name, version="1.0.0", description="test")
    return PluginContext(manifest, manager)


def test_registration_is_profile_local_and_disposable():
    manager = PluginManager()
    ctx = _context(manager)
    handle = ctx.register_capability_manifest(
        name="research",
        description="Current web research",
        toolsets=["web"],
        tools=["extra_search"],
        routing_keywords=["research", "latest"],
    )
    assert manager._capability_manifests["research"]["plugin"] == "router"
    handle.dispose()
    assert manager._capability_manifests == {}


def test_duplicate_manifest_name_is_rejected():
    manager = PluginManager()
    _context(manager, "first").register_capability_manifest(
        name="research", description="one"
    )
    with pytest.raises(ValueError, match="already registered"):
        _context(manager, "second").register_capability_manifest(
            name="research", description="two"
        )


def test_invalid_manifest_is_rejected():
    manager = PluginManager()
    ctx = _context(manager)
    with pytest.raises(ValueError, match="must match"):
        ctx.register_capability_manifest(name="Bad Name", description="bad")
    with pytest.raises(ValueError, match="description is required"):
        ctx.register_capability_manifest(name="valid", description="")


def test_plugin_toolsets_resolve_host_side(monkeypatch):
    import hermes_cli.plugins as plugins
    from toolsets import get_capability_manifests, resolve_toolset

    monkeypatch.setattr(
        plugins,
        "get_registered_capability_manifests",
        lambda: {
            "plugin-research": {
                "name": "plugin-research",
                "description": "Research with plugin metadata",
                "toolsets": ("web",),
                "tools": ("custom_search",),
                "routing_keywords": ("research",),
                "routing_examples": (),
                "routing_priority": 0,
            }
        },
    )
    manifest = get_capability_manifests()["plugin-research"]
    assert "custom_search" in manifest["tools"]
    assert set(resolve_toolset("web")).issubset(set(manifest["tools"]))


def test_plugin_manifest_cannot_grant_unauthorized_tool(monkeypatch):
    import hermes_cli.plugins as plugins
    from toolsets import get_capability_manifests

    monkeypatch.setattr(
        plugins,
        "get_registered_capability_manifests",
        lambda: {
            "danger": {
                "name": "danger",
                "description": "Delete production records",
                "toolsets": (),
                "tools": ("delete_production",),
                "routing_keywords": ("delete production",),
                "routing_examples": (),
                "routing_priority": 100,
            }
        },
    )
    authorized = [{
        "type": "function",
        "function": {
            "name": "clarify",
            "description": "Ask a question",
            "parameters": {"type": "object", "properties": {}},
        },
    }]
    plan = build_capability_plan(
        "delete production", tool_defs=authorized,
        manifests=get_capability_manifests(), direct_token_budget=2_000,
    )
    assert "delete_production" not in plan.direct_tools
    assert "delete_production" not in plan.deferred_tools
