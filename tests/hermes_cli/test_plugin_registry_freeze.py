"""Once the hosted boot freezes the registries, nothing a plugin kept from its load can change them.

A plugin may keep its ``ctx``, the registration handles ``ctx.register_*`` returned, or (for a
bundled platform) a deferred loader that runs on first lookup. Each of those reaches the plugin
registry without going through ``PluginManager._load_plugin`` / ``unload``, so each refuses on its
own. A local process never freezes, and the same calls keep working there.
"""

from __future__ import annotations

import asyncio

import pytest

import registry_freeze
from registry_freeze import RegistryFrozen, freeze_registries

from hermes_cli.plugins import PluginContext, PluginManager
from hermes_cli.plugins_manifest import PluginManifest


@pytest.fixture
def thawed(monkeypatch):
    """Start unfrozen; the freeze is one-way, so each test restores the module flag on teardown."""
    monkeypatch.setattr(registry_freeze, "_frozen", False)


@pytest.fixture
def manager(thawed):
    return PluginManager()


@pytest.fixture
def context(manager):
    return PluginContext(PluginManifest(name="probe", source="user"), manager)


def _callback(*_args, **_kwargs):
    return None


def _schema(name):
    return {"name": name, "description": "d", "parameters": {"type": "object", "properties": {}}}


def test_retained_context_cannot_register_middleware_after_freeze(manager, context):
    before = {kind: list(callbacks) for kind, callbacks in manager._middleware.items()}
    freeze_registries()
    with pytest.raises(RegistryFrozen, match="plugin registry"):
        context.register_middleware("llm_request", _callback)
    assert manager._middleware == before
    assert manager._registration_order == []


def _register_hook(ctx):
    ctx.register_hook("post_tool_call", _callback)


def _register_tool(ctx):
    ctx.register_tool("frozen_probe_tool", "frozen_probe", _schema("frozen_probe_tool"), _callback)


def _on_unload(ctx):
    ctx.on_unload(_callback)


def _subscribe(ctx):
    ctx.subscribe("other:event", _callback)


def _register_command(ctx):
    ctx.register_command("frozen-probe", _callback)


def _registry_state(manager):
    from tools.registry import registry
    return (
        {k: list(v) for k, v in manager._hooks.items()},
        {k: list(v) for k, v in manager._subscriptions.items()},
        dict(manager._plugin_commands),
        list(manager._registration_order),
        registry.get_entry("frozen_probe_tool", scope=manager.scope_key),
    )


@pytest.mark.parametrize("call", [_register_hook, _register_tool, _on_unload, _subscribe, _register_command])
def test_retained_context_entry_points_refuse_after_freeze(manager, context, call):
    before = _registry_state(manager)
    freeze_registries()
    with pytest.raises(RegistryFrozen, match="plugin registry"):
        call(context)
    assert _registry_state(manager) == before


def test_spawn_task_refuses_after_freeze_and_starts_nothing(manager, context):
    started = []

    async def work():
        started.append(True)

    async def scenario():
        coro = work()
        before = asyncio.all_tasks()
        freeze_registries()
        try:
            with pytest.raises(RegistryFrozen, match="plugin registry"):
                context.spawn_task(coro, name="frozen-probe")
            await asyncio.sleep(0)
            return asyncio.all_tasks() - before
        finally:
            coro.close()

    assert asyncio.run(scenario()) == set()
    assert started == []
    assert manager._registration_order == []


def test_handle_taken_before_freeze_cannot_dispose_after(manager, context):
    handle = context.register_hook("post_tool_call", _callback)
    freeze_registries()
    with pytest.raises(RegistryFrozen, match="plugin registry"):
        handle.dispose()
    assert handle.active
    assert manager._hooks["post_tool_call"] == [_callback]
    assert handle in manager._registration_order


def test_supervised_task_lease_still_releases_after_freeze(manager, context):
    """The lease only cancels the plugin's own task; its done-callback must not raise into the loop."""
    async def scenario():
        gate = asyncio.Event()

        async def work():
            await gate.wait()

        task = context.spawn_task(work(), name="frozen-probe")
        handle = next(r for r in manager._registration_order if r.kind == "background_task")
        freeze_registries()
        gate.set()
        await task
        await asyncio.sleep(0)  # the done-callback runs on the next loop iteration
        return handle

    handle = asyncio.run(scenario())
    assert not handle.active
    assert handle not in manager._registration_order


def _platform_manifest(tmp_path):
    plugin_dir = tmp_path / "frozenplat"
    plugin_dir.mkdir()
    (plugin_dir / "__init__.py").write_text("def register(ctx):\n    raise AssertionError('imported')\n")
    return PluginManifest(name="frozenplat-platform", kind="platform", source="bundled",
                          path=str(plugin_dir), key="frozenplat-platform")


@pytest.fixture
def no_frozenplat():
    from gateway.platform_registry import platform_registry
    yield
    platform_registry.unregister("frozenplat")


def test_deferred_platform_loader_refuses_after_freeze(manager, tmp_path, no_frozenplat, monkeypatch):
    """The loader leased at boot calls ``_load_plugin_scoped`` directly on first lookup. The freeze
    itself refuses while that loader is unresolved (see test_registry_freeze); a loader called
    after a freeze anyway (a handle retained past it) still publishes nothing."""
    from gateway.platform_registry import platform_registry
    manifest = _platform_manifest(tmp_path)
    manager._register_deferred_platform(manifest)
    loader = platform_registry.snapshot_registration("frozenplat", scope=manager.scope_key)[1]
    assert loader is not None
    before = list(manager._registration_order)
    with pytest.raises(registry_freeze.RegistriesIncomplete, match="frozenplat"):
        freeze_registries()
    monkeypatch.setattr(registry_freeze, "_frozen", True)  # past the completeness check
    with pytest.raises(RegistryFrozen, match="plugin registry"):
        loader()
    assert manager._registration_order == before
    assert manager._plugins["frozenplat-platform"].deferred


def test_registering_a_deferred_platform_refuses_after_freeze(manager, tmp_path, no_frozenplat):
    """A re-armed platform load publishes nothing into the platform registry once frozen."""
    from gateway.platform_registry import platform_registry
    freeze_registries()
    with pytest.raises(RegistryFrozen, match="plugin registry"):
        manager._register_deferred_platform(_platform_manifest(tmp_path))
    assert platform_registry.snapshot_registration("frozenplat", scope=manager.scope_key) == (None, None)
    assert "frozenplat-platform" not in manager._plugins


def test_unload_scoped_refuses_after_freeze(manager, context):
    handle = context.register_hook("post_tool_call", _callback)
    freeze_registries()
    with manager._discovery_lock, pytest.raises(RegistryFrozen, match="plugin registry"):
        manager._unload_scoped("probe")
    assert handle.active
    assert manager._hooks["post_tool_call"] == [_callback]


def test_fallback_hook_group_is_not_dropped_after_freeze(manager, context):
    """A memory provider re-loading at agent init must not strip the hooks every tenant shares."""
    source = ("probe", "/nowhere/probe/__init__.py")
    handle = manager._register_fallback_hook(context, source, "post_tool_call", _callback)
    freeze_registries()
    with pytest.raises(RegistryFrozen, match="plugin registry"):
        manager._drop_fallback_hooks(source)
    assert handle.active
    assert manager._memory_hook_registrations[source] == [handle]


def test_manager_hook_registration_refuses_after_freeze(manager, context):
    """``track_hook`` is public on the manager and writes the shared mapping without the ctx wrapper."""
    freeze_registries()
    with pytest.raises(RegistryFrozen, match="plugin registry"):
        manager.track_hook(context, "post_tool_call", _callback)
    assert "post_tool_call" not in manager._hooks
    assert manager._registration_order == []


def test_owned_fallback_hook_lease_refuses_after_freeze(manager, context):
    """When general discovery already owns the source, the memory loader leases an inert hook straight
    through the ledger, without any ctx entry point; the ledger itself refuses."""
    from types import SimpleNamespace
    from hermes_cli.plugins import LoadedPlugin
    manager._plugins["probe"] = LoadedPlugin(
        manifest=context.manifest, module=SimpleNamespace(__file__="/nowhere/probe/__init__.py"), enabled=True)
    source = ("probe", "/nowhere/probe/__init__.py")
    freeze_registries()
    with pytest.raises(RegistryFrozen, match="plugin registry"):
        manager._register_fallback_hook(context, source, "post_tool_call", _callback)
    assert manager._registration_order == []
    assert manager._ownership_ledger == {}


def test_manager_approval_transport_refuses_after_freeze(manager):
    freeze_registries()
    with pytest.raises(RegistryFrozen, match="plugin registry"):
        manager.register_approval_transport("frozen-probe", _callback, plugin_id="probe")
    assert "frozen-probe" not in manager._approval_transports


def test_unfrozen_context_registers_disposes_and_unloads(manager, context):
    """Local mode never freezes: the same calls keep working at runtime."""
    handle = context.register_middleware("llm_request", _callback)
    context.on_unload(_callback)
    context.subscribe("other:event", _callback)
    assert manager._middleware["llm_request"] == [_callback]
    handle.dispose()
    assert "llm_request" not in manager._middleware
    context.register_hook("post_tool_call", _callback)
    assert manager.unload("probe") is True
    assert "post_tool_call" not in manager._hooks


def test_registration_release_refuses_after_freeze(manager, context):
    """``release`` is the public name of a handle's inverse; it is the guarded ``dispose``."""
    handle = context.register_hook("post_tool_call", _callback)
    freeze_registries()
    with pytest.raises(RegistryFrozen, match="plugin registry"):
        handle.release()
    assert handle.active
    assert manager._hooks["post_tool_call"] == [_callback]


def test_registration_kind_and_inverse_cannot_be_rebound(manager, context):
    """``kind`` decides the background-task exemption; rebinding it (or the inverse) on a retained
    handle would let the plugin release a registry entry after the freeze."""
    handle = context.register_hook("post_tool_call", _callback)
    for name, value in (("kind", "background_task"), ("_release", lambda: None), ("key", "other")):
        with pytest.raises(AttributeError):
            setattr(handle, name, value)
    freeze_registries()
    with pytest.raises(RegistryFrozen):
        handle.dispose()
    assert manager._hooks["post_tool_call"] == [_callback]


def test_remove_plugin_skill_refuses_after_freeze(manager, context, tmp_path):
    path = tmp_path / "SKILL.md"
    path.write_text("skill")
    context.register_skill("probe", path)
    freeze_registries()
    with pytest.raises(RegistryFrozen, match="plugin registry"):
        manager.remove_plugin_skill("probe:probe")
    assert manager.find_plugin_skill("probe:probe") == path


def test_stale_plugin_skill_read_reports_missing_without_changing_the_registry(manager, context, tmp_path,
                                                                                monkeypatch):
    """A skill file deleted out of band reads as missing after the freeze; the frozen entry stays."""
    import json
    import hermes_cli.plugins as plugins_mod
    from tools.skills_tool import skill_view
    path = tmp_path / "SKILL.md"
    path.write_text("---\nname: probe\ndescription: d\n---\nbody\n")
    context.register_skill("probe", path)
    monkeypatch.setattr(plugins_mod, "get_plugin_manager", lambda: manager)
    monkeypatch.setattr(plugins_mod, "discover_plugins", lambda *a, **k: None)
    freeze_registries()
    path.unlink()
    result = json.loads(skill_view("probe:probe"))
    assert result["success"] is False and "no longer exists" in result["error"]
    assert manager.find_plugin_skill("probe:probe") == path


def test_memory_skill_prune_is_skipped_after_freeze(manager, context, tmp_path, monkeypatch):
    """Pruning an inactive memory provider's skills is a plugin-registry change: after the freeze it
    leaves the registry (and its own bookkeeping) as they are instead of raising into the read."""
    import hermes_cli.plugins as plugins_mod
    from plugins import memory
    path = tmp_path / "SKILL.md"
    path.write_text("skill")
    context.register_skill("probe", path)
    monkeypatch.setattr(plugins_mod, "get_plugin_manager", lambda: manager)
    tracked = {"probe:probe": path}
    monkeypatch.setattr(memory, "_registered_skills_for_active_home", lambda: tracked)
    freeze_registries()
    memory._prune_inactive_memory_provider_skills("other-provider")
    assert manager.find_plugin_skill("probe:probe") == path
    assert tracked == {"probe:probe": path}
