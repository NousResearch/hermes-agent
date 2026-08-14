"""End-to-end coverage for plugin registration ownership and reload cleanup."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from threading import Event
from time import monotonic, sleep
from types import MethodType

import yaml


def _write_plugin(hermes_home: Path) -> None:
    plugin_dir = hermes_home / "plugins" / "ledger_probe"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "ledger_probe",
                "version": "0.1.0",
                "description": "ownership ledger probe",
            }
        )
    )
    (plugin_dir / "SKILL.md").write_text("# Ledger probe\n")
    (plugin_dir / "__init__.py").write_text(
        "from pathlib import Path\n"
        "\n"
        "def _hook(**kwargs):\n"
        "    return {'hook': 'ledger'}\n"
        "\n"
        "def _middleware(**kwargs):\n"
        "    return {'middleware': 'ledger'}\n"
        "\n"
        "UNLOADED = []\n"
        "\n"
        "def register(ctx):\n"
        "    ctx.register_tool(\n"
        "        name='ledger_probe_tool',\n"
        "        toolset='plugin_ledger_probe',\n"
        "        schema={'name': 'ledger_probe_tool', 'parameters': {'type': 'object', 'properties': {}}},\n"
        "        handler=lambda args, **kwargs: 'ledger',\n"
        "    )\n"
        "    ctx.register_platform(\n"
        "        name='ledger_probe_platform',\n"
        "        label='Ledger probe',\n"
        "        adapter_factory=lambda config: object(),\n"
        "        check_fn=lambda: True,\n"
        "    )\n"
        "    ctx.register_cli_command(\n"
        "        'ledger-probe-cli', 'Ledger CLI', lambda parser: None,\n"
        "        handler_fn=lambda args: None,\n"
        "    )\n"
        "    ctx.register_command(\n"
        "        'ledger-probe-command', lambda args: args,\n"
        "        description='Ledger command',\n"
        "    )\n"
        "    ctx.register_hook('pre_tool_call', _hook)\n"
        "    ctx.register_middleware('tool_request', _middleware)\n"
        "    ctx.register_auxiliary_task(\n"
        "        key='ledger_probe_task',\n"
        "        display_name='Ledger probe task',\n"
        "        description='Ledger task',\n"
        "    )\n"
        "    ctx.register_skill(\n"
        "        'ledger-probe', Path(__file__).with_name('SKILL.md'),\n"
        "        'Ledger skill',\n"
        "    )\n"
        "    ctx.register_system_prompt_section(\n"
        "        'ledger-probe-section', 'ledger probe section content',\n"
        "    )\n"
        "    ctx.register_approval_transport(\n"
        "        'ledger_probe_transport', lambda request: 'deny',\n"
        "    )\n"
        "    ctx.on_unload(lambda: UNLOADED.append('ledger_probe'))\n"
    )
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": ["ledger_probe"]}})
    )


def _write_profile_probe(hermes_home: Path, marker: str) -> None:
    plugin_dir = hermes_home / "plugins" / "profile_probe"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "profile_probe",
                "version": "0.1.0",
                "description": f"profile probe {marker}",
            }
        )
    )
    (plugin_dir / "__init__.py").write_text(
        "def register(ctx):\n"
        "    ctx.register_tool(\n"
        "        name='shared_profile_tool',\n"
        "        toolset='profile_probe',\n"
        "        schema={'name': 'shared_profile_tool', 'parameters': {'type': 'object', 'properties': {}}},\n"
        f"        handler=lambda args, **kwargs: {marker!r},\n"
        "    )\n"
        "    ctx.register_platform(\n"
        "        name='shared_profile_platform',\n"
        f"        label={marker!r},\n"
        f"        adapter_factory=lambda config: {marker!r},\n"
        "        check_fn=lambda: True,\n"
        "    )\n"
    )
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": ["profile_probe"]}})
    )


def test_load_force_reload_and_unload_remove_every_manager_registration(
    tmp_path,
    monkeypatch,
):
    """A real temporary plugin has one live registration after each reload."""
    import hermes_cli.plugins as plugins_mod
    from gateway.platform_registry import platform_registry
    from hermes_cli.plugins import PluginManager
    from tools.registry import registry

    hermes_home = tmp_path / "hermes"
    _write_plugin(hermes_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(
        plugins_mod,
        "get_bundled_plugins_dir",
        lambda: tmp_path / "empty-bundled",
    )
    monkeypatch.setattr(PluginManager, "_scan_entry_points", lambda self: [])

    manager = PluginManager()
    manager.discover_and_load()

    first_tool = registry.get_entry("ledger_probe_tool")
    first_platform = platform_registry.get("ledger_probe_platform")
    first_hook = manager._hooks["pre_tool_call"][0]
    first_middleware = manager._middleware["tool_request"][0]
    first_command = manager._plugin_commands["ledger-probe-command"]
    first_cli_command = manager._cli_commands["ledger-probe-cli"]
    first_skill = manager._plugin_skills["ledger_probe:ledger-probe"]
    module_name = manager._plugins["ledger_probe"].module.__name__

    assert first_tool is not None
    assert first_platform is not None
    assert set(registration.kind for registration in manager._ownership_ledger["ledger_probe"]) == {
        "tool",
        "platform",
        "cli_command",
        "command",
        "hook",
        "middleware",
        "auxiliary_task",
        "skill",
        "tool_override_policy",
        "system_prompt_section",
        "approval_transport",
        "on_unload",
    }
    assert "ledger-probe-section" in manager._system_prompt_sections
    assert "ledger_probe_transport" in manager._approval_transports

    manager.discover_and_load(force=True)

    second_tool = registry.get_entry("ledger_probe_tool")
    second_platform = platform_registry.get("ledger_probe_platform")
    assert second_tool is not None and second_tool is not first_tool
    assert second_platform is not None and second_platform is not first_platform
    assert second_tool.handler is not first_tool.handler
    assert first_hook not in manager._hooks["pre_tool_call"]
    assert first_middleware not in manager._middleware["tool_request"]
    assert len(manager._hooks["pre_tool_call"]) == 1
    assert len(manager._middleware["tool_request"]) == 1
    assert manager._plugin_commands["ledger-probe-command"] is not first_command
    assert manager._cli_commands["ledger-probe-cli"] is not first_cli_command
    assert manager._plugin_skills["ledger_probe:ledger-probe"] is not first_skill
    assert len(manager._aux_tasks) == 1
    assert [
        entry
        for entry in platform_registry.plugin_entries()
        if entry.name == "ledger_probe_platform"
    ] == [second_platform]

    assert manager.unload("ledger_probe") is True
    assert registry.get_entry("ledger_probe_tool") is None
    assert not platform_registry.is_registered("ledger_probe_platform")
    assert "pre_tool_call" not in manager._hooks
    assert "tool_request" not in manager._middleware
    assert "ledger-probe-command" not in manager._plugin_commands
    assert "ledger-probe-cli" not in manager._cli_commands
    assert "ledger_probe:ledger-probe" not in manager._plugin_skills
    assert "ledger-probe-section" not in manager._system_prompt_sections
    assert "ledger_probe_transport" not in manager._approval_transports
    assert manager._plugins == {} or "ledger_probe" not in manager._plugins
    reloaded_module_name = None
    import sys as _sys
    for _name, _mod in list(_sys.modules.items()):
        if getattr(_mod, "UNLOADED", None) and "ledger_probe" in _mod.UNLOADED:
            reloaded_module_name = _name
            break
    assert reloaded_module_name is not None, "on_unload callback never fired"
    assert manager._aux_tasks == {}
    assert manager._ownership_ledger == {}
    assert registry.snapshot_plugin_override_policy(
        module_name, scope=manager.scope_key
    ) is None
    assert registry.plugin_scope_for_module(module_name) == manager.scope_key


def test_reverse_unload_restores_an_overridden_platform_registration():
    """Reverse teardown reveals an older entry before removing it."""
    from gateway.platform_registry import platform_registry
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    name = "ledger_override_platform"
    scope = platform_registry.current_scope_key()
    previous = platform_registry.snapshot_registration(name, scope=scope)
    manager_a = PluginManager(scope_key=scope)
    manager_b = PluginManager(scope_key=scope)
    context_a = PluginContext(
        PluginManifest(name="ledger_owner_a", key="ledger_owner_a"), manager_a
    )
    context_b = PluginContext(
        PluginManifest(name="ledger_owner_b", key="ledger_owner_b"), manager_b
    )

    try:
        handle_a = context_a.register_platform(
            name=name,
            label="Ledger A",
            adapter_factory=lambda config: "a",
            check_fn=lambda: True,
        )
        entry_a = platform_registry.get(name)
        handle_b = context_b.register_platform(
            name=name,
            label="Ledger B",
            adapter_factory=lambda config: "b",
            check_fn=lambda: True,
        )
        entry_b = platform_registry.get(name)

        assert handle_a is not None and handle_b is not None
        assert entry_a is not None and entry_b is not None
        assert entry_a is not entry_b

        handle_b.dispose()
        assert platform_registry.get(name) is entry_a
        handle_a.dispose()
        assert platform_registry.snapshot_registration(name, scope=scope) == previous
    finally:
        # The test uses a deliberately unique name, but restore any state that
        # a surrounding test may have installed under it.
        current = platform_registry.snapshot_registration(name, scope=scope)
        platform_registry.restore_registration(
            name, current, previous, scope=scope
        )


def test_targeted_unload_does_not_resurrect_an_older_tool_override():
    """The tool overlay follows the same arbitrary-order ownership contract."""
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    from tools.registry import registry

    name = "ledger_out_of_order_tool"
    scope = registry.current_scope_key()
    previous = registry.snapshot_registration(name, scope=scope)
    manager_a = PluginManager(scope_key=scope)
    manager_b = PluginManager(scope_key=scope)
    context_a = PluginContext(PluginManifest(name="tool_a", key="tool_a"), manager_a)
    context_b = PluginContext(PluginManifest(name="tool_b", key="tool_b"), manager_b)

    def register(context, marker):
        return context.register_tool(
            name=name,
            toolset="ledger_test",
            schema={
                "name": name,
                "parameters": {"type": "object", "properties": {}},
            },
            handler=lambda args, **kwargs: marker,
        )

    try:
        assert register(context_a, "a") is not None
        old_entry = registry.get_entry(name, scope=scope)
        assert register(context_b, "b") is not None
        new_entry = registry.get_entry(name, scope=scope)

        assert old_entry is not None and new_entry is not old_entry
        manager_a.unload("tool_a")
        assert registry.get_entry(name, scope=scope) is new_entry
        assert name not in manager_a._plugin_tool_names
        manager_b.unload("tool_b")
        assert registry.get_entry(name, scope=scope) is not old_entry
        assert registry.snapshot_registration(name, scope=scope) is previous
    finally:
        current = registry.snapshot_registration(name, scope=scope)
        if current is not None:
            registry.restore_registration(name, current, previous, scope=scope)


def test_rejected_tool_registration_does_not_claim_global_fallback():
    """Effective fallback identity cannot masquerade as a successful write."""
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    from tools.registry import registry

    name = "ledger_rejected_tool"
    previous = registry.snapshot_registration(name)

    def shared_handler(args, **kwargs):
        return "base"

    registry.register(
        name=name,
        toolset="ledger_base",
        schema={"name": name, "parameters": {"type": "object", "properties": {}}},
        handler=shared_handler,
    )
    base_entry = registry.snapshot_registration(name)
    manager = PluginManager()
    context = PluginContext(
        PluginManifest(name="rejected_owner", key="rejected_owner"), manager
    )
    try:
        handle = context.register_tool(
            name=name,
            toolset="different_toolset",
            schema={
                "name": name,
                "parameters": {"type": "object", "properties": {}},
            },
            handler=shared_handler,
        )
        assert handle is None
        assert name not in manager._plugin_tool_names
        assert manager._ownership_ledger == {}
        assert registry.snapshot_registration(name) is base_entry
    finally:
        if base_entry is not None:
            registry.restore_registration(name, base_entry, previous)


def test_plugin_context_cannot_shadow_same_toolset_global_with_core_callable():
    """Explicit context scope cannot launder an imported/core handler."""
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    from tools.registry import registry

    name = "ledger_same_toolset_global"
    previous = registry.snapshot_registration(name)
    registry.register(
        name=name,
        toolset="ledger_same_toolset",
        schema={"name": name, "parameters": {"type": "object", "properties": {}}},
        handler=lambda args, **kwargs: "base",
    )
    base_entry = registry.snapshot_registration(name)
    manager = PluginManager()
    context = PluginContext(
        PluginManifest(name="core_callable", key="core_callable"), manager
    )
    try:
        assert context.register_tool(
            name=name,
            toolset="ledger_same_toolset",
            schema={
                "name": name,
                "parameters": {"type": "object", "properties": {}},
            },
            handler=str,
        ) is None
        assert registry.get_entry(name, scope=manager.scope_key) is base_entry
        assert registry.snapshot_registration(name, scope=manager.scope_key) is None
    finally:
        if base_entry is not None:
            registry.restore_registration(name, base_entry, previous)


def test_rejected_tool_registration_does_not_claim_local_predecessor():
    """A same-handler rejection cannot manufacture a replacement lease."""
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    from tools.registry import registry

    name = "ledger_rejected_local_tool"
    scope = registry.current_scope_key()
    previous = registry.snapshot_registration(name, scope=scope)

    def shared_handler(args, **kwargs):
        return "shared"

    manager_a = PluginManager(scope_key=scope)
    manager_b = PluginManager(scope_key=scope)
    context_a = PluginContext(PluginManifest(name="local_owner", key="local_owner"), manager_a)
    context_b = PluginContext(PluginManifest(name="false_owner", key="false_owner"), manager_b)
    try:
        assert context_a.register_tool(
            name=name,
            toolset="owner_toolset",
            schema={"name": name, "parameters": {"type": "object", "properties": {}}},
            handler=shared_handler,
        ) is not None
        owner_entry = registry.snapshot_registration(name, scope=scope)
        assert context_b.register_tool(
            name=name,
            toolset="different_toolset",
            schema={"name": name, "parameters": {"type": "object", "properties": {}}},
            handler=shared_handler,
        ) is None
        assert manager_b._ownership_ledger == {}
        manager_a.unload("local_owner")
        assert registry.snapshot_registration(name, scope=scope) is previous
        assert owner_entry is not None
    finally:
        current = registry.snapshot_registration(name, scope=scope)
        if current is not None:
            registry.restore_registration(name, current, previous, scope=scope)


def test_scoped_plugin_cannot_deregister_a_process_global_tool():
    """Profile-local plugin cleanup must never mutate the shared base layer."""
    from unittest.mock import patch

    import pytest

    from tools.registry import ToolRegistry

    registry = ToolRegistry()
    name = "ledger_global_base"
    registry.register(
        name=name,
        toolset="ledger_base",
        schema={"name": name, "parameters": {"type": "object", "properties": {}}},
        handler=lambda args, **kwargs: "base",
    )
    module_name = "hermes_plugins.scoped_cleanup"
    registry.register_plugin_override_policy(
        module_name, True, scope="/profiles/isolated"
    )

    with patch.object(ToolRegistry, "_caller_module", return_value=module_name):
        with pytest.raises(PermissionError, match="process-global"):
            registry.deregister(name)

    assert registry.snapshot_registration(name) is not None


def test_shared_entrypoint_module_uses_the_active_profile_scope(tmp_path):
    """One pip module can serve A and B without becoming process-global."""
    import pytest

    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools.registry import ToolRegistry

    registry = ToolRegistry()
    module_name = "third_party.shared_hermes_plugin"
    home_a = str((tmp_path / "entrypoint-a").resolve())
    home_b = str((tmp_path / "entrypoint-b").resolve())
    home_c = str((tmp_path / "entrypoint-c").resolve())
    policy_a = registry.register_plugin_override_policy(
        module_name, False, scope=home_a
    )
    policy_b = registry.register_plugin_override_policy(
        module_name, False, scope=home_b
    )
    handler = eval("lambda args, **kwargs: 'shared'", {"__name__": module_name})

    def register_in(home, name):
        token = set_hermes_home_override(home)
        try:
            registry.register(
                name=name,
                toolset="entrypoint_shared",
                schema={
                    "name": name,
                    "parameters": {"type": "object", "properties": {}},
                },
                handler=handler,
            )
        finally:
            reset_hermes_home_override(token)

    register_in(home_a, "shared_entrypoint_a")
    register_in(home_b, "shared_entrypoint_b")
    assert registry.snapshot_registration("shared_entrypoint_a", scope=home_a) is not None
    assert registry.snapshot_registration("shared_entrypoint_a", scope=home_b) is None
    assert registry.snapshot_registration("shared_entrypoint_b", scope=home_b) is not None
    assert registry.snapshot_registration("shared_entrypoint_b") is None

    from unittest.mock import patch

    token = set_hermes_home_override(home_a)
    try:
        with patch.object(ToolRegistry, "_caller_module", return_value=module_name):
            registry.deregister("shared_entrypoint_a")
    finally:
        reset_hermes_home_override(token)
    assert registry.snapshot_registration("shared_entrypoint_a", scope=home_a) is None
    assert registry.snapshot_registration("shared_entrypoint_b", scope=home_b) is not None

    # Policy unload revokes authorization but durable scope attribution keeps
    # a stale delayed callback out of the process-global registry.
    registry.restore_plugin_override_policy(
        module_name, policy_a, None, scope=home_a
    )
    registry.restore_plugin_override_policy(
        module_name, policy_b, None, scope=home_b
    )
    register_in(home_a, "shared_entrypoint_stale")
    assert registry.snapshot_registration("shared_entrypoint_stale", scope=home_a)
    assert registry.snapshot_registration("shared_entrypoint_stale") is None
    with pytest.raises(PermissionError, match="multiple profiles"):
        register_in(home_c, "shared_entrypoint_ambiguous")
    assert registry.snapshot_registration("shared_entrypoint_ambiguous") is None


def test_decorated_plugin_callable_keeps_its_defining_module_scope(tmp_path):
    """functools.wraps must not replace the plugin wrapper's provenance."""
    import functools

    from tools.registry import ToolRegistry

    registry = ToolRegistry()
    module_name = "third_party.decorated_plugin"
    scope = str((tmp_path / "decorated-profile").resolve())
    registry.register_plugin_override_policy(module_name, False, scope=scope)
    namespace = {"__name__": module_name, "functools": functools}
    exec(
        "@functools.wraps(str)\n"
        "def handler(args, **kwargs):\n"
        "    return 'decorated'\n",
        namespace,
    )
    handler = namespace["handler"]
    registry.register(
        name="decorated_plugin_tool",
        toolset="decorated_plugin",
        schema={
            "name": "decorated_plugin_tool",
            "parameters": {"type": "object", "properties": {}},
        },
        handler=handler,
    )

    assert registry.snapshot_registration(
        "decorated_plugin_tool", scope=scope
    ) is not None
    assert registry.snapshot_registration("decorated_plugin_tool") is None


def test_entrypoint_policy_uses_the_most_specific_module_prefix(tmp_path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools.registry import ToolRegistry

    registry = ToolRegistry()
    broad_scope = str((tmp_path / "broad").resolve())
    narrow_scope = str((tmp_path / "narrow").resolve())
    registry.register_plugin_override_policy("vendor", True, scope=broad_scope)
    registry.register_plugin_override_policy(
        "vendor.plugin", False, scope=narrow_scope
    )
    handler = eval(
        "lambda args, **kwargs: 'narrow'",
        {"__name__": "vendor.plugin.handlers"},
    )
    token = set_hermes_home_override(narrow_scope)
    try:
        registry.register(
            name="specific_entrypoint_tool",
            toolset="specific_entrypoint",
            schema={
                "name": "specific_entrypoint_tool",
                "parameters": {"type": "object", "properties": {}},
            },
            handler=handler,
        )
    finally:
        reset_hermes_home_override(token)

    assert registry.snapshot_registration(
        "specific_entrypoint_tool", scope=narrow_scope
    ) is not None
    assert registry.snapshot_registration(
        "specific_entrypoint_tool", scope=broad_scope
    ) is None


def test_targeted_unload_does_not_resurrect_an_older_override():
    """Removing A under B tombstones A so B cannot restore it later."""
    from gateway.platform_registry import platform_registry
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    name = "ledger_out_of_order_platform"
    scope = platform_registry.current_scope_key()
    previous = platform_registry.snapshot_registration(name, scope=scope)
    manager_a = PluginManager(scope_key=scope)
    manager_b = PluginManager(scope_key=scope)
    context_a = PluginContext(
        PluginManifest(name="ledger_old", key="ledger_old"), manager_a
    )
    context_b = PluginContext(
        PluginManifest(name="ledger_new", key="ledger_new"), manager_b
    )

    try:
        context_a.register_platform(
            name=name,
            label="old",
            adapter_factory=lambda config: "old",
            check_fn=lambda: True,
        )
        old_entry = platform_registry.get(name)
        context_b.register_platform(
            name=name,
            label="new",
            adapter_factory=lambda config: "new",
            check_fn=lambda: True,
        )
        new_entry = platform_registry.get(name)

        assert old_entry is not None and new_entry is not None
        assert manager_a.unload("ledger_old") is True
        assert platform_registry.get(name) is new_entry
        assert name not in manager_a._plugin_platform_names
        assert manager_b.unload("ledger_new") is True
        assert platform_registry.get(name) is not old_entry
        assert platform_registry.snapshot_registration(name, scope=scope) == previous
    finally:
        current = platform_registry.snapshot_registration(name, scope=scope)
        platform_registry.restore_registration(
            name, current, previous, scope=scope
        )


def test_manager_local_override_does_not_resurrect_after_targeted_unload():
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    manager = PluginManager()
    context_a = PluginContext(PluginManifest(name="local_a", key="local_a"), manager)
    context_b = PluginContext(PluginManifest(name="local_b", key="local_b"), manager)

    context_a.register_cli_command("shared-local", "A", lambda parser: None)
    context_b.register_cli_command("shared-local", "B", lambda parser: None)

    assert manager.unload("local_a") is True
    assert manager._cli_commands["shared-local"]["plugin"] == "local_b"
    assert manager.unload("local_b") is True
    assert "shared-local" not in manager._cli_commands


def test_provider_overlay_switches_profiles_and_reveals_fresh_global_fallback(
    tmp_path,
    monkeypatch,
):
    """Provider consumers see A→B→A, and unload never pins a stale base."""
    from agent.image_gen_provider import ImageGenProvider
    import agent.image_gen_registry as image_registry
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    class Provider(ImageGenProvider):
        def __init__(self, marker):
            self.marker = marker

        @property
        def name(self):
            return "ledger_profile_provider"

        def generate(self, prompt, aspect_ratio="landscape", **kwargs):
            return {"marker": self.marker}

    name = "ledger_profile_provider"
    baseline = image_registry.snapshot_registration(name)
    global_a = Provider("global-a")
    global_b = Provider("global-b")
    provider_a = Provider("profile-a")
    provider_b = Provider("profile-b")
    home_a = str((tmp_path / "provider-a").resolve())
    home_b = str((tmp_path / "provider-b").resolve())
    manager_a = PluginManager(scope_key=home_a)
    manager_b = PluginManager(scope_key=home_b)
    context_a = PluginContext(PluginManifest(name="provider_a", key="provider_a"), manager_a)
    context_b = PluginContext(PluginManifest(name="provider_b", key="provider_b"), manager_b)
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"image_gen": {"provider": name}},
    )

    def active_for(home):
        token = set_hermes_home_override(home)
        try:
            return image_registry.get_active_provider()
        finally:
            reset_hermes_home_override(token)

    image_registry.register_provider(global_a)
    try:
        assert context_a.register_image_gen_provider(provider_a) is not None
        assert context_b.register_image_gen_provider(provider_b) is not None
        assert active_for(home_a) is provider_a
        assert active_for(home_b) is provider_b
        assert active_for(home_a) is provider_a

        assert manager_a.unload("provider_a") is True
        assert active_for(home_a) is global_a
        image_registry.register_provider(global_b)
        assert active_for(home_a) is global_b
        assert active_for(home_b) is provider_b

        assert manager_b.unload("provider_b") is True
        assert active_for(home_b) is global_b
    finally:
        manager_a.unload("provider_a")
        manager_b.unload("provider_b")
        current = image_registry.snapshot_registration(name)
        if current is not None:
            image_registry.restore_registration(name, current, baseline)


def test_reused_provider_singleton_keeps_registration_generations_distinct():
    """Reusing one object after unload must not revive its retired generation."""
    from agent.image_gen_provider import ImageGenProvider
    from agent.image_gen_registry import (
        get_provider,
        restore_registration,
        snapshot_registration,
    )
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    class ProbeProvider(ImageGenProvider):
        def __init__(self, marker):
            self.marker = marker

        @property
        def name(self):
            return "ledger_generation_provider"

        def generate(self, prompt, aspect_ratio="landscape", **kwargs):
            return {"marker": self.marker}

    managers = [PluginManager() for _ in range(4)]
    scope = managers[0].scope_key
    baseline = get_provider("ledger_generation_provider", scope=scope)
    baseline_local = snapshot_registration("ledger_generation_provider", scope=scope)
    provider_a = ProbeProvider("a")
    provider_b = ProbeProvider("b")
    provider_c = ProbeProvider("c")
    contexts = [
        PluginContext(
            PluginManifest(name=f"provider_{index}", key=f"provider_{index}"),
            manager,
        )
        for index, manager in enumerate(managers)
    ]

    try:
        contexts[0].register_image_gen_provider(provider_a)
        contexts[1].register_image_gen_provider(provider_b)
        managers[0].unload("provider_0")
        assert get_provider(provider_a.name, scope=scope) is provider_b

        # A fresh ownership generation deliberately reuses the same singleton.
        contexts[2].register_image_gen_provider(provider_a)
        contexts[3].register_image_gen_provider(provider_c)
        managers[3].unload("provider_3")
        assert get_provider(provider_a.name, scope=scope) is provider_a
        managers[2].unload("provider_2")
        assert get_provider(provider_a.name, scope=scope) is provider_b
        managers[1].unload("provider_1")
        assert get_provider(provider_a.name, scope=scope) is baseline
    finally:
        current = snapshot_registration("ledger_generation_provider", scope=scope)
        if current is not baseline_local and current is not None:
            restore_registration(
                "ledger_generation_provider", current, baseline_local, scope=scope
            )


def test_same_provider_singleton_can_have_two_live_owners():
    """Retiring an older identical lease must not remove the newer owner."""
    from agent.image_gen_provider import ImageGenProvider
    from agent.image_gen_registry import (
        get_provider,
        restore_registration,
        snapshot_registration,
    )
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    class SharedProvider(ImageGenProvider):
        @property
        def name(self):
            return "ledger_shared_singleton"

        def generate(self, prompt, aspect_ratio="landscape", **kwargs):
            return {"prompt": prompt}

    provider = SharedProvider()
    manager_a = PluginManager()
    manager_b = PluginManager()
    scope = manager_a.scope_key
    baseline = get_provider("ledger_shared_singleton", scope=scope)
    baseline_local = snapshot_registration("ledger_shared_singleton", scope=scope)
    context_a = PluginContext(PluginManifest(name="shared_a", key="shared_a"), manager_a)
    context_b = PluginContext(PluginManifest(name="shared_b", key="shared_b"), manager_b)

    try:
        assert context_a.register_image_gen_provider(provider) is not None
        assert context_b.register_image_gen_provider(provider) is not None
        manager_a.unload("shared_a")
        assert get_provider(provider.name, scope=scope) is provider
        manager_b.unload("shared_b")
        assert get_provider(provider.name, scope=scope) is baseline
    finally:
        current = snapshot_registration(provider.name, scope=scope)
        if current is not baseline_local and current is not None:
            restore_registration(
                provider.name, current, baseline_local, scope=scope
            )


def test_provider_cleanup_uses_the_captured_normalized_name():
    """Whitespace and later name mutation cannot orphan a provider entry."""
    from agent.image_gen_provider import ImageGenProvider
    from agent.image_gen_registry import (
        get_provider,
        restore_registration,
        snapshot_registration,
    )
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    class MutableProvider(ImageGenProvider):
        def __init__(self):
            self._name = "  ledger_mutable_provider  "

        @property
        def name(self):
            return self._name

        def generate(self, prompt, aspect_ratio="landscape", **kwargs):
            return {"prompt": prompt}

    key = "ledger_mutable_provider"
    provider = MutableProvider()
    manager = PluginManager()
    scope = manager.scope_key
    baseline = get_provider(key, scope=scope)
    baseline_local = snapshot_registration(key, scope=scope)
    context = PluginContext(
        PluginManifest(name="mutable_provider", key="mutable_provider"), manager
    )
    try:
        assert context.register_image_gen_provider(provider) is not None
        assert get_provider(key, scope=scope) is provider
        provider._name = "renamed_after_registration"
        manager.unload("mutable_provider")
        assert get_provider(key, scope=scope) is baseline
    finally:
        current = snapshot_registration(key, scope=scope)
        if current is not baseline_local and current is not None:
            restore_registration(key, current, baseline_local, scope=scope)


def test_registration_transaction_excludes_concurrent_disposal(monkeypatch):
    """A lease cannot be retired between another generation's write/acquire."""
    from agent.image_gen_provider import ImageGenProvider
    import agent.image_gen_registry as image_registry
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    class Provider(ImageGenProvider):
        def __init__(self, marker):
            self.marker = marker

        @property
        def name(self):
            return "ledger_transaction_provider"

        def generate(self, prompt, aspect_ratio="landscape", **kwargs):
            return {"marker": self.marker}

    provider_a = Provider("a")
    provider_b = Provider("b")
    manager_a = PluginManager()
    manager_b = PluginManager()
    scope = manager_a.scope_key
    baseline = image_registry.get_provider("ledger_transaction_provider", scope=scope)
    baseline_local = image_registry.snapshot_registration(
        "ledger_transaction_provider", scope=scope
    )
    context_a = PluginContext(PluginManifest(name="tx_a", key="tx_a"), manager_a)
    context_b = PluginContext(PluginManifest(name="tx_b", key="tx_b"), manager_b)
    context_a.register_image_gen_provider(provider_a)

    original_register = image_registry.register_provider
    wrote = Event()
    release = Event()

    def paused_register(provider, *, scope=None):
        original_register(provider, scope=scope)
        if provider is provider_b:
            wrote.set()
            assert release.wait(timeout=2)

    monkeypatch.setattr(image_registry, "register_provider", paused_register)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            registration = pool.submit(context_b.register_image_gen_provider, provider_b)
            assert wrote.wait(timeout=1)
            disposal = pool.submit(manager_a.unload, "tx_a")
            try:
                disposal.result(timeout=0.05)
            except TimeoutError:
                pass
            else:
                raise AssertionError("dispose interleaved with registration transaction")
            release.set()
            assert registration.result(timeout=1) is not None
            assert disposal.result(timeout=1) is True

        assert image_registry.get_provider(provider_a.name, scope=scope) is provider_b
        manager_b.unload("tx_b")
        assert image_registry.get_provider(provider_a.name, scope=scope) is baseline
    finally:
        release.set()
        current = image_registry.snapshot_registration(
            "ledger_transaction_provider", scope=scope
        )
        if current is not baseline_local and current is not None:
            image_registry.restore_registration(
                "ledger_transaction_provider",
                current,
                baseline_local,
                scope=scope,
            )


def test_deferred_platform_resolution_is_atomic_across_threads():
    """Concurrent first lookups both observe the materialized adapter."""
    from gateway.platform_registry import PlatformEntry, PlatformRegistry

    registry = PlatformRegistry()
    entry = PlatformEntry(
        name="threaded_platform",
        label="Threaded",
        adapter_factory=lambda config: object(),
        check_fn=lambda: True,
        source="plugin",
    )
    loads = []
    started = Event()
    release = Event()

    def load():
        loads.append(1)
        started.set()
        assert release.wait(timeout=2)
        registry.register(entry)

    registry.register_deferred("threaded_platform", load)
    with ThreadPoolExecutor(max_workers=3) as pool:
        first = pool.submit(registry.get, "threaded_platform")
        assert started.wait(timeout=1)
        assert registry.is_registered("threaded_platform")
        second = pool.submit(registry.get, "threaded_platform")
        try:
            second.result(timeout=0.05)
        except TimeoutError:
            pass
        else:
            raise AssertionError("concurrent lookup did not wait for materialization")
        enumeration = pool.submit(registry.all_entries)
        try:
            enumeration.result(timeout=0.05)
        except TimeoutError:
            pass
        else:
            raise AssertionError("enumeration omitted an in-flight platform")
        release.set()
        results = [first.result(timeout=1), second.result(timeout=1)]
        assert enumeration.result(timeout=1) == [entry]

    assert results == [entry, entry]
    assert loads == [1]


def test_deferred_platform_recursive_lookup_does_not_deadlock():
    """A loader that asks for its own entry fails fast until registration."""
    from gateway.platform_registry import PlatformEntry, PlatformRegistry

    registry = PlatformRegistry()
    nested_results = []
    entry = PlatformEntry(
        name="recursive_platform",
        label="Recursive",
        adapter_factory=lambda config: object(),
        check_fn=lambda: True,
        source="plugin",
    )

    def load():
        nested_results.append(registry.get("recursive_platform"))
        registry.register(entry)

    registry.register_deferred("recursive_platform", load)
    assert registry.get("recursive_platform") is entry
    assert nested_results == [None]


def test_resolved_deferred_platform_restores_its_displaced_loader():
    """Deferred-to-concrete loading remains one replacement chain."""
    from gateway.platform_registry import platform_registry
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    name = "ledger_transfer"
    scope = platform_registry.current_scope_key()
    previous = platform_registry.snapshot_registration(name, scope=scope)
    old_loader = lambda: None
    platform_registry.register_deferred(name, old_loader, scope=scope)
    displaced = platform_registry.snapshot_registration(name, scope=scope)
    manager = PluginManager(scope_key=scope)
    manifest = PluginManifest(
        name=f"{name}-platform",
        key=f"{name}-platform",
        source="bundled",
        path="unused",
    )

    def load_scoped(self, loaded_manifest):
        PluginContext(loaded_manifest, self).register_platform(
            name=name,
            label="Transferred",
            adapter_factory=lambda config: object(),
            check_fn=lambda: True,
        )

    manager._load_plugin_scoped = MethodType(load_scoped, manager)
    try:
        manager._register_deferred_platform(manifest)
        entry = platform_registry.get(name)
        assert entry is not None and entry.label == "Transferred"
        manager.unload(manifest)
        assert platform_registry.snapshot_registration(name, scope=scope) == displaced
    finally:
        current = platform_registry.snapshot_registration(name, scope=scope)
        platform_registry.restore_registration(
            name, current, previous, scope=scope
        )


def test_unload_cancels_a_deferred_platform_before_module_load():
    """Losing the in-flight race cannot publish registrations after unload."""
    from gateway.platform_registry import platform_registry
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    name = "ledger_cancel"
    scope = platform_registry.current_scope_key()
    previous = platform_registry.snapshot_registration(name, scope=scope)
    def old_loader():
        from gateway.platform_registry import PlatformEntry

        platform_registry.register(
            PlatformEntry(
                name=name,
                label="Restored predecessor",
                adapter_factory=lambda config: object(),
                check_fn=lambda: True,
                source="plugin",
            ),
            scope=scope,
        )
    platform_registry.register_deferred(name, old_loader, scope=scope)
    displaced = platform_registry.snapshot_registration(name, scope=scope)
    manager = PluginManager(scope_key=scope)
    manifest = PluginManifest(
        name=f"{name}-platform",
        key=f"{name}-platform",
        source="bundled",
        path="unused",
    )

    def load_scoped(self, loaded_manifest):
        PluginContext(loaded_manifest, self).register_platform(
            name=name,
            label="Should not publish",
            adapter_factory=lambda config: object(),
            check_fn=lambda: True,
        )

    manager._load_plugin_scoped = MethodType(load_scoped, manager)
    manager._register_deferred_platform(manifest)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            with manager._discovery_lock:
                lookup = pool.submit(platform_registry.get, name)
                deadline = monotonic() + 1
                while (scope, name) not in platform_registry._inflight:
                    if monotonic() >= deadline:
                        raise AssertionError("deferred loader never became in-flight")
                    sleep(0.001)
                assert manager.unload(manifest) is True
            restored_entry = lookup.result(timeout=1)
        assert restored_entry is not None
        assert restored_entry.label == "Restored predecessor"
        assert platform_registry.snapshot_registration(name, scope=scope)[0] is restored_entry
        assert name not in manager._plugin_platform_names
    finally:
        current = platform_registry.snapshot_registration(name, scope=scope)
        platform_registry.restore_registration(
            name, current, previous, scope=scope
        )


def test_direct_plugin_platform_registration_infers_immutable_scope(tmp_path):
    """The documented direct registry API cannot leak into another profile."""
    from gateway.platform_registry import PlatformEntry, platform_registry
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools.registry import registry as tool_registry

    home_a = str((tmp_path / "direct-a").resolve())
    home_b = tmp_path / "direct-b"
    module_name = "company.hermes.direct_platform_probe"
    policy = tool_registry.register_plugin_override_policy(
        module_name, False, scope=home_a
    )
    factory = eval(
        "lambda config: 'direct-a'",
        {"__name__": f"{module_name}.handlers"},
    )
    name = "ledger_direct_platform"
    previous_a = platform_registry.snapshot_registration(name, scope=home_a)
    previous_global = platform_registry.snapshot_registration(name)
    token = set_hermes_home_override(home_b)
    try:
        platform_registry.register(
            PlatformEntry(
                name=name,
                label="Direct A",
                adapter_factory=factory,
                check_fn=lambda: True,
                source="plugin",
            )
        )
        assert platform_registry.get(name) is None
    finally:
        reset_hermes_home_override(token)

    token = set_hermes_home_override(home_a)
    try:
        assert platform_registry.get(name).label == "Direct A"
    finally:
        reset_hermes_home_override(token)
        current_a = platform_registry.snapshot_registration(name, scope=home_a)
        platform_registry.restore_registration(
            name, current_a, previous_a, scope=home_a
        )
        current_global = platform_registry.snapshot_registration(name)
        if current_global != previous_global:
            platform_registry.restore_registration(
                name, current_global, previous_global
            )
        tool_registry.restore_plugin_override_policy(
            module_name, policy, None, scope=home_a
        )


def test_same_name_tool_and_platform_are_isolated_by_hermes_home(
    tmp_path,
    monkeypatch,
):
    """Real A→B→A profile switching keeps dispatch and adapters isolated."""
    import hermes_cli.plugins as plugins_mod
    from gateway.platform_registry import platform_registry
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from hermes_cli.plugins import PluginManager
    from tools.registry import registry

    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    _write_profile_probe(home_a, "profile-a")
    _write_profile_probe(home_b, "profile-b")
    monkeypatch.setattr(
        plugins_mod,
        "get_bundled_plugins_dir",
        lambda: tmp_path / "empty-bundled",
    )
    monkeypatch.setattr(PluginManager, "_scan_entry_points", lambda self: [])

    def load_profile(home: Path):
        token = set_hermes_home_override(home)
        try:
            manager = plugins_mod.get_plugin_manager()
            manager.discover_and_load()
            tool_entry = registry.get_entry("shared_profile_tool")
            platform_entry = platform_registry.get("shared_profile_platform")
            return manager, tool_entry, platform_entry
        finally:
            reset_hermes_home_override(token)

    manager_a, tool_a, platform_a = load_profile(home_a)
    manager_b, tool_b, platform_b = load_profile(home_b)

    assert manager_a is not manager_b
    assert tool_a is not None and tool_b is not None and tool_a is not tool_b
    assert platform_a is not None and platform_b is not None and platform_a is not platform_b

    token_a = set_hermes_home_override(home_a)
    try:
        assert registry.dispatch("shared_profile_tool", {}) == "profile-a"
        assert platform_registry.get("shared_profile_platform") is platform_a
    finally:
        reset_hermes_home_override(token_a)

    token_b = set_hermes_home_override(home_b)
    try:
        assert registry.dispatch("shared_profile_tool", {}) == "profile-b"
        assert platform_registry.get("shared_profile_platform") is platform_b
    finally:
        reset_hermes_home_override(token_b)

    token_a = set_hermes_home_override(home_a)
    try:
        assert registry.dispatch("shared_profile_tool", {}) == "profile-a"
        assert platform_registry.get("shared_profile_platform") is platform_a
    finally:
        reset_hermes_home_override(token_a)


def test_manager_discovery_uses_its_home_not_the_ambient_profile(
    tmp_path,
    monkeypatch,
):
    """A retained manager cannot scan another concurrently active profile."""
    import hermes_cli.plugins as plugins_mod
    from gateway.platform_registry import platform_registry
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from hermes_cli.plugins import PluginManager
    from tools.registry import registry

    home_a = tmp_path / "retained-a"
    home_b = tmp_path / "ambient-b"
    _write_profile_probe(home_a, "retained-a")
    _write_profile_probe(home_b, "ambient-b")
    monkeypatch.setattr(
        plugins_mod,
        "get_bundled_plugins_dir",
        lambda: tmp_path / "empty-bundled",
    )
    monkeypatch.setattr(PluginManager, "_scan_entry_points", lambda self: [])

    manager_a = PluginManager(scope_key=str(home_a.resolve()))
    ambient = set_hermes_home_override(home_b)
    try:
        manager_a.discover_and_load()
    finally:
        reset_hermes_home_override(ambient)

    tool_a = registry.get_entry("shared_profile_tool", scope=manager_a.scope_key)
    platform_a = platform_registry.snapshot_registration(
        "shared_profile_platform", scope=manager_a.scope_key
    )[0]
    assert tool_a is not None and tool_a.handler({}) == "retained-a"
    assert platform_a is not None and platform_a.label == "retained-a"
    assert registry.snapshot_registration(
        "shared_profile_tool", scope=str(home_b.resolve())
    ) is None
    assert platform_registry.snapshot_registration(
        "shared_profile_platform", scope=str(home_b.resolve())
    ) == (None, None)


def test_same_slug_profiles_allocate_distinct_modules_concurrently(
    tmp_path,
    monkeypatch,
):
    """Policy binding and import use one atomic profile-specific namespace."""
    import hermes_cli.plugins as plugins_mod
    from hermes_cli.plugins import PluginManager

    home_a = tmp_path / "concurrent-a"
    home_b = tmp_path / "concurrent-b"
    _write_profile_probe(home_a, "concurrent-a")
    _write_profile_probe(home_b, "concurrent-b")
    monkeypatch.setattr(
        plugins_mod,
        "get_bundled_plugins_dir",
        lambda: tmp_path / "empty-bundled",
    )
    monkeypatch.setattr(PluginManager, "_scan_entry_points", lambda self: [])
    managers = [
        PluginManager(scope_key=str(home_a.resolve())),
        PluginManager(scope_key=str(home_b.resolve())),
    ]

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda manager: manager.discover_and_load(), managers))

    modules = [manager._plugins["profile_probe"].module.__name__ for manager in managers]
    assert modules[0] != modules[1]


def test_spawned_supervised_task_is_cancelled_on_unload():
    """A plugin-spawned background task is tracked and cancelled on unload."""
    import asyncio

    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    manager = PluginManager()
    manifest = PluginManifest(
        name="task_probe", version="0.1", description="", source="user",
    )
    ctx = PluginContext(manifest, manager)
    cancelled = []

    async def scenario():
        async def forever():
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                cancelled.append(True)
                raise

        task = ctx.spawn_task(forever(), name="probe-task")
        await asyncio.sleep(0)
        kinds = {
            registration.kind
            for registration in manager._ownership_ledger["task_probe"]
        }
        assert "background_task" in kinds
        assert manager.unload("task_probe") is True
        with __import__("pytest").raises(asyncio.CancelledError):
            await task
        # Done-callback disposal removes the handle from the ledger.
        await asyncio.sleep(0)
        assert "task_probe" not in manager._ownership_ledger

    asyncio.run(scenario())
    assert cancelled == [True]


def test_on_unload_exception_does_not_block_other_teardown():
    """A raising on_unload callback is isolated; later cleanup still runs."""
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    manager = PluginManager()
    manifest = PluginManifest(
        name="boom_probe", version="0.1", description="", source="user",
    )
    ctx = PluginContext(manifest, manager)
    order = []

    ctx.on_unload(lambda: order.append("first"))

    def _boom():
        order.append("boom")
        raise RuntimeError("cleanup failed")

    ctx.on_unload(_boom)
    ctx.on_unload(lambda: order.append("last"))

    assert manager.unload("boom_probe") is True
    # Reverse acquisition order, exception isolated.
    assert order == ["last", "boom", "first"]
    assert "boom_probe" not in manager._ownership_ledger
