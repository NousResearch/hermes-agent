"""End-to-end coverage for plugin registration ownership and reload cleanup."""

from __future__ import annotations

from pathlib import Path

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
    )
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": ["ledger_probe"]}})
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
    }

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
    assert manager._aux_tasks == {}
    assert manager._ownership_ledger == {}


def test_reverse_unload_restores_an_overridden_platform_registration():
    """Reverse teardown reveals an older entry before removing it."""
    from gateway.platform_registry import platform_registry
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    name = "ledger_override_platform"
    previous = platform_registry.snapshot_registration(name)
    manager_a = PluginManager()
    manager_b = PluginManager()
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
        assert platform_registry.snapshot_registration(name) == previous
    finally:
        # The test uses a deliberately unique name, but restore any state that
        # a surrounding test may have installed under it.
        current = platform_registry.snapshot_registration(name)
        platform_registry.restore_registration(name, current, previous)
