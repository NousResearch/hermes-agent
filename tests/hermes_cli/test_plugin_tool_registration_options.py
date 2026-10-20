"""Plugin tool registration forwards registry options (regression for #138801)."""

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from tools.registry import registry


def test_plugin_tool_uses_custom_result_cap_and_dynamic_schema():
    scope = registry.current_scope_key()
    manager = PluginManager(scope_key=scope)
    context = PluginContext(
        PluginManifest(name="registration_options", key="registration_options"), manager
    )
    name = "registration_options_tool"
    previous = registry.snapshot_registration(name, scope=scope)
    schema = {"name": name, "description": "static", "parameters": {"type": "object"}}

    try:
        handle = context.register_tool(
            name=name,
            toolset="registration_options",
            schema=schema,
            handler=lambda args: "ok",
            max_result_size_chars=4096,
            dynamic_schema_overrides=lambda: {"description": "runtime"},
            execution_timeout=900,
        )

        assert handle is not None
        assert registry.get_max_result_size(name) == 4096
        assert registry.get_entry(name, scope=scope).execution_timeout == 900
        assert registry.get_definitions({name}) == [
            {"type": "function", "function": {**schema, "description": "runtime"}}
        ]
    finally:
        manager.unload("registration_options")
        current = registry.snapshot_registration(name, scope=scope)
        if current is not previous:
            registry.restore_registration(name, current, previous, scope=scope)
