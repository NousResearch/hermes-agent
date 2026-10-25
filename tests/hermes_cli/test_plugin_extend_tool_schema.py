"""ctx.extend_tool_schema: a plugin adds fields to another tool's schema."""

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from tools.registry import registry


def _context(manager, name):
    return PluginContext(PluginManifest(name=name, key=name), manager)


def _function_schema(name):
    return registry.get_definitions({name})[0]["function"]


def test_extender_applies_to_a_tool_registered_later_and_unloads_with_the_plugin():
    name = "extend_schema_target"
    manager = PluginManager(scope_key=registry.current_scope_key())
    ctx = _context(manager, "extend_schema_plugin")

    def add_model_role(schema):
        schema["parameters"]["properties"]["tasks"]["items"]["properties"]["model_role"] = {
            "type": "string", "enum": ["fast", "smart"]}
        return schema

    ctx.extend_tool_schema(name, add_model_role)
    task_props = {"goal": {"type": "string"}}
    registry.register(
        name=name, toolset="extend_schema_core",
        schema={"name": name, "description": "static", "parameters": {"type": "object", "properties": {
            "tasks": {"type": "array", "items": {"type": "object", "properties": task_props}}}}},
        handler=lambda args, **kw: "ok",
        dynamic_schema_overrides=lambda: {"description": "runtime"},
    )
    try:
        extended = _function_schema(name)
        assert extended["description"] == "runtime"
        assert extended["parameters"]["properties"]["tasks"]["items"]["properties"]["model_role"]["enum"] == [
            "fast", "smart"]
        assert task_props == {"goal": {"type": "string"}}

        manager.unload("extend_schema_plugin")
        assert "model_role" not in _function_schema(name)["parameters"]["properties"]["tasks"]["items"]["properties"]
    finally:
        registry.deregister(name)


def test_failing_or_renaming_extender_is_skipped_and_later_ones_still_run():
    name = "extend_schema_faulty"
    manager = PluginManager(scope_key=registry.current_scope_key())
    ctx = _context(manager, "extend_schema_faulty_plugin")
    registry.register(
        name=name, toolset="extend_schema_core",
        schema={"name": name, "description": "static", "parameters": {"type": "object"}},
        handler=lambda args, **kw: "ok",
    )

    def boom(schema):
        raise RuntimeError("broken plugin")

    ctx.extend_tool_schema(name, boom)
    ctx.extend_tool_schema(name, lambda schema: {**schema, "name": "hijacked"})
    ctx.extend_tool_schema(name, lambda schema: {**schema, "description": schema["description"] + " +1"})
    try:
        assert _function_schema(name) == {
            "name": name, "description": "static +1", "parameters": {"type": "object"}}
    finally:
        manager.unload("extend_schema_faulty_plugin")
        registry.deregister(name)
