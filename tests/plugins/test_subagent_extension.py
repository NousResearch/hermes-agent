"""Native discovery must scope and revoke delegation extensions like other plugin registrations.

These frozen consumers guard profile isolation, tool-override consent, and unload:
without them a plugin could silently change another profile's delegate tool.
"""
import json
from pathlib import Path

import pytest


PLUGIN = '''
from agent.subagent_extension import SubagentExtension
class Extension(SubagentExtension):
    def extend_schema(self, schema):
        schema["parameters"]["properties"]["extension_probe"] = {"type": "string"}
        return schema
    def handle_action(self, action, request):
        import json
        return json.dumps({"echo": request["extension_probe"]}) if action == "probe" else None

def register(ctx):
    ctx.register_subagent_extension(Extension())
'''


@pytest.fixture
def homes(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    first, second = tmp_path / "first", tmp_path / "second"
    for home in (first, second):
        home.mkdir()
        (home / "config.yaml").write_text('{}')
    monkeypatch.setenv("HERMES_HOME", str(first))
    plugin = first / "plugins" / "extension-example"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text('name: extension-example\nversion: 0.1.0\n')
    (plugin / "__init__.py").write_text(PLUGIN)
    return first, second


def activate(home, *, allow):
    from hermes_cli.plugins import discover_plugins, get_plugin_manager
    (home / "config.yaml").write_text(json.dumps({"plugins": {
        "enabled": ["extension-example"], "entries": {"extension-example": {"allow_tool_override": allow}}}}))
    discover_plugins(force=True)
    return get_plugin_manager()


def parameters():
    from tools.delegate_tool import _build_dynamic_schema_overrides
    return _build_dynamic_schema_overrides()["parameters"]["properties"]


def test_extension_is_profile_scoped_and_unload_restores_native_schema(homes):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    first, second = homes
    manager = activate(first, allow=True)
    assert "extension_probe" in parameters()
    token = set_hermes_home_override(second)
    try:
        assert "extension_probe" not in parameters()
    finally:
        reset_hermes_home_override(token)
    assert "extension_probe" in parameters()
    manager.unload("extension-example")
    assert "extension_probe" not in parameters()


def test_extension_requires_existing_tool_override_consent(homes):
    activate(homes[0], allow=False)
    assert "extension_probe" not in parameters()


def test_plugin_arguments_cross_inline_and_registry_dispatch(homes):
    from run_agent import AIAgent
    from tools.registry import registry
    activate(homes[0], allow=True)
    parent = object.__new__(AIAgent)
    args = {"action": "probe", "extension_probe": "new plugin-owned parameter"}
    inline = json.loads(parent._dispatch_delegate_task(args))
    registered = json.loads(registry.get_entry("delegate_task").handler(args, parent_agent=parent))
    assert inline == registered == {"echo": args["extension_probe"]}
