"""Exact grants cover raw dispatch and the uncollapsed deferred catalog."""
import json
from unittest.mock import MagicMock, patch

import pytest

import model_tools
from tools.registry import registry


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    handlers = {}
    for name, toolset in [
        ("permission_builtin", "web"),
        ("permission_plugin", "permission_plugin_set"),
        ("mcp_permission_demo_read", "mcp_permission_demo"),
        ("mcp_permission_demo_delete", "mcp_permission_demo"),
        ("permission_client", "desktop_ui"),
    ]:
        handlers[name] = MagicMock(return_value='{"executed": true}')
        registry.register(name=name, toolset=toolset, schema={
            "name": name, "description": "permission probe",
            "parameters": {"type": "object", "properties": {}},
        }, handler=handlers[name], check_fn=lambda: True)
    yield tmp_path, handlers
    for name in handlers:
        registry._tools.pop(name, None)
    model_tools._clear_tool_defs_cache()


@pytest.mark.parametrize("allowed", [[], "permission_builtin", False, 1, {}, ["permission_builtin", None], ["permission_builtin", 2], [""], [" permission_builtin"]])
def test_malformed_or_empty_grants_deny_every_tool(catalog, allowed):
    home, handlers = catalog
    configure(home, allowed)
    assert model_tools.get_tool_definitions(quiet_mode=True) == []
    assert "agent.allowed_tools" in model_tools.handle_function_call("permission_builtin", {})
    handlers["permission_builtin"].assert_not_called()


def test_unset_or_null_preserves_existing_dispatch(catalog):
    home, handlers = catalog
    for config in [{}, {"agent": {"allowed_tools": None}}]:
        (home / "config.yaml").write_text(json.dumps(config))
        assert json.loads(model_tools.handle_function_call("permission_builtin", {}))["executed"]
    assert handlers["permission_builtin"].call_count == 2


def test_default_registry_exposes_backward_compatible_unrestricted_option():
    from hermes_cli.config import DEFAULT_CONFIG
    assert DEFAULT_CONFIG["agent"]["allowed_tools"] is None


def test_bound_profile_grant_does_not_borrow_launch_profile(catalog):
    home, handlers = catalog
    configure(home, None)
    other = home / "other"; other.mkdir()
    configure(other, [])
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    token = set_hermes_home_override(other)
    try:
        assert "agent.allowed_tools" in model_tools.handle_function_call("permission_builtin", {})
        assert model_tools.get_tool_definitions(quiet_mode=True) == []
    finally:
        reset_hermes_home_override(token)
    assert json.loads(model_tools.handle_function_call("permission_builtin", {}))["executed"]


def configure(home, allowed):
    (home / "config.yaml").write_text(json.dumps({
        "agent": {"allowed_tools": allowed}, "tool_search": {"enabled": "off"},
    }))


@pytest.mark.parametrize("name", ["permission_builtin", "permission_plugin", "mcp_permission_demo_delete", "permission_client"])
def test_raw_dispatch_denies_without_relying_on_visible_schemas(catalog, name):
    home, handlers = catalog
    configure(home, ["mcp_permission_demo_read"])
    result = model_tools.handle_function_call(name, {}, enabled_tools=[name])
    assert "agent.allowed_tools" in result
    handlers[name].assert_not_called()


def test_uncollapsed_catalog_filters_exact_names(catalog):
    home, handlers = catalog
    configure(home, ["mcp_permission_demo_read"])
    definitions = model_tools.get_tool_definitions(quiet_mode=True, skip_tool_search_assembly=True)
    assert {d["function"]["name"] for d in definitions} == {"mcp_permission_demo_read"}


def test_bridge_grant_is_not_a_grant_to_underlying_tool(catalog):
    home, handlers = catalog
    configure(home, ["tool_call", "tool_search", "tool_describe", "mcp_permission_demo_read"])
    denied = model_tools.handle_function_call("tool_call", {
        "name": "mcp_permission_demo_delete", "arguments": {},
    })
    assert "agent.allowed_tools" in denied
    handlers["mcp_permission_demo_delete"].assert_not_called()
    allowed = model_tools.handle_function_call("tool_call", {
        "name": "mcp_permission_demo_read", "arguments": {},
    })
    assert json.loads(allowed)["executed"] is True
    handlers["mcp_permission_demo_read"].assert_called_once()
