"""Profile-local plugins must be discovered before a multiplexed turn pins tools."""

from gateway.run import GatewayRunner
from gateway.config import Platform
from gateway.session import SessionSource
from hermes_constants import hermes_home_key, reset_hermes_home_override, set_hermes_home_override
from tools.registry import registry


def test_profile_plugin_tool_is_in_gateway_turn_surface(tmp_path, monkeypatch):
    from hermes_cli import plugins, tools_config
    from model_tools import get_tool_definitions

    # model_tools was already imported under the host profile. A new profile's
    # plugin has not yet been discovered, as on the multiplexed gateway.
    name = "profile_probe_receipt_tool"
    scope = hermes_home_key(tmp_path)
    calls = []

    def discover_profile_plugin():
        calls.append(scope)
        registry.register(
            name=name, toolset="profile_probe_receipts",
            schema={"name": name, "description": "Read a scoped receipt", "parameters": {"type": "object", "properties": {}}},
            handler=lambda args: "read only", scope=scope,
        )

    monkeypatch.setattr(plugins, "discover_plugins", discover_profile_plugin)
    # An in-flight scan can report its persisted key before registering tools.
    monkeypatch.setattr(tools_config, "_get_platform_tools", lambda config, platform: {"clarify", "profile_probe_receipts"})
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._delivery_adapter_for = lambda source: None
    token = set_hermes_home_override(str(tmp_path))
    try:
        selected = runner._resolve_enabled_toolsets_for_source({}, SessionSource(platform=Platform.DISCORD, user_id="u", chat_id="c", chat_type="dm"), "discord")
        names = {tool["function"]["name"] for tool in get_tool_definitions(selected, quiet_mode=True, skip_tool_search_assembly=True)}
        assert name in names
        assert calls == [scope]
    finally:
        registry.deregister(name, scope=scope)
        reset_hermes_home_override(token)
