"""An opt-in channel catalog must bind to the actual receiving bot and workspace."""

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.protected_channel_tools import (
    ProtectedChannelRejected, ProtectedChannelToolsets, allowed_tool_names, policy_for_source,
)
from gateway.session_identity import IdentityUnresolved, identity_of, resolve_identity


POLICY = {
    "platform": "slack", "chat_id": "channel-a", "scope_id": "workspace-a",
    "bot_profile": "default", "toolsets": ["web"],
}


class _Stub(BasePlatformAdapter):
    pass


_Stub.__abstractmethods__ = frozenset()


def _rig(tmp_path, monkeypatch):
    from gateway.run import GatewayRunner

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig.from_dict({"multiplex_profiles": False,
                                             "protected_channel_toolsets": [POLICY]})
    runner._primary_profile_name = "default"
    bot = _Stub(PlatformConfig(enabled=True, extra={}), Platform.SLACK)
    bot.gateway_runner = runner
    runner.adapters = {Platform.SLACK: bot}
    runner._profile_adapters = {}
    return runner, bot


def _source(bot, *, chat="channel-a", scope="workspace-a"):
    return bot.build_source(chat_id=chat, chat_type="channel", user_id="person", scope_id=scope)


def test_exact_channel_uses_selected_catalog_and_other_channel_is_unchanged(tmp_path, monkeypatch):
    runner, bot = _rig(tmp_path, monkeypatch)
    protected = _source(bot)
    identity = resolve_identity(protected, runner=runner)
    assert identity.adapter() is bot
    assert runner._resolve_enabled_toolsets_for_source({}, protected, "slack") == ["web", "no_mcp"]
    assert runner._protected_channel_tool_names(protected, None)

    other = _source(bot, chat="channel-b")
    resolve_identity(other, runner=runner)
    assert runner._protected_channel_tool_names(other, None) is None
    assert runner._resolve_enabled_toolsets_for_source(
        {"platform_toolsets": {"slack": ["terminal"]}}, other, "slack") == ["terminal"]


@pytest.mark.parametrize("change", ["workspace", "bot", "missing_adapter"])
def test_partial_or_spoofed_binding_never_gets_a_default_catalog(tmp_path, monkeypatch, change):
    runner, bot = _rig(tmp_path, monkeypatch)
    source = _source(bot, scope="workspace-b" if change == "workspace" else "workspace-a")
    if change == "bot":
        # The callback's claimed profile disagrees with the registered bot.
        resolve = lambda: resolve_identity(source, runner=runner, transport_profile="other")
    elif change == "missing_adapter":
        del source._transport_adapter_ref
        resolve = lambda: resolve_identity(source, runner=runner)
    else:
        resolve = lambda: resolve_identity(source, runner=runner)
    with pytest.raises(IdentityUnresolved):
        resolve()
    assert source.profile_route_rejected is True
    assert identity_of(source) is None
    with pytest.raises(ProtectedChannelRejected):
        runner._resolve_enabled_toolsets_for_source({}, source, "slack")


@pytest.mark.parametrize("bad", [
    None,
    {**POLICY, "scope_id": None},
    {**POLICY, "toolsets": "web"},
    {**POLICY, "toolsets": ["web", "web"]},
    {**POLICY, "bot_profile": "../../default"},
    {**POLICY, "platform": "slak"},
])
def test_invalid_protection_fails_config_load(bad):
    with pytest.raises(ValueError):
        GatewayConfig.from_dict({"protected_channel_toolsets": bad if bad is None else [bad]})


def test_config_roundtrip_preserves_exact_policy():
    cfg = GatewayConfig.from_dict({"protected_channel_toolsets": [POLICY]})
    restored = GatewayConfig.from_dict(cfg.to_dict())
    assert restored.protected_channel_toolsets == cfg.protected_channel_toolsets


def test_gateway_yaml_layer_loads_policy_and_rejects_invalid_entry(tmp_path, monkeypatch):
    from gateway.config import load_gateway_config

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = tmp_path / "config.yaml"
    config.write_text(
        "gateway:\n  protected_channel_toolsets:\n"
        "    - platform: slack\n      scope_id: workspace-a\n      chat_id: channel-a\n"
        "      bot_profile: default\n      toolsets: [web]\n")
    assert load_gateway_config().protected_channel_toolsets[0].toolsets == ("web",)

    config.write_text(config.read_text().replace("toolsets: [web]", "toolsets: web"))
    with pytest.raises(ValueError, match="toolsets"):
        load_gateway_config()


def test_unavailable_toolset_refuses_the_protected_turn(tmp_path, monkeypatch):
    runner, bot = _rig(tmp_path, monkeypatch)
    runner.config = GatewayConfig.from_dict({
        "protected_channel_toolsets": [{**POLICY, "toolsets": ["missing_toolset_for_test"]}]})
    source = _source(bot)
    resolve_identity(source, runner=runner)
    with pytest.raises(ProtectedChannelRejected, match="unavailable"):
        runner._protected_channel_tool_names(source, None)


def test_joint_catalog_keeps_only_selected_mcp_and_no_mcp_stops_later_expansion(monkeypatch):
    from agent.inline_tool_executors import _scope_in_connected_mcp_servers
    from model_tools import get_tool_definitions
    from tools.registry import ToolRegistry
    import model_tools

    registry = ToolRegistry()
    for name, toolset in (("mcp_selected_answer", "mcp-selected"),
                          ("mcp_ambient_raw", "mcp-ambient")):
        registry.register(
            name=name, toolset=toolset, handler=lambda *_a, **_k: "{}",
            schema={"name": name, "description": name,
                    "parameters": {"type": "object", "properties": {}}},
        )
    monkeypatch.setattr("tools.registry.registry", registry)
    monkeypatch.setattr(model_tools, "registry", registry)
    policy = ProtectedChannelToolsets("slack", "channel-a", "workspace-a", "default",
                                      ("mcp-selected", "no_mcp"))
    selected = allowed_tool_names(policy)
    assert "mcp_selected_answer" in selected
    assert "mcp_ambient_raw" not in selected
    actual = {d["function"]["name"] for d in get_tool_definitions(
        enabled_toolsets=list(policy.toolsets), quiet_mode=True, skip_tool_search_assembly=True)}
    assert actual == {"mcp_selected_answer"}

    agent = type("Agent", (), {"enabled_toolsets": list(policy.toolsets)})()
    _scope_in_connected_mcp_servers(agent, '{"targets":[{"name":"mcp-ambient",'
                                           '"kind":"mcp","state":"connected","tools":["raw"]}]}')
    assert agent.enabled_toolsets == list(policy.toolsets)


@pytest.mark.asyncio
async def test_native_durable_restore_keeps_protected_catalog_without_inferred_identity(tmp_path, monkeypatch):
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.session_envelope import restore_native
    from gateway.session_ingress_context import native_callback, register_transport_home
    from hermes_state_runtime import list_session_admissions

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner = GatewayRunner(GatewayConfig.from_dict({"protected_channel_toolsets": [POLICY]}))
    runner._primary_profile_name = "default"
    bot = _Stub(PlatformConfig(enabled=True, extra={}), Platform.SLACK)
    bot.gateway_runner = runner
    runner.adapters = {Platform.SLACK: bot}
    runner._is_user_authorized_for_source = lambda *_a, **_k: True
    authority = await initialize_session_authority(runner, profile_id="default", instance_id="protected")
    register_transport_home(runner, None, tmp_path)
    try:
        event = MessageEvent(text="test", source=_source(bot), message_id="native-1")
        with native_callback(runner, event, tmp_path):
            receipt = await authority.admit_native(event)
        rows = list_session_admissions(authority.db, session_id=receipt.ref.session_id, pending_only=False)
        assert len(rows) == 1
        payload = rows[0]["payload"]
        restored = restore_native(payload, runner)
        assert identity_of(restored.source) is None
        assert restored.source._transport_adapter_ref() is bot
        assert restored.source._native_provenance_verified is True
        assert runner._resolve_enabled_toolsets_for_source({}, restored.source, "slack") == ["web", "no_mcp"]
        assert runner._protected_channel_tool_names(restored.source, None)

        restored.source._native_provenance_verified = False
        with pytest.raises(ProtectedChannelRejected):
            runner._resolve_enabled_toolsets_for_source({}, restored.source, "slack")
        restored.source._native_provenance_verified = True

        # Rebind the queued row after a connector restart. The persisted proof
        # is checked against the newly registered receiver, not the old object.
        replacement = _Stub(PlatformConfig(enabled=True, extra={}), Platform.SLACK)
        replacement.gateway_runner = runner
        runner.adapters = {Platform.SLACK: replacement}
        restarted = restore_native(payload, runner)
        assert restarted.source._transport_adapter_ref() is replacement
        assert runner._resolve_enabled_toolsets_for_source({}, restarted.source, "slack") == ["web", "no_mcp"]

        # A removed receiver cannot be inferred from the stored row or channel.
        runner.adapters = {}
        with pytest.raises(ProtectedChannelRejected):
            runner._resolve_enabled_toolsets_for_source({}, restarted.source, "slack")
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_protected_turn_cannot_escape_through_proxy(tmp_path, monkeypatch):
    runner, bot = _rig(tmp_path, monkeypatch)
    source = _source(bot)
    resolve_identity(source, runner=runner)
    result = await runner._run_agent_via_proxy("question", "", [], source, "session")
    assert result["api_calls"] == 0
    assert "cannot be enforced" in result["final_response"]


def test_policy_deletion_is_no_longer_enforced_and_needs_external_drift_check(tmp_path, monkeypatch):
    runner, bot = _rig(tmp_path, monkeypatch)
    source = _source(bot)
    resolve_identity(source, runner=runner)
    runner.config.protected_channel_toolsets = []
    assert policy_for_source(runner.config, source, transport_profile="default", live_adapter=True) is None
