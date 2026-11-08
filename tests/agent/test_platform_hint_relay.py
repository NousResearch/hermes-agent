"""Relay handshake hints reach new prompts without changing a stored prompt."""

import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.prompt_builder import PLATFORM_HINTS
from agent.system_prompt import build_system_prompt, platform_hint
from gateway.authz_mixin import GatewayAuthorizationMixin
from gateway.config import Platform, PlatformConfig
from gateway.platform_registry import PlatformEntry, PlatformRegistry
from gateway.relay.adapter import RelayAdapter
from gateway.relay.descriptor import CapabilityDescriptor
from gateway.relay.ws_transport import WebSocketRelayTransport
from hermes_state import SessionDB
from tests.agent.test_system_prompt import _make_agent
from tests.agent.test_system_prompt_switch_invalidation import _turn_agent
from tests.gateway.relay.stub_connector import StubConnector
from tests.gateway.relay.test_relay_per_platform_caps import MultiDescriptorStub


def _descriptor(platform="relay", hint="Client formatting instructions"):
    return CapabilityDescriptor(
        contract_version=1, platform=platform, label=platform,
        max_message_length=4096, supports_draft_streaming=False,
        supports_edit=True, supports_threads=False, markdown_dialect="plain",
        len_unit="chars", platform_hint=hint,
    )


@pytest.fixture
def relay_gateway(monkeypatch):
    import gateway.run

    registry = PlatformRegistry()
    monkeypatch.setattr("gateway.platform_registry.platform_registry", registry)
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
    runner = GatewayAuthorizationMixin()
    runner.adapters = {}
    runner._profile_adapters = {}
    runner._primary_profile_name = "custom"
    runner.config = SimpleNamespace(multiplex_profiles=False)
    monkeypatch.setattr(gateway.run, "_gateway_runner_ref", lambda: runner)
    return runner, registry


async def _connect(runner, *descriptors):
    transport = MultiDescriptorStub(*descriptors)
    adapter = RelayAdapter(PlatformConfig(), _descriptor(hint=""), transport)
    assert await adapter.connect()
    runner.adapters[Platform.RELAY] = adapter
    return adapter, transport


@pytest.mark.asyncio
async def test_negotiated_hint_reaches_system_prompt(relay_gateway):
    runner, _ = relay_gateway
    # A real client's hint is much larger than a chat message.
    hint = "Native answer blocks and buttons.\n" * 600
    await _connect(runner, _descriptor(hint=hint))
    assert hint in build_system_prompt(_make_agent(platform="relay"))


@pytest.mark.asyncio
async def test_builtin_and_plugin_hints_take_precedence(relay_gateway):
    runner, registry = relay_gateway
    await _connect(runner, _descriptor("tui"), _descriptor("relay"))
    assert platform_hint(_make_agent(platform="tui")) == PLATFORM_HINTS["tui"]
    registry.register(PlatformEntry(
        name="relay", label="Plugin", adapter_factory=lambda _: None,
        check_fn=lambda: True, platform_hint="Plugin instructions",
    ), scope=registry.current_scope_key())
    assert platform_hint(_make_agent(platform="relay")) == "Plugin instructions"


@pytest.mark.asyncio
@pytest.mark.parametrize("override, expected", [
    ({"append": "Extra"}, "Client formatting instructions\n\nExtra"),
    ({"replace": "Replacement"}, "Replacement"),
])
async def test_config_override_applies_to_relay_hint(relay_gateway, override, expected):
    runner, _ = relay_gateway
    await _connect(runner, _descriptor())
    agent = _make_agent(platform="relay", _platform_hint_overrides={"relay": override})
    assert platform_hint(agent) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("hint", ["", None, 123, ["bad"], {"bad": "shape"}])
async def test_empty_or_malformed_hint_is_ignored(relay_gateway, hint):
    runner, _ = relay_gateway
    await _connect(runner, _descriptor(hint=hint))
    assert platform_hint(_make_agent(platform="relay")) == ""


@pytest.mark.asyncio
async def test_omitted_wire_hint_keeps_prompt_without_a_hint(relay_gateway):
    runner, _ = relay_gateway
    wire = dict(_descriptor().__dict__)
    del wire["platform_hint"]
    await _connect(runner, CapabilityDescriptor.from_json(json.dumps(wire)))
    assert platform_hint(_make_agent(platform="relay")) == ""


@pytest.mark.asyncio
async def test_missing_descriptor_does_not_use_other_platform_hint(relay_gateway):
    runner, _ = relay_gateway
    await _connect(runner, _descriptor("other", "Other platform only"))
    assert platform_hint(_make_agent(platform="relay")) == ""


@pytest.mark.asyncio
async def test_fronted_platforms_get_their_own_hints(relay_gateway):
    runner, _ = relay_gateway
    await _connect(runner, _descriptor("relay", "First"), _descriptor("other", "Second"))
    assert platform_hint(_make_agent(platform="relay")) == "First"
    assert platform_hint(_make_agent(platform="other")) == "Second"


@pytest.mark.asyncio
async def test_legacy_transport_uses_its_matching_handshake_descriptor(relay_gateway):
    runner, _ = relay_gateway
    adapter = RelayAdapter(PlatformConfig(), _descriptor(hint=""), StubConnector(_descriptor()))
    assert await adapter.connect()
    runner.adapters[Platform.RELAY] = adapter
    assert platform_hint(_make_agent(platform="relay")) == "Client formatting instructions"
    assert platform_hint(_make_agent(platform="other")) == ""


@pytest.mark.asyncio
async def test_rehandshake_changes_new_prompt_but_keeps_stored_prompt(relay_gateway, tmp_path):
    from agent.conversation_loop import _restore_or_build_system_prompt

    runner, _ = relay_gateway
    old = _descriptor(hint="Old hint")
    transport = WebSocketRelayTransport("wss://connector.example/relay", "relay", "bot")
    await transport._handle_frame(json.dumps({"type": "descriptor", "descriptor": old.__dict__}))
    runner.adapters[Platform.RELAY] = RelayAdapter(PlatformConfig(), old, transport)
    original = build_system_prompt(_make_agent(platform="relay"))
    assert "Old hint" in original

    with SessionDB(db_path=tmp_path / "state.db") as db:
        db.create_session("switch-session", source="relay", model="model")
        db.update_system_prompt("switch-session", original)
        new = replace(old, platform_hint="New hint")
        await transport._handle_frame(json.dumps({"type": "descriptor", "descriptor": new.__dict__}))
        assert "New hint" in build_system_prompt(_make_agent(platform="relay"))
        continuing = _turn_agent(db, "", "")
        continuing.platform = "relay"
        _restore_or_build_system_prompt(continuing, None, [{"role": "user", "content": "hi"}])
        continuing._build_system_prompt.assert_not_called()
        assert continuing._cached_system_prompt == original
        assert continuing._surface_switch_note == ""
        assert db.get_session("switch-session")["system_prompt"] == original


@pytest.mark.asyncio
async def test_profile_scope_selects_only_its_own_relay(relay_gateway, tmp_path, monkeypatch):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    runner, registry = relay_gateway
    profiles_root = tmp_path / "profiles"
    monkeypatch.setattr("hermes_cli.profiles._get_profiles_root", lambda: profiles_root)
    await _connect(runner, _descriptor(hint="Primary private hint"))
    secondary = RelayAdapter(
        PlatformConfig(), _descriptor(hint=""), StubConnector(_descriptor(hint="Secondary hint")),
    )
    assert await secondary.connect()
    runner._profile_adapters["secondary"] = {Platform.RELAY: secondary}
    token = set_hermes_home_override(profiles_root / "secondary")
    try:
        assert platform_hint(_make_agent(platform="relay")) == "Secondary hint"
        registry.register(PlatformEntry(
            name="relay", label="Scoped", adapter_factory=lambda _: None,
            check_fn=lambda: True, platform_hint="Scoped plugin hint",
        ), scope=registry.current_scope_key())
        assert platform_hint(_make_agent(platform="relay")) == "Scoped plugin hint"
    finally:
        reset_hermes_home_override(token)
    token = set_hermes_home_override(profiles_root / "unserved")
    try:
        assert platform_hint(_make_agent(platform="relay")) == ""
    finally:
        reset_hermes_home_override(token)


def test_no_gateway_leaves_hint_empty(relay_gateway, monkeypatch):
    monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: None)
    assert platform_hint(_make_agent(platform="relay")) == ""


def test_failed_relay_lookup_does_not_break_prompt(relay_gateway):
    runner, _ = relay_gateway
    runner._adapters_for_profile = Mock(side_effect=RuntimeError("adapter unavailable"))
    assert platform_hint(_make_agent(platform="relay")) == ""
