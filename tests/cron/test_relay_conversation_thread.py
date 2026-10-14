"""A cold relay thread must carry the resolved logical platform and runtime identity."""
import asyncio
from pathlib import Path

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize("primary_supports_thread,secondary_supports_thread,multiplex", [(True, True, True), (False, True, True), (True, False, True), (True, True, False)])
async def test_cold_relay_cron_thread_keeps_its_resolved_address(tmp_path, monkeypatch, primary_supports_thread, secondary_supports_thread, multiplex):
    from cron.scheduler_delivery import _prepare_target_delivery
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.conversation import conversation_origin
    from gateway.profile_routing import parse_profile_routes
    from gateway.relay.adapter import RelayAdapter
    from gateway.relay.descriptor import CapabilityDescriptor
    from gateway.relay.ws_transport import WebSocketRelayTransport
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource
    from gateway.session_identity import identity_of

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    ops = home / "profiles" / "ops"
    ops.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_cli.profiles.get_active_profile_name", lambda: "default")
    monkeypatch.setattr("hermes_cli.profiles.profiles_to_serve", lambda **kwargs: [("default", home), ("ops", ops)])
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: ops)
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name == "ops")
    transport = WebSocketRelayTransport(
        "ws://unused.invalid", "discord", "discord-bot",
        identities=[("discord", "discord-bot"), ("slack", "slack-bot")],
    )
    transport._ws = object()
    frames = []

    async def send_frame(frame):
        frames.append(frame)
        transport._pending[frame["requestId"]].set_result({"success": True, "thread_id": "root-ts"})

    monkeypatch.setattr(transport, "_send", send_frame)
    desc = CapabilityDescriptor(1, "discord", "Discord", 2000, False, True, primary_supports_thread,
                                "discord", "chars", supported_ops=("send", "thread_create") if primary_supports_thread else ("send",))
    transport._descriptors_by_platform["slack"] = CapabilityDescriptor(
        1, "slack", "Slack", 4000, False, True, secondary_supports_thread, "slack", "chars",
        supported_ops=("send", "thread_create") if secondary_supports_thread else ("send",))
    pconfig = PlatformConfig(enabled=True)
    adapter = RelayAdapter(pconfig, desc, transport=transport)
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=multiplex, platforms={Platform.RELAY: pconfig})
    runner.config.profile_routes = parse_profile_routes([
        {"platform": "slack", "chat_id": "C1", "profile": "ops"},
    ])
    runner._primary_profile_name = "default"
    runner.adapters = {Platform.RELAY: adapter}
    runner._profile_adapters = {"ops": {}}
    adapter.gateway_runner = runner
    source = SessionSource(Platform.SLACK, "C1", chat_type="group", user_id="alice", scope_id="TB")
    adapter._canonicalize(source)
    assert identity_of(source).runtime_profile == ("ops" if multiplex else "default")
    assert identity_of(source).transport_profile == "default"
    origin = {**source.to_dict(), "conversation": conversation_origin(source, thread_id=None)}
    errors = []
    loop = asyncio.get_running_loop()
    delivery = await asyncio.to_thread(
        _prepare_target_delivery,
        {"id": "digest", "origin": origin, "deliver": "origin", "attach_to_session": True},
        {"platform": "slack", "chat_id": "C1"},
        adapters=runner.adapters, loop=loop, config=runner.config,
        notify_delivery=True, mirror_enabled=True, mirror_text="brief", delivery_errors=errors,
    )
    assert delivery is not None, errors
    assert identity_of(delivery.session_source).runtime_profile == ("ops" if multiplex else "default")
    if not secondary_supports_thread:
        assert not frames
        assert delivery.thread_id is None
        return
    assert len(frames) == 1
    frame = frames[0]
    observed = {
        "platform": frame.get("platform"), "botId": frame.get("botId"),
        "profile": frame["action"]["metadata"].get("profile"),
        "scope_id": frame["action"]["metadata"].get("scope_id"),
        "user_id": frame["action"]["metadata"].get("user_id"),
    }
    assert observed == {"platform": "slack", "botId": "slack-bot", "profile": "ops" if multiplex else None, "scope_id": "TB", "user_id": "alice"}
