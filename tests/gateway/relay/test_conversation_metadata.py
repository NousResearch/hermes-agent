"""Legacy continuation must query metadata at the known logical address, even with cold relay caches."""
import asyncio

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize("primary_has_lookup", [True, False])
async def test_legacy_relay_kind_comes_from_the_target_platform_and_scope(primary_has_lookup):
    from gateway.config import PlatformConfig
    from gateway.conversation import resolve_delivery_source
    from gateway.relay.adapter import RelayAdapter
    from gateway.relay.descriptor import CapabilityDescriptor
    from gateway.relay.ws_transport import WebSocketRelayTransport, _event_from_wire
    from gateway.session import build_session_key

    transport = WebSocketRelayTransport(
        "ws://unused.invalid", "discord", "discord-bot",
        identities=[("discord", "discord-bot"), ("slack", "slack-bot")],
    )
    transport._ws = object()
    frames = []

    async def send_frame(frame):
        frames.append(frame)
        action = frame["action"]
        assert action["op"] == "get_chat_info"
        intended = frame.get("platform") == "slack" and action.get("metadata", {}).get("scope_id") == "TB"
        # The socket's default identity and the requested logical identity can answer different kinds.
        transport._pending[frame["requestId"]].set_result(
            {"success": True, "chat_info": {"type": "group" if intended else "dm"}})

    transport._send = send_frame
    primary = CapabilityDescriptor(1, "discord", "Discord", 2000, False, True, True,
        "discord", "chars", supported_ops=("send", "get_chat_info") if primary_has_lookup else ("send",))
    transport._descriptors_by_platform["slack"] = CapabilityDescriptor(
        1, "slack", "Slack", 4000, False, True, True, "slack", "chars", supported_ops=("send", "get_chat_info"))
    adapter = RelayAdapter(PlatformConfig(enabled=True), primary, transport=transport)
    origin = {"platform": "slack", "chat_id": "C1", "scope_id": "TB", "user_id": "alice"}
    seed = await resolve_delivery_source(adapter, "slack", "C1", None, origin)
    reply = _event_from_wire({"text": "continue", "source": {**origin, "chat_type": "group"}})
    assert build_session_key(seed) == build_session_key(reply.source), frames
    assert frames[0]["platform"] == "slack"
    assert frames[0]["botId"] == "slack-bot"
    assert frames[0]["action"]["metadata"]["scope_id"] == "TB"
