"""A relayed Discord interaction that is not a prompt button answers a prompt only as an unadmitted sender.

Discord application commands (``/approve``, ``/deny``) and component presses with a plain custom_id
travel the relay's passthrough lane, which the connector forwards to every gateway of the tenant
without its per-author check. They reach the runner as typed commands, so the runner's typed-answer
checks must treat their sender as not admitted, exactly as a prompt button press is: under
``approvals.approvers: ["*"]`` only the requester, a listed approver or an admin answers.

Real path: authenticated WS reader -> relay adapter -> GatewayRunner with its SQLite authority.
"""

import base64
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from tests.gateway.test_approval_requester_binding import write_approvers

THREAD = "t1"


def _frame(interaction_type, user, data):
    body = {"id": f"i-{user}-{interaction_type}", "type": interaction_type, "channel_id": THREAD, "guild_id": "g1",
            "channel": {"type": 11, "parent_id": "c1"}, "member": {"user": {"id": user, "username": user}},
            "data": data}
    return {"type": "passthrough_forward", "bufferId": f"b-{user}", "forward": {
        "platform": "discord", "botId": "app", "method": "POST", "path": "/interactions", "headers": [],
        "bodyB64": base64.b64encode(json.dumps(body).encode()).decode()}}


@pytest_asyncio.fixture
async def gateway(tmp_path, monkeypatch):
    import gateway.run as gr
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.base import SendResult
    from gateway.relay.adapter import RelayAdapter
    from gateway.relay.ws_transport import WebSocketRelayTransport
    from gateway.session_authority import initialize_session_authority
    from tests.gateway.relay.test_relay_passthrough import _desc
    from tools import approval

    write_approvers(["*"])
    monkeypatch.setattr(gr, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})
    runner = gr.GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    transport = WebSocketRelayTransport("ws://127.0.0.1:1/relay", "discord", "app",
                                        gateway_id="t", upgrade_secret="test-only")
    desc = _desc()
    transport._ws = MagicMock()
    transport._descriptor = desc
    transport._descriptors_by_platform = {"discord": desc}
    adapter = RelayAdapter(PlatformConfig(enabled=True), desc, transport=transport)
    runner.adapters = {Platform.RELAY: adapter}
    adapter.gateway_runner = runner
    authority = await initialize_session_authority(runner, profile_id="default", instance_id="t")
    runner._wire_adapter_handlers(adapter)
    transport.set_passthrough_handler(adapter._on_passthrough)
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="sent"))
    try:
        yield runner, transport, adapter
    finally:
        approval._gateway_queues.clear()
        authority.db.close()


def _thread_session_key(runner):
    from gateway.config import Platform
    from gateway.session import SessionSource

    return runner._session_key_for_source(SessionSource(
        platform=Platform.DISCORD, chat_id=THREAD, chat_type="thread", thread_id=THREAD, user_id="alice",
        delivered_via_upstream_relay=True))


def _alice_approval(session_key):
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools import approval
    from tools.approval_gateway_wait import _ApprovalEntry

    tokens = set_session_vars(platform="discord", user_id="alice", user_name="Alice", chat_type="thread",
                              session_key=session_key)
    try:
        entry = _ApprovalEntry({"command": "rm -rf build"})
    finally:
        clear_session_vars(tokens)
    approval._gateway_queues[session_key] = [entry]
    return entry


@pytest.mark.asyncio
@pytest.mark.parametrize(("interaction_type", "data"), [
    (2, {"name": "approve"}),
    (3, {"custom_id": "/approve"}),
])
async def test_discord_interaction_from_another_member_does_not_approve(gateway, interaction_type, data):
    from tests.waiting import async_wait_until

    runner, transport, adapter = gateway
    entry = _alice_approval(_thread_session_key(runner))
    replies = adapter.send.await_count

    await transport._handle_frame(json.dumps(_frame(interaction_type, "mallory", data)))

    # The refusal reply is the end of the command's handling.
    await async_wait_until(lambda: adapter.send.await_count > replies, timeout=10)
    assert entry.result is None
    assert not entry.event.is_set()


@pytest.mark.asyncio
async def test_discord_approve_command_from_the_requester_still_approves(gateway):
    from tests.waiting import async_wait_until

    runner, transport, _adapter = gateway
    entry = _alice_approval(_thread_session_key(runner))

    await transport._handle_frame(json.dumps(_frame(2, "alice", {"name": "approve"})))

    await async_wait_until(entry.event.is_set, timeout=10)
    assert entry.result == "once"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["slash_confirm", "clarify"])
@pytest.mark.parametrize(("sender", "relayed_interaction", "refused"), [
    ("mallory", True, True), ("alice", True, False), ("mallory", False, False)])
async def test_typed_answer_from_a_discord_interaction_is_unadmitted(gateway, kind, sender, relayed_interaction,
                                                                     refused):
    """The runner's typed slash-confirm and clarify checks: a relayed Discord interaction's sender is not
    admitted, a relayed message's sender is (``approvers: ["*"]``, any admitted member answers)."""
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent
    from gateway.session import SessionSource
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools import clarify_gateway, slash_confirm

    runner, _transport, _adapter = gateway
    session_key = _thread_session_key(runner)
    tokens = set_session_vars(platform="discord", user_id="alice", user_name="Alice", chat_type="thread",
                              session_key=session_key)
    try:
        if kind == "slash_confirm":
            slash_confirm.register(session_key, "cf1", "reload-mcp", AsyncMock(return_value="done"))
        else:
            clarify_gateway.register("cl1", session_key, "Which one?", ["red", "blue"])
    finally:
        clear_session_vars(tokens)
    event = MessageEvent(text="red", source=SessionSource(
        platform=Platform.DISCORD, chat_id=THREAD, chat_type="thread", thread_id=THREAD, user_id=sender,
        delivered_via_upstream_relay=True))
    if relayed_interaction:
        event._unadmitted_author = True
    try:
        if kind == "slash_confirm":
            refusal = runner._refused_slash_confirm(event, session_key, "cf1")
        else:
            refusal = runner._refused_clarify(event, "cl1")
    finally:
        slash_confirm.clear(session_key)
        clarify_gateway.clear_session(session_key)

    assert (refusal is not None) is refused
