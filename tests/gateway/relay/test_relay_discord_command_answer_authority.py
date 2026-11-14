"""A relayed Discord interaction answers a prompt as admitted only for a sender the connector admitted.

Discord application commands (``/approve``, ``/deny``) and component presses with a plain custom_id
travel the relay's passthrough lane, which the connector forwards to every gateway of the tenant
without its per-author check, and reach the runner as typed commands, possibly rebuilt (``/queue``,
``/steer``), rewritten by a plugin or merged. The runner's typed-answer checks judge the sender, not
the event: a relayed Discord sender whose messages never reached this gateway is not admitted, so
under ``approvals.approvers: ["*"]`` only the requester, a listed approver or an admin answers.
Any other interaction from such a sender is dropped at the relay intake unless they are an admin.

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


@pytest.fixture(autouse=True)
def _no_admitted_authors(monkeypatch):
    from gateway.relay import admitted_authors

    monkeypatch.setattr(admitted_authors, "_admitted", type(admitted_authors._admitted)())


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
@pytest.mark.parametrize(("sender", "delivered_here", "refused"), [
    ("mallory", False, True), ("alice", False, False), ("mallory", True, False)])
async def test_typed_answer_from_an_unadmitted_discord_sender_is_refused(gateway, kind, sender, delivered_here,
                                                                         refused):
    """The runner's typed slash-confirm and clarify checks: a relayed Discord sender is admitted once
    the connector delivered their message here (``approvers: ["*"]``, any admitted member answers)."""
    from gateway.relay.admitted_authors import note_delivered
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
    if delivered_here:
        note_delivered(event.source)
    try:
        if kind == "slash_confirm":
            refusal = runner._refused_slash_confirm(event, session_key, "cf1")
        else:
            refusal = runner._refused_clarify(event, "cl1")
    finally:
        slash_confirm.clear(session_key)
        clarify_gateway.clear_session(session_key)

    assert (refusal is not None) is refused


@pytest.mark.asyncio
async def test_a_pre_gateway_dispatch_rewrite_keeps_the_interaction_unadmitted(gateway, monkeypatch):
    """A plugin hook that rewrites the text (here: to the same text) replaces the event with a copy;
    the copy still answers only as the unadmitted sender."""
    from hermes_cli.plugins import get_plugin_manager
    from tests.waiting import async_wait_until

    async def rewrite(event, **_kwargs):
        return {"action": "rewrite", "text": event.text}

    manager = get_plugin_manager()
    monkeypatch.setattr(manager, "_discovered", True)
    monkeypatch.setattr(manager, "_hooks", {**manager._hooks, "pre_gateway_dispatch": [rewrite]})
    runner, transport, adapter = gateway
    entry = _alice_approval(_thread_session_key(runner))
    replies = adapter.send.await_count

    await transport._handle_frame(json.dumps(_frame(2, "mallory", {"name": "approve"})))

    await async_wait_until(lambda: adapter.send.await_count > replies, timeout=10)
    assert entry.result is None


def test_admission_follows_the_sender_through_rebuilds_and_merges():
    """``/queue`` and ``/steer`` rebuild the event from its source, a plugin rewrite copies it, a merge
    of two senders clears its user: none of them changes who the connector admitted."""
    import dataclasses

    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent
    from gateway.relay.admitted_authors import answer_admitted, note_delivered
    from gateway.session import SessionSource

    def relayed(user, platform=Platform.DISCORD):
        return SessionSource(platform=platform, chat_id=THREAD, chat_type="thread", user_id=user,
                             delivered_via_upstream_relay=True)

    note_delivered(relayed("alice"))
    interaction = MessageEvent(text="/queue /approve", source=relayed("mallory"))
    rebuilt = MessageEvent(text="/approve", source=interaction.source)
    assert [answer_admitted(e) for e in (interaction, rebuilt, dataclasses.replace(rebuilt, text="/approve"))] == [
        False, False, False]
    merged = MessageEvent(text="hi", source=relayed("alice"))
    merged.absorb_turn_input(interaction)
    assert merged.sender_id() == "" and answer_admitted(merged) is False
    assert answer_admitted(MessageEvent(text="/approve", source=relayed("alice"))) is True
    # Other relayed platforms and native adapters are admitted by their own lanes, unchanged.
    assert answer_admitted(MessageEvent(text="/approve", source=relayed("mallory", Platform.TELEGRAM))) is True
    assert answer_admitted(MessageEvent(text="/approve", source=SessionSource(
        platform=Platform.DISCORD, chat_id=THREAD, chat_type="thread", user_id="mallory"))) is True


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/queue /approve", "/steer /approve"])
async def test_a_follow_up_rebuilt_from_a_discord_interaction_does_not_approve(gateway, monkeypatch, command):
    """A busy session turns ``/queue`` / a fallback ``/steer`` into a rebuilt follow-up event. The relay
    intake drops these from an unadmitted sender; past it, the runner's answer check still refuses."""
    from tests.waiting import async_wait_until

    runner, transport, adapter = gateway
    key = _thread_session_key(runner)
    entry = _alice_approval(key)
    monkeypatch.setattr(runner, "_is_session_running", lambda _session_key: True)
    monkeypatch.setattr(adapter, "_interaction_admitted", lambda _event: True)
    replies = adapter.send.await_count

    await transport._handle_frame(json.dumps(_frame(3, "mallory", {"custom_id": command})))

    # The queued acknowledgement, then the rebuilt /approve's own answer.
    await async_wait_until(lambda: adapter.send.await_count >= replies + 2 or entry.event.is_set(), timeout=15)
    assert entry.result is None


def _spy_dispatch(adapter, monkeypatch):
    calls = []
    original = adapter.handle_message

    async def spy(event):
        calls.append(event.text)
        return await original(event)

    monkeypatch.setattr(adapter, "handle_message", spy)
    return calls


UNADMITTED_INTERACTIONS = [
    (2, {"name": "status"}),
    (2, {"name": "title", "options": [{"type": 3, "name": "name", "value": "renamed-by-mallory"}]}),
    (2, {"name": "queue", "options": [{"type": 3, "name": "prompt", "value": "/approve"}]}),
    (3, {"custom_id": "/status"}),
    (5, {"custom_id": "form", "components": [{"type": 1, "components": [
        {"type": 4, "custom_id": "q", "value": "summarize the thread"}]}]}),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("interaction_type", "data"), UNADMITTED_INTERACTIONS)
async def test_an_unadmitted_discord_interaction_reaches_no_command(gateway, monkeypatch, interaction_type, data):
    """The connector forwards every interaction to every gateway of the tenant: one from a sender it never
    delivered a message from here is dropped before the runner, without a reply."""
    _runner, transport, adapter = gateway
    dispatched = _spy_dispatch(adapter, monkeypatch)
    replies = adapter.send.await_count

    await transport._handle_frame(json.dumps(_frame(interaction_type, "mallory", data)))

    assert dispatched == []
    assert adapter.send.await_count == replies


@pytest.mark.asyncio
@pytest.mark.parametrize("sender", ["alice", "owner-admin"])
async def test_a_discord_interaction_from_an_admitted_sender_or_an_admin_runs(gateway, monkeypatch, sender):
    """A member whose message the connector delivered here, or an admin of the scope, still runs commands."""
    import gateway.run as gr
    from gateway.config import Platform, PlatformConfig
    from gateway.relay.admitted_authors import note_delivered
    from gateway.session import SessionSource
    from tests.waiting import async_wait_until

    runner, transport, adapter = gateway
    runner.config.platforms[Platform.DISCORD] = PlatformConfig(
        enabled=True, extra={"group_allow_admin_from": ["owner-admin"], "user_allowed_commands": ["status"]})
    monkeypatch.setattr(gr, "_gateway_runner_ref", lambda: runner)
    note_delivered(SessionSource(platform=Platform.DISCORD, chat_id=THREAD, user_id="alice",
                                 delivered_via_upstream_relay=True))
    dispatched = _spy_dispatch(adapter, monkeypatch)
    replies = adapter.send.await_count

    await transport._handle_frame(json.dumps(_frame(2, sender, {"name": "status"})))

    assert dispatched == ["/status"]
    await async_wait_until(lambda: adapter.send.await_count > replies, timeout=10)


@pytest.mark.asyncio
@pytest.mark.parametrize("data", [{"name": "approve"}, {"name": "deny"}])
async def test_an_unadmitted_discord_prompt_answer_still_reaches_the_runner(gateway, monkeypatch, data):
    """``/approve`` and ``/deny`` pass the intake: the runner answers them as the unadmitted sender."""
    _runner, transport, adapter = gateway
    dispatched = _spy_dispatch(adapter, monkeypatch)

    await transport._handle_frame(json.dumps(_frame(2, "mallory", data)))

    assert dispatched == [f"/{data['name']}"]
