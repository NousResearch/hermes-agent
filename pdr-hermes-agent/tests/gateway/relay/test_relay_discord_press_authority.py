"""A relayed Discord button press answers a prompt only for an admitted sender, its requester or an admin.

The connector forwards a Discord component press to every gateway of the tenant without the
per-author check it runs on messages, so under the default ``approvals.approvers: ["*"]`` a press
from any guild member, linked or not, settled another user's exec approval, slash confirmation or
clarify question. A Discord member whose messages the connector delivers here is admitted and keeps
"any member answers"; so does the inbound lane (Telegram/Slack/WhatsApp presses), which the connector
narrows to admitted authors.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from tests.gateway.test_approval_requester_binding import write_approvers

SESSION = "agent:main:discord:group:c1"


@pytest.fixture(autouse=True)
def _no_admitted_authors(monkeypatch):
    from gateway.relay import admitted_authors

    monkeypatch.setattr(admitted_authors, "_admitted", type(admitted_authors._admitted)())


@pytest.fixture
def relay(monkeypatch):
    import gateway.run as gateway_run
    from gateway.relay.adapter import RelayAdapter
    from tests.gateway.relay.stub_connector import StubConnector
    from tests.gateway.relay.test_relay_interactive import make_desc
    from tools import approval, clarify_gateway, slash_confirm

    write_approvers(["*"])
    admins = {"group_allow_admin_from": ["owner-admin"]}
    monkeypatch.setattr(gateway_run, "_gateway_runner_ref", lambda: SimpleNamespace(config=GatewayConfig(
        platforms={p: PlatformConfig(enabled=True, extra=dict(admins)) for p in (Platform.DISCORD, Platform.TELEGRAM)})))
    stub = StubConnector(make_desc())
    adapter = RelayAdapter(PlatformConfig(), make_desc(), transport=stub)
    yield adapter, stub
    approval._gateway_queues.clear()
    slash_confirm.clear(SESSION)
    clarify_gateway.clear_session(SESSION)


def _no_admin_list(monkeypatch):
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_gateway_runner_ref", lambda: SimpleNamespace(config=GatewayConfig(
        platforms={p: PlatformConfig(enabled=True) for p in (Platform.DISCORD, Platform.TELEGRAM)})))


async def _raise_as_alice(adapter, stub, kind):
    """Raise *kind* inside Alice's turn and post its card; returns (prompt_id, option, answered?)."""
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools import approval, clarify_gateway, slash_confirm
    from tools.approval_gateway_wait import _ApprovalEntry

    ran = []
    clarify = None

    async def handler(choice):
        ran.append(choice)
        return "done"

    tokens = set_session_vars(platform="discord", user_id="alice", user_name="Alice", chat_type="group",
                              session_key=SESSION)
    try:
        if kind == "exec_approval":
            entry = _ApprovalEntry({"command": "rm -rf build"})
            approval._gateway_queues[SESSION] = [entry]
            prompt_id = adapter._mint_prompt("exec_approval", {
                "session_key": SESSION, "request_id": entry.data["request_id"], "chat_id": "c1"})
            return prompt_id, "once", lambda: entry.result == "once"
        if kind == "slash_confirm":
            slash_confirm.register(SESSION, "cf1", "reload-mcp", handler)
        else:
            clarify = clarify_gateway.register("cl1", SESSION, "Which one?", ["red", "blue"])
    finally:
        clear_session_vars(tokens)
    if kind == "slash_confirm":
        await adapter.send_slash_confirm("c1", "Confirm", "reload?", SESSION, "cf1")
        return stub.sent[-1]["prompt_id"], "always", lambda: bool(ran)
    await adapter.send_clarify("c1", "Which one?", ["red", "blue"], "cl1", SESSION)
    return stub.sent[-1]["prompt_id"], "c0", lambda: clarify is not None and clarify.event.is_set()


async def _discord_press(adapter, clicker, prompt_id, option):
    from gateway.relay.ws_transport import PassthroughForward

    body = json.dumps({"type": 3, "guild_id": "guild", "channel_id": "c1", "member": {"user": {"id": clicker}},
                       "data": {"custom_id": f"hp1:{prompt_id}:{option}"}}).encode()
    await adapter._on_passthrough(PassthroughForward(platform="discord", bot_id="bot", method="POST", path="/",
                                                     headers=[], body=body))


async def _inbound_press(adapter, clicker, prompt_id, option):
    from gateway.relay.ws_transport import _event_from_wire

    await adapter._on_inbound(_event_from_wire({
        "text": f"/{option}", "message_type": "command",
        "source": {"platform": "telegram", "user_id": clicker, "chat_id": "c1", "chat_type": "group"},
        "prompt_response": {"prompt_id": prompt_id, "option_id": option}}))


KINDS = ["exec_approval", "slash_confirm", "clarify"]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_discord_press_from_another_member_leaves_the_prompt_pending(relay, kind, monkeypatch):
    adapter, stub = relay
    _no_admin_list(monkeypatch)  # the default setup: no admin list, approvers ["*"]
    prompt_id, option, answered = await _raise_as_alice(adapter, stub, kind)

    await _discord_press(adapter, "mallory", prompt_id, option)

    assert answered() is False
    assert prompt_id in adapter._pending_prompts
    await _discord_press(adapter, "alice", prompt_id, option)
    assert answered() is True


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("clicker", ["owner-admin", "alice", "mallory"])
async def test_discord_press_with_an_admin_list_follows_the_slash_policy(relay, kind, clicker):
    """With ``group_allow_admin_from`` set, answering an approval or slash confirm is admin-only,
    for the requester too (their typed ``/approve`` is refused); a clarify answer runs no command."""
    adapter, stub = relay
    prompt_id, option, answered = await _raise_as_alice(adapter, stub, kind)
    acks = []
    adapter._send_lifecycle_ack = lambda chat_id, text, meta: acks.append(text)

    await _discord_press(adapter, clicker, prompt_id, option)

    expected = clicker == "owner-admin" or (clicker == "alice" and kind == "clarify")
    assert answered() is expected
    # The refusal names the check that failed: the slash policy (admin-only) before the requester.
    if kind != "clarify" and clicker != "owner-admin":
        assert "admin-only" in acks[0] and "alice" not in acks[0].lower()
    elif not expected:
        assert "alice" in acks[0].lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_inbound_lane_press_from_another_member_still_answers(relay, kind, monkeypatch):
    adapter, stub = relay
    _no_admin_list(monkeypatch)
    prompt_id, option, answered = await _raise_as_alice(adapter, stub, kind)

    await _inbound_press(adapter, "bob", prompt_id, option)

    assert answered() is True


def test_unadmitted_card_answer_settles_nothing_for_another_member(relay):
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools import approval
    from tools.approval_gateway_wait import _ApprovalEntry

    tokens = set_session_vars(platform="discord", user_id="alice", user_name="Alice", chat_type="group",
                              session_key=SESSION)
    try:
        entry = _ApprovalEntry({"command": "rm -rf build"})
    finally:
        clear_session_vars(tokens)
    approval._gateway_queues[SESSION] = [entry]
    request_id = entry.data["request_id"]

    assert approval.resolve_gateway_approval_card(SESSION, "once", request_id, actor_id="mallory",
                                                  admitted=False) == 0
    assert entry.result is None
    assert approval.resolve_gateway_approval_card(SESSION, "once", request_id, actor_id="alice",
                                                  admitted=False) == 1
    assert entry.result == "once"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_discord_press_from_a_member_whose_messages_arrive_here_still_answers(relay, kind, monkeypatch):
    """``approvers: ["*"]``: a teammate the connector admitted (their Discord message reached this
    gateway) answers another member's prompt with a button, as before."""
    from gateway.relay.ws_transport import _event_from_wire

    adapter, stub = relay
    _no_admin_list(monkeypatch)
    adapter.handle_message = AsyncMock()  # the plain message is only the connector's admission here
    await adapter._on_inbound(_event_from_wire({"text": "hi", "message_id": "m-bob", "source": {
        "platform": "discord", "user_id": "bob", "chat_id": "c1", "chat_type": "group"}}))
    prompt_id, option, answered = await _raise_as_alice(adapter, stub, kind)

    await _discord_press(adapter, "bob", prompt_id, option)

    assert answered() is True

