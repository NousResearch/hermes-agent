"""The ``/approve`` text fallback on an adapter whose ``send`` caps a split message (Discord's
``MAX_SPLIT_MESSAGES``) or cannot report a lost chunk (the relay): the uncut command must not push the
answer instructions and the deadline past what is delivered (#62359)."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from hermes_agent.gateway.config import PlatformConfig
from hermes_agent.gateway.run_turn_runner import TurnRunner
from hermes_agent.tools import approval
from hermes_agent.tools.approval_gateway_wait import _ApprovalEntry

SESSION = "agent:main:discord:dm:555"
TAIL = "&& rm -rf ~ # TAIL_MARKER"


def _discord():
    from plugins.platforms.discord.adapter import DiscordAdapter

    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    sent = []

    async def fake_send(**kwargs):
        sent.append(kwargs.get("content", ""))
        return SimpleNamespace(id=len(sent))

    channel = SimpleNamespace(send=AsyncMock(side_effect=fake_send))
    adapter._client = SimpleNamespace(get_channel=lambda _chat_id: channel, fetch_channel=AsyncMock())

    async def card_fails(**_kwargs):  # the button card could not be sent: the text fallback answers
        raise RuntimeError("card send failed")
    adapter.send_exec_approval = card_fails
    adapter.pause_typing_for_chat = lambda _c: None
    return adapter, sent


def _render(adapter, command: str) -> None:
    """Drive the production fallback path (``TurnRunner._render_approval_sync``) on ``adapter``."""
    loop = asyncio.new_event_loop()
    turn = object.__new__(TurnRunner)
    turn._ctx = SimpleNamespace(_status_adapter=adapter, _status_chat_id="555", session_key=SESSION,
                                _status_thread_metadata=None, _run_still_current=lambda: True)
    turn._schedule = lambda coro, _m: SimpleNamespace(result=lambda timeout=None: loop.run_until_complete(coro))
    turn._close_native_stream_boundary = lambda *_a, **_k: True
    try:
        turn._render_approval_sync({"command": command, "description": "recursive delete", "request_id": "r1"})
    finally:
        loop.close()


@pytest.fixture()
def pending():
    entry = _ApprovalEntry({"request_id": "r1", "command": "x"})
    with approval._lock:
        approval._gateway_queues[SESSION] = [entry]
    yield entry
    with approval._lock:
        approval._gateway_queues.clear()


def test_discord_fallback_past_the_chunk_cap_keeps_instructions_and_offers_only_deny(pending):
    adapter, sent = _discord()
    _render(adapter, "echo " + "A" * 20000 + " " + TAIL)
    delivered = "".join(sent)
    assert "/deny" in delivered and "it will NOT run" in delivered  # instructions and deadline survive
    assert TAIL not in delivered and "/approve" not in delivered  # an unseen tail is never offered for approval
    assert len(sent) <= adapter.MAX_SPLIT_MESSAGES


def test_relay_fallback_longer_than_one_message_keeps_instructions_and_offers_only_deny(pending):
    from tests.gateway.relay.test_relay_full_command_proof import _Connector, _relay

    connector = _Connector()
    adapter = _relay(connector)

    async def card_fails(**_kwargs):
        raise RuntimeError("card send failed")
    adapter.send_exec_approval, adapter.pause_typing_for_chat = card_fails, lambda _c: None
    _render(adapter, "echo " + "A" * 6000 + " " + TAIL)
    assert len(connector.sends) == 1 and len(connector.sends[0]) <= 2000  # nothing left for the connector to split
    assert "/deny" in connector.sends[0] and "/approve" not in connector.sends[0] and TAIL not in connector.sends[0]


def test_discord_fallback_within_the_chunk_cap_keeps_every_choice(pending):
    adapter, sent = _discord()
    _render(adapter, "echo " + "A" * 5000 + " " + TAIL)
    delivered = "".join(sent)
    assert TAIL in delivered and "/approve" in delivered and "/deny" in delivered
