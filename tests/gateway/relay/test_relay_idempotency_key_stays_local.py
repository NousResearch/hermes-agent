"""The delivery ledger's ``idempotency_key`` never reaches the Relay connector.

``send_final_ledgered`` stamps every send of a ledgered final with the obligation id as
``metadata["idempotency_key"]`` so an adapter with platform-side dedup (Discord's message nonce)
can recognise a re-send. That key is gateway-internal: the Relay adapter copies ``metadata`` into
its wire frames, and the connector contract has no such field, so ``_with_scope`` (the one place
every chat-addressed frame's metadata is built) drops it while keeping everything else.
"""

import os

import pytest

from gateway import delivery_ledger as dl
from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.relay.adapter import RelayAdapter
from gateway.session import SessionSource
from tests.gateway.relay.stub_connector import StubConnector
from tests.gateway.relay.test_relay_live_cards import make_desc


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(dl, "_db_path", lambda: home / "state.db")
    monkeypatch.setattr(dl, "_owner_stamp", lambda: (os.getpid(), 202))
    monkeypatch.setattr(dl, "ledger_enabled", lambda config=None: True)


@pytest.mark.asyncio
async def test_a_ledgered_final_through_relay_carries_no_idempotency_key_on_the_wire(ledger):
    stub = StubConnector(make_desc())
    adapter = RelayAdapter(PlatformConfig(), make_desc(), transport=stub)
    source = SessionSource(platform=Platform.RELAY, chat_id="C1", chat_type="channel")
    event = MessageEvent(text="question", source=source, message_id="m1")

    result, _ = await adapter.send_final_ledgered(
        event, "agent:main:relay:channel:C1", "the answer", {"thread_id": "171.001"}, reply_to=None)

    with dl._connect() as conn:
        (state,) = conn.execute("SELECT state FROM delivery_obligations").fetchone()
    assert result.success and state == "delivered"  # the key was in play: the send was ledgered
    [frame] = [f for f in stub.sent if f.get("op") == "send"]
    assert frame["content"] == "the answer"
    assert "idempotency_key" not in frame["metadata"]
    assert frame["metadata"]["thread_id"] == "171.001"
