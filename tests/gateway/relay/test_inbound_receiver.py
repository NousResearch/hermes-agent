"""Unit tests for gateway/relay/inbound_receiver.py.

Covers the verify-then-dispatch core (handle_raw): a correctly-signed message
delivery is verified + dispatched; an interrupt delivery routes to the interrupt
handler; unsigned/tampered/expired/no-key deliveries are rejected 401; malformed
JSON is 400. Signatures are produced with the SAME auth primitives the connector
uses (gateway/relay/auth.py sign), so this exercises the real verify path.
"""

from __future__ import annotations

import json
import time

import pytest

from gateway.relay.auth import sign
from gateway.relay.inbound_receiver import InboundDeliveryReceiver

_KEY = "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff"


def _signed(body_obj: dict, key: str = _KEY, ts: int | None = None) -> tuple[bytes, str, str]:
    """Serialize compactly (as the connector's JSON.stringify does), sign it."""
    body = json.dumps(body_obj, separators=(",", ":"))
    raw = body.encode("utf-8")
    t = ts if ts is not None else int(time.time())
    return raw, str(t), sign(f"{t}.{body}", key)


def _receiver(**kw):
    received: list = []
    interrupts: list = []

    async def on_message(ev):
        received.append(ev)

    async def on_interrupt(sk, chat):
        interrupts.append((sk, chat))

    r = InboundDeliveryReceiver(
        delivery_key_verify_list=lambda: [_KEY],
        on_message=on_message,
        on_interrupt=on_interrupt,
        **kw,
    )
    return r, received, interrupts


@pytest.mark.asyncio
async def test_valid_message_delivery_dispatched():
    r, received, _ = _receiver()
    raw, ts, sig = _signed(
        {
            "type": "message",
            "event": {
                "text": "hello",
                "message_type": "text",
                "source": {"platform": "discord", "chat_id": "chan1", "chat_type": "group", "guild_id": "guildA"},
            },
        }
    )
    status, body = await r.handle_raw(raw_body=raw, timestamp=ts, signature=sig, is_interrupt=False)
    assert status == 200 and body == {"ok": True}
    assert len(received) == 1
    assert received[0].text == "hello"
    assert received[0].source.guild_id == "guildA"


@pytest.mark.asyncio
async def test_valid_interrupt_delivery_routes_to_interrupt_handler():
    r, _, interrupts = _receiver()
    raw, ts, sig = _signed({"type": "interrupt", "session_key": "agent:main:discord:group:c:u", "reason": "stop"})
    status, _ = await r.handle_raw(raw_body=raw, timestamp=ts, signature=sig, is_interrupt=True)
    assert status == 200
    assert interrupts and interrupts[0][0] == "agent:main:discord:group:c:u"


@pytest.mark.asyncio
async def test_tampered_body_rejected_401():
    r, received, _ = _receiver()
    raw, ts, sig = _signed({"type": "message", "event": {"text": "x", "source": {"chat_id": "c"}}})
    status, _ = await r.handle_raw(raw_body=raw + b" ", timestamp=ts, signature=sig, is_interrupt=False)
    assert status == 401
    assert received == []


@pytest.mark.asyncio
async def test_unsigned_rejected_401():
    r, _, _ = _receiver()
    raw, _, _ = _signed({"type": "message", "event": {"text": "x", "source": {"chat_id": "c"}}})
    status, _ = await r.handle_raw(raw_body=raw, timestamp=None, signature=None, is_interrupt=False)
    assert status == 401


@pytest.mark.asyncio
async def test_expired_timestamp_rejected_401():
    r, _, _ = _receiver(max_skew_seconds=300)
    raw, _, sig = _signed({"type": "message", "event": {"text": "x", "source": {"chat_id": "c"}}}, ts=1)
    # ts=1 (1970) is far outside the 300s window vs now.
    status, _ = await r.handle_raw(raw_body=raw, timestamp="1", signature=sig, is_interrupt=False)
    assert status == 401


@pytest.mark.asyncio
async def test_wrong_key_rejected_401():
    r, _, _ = _receiver()
    other = "ffeeddccbbaa99887766554433221100ffeeddccbbaa99887766554433221100"
    raw, ts, sig = _signed({"type": "message", "event": {"text": "x", "source": {"chat_id": "c"}}}, key=other)
    status, _ = await r.handle_raw(raw_body=raw, timestamp=ts, signature=sig, is_interrupt=False)
    assert status == 401


@pytest.mark.asyncio
async def test_no_delivery_key_fails_closed_401():
    async def on_message(ev):
        pass

    r = InboundDeliveryReceiver(delivery_key_verify_list=lambda: [], on_message=on_message)
    raw, ts, sig = _signed({"type": "message", "event": {"text": "x", "source": {"chat_id": "c"}}})
    status, _ = await r.handle_raw(raw_body=raw, timestamp=ts, signature=sig, is_interrupt=False)
    assert status == 401


@pytest.mark.asyncio
async def test_rotation_secondary_key_accepted():
    new = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    received: list = []

    async def on_message(ev):
        received.append(ev)

    # Connector still signs with the OLD key (secondary); verify list has both.
    r = InboundDeliveryReceiver(
        delivery_key_verify_list=lambda: [new, _KEY], on_message=on_message
    )
    raw, ts, sig = _signed({"type": "message", "event": {"text": "x", "source": {"chat_id": "c"}}}, key=_KEY)
    status, _ = await r.handle_raw(raw_body=raw, timestamp=ts, signature=sig, is_interrupt=False)
    assert status == 200 and len(received) == 1


@pytest.mark.asyncio
async def test_malformed_json_after_valid_signature_is_400():
    r, _, _ = _receiver()
    # Sign a non-JSON body so the signature passes but json.loads fails.
    raw = b"not json at all"
    ts = str(int(time.time()))
    sig = sign(f"{ts}.{raw.decode()}", _KEY)
    status, body = await r.handle_raw(raw_body=raw, timestamp=ts, signature=sig, is_interrupt=False)
    assert status == 400
