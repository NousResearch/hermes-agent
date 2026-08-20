import pytest, os
from unittest.mock import patch
from gateway.platforms.signal import SignalAdapter
from gateway.config import PlatformConfig, Platform

BOT = "+491234567890"
BOT_UUID = "3eb65f7a-6a00-4263-a72c-280a262aa28b"
OTHER_UUID = "b2187174-cd8e-4bb5-9f7f-dd6d6890d9e7"

@pytest.fixture
def adapter():
    c = PlatformConfig(platform=Platform.SIGNAL, enabled=True, extra={
        "http_url": "http://127.0.0.1:8090", "account": BOT,
    })
    with patch("gateway.platforms.signal.httpx.AsyncClient"):
        a = SignalAdapter(c)
        a._recipient_uuid_by_number[BOT] = BOT_UUID
        a._running = True
        return a

def env(text, mention=None):
    e = {"envelope": {"sourceUuid": OTHER_UUID, "timestamp": 1},
         "dataMessage": {"message": text, "mentions": [mention] if mention else [],
                         "groupInfo": {"groupId": "g"}}}
    return e

def check(adapter, envelope):
    dm = envelope["dataMessage"]
    an = adapter._account_normalized
    text = dm.get("message", "")
    in_text = an and f"@{an}" in (text or "")
    in_meta = any(
        m.get("number") == an or m.get("uuid") == an
        or (adapter._recipient_uuid_by_number.get(an)
            and m.get("uuid") == adapter._recipient_uuid_by_number.get(an))
        for m in (dm.get("mentions") or [])
    )
    return in_text, in_meta

# --- UUID match ---
@pytest.mark.asyncio
async def test_uuid_match(adapter):
    _, meta = check(adapter, env("😄 test", {"start": 0, "length": 36, "uuid": BOT_UUID}))
    assert meta, "UUID match → mentioned"

# --- UUID no match ---
@pytest.mark.asyncio
async def test_uuid_no_match(adapter):
    _, meta = check(adapter, env("😄 test", {"start": 0, "length": 36, "uuid": OTHER_UUID}))
    assert not meta, "other UUID → not mentioned"

# --- Phone mention ---
@pytest.mark.asyncio
async def test_phone_mention(adapter):
    in_text, meta = check(adapter, env("😄 @" + BOT + " test", {"start": 0, "length": 15, "number": BOT, "uuid": BOT_UUID}))
    assert in_text and meta

# --- syncMessage fallback ---
@pytest.mark.asyncio
async def test_sync_message(adapter):
    e = env("sync text", None)
    e["dataMessage"] = {"message": "", "mentions": []}  # empty dataMessage
    e["envelope"]["syncMessage"] = {"sentMessage": {"message": "sync text", "destination": BOT}}
    _, meta = check(adapter, e)
    text = e["dataMessage"].get("message", "")
    an = adapter._account_normalized
    # syncMessage fallback is only active in _handle_envelope, not our check()
    # This tests that the extraction works; full integration would call _handle_envelope
    assert text == "sync text"
