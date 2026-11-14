"""Inbound Signal attachments are fetched only for senders the gateway authorizes.

Drives the real adapter with the real GatewayRunner authorization callback wired as startup wires it;
only signal-cli's getAttachment RPC is stubbed.
"""
import base64

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig

OWNER, STRANGER = "+15550000001", "+15550000002"


@pytest.fixture
def signal(tmp_path, monkeypatch):
    from gateway.pairing import PairingStore
    from gateway.run import GatewayRunner
    from plugins.platforms.signal.adapter import SignalAdapter

    for key in ("GATEWAY_ALLOWED_USERS", "GATEWAY_ALLOW_ALL_USERS", "SIGNAL_ALLOW_ALL_USERS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SIGNAL_ALLOWED_USERS", OWNER)
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "")
    monkeypatch.setattr("gateway.pairing.PAIRING_DIR", tmp_path / "pairing")

    adapter = SignalAdapter(PlatformConfig(enabled=True, extra={"http_url": "http://127.0.0.1:1", "account": "+15559999999"}))
    rpc_calls = []

    async def rpc(method, params, rpc_id=None):
        rpc_calls.append(method)
        return {"data": base64.b64encode(b"%PDF-1.4 fake").decode()}

    adapter._rpc = rpc
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.SIGNAL: adapter.config})
    runner.adapters = {Platform.SIGNAL: adapter}
    runner.pairing_store = PairingStore()
    events = []

    async def capture(event):
        events.append(event)
    adapter.handle_message = capture

    async def dm(sender, text="see attached"):
        before = len(events)
        await adapter._handle_envelope({"envelope": {
            "sourceNumber": sender, "sourceName": "someone", "timestamp": 1700000000000,
            "dataMessage": {"timestamp": 1700000000000, "message": text, "attachments": [
                {"id": "att1.pdf", "contentType": "application/pdf", "size": 13}]}}})
        return events[before] if len(events) > before else None

    return adapter, runner, dm, rpc_calls


@pytest.mark.asyncio
async def test_attachments_are_fetched_only_for_authorized_senders(signal):
    adapter, runner, dm, rpc_calls = signal
    adapter.set_authorization_check(runner._make_adapter_auth_check(Platform.SIGNAL))

    denied = await dm(STRANGER)
    assert rpc_calls == []
    assert denied.media_urls == []
    assert denied.source.user_id == STRANGER  # still dispatched so the runner can deny/pair
    assert await dm(STRANGER, text="") is None  # attachment-only from a stranger is contentless
    assert rpc_calls == []

    allowed = await dm(OWNER)
    assert rpc_calls == ["getAttachment"]
    assert len(allowed.media_urls) == 1


@pytest.mark.asyncio
async def test_unwired_authorization_check_fetches_nothing(signal):
    _adapter, _runner, dm, rpc_calls = signal

    event = await dm(OWNER)

    assert rpc_calls == []
    assert event.media_urls == []


@pytest.mark.asyncio
async def test_bot_users_sender_in_loop_guard_cooldown_gets_no_attachment_fetch(signal):
    """An allowlisted ``bot_users`` sender is judged as a bot: while its chat cools down, nothing is fetched."""
    from gateway.bot_loop_guard import BotLoopGuard
    adapter, runner, dm, rpc_calls = signal
    adapter.config.extra["bot_users"] = [OWNER]
    runner._bot_loop_guard = guard = BotLoopGuard()
    guard.blocked = lambda conversation: True
    adapter.set_authorization_check(runner._make_adapter_auth_check(Platform.SIGNAL))

    event = await dm(OWNER)

    assert event.source.is_bot
    assert rpc_calls == []
    assert event.media_urls == []
