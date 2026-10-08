"""Restart durability contract for accepted webhook deliveries."""

import asyncio
import hashlib
import hmac
import json
import threading

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms import webhook_inbox
from gateway.platforms.webhook import WebhookAdapter, _WebhookDeliveryIdentity

SECRET = "fake-signing-secret"


def _config(**route_extra):
    route = {
        "secret": SECRET,
        "deliver": "telegram",
        "deliver_extra": {"chat_id": "fake-chat"},
        "prompt": "Alert: {message}",
        **route_extra,
    }
    return PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "routes": {"alerts": route}})


def _signed(delivery_id, message="fake payload"):
    body = json.dumps({"message": message}).encode()
    headers = {
        "Content-Type": "application/json",
        "X-GitHub-Delivery": delivery_id,
        "X-Hub-Signature-256": "sha256=" + hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest(),
    }
    return body, headers


def _adapter(config, received):
    adapter = WebhookAdapter(config)

    async def capture(event):
        received.append(event)

    adapter.handle_message = capture
    return adapter


async def _post(adapter, delivery_id, message="fake payload"):
    app = web.Application()
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
    body, headers = _signed(delivery_id, message)
    async with TestClient(TestServer(app)) as client:
        response = await client.post("/webhooks/alerts", data=body, headers=headers)
        payload = await response.json() if response.content_type == "application/json" else None
        return response.status, payload


async def _settle(adapter):
    """Let the hand-off and its background checkpoint finish (a live gateway keeps running)."""
    for _ in range(20):
        pending = [t for t in adapter._background_tasks if not t.done()]
        if not pending:
            return
        await asyncio.gather(*pending, return_exceptions=True)


def _inbox_states(home):
    root = home / "state" / webhook_inbox.INBOX_DIRNAME
    return sorted(json.loads(p.read_text())["state"] for p in root.glob("*.json"))


@pytest.mark.asyncio
async def test_restart_preserves_idempotency_and_delivery_envelope(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    received = []
    first = _adapter(_config(), received)
    assert (await _post(first, "fake-delivery-1"))[0] == 202
    await _settle(first)
    assert _inbox_states(tmp_path) == ["started"]

    restarted = _adapter(_config(), received)
    restarted._replay_pending_deliveries()
    status, body = await _post(restarted, "fake-delivery-1")
    assert (status, body["status"]) == (200, "duplicate")

    assert len(received) == 1
    identity = _WebhookDeliveryIdentity.from_parts(None, "alerts", "fake-delivery-1")
    assert identity in restarted._seen_deliveries
    envelope = restarted._delivery_info[identity.session_chat_id]
    assert envelope["deliver"] == "telegram"
    assert envelope["deliver_extra"] == {"chat_id": "fake-chat"}


@pytest.mark.asyncio
async def test_crash_after_admission_before_dispatch_replays_exactly_once(tmp_path, monkeypatch):
    """seen persisted -> crash -> no dispatch: the accepted work must not be lost to the
    idempotency tombstone. The pending record carries the payload/prompt, the restart
    replays it, the sender's retry is a duplicate, and a second restart replays nothing."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    received = []
    crashed = _adapter(_config(), received)

    def die_before_dispatch(*_args, **_kwargs):
        raise RuntimeError("process killed between admission and dispatch")

    monkeypatch.setattr(crashed, "_run_admitted", die_before_dispatch)
    status, _ = await _post(crashed, "fake-delivery-2", "lost?")
    assert status == 500
    assert received == []
    assert _inbox_states(tmp_path) == ["pending"]

    restarted = _adapter(_config(), received)
    status, body = await _post(restarted, "fake-delivery-2", "lost?")  # sender retry before replay
    assert (status, body["status"]) == (200, "duplicate")
    restarted._replay_pending_deliveries()
    await _settle(restarted)
    assert [e.text for e in received] == ["Alert: lost?"]
    assert received[0].message_id == "fake-delivery-2"
    assert _inbox_states(tmp_path) == ["started"]

    again = _adapter(_config(), received)
    again._replay_pending_deliveries()
    await _settle(again)
    assert len(received) == 1
    assert (await _post(again, "fake-delivery-2", "lost?"))[1]["status"] == "duplicate"


@pytest.mark.asyncio
async def test_admission_write_failure_is_retryable_not_a_silent_duplicate(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    received = []
    adapter = _adapter(_config(), received)
    real_write = adapter._inbox.write

    def disk_full(_record):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(adapter._inbox, "write", disk_full)
    status, body = await _post(adapter, "fake-delivery-3")
    assert status == 503
    assert received == []

    monkeypatch.setattr(adapter._inbox, "write", real_write)
    status, body = await _post(adapter, "fake-delivery-3")
    assert (status, body["status"]) == (202, "accepted")
    await _settle(adapter)
    assert len(received) == 1


@pytest.mark.asyncio
async def test_per_delivery_checkpoint_is_constant_size_and_off_the_event_loop(tmp_path, monkeypatch):
    """Each delivery writes only its own small record, on a worker thread. A whole-state
    snapshot per delivery grew linearly with the live set (O(n^2) bytes per TTL window)
    and fsynced on the aiohttp loop."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    received = []
    adapter = _adapter(_config(), received)
    adapter._rate_limit = 10_000
    loop_thread = threading.get_ident()
    writes = []
    real_write = adapter._inbox.write

    def spy(record):
        writes.append((record["state"], len(json.dumps(record, default=str)), threading.get_ident()))
        real_write(record)

    monkeypatch.setattr(adapter._inbox, "write", spy)
    for index in range(150):
        assert (await _post(adapter, f"bulk-{index:04d}"))[0] == 202
    await _settle(adapter)

    assert len(received) == 150
    assert sorted({state for state, _, _ in writes}) == ["pending", "started"]
    assert len(writes) == 300  # exactly one admission write + one hand-off write per delivery
    pending_sizes = [size for state, size, _ in writes if state == "pending"]
    assert max(pending_sizes) - min(pending_sizes) <= 8  # independent of how many are live
    assert all(thread != loop_thread for _, _, thread in writes)


@pytest.mark.asyncio
async def test_superseded_coalesced_event_is_retired_and_pending_group_replays(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    received = []
    config = _config(coalesce={"key": "message", "window_seconds": 30})
    adapter = _adapter(config, received)
    assert (await _post(adapter, "c-1", "pr-7"))[1]["status"] == "coalesced"
    assert (await _post(adapter, "c-2", "pr-7"))[1]["status"] == "coalesced"
    await _settle(adapter)
    # Crash with the group still buffered: c-1 was superseded, c-2 is accepted work.
    for task in adapter._coalescer._timers.values():
        task.cancel()
    assert _inbox_states(tmp_path) == ["pending", "superseded"]

    restarted = _adapter(config, received)
    restarted._replay_pending_deliveries()
    await restarted._coalescer.flush()
    await _settle(restarted)
    assert [e.message_id for e in received] == ["c-2"]
    assert _inbox_states(tmp_path) == ["started", "superseded"]
