"""Urgent-only webhook egress: enforcement includes resumed background turns."""
import asyncio
import json
import os
import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.webhook import WebhookAdapter, _INSECURE_NO_AUTH, _WebhookDeliveryIdentity

URGENT = "[NEEDS-APPROVAL] Server remains offline after automated recovery\nAction required: Open the provider console and check the boot failure.\nEvidence: Two independent HTTP probes time out."


def _adapter():
    a = WebhookAdapter(PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": 0}))
    target = AsyncMock()
    target.send.return_value = SendResult(success=True)
    runner = MagicMock()
    runner._authorization_adapter = lambda platform, profile=None: target if platform == Platform.DISCORD else None
    a.gateway_runner = runner
    return a, target


def _route(**kwargs):
    return {"secret": _INSECURE_NO_AUTH, "deliver": "discord", "deliver_extra": {"chat_id": "test-only"}, **kwargs}


def _subscriptions(tmp_path, routes):
    path = tmp_path / "webhook_subscriptions.json"
    path.write_text(json.dumps(routes))
    # Deliberately change mtime: live reload uses a monotonically increasing mtime.
    ts = max(time.time(), path.stat().st_mtime) + 1
    os.utime(path, (ts, ts))
    return path


@pytest.mark.parametrize("text,metadata,allowed", [
    ("VERDICT: [AUTO-FIX] staticSites rung 1; HOLD-GATE PASS. 7 LINK sites recovered.", None, False),
    ("GATE-PASS — docs.dilantimedia.com resolved on its own. No action needed.", None, False),
    ("VERDICT: staticSites episode #16 gate PASSED — self-drained. [SILENT]\nGate evidence: zero stalls.", None, False),
    ("VERDICT: Contabo VPS2 incident fully closed — monitors green. [SILENT]\nReboot verified.", None, False),
    ("Successful reboot. A future tuning decision needs owner approval. No current outage.", None, False),
    ("Status: watcher pending\n[NEEDS-APPROVAL] this tag is buried\nAction required: Check later.", None, False),
    (URGENT + "\nEvidence note: [SILENT] because this recovered.", None, False),
    (URGENT, {"_interim_send": True}, False),
    ("[NEEDS-APPROVAL] Offline\nAction required: none", None, False),
    ("[NEEDS-APPROVAL] Offline\nAction required: No action needed.", None, False),
    ("[NEEDS-APPROVAL]\nAction required: Inspect console.", None, False),
    ("[NEEDS-APPROVAL] Offline\nEvidence: Unreachable", None, False),
    (URGENT, None, True),
    (URGENT.replace("[NEEDS-APPROVAL]", "[UNRESOLVED]"), None, True),
    (URGENT.replace("[NEEDS-APPROVAL]", "[SECURITY-INCIDENT]"), None, True),
])
def test_current_policy_controls_existing_session_egress(tmp_path, monkeypatch, text, metadata, allowed):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    a, target = _adapter()
    # Production chat ids are `webhook:v2:<token>`; the legacy spelling below is kept
    # only in the decoder for admissions queued before that upgrade.
    chat = _WebhookDeliveryIdentity.from_parts(None, "incident", "existing-0").session_chat_id
    a._delivery_info[chat] = {"deliver": "discord", "deliver_extra": {"chat_id": "test-only"}}
    _subscriptions(tmp_path, {"incident": _route()})

    async def exercise():
        # Default/unconfigured routes preserve existing behavior.
        assert (await a.send(chat, "An ordinary ungated reply")).success
        assert target.send.await_count == 1
        target.send.reset_mock()
        # Apply to a cached session without receiving another HTTP event.
        _subscriptions(tmp_path, {"incident": _route(response_policy="urgent_only")})
        result = await a.send(chat, text, metadata=metadata)
        assert result.success
        assert target.send.await_count == int(allowed)
        assert chat in a._delivery_info
        target.send.reset_mock()
        # A suppressed reply must not consume later urgent delivery.
        assert (await a.send(chat, URGENT)).success
        assert target.send.await_count == 1

    asyncio.run(exercise())


def test_v2_session_without_cached_policy_still_gated(tmp_path, monkeypatch):
    """Regression: a `webhook:v2:<token>` chat id must decode its route via the canonical
    decoder, not a raw split — with the in-memory policy cache expired (gateway restart,
    TTL prune), an urgent_only route still suppresses a routine final."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    a, target = _adapter()
    chat = _WebhookDeliveryIdentity.from_parts(None, "incident", "late-final").session_chat_id
    assert chat.startswith("webhook:v2:")  # the format the old `split(":", 2)[1]` mis-parsed
    _subscriptions(tmp_path, {"incident": _route(response_policy="urgent_only")})

    async def exercise():
        # No _delivery_info/_delivery_policies entry: only the route config can gate.
        # A routine final is suppressed (success without contacting the target); an
        # urgent final passes the gate and only then misses its (here absent) destination.
        assert (await a.send(chat, "Routine close-out: incident recovered on its own.")).success
        assert target.send.await_count == 0
        urgent = await a.send(chat, URGENT)
        assert not urgent.success  # passed the policy gate, failed at destination resolution

    asyncio.run(exercise())


@pytest.mark.parametrize("content,direct,expected", [
    ("Recovered, nothing for the owner to do.", False, 0),
    (URGENT, False, 1),
    ("MONEY SITE DOWN: verified live alert", True, 1),
])
def test_http_dispatch_preserves_policy_and_direct_alerts(tmp_path, monkeypatch, content, direct, expected):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    a = WebhookAdapter(PlatformConfig(enabled=True, extra={
        "host": "127.0.0.1", "port": 0,
        "routes": {"incident": _route(response_policy="urgent_only", prompt=content, deliver_only=direct)}}))
    captured = []

    async def handle(event):
        captured.append(event)

    a.handle_message = handle
    target = AsyncMock()
    target.send.return_value = SendResult(success=True)
    mock_runner = MagicMock()
    mock_runner._profile_name_for_source.return_value = None
    mock_runner.adapters = {Platform.DISCORD: target}
    mock_runner._authorization_adapter = lambda platform, profile=None: mock_runner.adapters.get(platform)
    mock_runner.config.get_home_channel.return_value = None
    a.gateway_runner = mock_runner

    async def exercise():
        # Current main routes agent-run admission through the durable producer, so the
        # route-authority fixture (tests/gateway/fixtures/webhook_route_authority.py)
        # must mount a real runner/authority — a bare MagicMock runner cannot admit.
        app = web.Application()
        from tests.gateway.fixtures.webhook_route_authority import mount_authority
        mount_authority(app, a)
        app.router.add_post("/webhooks/{route_name}", a._handle_webhook)
        async with TestClient(TestServer(app)) as client:
            response = await client.post("/webhooks/incident", json={"type": "alert"},
                                         headers={"X-GitHub-Delivery": "test-dispatch"})
            assert response.status == (200 if direct else 202)
        if not direct:
            # The admitted session's reply leg is subject to the route policy.
            chat_id = captured[0].source.chat_id
            assert (await a.send(chat_id, content)).success
            assert a._delivery_policies.get(chat_id) == "urgent_only"
        assert target.send.await_count == expected
        with pytest.raises(ValueError, match="response_policy"):
            a._validate_route("invalid", _route(response_policy="urgent_ony"))

    asyncio.run(exercise())
