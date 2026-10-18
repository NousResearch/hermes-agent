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
from gateway.platforms.webhook import WebhookAdapter, _INSECURE_NO_AUTH

URGENT = "[NEEDS-KIM] Server remains offline after automated recovery\nAction required: Open the provider console and check the boot failure.\nEvidence: Two independent HTTP probes time out."


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
    ("Successful reboot. A future tuning decision needs Kim. No current outage.", None, False),
    ("Status: watcher pending\n[NEEDS-KIM] this tag is buried\nAction required: Check later.", None, False),
    (URGENT + "\nEvidence note: [SILENT] because this recovered.", None, False),
    (URGENT, {"_interim_send": True}, False),
    ("[NEEDS-KIM] Offline\nAction required: none", None, False),
    ("[NEEDS-KIM] Offline\nAction required: No action needed.", None, False),
    ("[NEEDS-KIM]\nAction required: Inspect console.", None, False),
    ("[NEEDS-KIM] Offline\nEvidence: Unreachable", None, False),
    (URGENT, None, True),
    (URGENT.replace("[NEEDS-KIM]", "[UNRESOLVED]"), None, True),
    (URGENT.replace("[NEEDS-KIM]", "[SECURITY-INCIDENT]"), None, True),
])
def test_current_policy_controls_existing_session_egress(tmp_path, monkeypatch, text, metadata, allowed):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    a, target = _adapter()
    chat = "webhook:incident:q:existing:0"
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


@pytest.mark.parametrize("content,direct,expected", [
    ("Recovered, nothing for Kim to do.", False, 0),
    (URGENT, False, 1),
    ("MONEY SITE DOWN: verified live alert", True, 1),
])
def test_http_dispatch_preserves_policy_and_direct_alerts(tmp_path, monkeypatch, content, direct, expected):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    a, target = _adapter()
    _subscriptions(tmp_path, {"incident": _route(response_policy="urgent_only", prompt=content, deliver_only=direct)})
    captured = []

    async def handle(event):
        captured.append(event)
        assert (await a.send(event.source.chat_id, content)).success

    a.handle_message = handle

    async def exercise():
        app = web.Application()
        app.router.add_post("/webhooks/{route_name}", a._handle_webhook)
        async with TestClient(TestServer(app)) as client:
            response = await client.post("/webhooks/incident", json={"type": "alert"}, headers={"X-Request-ID": "test-dispatch"})
            assert response.status == (200 if direct else 202)
            if a._background_tasks:
                await asyncio.gather(*list(a._background_tasks))
        assert target.send.await_count == expected
        if not direct:
            info = a._delivery_info[captured[0].source.chat_id]
            assert info.get("response_policy") == "urgent_only"
        with pytest.raises(ValueError, match="response_policy"):
            a._validate_route("invalid", _route(response_policy="urgent_ony"))

    asyncio.run(exercise())
