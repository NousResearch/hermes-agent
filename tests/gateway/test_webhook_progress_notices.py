"""A signed webhook delivers routine compaction notices to chat only on opt-in (#157743)."""

import hashlib
import hmac
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from agent.conversation_compression import COMPACTION_DONE_STATUS, COMPACTION_STATUS
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.webhook import WebhookAdapter, _WebhookDeliveryIdentity
from gateway.run import _send_or_update_status_coro
from hermes_constants import get_hermes_home
from tests.gateway.fixtures.webhook_route_authority import mount_authority


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True], ids=["default", "opt-in"])
async def test_webhook_chat_compaction_progress_respects_opt_in(enabled, monkeypatch):
    monkeypatch.setattr("gateway.run._hermes_home", get_hermes_home())
    if enabled:
        (get_hermes_home() / "config.yaml").write_text("compression:\n  progress_notices: true\n")
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={
        "secret": "test-secret",
        "routes": {"fixture": {
            "prompt": "{text}", "deliver": "whatsapp",
            "deliver_extra": {"chat_id": "fixture-chat"},
        }},
    }))
    app = web.Application()
    mount_authority(app, adapter)
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)

    async with TestClient(TestServer(app)) as client:
        runner = adapter._message_handler.__self__
        adapter.gateway_runner = runner
        sink = AsyncMock(return_value=SendResult(success=True))
        runner.adapters[Platform.WHATSAPP] = SimpleNamespace(send=sink)
        payload = b'{"text":"send the digest"}'
        signature = hmac.new(b"test-secret", payload, hashlib.sha256).hexdigest()
        response = await client.post("/webhooks/fixture", data=payload, headers={
            "X-Hub-Signature-256": f"sha256={signature}", "X-GitHub-Delivery": "progress",
        })
        assert response.status == 202, await response.text()

        chat_id = _WebhookDeliveryIdentity.from_parts(None, "fixture", "progress").session_chat_id
        assert (await _send_or_update_status_coro(adapter, chat_id, "lifecycle", COMPACTION_STATUS, {})).success
        assert (await _send_or_update_status_coro(adapter, chat_id, "compacted", COMPACTION_DONE_STATUS, {})).success
        assert (await adapter.send(chat_id, "completed digest")).success
        # A final answer containing this wording is not a status callback.
        assert (await adapter.send(chat_id, COMPACTION_DONE_STATUS)).success

        expected = [COMPACTION_STATUS, COMPACTION_DONE_STATUS] if enabled else []
        assert [call.args[:2] for call in sink.await_args_list] == [
            ("fixture-chat", message) for message in [*expected, "completed digest", COMPACTION_DONE_STATUS]
        ]
