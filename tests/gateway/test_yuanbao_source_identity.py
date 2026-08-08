import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import trusted_source_message_id
from gateway.platforms.yuanbao import (
    AccessPolicy,
    InboundContext,
    InboundPipelineBuilder,
    YuanbaoAdapter,
)


def _adapter() -> YuanbaoAdapter:
    config = PlatformConfig(
        enabled=True,
        extra={
            "app_id": "test_key",
            "app_secret": "test_secret",
            "ws_url": "wss://test.example.com/ws",
            "api_domain": "https://test.example.com",
        },
    )
    adapter = YuanbaoAdapter(config)
    adapter._bot_id = "bot_123"
    adapter._access_policy = AccessPolicy(
        dm_policy="open",
        dm_allow_from=[],
        group_policy="open",
        group_allow_from=[],
    )
    adapter.handle_message = AsyncMock()
    adapter._resolve_inbound_media_urls = AsyncMock(return_value=([], []))
    return adapter


def _push(text: str, message_id: str) -> bytes:
    return json.dumps(
        {
            "CallbackCommand": "C2C.CallbackAfterSendMsg",
            "From_Account": "alice",
            "To_Account": "bot_123",
            "MsgBody": [
                {
                    "MsgType": "TIMTextElem",
                    "MsgContent": {"text": text},
                }
            ],
            "MsgKey": message_id,
        }
    ).encode("utf-8")


@pytest.mark.asyncio
async def test_yuanbao_aggregated_frames_fail_closed_source_identity(monkeypatch):
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
    adapter = _adapter()
    ctx = InboundContext(
        adapter=adapter,
        raw_frames=[
            _push("approved request", "approved-1"),
            _push("later request", "later-2"),
        ],
    )

    await InboundPipelineBuilder.build().execute(ctx)
    if adapter._inbound_tasks:
        await asyncio.gather(*list(adapter._inbound_tasks))

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert "approved request" in event.text
    assert "later request" in event.text
    assert trusted_source_message_id(event) is None
