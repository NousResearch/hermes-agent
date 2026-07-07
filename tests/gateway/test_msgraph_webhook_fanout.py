from __future__ import annotations

import asyncio

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.msgraph_webhook import MSGraphWebhookAdapter


class _FakeRequest:
    def __init__(self, payload):
        self.query = {}
        self._payload = payload
        self.remote = "127.0.0.1"

    async def json(self):
        return self._payload


@pytest.mark.anyio
async def test_msgraph_webhook_fans_out_to_named_schedulers():
    adapter = MSGraphWebhookAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "host": "127.0.0.1",
                "client_state": "state",
                "accepted_resources": ["chats"],
            },
        )
    )
    seen: list[str] = []

    async def first(notification, event):
        seen.append(f"first:{notification['id']}:{event.message_id}")

    def second(notification, event):
        seen.append(f"second:{notification['id']}:{event.message_id}")

    adapter.register_notification_scheduler("first", first)
    adapter.register_notification_scheduler("second", second)

    response = await adapter._handle_notification(
        _FakeRequest(
            {
                "value": [
                    {
                        "id": "n1",
                        "subscriptionId": "sub",
                        "changeType": "created",
                        "resource": "chats/19:chat/messages/msg-1",
                        "clientState": "state",
                    }
                ]
            }
        )
    )

    assert response.status == 202
    await asyncio.sleep(0.05)
    assert len(seen) == 2
    assert seen[0].startswith("second:") or seen[0].startswith("first:")
