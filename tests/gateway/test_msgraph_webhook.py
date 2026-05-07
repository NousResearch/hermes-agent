"""Tests for the Microsoft Graph webhook adapter."""

import asyncio
import json

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig, _apply_env_overrides
from gateway.platforms.msgraph_webhook import MSGraphWebhookAdapter


def _make_adapter(**extra_overrides) -> MSGraphWebhookAdapter:
    extra = {
        "client_state": "expected-client-state",
        "accepted_resources": ["communications/onlineMeetings"],
    }
    extra.update(extra_overrides)
    return MSGraphWebhookAdapter(PlatformConfig(enabled=True, extra=extra))


class _FakeRequest:
    def __init__(self, *, query=None, json_payload=None):
        self.query = query or {}
        self._json_payload = json_payload

    async def json(self):
        if isinstance(self._json_payload, Exception):
            raise self._json_payload
        return self._json_payload


class TestMSGraphWebhookConfig:
    def test_gateway_config_accepts_msgraph_webhook_platform(self):
        config = GatewayConfig.from_dict(
            {
                "platforms": {
                    "msgraph_webhook": {
                        "enabled": True,
                        "extra": {"client_state": "expected"},
                    }
                }
            }
        )

        assert Platform.MSGRAPH_WEBHOOK in config.platforms
        assert Platform.MSGRAPH_WEBHOOK in config.get_connected_platforms()

    def test_env_overrides_apply_to_existing_msgraph_webhook_platform(self, monkeypatch):
        config = GatewayConfig(
            platforms={Platform.MSGRAPH_WEBHOOK: PlatformConfig(enabled=True, extra={})}
        )

        monkeypatch.setenv("MSGRAPH_WEBHOOK_PORT", "8650")
        monkeypatch.setenv("MSGRAPH_WEBHOOK_CLIENT_STATE", "env-state")
        monkeypatch.setenv(
            "MSGRAPH_WEBHOOK_ACCEPTED_RESOURCES",
            "communications/onlineMeetings, chats/getAllMessages",
        )

        _apply_env_overrides(config)

        extra = config.platforms[Platform.MSGRAPH_WEBHOOK].extra
        assert extra["port"] == 8650
        assert extra["client_state"] == "env-state"
        assert extra["accepted_resources"] == [
            "communications/onlineMeetings",
            "chats/getAllMessages",
        ]


class TestMSGraphValidationHandshake:
    @pytest.mark.anyio
    async def test_validation_token_echo(self):
        adapter = _make_adapter()
        resp = await adapter._handle_notification(
            _FakeRequest(query={"validationToken": "abc123"})
        )
        assert resp.status == 200
        assert resp.text == "abc123"
        assert resp.content_type == "text/plain"


class TestMSGraphNotifications:
    @pytest.mark.anyio
    async def test_valid_notification_accepted_and_scheduled(self):
        adapter = _make_adapter()
        scheduled: list[tuple[dict, object]] = []

        async def _capture(notification, event):
            scheduled.append((notification, event))

        adapter.set_notification_scheduler(_capture)
        payload = {
            "value": [
                {
                    "id": "notif-1",
                    "subscriptionId": "sub-1",
                    "changeType": "updated",
                    "resource": "communications/onlineMeetings/meeting-1",
                    "clientState": "expected-client-state",
                    "resourceData": {"id": "meeting-1"},
                }
            ]
        }

        resp = await adapter._handle_notification(_FakeRequest(json_payload=payload))
        assert resp.status == 202
        data = json.loads(resp.text)
        assert data["accepted"] == 1
        assert data["duplicates"] == 0
        assert data["rejected"] == 0
        assert data["scheduled"] == 1

        await asyncio.sleep(0.05)

        assert len(scheduled) == 1
        notification, event = scheduled[0]
        assert notification["id"] == "notif-1"
        assert event.source.platform == Platform.MSGRAPH_WEBHOOK
        assert event.source.chat_type == "webhook"
        assert event.message_id == "id:notif-1"

    @pytest.mark.anyio
    async def test_bad_client_state_rejected(self):
        adapter = _make_adapter()
        scheduled: list[tuple[dict, object]] = []

        async def _capture(notification, event):
            scheduled.append((notification, event))

        adapter.set_notification_scheduler(_capture)
        payload = {
            "value": [
                {
                    "id": "notif-2",
                    "subscriptionId": "sub-1",
                    "changeType": "updated",
                    "resource": "communications/onlineMeetings/meeting-2",
                    "clientState": "wrong-state",
                }
            ]
        }

        resp = await adapter._handle_notification(_FakeRequest(json_payload=payload))
        assert resp.status == 403
        data = json.loads(resp.text)
        assert data["accepted"] == 0
        assert data["duplicates"] == 0
        assert data["rejected"] == 1

        await asyncio.sleep(0.05)

        assert scheduled == []

    @pytest.mark.anyio
    async def test_duplicate_notification_deduped(self):
        adapter = _make_adapter()
        scheduled: list[tuple[dict, object]] = []

        async def _capture(notification, event):
            scheduled.append((notification, event))

        adapter.set_notification_scheduler(_capture)
        payload = {
            "value": [
                {
                    "id": "notif-dup",
                    "subscriptionId": "sub-1",
                    "changeType": "updated",
                    "resource": "communications/onlineMeetings/meeting-3",
                    "clientState": "expected-client-state",
                }
            ]
        }

        first = await adapter._handle_notification(_FakeRequest(json_payload=payload))
        assert first.status == 202
        second = await adapter._handle_notification(_FakeRequest(json_payload=payload))
        assert second.status == 202
        second_data = json.loads(second.text)
        assert second_data["accepted"] == 0
        assert second_data["duplicates"] == 1
        assert second_data["scheduled"] == 0

        await asyncio.sleep(0.05)

        assert len(scheduled) == 1

    @pytest.mark.anyio
    async def test_seen_receipts_are_bounded(self):
        adapter = _make_adapter(max_seen_receipts=2)

        async def _capture(notification, event):
            return None

        adapter.set_notification_scheduler(_capture)

        async def _post(notification_id: str):
            payload = {
                "value": [
                    {
                        "id": notification_id,
                        "subscriptionId": "sub-1",
                        "changeType": "updated",
                        "resource": "communications/onlineMeetings/meeting-3",
                        "clientState": "expected-client-state",
                    }
                ]
            }
            return await adapter._handle_notification(_FakeRequest(json_payload=payload))

        first = await _post("notif-a")
        second = await _post("notif-b")
        third = await _post("notif-c")

        assert first.status == 202
        assert second.status == 202
        assert third.status == 202
        assert len(adapter._seen_receipts) == 2
        assert list(adapter._seen_receipt_order) == ["id:notif-b", "id:notif-c"]

        replay = await _post("notif-a")
        replay_data = json.loads(replay.text)
        assert replay_data["accepted"] == 1
        assert replay_data["duplicates"] == 0
