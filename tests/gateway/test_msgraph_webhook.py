"""Tests for the Microsoft Graph webhook adapter."""

import asyncio
import json

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig, _apply_env_overrides
from gateway.platforms.msgraph_webhook import AIOHTTP_AVAILABLE, MSGraphWebhookAdapter


def _make_adapter(**extra_overrides) -> MSGraphWebhookAdapter:
    extra = {
        "host": "127.0.0.1",
        "client_state": "expected-client-state",
        "accepted_resources": ["communications/onlineMeetings"],
    }
    extra.update(extra_overrides)
    return MSGraphWebhookAdapter(PlatformConfig(enabled=True, extra=extra))


class _FakeRequest:
    def __init__(
        self,
        *,
        query=None,
        json_payload=None,
        raw_body: bytes | None = None,
        content_length: int | None = None,
        remote="127.0.0.1",
    ):
        self.query = query or {}
        self._json_payload = json_payload
        self._raw_body = raw_body
        self.content_length = content_length
        self.remote = remote

    async def json(self):
        if isinstance(self._json_payload, Exception):
            raise self._json_payload
        return self._json_payload

    async def read(self):
        if self._raw_body is not None:
            return self._raw_body
        return json.dumps(self._json_payload or {}).encode("utf-8")


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


class TestMSGraphValidationHandshake:


    @pytest.mark.anyio
    async def test_connect_allows_loopback_without_source_allowlist(self):
        if not AIOHTTP_AVAILABLE:
            pytest.skip("aiohttp not installed")
        adapter = _make_adapter(host="127.0.0.1", port=0, allowed_source_cidrs=[])
        try:
            connected = await adapter.connect()
            assert connected is True
            assert adapter.is_connected is True
        finally:
            await adapter.disconnect()

    @pytest.mark.anyio
    async def test_validation_token_echo_on_get(self):
        adapter = _make_adapter()
        resp = await adapter._handle_validation(
            _FakeRequest(query={"validationToken": "abc123"})
        )
        assert resp.status == 200
        assert resp.text == "abc123"
        assert resp.content_type == "text/plain"


class TestMSGraphNotifications:
    @pytest.mark.anyio
    async def test_missing_client_state_is_auth_rejected(self):
        adapter = _make_adapter(client_state=None)
        payload = {
            "value": [
                {
                    "id": "notif-no-client-state",
                    "subscriptionId": "sub-1",
                    "changeType": "updated",
                    "resource": "communications/onlineMeetings/meeting-1",
                }
            ]
        }
        resp = await adapter._handle_notification(_FakeRequest(json_payload=payload))
        assert resp.status == 403

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
        # Success is 202 with empty body: internal counters must not leak to
        # the wire. Counters are still observable via /health.
        assert resp.status == 202
        assert resp.body is None or not resp.body

        await asyncio.sleep(0.05)

        assert len(scheduled) == 1
        notification, event = scheduled[0]
        assert notification["id"] == "notif-1"
        assert event.source.platform == Platform.MSGRAPH_WEBHOOK
        assert event.source.chat_type == "webhook"
        assert event.message_id == "id:notif-1"

    @pytest.mark.anyio
    async def test_oversized_notification_rejected_by_content_length(self):
        adapter = _make_adapter(max_body_bytes=100)
        payload = {
            "value": [
                {
                    "id": "notif-oversized",
                    "subscriptionId": "sub-1",
                    "changeType": "updated",
                    "resource": "communications/onlineMeetings/meeting-1",
                    "clientState": "expected-client-state",
                }
            ]
        }

        resp = await adapter._handle_notification(
            _FakeRequest(json_payload=payload, content_length=101)
        )

        assert resp.status == 413


    @pytest.mark.anyio
    async def test_non_ascii_client_state_rejected_without_raising(self):
        """A non-ASCII clientState (attacker-controlled request body) must be
        rejected with 403, not crash the handler: hmac.compare_digest raises
        TypeError on a str containing non-ASCII characters."""
        adapter = _make_adapter()
        payload = {
            "value": [
                {
                    "id": "notif-nonascii",
                    "subscriptionId": "sub-1",
                    "changeType": "updated",
                    "resource": "communications/onlineMeetings/meeting-x",
                    "clientState": "ské-not-the-secret",
                }
            ]
        }
        response = await adapter._handle_notification(
            _FakeRequest(json_payload=payload)
        )
        assert response.status == 403


    @pytest.mark.anyio
    async def test_resource_patterns_accept_leading_slash(self):
        adapter = _make_adapter(accepted_resources=["/communications/onlineMeetings"])
        payload = {
            "value": [
                {
                    "id": "notif-slash",
                    "subscriptionId": "sub-1",
                    "changeType": "updated",
                    "resource": "communications/onlineMeetings/meeting-4",
                    "clientState": "expected-client-state",
                }
            ]
        }

        resp = await adapter._handle_notification(_FakeRequest(json_payload=payload))
        assert resp.status == 202


class TestMSGraphSourceIPAllowlist:
    @pytest.mark.anyio
    async def test_public_bind_without_allowlist_fails_closed(self):
        """Public binds must not accept requests until a source allowlist is configured."""
        adapter = _make_adapter(host="0.0.0.0", allowed_source_cidrs=[])
        payload = {
            "value": [
                {
                    "id": "notif-ip",
                    "resource": "communications/onlineMeetings/m",
                    "clientState": "expected-client-state",
                }
            ]
        }
        resp = await adapter._handle_notification(
            _FakeRequest(json_payload=payload, remote="203.0.113.99")
        )
        assert resp.status == 403

    @pytest.mark.anyio
    async def test_loopback_bind_without_allowlist_still_accepts_local_requests(self):
        """Loopback-only listeners may rely on local proxying/tunnels instead of CIDRs."""
        adapter = _make_adapter(host="127.0.0.1", allowed_source_cidrs=[])
        payload = {
            "value": [
                {
                    "id": "notif-ip-local",
                    "resource": "communications/onlineMeetings/m",
                    "clientState": "expected-client-state",
                }
            ]
        }
        resp = await adapter._handle_notification(
            _FakeRequest(json_payload=payload, remote="127.0.0.1")
        )
        assert resp.status == 202

    @pytest.mark.anyio
    async def test_post_from_disallowed_ip_rejected(self):
        adapter = _make_adapter(allowed_source_cidrs=["10.0.0.0/8"])
        payload = {
            "value": [
                {
                    "id": "notif-ip-bad",
                    "resource": "communications/onlineMeetings/m",
                    "clientState": "expected-client-state",
                }
            ]
        }
        resp = await adapter._handle_notification(
            _FakeRequest(json_payload=payload, remote="203.0.113.99")
        )
        assert resp.status == 403


