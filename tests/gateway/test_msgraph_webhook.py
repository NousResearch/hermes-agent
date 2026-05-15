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
    def __init__(self, *, query=None, json_payload=None, remote="127.0.0.1"):
        self.query = query or {}
        self._json_payload = json_payload
        self.remote = remote

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
    async def test_validation_token_echo_on_get(self):
        adapter = _make_adapter()
        resp = await adapter._handle_validation(
            _FakeRequest(query={"validationToken": "abc123"})
        )
        assert resp.status == 200
        assert resp.text == "abc123"
        assert resp.content_type == "text/plain"

    @pytest.mark.anyio
    async def test_bare_get_without_validation_token_rejected(self):
        """GET without validationToken is 400 so the endpoint can't be enumerated."""
        adapter = _make_adapter()
        resp = await adapter._handle_validation(_FakeRequest())
        assert resp.status == 400

    @pytest.mark.anyio
    async def test_post_with_validation_token_still_echoes(self):
        """Tolerate defensive clients that send validationToken on POST."""
        adapter = _make_adapter()
        resp = await adapter._handle_notification(
            _FakeRequest(query={"validationToken": "abc123"})
        )
        assert resp.status == 200
        assert resp.text == "abc123"


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
    async def test_bad_client_state_rejected_as_auth_failure(self):
        """Every-item-bad-clientState batches return 403 so forged POSTs stop retrying."""
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

        await asyncio.sleep(0.05)

        assert scheduled == []

    @pytest.mark.anyio
    async def test_client_state_compare_is_timing_safe(self, monkeypatch):
        """Ensure hmac.compare_digest is used for clientState comparison."""
        import hmac

        calls: list[tuple[str, str]] = []
        real_compare = hmac.compare_digest

        def _spy(a, b):
            calls.append((a, b))
            return real_compare(a, b)

        monkeypatch.setattr(
            "gateway.platforms.msgraph_webhook.hmac.compare_digest", _spy
        )

        adapter = _make_adapter()
        payload = {
            "value": [
                {
                    "id": "notif-timing",
                    "subscriptionId": "sub-1",
                    "changeType": "updated",
                    "resource": "communications/onlineMeetings/meeting-x",
                    "clientState": "expected-client-state",
                }
            ]
        }
        await adapter._handle_notification(_FakeRequest(json_payload=payload))

        assert calls, "hmac.compare_digest was never called; clientState check is not timing-safe"
        provided, expected = calls[0]
        assert provided == "expected-client-state"
        assert expected == "expected-client-state"

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
        # Duplicate-only batch still returns 202 so Graph stops retrying.
        assert second.status == 202
        assert adapter._duplicate_count == 1

        await asyncio.sleep(0.05)

        assert len(scheduled) == 1

    @pytest.mark.anyio
    async def test_notifications_without_id_are_not_deduped(self):
        adapter = _make_adapter()
        scheduled: list[tuple[dict, object]] = []

        async def _capture(notification, event):
            scheduled.append((notification, event))

        adapter.set_notification_scheduler(_capture)
        payload = {
            "value": [
                {
                    "subscriptionId": "sub-1",
                    "changeType": "updated",
                    "resource": "communications/onlineMeetings/meeting-3",
                    "clientState": "expected-client-state",
                    "resourceData": {"id": "meeting-3"},
                }
            ]
        }

        first = await adapter._handle_notification(_FakeRequest(json_payload=payload))
        second = await adapter._handle_notification(_FakeRequest(json_payload=payload))

        assert first.status == 202
        assert second.status == 202

        await asyncio.sleep(0.05)

        assert len(scheduled) == 2

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

    @pytest.mark.anyio
    async def test_resource_not_in_allowlist_returns_400(self):
        """Every-item-rejected-for-non-auth returns 400 (configuration issue)."""
        adapter = _make_adapter(accepted_resources=["communications/onlineMeetings"])
        payload = {
            "value": [
                {
                    "id": "notif-bad-resource",
                    "resource": "users/u1/messages",
                    "clientState": "expected-client-state",
                }
            ]
        }
        resp = await adapter._handle_notification(_FakeRequest(json_payload=payload))
        assert resp.status == 400

    @pytest.mark.anyio
    async def test_malformed_body_returns_400(self):
        adapter = _make_adapter()
        resp = await adapter._handle_notification(
            _FakeRequest(json_payload=ValueError("bad json"))
        )
        assert resp.status == 400

    @pytest.mark.anyio
    async def test_missing_value_array_returns_400(self):
        adapter = _make_adapter()
        resp = await adapter._handle_notification(
            _FakeRequest(json_payload={"not_value": []})
        )
        assert resp.status == 400

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
        # notif-a evicted from the bounded cache, so it's accepted again (202)
        # rather than treated as a duplicate.
        assert replay.status == 202
        assert adapter._accepted_count == 4


class TestMSGraphSourceIPAllowlist:
    @pytest.mark.anyio
    async def test_disabled_by_default_allows_all(self):
        """Empty allowlist preserves pre-existing behavior (dev tunnels, localhost)."""
        adapter = _make_adapter()  # no allowed_source_cidrs set
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

    @pytest.mark.anyio
    async def test_post_from_allowed_ip_accepted(self):
        adapter = _make_adapter(allowed_source_cidrs=["10.0.0.0/8", "203.0.113.0/24"])
        payload = {
            "value": [
                {
                    "id": "notif-ip-ok",
                    "resource": "communications/onlineMeetings/m",
                    "clientState": "expected-client-state",
                }
            ]
        }
        resp = await adapter._handle_notification(
            _FakeRequest(json_payload=payload, remote="203.0.113.5")
        )
        assert resp.status == 202

    @pytest.mark.anyio
    async def test_validation_handshake_also_respects_allowlist(self):
        """A disallowed IP shouldn't be able to probe the handshake endpoint."""
        adapter = _make_adapter(allowed_source_cidrs=["10.0.0.0/8"])
        resp = await adapter._handle_validation(
            _FakeRequest(query={"validationToken": "probe"}, remote="203.0.113.99")
        )
        assert resp.status == 403

    @pytest.mark.anyio
    async def test_invalid_cidr_entries_are_ignored_at_init(self):
        """Malformed CIDR strings should log a warning and be ignored, not crash."""
        adapter = _make_adapter(
            allowed_source_cidrs=["10.0.0.0/8", "not-a-cidr", "", "203.0.113.0/24"]
        )
        assert len(adapter._allowed_source_networks) == 2

    @pytest.mark.anyio
    async def test_cidr_list_accepts_comma_string(self):
        """Env-var-style 'cidr1, cidr2' strings parse as a list."""
        adapter = _make_adapter(allowed_source_cidrs="10.0.0.0/8, 203.0.113.0/24")
        assert len(adapter._allowed_source_networks) == 2
