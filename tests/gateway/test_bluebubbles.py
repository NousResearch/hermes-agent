"""Tests for the BlueBubbles iMessage gateway adapter."""
import asyncio
import json

import pytest

from gateway.config import Platform, PlatformConfig


def _make_adapter(monkeypatch, **extra):
    monkeypatch.setenv("BLUEBUBBLES_SERVER_URL", "http://localhost:1234")
    monkeypatch.setenv("BLUEBUBBLES_PASSWORD", "secret")
    from gateway.platforms.bluebubbles import BlueBubblesAdapter

    cfg = PlatformConfig(
        enabled=True,
        extra={
            "server_url": "http://localhost:1234",
            "password": "secret",
            **extra,
        },
    )
    return BlueBubblesAdapter(cfg)


class TestBlueBubblesConfigLoading:
    def test_apply_env_overrides_bluebubbles(self, monkeypatch):
        monkeypatch.setenv("BLUEBUBBLES_SERVER_URL", "http://localhost:1234")
        monkeypatch.setenv("BLUEBUBBLES_PASSWORD", "secret")
        monkeypatch.setenv("BLUEBUBBLES_WEBHOOK_PORT", "9999")
        monkeypatch.setenv("BLUEBUBBLES_REQUIRE_MENTION", "true")
        monkeypatch.setenv("BLUEBUBBLES_MENTION_PATTERNS", r'["(?i)^amos\\b"]')
        from gateway.config import GatewayConfig, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)
        assert Platform.BLUEBUBBLES in config.platforms
        bc = config.platforms[Platform.BLUEBUBBLES]
        assert bc.enabled is True
        assert bc.extra["server_url"] == "http://localhost:1234"
        assert bc.extra["password"] == "secret"
        assert bc.extra["webhook_port"] == 9999
        assert bc.extra["require_mention"] is True
        assert bc.extra["mention_patterns"] == ["(?i)^amos\\b"]


class TestBlueBubblesHelpers:
    def test_check_requirements(self, monkeypatch):
        monkeypatch.setenv("BLUEBUBBLES_SERVER_URL", "http://localhost:1234")
        monkeypatch.setenv("BLUEBUBBLES_PASSWORD", "secret")
        from gateway.platforms.bluebubbles import check_bluebubbles_requirements

        assert check_bluebubbles_requirements() is True


    def test_format_message_preserves_underscores_in_identifiers(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        text = "Use /api_v2 with FEATURE_FLAG_NAME and config_file.json"
        assert adapter.format_message(text) == text

    def test_strip_markdown_headers(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        assert adapter.format_message("## Heading\ntext") == "Heading\ntext"


    def test_init_normalizes_webhook_path(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, webhook_path="bluebubbles-webhook")
        assert adapter.webhook_path == "/bluebubbles-webhook"


    def test_server_url_normalized(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, server_url="http://localhost:1234/")
        assert adapter.server_url == "http://localhost:1234"


class _FakeBlueBubblesRequest:
    def __init__(self, payload, password="secret"):
        self.query = {"password": password}
        self.headers = {}
        self._body = json.dumps(payload).encode("utf-8")

    async def read(self):
        return self._body


class TestBlueBubblesDuplicateDelivery:
    @pytest.mark.asyncio
    async def test_v019_new_and_updated_chat_variants_dispatch_once(self, monkeypatch):
        """Regression for #30708/#34372 as reproduced on Hermes v0.19.0."""
        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer

        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        handled = []

        async def fake_handle_message(event):
            handled.append(event)

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        app = web.Application()
        app.router.add_post("/bluebubbles-webhook", adapter._handle_webhook)

        async with TestClient(TestServer(app)) as client:
            first = await client.post(
                "/bluebubbles-webhook?password=secret",
                json={
                    "type": "new-message",
                    "data": {
                        "guid": "v019-msg-1",
                        "text": "approve",
                        "chatGuid": "any;-;+15555550100",
                        "chatIdentifier": "+15555550100",
                        "handle": {"address": "+15555550100"},
                        "isFromMe": False,
                    },
                },
            )
            second = await client.post(
                "/bluebubbles-webhook?password=secret",
                json={
                    "type": "updated-message",
                    "data": {
                        "guid": "v019-msg-1",
                        "text": "approve",
                        "chatIdentifier": "+15555550100",
                        "handle": {"address": "+15555550100"},
                        "isFromMe": False,
                    },
                },
            )
            await asyncio.sleep(0)

        assert first.status == 200
        assert second.status == 200
        assert [(event.message_id, event.text) for event in handled] == [
            ("v019-msg-1", "approve")
        ]

    @pytest.mark.asyncio
    async def test_duplicate_guid_is_dropped_but_same_text_new_guid_is_kept(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        handled = []

        async def fake_handle_message(event):
            handled.append(event)

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        payload = {
            "type": "new-message",
            "data": {
                "guid": "duplicate-guid-1",
                "text": "hello",
                "chatIdentifier": "user@example.com",
                "handle": {"address": "user@example.com"},
                "isFromMe": False,
            },
        }

        first = await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))
        second = await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))
        distinct = {**payload, "data": {**payload["data"], "guid": "duplicate-guid-2"}}
        third = await adapter._handle_webhook(_FakeBlueBubblesRequest(distinct))
        await asyncio.sleep(0)

        assert first.status == 200
        assert second.status == 200
        assert third.status == 200
        assert [event.message_id for event in handled] == [
            "duplicate-guid-1",
            "duplicate-guid-2",
        ]

    @pytest.mark.asyncio
    async def test_failed_handoff_releases_guid_for_redelivery(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        attempts = 0
        handled = []

        async def flaky_handle_message(event):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("transient handoff failure")
            handled.append(event)

        monkeypatch.setattr(adapter, "handle_message", flaky_handle_message)
        payload = {
            "type": "new-message",
            "data": {
                "guid": "retry-guid-1",
                "text": "retry me",
                "chatIdentifier": "user@example.com",
                "handle": {"address": "user@example.com"},
                "isFromMe": False,
            },
        }

        first = await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))
        second = await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))

        assert first.status == 503
        assert second.status == 200
        assert attempts == 2
        assert [event.message_id for event in handled] == ["retry-guid-1"]

    @pytest.mark.asyncio
    async def test_cancelled_handoff_releases_guid_for_redelivery(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        attempts = 0

        async def cancelled_once(event):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise asyncio.CancelledError

        monkeypatch.setattr(adapter, "handle_message", cancelled_once)
        payload = {
            "type": "new-message",
            "data": {
                "guid": "cancelled-guid-1",
                "text": "retry after cancellation",
                "chatIdentifier": "user@example.com",
                "handle": {"address": "user@example.com"},
                "isFromMe": False,
            },
        }

        with pytest.raises(asyncio.CancelledError):
            await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))
        retry = await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))

        assert retry.status == 200
        assert attempts == 2

    @pytest.mark.asyncio
    async def test_single_delivery_retries_attachment_and_preserves_caption(
        self, monkeypatch
    ):
        """BlueBubbles does not redeliver a webhook after a non-2xx response."""
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        attempts = 0
        handled = []

        async def failed_download(attachment_guid, metadata):
            nonlocal attempts
            attempts += 1
            return None

        async def fake_handle_message(event):
            handled.append(event)

        monkeypatch.setattr(adapter, "_download_attachment", failed_download)
        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        monkeypatch.setattr(
            "gateway.platforms.bluebubbles._ATTACHMENT_RETRY_DELAYS", (0, 0)
        )
        payload = {
            "type": "new-message",
            "data": {
                "guid": "attachment-guid-1",
                "text": "image caption",
                "chatIdentifier": "user@example.com",
                "handle": {"address": "user@example.com"},
                "isFromMe": False,
                "attachments": [
                    {
                        "guid": "file-guid-1",
                        "mimeType": "image/png",
                        "transferName": "image.png",
                    }
                ],
            },
        }

        response = await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))

        assert response.status == 200
        assert attempts == 3
        assert [(event.text, event.media_urls) for event in handled] == [
            ("image caption", [])
        ]

    @pytest.mark.asyncio
    async def test_single_delivery_recovers_attachment_on_internal_retry(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        attempts = 0
        handled = []

        async def transient_download(attachment_guid, metadata):
            nonlocal attempts
            attempts += 1
            return None if attempts == 1 else "/tmp/recovered-image.png"

        async def fake_handle_message(event):
            handled.append(event)

        monkeypatch.setattr(adapter, "_download_attachment", transient_download)
        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        monkeypatch.setattr(
            "gateway.platforms.bluebubbles._ATTACHMENT_RETRY_DELAYS", (0, 0)
        )
        payload = {
            "type": "new-message",
            "data": {
                "guid": "attachment-guid-recovered",
                "text": "recovered caption",
                "chatIdentifier": "user@example.com",
                "handle": {"address": "user@example.com"},
                "attachments": [
                    {"guid": "transient-file", "mimeType": "image/png"},
                ],
            },
        }

        response = await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))

        assert response.status == 200
        assert attempts == 2
        assert [(event.text, event.media_urls) for event in handled] == [
            ("recovered caption", ["/tmp/recovered-image.png"])
        ]

    @pytest.mark.asyncio
    async def test_single_delivery_preserves_successful_attachment_siblings(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        attempts = {"good-file": 0, "bad-file": 0}
        handled = []

        async def partial_download(attachment_guid, metadata):
            attempts[attachment_guid] += 1
            if attachment_guid == "good-file":
                return "/tmp/good-image.png"
            return None

        async def fake_handle_message(event):
            handled.append(event)

        monkeypatch.setattr(adapter, "_download_attachment", partial_download)
        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        monkeypatch.setattr(
            "gateway.platforms.bluebubbles._ATTACHMENT_RETRY_DELAYS", (0, 0)
        )
        payload = {
            "type": "new-message",
            "data": {
                "guid": "attachment-guid-2",
                "text": "two files",
                "chatIdentifier": "user@example.com",
                "handle": {"address": "user@example.com"},
                "isFromMe": False,
                "attachments": [
                    {"guid": "good-file", "mimeType": "image/png"},
                    {"guid": "bad-file", "mimeType": "application/pdf"},
                ],
            },
        }

        response = await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))

        assert response.status == 200
        assert attempts == {"good-file": 1, "bad-file": 3}
        assert [(event.text, event.media_urls) for event in handled] == [
            ("two files", ["/tmp/good-image.png"])
        ]

    @pytest.mark.asyncio
    async def test_single_delivery_acknowledges_unrecoverable_attachment_only_message(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        attempts = 0
        handled = []

        async def failed_download(attachment_guid, metadata):
            nonlocal attempts
            attempts += 1
            return None

        async def fake_handle_message(event):
            handled.append(event)

        monkeypatch.setattr(adapter, "_download_attachment", failed_download)
        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        monkeypatch.setattr(
            "gateway.platforms.bluebubbles._ATTACHMENT_RETRY_DELAYS", (0, 0)
        )
        payload = {
            "type": "new-message",
            "data": {
                "guid": "attachment-guid-3",
                "text": "",
                "chatIdentifier": "user@example.com",
                "handle": {"address": "user@example.com"},
                "isFromMe": False,
                "attachments": [
                    {"guid": "bad-file", "mimeType": "image/png"},
                ],
            },
        }

        response = await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))

        assert response.status == 200
        assert attempts == 3
        assert [(event.text, event.media_urls) for event in handled] == [
            ("(attachment unavailable)", [])
        ]


class TestBlueBubblesMentionGating:
    @pytest.mark.asyncio
    async def test_group_message_without_mention_is_acknowledged_and_skipped(self, monkeypatch):
        adapter = _make_adapter(
            monkeypatch,
            require_mention=True,
            send_read_receipts=False,
        )
        handled = []

        async def fake_handle_message(event):
            handled.append(event)

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        response = await adapter._handle_webhook(_FakeBlueBubblesRequest({
            "type": "new-message",
            "data": {
                "guid": "msg-1",
                "text": "casual family chatter",
                "handle": {"address": "+15555550100"},
                "isFromMe": False,
                "isGroup": True,
                "chats": [{"guid": "iMessage;+;group-chat"}],
            },
        }))
        await asyncio.sleep(0)

        assert response.status == 200
        assert handled == []


class TestBlueBubblesWebhookParsing:

    def test_webhook_can_fall_back_to_sender_when_chat_fields_missing(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        payload = {
            "data": {
                "guid": "MESSAGE-GUID",
                "text": "hello",
                "handle": {"address": "user@example.com"},
                "isFromMe": False,
            }
        }
        record = adapter._extract_payload_record(payload) or {}
        chat_guid = adapter._value(
            record.get("chatGuid"),
            payload.get("chatGuid"),
            record.get("chat_guid"),
            payload.get("chat_guid"),
            payload.get("guid"),
        )
        chat_identifier = adapter._value(
            record.get("chatIdentifier"),
            record.get("identifier"),
            payload.get("chatIdentifier"),
            payload.get("identifier"),
        )
        sender = (
            adapter._value(
                record.get("handle", {}).get("address")
                if isinstance(record.get("handle"), dict)
                else None,
                record.get("sender"),
                record.get("from"),
                record.get("address"),
            )
            or chat_identifier
            or chat_guid
        )
        if not (chat_guid or chat_identifier) and sender:
            chat_identifier = sender
        assert chat_identifier == "user@example.com"


    def test_extract_payload_record_accepts_list_data(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        payload = {
            "type": "new-message",
            "data": [
                {
                    "text": "hello",
                    "chatGuid": "iMessage;-;user@example.com",
                    "chatIdentifier": "user@example.com",
                }
            ],
        }
        record = adapter._extract_payload_record(payload)
        assert record == payload["data"][0]


class TestBlueBubblesGuidResolution:


    @pytest.mark.asyncio
    async def test_participant_only_match_does_not_resolve_to_group(self, monkeypatch):
        """Regression for #24157: contact appearing as a participant in a group
        chat must NOT be selected when no DM with that exact chatIdentifier exists.

        Otherwise an outbound DM reply leaks into the group thread.
        """
        adapter = _make_adapter(monkeypatch)

        async def fake_api_post(path, payload):
            return {
                "data": [
                    {
                        "guid": "iMessage;+;chat0000000000-family-group",
                        "chatIdentifier": "chat0000000000",
                        "participants": [
                            {"address": "user@example.com"},
                            {"address": "+15555550100"},
                        ],
                    }
                ]
            }

        monkeypatch.setattr(adapter, "_api_post", fake_api_post)
        result = await adapter._resolve_chat_guid("user@example.com")
        assert result is None, (
            "participant-only match must not resolve to a group GUID — DM "
            "replies would leak into the group thread"
        )


    @pytest.mark.asyncio
    async def test_unresolved_target_is_not_cached(self, monkeypatch):
        """When no exact match is found, the resolver must NOT cache anything.

        Otherwise a later attempt — after the DM has been created — would
        keep returning the stale ``None`` from cache. Also guards against a
        latent variant of #24157 where a group GUID could be cached under a
        bare address key and persist across calls.
        """
        adapter = _make_adapter(monkeypatch)

        async def fake_api_post(path, payload):
            return {
                "data": [
                    {
                        "guid": "iMessage;+;chat0000000000-family-group",
                        "chatIdentifier": "chat0000000000",
                        "participants": [{"address": "user@example.com"}],
                    }
                ]
            }

        monkeypatch.setattr(adapter, "_api_post", fake_api_post)
        await adapter._resolve_chat_guid("user@example.com")
        assert "user@example.com" not in adapter._guid_cache


class TestBlueBubblesAttachmentDownload:
    """Verify _download_attachment routes to the correct cache helper."""

    def test_download_image_uses_image_cache(self, monkeypatch):
        """Image MIME routes to cache_image_from_bytes."""
        adapter = _make_adapter(monkeypatch)
        import asyncio

        # Mock the HTTP client response
        class MockResponse:
            status_code = 200
            content = b"\x89PNG\r\n\x1a\n"

            def raise_for_status(self):
                pass

        async def mock_get(*args, **kwargs):
            return MockResponse()

        adapter.client = type("MockClient", (), {"get": mock_get})()

        cached_path = None

        def mock_cache_image(data, ext):
            nonlocal cached_path
            cached_path = f"/tmp/test_image{ext}"
            return cached_path

        monkeypatch.setattr(
            "gateway.platforms.bluebubbles.cache_image_from_bytes",
            mock_cache_image,
        )

        att_meta = {"mimeType": "image/png", "transferName": "photo.png"}
        result = asyncio.get_event_loop().run_until_complete(
            adapter._download_attachment("att-guid-123", att_meta)
        )
        assert result == "/tmp/test_image.png"


# ---------------------------------------------------------------------------
# Webhook registration
# ---------------------------------------------------------------------------


class TestBlueBubblesWebhookUrl:
    """_webhook_url property normalises local hosts to 'localhost'."""

    def test_default_host(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        # Default webhook_host is 0.0.0.0 → normalized to localhost
        assert "localhost" in adapter._webhook_url
        assert str(adapter.webhook_port) in adapter._webhook_url
        assert adapter.webhook_path in adapter._webhook_url


    def test_register_url_omits_query_when_no_password(self, monkeypatch):
        """If no password is configured, the register URL should be the bare URL."""
        monkeypatch.delenv("BLUEBUBBLES_PASSWORD", raising=False)
        from gateway.platforms.bluebubbles import BlueBubblesAdapter
        cfg = PlatformConfig(
            enabled=True,
            extra={"server_url": "http://localhost:1234", "password": ""},
        )
        adapter = BlueBubblesAdapter(cfg)
        assert adapter._webhook_register_url == adapter._webhook_url


class TestBlueBubblesWebhookRegistration:
    """Tests for _register_webhook, _unregister_webhook, _find_registered_webhooks."""

    @staticmethod
    def _mock_client(get_response=None, post_response=None, delete_ok=True):
        """Build a tiny mock httpx.AsyncClient."""

        async def mock_get(*args, **kwargs):
            class R:
                status_code = 200
                def raise_for_status(self):
                    pass
                def json(self):
                    return get_response or {"status": 200, "data": []}
            return R()

        async def mock_post(*args, **kwargs):
            class R:
                status_code = 200
                def raise_for_status(self):
                    pass
                def json(self):
                    return post_response or {"status": 200, "data": {}}
            return R()

        async def mock_delete(*args, **kwargs):
            class R:
                status_code = 200 if delete_ok else 500
                def raise_for_status(self_inner):
                    if not delete_ok:
                        raise Exception("delete failed")
            return R()

        return type(
            "MockClient", (),
            {"get": mock_get, "post": mock_post, "delete": mock_delete},
        )()

    # -- _find_registered_webhooks --

    def test_find_registered_webhooks_returns_matches(self, monkeypatch):
        import asyncio
        adapter = _make_adapter(monkeypatch)
        url = adapter._webhook_url
        adapter.client = self._mock_client(
            get_response={"status": 200, "data": [
                {"id": 1, "url": url, "events": ["new-message"]},
                {"id": 2, "url": "http://other:9999/hook", "events": ["message"]},
            ]}
        )
        result = asyncio.get_event_loop().run_until_complete(
            adapter._find_registered_webhooks(url)
        )
        assert len(result) == 1
        assert result[0]["id"] == 1


    # -- _register_webhook --

    def test_register_fresh(self, monkeypatch):
        """No existing webhook → POST creates one."""
        import asyncio
        adapter = _make_adapter(monkeypatch)
        adapter.client = self._mock_client(
            get_response={"status": 200, "data": []},
            post_response={"status": 200, "data": {"id": 42}},
        )
        ok = asyncio.get_event_loop().run_until_complete(
            adapter._register_webhook()
        )
        assert ok is True


    def test_register_reuses_existing(self, monkeypatch):
        """Crash resilience — existing registration is reused, no POST needed."""
        import asyncio
        adapter = _make_adapter(monkeypatch)
        url = adapter._webhook_register_url
        adapter.client = self._mock_client(
            get_response={"status": 200, "data": [
                {"id": 7, "url": url, "events": ["new-message"]},
            ]},
        )

        # Track whether POST was called
        post_called = False
        orig_api_post = adapter._api_post
        async def tracking_post(path, payload):
            nonlocal post_called
            post_called = True
            return await orig_api_post(path, payload)
        adapter._api_post = tracking_post

        ok = asyncio.get_event_loop().run_until_complete(
            adapter._register_webhook()
        )
        assert ok is True
        assert not post_called, "Should reuse existing, not POST again"


    # -- _unregister_webhook --


    def test_unregister_removes_all_duplicates(self, monkeypatch):
        """Multiple orphaned registrations for same URL — all get removed."""
        import asyncio
        adapter = _make_adapter(monkeypatch)
        url = adapter._webhook_register_url
        deleted_ids = []

        async def mock_delete(*args, **kwargs):
            # Extract ID from URL
            url_str = args[0] if args else ""
            deleted_ids.append(url_str)
            class R:
                status_code = 200
                def raise_for_status(self):
                    pass
            return R()

        adapter.client = self._mock_client(
            get_response={"status": 200, "data": [
                {"id": 1, "url": url},
                {"id": 2, "url": url},
                {"id": 3, "url": "http://other/hook"},
            ]},
        )
        adapter.client.delete = mock_delete

        ok = asyncio.get_event_loop().run_until_complete(
            adapter._unregister_webhook()
        )
        assert ok is True
        assert len(deleted_ids) == 2


