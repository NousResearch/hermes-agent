"""Tests for the BlueBubbles iMessage gateway adapter."""
import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter


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


def _record_sleeps(monkeypatch):
    """Capture every ``asyncio.sleep`` delay; the fake still yields to the loop once per call.

    bluebubbles reaches sleep through the module attribute, so patching the module covers it. Use
    the returned ``real_sleep`` for the test's own yields so they are not recorded — a non-empty
    record then proves the code under test actually slept."""
    real_sleep = asyncio.sleep
    slept = []

    async def fake_sleep(delay):
        slept.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    return slept, real_sleep


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

        async def mock_cache_image(data, ext):
            nonlocal cached_path
            cached_path = f"/tmp/test_image{ext}"
            return cached_path

        monkeypatch.setattr(
            "gateway.platforms.bluebubbles.cache_image_from_bytes_async",
            mock_cache_image,
        )

        att_meta = {"mimeType": "image/png", "transferName": "photo.png"}
        result = asyncio.get_event_loop().run_until_complete(
            adapter._download_attachment("att-guid-123", att_meta)
        )
        assert result == "/tmp/test_image.png"


class TestBlueBubblesAttachmentSend:
    @pytest.mark.asyncio
    async def test_attachment_payload_is_read_before_async_upload(self, monkeypatch, tmp_path):
        adapter = _make_adapter(monkeypatch)
        file_path = tmp_path / "payload.bin"
        payload = b"attachment-payload"
        file_path.write_bytes(payload)

        captured = {}

        async def fake_resolve_chat_guid(chat_id):
            return "iMessage;+;chat-guid"

        class MockResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"status": 200, "data": {"guid": "message-guid"}}

        class MockClient:
            async def post(self, url, *, files, data, timeout):
                captured.update(url=url, files=files, data=data, timeout=timeout)
                return MockResponse()

        monkeypatch.setattr(adapter, "_resolve_chat_guid", fake_resolve_chat_guid)
        adapter.client = MockClient()

        result = await adapter._send_attachment(
            "target", str(file_path), filename="payload.bin"
        )

        assert result.success is True
        assert captured["files"]["attachment"] == (
            "payload.bin",
            payload,
            "application/octet-stream",
        )
        assert captured["data"]["chatGuid"] == "iMessage;+;chat-guid"


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


# ---------------------------------------------------------------------------
# Regression for #78183: httpx timeout exceptions stringify to "" which
# defeats _is_timeout_error, causing the plain-text fallback to re-send an
# already-delivered message (duplicate delivery).
# ---------------------------------------------------------------------------

class TestBlueBubblesTimeoutErrorNormalization:
    """When an httpx timeout has an empty string representation, the adapter
    must fall back to the exception type name so the base-layer timeout guard
    can still recognise it."""

    @pytest.mark.asyncio
    async def test_send_read_timeout_produces_matchable_error(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)

        async def fake_resolve(chat_id):
            return "iMessage;+;chat-123"
        monkeypatch.setattr(adapter, "_resolve_chat_guid", fake_resolve)

        async def fake_api_post(path, payload):
            raise httpx.ReadTimeout("")
        monkeypatch.setattr(adapter, "_api_post", fake_api_post)

        result = await adapter.send("chat-1", "hello world")

        assert not result.success
        assert result.error, "error must not be empty"
        assert BasePlatformAdapter._is_timeout_error(result.error), (
            f"_is_timeout_error must recognise {result.error!r}"
        )


    @pytest.mark.asyncio
    async def test_create_chat_for_handle_timeout_produces_matchable_error(
        self, monkeypatch,
    ):
        """Sibling call path — _create_chat_for_handle has the same
        error=str(exc) pattern and must also preserve the exception type."""
        adapter = _make_adapter(monkeypatch)

        async def fake_api_post(path, payload):
            raise httpx.ReadTimeout("")
        monkeypatch.setattr(adapter, "_api_post", fake_api_post)

        result = await adapter._create_chat_for_handle("test@example.com", "hi")

        assert not result.success
        assert result.error
        assert BasePlatformAdapter._is_timeout_error(result.error)

    @pytest.mark.asyncio
    async def test_non_empty_error_string_is_unchanged(self, monkeypatch):
        """A normal exception with a message must keep its original text."""
        adapter = _make_adapter(monkeypatch)

        async def fake_resolve(chat_id):
            return "iMessage;+;chat-123"
        monkeypatch.setattr(adapter, "_resolve_chat_guid", fake_resolve)

        async def fake_api_post(path, payload):
            raise RuntimeError("Server error '500 Internal Server Error'")
        monkeypatch.setattr(adapter, "_api_post", fake_api_post)

        result = await adapter.send("chat-1", "hello world")

        assert not result.success
        assert "500 Internal Server Error" in (result.error or "")




class TestBlueBubblesGateBeforeDownload:
    """The require_mention gate must run BEFORE attachments are downloaded (review follow-up)."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("text, downloads, handled_count", [
        ("look at this", 0, 0),          # unmentioned group attachment: never fetched
        ("hermes look at this", 1, 1),   # mentioned: fetched and dispatched
    ])
    async def test_unmentioned_group_attachment_is_not_downloaded(
            self, monkeypatch, text, downloads, handled_count):
        adapter = _make_adapter(monkeypatch, require_mention=True, send_read_receipts=False)
        handled = []

        async def fake_handle_message(event):
            event._gateway_accepted = True
            handled.append(event)

        download = AsyncMock(return_value="/tmp/cached.jpg")
        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        monkeypatch.setattr(adapter, "_download_attachment", download)
        response = await adapter._handle_webhook(_FakeBlueBubblesRequest({
            "type": "new-message",
            "data": {
                "guid": "msg-att-1",
                "text": text,
                "handle": {"address": "+15555550100"},
                "isFromMe": False,
                "isGroup": True,
                "chats": [{"guid": "iMessage;+;group-chat"}],
                "attachments": [{"guid": "att-1", "mimeType": "image/jpeg"}],
            },
        }))
        await asyncio.sleep(0)

        assert response.status == 200
        assert download.await_count == downloads
        assert len(handled) == handled_count


class TestBlueBubblesAttachmentDownloadInBand:
    """_download_attachment makes exactly one in-band attempt; retries live in background recovery."""

    @pytest.mark.asyncio
    async def test_download_makes_single_attempt_without_sleeping(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        slept, _ = _record_sleeps(monkeypatch)
        calls = {"count": 0}

        async def mock_get(*args, **kwargs):
            calls["count"] += 1
            raise RuntimeError("500 Internal Server Error")

        adapter.client = type("MockClient", (), {"get": mock_get})()

        result = await adapter._download_attachment("att-single-1", {"mimeType": "image/jpeg"})
        assert result is None
        assert calls["count"] == 1  # no in-band retry loop: it would block the webhook response
        assert slept == []  # the webhook path never waits on a retry budget

    @pytest.mark.asyncio
    async def test_download_success_returns_cached_path(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        calls = {"count": 0}

        class MockResponse:
            status_code = 200
            content = b"png-bytes"

            def raise_for_status(self):
                pass

        async def mock_get(*args, **kwargs):
            calls["count"] += 1
            return MockResponse()

        adapter.client = type("MockClient", (), {"get": mock_get})()

        async def mock_cache_image(data, ext):
            return f"/tmp/cached{ext}"

        monkeypatch.setattr(
            "gateway.platforms.bluebubbles.cache_image_from_bytes_async",
            mock_cache_image,
        )

        result = await adapter._download_attachment(
            "att-ok-1", {"mimeType": "image/png", "transferName": "photo.png"})
        assert result == "/tmp/cached.png"
        assert calls["count"] == 1

    @pytest.mark.asyncio
    async def test_download_without_client_returns_none(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter.client = None
        assert await adapter._download_attachment("att-1", {}) is None


class TestBlueBubblesAttachmentFailureDelivery:
    """A failed attachment download must not drop the message or stay silent."""

    @pytest.mark.asyncio
    async def test_image_only_message_with_failed_download_is_delivered(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        handled = []

        async def fake_handle_message(event):
            event._gateway_accepted = True
            handled.append(event)

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        monkeypatch.setattr(adapter, "_download_attachment", AsyncMock(return_value=None))
        recovery = AsyncMock()
        monkeypatch.setattr(adapter, "_recover_late_attachments", recovery)

        response = await adapter._handle_webhook(_FakeBlueBubblesRequest({
            "type": "new-message",
            "data": {
                "guid": "msg-att-fail-1",
                "text": "",
                "handle": {"address": "+155****0100"},
                "isFromMe": False,
                "chats": [{"guid": "iMessage;+;dm-chat"}],
                "attachments": [{"guid": "att-fail-1", "mimeType": "image/heic",
                                 "transferName": "IMG_0001.HEIC"}],
            },
        }))
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert response.status == 200
        assert len(handled) == 1
        assert "attachment download failed" in handled[0].text
        assert "IMG_0001.HEIC" in handled[0].text
        assert handled[0].media_urls == []
        assert recovery.await_count == 1

    @pytest.mark.asyncio
    async def test_text_message_with_failed_download_gets_notice_appended(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        handled = []

        async def fake_handle_message(event):
            event._gateway_accepted = True
            handled.append(event)

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        monkeypatch.setattr(adapter, "_download_attachment", AsyncMock(return_value=None))
        monkeypatch.setattr(adapter, "_recover_late_attachments", AsyncMock())

        response = await adapter._handle_webhook(_FakeBlueBubblesRequest({
            "type": "new-message",
            "data": {
                "guid": "msg-att-fail-2",
                "text": "look at this",
                "handle": {"address": "+155****0100"},
                "isFromMe": False,
                "chats": [{"guid": "iMessage;+;dm-chat"}],
                "attachments": [{"guid": "att-fail-2", "mimeType": "image/jpeg"}],
            },
        }))
        await asyncio.sleep(0)

        assert response.status == 200
        assert len(handled) == 1
        assert handled[0].text.startswith("look at this\n[attachment download failed:")

    @pytest.mark.asyncio
    async def test_successful_attachment_delivers_without_notice_or_recovery(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        handled = []

        async def fake_handle_message(event):
            event._gateway_accepted = True
            handled.append(event)

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        monkeypatch.setattr(adapter, "_download_attachment", AsyncMock(return_value="/tmp/cached.jpg"))
        recovery = AsyncMock()
        monkeypatch.setattr(adapter, "_recover_late_attachments", recovery)

        response = await adapter._handle_webhook(_FakeBlueBubblesRequest({
            "type": "new-message",
            "data": {
                "guid": "msg-att-ok-1",
                "text": "",
                "handle": {"address": "+155****0100"},
                "isFromMe": False,
                "chats": [{"guid": "iMessage;+;dm-chat"}],
                "attachments": [{"guid": "att-ok-1", "mimeType": "image/jpeg"}],
            },
        }))
        await asyncio.sleep(0)

        assert response.status == 200
        assert len(handled) == 1
        assert handled[0].media_urls == ["/tmp/cached.jpg"]
        assert "failed" not in handled[0].text
        assert recovery.await_count == 0
        assert "att-ok-1" in adapter._late_recovery_done
        assert "att-ok-1" not in adapter._delivery_inflight

    @pytest.mark.asyncio
    async def test_failed_attachment_never_sleeps_on_the_webhook_path(self, monkeypatch):
        """The webhook must not wait out retry backoffs: one in-band attempt, then 200.

        Retries are the background task's job. A gutted retry (or a re-added sleep) shows up
        here directly because the recorder sees every asyncio.sleep the code makes."""
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        slept, real_sleep = _record_sleeps(monkeypatch)
        handled = []

        async def fake_handle_message(event):
            event._gateway_accepted = True
            handled.append(event)

        download = AsyncMock(return_value=None)
        recovery = AsyncMock()
        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        monkeypatch.setattr(adapter, "_download_attachment", download)
        monkeypatch.setattr(adapter, "_recover_late_attachments", recovery)

        response = await adapter._handle_webhook(_FakeBlueBubblesRequest({
            "type": "new-message",
            "data": {
                "guid": "msg-fast-fail-1",
                "text": "",
                "handle": {"address": "+155****0100"},
                "isFromMe": False,
                "chats": [{"guid": "iMessage;+;dm-chat"}],
                "attachments": [{"guid": "att-fast-fail-1", "mimeType": "image/jpeg"}],
            },
        }))
        await real_sleep(0)
        await real_sleep(0)

        assert response.status == 200
        assert download.await_count == 1  # exactly one in-band attempt
        assert slept == []  # no retry budget spent before the HTTP response
        assert len(handled) == 1
        assert "attachment download failed" in handled[0].text

    @pytest.mark.asyncio
    async def test_failed_attachment_is_scheduled_for_background_recovery(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        handled = []

        async def fake_handle_message(event):
            event._gateway_accepted = True
            handled.append(event)

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        monkeypatch.setattr(adapter, "_download_attachment", AsyncMock(return_value=None))
        recovery = AsyncMock()
        monkeypatch.setattr(adapter, "_recover_late_attachments", recovery)

        response = await adapter._handle_webhook(_FakeBlueBubblesRequest({
            "type": "new-message",
            "data": {
                "guid": "msg-sched-1",
                "text": "",
                "handle": {"address": "+155****0100"},
                "isFromMe": False,
                "chats": [{"guid": "iMessage;+;dm-chat"}],
                "attachments": [{"guid": "att-sched-1", "mimeType": "image/jpeg"}],
            },
        }))
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert response.status == 200
        assert "att-sched-1" in adapter._late_recovery_pending
        assert recovery.await_count == 1

    def test_attachment_failure_notice_redacts_contact_info(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        notice = adapter._attachment_failure_notice([
            {"guid": "att-1", "transferName": "IMG_+8613812345678.jpg"},
            {"guid": "att-2", "transferName": "IMG_user@example.com.heic"},
        ])
        assert "+8613812345678" not in notice
        assert "user@example.com" not in notice
        assert "[REDACTED]" in notice


class TestBlueBubblesLateAttachmentRecovery:
    """Background recovery for attachments that sync to disk after the webhook was handled."""

    @staticmethod
    def _source(adapter):
        return adapter.build_source(chat_id="iMessage;+;dm-chat", chat_name="+155****0100",
                                    chat_type="dm", user_id="+155****0100", user_name="+155****0100")

    @pytest.mark.asyncio
    async def test_late_recovery_delivers_follow_up(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        adapter.client = object()  # non-None: the recovery loop requires a live client
        slept, real_sleep = _record_sleeps(monkeypatch)
        handled = []

        async def fake_handle_message(event):
            event._gateway_accepted = True
            handled.append(event)

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        fetch = AsyncMock(side_effect=[RuntimeError("still syncing"), "/tmp/late.jpg"])
        monkeypatch.setattr(adapter, "_fetch_attachment_once", fetch)
        adapter._late_recovery_pending.add("att-late-1")

        await adapter._recover_late_attachments(
            [{"guid": "att-late-1", "mimeType": "image/jpeg", "transferName": "late.jpg"}],
            source=self._source(adapter), reply_to_message_id="msg-42")
        await real_sleep(0)
        await real_sleep(0)

        assert fetch.await_count == 2
        assert slept == [5.0, 20.0]  # real backoff constants: first retry does not wait 90s
        assert len(handled) == 1
        assert handled[0].media_urls == ["/tmp/late.jpg"]
        assert handled[0].media_types == ["image/jpeg"]
        assert "late attachment" in handled[0].text
        assert handled[0].reply_to_message_id == "msg-42"
        assert "att-late-1" in adapter._late_recovery_done
        assert "att-late-1" not in adapter._late_recovery_pending
        assert "att-late-1" not in adapter._delivery_inflight

    @pytest.mark.asyncio
    async def test_late_recovery_backoff_sequence_then_permanent_failure(self, monkeypatch):
        """Transport failures walk the real retry budget; a ValueError is dropped, not retried."""
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        adapter.client = object()
        # Bound the window so a broken retry loop fails the test instead of spinning for the
        # real 15-minute window (the fake sleeps below never advance the clock).
        monkeypatch.setattr("gateway.platforms.bluebubbles._LATE_ATTACHMENT_RECOVERY_WINDOW_S", 5.0)
        slept, real_sleep = _record_sleeps(monkeypatch)
        handled = []

        async def fake_handle_message(event):
            event._gateway_accepted = True
            handled.append(event)

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        fetch = AsyncMock(side_effect=[
            RuntimeError("still syncing"),
            RuntimeError("still syncing"),
            RuntimeError("still syncing"),
            ValueError("Refusing to cache non-image data"),
        ])
        monkeypatch.setattr(adapter, "_fetch_attachment_once", fetch)
        adapter._late_recovery_pending.add("att-backoff-1")

        await adapter._recover_late_attachments(
            [{"guid": "att-backoff-1", "mimeType": "image/jpeg", "transferName": "photo.jpg"}],
            source=self._source(adapter))
        await real_sleep(0)

        assert slept == [5.0, 20.0, 60.0, 90.0]  # (5, 20, 60) budget, then the 90s poll interval
        assert fetch.await_count == 4
        assert handled == []  # nothing was ever delivered
        assert adapter._late_recovery_pending == set()
        assert "att-backoff-1" not in adapter._late_recovery_done

    @pytest.mark.asyncio
    async def test_late_recovery_value_error_gives_up_immediately(self, monkeypatch):
        """A permanent failure on the first attempt: one wait, one fetch, then dropped."""
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        adapter.client = object()
        # Bound the window so a broken retry loop fails the test instead of spinning for the
        # real 15-minute window (the fake sleeps below never advance the clock).
        monkeypatch.setattr("gateway.platforms.bluebubbles._LATE_ATTACHMENT_RECOVERY_WINDOW_S", 5.0)
        slept, _ = _record_sleeps(monkeypatch)
        fetch = AsyncMock(side_effect=ValueError("Inbound image payload is too large"))
        monkeypatch.setattr(adapter, "_fetch_attachment_once", fetch)
        adapter._late_recovery_pending.add("att-oversize-1")

        await adapter._recover_late_attachments(
            [{"guid": "att-oversize-1", "mimeType": "image/png"}], source=self._source(adapter))

        assert slept == [5.0]  # one backoff wait, then giving up
        assert fetch.await_count == 1  # never retried
        assert adapter._late_recovery_pending == set()

    @pytest.mark.asyncio
    async def test_late_recovery_stops_after_window(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        adapter.client = object()
        monkeypatch.setattr("gateway.platforms.bluebubbles._LATE_ATTACHMENT_RECOVERY_WINDOW_S", 0.05)
        slept, real_sleep = _record_sleeps(monkeypatch)
        handled = []

        async def fake_handle_message(event):
            event._gateway_accepted = True
            handled.append(event)

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)

        async def slow_fail(guid, att):
            await real_sleep(0.2)  # the fetch outlives the recovery window
            raise RuntimeError("never synced")

        fetch = AsyncMock(side_effect=slow_fail)
        monkeypatch.setattr(adapter, "_fetch_attachment_once", fetch)
        adapter._late_recovery_pending.add("att-late-2")

        await adapter._recover_late_attachments(
            [{"guid": "att-late-2", "mimeType": "image/jpeg"}], source=self._source(adapter))

        assert slept == [5.0]
        assert fetch.await_count == 1  # the window expired while the first fetch ran
        assert handled == []
        assert "att-late-2" not in adapter._late_recovery_pending

    @pytest.mark.asyncio
    async def test_late_recovery_skips_already_delivered_guid(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        adapter.client = object()
        slept, _ = _record_sleeps(monkeypatch)
        handled = []

        async def fake_handle_message(event):
            event._gateway_accepted = True
            handled.append(event)

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        fetch = AsyncMock(return_value="/tmp/dup.jpg")
        monkeypatch.setattr(adapter, "_fetch_attachment_once", fetch)
        adapter._remember_delivered_attachment("att-done-1")

        await adapter._recover_late_attachments(
            [{"guid": "att-done-1", "mimeType": "image/jpeg"}], source=self._source(adapter))

        assert slept == [5.0]
        assert fetch.await_count == 0
        assert handled == []
        assert "att-done-1" in adapter._late_recovery_done

    def test_delivered_guid_lru_evicts_oldest(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        monkeypatch.setattr("gateway.platforms.bluebubbles._LATE_ATTACHMENT_RECOVERY_CAP", 3)
        for guid in ("g1", "g2", "g3", "g4"):
            adapter._remember_delivered_attachment(guid)
        assert list(adapter._late_recovery_done) == ["g2", "g3", "g4"]
        adapter._remember_delivered_attachment("g2")  # refresh moves the guid to the end
        assert list(adapter._late_recovery_done) == ["g3", "g4", "g2"]

    @pytest.mark.asyncio
    async def test_late_delivery_notice_redacts_contact_info(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        handled = []

        async def fake_handle_message(event):
            event._gateway_accepted = True
            handled.append(event)

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)

        await adapter._deliver_late_attachments(
            [("/tmp/late.jpg", "image/jpeg",
              {"guid": "att-redact-1", "mimeType": "image/jpeg",
               "transferName": "IMG_+8613812345678.jpg"})],
            source=self._source(adapter))
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert len(handled) == 1
        assert "+8613812345678" not in handled[0].text
        assert "[REDACTED]" in handled[0].text


class TestBlueBubblesAttachmentDeduplication:
    """Attachment delivery claims are shared between the webhook path and late recovery."""

    @pytest.mark.asyncio
    async def test_replayed_event_skips_claimed_attachments(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        adapter._remember_delivered_attachment("att-done")
        adapter._delivery_inflight.add("att-inflight")
        handled = []

        async def fake_handle_message(event):
            event._gateway_accepted = True
            handled.append(event)

        download = AsyncMock(return_value="/tmp/fresh.jpg")
        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        monkeypatch.setattr(adapter, "_download_attachment", download)

        response = await adapter._handle_webhook(_FakeBlueBubblesRequest({
            "type": "updated-message",  # BlueBubbles replays messages through this event too
            "data": {
                "guid": "msg-replay-1",
                "text": "same message again",
                "handle": {"address": "+155****0100"},
                "isFromMe": False,
                "chats": [{"guid": "iMessage;+;dm-chat"}],
                "attachments": [
                    {"guid": "att-done", "mimeType": "image/jpeg"},
                    {"guid": "att-inflight", "mimeType": "image/jpeg"},
                    {"guid": "att-fresh", "mimeType": "image/jpeg"},
                ],
            },
        }))
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert response.status == 200
        assert download.await_count == 1  # claimed guids are not downloaded again
        assert download.await_args.args[0] == "att-fresh"
        assert len(handled) == 1
        assert handled[0].media_urls == ["/tmp/fresh.jpg"]  # ...and not re-delivered
        assert "failed" not in handled[0].text  # skipped guids are not counted as failed
        assert "att-inflight" in adapter._delivery_inflight  # someone else's claim is untouched
        assert "att-fresh" in adapter._late_recovery_done

    @pytest.mark.asyncio
    async def test_claim_is_inflight_until_gateway_accepts(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        handled = []

        async def accept(event):
            event._gateway_accepted = True
            handled.append(event)

        monkeypatch.setattr(adapter, "handle_message", accept)
        monkeypatch.setattr(adapter, "_download_attachment", AsyncMock(return_value="/tmp/a.jpg"))

        response = await adapter._handle_webhook(_FakeBlueBubblesRequest({
            "type": "new-message",
            "data": {
                "guid": "msg-accept-1",
                "text": "",
                "handle": {"address": "+155****0100"},
                "isFromMe": False,
                "chats": [{"guid": "iMessage;+;dm-chat"}],
                "attachments": [{"guid": "att-accept-1", "mimeType": "image/jpeg"}],
            },
        }))

        assert response.status == 200
        # claimed synchronously before dispatch: a concurrent replay cannot slip in
        assert "att-accept-1" in adapter._delivery_inflight
        assert "att-accept-1" not in adapter._late_recovery_done
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        # accepted by the gateway → the claim settles into the delivered LRU
        assert "att-accept-1" in adapter._late_recovery_done
        assert "att-accept-1" not in adapter._delivery_inflight
        assert len(handled) == 1

    @pytest.mark.asyncio
    async def test_rejected_delivery_releases_claim_for_replay(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        handled = []

        async def reject(event):
            handled.append(event)  # gateway did not accept it; _gateway_accepted stays False

        download = AsyncMock(return_value="/tmp/retry.jpg")
        monkeypatch.setattr(adapter, "handle_message", reject)
        monkeypatch.setattr(adapter, "_download_attachment", download)
        monkeypatch.setattr(adapter, "_recover_late_attachments", AsyncMock())

        payload = {
            "type": "new-message",
            "data": {
                "guid": "msg-reject-1",
                "text": "look at this",
                "handle": {"address": "+155****0100"},
                "isFromMe": False,
                "chats": [{"guid": "iMessage;+;dm-chat"}],
                "attachments": [{"guid": "att-replay-1", "mimeType": "image/jpeg"}],
            },
        }
        response = await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert response.status == 200
        assert "att-replay-1" not in adapter._late_recovery_done
        assert "att-replay-1" not in adapter._delivery_inflight

        # the replay can fetch and deliver the attachment again (it was never delivered)
        async def accept(event):
            event._gateway_accepted = True
            handled.append(event)

        monkeypatch.setattr(adapter, "handle_message", accept)
        response = await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert response.status == 200
        assert download.await_count == 2
        assert len(handled) == 2
        assert "att-replay-1" in adapter._late_recovery_done
        assert "att-replay-1" not in adapter._delivery_inflight

    @pytest.mark.asyncio
    async def test_claimed_failed_attachment_is_not_scheduled_for_recovery(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        adapter._remember_delivered_attachment("att-claimed-1")
        handled = []

        async def fake_handle_message(event):
            event._gateway_accepted = True
            handled.append(event)

        download = AsyncMock(return_value=None)
        recovery = AsyncMock()
        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        monkeypatch.setattr(adapter, "_download_attachment", download)
        monkeypatch.setattr(adapter, "_recover_late_attachments", recovery)

        response = await adapter._handle_webhook(_FakeBlueBubblesRequest({
            "type": "new-message",
            "data": {
                "guid": "msg-claimed-1",
                "text": "look at this",
                "handle": {"address": "+155****0100"},
                "isFromMe": False,
                "chats": [{"guid": "iMessage;+;dm-chat"}],
                "attachments": [{"guid": "att-claimed-1", "mimeType": "image/jpeg"}],
            },
        }))
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert response.status == 200
        assert download.await_count == 0  # already delivered: not fetched again
        assert adapter._late_recovery_pending == set()
        assert recovery.await_count == 0
        assert len(handled) == 1
