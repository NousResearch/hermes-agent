"""Tests for the BlueBubbles iMessage gateway adapter."""
import asyncio
import json

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


class TestBlueBubblesConnect:
    def test_connect_fails_when_webhook_registration_fails(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, webhook_port="0")

        async def api_get(path):
            if path == "/api/v1/server/info":
                return {"data": {"private_api": True, "helper_connected": True}}
            return {"status": 200}

        async def fail_register():
            return False

        monkeypatch.setattr(adapter, "_api_get", api_get)
        monkeypatch.setattr(adapter, "_register_webhook", fail_register)

        connected = asyncio.get_event_loop().run_until_complete(adapter.connect())

        assert connected is False
        assert adapter._runner is None
        assert adapter.client is None

    def test_connect_cleans_up_when_webhook_registration_raises(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, webhook_port="0")

        async def api_get(path):
            if path == "/api/v1/server/info":
                return {"data": {"private_api": True, "helper_connected": True}}
            return {"status": 200}

        async def raise_during_register():
            raise ValueError("malformed response")

        monkeypatch.setattr(adapter, "_api_get", api_get)
        monkeypatch.setattr(adapter, "_register_webhook", raise_during_register)

        connected = asyncio.get_event_loop().run_until_complete(adapter.connect())

        assert connected is False
        assert adapter._runner is None
        assert adapter.client is None

    def test_connect_does_not_log_server_password_on_api_failure(self, monkeypatch, caplog):
        adapter = _make_adapter(monkeypatch)

        async def fail_api(path):
            url = adapter._api_url(path)
            request = httpx.Request("GET", url)
            response = httpx.Response(500, request=request)
            raise httpx.HTTPStatusError(f"failed: {url}", request=request, response=response)

        monkeypatch.setattr(adapter, "_api_get", fail_api)

        connected = asyncio.get_event_loop().run_until_complete(adapter.connect())

        assert connected is False
        assert adapter.client is None
        assert "secret" not in caplog.text


class TestBlueBubblesWebhookUrl:
    def test_default_ipv4_listener_advertises_same_address_family(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)

        assert adapter._webhook_url == "http://127.0.0.1:8645/bluebubbles-webhook"

    @pytest.mark.parametrize(
        ("listener", "advertised"),
        [
            ("0.0.0.0", "127.0.0.1"),
            ("::", "[::1]"),
            ("::1", "[::1]"),
            ("localhost", "localhost"),
            ("bridge.internal", "bridge.internal"),
        ],
    )
    def test_callback_host_is_reachable_and_ipv6_safe(self, monkeypatch, listener, advertised):
        adapter = _make_adapter(monkeypatch, webhook_host=listener)

        assert adapter._webhook_url == f"http://{advertised}:8645/bluebubbles-webhook"


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


    def test_register_subscribes_to_creation_events_only(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        monkeypatch.setattr(adapter, "client", self._mock_client())
        payloads = []

        async def no_existing(url):
            return []

        async def capture_post(path, payload):
            payloads.append(payload)
            return {"status": 200, "data": {"id": 42}}

        monkeypatch.setattr(adapter, "_find_registered_webhooks", no_existing)
        monkeypatch.setattr(adapter, "_api_post", capture_post)

        ok = asyncio.get_event_loop().run_until_complete(adapter._register_webhook())

        assert ok is True
        assert payloads[0]["events"] == ["new-message"]


    def test_register_fails_closed_when_webhooks_cannot_be_listed(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        monkeypatch.setattr(adapter, "client", self._mock_client())
        posted_payloads = []

        async def fail_list(path):
            raise httpx.ConnectError("offline")

        async def capture_post(path, payload):
            posted_payloads.append(payload)
            return {"status": 200, "data": {"id": 42}}

        monkeypatch.setattr(adapter, "_api_get", fail_list)
        monkeypatch.setattr(adapter, "_api_post", capture_post)

        ok = asyncio.get_event_loop().run_until_complete(adapter._register_webhook())

        assert ok is False
        assert posted_payloads == []


    @pytest.mark.parametrize(
        "list_response",
        [
            {"status": 500, "data": []},
            {"status": 200, "data": {}},
        ],
    )
    def test_register_fails_closed_on_invalid_list_response(self, monkeypatch, list_response):
        adapter = _make_adapter(monkeypatch)
        monkeypatch.setattr(adapter, "client", self._mock_client())
        posted_payloads = []

        async def invalid_list(path):
            return list_response

        async def capture_post(path, payload):
            posted_payloads.append(payload)
            return {"status": 200, "data": {"id": 42}}

        monkeypatch.setattr(adapter, "_api_get", invalid_list)
        monkeypatch.setattr(adapter, "_api_post", capture_post)

        ok = asyncio.get_event_loop().run_until_complete(adapter._register_webhook())

        assert ok is False
        assert posted_payloads == []


    @pytest.mark.parametrize(
        "registration",
        [
            "not-an-object",
            {"url": "http://127.0.0.1:8645/bluebubbles-webhook?password=secret", "events": {}},
            {"url": "http://other/webhook", "events": ["new-message"]},
        ],
    )
    def test_register_fails_closed_on_malformed_registration(self, monkeypatch, registration):
        adapter = _make_adapter(monkeypatch)
        monkeypatch.setattr(adapter, "client", self._mock_client())
        posted_payloads = []

        async def malformed_list(path):
            return {"status": 200, "data": [registration]}

        async def capture_post(path, payload):
            posted_payloads.append(payload)
            return {"status": 200, "data": {"id": 42}}

        monkeypatch.setattr(adapter, "_api_get", malformed_list)
        monkeypatch.setattr(adapter, "_api_post", capture_post)

        ok = asyncio.get_event_loop().run_until_complete(adapter._register_webhook())

        assert ok is False
        assert posted_payloads == []


    def test_register_keeps_stale_callback_when_replacement_post_fails(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        current_url = adapter._webhook_register_url
        legacy_url = current_url.replace("127.0.0.1", "localhost")
        client = self._mock_client(
            get_response={"status": 200, "data": [
                {"id": 8, "url": legacy_url, "events": ["new-message"]},
            ]},
        )
        deleted_ids = []

        async def capture_delete(url):
            deleted_ids.append(url)
            return type("Response", (), {"raise_for_status": lambda self: None})()

        async def fail_post(path, payload):
            return {"status": 500, "message": "failed"}

        monkeypatch.setattr(client, "delete", capture_delete, raising=False)
        monkeypatch.setattr(adapter, "client", client)
        monkeypatch.setattr(adapter, "_api_post", fail_post)

        ok = asyncio.get_event_loop().run_until_complete(adapter._register_webhook())

        assert ok is False
        assert deleted_ids == []


    def test_register_rolls_back_new_callback_when_stale_delete_fails(self, monkeypatch, caplog):
        adapter = _make_adapter(monkeypatch)
        current_url = adapter._webhook_register_url
        legacy_url = current_url.replace("127.0.0.1", "localhost")
        client = self._mock_client(
            get_response={"status": 200, "data": [
                {"id": 8, "url": legacy_url, "events": ["new-message"]},
            ]},
            post_response={"status": 200, "data": {"id": 42}},
        )
        deleted_ids = []

        async def delete_with_stale_failure(url):
            webhook_id = url.rsplit("/", 1)[-1].split("?", 1)[0]
            deleted_ids.append(webhook_id)
            request = httpx.Request("DELETE", url)
            if webhook_id == "8":
                response = httpx.Response(500, request=request)
                raise httpx.HTTPStatusError(f"failed: {url}", request=request, response=response)
            return httpx.Response(204, request=request)

        monkeypatch.setattr(client, "delete", delete_with_stale_failure, raising=False)
        monkeypatch.setattr(adapter, "client", client)

        ok = asyncio.get_event_loop().run_until_complete(adapter._register_webhook())

        assert ok is False
        assert deleted_ids == ["8", "42"]
        assert "secret" not in caplog.text


    def test_register_replaces_stale_and_legacy_local_callbacks(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        current_url = adapter._webhook_register_url
        legacy_url = current_url.replace("127.0.0.1", "localhost")
        client = self._mock_client(
            get_response={"status": 200, "data": [
                {"id": 7, "url": current_url, "events": ["new-message", "updated-message"]},
                {"id": 8, "url": legacy_url, "events": ["new-message"]},
            ]},
        )
        deleted_ids = []
        posted_payloads = []

        async def capture_delete(url):
            deleted_ids.append(url.rsplit("/", 1)[-1].split("?", 1)[0])
            return type("Response", (), {"raise_for_status": lambda self: None})()

        async def capture_post(path, payload):
            posted_payloads.append(payload)
            return {"status": 200, "data": {"id": 42}}

        monkeypatch.setattr(client, "delete", capture_delete, raising=False)
        monkeypatch.setattr(adapter, "client", client)
        monkeypatch.setattr(adapter, "_api_post", capture_post)

        ok = asyncio.get_event_loop().run_until_complete(adapter._register_webhook())

        assert ok is True
        assert deleted_ids == ["7", "8"]
        assert posted_payloads == [{"url": current_url, "events": ["new-message"]}]


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
    async def test_send_write_timeout_produces_matchable_error(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)

        async def fake_resolve(chat_id):
            return "iMessage;+;chat-123"
        monkeypatch.setattr(adapter, "_resolve_chat_guid", fake_resolve)

        async def fake_api_post(path, payload):
            raise httpx.WriteTimeout("")
        monkeypatch.setattr(adapter, "_api_post", fake_api_post)

        result = await adapter.send("chat-1", "hello world")

        assert not result.success
        assert result.error
        assert BasePlatformAdapter._is_timeout_error(result.error)

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


