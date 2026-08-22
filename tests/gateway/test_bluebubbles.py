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
    """_webhook_url keeps advertise family consistent with TCPSite bind.

    IPv4 loopback/wildcard binds advertise ``127.0.0.1`` (macOS Node.js
    resolves bare localhost to ``::1`` first). IPv6 any-address binds
    advertise ``[::1]`` — never rewrite IPv6 bind → IPv4 register.
    """

    def test_default_host(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        # Default webhook_host is 127.0.0.1 → normalised to 127.0.0.1
        assert adapter._webhook_url.startswith("http://127.0.0.1:")
        assert str(adapter.webhook_port) in adapter._webhook_url
        assert adapter.webhook_path in adapter._webhook_url

    @pytest.mark.parametrize("host", ["0.0.0.0", "127.0.0.1", "localhost"])
    def test_ipv4_bind_hosts_advertise_127_0_0_1(self, monkeypatch, host):
        adapter = _make_adapter(monkeypatch, webhook_host=host)
        assert adapter._webhook_url.startswith("http://127.0.0.1:")

    @pytest.mark.parametrize("host", ["::", "::0", "[::]"])
    def test_ipv6_any_bind_advertises_loopback_v6(self, monkeypatch, host):
        """Bind family :: must not be rewritten to IPv4 127.0.0.1."""
        adapter = _make_adapter(monkeypatch, webhook_host=host)
        assert adapter._webhook_url.startswith("http://[::1]:")
        assert "127.0.0.1" not in adapter._webhook_url

    def test_custom_host_preserved(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, webhook_host="192.168.1.50")
        assert "192.168.1.50" in adapter._webhook_url

    def test_legacy_webhook_urls_returns_localhost_variant(self, monkeypatch):
        """When using a loopback host, _legacy_webhook_urls includes the old
        ``localhost``-based URL that a prior Hermes version would have
        registered.
        """
        adapter = _make_adapter(monkeypatch)
        # Default webhook_host is 127.0.0.1 → legacy was localhost
        assert len(adapter._legacy_webhook_urls) == 1
        assert adapter._legacy_webhook_urls[0].startswith("http://localhost:")

    def test_legacy_webhook_urls_empty_when_custom_host(self, monkeypatch):
        """A non-loopback host was never normalised to localhost."""
        adapter = _make_adapter(monkeypatch, webhook_host="192.168.1.50")
        assert adapter._legacy_webhook_urls == []

    def test_register_url_embeds_password(self, monkeypatch):
        """_webhook_register_url should append ?password=... for inbound auth."""
        adapter = _make_adapter(monkeypatch, password="secret123")
        assert adapter._webhook_register_url.endswith("?password=secret123")
        assert adapter._webhook_register_url.startswith(adapter._webhook_url)

    def test_register_url_url_encodes_password(self, monkeypatch):
        """Passwords with special characters must be URL-encoded."""
        adapter = _make_adapter(monkeypatch, password="W9fTC&L5JL*@")
        assert "password=W9fTC%26L5JL%2A%40" in adapter._webhook_register_url

    def test_register_url_for_log_masks_password(self, monkeypatch):
        """Log-safe webhook URLs must never expose the webhook password."""
        adapter = _make_adapter(monkeypatch, password="W9fTC&L5JL*@")
        safe_url = adapter._webhook_register_url_for_log
        assert safe_url.endswith("?password=***")
        assert "W9fTC" not in safe_url
        assert "%26" not in safe_url

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


    # -- legacy localhost migration --

    def test_register_migrates_legacy_localhost_webhook(self, monkeypatch):
        """When a stale ``localhost`` registration exists, _register_webhook
        removes it before creating the new ``127.0.0.1`` registration.
        This is the upgrade path from pre-IPv4-fix Hermes versions.
        """
        import asyncio
        adapter = _make_adapter(monkeypatch, password="secret123")
        current_url = adapter._webhook_register_url
        legacy_url = adapter._legacy_webhook_urls[0] + "?password=secret123"

        deleted_ids = []
        all_webhooks = [
            {"id": 42, "url": legacy_url, "events": ["new-message"]},
        ]

        async def mock_get(url, *args, **kwargs):
            class R:
                status_code = 200
                def raise_for_status(self): pass
                def json(self_inner):
                    return {"status": 200, "data": list(all_webhooks)}
            return R()

        async def mock_delete(url, *args, **kwargs):
            deleted_ids.append("42")
            all_webhooks.clear()
            class R:
                status_code = 200
                def raise_for_status(self): pass
            return R()

        async def mock_post(path, payload, *args, **kwargs):
            class R:
                status_code = 200
                def raise_for_status(self): pass
                def json(self_inner):
                    return {"status": 200, "data": {}}
            return R()

        adapter.client = type("MockClient", (), {
            "get": mock_get, "post": mock_post, "delete": mock_delete,
        })()

        ok = asyncio.get_event_loop().run_until_complete(
            adapter._register_webhook()
        )
        assert ok is True
        # The stale localhost registration should have been deleted
        assert len(deleted_ids) > 0, "Legacy localhost webhook should be migrated"

    
    def test_register_migrates_bare_legacy_localhost_without_password_query(
        self, monkeypatch
    ):
        """Pre-326cbbe registrations used bare localhost URLs (no ?password=).

        Even when the current install has a password, migration must still
        find and remove the bare historical form.
        """
        import asyncio

        adapter = _make_adapter(monkeypatch, password="secret123")
        bare_legacy = adapter._legacy_webhook_urls[0]
        assert "?" not in bare_legacy

        deleted_ids = []
        all_webhooks = [
            {"id": 77, "url": bare_legacy, "events": ["new-message"]},
        ]

        async def mock_get(url, *args, **kwargs):
            class R:
                status_code = 200

                def raise_for_status(self):
                    pass

                def json(self_inner):
                    return {"status": 200, "data": list(all_webhooks)}

            return R()

        async def mock_delete(url, *args, **kwargs):
            deleted_ids.append(77)
            all_webhooks.clear()

            class R:
                status_code = 200

                def raise_for_status(self):
                    pass

            return R()

        async def mock_post(path, payload, *args, **kwargs):
            class R:
                status_code = 200

                def raise_for_status(self):
                    pass

                def json(self_inner):
                    return {"status": 200, "data": {}}

            return R()

        adapter.client = type(
            "MockClient",
            (),
            {"get": mock_get, "post": mock_post, "delete": mock_delete},
        )()

        ok = asyncio.get_event_loop().run_until_complete(adapter._register_webhook())
        assert ok is True
        assert 77 in deleted_ids, "Bare pre-password localhost webhook must be migrated"

    def test_register_skips_migration_when_no_legacy(self, monkeypatch):
        """No stale registrations → migration is a no-op, normal register proceeds."""
        import asyncio
        adapter = _make_adapter(monkeypatch, webhook_host="192.168.1.50")
        # Custom host → _legacy_webhook_urls is empty → no migration

        current_url = adapter._webhook_register_url

        async def mock_get(path, *args, **kwargs):
            class R:
                status_code = 200
                def raise_for_status(self): pass
                def json(self_inner):
                    return {"status": 200, "data": []}
            return R()

        async def mock_post(path, payload, *args, **kwargs):
            class R:
                status_code = 200
                def raise_for_status(self): pass
                def json(self_inner):
                    return {"status": 200, "data": {}}
            return R()

        adapter.client = type("MockClient", (), {
            "get": mock_get, "post": mock_post, "delete": lambda *a, **k: None,
        })()

        ok = asyncio.get_event_loop().run_until_complete(
            adapter._register_webhook()
        )
        assert ok is True

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


