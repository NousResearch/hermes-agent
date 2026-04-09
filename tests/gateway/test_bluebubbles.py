"""Tests for the BlueBubbles iMessage gateway adapter."""
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


class TestBlueBubblesPlatformEnum:
    def test_bluebubbles_enum_exists(self):
        assert Platform.BLUEBUBBLES.value == "bluebubbles"


class TestBlueBubblesConfigLoading:
    def test_apply_env_overrides_bluebubbles(self, monkeypatch):
        monkeypatch.setenv("BLUEBUBBLES_SERVER_URL", "http://localhost:1234")
        monkeypatch.setenv("BLUEBUBBLES_PASSWORD", "secret")
        monkeypatch.setenv("BLUEBUBBLES_WEBHOOK_PORT", "9999")
        from gateway.config import GatewayConfig, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)
        assert Platform.BLUEBUBBLES in config.platforms
        bc = config.platforms[Platform.BLUEBUBBLES]
        assert bc.enabled is True
        assert bc.extra["server_url"] == "http://localhost:1234"
        assert bc.extra["password"] == "secret"
        assert bc.extra["webhook_port"] == 9999

    def test_connected_platforms_includes_bluebubbles(self, monkeypatch):
        monkeypatch.setenv("BLUEBUBBLES_SERVER_URL", "http://localhost:1234")
        monkeypatch.setenv("BLUEBUBBLES_PASSWORD", "secret")
        from gateway.config import GatewayConfig, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)
        assert Platform.BLUEBUBBLES in config.get_connected_platforms()

    def test_home_channel_set_from_env(self, monkeypatch):
        monkeypatch.setenv("BLUEBUBBLES_SERVER_URL", "http://localhost:1234")
        monkeypatch.setenv("BLUEBUBBLES_PASSWORD", "secret")
        monkeypatch.setenv("BLUEBUBBLES_HOME_CHANNEL", "user@example.com")
        from gateway.config import GatewayConfig, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)
        hc = config.platforms[Platform.BLUEBUBBLES].home_channel
        assert hc is not None
        assert hc.chat_id == "user@example.com"

    def test_not_connected_without_password(self, monkeypatch):
        monkeypatch.setenv("BLUEBUBBLES_SERVER_URL", "http://localhost:1234")
        monkeypatch.delenv("BLUEBUBBLES_PASSWORD", raising=False)
        from gateway.config import GatewayConfig, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)
        assert Platform.BLUEBUBBLES not in config.get_connected_platforms()


class TestBlueBubblesHelpers:
    def test_check_requirements(self, monkeypatch):
        monkeypatch.setenv("BLUEBUBBLES_SERVER_URL", "http://localhost:1234")
        monkeypatch.setenv("BLUEBUBBLES_PASSWORD", "secret")
        from gateway.platforms.bluebubbles import check_bluebubbles_requirements

        assert check_bluebubbles_requirements() is True

    def test_format_message_strips_markdown(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        assert adapter.format_message("**Hello** `world`") == "Hello world"

    def test_strip_markdown_headers(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        assert adapter.format_message("## Heading\ntext") == "Heading\ntext"

    def test_strip_markdown_links(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        assert adapter.format_message("[click here](http://example.com)") == "click here"

    def test_init_normalizes_webhook_path(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, webhook_path="bluebubbles-webhook")
        assert adapter.webhook_path == "/bluebubbles-webhook"

    def test_init_preserves_leading_slash(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, webhook_path="/my-hook")
        assert adapter.webhook_path == "/my-hook"

    def test_server_url_normalized(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, server_url="http://localhost:1234/")
        assert adapter.server_url == "http://localhost:1234"

    def test_server_url_adds_scheme(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, server_url="localhost:1234")
        assert adapter.server_url == "http://localhost:1234"


class TestBlueBubblesWebhookParsing:
    def test_webhook_prefers_chat_guid_over_message_guid(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        payload = {
            "guid": "MESSAGE-GUID",
            "chatGuid": "iMessage;-;user@example.com",
            "chatIdentifier": "user@example.com",
        }
        record = adapter._extract_payload_record(payload) or {}
        chat_guid = adapter._value(
            record.get("chatGuid"),
            payload.get("chatGuid"),
            record.get("chat_guid"),
            payload.get("chat_guid"),
            payload.get("guid"),
        )
        assert chat_guid == "iMessage;-;user@example.com"

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

    def test_extract_payload_record_dict_data(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        payload = {"data": {"text": "hello", "chatGuid": "iMessage;-;+1234"}}
        record = adapter._extract_payload_record(payload)
        assert record["text"] == "hello"

    def test_extract_payload_record_fallback_to_message(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        payload = {"message": {"text": "hello"}}
        record = adapter._extract_payload_record(payload)
        assert record["text"] == "hello"


class TestBlueBubblesGuidResolution:
    def test_raw_guid_returned_as_is(self, monkeypatch):
        """If target already contains ';' it's a raw GUID — return unchanged."""
        adapter = _make_adapter(monkeypatch)
        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            adapter._resolve_chat_guid("iMessage;-;user@example.com")
        )
        assert result == "iMessage;-;user@example.com"

    def test_empty_target_returns_none(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            adapter._resolve_chat_guid("")
        )
        assert result is None


class TestBlueBubblesToolsetIntegration:
    def test_toolset_exists(self):
        from toolsets import TOOLSETS

        assert "hermes-bluebubbles" in TOOLSETS

    def test_toolset_in_gateway_composite(self):
        from toolsets import TOOLSETS

        gateway = TOOLSETS["hermes-gateway"]
        assert "hermes-bluebubbles" in gateway["includes"]


class TestBlueBubblesPromptHint:
    def test_platform_hint_exists(self):
        from agent.prompt_builder import PLATFORM_HINTS

        assert "bluebubbles" in PLATFORM_HINTS
        hint = PLATFORM_HINTS["bluebubbles"]
        assert "iMessage" in hint
        assert "plain text" in hint


class TestBlueBubblesAttachmentDownload:
    """Verify _download_attachment routes to the correct cache helper."""

    def test_download_image_uses_image_cache(self, monkeypatch):
        """Image MIME routes to cache_image_from_bytes."""
        adapter = _make_adapter(monkeypatch)
        import asyncio
        import httpx

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

    def test_download_audio_uses_audio_cache(self, monkeypatch):
        """Audio MIME routes to cache_audio_from_bytes."""
        adapter = _make_adapter(monkeypatch)
        import asyncio

        class MockResponse:
            status_code = 200
            content = b"fake-audio-data"

            def raise_for_status(self):
                pass

        async def mock_get(*args, **kwargs):
            return MockResponse()

        adapter.client = type("MockClient", (), {"get": mock_get})()

        cached_path = None

        def mock_cache_audio(data, ext):
            nonlocal cached_path
            cached_path = f"/tmp/test_audio{ext}"
            return cached_path

        monkeypatch.setattr(
            "gateway.platforms.bluebubbles.cache_audio_from_bytes",
            mock_cache_audio,
        )

        att_meta = {"mimeType": "audio/mpeg", "transferName": "voice.mp3"}
        result = asyncio.get_event_loop().run_until_complete(
            adapter._download_attachment("att-guid-456", att_meta)
        )
        assert result == "/tmp/test_audio.mp3"

    def test_download_document_uses_document_cache(self, monkeypatch):
        """Non-image/audio MIME routes to cache_document_from_bytes."""
        adapter = _make_adapter(monkeypatch)
        import asyncio

        class MockResponse:
            status_code = 200
            content = b"fake-doc-data"

            def raise_for_status(self):
                pass

        async def mock_get(*args, **kwargs):
            return MockResponse()

        adapter.client = type("MockClient", (), {"get": mock_get})()

        cached_path = None

        def mock_cache_doc(data, filename):
            nonlocal cached_path
            cached_path = f"/tmp/{filename}"
            return cached_path

        monkeypatch.setattr(
            "gateway.platforms.bluebubbles.cache_document_from_bytes",
            mock_cache_doc,
        )

        att_meta = {"mimeType": "application/pdf", "transferName": "report.pdf"}
        result = asyncio.get_event_loop().run_until_complete(
            adapter._download_attachment("att-guid-789", att_meta)
        )
        assert result == "/tmp/report.pdf"

    def test_download_returns_none_without_client(self, monkeypatch):
        """No client → returns None gracefully."""
        adapter = _make_adapter(monkeypatch)
        adapter.client = None
        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            adapter._download_attachment("att-guid", {"mimeType": "image/png"})
        )
        assert result is None
