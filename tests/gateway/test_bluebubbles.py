"""Tests for the BlueBubbles iMessage gateway adapter."""
import asyncio
import json

import pytest

from gateway.config import Platform, PlatformConfig


def _make_adapter(monkeypatch, **extra):
    monkeypatch.setenv("BLUEBUBBLES_SERVER_URL", "http://localhost:1234")
    monkeypatch.setenv("BLUEBUBBLES_PASSWORD", "secret")
    monkeypatch.setenv("BLUEBUBBLES_WEBHOOK_HOST", "127.0.0.1")
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

    def test_supports_message_editing_is_false(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        assert adapter.SUPPORTS_MESSAGE_EDITING is False

    def test_truncate_message_omits_pagination_suffixes(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        chunks = adapter.truncate_message("abcdefghij", max_length=6)
        assert len(chunks) > 1
        assert "".join(chunks) == "abcdefghij"
        assert all("(" not in chunk for chunk in chunks)

    @pytest.mark.asyncio
    async def test_send_splits_paragraphs_into_multiple_bubbles(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        sent = []

        async def fake_resolve_chat_guid(chat_id):
            return "iMessage;-;user@example.com"

        async def fake_api_post(path, payload):
            sent.append(payload["message"])
            return {"data": {"guid": f"msg-{len(sent)}"}}

        monkeypatch.setattr(adapter, "_resolve_chat_guid", fake_resolve_chat_guid)
        monkeypatch.setattr(adapter, "_api_post", fake_api_post)

        result = await adapter.send("user@example.com", "first thought\n\nsecond thought")

        assert result.success is True
        assert sent == ["first thought", "second thought"]

    def test_format_message_strips_markdown(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        assert adapter.format_message("**Hello** `world`") == "Hello world"

    def test_format_message_preserves_underscores_in_identifiers(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        text = "Use /api_v2 with FEATURE_FLAG_NAME and config_file.json"
        assert adapter.format_message(text) == text

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

    def test_default_mention_patterns_match_hermes_variants(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, require_mention=True)

        assert adapter.require_mention is True
        assert adapter._message_matches_mention_patterns("Hermes, summarize this")
        assert adapter._message_matches_mention_patterns("@Hermes agent help")
        assert not adapter._message_matches_mention_patterns("casual family chatter")
        assert not adapter._message_matches_mention_patterns("antihermes should not match")

    def test_custom_mention_patterns_override_defaults(self, monkeypatch):
        adapter = _make_adapter(
            monkeypatch,
            require_mention=True,
            mention_patterns=[r"(?<![\w@])@?amos\b[,:\-]?"],
        )

        assert adapter._message_matches_mention_patterns("Amos what is next?")
        assert not adapter._message_matches_mention_patterns("Hermes what is next?")

    def test_clean_mention_text_strips_leading_wake_word(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, require_mention=True)

        assert adapter._clean_mention_text("Hermes, summarize this") == "summarize this"
        assert adapter._clean_mention_text("Hermes agent: summarize this") == "summarize this"
        assert adapter._clean_mention_text("please ask Hermes about this") == "please ask Hermes about this"


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

    @pytest.mark.asyncio
    async def test_group_message_with_default_mention_is_dispatched_cleaned(self, monkeypatch):
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
                "guid": "msg-2",
                "text": "Hermes, summarize this",
                "handle": {"address": "+15555550100"},
                "isFromMe": False,
                "isGroup": True,
                "chats": [{"guid": "iMessage;+;group-chat"}],
            },
        }))
        await asyncio.sleep(0)

        assert response.status == 200
        assert [event.text for event in handled] == ["summarize this"]

    @pytest.mark.asyncio
    async def test_dm_message_does_not_require_mention(self, monkeypatch):
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
                "guid": "msg-3",
                "text": "hello from a dm",
                "handle": {"address": "user@example.com"},
                "isFromMe": False,
                "chatGuid": "iMessage;-;user@example.com",
                "chatIdentifier": "user@example.com",
            },
        }))
        await asyncio.sleep(0)

        assert response.status == 200
        assert [event.text for event in handled] == ["hello from a dm"]


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

    def test_webhook_extracts_chat_guid_from_chats_array_dm(self, monkeypatch):
        """BB v1.9+ webhook payloads omit top-level chatGuid; GUID is in chats[0].guid."""
        adapter = _make_adapter(monkeypatch)
        payload = {
            "type": "new-message",
            "data": {
                "guid": "MESSAGE-GUID",
                "text": "hello",
                "handle": {"address": "+15551234567"},
                "isFromMe": False,
                "chats": [
                    {"guid": "any;-;+15551234567", "chatIdentifier": "+15551234567"}
                ],
            },
        }
        record = adapter._extract_payload_record(payload) or {}
        chat_guid = adapter._value(
            record.get("chatGuid"),
            payload.get("chatGuid"),
            record.get("chat_guid"),
            payload.get("chat_guid"),
            payload.get("guid"),
        )
        if not chat_guid:
            _chats = record.get("chats") or []
            if _chats and isinstance(_chats[0], dict):
                chat_guid = _chats[0].get("guid") or _chats[0].get("chatGuid")
        assert chat_guid == "any;-;+15551234567"

    def test_webhook_extracts_chat_guid_from_chats_array_group(self, monkeypatch):
        """Group chat GUIDs contain ;+; and must be extracted from chats array."""
        adapter = _make_adapter(monkeypatch)
        payload = {
            "type": "new-message",
            "data": {
                "guid": "MESSAGE-GUID",
                "text": "hello everyone",
                "handle": {"address": "+15551234567"},
                "isFromMe": False,
                "isGroup": True,
                "chats": [{"guid": "any;+;chat-uuid-abc123"}],
            },
        }
        record = adapter._extract_payload_record(payload) or {}
        chat_guid = adapter._value(
            record.get("chatGuid"),
            payload.get("chatGuid"),
            record.get("chat_guid"),
            payload.get("chat_guid"),
            payload.get("guid"),
        )
        if not chat_guid:
            _chats = record.get("chats") or []
            if _chats and isinstance(_chats[0], dict):
                chat_guid = _chats[0].get("guid") or _chats[0].get("chatGuid")
        assert chat_guid == "any;+;chat-uuid-abc123"

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

    @pytest.mark.asyncio
    async def test_exact_chat_identifier_match_returns_dm_guid(self, monkeypatch):
        """A 1:1 DM whose chatIdentifier equals the target resolves to its guid."""
        adapter = _make_adapter(monkeypatch)

        async def fake_api_post(path, payload):
            return {
                "data": [
                    {
                        "guid": "iMessage;-;user@example.com",
                        "chatIdentifier": "user@example.com",
                        "participants": [{"address": "user@example.com"}],
                    }
                ]
            }

        monkeypatch.setattr(adapter, "_api_post", fake_api_post)
        result = await adapter._resolve_chat_guid("user@example.com")
        assert result == "iMessage;-;user@example.com"

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
    async def test_dm_chosen_over_group_when_both_contain_contact(self, monkeypatch):
        """Even when a group chat is returned BEFORE a DM in the query result,
        the resolver must lock onto the DM by chatIdentifier and not the
        group via participant fallback.
        """
        adapter = _make_adapter(monkeypatch)

        async def fake_api_post(path, payload):
            return {
                "data": [
                    {
                        "guid": "iMessage;+;chat0000000000-family-group",
                        "chatIdentifier": "chat0000000000",
                        "participants": [{"address": "user@example.com"}],
                    },
                    {
                        "guid": "iMessage;-;user@example.com",
                        "chatIdentifier": "user@example.com",
                        "participants": [{"address": "user@example.com"}],
                    },
                ]
            }

        monkeypatch.setattr(adapter, "_api_post", fake_api_post)
        result = await adapter._resolve_chat_guid("user@example.com")
        assert result == "iMessage;-;user@example.com"

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

    @pytest.mark.parametrize("host", ["0.0.0.0", "127.0.0.1", "localhost", "::"])
    def test_local_hosts_normalized(self, monkeypatch, host):
        adapter = _make_adapter(monkeypatch, webhook_host=host)
        assert adapter._webhook_url.startswith("http://localhost:")

    def test_custom_host_preserved(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, webhook_host="192.168.1.50")
        assert "192.168.1.50" in adapter._webhook_url

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

    def test_find_registered_webhooks_empty_when_none(self, monkeypatch):
        import asyncio
        adapter = _make_adapter(monkeypatch)
        adapter.client = self._mock_client(
            get_response={"status": 200, "data": []}
        )
        result = asyncio.get_event_loop().run_until_complete(
            adapter._find_registered_webhooks(adapter._webhook_url)
        )
        assert result == []

    def test_find_registered_webhooks_reports_api_error(self, monkeypatch):
        import asyncio
        adapter = _make_adapter(monkeypatch)
        adapter.client = self._mock_client()

        # Override _api_get to raise
        async def bad_get(path):
            raise ConnectionError("server down")
        adapter._api_get = bad_get

        result = asyncio.get_event_loop().run_until_complete(
            adapter._find_registered_webhooks(adapter._webhook_url)
        )
        assert result is None

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

    def test_register_accepts_201(self, monkeypatch):
        """BB might return 201 Created — must still succeed."""
        import asyncio
        adapter = _make_adapter(monkeypatch)
        adapter.client = self._mock_client(
            get_response={"status": 200, "data": []},
            post_response={"status": 201, "data": {"id": 43}},
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

    @pytest.mark.asyncio
    async def test_register_migrates_realistic_idempotent_same_url_webhook(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch)
        url = adapter._webhook_register_url
        calls = []
        registrations = [
            {
                "id": 7,
                "url": url,
                "events": ["new-message", "updated-message"],
            }
        ]
        adapter.client = self._mock_client()
        original_delete = adapter.client.delete

        async def fake_find_registered_webhooks(candidate_url):
            return [item for item in registrations if item["url"] == candidate_url]

        async def tracking_post(path, payload):
            calls.append(("post", payload))
            # BlueBubbles addWebhook() returns an existing same-URL row
            # unchanged instead of updating its event list.
            existing = next(
                (item for item in registrations if item["url"] == payload["url"]),
                None,
            )
            if existing:
                return {"status": 200, "data": dict(existing)}
            created = {"id": 8, **payload}
            registrations.append(created)
            return {"status": 200, "data": dict(created)}

        async def tracking_delete(*args, **kwargs):
            calls.append(("delete", args[0]))
            registrations[:] = [item for item in registrations if item["id"] != 7]
            return await original_delete(*args, **kwargs)

        monkeypatch.setattr(
            adapter, "_find_registered_webhooks", fake_find_registered_webhooks
        )
        monkeypatch.setattr(adapter, "_api_post", tracking_post)
        adapter.client.delete = tracking_delete

        assert await adapter._register_webhook() is True
        assert calls == [
            ("delete", adapter._api_url("/api/v1/webhook/7")),
            ("post", {"url": url, "events": ["new-message"]}),
        ]
        assert registrations == [
            {"id": 8, "url": url, "events": ["new-message"]}
        ]
        assert adapter._owned_webhook_ids == set()

    @pytest.mark.asyncio
    async def test_reused_registration_is_not_owned_or_removed_on_disconnect(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch)
        url = adapter._webhook_register_url
        deleted = []
        adapter.client = self._mock_client(
            get_response={
                "status": 200,
                "data": [{"id": 7, "url": url, "events": ["new-message"]}],
            }
        )
        original_delete = adapter.client.delete

        async def tracking_delete(*args, **kwargs):
            deleted.append(args[0])
            return await original_delete(*args, **kwargs)

        adapter.client.delete = tracking_delete

        assert await adapter._register_webhook() is True
        assert await adapter._unregister_webhook() is False
        assert deleted == []

    @pytest.mark.asyncio
    async def test_cancelled_fresh_registration_post_settles_without_claiming_owner(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch)
        started = asyncio.Event()
        release = asyncio.Event()
        posted = []
        adapter.client = self._mock_client(get_response={"status": 200, "data": []})

        async def delayed_post(path, payload):
            posted.append((path, payload))
            started.set()
            await release.wait()
            return {"status": 200, "data": {"id": 8, **payload}}

        adapter._api_post = delayed_post
        registration = asyncio.create_task(adapter._register_webhook())
        await started.wait()
        registration.cancel()
        await asyncio.sleep(0)
        release.set()

        with pytest.raises(asyncio.CancelledError):
            await registration

        assert posted == [
            (
                "/api/v1/webhook",
                {"url": adapter._webhook_register_url, "events": ["new-message"]},
            )
        ]
        assert adapter._owned_webhook_ids == set()
        assert await adapter._unregister_webhook() is False

    @pytest.mark.asyncio
    async def test_cancelled_registration_post_does_not_claim_ambiguous_owner(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch)
        url = adapter._webhook_register_url
        post_started = asyncio.Event()
        release_post = asyncio.Event()
        deleted = []
        registrations = [
            {"id": 7, "url": url, "events": ["new-message", "updated-message"]}
        ]
        adapter.client = self._mock_client()
        original_delete = adapter.client.delete

        async def fake_find_registered_webhooks(candidate_url):
            return list(registrations)

        async def delayed_post(path, payload):
            post_started.set()
            await release_post.wait()
            registrations.append(
                {"id": 8, "url": url, "events": list(payload["events"])}
            )
            return {"status": 200, "data": {"id": 8}}

        async def tracking_delete(*args, **kwargs):
            deleted.append(args[0])
            return await original_delete(*args, **kwargs)

        monkeypatch.setattr(
            adapter, "_find_registered_webhooks", fake_find_registered_webhooks
        )
        monkeypatch.setattr(adapter, "_api_post", delayed_post)
        adapter.client.delete = tracking_delete

        registration = asyncio.create_task(adapter._register_webhook())
        await post_started.wait()
        registration.cancel()
        release_post.set()

        with pytest.raises(asyncio.CancelledError):
            await registration

        assert adapter._owned_webhook_ids == set()
        assert await adapter._unregister_webhook() is False
        assert deleted == [adapter._api_url("/api/v1/webhook/7")]

    def test_register_does_not_post_when_lookup_fails(self, monkeypatch):
        import asyncio

        adapter = _make_adapter(monkeypatch)
        posted = []
        adapter.client = self._mock_client()

        async def failed_lookup(url):
            return None

        async def tracking_post(path, payload):
            posted.append((path, payload))
            return {"status": 200}

        adapter._find_registered_webhooks = failed_lookup
        adapter._api_post = tracking_post

        ok = asyncio.get_event_loop().run_until_complete(adapter._register_webhook())

        assert ok is False
        assert posted == []

    @pytest.mark.asyncio
    async def test_partial_stale_delete_failure_restores_when_url_is_empty(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch)
        url = adapter._webhook_register_url
        registrations = [
            {"id": 7, "url": url, "events": ["updated-message"]},
            {"id": 8, "url": url, "events": ["updated-message"]},
        ]
        posted = []
        adapter.client = self._mock_client()

        async def fake_find_registered_webhooks(candidate_url):
            return [item for item in registrations if item["url"] == candidate_url]

        async def uncertain_delete(endpoint, **kwargs):
            webhook_id = int(endpoint.split("/api/v1/webhook/")[1].split("?")[0])
            registrations[:] = [item for item in registrations if item["id"] != webhook_id]
            if webhook_id == 8:
                raise TimeoutError("delete response lost after commit")

            class Response:
                def raise_for_status(self):
                    return None

            return Response()

        async def restore_post(path, payload):
            posted.append(payload)
            restored = {"id": 9, **payload}
            registrations.append(restored)
            return {"status": 200, "data": restored}

        monkeypatch.setattr(
            adapter, "_find_registered_webhooks", fake_find_registered_webhooks
        )
        adapter.client.delete = uncertain_delete
        adapter._api_post = restore_post

        assert await adapter._register_webhook() is False
        assert posted == [{"url": url, "events": ["updated-message"]}]
        assert registrations == [
            {"id": 9, "url": url, "events": ["updated-message"]}
        ]

    @pytest.mark.asyncio
    async def test_register_post_failure_restores_stale_hook(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        url = adapter._webhook_register_url
        deleted = []
        posted = []
        registrations = [
            {
                "id": 7,
                "url": url,
                "events": ["new-message", "updated-message"],
            }
        ]
        adapter.client = self._mock_client()
        original_delete = adapter.client.delete

        async def fake_find_registered_webhooks(candidate_url):
            return [item for item in registrations if item["url"] == candidate_url]

        async def tracking_delete(*args, **kwargs):
            deleted.append(args[0])
            registrations.clear()
            return await original_delete(*args, **kwargs)

        async def replacement_then_rollback(path, payload):
            posted.append(payload)
            if len(posted) == 1:
                return {"status": 500, "message": "internal error"}
            restored = {"id": 9, **payload}
            registrations.append(restored)
            return {"status": 200, "data": restored}

        monkeypatch.setattr(
            adapter, "_find_registered_webhooks", fake_find_registered_webhooks
        )
        adapter.client.delete = tracking_delete
        adapter._api_post = replacement_then_rollback

        assert await adapter._register_webhook() is False
        assert deleted == [adapter._api_url("/api/v1/webhook/7")]
        assert posted == [
            {"url": url, "events": ["new-message"]},
            {"url": url, "events": ["new-message", "updated-message"]},
        ]
        assert registrations == [
            {
                "id": 9,
                "url": url,
                "events": ["new-message", "updated-message"],
            }
        ]
        assert adapter._owned_webhook_ids == set()

    @pytest.mark.asyncio
    async def test_register_reconciles_replacement_committed_before_timeout(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch)
        url = adapter._webhook_register_url
        registrations = [
            {
                "id": 7,
                "url": url,
                "events": ["new-message", "updated-message"],
            }
        ]
        posted = []
        adapter.client = self._mock_client()
        original_delete = adapter.client.delete

        async def fake_find_registered_webhooks(candidate_url):
            return [item for item in registrations if item["url"] == candidate_url]

        async def tracking_delete(*args, **kwargs):
            registrations.clear()
            return await original_delete(*args, **kwargs)

        async def committed_then_timed_out(path, payload):
            posted.append(payload)
            committed = {"id": 8, **payload}
            registrations.append(committed)
            raise TimeoutError("response lost after commit")

        monkeypatch.setattr(
            adapter, "_find_registered_webhooks", fake_find_registered_webhooks
        )
        adapter.client.delete = tracking_delete
        adapter._api_post = committed_then_timed_out

        assert await adapter._register_webhook() is True
        assert posted == [{"url": url, "events": ["new-message"]}]
        assert registrations == [
            {"id": 8, "url": url, "events": ["new-message"]}
        ]
        assert adapter._owned_webhook_ids == set()

    @pytest.mark.asyncio
    async def test_register_does_not_delete_unexpected_post_failure_owner(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch)
        url = adapter._webhook_register_url
        registrations = [
            {
                "id": 7,
                "url": url,
                "events": ["new-message", "updated-message"],
            }
        ]
        adapter.client = self._mock_client()
        original_delete = adapter.client.delete

        async def fake_find_registered_webhooks(candidate_url):
            return [item for item in registrations if item["url"] == candidate_url]

        async def tracking_delete(*args, **kwargs):
            registrations.clear()
            return await original_delete(*args, **kwargs)

        async def unexpected_owner(path, payload):
            occupied = {
                "id": 8,
                "url": url,
                "events": ["updated-message"],
            }
            registrations.append(occupied)
            return {"status": 200, "data": occupied}

        monkeypatch.setattr(
            adapter, "_find_registered_webhooks", fake_find_registered_webhooks
        )
        adapter.client.delete = tracking_delete
        adapter._api_post = unexpected_owner

        assert await adapter._register_webhook() is False
        assert registrations == [
            {"id": 8, "url": url, "events": ["updated-message"]}
        ]
        assert adapter._owned_webhook_ids == set()

    def test_register_returns_false_without_client(self, monkeypatch):
        import asyncio
        adapter = _make_adapter(monkeypatch)
        adapter.client = None
        ok = asyncio.get_event_loop().run_until_complete(
            adapter._register_webhook()
        )
        assert ok is False

    def test_register_returns_false_on_server_error(self, monkeypatch):
        import asyncio
        adapter = _make_adapter(monkeypatch)
        adapter.client = self._mock_client(
            get_response={"status": 200, "data": []},
            post_response={"status": 500, "message": "internal error"},
        )
        ok = asyncio.get_event_loop().run_until_complete(
            adapter._register_webhook()
        )
        assert ok is False

    @pytest.mark.asyncio
    async def test_connect_registration_failure_does_not_unregister_existing(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch, webhook_port="0")
        unregister_calls = []

        async def fake_api_get(path):
            if path == "/api/v1/server/info":
                return {"data": {}}
            return {"status": 200}

        async def failed_registration():
            return False

        async def tracking_unregister():
            unregister_calls.append(True)
            return True

        monkeypatch.setattr(adapter, "_api_get", fake_api_get)
        monkeypatch.setattr(adapter, "_register_webhook", failed_registration)
        monkeypatch.setattr(adapter, "_unregister_webhook", tracking_unregister)

        assert await adapter.connect() is False
        assert unregister_calls == []
        assert adapter.client is None
        assert adapter._runner is None

    @pytest.mark.asyncio
    async def test_connect_cancellation_cleans_owned_registration_and_listener(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch, webhook_port="0")
        cleanup_calls = []

        async def fake_api_get(path):
            if path == "/api/v1/server/info":
                return {"data": {}}
            return {"status": 200}

        async def cancelled_registration():
            adapter._owned_webhook_ids.add("8")
            raise asyncio.CancelledError

        async def tracking_unregister():
            cleanup_calls.append("unregister")
            adapter._owned_webhook_ids.clear()
            return True

        monkeypatch.setattr(adapter, "_api_get", fake_api_get)
        monkeypatch.setattr(adapter, "_register_webhook", cancelled_registration)
        monkeypatch.setattr(adapter, "_unregister_webhook", tracking_unregister)

        with pytest.raises(asyncio.CancelledError):
            await adapter.connect()

        assert cleanup_calls == ["unregister"]
        assert adapter._owned_webhook_ids == set()
        assert adapter.client is None
        assert adapter._runner is None

    # -- _unregister_webhook --

    def test_unregister_removes_matching(self, monkeypatch):
        import asyncio
        adapter = _make_adapter(monkeypatch)
        url = adapter._webhook_register_url
        adapter.client = self._mock_client(
            get_response={"status": 200, "data": [
                {"id": 10, "url": url},
            ]},
        )
        adapter._owned_webhook_ids = {"10"}
        ok = asyncio.get_event_loop().run_until_complete(
            adapter._unregister_webhook()
        )
        assert ok is True

    def test_unregister_removes_only_owned_registrations(self, monkeypatch):
        """Disconnect must not remove same-URL registrations it did not create."""
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
                {"id": 4, "url": url},
                {"id": 3, "url": "http://other/hook"},
            ]},
        )
        adapter.client.delete = mock_delete
        adapter._owned_webhook_ids = {"1", "2"}

        ok = asyncio.get_event_loop().run_until_complete(
            adapter._unregister_webhook()
        )
        assert ok is True
        assert len(deleted_ids) == 2
        assert set(deleted_ids) == {
            adapter._api_url("/api/v1/webhook/1"),
            adapter._api_url("/api/v1/webhook/2"),
        }

    def test_unregister_returns_false_without_client(self, monkeypatch):
        import asyncio
        adapter = _make_adapter(monkeypatch)
        adapter.client = None
        ok = asyncio.get_event_loop().run_until_complete(
            adapter._unregister_webhook()
        )
        assert ok is False

    def test_unregister_handles_api_failure_gracefully(self, monkeypatch):
        import asyncio
        adapter = _make_adapter(monkeypatch)
        adapter.client = self._mock_client()

        async def bad_get(path):
            raise ConnectionError("server down")
        adapter._api_get = bad_get

        ok = asyncio.get_event_loop().run_until_complete(
            adapter._unregister_webhook()
        )
        assert ok is False
