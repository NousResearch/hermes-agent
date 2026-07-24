"""Tests for the BlueBubbles iMessage gateway adapter."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
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

    @pytest.mark.asyncio
    async def test_read_timeout_is_ambiguous_and_does_not_send_fallback(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch)
        monkeypatch.setattr(
            adapter,
            "_resolve_chat_guid",
            AsyncMock(return_value="iMessage;-;user@example.com"),
        )
        post = AsyncMock(side_effect=httpx.ReadTimeout(""))
        monkeypatch.setattr(adapter, "_api_post", post)

        result = await adapter._send_with_retry("user@example.com", "hello")

        assert result.success is False
        assert result.error == "ReadTimeout"
        assert post.await_count == 1

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


class TestBlueBubblesInboundDeduplication:
    @staticmethod
    def _payload(
        message_guid,
        *,
        event_type="new-message",
        chat_guid=None,
        text="hello",
        attachments=None,
    ):
        return {
            "type": event_type,
            "data": {
                "guid": message_guid,
                "text": text,
                "handle": {"address": "user@example.com"},
                "isFromMe": False,
                "chatGuid": chat_guid or "iMessage;-;user@example.com",
                "chatIdentifier": "user@example.com",
                "attachments": attachments or [],
            },
        }

    @pytest.mark.asyncio
    async def test_same_message_guid_dispatches_once_across_new_and_updated_events(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        handled = []
        monkeypatch.setattr(adapter, "handle_message", AsyncMock(side_effect=handled.append))

        first = self._payload("stable-guid", event_type="new-message")
        updated = self._payload(
            "stable-guid",
            event_type="updated-message",
            chat_guid="iMessage;-;different-chat-fields@example.com",
        )

        assert (await adapter._handle_webhook(_FakeBlueBubblesRequest(first))).status == 200
        assert (await adapter._handle_webhook(_FakeBlueBubblesRequest(updated))).status == 200
        await asyncio.sleep(0)

        assert [event.message_id for event in handled] == ["stable-guid"]

    @pytest.mark.asyncio
    async def test_distinct_message_guids_still_dispatch(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        handled = []
        monkeypatch.setattr(adapter, "handle_message", AsyncMock(side_effect=handled.append))

        for guid in ("guid-1", "guid-2"):
            response = await adapter._handle_webhook(
                _FakeBlueBubblesRequest(self._payload(guid))
            )
            assert response.status == 200
        await asyncio.sleep(0)

        assert [event.message_id for event in handled] == ["guid-1", "guid-2"]

    @pytest.mark.asyncio
    async def test_malformed_delivery_does_not_poison_valid_retry(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        handled = []
        monkeypatch.setattr(adapter, "handle_message", AsyncMock(side_effect=handled.append))
        malformed = self._payload("retry-guid", text="")
        valid = self._payload("retry-guid", text="valid retry")

        assert (await adapter._handle_webhook(_FakeBlueBubblesRequest(malformed))).status == 400
        assert (await adapter._handle_webhook(_FakeBlueBubblesRequest(valid))).status == 200
        await asyncio.sleep(0)

        assert [event.text for event in handled] == ["valid retry"]

    @pytest.mark.asyncio
    async def test_seen_guid_cache_has_size_bound_and_ttl(self, monkeypatch):
        import gateway.platforms.bluebubbles as bluebubbles

        now = [100.0]
        monkeypatch.setattr(bluebubbles, "_MESSAGE_DEDUP_CACHE_SIZE", 2)
        monkeypatch.setattr(bluebubbles, "_MESSAGE_DEDUP_TTL_SECONDS", 5.0)
        monkeypatch.setattr(bluebubbles.time, "monotonic", lambda: now[0])
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        handled = []
        monkeypatch.setattr(adapter, "handle_message", AsyncMock(side_effect=handled.append))

        for guid in ("guid-1", "guid-2", "guid-3"):
            response = await adapter._handle_webhook(
                _FakeBlueBubblesRequest(self._payload(guid))
            )
            assert response.status == 200
            if adapter._background_tasks:
                await asyncio.gather(*list(adapter._background_tasks))
        assert len(adapter._seen_message_guids) == 2

        now[0] += 6.0
        await adapter._handle_webhook(
            _FakeBlueBubblesRequest(self._payload("guid-3"))
        )
        if adapter._background_tasks:
            await asyncio.gather(*list(adapter._background_tasks))

        assert [event.message_id for event in handled] == [
            "guid-1",
            "guid-2",
            "guid-3",
            "guid-3",
        ]

    @pytest.mark.asyncio
    async def test_overlapping_duplicate_webhooks_download_and_dispatch_once(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        download_started = asyncio.Event()
        release_download = asyncio.Event()
        handled = []

        async def download_once(*_args):
            download_started.set()
            await release_download.wait()
            return "/cache/photo.jpg"

        download = AsyncMock(side_effect=download_once)
        monkeypatch.setattr(adapter, "_download_attachment", download)
        monkeypatch.setattr(
            adapter, "handle_message", AsyncMock(side_effect=handled.append)
        )
        payload = self._payload(
            "overlap-guid",
            attachments=[{"guid": "att-1", "mimeType": "image/jpeg"}],
        )

        first = asyncio.create_task(
            adapter._handle_webhook(_FakeBlueBubblesRequest(payload))
        )
        await download_started.wait()
        second = asyncio.create_task(
            adapter._handle_webhook(_FakeBlueBubblesRequest(payload))
        )
        await asyncio.sleep(0)
        release_download.set()
        responses = await asyncio.gather(first, second)
        await asyncio.sleep(0)

        assert [response.status for response in responses] == [200, 200]
        assert download.await_count == 1
        assert [event.message_id for event in handled] == ["overlap-guid"]

    @pytest.mark.asyncio
    async def test_late_updated_message_dispatches_attachment_only_enrichment(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        handled = []
        monkeypatch.setattr(
            adapter, "handle_message", AsyncMock(side_effect=handled.append)
        )
        download = AsyncMock(return_value="/cache/enriched-photo.jpg")
        monkeypatch.setattr(adapter, "_download_attachment", download)

        original = self._payload("enrich-guid", event_type="new-message")
        enriched = self._payload(
            "enrich-guid",
            event_type="updated-message",
            attachments=[{"guid": "att-new", "mimeType": "image/png"}],
        )

        assert (
            await adapter._handle_webhook(_FakeBlueBubblesRequest(original))
        ).status == 200
        await asyncio.sleep(0)
        assert (
            await adapter._handle_webhook(_FakeBlueBubblesRequest(enriched))
        ).status == 200
        await asyncio.sleep(0)

        assert download.await_count == 1
        assert len(handled) == 2
        assert handled[0].text == "hello"
        assert handled[0].media_urls == []
        assert handled[1].text == "(attachment)"
        assert handled[1].media_urls == ["/cache/enriched-photo.jpg"]
        assert handled[1].media_types == ["image/png"]
        assert handled[1].message_type.value == "photo"

    @pytest.mark.asyncio
    async def test_failed_attachment_download_is_retryable_on_updated_message(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        handled = []
        monkeypatch.setattr(
            adapter, "handle_message", AsyncMock(side_effect=handled.append)
        )
        download = AsyncMock(side_effect=[None, "/cache/retried.jpg"])
        monkeypatch.setattr(adapter, "_download_attachment", download)
        payload = self._payload(
            "retry-download-guid",
            attachments=[{"guid": "retry-att", "mimeType": "image/jpeg"}],
        )

        assert (
            await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))
        ).status == 200
        await asyncio.sleep(0)
        assert (
            await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))
        ).status == 200
        await asyncio.sleep(0)

        assert download.await_count == 2
        assert len(handled) == 2
        assert handled[0].media_urls == []
        assert handled[1].text == "(attachment)"
        assert handled[1].media_urls == ["/cache/retried.jpg"]
        assert handled[1].message_type.value == "photo"

    @pytest.mark.asyncio
    async def test_caf_uti_attachment_remains_voice_message(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        handled = []
        monkeypatch.setattr(
            adapter, "handle_message", AsyncMock(side_effect=handled.append)
        )
        monkeypatch.setattr(
            adapter,
            "_download_attachment",
            AsyncMock(return_value="/cache/voice.caf"),
        )
        payload = self._payload(
            "caf-guid",
            attachments=[
                {
                    "guid": "caf-att",
                    "mimeType": "application/octet-stream",
                    "uti": "com.apple.caf",
                }
            ],
        )

        assert (
            await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))
        ).status == 200
        await asyncio.sleep(0)

        assert len(handled) == 1
        assert handled[0].message_type.value == "voice"
        assert handled[0].media_types == ["application/octet-stream"]

    @pytest.mark.asyncio
    async def test_waiting_duplicate_takes_over_after_owner_dispatch_failure(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        attempts = []

        async def flaky_handle(event):
            attempts.append(event.message_id)
            if len(attempts) == 1:
                first_started.set()
                await release_first.wait()
                raise RuntimeError("owner dispatch failed")

        monkeypatch.setattr(adapter, "handle_message", flaky_handle)
        payload = self._payload("joined-retry-guid")

        first_response = await adapter._handle_webhook(
            _FakeBlueBubblesRequest(payload)
        )
        await first_started.wait()
        duplicate = asyncio.create_task(
            adapter._handle_webhook(_FakeBlueBubblesRequest(payload))
        )
        await asyncio.sleep(0)
        assert not duplicate.done()

        release_first.set()
        duplicate_response = await duplicate
        await asyncio.sleep(0)

        assert first_response.status == 200
        assert duplicate_response.status == 200
        assert attempts == ["joined-retry-guid", "joined-retry-guid"]

    @pytest.mark.asyncio
    async def test_capacity_pressure_returns_503_without_evicting_inflight(
        self, monkeypatch
    ):
        import gateway.platforms.bluebubbles as bluebubbles

        monkeypatch.setattr(bluebubbles, "_MESSAGE_DEDUP_CACHE_SIZE", 1)
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        download_started = asyncio.Event()
        release_download = asyncio.Event()

        async def blocked_download(*_args):
            download_started.set()
            await release_download.wait()
            return "/cache/photo.jpg"

        monkeypatch.setattr(adapter, "_download_attachment", blocked_download)
        monkeypatch.setattr(adapter, "handle_message", AsyncMock())
        first_payload = self._payload(
            "capacity-1",
            attachments=[{"guid": "att-1", "mimeType": "image/jpeg"}],
        )
        first = asyncio.create_task(
            adapter._handle_webhook(_FakeBlueBubblesRequest(first_payload))
        )
        await download_started.wait()

        second_response = await adapter._handle_webhook(
            _FakeBlueBubblesRequest(self._payload("capacity-2"))
        )
        assert second_response.status == 503
        assert "capacity-1" in adapter._seen_message_guids

        release_download.set()
        assert (await first).status == 200

    @pytest.mark.asyncio
    async def test_no_guid_message_still_enforces_attachment_bound(
        self, monkeypatch
    ):
        import gateway.platforms.bluebubbles as bluebubbles

        monkeypatch.setattr(bluebubbles, "_MESSAGE_DEDUP_MAX_ATTACHMENTS", 2)
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        download = AsyncMock(return_value="/cache/photo.jpg")
        monkeypatch.setattr(adapter, "_download_attachment", download)
        payload = self._payload(
            None,
            attachments=[
                {"guid": f"att-{index}", "mimeType": "image/jpeg"}
                for index in range(3)
            ],
        )

        response = await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))

        assert response.status == 413
        assert download.await_count == 0

    @pytest.mark.asyncio
    async def test_duplicate_join_has_bounded_wait(self, monkeypatch):
        import gateway.platforms.bluebubbles as bluebubbles

        monkeypatch.setattr(
            bluebubbles, "_MESSAGE_DEDUP_JOIN_TIMEOUT_SECONDS", 0.01
        )
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        owner_started = asyncio.Event()
        release_owner = asyncio.Event()

        async def blocked_handle(_event):
            owner_started.set()
            await release_owner.wait()

        monkeypatch.setattr(adapter, "handle_message", blocked_handle)
        payload = self._payload("bounded-join")
        assert (
            await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))
        ).status == 200
        await owner_started.wait()

        duplicate_response = await asyncio.wait_for(
            adapter._handle_webhook(_FakeBlueBubblesRequest(payload)),
            timeout=0.1,
        )

        assert duplicate_response.status == 503
        release_owner.set()

    @pytest.mark.asyncio
    async def test_late_enrichment_download_exception_rolls_back_for_retry(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        handled = []
        monkeypatch.setattr(
            adapter, "handle_message", AsyncMock(side_effect=handled.append)
        )
        original = self._payload("late-failure")
        enriched = self._payload(
            "late-failure",
            event_type="updated-message",
            attachments=[{"guid": "late-att", "mimeType": "image/jpeg"}],
        )

        assert (
            await adapter._handle_webhook(_FakeBlueBubblesRequest(original))
        ).status == 200
        if adapter._background_tasks:
            await asyncio.gather(*list(adapter._background_tasks))
        monkeypatch.setattr(
            adapter,
            "_download_attachment",
            AsyncMock(side_effect=RuntimeError("download failed")),
        )

        with pytest.raises(RuntimeError, match="download failed"):
            await adapter._handle_webhook(_FakeBlueBubblesRequest(enriched))

        reservation = adapter._seen_message_guids["late-failure"]
        assert reservation["state"] == "complete"
        assert "late-att" not in reservation["attachment_guids"]

        monkeypatch.setattr(
            adapter,
            "_download_attachment",
            AsyncMock(return_value="/cache/retried-late.jpg"),
        )
        assert (
            await adapter._handle_webhook(_FakeBlueBubblesRequest(enriched))
        ).status == 200
        await asyncio.sleep(0)
        assert len(handled) == 2
        assert handled[1].media_urls == ["/cache/retried-late.jpg"]

    @pytest.mark.asyncio
    async def test_new_attachment_waits_for_inflight_owner_then_dispatches(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        owner_started = asyncio.Event()
        release_owner = asyncio.Event()
        download_started = asyncio.Event()
        release_download = asyncio.Event()
        handled = []

        async def handle(event):
            handled.append(event)
            if len(handled) == 1:
                owner_started.set()
                await release_owner.wait()

        async def download(*_args):
            download_started.set()
            await release_download.wait()
            return "/cache/serialized.jpg"

        monkeypatch.setattr(adapter, "handle_message", handle)
        monkeypatch.setattr(adapter, "_download_attachment", download)
        original = self._payload("serialize-enrichment")
        enriched = self._payload(
            "serialize-enrichment",
            event_type="updated-message",
            attachments=[{"guid": "serialized-att", "mimeType": "image/jpeg"}],
        )

        assert (
            await adapter._handle_webhook(_FakeBlueBubblesRequest(original))
        ).status == 200
        await owner_started.wait()
        enrichment = asyncio.create_task(
            adapter._handle_webhook(_FakeBlueBubblesRequest(enriched))
        )
        await asyncio.sleep(0)
        assert not download_started.is_set()

        release_owner.set()
        await download_started.wait()
        release_download.set()
        assert (await enrichment).status == 200
        await asyncio.sleep(0)

        assert len(handled) == 2
        assert handled[1].text == "(attachment)"
        assert handled[1].media_urls == ["/cache/serialized.jpg"]

    @pytest.mark.asyncio
    async def test_completed_reservation_does_not_retain_raw_message_event(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        monkeypatch.setattr(adapter, "handle_message", AsyncMock())
        payload = self._payload("no-event-retention", text="x" * 100_000)

        assert (
            await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))
        ).status == 200
        if adapter._background_tasks:
            await asyncio.gather(*list(adapter._background_tasks))

        reservation = adapter._seen_message_guids["no-event-retention"]
        assert reservation["state"] == "complete"
        assert "event" not in reservation

    @pytest.mark.asyncio
    async def test_reservation_preserves_bluebubbles_attachment_order(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)

        delivery_kind, _reservation, new_guids = adapter._reserve_message_delivery(
            "ordered-attachments", ["z-last-lexically", "a-first-lexically"]
        )

        assert delivery_kind == "new"
        assert new_guids == ["z-last-lexically", "a-first-lexically"]

    @pytest.mark.asyncio
    async def test_duplicate_join_loop_has_request_wide_attempt_bound(
        self, monkeypatch
    ):
        import gateway.platforms.bluebubbles as bluebubbles

        monkeypatch.setattr(bluebubbles, "_MESSAGE_DEDUP_MAX_JOIN_ATTEMPTS", 2)
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        outcome = asyncio.get_running_loop().create_future()
        outcome.set_result(False)
        reservation = {"outcome": outcome, "waiters": 0}
        reserve = Mock(return_value=("duplicate_wait", reservation, set()))
        monkeypatch.setattr(adapter, "_reserve_message_delivery", reserve)
        monkeypatch.setattr(
            adapter, "_join_message_reservation", AsyncMock(return_value=False)
        )

        response = await adapter._handle_webhook(
            _FakeBlueBubblesRequest(self._payload("bounded-attempts"))
        )

        assert response.status == 503
        assert reserve.call_count == 3
        assert adapter._join_message_reservation.await_count == 2

    @pytest.mark.asyncio
    async def test_duplicate_rechecks_released_attachment_after_owner_success(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        first_download_started = asyncio.Event()
        release_first_download = asyncio.Event()
        downloads = 0
        handled = []

        async def retrying_download(*_args):
            nonlocal downloads
            downloads += 1
            if downloads == 1:
                first_download_started.set()
                await release_first_download.wait()
                return None
            return "/cache/join-retry.jpg"

        monkeypatch.setattr(adapter, "_download_attachment", retrying_download)
        monkeypatch.setattr(
            adapter, "handle_message", AsyncMock(side_effect=handled.append)
        )
        payload = self._payload(
            "joined-download-retry",
            attachments=[{"guid": "joined-att", "mimeType": "image/jpeg"}],
        )

        owner = asyncio.create_task(
            adapter._handle_webhook(_FakeBlueBubblesRequest(payload))
        )
        await first_download_started.wait()
        duplicate = asyncio.create_task(
            adapter._handle_webhook(_FakeBlueBubblesRequest(payload))
        )
        await asyncio.sleep(0)
        release_first_download.set()

        responses = await asyncio.gather(owner, duplicate)
        if adapter._background_tasks:
            await asyncio.gather(*list(adapter._background_tasks))

        assert [response.status for response in responses] == [200, 200]
        assert downloads == 2
        assert len(handled) == 2
        assert handled[1].media_urls == ["/cache/join-retry.jpg"]

    @pytest.mark.asyncio
    async def test_attachment_guid_bound_rejects_oversized_message(self, monkeypatch):
        import gateway.platforms.bluebubbles as bluebubbles

        monkeypatch.setattr(bluebubbles, "_MESSAGE_DEDUP_MAX_ATTACHMENTS", 2)
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        download = AsyncMock(return_value="/cache/photo.jpg")
        monkeypatch.setattr(adapter, "_download_attachment", download)
        payload = self._payload(
            "too-many-attachments",
            attachments=[
                {"guid": f"att-{index}", "mimeType": "image/jpeg"}
                for index in range(3)
            ],
        )

        response = await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))

        assert response.status == 413
        assert download.await_count == 0
        assert "too-many-attachments" not in adapter._seen_message_guids

    @pytest.mark.asyncio
    async def test_failed_webhook_task_scheduling_releases_reservation_for_retry(
        self, monkeypatch
    ):
        import gateway.platforms.bluebubbles as bluebubbles

        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        handled = []
        monkeypatch.setattr(
            adapter, "handle_message", AsyncMock(side_effect=handled.append)
        )
        real_create_task = asyncio.create_task
        scheduling_attempts = 0

        def fail_first_schedule(coro):
            nonlocal scheduling_attempts
            scheduling_attempts += 1
            if scheduling_attempts == 1:
                coro.close()
                raise RuntimeError("task scheduler unavailable")
            return real_create_task(coro)

        monkeypatch.setattr(bluebubbles.asyncio, "create_task", fail_first_schedule)
        payload = self._payload("retry-schedule-guid")

        with pytest.raises(RuntimeError, match="scheduler unavailable"):
            await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))
        assert (
            await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))
        ).status == 200
        await asyncio.sleep(0)

        assert [event.message_id for event in handled] == ["retry-schedule-guid"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("failure_site", ["build_source", "message_event", "apply_media"])
    async def test_synchronous_event_setup_failure_releases_reservation_for_retry(
        self, monkeypatch, failure_site
    ):
        import gateway.platforms.bluebubbles as bluebubbles

        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        handled = []
        monkeypatch.setattr(
            adapter, "handle_message", AsyncMock(side_effect=handled.append)
        )
        setup_attempts = 0

        def fail_once_then(call):
            def wrapped(*args, **kwargs):
                nonlocal setup_attempts
                setup_attempts += 1
                if setup_attempts == 1:
                    raise RuntimeError("event setup failed")
                return call(*args, **kwargs)

            return wrapped

        if failure_site == "build_source":
            monkeypatch.setattr(
                adapter, "build_source", fail_once_then(adapter.build_source)
            )
        elif failure_site == "message_event":
            monkeypatch.setattr(
                bluebubbles,
                "MessageEvent",
                fail_once_then(bluebubbles.MessageEvent),
            )
        else:
            monkeypatch.setattr(
                adapter,
                "_apply_reservation_media",
                fail_once_then(adapter._apply_reservation_media),
            )
        payload = self._payload(f"retry-{failure_site}-guid")

        with pytest.raises(RuntimeError, match="event setup failed"):
            await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))
        assert f"retry-{failure_site}-guid" not in adapter._seen_message_guids
        assert (
            await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))
        ).status == 200
        await asyncio.sleep(0)

        assert [event.message_id for event in handled] == [
            f"retry-{failure_site}-guid"
        ]

    @pytest.mark.asyncio
    async def test_late_enrichment_setup_failure_restores_reservation_for_retry(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        handled = []
        monkeypatch.setattr(
            adapter, "handle_message", AsyncMock(side_effect=handled.append)
        )
        monkeypatch.setattr(
            adapter,
            "_download_attachment",
            AsyncMock(return_value="/cache/retried-enrichment.jpg"),
        )
        original = self._payload("retry-late-setup-guid")
        enriched = self._payload(
            "retry-late-setup-guid",
            event_type="updated-message",
            attachments=[{"guid": "late-setup-att", "mimeType": "image/jpeg"}],
        )

        assert (
            await adapter._handle_webhook(_FakeBlueBubblesRequest(original))
        ).status == 200
        if adapter._background_tasks:
            await asyncio.gather(*list(adapter._background_tasks))

        original_apply = adapter._apply_reservation_media
        setup_attempts = 0

        def fail_first_enrichment(event, reservation):
            nonlocal setup_attempts
            setup_attempts += 1
            if setup_attempts == 1:
                raise RuntimeError("late setup failed")
            return original_apply(event, reservation)

        monkeypatch.setattr(
            adapter, "_apply_reservation_media", fail_first_enrichment
        )
        with pytest.raises(RuntimeError, match="late setup failed"):
            await adapter._handle_webhook(_FakeBlueBubblesRequest(enriched))
        reservation = adapter._seen_message_guids["retry-late-setup-guid"]
        assert reservation["state"] == "complete"
        assert "late-setup-att" not in reservation["attachment_guids"]

        assert (
            await adapter._handle_webhook(_FakeBlueBubblesRequest(enriched))
        ).status == 200
        if adapter._background_tasks:
            await asyncio.gather(*list(adapter._background_tasks))

        assert len(handled) == 2
        assert handled[1].media_urls == ["/cache/retried-enrichment.jpg"]

    @pytest.mark.asyncio
    async def test_observable_dispatch_failure_releases_reservation_for_retry(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        first_failed = asyncio.Event()
        attempts = []

        async def flaky_handle(event):
            attempts.append(event.message_id)
            if len(attempts) == 1:
                first_failed.set()
                raise RuntimeError("dispatch setup failed")

        monkeypatch.setattr(adapter, "handle_message", flaky_handle)
        payload = self._payload("retry-dispatch-guid")

        assert (
            await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))
        ).status == 200
        await first_failed.wait()
        await asyncio.sleep(0)
        assert (
            await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))
        ).status == 200
        await asyncio.sleep(0)

        assert attempts == ["retry-dispatch-guid", "retry-dispatch-guid"]


def _quick_ack_runner(monkeypatch, config, *, platform=Platform.BLUEBUBBLES):
    from gateway.platforms.base import MessageEvent, SessionSource
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._pending_turn_sidecar_notes = {}
    adapter = _make_adapter(monkeypatch)
    adapter.send = AsyncMock(return_value=SimpleNamespace(success=True))
    runner._adapter_for_source = lambda source: adapter
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: config)
    source = SessionSource(
        platform=platform,
        chat_id="iMessage;-;user@example.com",
        user_id="user@example.com",
        chat_type="dm",
    )
    event = MessageEvent(text="Please compare these two contracts", source=source)
    return runner, adapter, event, source


def _quick_ack_config(**overrides):
    return {
        "display": {
            "platforms": {
                "bluebubbles": {
                    "quick_ack_enabled": True,
                    **overrides,
                }
            }
        }
    }


def _aux_response(text):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


class TestBlueBubblesQuickAcknowledgment:
    @pytest.mark.asyncio
    async def test_enabled_bluebubbles_sends_contextual_ack_before_main_turn(self, monkeypatch):
        runner, adapter, event, source = _quick_ack_runner(
            monkeypatch,
            _quick_ack_config(quick_ack_model="fast-model", quick_ack_timeout_seconds=2),
        )
        aux = AsyncMock(return_value=_aux_response('"I’ll compare both carefully."'))
        monkeypatch.setattr("agent.auxiliary_client.async_call_llm", aux)
        notes = []

        ack = await runner._maybe_send_bluebubbles_quick_ack(
            event, source, event.text, notes
        )

        assert ack == "I’ll compare both carefully."
        adapter.send.assert_awaited_once_with(source.chat_id, ack)
        kwargs = aux.await_args.kwargs
        assert kwargs["model"] == "fast-model"
        assert kwargs["timeout"] == 2.0
        assert kwargs.get("tools") is None
        assert "stream" not in kwargs
        prompt = kwargs["messages"][0]["content"]
        assert "under 8 words" in prompt
        assert "no quotes or Markdown" in prompt
        assert "must not claim" in prompt
        assert [message["role"] for message in kwargs["messages"]] == ["system", "user"]
        assert event.text in kwargs["messages"][1]["content"]

    @pytest.mark.asyncio
    async def test_disabled_setting_skips_ack(self, monkeypatch):
        runner, adapter, event, source = _quick_ack_runner(monkeypatch, {})
        aux = AsyncMock(return_value=_aux_response("Taking a look now."))
        monkeypatch.setattr("agent.auxiliary_client.async_call_llm", aux)

        ack = await runner._maybe_send_bluebubbles_quick_ack(
            event, source, event.text, []
        )

        assert ack is None
        aux.assert_not_awaited()
        adapter.send.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_timeout_setting_is_bounded(self, monkeypatch):
        runner, _adapter, event, source = _quick_ack_runner(
            monkeypatch,
            _quick_ack_config(quick_ack_timeout_seconds=999),
        )
        aux = AsyncMock(return_value=_aux_response("I’ll inspect this now."))
        monkeypatch.setattr("agent.auxiliary_client.async_call_llm", aux)

        await runner._maybe_send_bluebubbles_quick_ack(
            event, source, event.text, []
        )

        assert aux.await_args.kwargs["timeout"] == 10.0

    @pytest.mark.asyncio
    async def test_non_bluebubbles_message_skips_ack(self, monkeypatch):
        runner, adapter, event, source = _quick_ack_runner(
            monkeypatch, _quick_ack_config(), platform=Platform.TELEGRAM
        )
        aux = AsyncMock(return_value=_aux_response("Taking a look now."))
        monkeypatch.setattr("agent.auxiliary_client.async_call_llm", aux)

        ack = await runner._maybe_send_bluebubbles_quick_ack(
            event, source, event.text, []
        )

        assert ack is None
        aux.assert_not_awaited()
        adapter.send.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("text", ["/help", "hi", "ping", "thanks", "yes", "no"])
    async def test_slash_and_trivial_messages_skip_ack(self, monkeypatch, text):
        runner, adapter, event, source = _quick_ack_runner(
            monkeypatch, _quick_ack_config()
        )
        event.text = text
        aux = AsyncMock(return_value=_aux_response("Taking a look now."))
        monkeypatch.setattr("agent.auxiliary_client.async_call_llm", aux)

        ack = await runner._maybe_send_bluebubbles_quick_ack(
            event, source, event.text, []
        )

        assert ack is None
        aux.assert_not_awaited()
        adapter.send.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_auxiliary_failure_sends_configured_fallback(self, monkeypatch):
        runner, adapter, event, source = _quick_ack_runner(
            monkeypatch,
            _quick_ack_config(quick_ack_fallback="Got it — I’m checking."),
        )
        monkeypatch.setattr(
            "agent.auxiliary_client.async_call_llm",
            AsyncMock(side_effect=RuntimeError("aux unavailable")),
        )
        notes = []

        ack = await runner._maybe_send_bluebubbles_quick_ack(
            event, source, event.text, notes
        )

        assert ack == "Got it — I’m checking."
        adapter.send.assert_awaited_once_with(source.chat_id, ack)
        assert ack in notes[0]

    @pytest.mark.asyncio
    async def test_ack_send_failure_does_not_abort_main_turn(self, monkeypatch):
        runner, adapter, event, source = _quick_ack_runner(
            monkeypatch, _quick_ack_config()
        )
        monkeypatch.setattr(
            "agent.auxiliary_client.async_call_llm",
            AsyncMock(return_value=_aux_response("I’ll inspect this now.")),
        )
        adapter.send.side_effect = RuntimeError("BlueBubbles offline")
        main_turn = AsyncMock(return_value="main response")
        notes = []

        ack = await runner._maybe_send_bluebubbles_quick_ack(
            event, source, event.text, notes
        )
        result = await main_turn()

        assert ack is None
        assert notes == []
        assert result == "main response"
        main_turn.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_prompt_injected_completion_claim_uses_safe_fallback(
        self, monkeypatch
    ):
        runner, adapter, event, source = _quick_ack_runner(
            monkeypatch,
            _quick_ack_config(
                quick_ack_fallback="Got it — I’m checking.",
            ),
        )
        aux = AsyncMock(return_value=_aux_response("Done — I sent it."))
        monkeypatch.setattr("agent.auxiliary_client.async_call_llm", aux)

        ack = await runner._maybe_send_bluebubbles_quick_ack(
            event, source, event.text, []
        )

        assert ack == "Got it — I’m checking."
        adapter.send.assert_awaited_once_with(source.chat_id, ack)
        messages = aux.await_args.kwargs["messages"]
        assert [message["role"] for message in messages] == ["system", "user"]
        assert "must not claim" in messages[0]["content"].lower()
        assert event.text in messages[1]["content"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "unsafe_text",
        [
            "I've handled that for you.",
            "Your email is on its way.",
            "Got it — your request is fulfilled.",
        ],
    )
    async def test_non_pending_generated_ack_uses_safe_default(
        self, monkeypatch, unsafe_text
    ):
        import gateway.platforms.bluebubbles as bluebubbles

        runner, adapter, event, source = _quick_ack_runner(
            monkeypatch,
            _quick_ack_config(quick_ack_fallback="Done"),
        )
        monkeypatch.setattr(
            "agent.auxiliary_client.async_call_llm",
            AsyncMock(return_value=_aux_response(unsafe_text)),
        )

        ack = await runner._maybe_send_bluebubbles_quick_ack(
            event, source, event.text, []
        )

        assert ack == bluebubbles._QUICK_ACK_DEFAULT_FALLBACK
        adapter.send.assert_awaited_once_with(source.chat_id, ack)

    @pytest.mark.asyncio
    async def test_parent_cancellation_cancels_quick_ack_child(self, monkeypatch):
        runner, adapter, event, source = _quick_ack_runner(
            monkeypatch, _quick_ack_config()
        )
        generation_started = asyncio.Event()
        generation_cancelled = asyncio.Event()

        async def cancellable_generation(**_kwargs):
            generation_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                generation_cancelled.set()
                raise

        monkeypatch.setattr(
            "agent.auxiliary_client.async_call_llm", cancellable_generation
        )
        ack_task = asyncio.create_task(
            runner._maybe_send_bluebubbles_quick_ack(
                event, source, event.text, []
            )
        )
        await generation_started.wait()
        ack_task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await ack_task
        await asyncio.wait_for(generation_cancelled.wait(), timeout=0.1)
        adapter.send.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_deadline_does_not_wait_for_generation_cancellation_cleanup(
        self, monkeypatch
    ):
        import gateway.platforms.bluebubbles as bluebubbles

        monkeypatch.setattr(bluebubbles, "_QUICK_ACK_MIN_TIMEOUT_SECONDS", 0.01)
        runner, adapter, event, source = _quick_ack_runner(
            monkeypatch,
            _quick_ack_config(
                quick_ack_timeout_seconds=0.06,
                quick_ack_fallback="Got it — I’m checking.",
            ),
        )

        async def cancellation_delayed_generation(**_kwargs):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await asyncio.sleep(0.15)
                raise

        monkeypatch.setattr(
            "agent.auxiliary_client.async_call_llm",
            cancellation_delayed_generation,
        )
        started = asyncio.get_running_loop().time()
        ack = await runner._maybe_send_bluebubbles_quick_ack(
            event, source, event.text, []
        )
        elapsed = asyncio.get_running_loop().time() - started

        assert ack == "Got it — I’m checking."
        assert elapsed < 0.11
        adapter.send.assert_awaited_once_with(source.chat_id, ack)

    @pytest.mark.asyncio
    async def test_generation_timeout_still_sends_fallback_with_reserved_budget(
        self, monkeypatch
    ):
        import gateway.platforms.bluebubbles as bluebubbles

        monkeypatch.setattr(bluebubbles, "_QUICK_ACK_MIN_TIMEOUT_SECONDS", 0.01)
        runner, adapter, event, source = _quick_ack_runner(
            monkeypatch,
            _quick_ack_config(
                quick_ack_timeout_seconds=0.08,
                quick_ack_fallback="Checking now.",
            ),
        )

        async def hung_generation(**_kwargs):
            await asyncio.Event().wait()

        monkeypatch.setattr("agent.auxiliary_client.async_call_llm", hung_generation)
        started = asyncio.get_running_loop().time()
        ack = await runner._maybe_send_bluebubbles_quick_ack(
            event, source, event.text, []
        )
        elapsed = asyncio.get_running_loop().time() - started

        assert ack == "Checking now."
        assert elapsed < 0.12
        adapter.send.assert_awaited_once_with(source.chat_id, "Checking now.")

    @pytest.mark.asyncio
    async def test_end_to_end_timeout_bounds_hung_send_without_fallback_retry(
        self, monkeypatch
    ):
        import gateway.platforms.bluebubbles as bluebubbles

        monkeypatch.setattr(bluebubbles, "_QUICK_ACK_MIN_TIMEOUT_SECONDS", 0.01)
        runner, adapter, event, source = _quick_ack_runner(
            monkeypatch,
            _quick_ack_config(quick_ack_timeout_seconds=0.05),
        )
        monkeypatch.setattr(
            "agent.auxiliary_client.async_call_llm",
            AsyncMock(return_value=_aux_response("I’ll inspect this now.")),
        )

        async def hung_send(*_args, **_kwargs):
            await asyncio.Event().wait()

        adapter.send.side_effect = hung_send
        started = asyncio.get_running_loop().time()
        ack = await asyncio.wait_for(
            runner._maybe_send_bluebubbles_quick_ack(
                event, source, event.text, []
            ),
            timeout=0.15,
        )
        elapsed = asyncio.get_running_loop().time() - started

        assert ack is None
        assert elapsed < 0.15
        assert adapter.send.await_count == 1

    @pytest.mark.asyncio
    async def test_fallback_send_uses_only_deadline_remaining_after_generation_failure(
        self, monkeypatch
    ):
        import gateway.platforms.bluebubbles as bluebubbles

        monkeypatch.setattr(bluebubbles, "_QUICK_ACK_MIN_TIMEOUT_SECONDS", 0.01)
        runner, adapter, event, source = _quick_ack_runner(
            monkeypatch,
            _quick_ack_config(
                quick_ack_timeout_seconds=0.06,
                quick_ack_fallback="Got it — I’m checking.",
            ),
        )

        async def delayed_generation_failure(**_kwargs):
            await asyncio.sleep(0.04)
            raise RuntimeError("aux unavailable")

        async def hung_send(*_args, **_kwargs):
            await asyncio.Event().wait()

        monkeypatch.setattr(
            "agent.auxiliary_client.async_call_llm", delayed_generation_failure
        )
        adapter.send.side_effect = hung_send
        started = asyncio.get_running_loop().time()
        ack = await asyncio.wait_for(
            runner._maybe_send_bluebubbles_quick_ack(
                event, source, event.text, []
            ),
            timeout=0.12,
        )
        elapsed = asyncio.get_running_loop().time() - started

        assert ack is None
        assert elapsed < 0.11
        assert adapter.send.await_count == 1

    @pytest.mark.asyncio
    async def test_proxy_mode_receives_turn_notes_without_mutating_user_message(
        self, monkeypatch
    ):
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = SimpleNamespace(multiplex_profiles=False)
        monkeypatch.setattr(runner, "_get_proxy_url", lambda: "http://proxy.test")
        captured = {}

        async def fake_proxy(**kwargs):
            captured.update(kwargs)
            return {"final_response": "ok"}

        monkeypatch.setattr(runner, "_run_agent_via_proxy", fake_proxy)
        result = await runner._run_agent(
            message="original user text",
            context_prompt="base context",
            history=[],
            source=SimpleNamespace(),
            session_id="session-id",
            session_key="session-key",
            turn_sidecar_notes=["visible quick acknowledgment: checking now"],
        )

        assert result == {"final_response": "ok"}
        assert captured["message"] == "original user text"
        assert captured["context_prompt"].startswith("base context")
        assert "visible quick acknowledgment" in captured["context_prompt"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("failure", [RuntimeError("boom"), asyncio.CancelledError()])
    async def test_turn_local_ack_context_cannot_leak_after_failed_run(
        self, monkeypatch, failure
    ):
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = SimpleNamespace(multiplex_profiles=False)
        forwarded = []

        async def fake_inner(*_args, turn_sidecar_notes=None, **_kwargs):
            forwarded.append(list(turn_sidecar_notes or []))
            if len(forwarded) == 1:
                raise failure
            return {"final_response": "ok"}

        monkeypatch.setattr(runner, "_run_agent_inner", fake_inner)
        common = {
            "message": "message",
            "context_prompt": "context",
            "history": [],
            "source": SimpleNamespace(),
            "session_id": "session-id",
            "session_key": "session-key",
        }

        with pytest.raises(type(failure)):
            await runner._run_agent(
                **common,
                turn_sidecar_notes=["visible quick acknowledgment: first-turn"],
            )
        result = await runner._run_agent(**common, turn_sidecar_notes=[])

        assert result == {"final_response": "ok"}
        assert forwarded == [
            ["visible quick acknowledgment: first-turn"],
            [],
        ]

    @pytest.mark.asyncio
    async def test_visible_ack_is_one_shot_sidecar_not_user_authored_text(self, monkeypatch):
        """The ack note changes no role/content history and is consumed once.

        It is composed onto the API copy of this user turn, preserving strict
        alternation and the persisted clean user text while keeping later prompt
        prefixes byte-stable through the existing api_content sidecar.
        """
        from agent.turn_context import compose_user_api_content

        runner, _adapter, event, source = _quick_ack_runner(
            monkeypatch, _quick_ack_config()
        )
        monkeypatch.setattr(
            "agent.auxiliary_client.async_call_llm",
            AsyncMock(return_value=_aux_response("I’ll compare both carefully.")),
        )
        original_user_text = event.text
        notes = []

        await runner._maybe_send_bluebubbles_quick_ack(
            event, source, event.text, notes
        )
        api_content = compose_user_api_content(
            original_user_text, "", "\n\n".join(notes)
        )

        assert event.text == original_user_text
        assert len(notes) == 1
        assert "visible quick acknowledgment" in notes[0]
        assert "I’ll compare both carefully." in api_content
        assert api_content.startswith(original_user_text)


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
    """_webhook_url keeps local callbacks on explicit IPv4 loopback."""

    def test_default_host(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        # Default webhook_host is 0.0.0.0 → safe local registration target.
        assert "127.0.0.1" in adapter._webhook_url
        assert str(adapter.webhook_port) in adapter._webhook_url
        assert adapter.webhook_path in adapter._webhook_url

    @pytest.mark.parametrize("host", ["0.0.0.0", "127.0.0.1", "localhost", "::"])
    def test_local_hosts_normalized(self, monkeypatch, host):
        adapter = _make_adapter(monkeypatch, webhook_host=host)
        assert adapter._webhook_url.startswith("http://127.0.0.1:")

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

    def test_find_registered_webhooks_treats_loopback_aliases_as_equivalent(self, monkeypatch):
        import asyncio

        adapter = _make_adapter(monkeypatch, webhook_host="127.0.0.1")
        canonical = adapter._webhook_register_url
        alias = (
            canonical.replace("127.0.0.1", "localhost")
            if "127.0.0.1" in canonical
            else canonical.replace("localhost", "127.0.0.1")
        )
        adapter.client = self._mock_client(
            get_response={"status": 200, "data": [
                {"id": 1, "url": canonical, "events": ["new-message", "updated-message"]},
                {"id": 2, "url": alias, "events": ["new-message", "updated-message"]},
            ]}
        )

        result = asyncio.get_event_loop().run_until_complete(
            adapter._find_registered_webhooks(canonical)
        )

        assert [item["id"] for item in result] == [1, 2]

    def test_webhook_equivalence_preserves_userinfo_and_query_order(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        canonical = (
            "http://127.0.0.1:8645/bluebubbles-webhook"
            "?password=secret&event=new-message&event=updated-message"
        )
        alias = canonical.replace("127.0.0.1", "localhost")
        with_userinfo = alias.replace("http://", "http://user@")
        reordered = (
            "http://localhost:8645/bluebubbles-webhook"
            "?event=updated-message&event=new-message&password=secret"
        )

        assert adapter._normalized_webhook_url(alias) == adapter._normalized_webhook_url(canonical)
        assert adapter._normalized_webhook_url(with_userinfo) != adapter._normalized_webhook_url(canonical)
        assert adapter._normalized_webhook_url(reordered) != adapter._normalized_webhook_url(canonical)

    def test_webhook_normalization_preserves_custom_ipv6_brackets(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        url = "http://[2001:db8::1]:8645/bluebubbles-webhook?password=secret"

        assert adapter._normalized_webhook_url(url) == url

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

    def test_find_registered_webhooks_handles_api_error(self, monkeypatch):
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
        assert result == []

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
                {"id": 7, "url": url, "events": ["new-message", "updated-message"]},
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
    async def test_register_collapses_duplicate_equivalent_webhooks(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, webhook_host="127.0.0.1")
        canonical = adapter._webhook_register_url
        alias = canonical.replace("127.0.0.1", "localhost")
        delete_response = SimpleNamespace(raise_for_status=lambda: None)
        adapter.client = SimpleNamespace(
            delete=AsyncMock(return_value=delete_response),
        )
        monkeypatch.setattr(
            adapter,
            "_find_registered_webhooks",
            AsyncMock(return_value=[
                {"id": 7, "url": canonical, "events": ["new-message", "updated-message"]},
                {"id": 8, "url": alias, "events": ["new-message", "updated-message"]},
            ]),
        )
        post = AsyncMock(return_value={"status": 200, "data": {"id": 9}})
        monkeypatch.setattr(adapter, "_api_post", post)

        assert await adapter._register_webhook() is True
        client = adapter.client
        assert client is not None
        client.delete.assert_awaited_once()
        assert client.delete.await_args.args[0].endswith("/api/v1/webhook/8?password=secret")
        post.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_register_post_failure_preserves_existing_alias(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, webhook_host="127.0.0.1")
        alias = adapter._webhook_register_url.replace("127.0.0.1", "localhost")
        adapter.client = SimpleNamespace(delete=AsyncMock())
        monkeypatch.setattr(
            adapter,
            "_find_registered_webhooks",
            AsyncMock(return_value=[
                {"id": 8, "url": alias, "events": ["new-message", "updated-message"]},
            ]),
        )
        monkeypatch.setattr(
            adapter,
            "_api_post",
            AsyncMock(return_value={"status": 500, "message": "server error"}),
        )

        assert await adapter._register_webhook() is False
        client = adapter.client
        assert client is not None
        client.delete.assert_not_awaited()

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
        ok = asyncio.get_event_loop().run_until_complete(
            adapter._unregister_webhook()
        )
        assert ok is True

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
