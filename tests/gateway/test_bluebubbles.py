"""Tests for the BlueBubbles iMessage gateway adapter."""
import asyncio
import json
from unittest.mock import AsyncMock

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

    @pytest.mark.asyncio
    async def test_group_tapback_bypasses_mention_gate(self, monkeypatch):
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
                "guid": "group-tapback-1",
                "text": "Liked “Smoke test passed”",
                "handle": {"address": "+155****0100"},
                "isFromMe": False,
                "isGroup": True,
                "chats": [{"guid": "iMessage;+;group-chat"}],
            },
        }))
        await asyncio.sleep(0)

        assert response.status == 200
        assert [event.text for event in handled] == [
            "Reaction: User liked this message: Smoke test passed"
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("update_fields", "expected_fragment"),
        [
            ({"text": "second draft", "dateEdited": 123456789}, "Message edited."),
            ({"dateRetracted": 123456789}, "Message retracted/unsent."),
        ],
    )
    async def test_group_update_notification_bypasses_mention_gate(
        self, monkeypatch, update_fields, expected_fragment
    ):
        adapter = _make_adapter(
            monkeypatch,
            require_mention=True,
            send_read_receipts=False,
        )
        handled = []

        async def fake_handle_message(event):
            handled.append(event)

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        base = {
            "guid": "group-update-1",
            "handle": {"address": "+155****0100"},
            "isFromMe": False,
            "isGroup": True,
            "chats": [{"guid": "iMessage;+;group-chat"}],
        }
        await adapter._handle_webhook(_FakeBlueBubblesRequest({
            "type": "new-message",
            "data": {**base, "text": "first draft"},
        }))
        response = await adapter._handle_webhook(_FakeBlueBubblesRequest({
            "type": "updated-message",
            "data": {**base, **update_fields},
        }))
        await asyncio.sleep(0)

        assert response.status == 200
        assert len(handled) == 1
        assert expected_fragment in handled[0].text
        assert "first draft" in handled[0].text


class TestBlueBubblesUpdatedMessageHandling:
    async def _dispatch(self, adapter, payload):
        response = await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))
        await asyncio.sleep(0)
        return response

    @pytest.mark.asyncio
    async def test_updated_message_same_guid_and_text_is_deduped(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        handled = []

        async def fake_handle_message(event):
            handled.append(event)

        async def fake_mark_read(chat_id):
            return False

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        monkeypatch.setattr(adapter, "mark_read", fake_mark_read)

        base = {
            "data": {
                "guid": "MSG-GUID-1",
                "text": "hello",
                "handle": {"address": "user@example.com"},
                "isFromMe": False,
                "chats": [{"guid": "any;-;user@example.com"}],
            }
        }
        await self._dispatch(adapter, {"type": "new-message", **base})
        await self._dispatch(adapter, {"type": "updated-message", **base})

        assert len(handled) == 1
        assert handled[0].text == "hello"

    @pytest.mark.asyncio
    async def test_invalid_update_does_not_poison_valid_retry(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        handled = []

        async def fake_handle_message(event):
            handled.append(event)

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        monkeypatch.setattr(adapter, "mark_read", AsyncMock(return_value=False))

        invalid = {
            "type": "updated-message",
            "data": {
                "guid": "MSG-GUID-RETRY",
                "text": "corrected text",
                "isFromMe": False,
                "dateEdited": 123456789,
                "chats": {"guid": "malformed-chat-container"},
            },
        }
        invalid_response = await self._dispatch(adapter, invalid)

        assert invalid_response.status == 400
        assert handled == []
        assert getattr(adapter, "_recent_update_event_keys") == {}
        assert adapter._recent_message_texts == {}

        valid = {
            **invalid,
            "data": {
                **invalid["data"],
                "handle": {"address": "user@example.com"},
                "chats": [{"guid": "any;-;user@example.com"}],
            },
        }
        valid_response = await self._dispatch(adapter, valid)

        assert valid_response.status == 200
        assert len(handled) == 1
        assert handled[0].text == "Message edited.\nNew text: corrected text"

    @pytest.mark.asyncio
    async def test_concurrent_update_retry_is_reserved_before_attachment_download(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        handled = []

        async def fake_handle_message(event):
            handled.append(event)

        async def slow_download(att_guid, attachment):
            await asyncio.sleep(0.01)
            return "/tmp/update-image.png"

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        download_attachment = AsyncMock(side_effect=slow_download)
        monkeypatch.setattr(adapter, "_download_attachment", download_attachment)

        payload = {
            "type": "updated-message",
            "data": {
                "guid": "MSG-GUID-CONCURRENT",
                "text": "corrected text",
                "dateEdited": 123456789,
                "handle": {"address": "user@example.com"},
                "isFromMe": False,
                "chats": [{"guid": "any;-;user@example.com"}],
                "attachments": [
                    {
                        "guid": "ATTACHMENT-GUID",
                        "mimeType": {"malformed": True},
                        "uti": {"malformed": True},
                    }
                ],
            },
        }
        responses = await asyncio.gather(
            adapter._handle_webhook(_FakeBlueBubblesRequest(payload)),
            adapter._handle_webhook(_FakeBlueBubblesRequest(payload)),
        )
        await asyncio.sleep(0)

        assert [response.status for response in responses] == [200, 200]
        assert len(handled) == 1
        download_attachment.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_updated_message_receipt_without_text_is_acknowledged(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        handled = []

        async def fake_handle_message(event):
            handled.append(event)

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        download_attachment = AsyncMock(return_value="/tmp/duplicate-sensitive-media.png")
        monkeypatch.setattr(adapter, "_download_attachment", download_attachment)

        response = await self._dispatch(
            adapter,
            {
                "type": "updated-message",
                "data": {
                    "guid": "MSG-GUID-2",
                    "handle": {"address": "user@example.com"},
                    "isFromMe": False,
                    "chats": [{"guid": "any;-;user@example.com"}],
                    "dateEdited": None,
                    "dateRetracted": None,
                    "attachments": [
                        {"guid": "duplicate-attachment", "mimeType": "image/png"}
                    ],
                },
            },
        )

        assert response.status == 200
        assert handled == []
        download_attachment.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_updated_message_edit_forwards_before_after_context(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        handled = []

        async def fake_handle_message(event):
            handled.append(event)

        async def fake_mark_read(chat_id):
            return False

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        monkeypatch.setattr(adapter, "mark_read", fake_mark_read)

        original = {
            "type": "new-message",
            "data": {
                "guid": "MSG-GUID-3",
                "text": "first draft",
                "handle": {"address": "user@example.com"},
                "isFromMe": False,
                "chats": [{"guid": "any;-;user@example.com"}],
            },
        }
        edited = {
            "type": "updated-message",
            "data": {
                "guid": "MSG-GUID-3",
                "text": "second draft",
                "handle": {"address": "user@example.com"},
                "isFromMe": False,
                "chats": [{"guid": "any;-;user@example.com"}],
                "dateEdited": 123456789,
            },
        }

        await self._dispatch(adapter, original)
        await self._dispatch(adapter, edited)
        await self._dispatch(adapter, edited)

        assert len(handled) == 2
        assert "edited" in handled[1].text.lower()
        assert "first draft" in handled[1].text
        assert "second draft" in handled[1].text

        edited_back = {
            **edited,
            "data": {**edited["data"], "text": "first draft"},
        }
        await self._dispatch(adapter, edited_back)
        await self._dispatch(adapter, edited)

        assert len(handled) == 4
        assert "second draft" in handled[3].text

    @pytest.mark.asyncio
    async def test_updated_message_retraction_notifies_agent_with_cached_text(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        handled = []

        async def fake_handle_message(event):
            handled.append(event)

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        mark_read = AsyncMock(return_value=False)
        monkeypatch.setattr(adapter, "mark_read", mark_read)

        await self._dispatch(
            adapter,
            {
                "type": "new-message",
                "data": {
                    "guid": "MSG-GUID-4",
                    "text": "please unsend me",
                    "handle": {"address": "user@example.com"},
                    "isFromMe": False,
                    "chats": [{"guid": "any;-;user@example.com"}],
                },
            },
        )
        retraction = {
            "type": "updated-message",
            "data": {
                "guid": "MSG-GUID-4",
                "handle": {"address": "user@example.com"},
                "isFromMe": False,
                "chats": [{"guid": "any;-;user@example.com"}],
                "dateRetracted": 123456789,
            },
        }
        await self._dispatch(adapter, retraction)
        await self._dispatch(adapter, retraction)

        assert len(handled) == 2
        assert "retracted" in handled[1].text.lower() or "unsent" in handled[1].text.lower()
        assert "please unsend me" in handled[1].text
        assert mark_read.await_count == 2

    @pytest.mark.asyncio
    async def test_tapback_text_is_forwarded_as_reaction_event(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        handled = []

        async def fake_handle_message(event):
            handled.append(event)

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        mark_read = AsyncMock(return_value=False)
        monkeypatch.setattr(adapter, "mark_read", mark_read)

        tapback = {
            "type": "new-message",
            "data": {
                "guid": "TAPBACK-GUID-1",
                "text": "Liked “Smoke test passed”",
                "handle": {"address": "user@example.com"},
                "isFromMe": False,
                "chats": [{"guid": "any;-;user@example.com"}],
            },
        }
        await self._dispatch(adapter, tapback)
        await self._dispatch(adapter, tapback)

        assert len(handled) == 1
        assert handled[0].text == "Reaction: User liked this message: Smoke test passed"
        assert handled[0].source.chat_id == "user@example.com"
        mark_read.assert_awaited_once_with("user@example.com")

    @pytest.mark.asyncio
    async def test_removed_tapback_text_is_forwarded_as_reaction_removal(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        handled = []

        async def fake_handle_message(event):
            handled.append(event)

        async def fake_mark_read(chat_id):
            return False

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        monkeypatch.setattr(adapter, "mark_read", fake_mark_read)

        await self._dispatch(
            adapter,
            {
                "type": "new-message",
                "data": {
                    "guid": "TAPBACK-GUID-2",
                    "text": "Removed a like from “Smoke test passed”",
                    "handle": {"address": "user@example.com"},
                    "isFromMe": False,
                    "chats": [{"guid": "any;-;user@example.com"}],
                },
            },
        )

        assert len(handled) == 1
        assert handled[0].text == "Reaction removed: User removed a like Tapback from: Smoke test passed"

    @pytest.mark.asyncio
    async def test_associated_message_type_string_is_forwarded_as_reaction(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        handled = []

        async def fake_handle_message(event):
            handled.append(event)

        async def fake_mark_read(chat_id):
            return False

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        monkeypatch.setattr(adapter, "mark_read", fake_mark_read)

        tapback = {
            "type": "new-message",
            "data": {
                "guid": "TAPBACK-GUID-3",
                "text": "Smoke test passed",
                "associatedMessageType": "2003",
                "handle": {"address": "user@example.com"},
                "isFromMe": False,
                "chats": [{"guid": "any;-;user@example.com"}],
            },
        }
        await self._dispatch(adapter, tapback)
        numeric_tapback = {
            **tapback,
            "data": {**tapback["data"], "associatedMessageType": 2003},
        }
        await self._dispatch(adapter, numeric_tapback)

        assert len(handled) == 1
        assert handled[0].text == "Reaction: User added a laugh Tapback to: Smoke test passed"

    @pytest.mark.asyncio
    async def test_dm_chat_guid_and_plain_identifier_use_same_session(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        handled = []

        async def fake_handle_message(event):
            handled.append(event)

        async def fake_mark_read(chat_id):
            return False

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        monkeypatch.setattr(adapter, "mark_read", fake_mark_read)

        await self._dispatch(
            adapter,
            {
                "type": "new-message",
                "data": {
                    "guid": "MSG-GUID-5",
                    "text": "first draft",
                    "handle": {"address": "user@example.com"},
                    "isFromMe": False,
                    "chats": [{"guid": "any;-;user@example.com"}],
                },
            },
        )
        await self._dispatch(
            adapter,
            {
                "type": "updated-message",
                "data": {
                    "guid": "MSG-GUID-5",
                    "text": "second draft",
                    "chatIdentifier": "user@example.com",
                    "handle": {"address": "user@example.com"},
                    "isFromMe": False,
                    "dateEdited": 123456789,
                },
            },
        )

        assert len(handled) == 2
        assert handled[0].source.chat_id == "user@example.com"
        assert handled[1].source.chat_id == "user@example.com"
        assert "first draft" in handled[1].text
        assert "second draft" in handled[1].text


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


