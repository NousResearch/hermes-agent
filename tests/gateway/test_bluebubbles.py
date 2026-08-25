"""Tests for the BlueBubbles iMessage gateway adapter."""
import asyncio
import json

import httpx
import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageType


def _make_adapter(monkeypatch, **extra):
    monkeypatch.setenv("BLUEBUBBLES_SERVER_URL", "http://localhost:1234")
    monkeypatch.setenv("BLUEBUBBLES_PASSWORD", "secret")
    from gateway.platforms.bluebubbles import BlueBubblesAdapter

    cfg = PlatformConfig(
        enabled=True,
        extra={
            "server_url": "http://localhost:1234",
            "password": "secret",
            "message_revision_wait_seconds": 0,
            **extra,
        },
    )
    adapter = BlueBubblesAdapter(cfg)

    async def verified_test_membership(message_guid, candidate_chat_guid):
        if candidate_chat_guid:
            return candidate_chat_guid
        return await adapter._resolve_exact_message_chat_guid(message_guid)

    monkeypatch.setattr(
        adapter,
        "_verify_inbound_message_membership",
        verified_test_membership,
    )
    return adapter


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


    def test_top_level_participant_names_are_bridged_to_adapter_extra(
        self, monkeypatch, tmp_path
    ):
        (tmp_path / "config.yaml").write_text(
            """
platforms:
  bluebubbles:
    enabled: true
    participant_names:
      "+15550000100": "Mark"
""".strip(),
            encoding="utf-8",
        )
        monkeypatch.setattr("gateway.config.get_hermes_home", lambda: tmp_path)
        monkeypatch.setenv("BLUEBUBBLES_SERVER_URL", "http://localhost:1234")
        monkeypatch.setenv("BLUEBUBBLES_PASSWORD", "secret")

        from gateway.config import load_gateway_config

        config = load_gateway_config()

        assert config.platforms[Platform.BLUEBUBBLES].extra["participant_names"] == {
            "+15550000100": "Mark"
        }


class TestBlueBubblesHelpers:
    def test_group_reply_anchor_is_top_level_unless_user_used_inline_reply(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch)
        from gateway.platforms.base import MessageEvent, _reply_anchor_for_event

        source = adapter.build_source(
            chat_id="iMessage;+;family-group",
            chat_name="family",
            chat_type="group",
            user_id="user@example.com",
        )
        top_level = MessageEvent(
            text="ducky, answer this",
            source=source,
            message_id="current-message-guid",
        )
        inline = MessageEvent(
            text="ducky, follow up",
            source=source,
            message_id="current-reply-guid",
            reply_to_message_id="original-thread-guid",
        )

        assert _reply_anchor_for_event(top_level) is None
        assert _reply_anchor_for_event(inline) == "original-thread-guid"

    def test_dm_reply_anchor_keeps_current_message(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        from gateway.platforms.base import MessageEvent, _reply_anchor_for_event

        event = MessageEvent(
            text="hello",
            source=adapter.build_source(
                chat_id="iMessage;-;user@example.com",
                chat_name="user@example.com",
                chat_type="dm",
                user_id="user@example.com",
            ),
            message_id="current-message-guid",
        )

        assert _reply_anchor_for_event(event) == "current-message-guid"

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
    async def test_send_keeps_paragraphs_in_one_bubble_when_under_limit(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        sent = []

        async def fake_resolve_chat_guid(chat_id):
            return "iMessage;-;user@example.com"

        async def fake_api_post(path, payload):
            sent.append(payload["message"])
            return {"data": {"guid": f"msg-{len(sent)}"}}

        monkeypatch.setattr(adapter, "_resolve_chat_guid", fake_resolve_chat_guid)
        monkeypatch.setattr(adapter, "_api_post", fake_api_post)

        content = "first thought\n\nsecond thought"
        result = await adapter.send("user@example.com", content)

        assert result.success is True
        assert sent == [content]

    @pytest.mark.asyncio
    async def test_send_deduplicates_concurrent_content_for_same_origin(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        sent = []

        async def fake_resolve_chat_guid(chat_id):
            return "iMessage;-;user@example.com"

        async def fake_api_post(path, payload):
            sent.append(payload)
            await asyncio.sleep(0.01)
            return {"data": {"guid": "assistant-message-guid"}}

        monkeypatch.setattr(adapter, "_resolve_chat_guid", fake_resolve_chat_guid)
        monkeypatch.setattr(adapter, "_api_post", fake_api_post)
        metadata = {"reply_to_message_id": "origin-message-guid"}

        first, second = await asyncio.gather(
            adapter.send("user@example.com", "same answer", metadata=metadata),
            adapter.send("user@example.com", "same answer", metadata=metadata),
        )

        assert first.success is True
        assert second.success is True
        assert [payload["message"] for payload in sent] == ["same answer"]
        assert {first.raw_response.get("deduplicated"), second.raw_response.get("deduplicated")} == {None, True}

    @pytest.mark.asyncio
    async def test_send_without_origin_does_not_deduplicate_legitimate_repeats(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        sent = []

        async def fake_resolve_chat_guid(chat_id):
            return "iMessage;-;user@example.com"

        async def fake_api_post(path, payload):
            sent.append(payload["message"])
            return {"data": {"guid": f"message-{len(sent)}"}}

        monkeypatch.setattr(adapter, "_resolve_chat_guid", fake_resolve_chat_guid)
        monkeypatch.setattr(adapter, "_api_post", fake_api_post)

        await adapter.send("user@example.com", "legitimate repeat")
        await adapter.send("user@example.com", "legitimate repeat")

        assert sent == ["legitimate repeat", "legitimate repeat"]

    @pytest.mark.asyncio
    async def test_send_reuses_idempotency_key_after_ambiguous_failure(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        payloads = []

        async def fake_resolve_chat_guid(chat_id):
            return "iMessage;-;user@example.com"

        async def fake_api_post(path, payload):
            payloads.append(payload)
            if len(payloads) == 1:
                raise TimeoutError("response lost")
            return {"data": {"guid": "assistant-message-guid"}}

        monkeypatch.setattr(adapter, "_resolve_chat_guid", fake_resolve_chat_guid)
        monkeypatch.setattr(adapter, "_api_post", fake_api_post)
        metadata = {"reply_to_message_id": "origin-message-guid"}

        first = await adapter.send("user@example.com", "same answer", metadata=metadata)
        second = await adapter.send("user@example.com", "same answer", metadata=metadata)

        assert first.success is False
        assert second.success is True
        assert payloads[0]["tempGuid"] == payloads[1]["tempGuid"]

    @pytest.mark.asyncio
    async def test_send_suppresses_only_structured_internal_notice_over_sms(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        api_calls = []

        async def fake_resolve_chat_guid(chat_id):
            return "SMS;-;+155****0100"

        async def fake_api_post(path, payload):
            api_calls.append((path, payload))
            return {"data": {"guid": "sent-message"}}

        monkeypatch.setattr(adapter, "_resolve_chat_guid", fake_resolve_chat_guid)
        monkeypatch.setattr(adapter, "_api_post", fake_api_post)
        text = "Gateway shutting down for maintenance"

        legitimate = await adapter.send("+155****0100", text)
        suppressed = await adapter.send(
            "+155****0100",
            text,
            metadata={"internal_notice": True},
        )

        assert legitimate.success is True
        assert suppressed.success is True
        assert suppressed.raw_response == {"suppressed": "internal_sms_notice"}
        assert [payload["message"] for _, payload in api_calls] == [text]

    @pytest.mark.asyncio
    async def test_internal_notice_does_not_create_unresolved_phone_chat(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._private_api_enabled = True
        create_calls = []

        async def fake_resolve_chat_guid(chat_id):
            return None

        async def fake_create_chat(address, message, temp_guid=None):
            create_calls.append((address, message, temp_guid))
            raise AssertionError("internal notice must not create an unresolved phone chat")

        monkeypatch.setattr(adapter, "_resolve_chat_guid", fake_resolve_chat_guid)
        monkeypatch.setattr(adapter, "_create_chat_for_handle", fake_create_chat)

        result = await adapter.send(
            "+15555550100",
            "Gateway restarting",
            metadata={"internal_notice": True},
        )

        assert result.success is True
        assert result.raw_response == {"suppressed": "internal_sms_notice"}
        assert create_calls == []

    @pytest.mark.asyncio
    async def test_internal_notice_allows_plus_prefixed_email_handle(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._private_api_enabled = True
        payloads = []

        async def fake_resolve_chat_guid(chat_id):
            return None

        async def fake_api_post(path, payload):
            payloads.append((path, payload))
            return {"data": {"guid": "new-chat-message"}}

        monkeypatch.setattr(adapter, "_resolve_chat_guid", fake_resolve_chat_guid)
        monkeypatch.setattr(adapter, "_api_post", fake_api_post)

        result = await adapter.send(
            "+15555550100@example.com",
            "Internal iMessage notice",
            metadata={"internal_notice": True},
        )

        assert result.success is True
        assert [path for path, _ in payloads] == ["/api/v1/chat/new"]

    @pytest.mark.asyncio
    async def test_new_chat_retry_reuses_idempotency_key(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._private_api_enabled = True
        payloads = []

        async def fake_resolve_chat_guid(chat_id):
            return None

        async def fake_api_post(path, payload):
            assert path == "/api/v1/chat/new"
            payloads.append(payload)
            if len(payloads) == 1:
                raise TimeoutError("response lost")
            return {"data": {"guid": "new-chat-message"}}

        monkeypatch.setattr(adapter, "_resolve_chat_guid", fake_resolve_chat_guid)
        monkeypatch.setattr(adapter, "_api_post", fake_api_post)
        metadata = {"reply_to_message_id": "origin-message-guid"}

        first = await adapter.send("user@example.com", "hello", metadata=metadata)
        second = await adapter.send("user@example.com", "hello", metadata=metadata)

        assert first.success is False
        assert second.success is True
        assert payloads[0]["tempGuid"] == payloads[1]["tempGuid"]

    @pytest.mark.asyncio
    async def test_new_chat_sends_remaining_chunks_after_creation(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._private_api_enabled = True
        resolve_calls = 0
        payloads = []

        async def fake_resolve_chat_guid(chat_id):
            nonlocal resolve_calls
            resolve_calls += 1
            return None if resolve_calls == 1 else "iMessage;-;user@example.com"

        async def fake_api_post(path, payload):
            payloads.append((path, payload))
            return {"data": {"guid": f"message-{len(payloads)}"}}

        monkeypatch.setattr(adapter, "_resolve_chat_guid", fake_resolve_chat_guid)
        monkeypatch.setattr(adapter, "_api_post", fake_api_post)
        text = "x" * (adapter.MAX_MESSAGE_LENGTH + 1)

        result = await adapter.send(
            "user@example.com",
            text,
            metadata={"reply_to_message_id": "origin-message-guid"},
        )

        assert result.success is True
        assert [path for path, _ in payloads] == [
            "/api/v1/chat/new",
            "/api/v1/message/text",
        ]
        assert "".join(payload["message"] for _, payload in payloads) == text

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


    def test_init_normalizes_webhook_path(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, webhook_path="bluebubbles-webhook")
        assert adapter.webhook_path == "/bluebubbles-webhook"


    def test_server_url_normalized(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, server_url="http://localhost:1234/")
        assert adapter.server_url == "http://localhost:1234"


class TestBlueBubblesReactions:
    @pytest.mark.asyncio
    async def test_processing_start_never_generates_automatic_tapback(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, ack_reaction="like")
        from gateway.platforms.base import MessageEvent

        calls = []

        async def fake_send_reaction(chat_id, message_guid, reaction, part_index=0):
            calls.append((chat_id, message_guid, reaction, part_index))

        monkeypatch.setattr(adapter, "send_reaction", fake_send_reaction)
        adapter.set_authorization_check(lambda user_id, chat_type, chat_id: True)
        event = MessageEvent(
            text="check this",
            source=adapter.build_source(
                chat_id="iMessage;+;exact-family-guid",
                chat_name="family-group",
                chat_type="group",
                user_id="+155****0100",
            ),
            message_id="inbound-message-guid",
        )

        await adapter.on_processing_start(event)

        assert calls == []

    @pytest.mark.asyncio
    async def test_processing_start_sends_no_ack_without_positive_authorization(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch, ack_reaction="like")
        from gateway.platforms.base import MessageEvent

        calls = []

        async def fake_send_reaction(*args, **kwargs):
            calls.append((args, kwargs))

        monkeypatch.setattr(adapter, "send_reaction", fake_send_reaction)
        adapter.set_authorization_check(lambda user_id, chat_type, chat_id: False)
        event = MessageEvent(
            text="check this",
            source=adapter.build_source(
                chat_id="iMessage;+;exact-family-guid",
                chat_name="family-group",
                chat_type="group",
                user_id="+155****0100",
            ),
            message_id="inbound-message-guid",
        )

        await adapter.on_processing_start(event)

        assert calls == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "reaction",
        [
            "love",
            "like",
            "dislike",
            "laugh",
            "emphasize",
            "question",
            "-love",
            "-like",
            "-dislike",
            "-laugh",
            "-emphasize",
            "-question",
        ],
    )
    async def test_send_reaction_posts_supported_tapbacks_to_exact_chat(
        self, monkeypatch, reaction
    ):
        adapter = _make_adapter(monkeypatch)
        adapter._private_api_enabled = True
        adapter._helper_connected = True
        adapter.client = object()
        calls = []

        async def fake_resolve_chat_guid(chat_id):
            assert chat_id == "family-group"
            return "iMessage;+;exact-family-guid"

        async def fake_api_post(path, payload):
            calls.append((path, payload))
            if path == "/api/v1/message/query":
                return {
                    "data": [
                        {
                            "guid": "target-message-guid",
                            "chats": [{"guid": "iMessage;+;exact-family-guid"}],
                        }
                    ]
                }
            assert path == "/api/v1/message/react"
            return {"status": 200, "data": {"guid": "reaction-guid"}}

        async def fake_refresh_helper_state():
            return True

        monkeypatch.setattr(adapter, "_resolve_chat_guid", fake_resolve_chat_guid)
        monkeypatch.setattr(adapter, "_api_post", fake_api_post)
        monkeypatch.setattr(adapter, "_refresh_helper_state", fake_refresh_helper_state)

        result = await adapter.send_reaction(
            "family-group", "target-message-guid", reaction, part_index=2
        )

        assert result.success is True
        assert result.message_id == "reaction-guid"
        assert calls == [
            (
                "/api/v1/message/query",
                {
                    "limit": 1,
                    "offset": 0,
                    "chatGuid": "iMessage;+;exact-family-guid",
                    "with": ["chats"],
                    "where": [
                        {
                            "statement": "message.guid = :guid",
                            "args": {"guid": "target-message-guid"},
                        }
                    ],
                },
            ),
            (
                "/api/v1/message/react",
                {
                    "chatGuid": "iMessage;+;exact-family-guid",
                    "selectedMessageGuid": "target-message-guid",
                    "reaction": reaction,
                    "partIndex": 2,
                },
            )
        ]

    @pytest.mark.asyncio
    async def test_send_reaction_fails_explicitly_without_live_private_helper(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch)
        adapter._private_api_enabled = True
        adapter._helper_connected = True
        adapter.client = None

        result = await adapter.send_reaction(
            "iMessage;-;user@example.com", "target-message-guid", "like"
        )

        assert result.success is False
        assert result.error == "BlueBubbles is not connected"

    @pytest.mark.asyncio
    async def test_send_reaction_refreshes_stale_private_helper_gate(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._private_api_enabled = True
        adapter._helper_connected = False
        adapter.client = object()
        posts = []

        async def fake_api_get(path):
            assert path == "/api/v1/server/info"
            return {"data": {"private_api": True, "helper_connected": True}}

        async def fake_api_post(path, payload):
            posts.append((path, payload))
            if path == "/api/v1/message/query":
                return {
                    "data": [
                        {
                            "guid": "target-message-guid",
                            "chats": [{"guid": "iMessage;-;user@example.com"}],
                        }
                    ]
                }
            return {"data": {"guid": "reaction-guid"}}

        monkeypatch.setattr(adapter, "_api_get", fake_api_get)
        monkeypatch.setattr(adapter, "_api_post", fake_api_post)

        result = await adapter.send_reaction(
            "iMessage;-;user@example.com", "target-message-guid", "love"
        )

        assert result.success is True
        assert [path for path, _ in posts] == [
            "/api/v1/message/query",
            "/api/v1/message/react",
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("message_guid", "reaction", "part_index", "error_fragment"),
        [
            ("target-message-guid", "party", 0, "Unsupported"),
            ("", "like", 0, "requires message GUID"),
            ("target-message-guid", "like", -1, "part index"),
        ],
    )
    async def test_send_reaction_rejects_unsafe_request_before_io(
        self,
        monkeypatch,
        message_guid,
        reaction,
        part_index,
        error_fragment,
    ):
        adapter = _make_adapter(monkeypatch)
        adapter.client = object()

        async def fail_refresh():
            raise AssertionError("invalid reaction must fail before helper I/O")

        monkeypatch.setattr(adapter, "_refresh_helper_state", fail_refresh)

        result = await adapter.send_reaction(
            "iMessage;-;user@example.com",
            message_guid,
            reaction,
            part_index=part_index,
        )

        assert result.success is False
        assert error_fragment in result.error

    @pytest.mark.asyncio
    async def test_send_reaction_rejects_message_from_a_different_chat(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._private_api_enabled = True
        adapter._helper_connected = True
        adapter.client = object()
        calls = []

        async def fake_api_post(path, payload):
            calls.append((path, payload))
            if path == "/api/v1/message/query":
                return {
                    "data": [
                        {
                            "guid": "target-message-guid",
                            "chats": [{"guid": "iMessage;+;other-family-guid"}],
                        }
                    ]
                }
            raise AssertionError("cross-chat reaction must fail before mutation")

        async def fake_refresh_helper_state():
            return True

        monkeypatch.setattr(adapter, "_api_post", fake_api_post)
        monkeypatch.setattr(adapter, "_refresh_helper_state", fake_refresh_helper_state)

        result = await adapter.send_reaction(
            "iMessage;+;exact-family-guid", "target-message-guid", "like"
        )

        assert result.success is False
        assert "does not belong to chat" in (result.error or "")
        assert [path for path, _ in calls] == ["/api/v1/message/query"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("authorization", [None, False])
    async def test_unauthorized_tapback_with_attachment_has_no_side_effects(
        self, monkeypatch, authorization
    ):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        downloads = []
        platform_events = []

        if authorization is not None:
            adapter.set_authorization_check(
                lambda user_id, chat_type, chat_id: authorization
            )

        async def capture_download(att_guid, att_meta):
            downloads.append((att_guid, att_meta))
            return f"/tmp/{att_guid}"

        async def capture_platform_event(event, source):
            platform_events.append((event, source))

        monkeypatch.setattr(adapter, "_download_attachment", capture_download)
        adapter.set_platform_event_handler(capture_platform_event)

        response = await adapter._handle_webhook(
            _FakeBlueBubblesRequest(
                {
                    "type": "new-message",
                    "data": {
                        "guid": "unauthorized-tapback-guid",
                        "text": "",
                        "associatedMessageType": 2001,
                        "associatedMessageGuid": "p:0/target-message-guid",
                        "handle": {"address": "+15555550100"},
                        "isFromMe": False,
                        "chatGuid": "iMessage;-;+15555550100",
                        "attachments": [
                            {"guid": "must-not-download", "mimeType": "image/png"}
                        ],
                    },
                }
            )
        )

        assert response.status == 200
        assert downloads == []
        assert platform_events == []
        assert adapter._message_revision_serials == {}
        assert adapter._pending_message_identities == set()
        assert adapter._active_attachment_revisions == {}
        assert adapter._active_attachment_leases == set()

    @pytest.mark.asyncio
    async def test_inbound_tapback_rejects_target_outside_exact_chat(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        adapter.set_authorization_check(lambda user_id, chat_type, chat_id: True)
        downloads = []
        platform_events = []
        exact_queries = []

        async def return_cross_chat_target(path):
            exact_queries.append(path)
            return {
                "data": {
                    "guid": "other-chat-target-guid",
                    "chats": [{"guid": "iMessage;+;other-family-guid"}],
                }
            }

        async def capture_platform_event(event, source):
            platform_events.append((event, source))

        async def capture_download(att_guid, att_meta):
            downloads.append((att_guid, att_meta))
            return f"/tmp/{att_guid}"

        monkeypatch.setattr(adapter, "_api_get", return_cross_chat_target)
        monkeypatch.setattr(adapter, "_download_attachment", capture_download)
        adapter.set_platform_event_handler(capture_platform_event)

        response = await adapter._handle_webhook(
            _FakeBlueBubblesRequest(
                {
                    "type": "new-message",
                    "data": {
                        "guid": "cross-chat-tapback-guid",
                        "text": "",
                        "associatedMessageType": 2001,
                        "associatedMessageGuid": "p:0/other-chat-target-guid",
                        "handle": {"address": "+15555550100"},
                        "isFromMe": False,
                        "chatGuid": "iMessage;+;exact-family-guid",
                        "chatIdentifier": "family-group",
                        "attachments": [
                            {
                                "guid": "cross-chat-must-not-download",
                                "mimeType": "image/png",
                            }
                        ],
                    },
                }
            )
        )

        assert response.status == 200
        assert exact_queries == [
            "/api/v1/message/other-chat-target-guid?with=chats"
        ]
        assert downloads == []
        assert platform_events == []
        assert adapter._message_revision_serials == {}
        assert adapter._pending_message_revisions == {}
        assert adapter._pending_message_identities == set()
        assert adapter._seen_message_guids == {}
        assert adapter._active_attachment_revisions == {}
        assert adapter._active_attachment_leases == set()
        assert adapter._active_attachment_identity_leases == {}
        assert adapter._message_revision_text == {}
        assert adapter._tapback_event_states == {}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("associated_type", "action", "reaction"),
        [
            (2001, "added", "like"),
            ("2001", "added", "like"),
            (3005, "removed", "question"),
            ("3005", "removed", "question"),
        ],
    )
    async def test_inbound_tapback_uses_authorized_platform_event_boundary_once(
        self, monkeypatch, associated_type, action, reaction
    ):
        adapter = _make_adapter(
            monkeypatch,
            require_mention=True,
            send_read_receipts=False,
        )
        platform_events = []
        normal_messages = []

        async def capture_platform_event(event, source):
            platform_events.append((event, source))

        async def capture_normal_message(event):
            normal_messages.append(event)

        async def exact_target(chat_guid, message_guid, **kwargs):
            return {
                "guid": message_guid,
                "chats": [{"guid": chat_guid}],
            }

        adapter.set_authorization_check(lambda user_id, chat_type, chat_id: True)
        adapter.set_platform_event_handler(capture_platform_event)
        monkeypatch.setattr(adapter, "handle_message", capture_normal_message)
        monkeypatch.setattr(adapter, "_query_exact_message", exact_target)

        response = await adapter._handle_webhook(
            _FakeBlueBubblesRequest(
                {
                    "type": "new-message",
                    "data": {
                        "guid": "tapback-event-guid",
                        "text": "",
                        "associatedMessageType": associated_type,
                        "associatedMessageGuid": "p:0/target-message-guid",
                        "handle": {"address": "+15555550100"},
                        "isFromMe": False,
                        "isGroup": True,
                        "chatGuid": "iMessage;+;exact-family-guid",
                        "chatIdentifier": "family-group",
                    },
                }
            )
        )
        await asyncio.sleep(0)

        assert response.status == 200
        assert normal_messages == []
        assert len(platform_events) == 1
        event, source = platform_events[0]
        assert event["event_type"] == "reaction"
        assert event["payload"]["action"] == action
        assert event["payload"]["reaction"] == reaction
        assert event["payload"]["message_id"] == "target-message-guid"
        assert event["payload"]["reaction_message_id"] == "tapback-event-guid"
        assert source.chat_id == "iMessage;+;exact-family-guid"
        assert source.chat_type == "group"
        assert source.user_id == event["payload"]["user_id"]

    @pytest.mark.asyncio
    async def test_duplicate_inbound_tapback_webhooks_emit_one_platform_event(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        platform_events = []

        async def capture_platform_event(event, source):
            platform_events.append((event, source))

        async def exact_target(chat_guid, message_guid, **kwargs):
            return {
                "guid": message_guid,
                "chats": [{"guid": chat_guid}],
            }

        adapter.set_authorization_check(lambda user_id, chat_type, chat_id: True)
        adapter.set_platform_event_handler(capture_platform_event)
        monkeypatch.setattr(adapter, "_query_exact_message", exact_target)
        payload = {
            "type": "updated-message",
            "data": {
                "guid": "tapback-event-guid",
                "text": "",
                "associatedMessageType": 2001,
                "associatedMessageGuid": "p:0/target-message-guid",
                "handle": {"address": "+155****0100"},
                "isFromMe": False,
                "isGroup": True,
                "chatGuid": "iMessage;+;exact-family-guid",
                "chatIdentifier": "family-group",
            },
        }

        first = await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))
        second = await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))

        assert first.status == 200
        assert second.status == 200
        assert len(platform_events) == 1

    @pytest.mark.asyncio
    async def test_tapback_dedup_preserves_add_remove_add_with_reused_guid(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        adapter.set_authorization_check(lambda user_id, chat_type, chat_id: True)
        platform_events = []

        async def exact_target(chat_guid, message_guid, **kwargs):
            return {
                "guid": message_guid,
                "chats": [{"guid": chat_guid}],
            }

        async def capture_platform_event(event, source):
            platform_events.append((event, source))

        monkeypatch.setattr(adapter, "_query_exact_message", exact_target)
        adapter.set_platform_event_handler(capture_platform_event)
        base = {
            "guid": "reused-tapback-guid",
            "text": "",
            "associatedMessageGuid": "p:2/target-message-guid",
            "handle": {"address": "+15555550100"},
            "isFromMe": False,
            "isGroup": True,
            "chatGuid": "iMessage;+;exact-family-guid",
            "chatIdentifier": "family-group",
        }

        for associated_type in (2001, 2001, 3001, 2001):
            await adapter._handle_webhook(
                _FakeBlueBubblesRequest(
                    {
                        "type": "updated-message",
                        "data": {
                            **base,
                            "associatedMessageType": associated_type,
                        },
                    }
                )
            )

        assert [
            (event["payload"]["action"], event["payload"]["reaction"])
            for event, _source in platform_events
        ] == [
            ("added", "like"),
            ("removed", "like"),
            ("added", "like"),
        ]


class TestBlueBubblesHelperState:
    @pytest.mark.asyncio
    async def test_helper_state_does_not_poll_when_live_state_is_already_true(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch)
        adapter._private_api_enabled = True
        adapter._helper_connected = True
        adapter.client = object()

        async def fail_api_get(path):
            raise AssertionError(f"live helper state should not repoll {path}")

        monkeypatch.setattr(adapter, "_api_get", fail_api_get)

        assert await adapter._refresh_helper_state() is True

    @pytest.mark.asyncio
    async def test_concurrent_stale_helper_refreshes_share_one_server_info_request(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch)
        adapter._private_api_enabled = False
        adapter._helper_connected = False
        adapter.client = object()
        calls = 0
        release = asyncio.Event()

        async def fake_api_get(path):
            nonlocal calls
            calls += 1
            await release.wait()
            return {"data": {"private_api": True, "helper_connected": True}}

        monkeypatch.setattr(adapter, "_api_get", fake_api_get)
        first = asyncio.create_task(adapter._refresh_helper_state())
        second = asyncio.create_task(adapter._refresh_helper_state())
        await asyncio.sleep(0)
        release.set()

        assert await asyncio.gather(first, second) == [True, True]
        assert calls == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("raises", [False, True])
    async def test_concurrent_negative_helper_refreshes_share_result_and_retry_after_ttl(
        self, monkeypatch, raises
    ):
        adapter = _make_adapter(monkeypatch)
        adapter._private_api_enabled = False
        adapter._helper_connected = False
        adapter.client = object()
        calls = 0
        release = asyncio.Event()

        async def fake_api_get(path):
            nonlocal calls
            calls += 1
            await release.wait()
            if raises:
                raise RuntimeError("server info unavailable")
            return {"data": {"private_api": True, "helper_connected": False}}

        monkeypatch.setattr(adapter, "_api_get", fake_api_get)
        first = asyncio.create_task(adapter._refresh_helper_state())
        second = asyncio.create_task(adapter._refresh_helper_state())
        third = asyncio.create_task(adapter._refresh_helper_state())
        await asyncio.sleep(0)
        release.set()

        assert await asyncio.gather(first, second, third) == [False, False, False]
        assert calls == 1

        adapter._helper_last_refresh_at = 0.0
        assert await adapter._refresh_helper_state() is False
        assert calls == 2

    @pytest.mark.asyncio
    async def test_send_typing_refreshes_stale_helper_state(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._private_api_enabled = True
        adapter._helper_connected = False
        posts = []

        class FakeClient:
            async def post(self, url, timeout):
                posts.append((url, timeout))

        async def fake_api_get(path):
            assert path == "/api/v1/server/info"
            return {"data": {"private_api": True, "helper_connected": True}}

        adapter.client = FakeClient()
        monkeypatch.setattr(adapter, "_api_get", fake_api_get)

        await adapter.send_typing("iMessage;-;user@example.com")

        assert len(posts) == 1
        assert "/api/v1/chat/iMessage%3B-%3Buser%40example.com/typing" in posts[0][0]

    @pytest.mark.asyncio
    async def test_stop_typing_refreshes_cold_boot_helper_state(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch)
        adapter._private_api_enabled = None
        adapter._helper_connected = False
        deletes = []
        info_calls = []

        class FakeClient:
            async def delete(self, url, timeout):
                deletes.append((url, timeout))

        async def fake_api_get(path):
            info_calls.append(path)
            return {
                "data": {
                    "private_api": True,
                    "helper_connected": True,
                }
            }

        adapter.client = FakeClient()
        monkeypatch.setattr(adapter, "_api_get", fake_api_get)

        await adapter.stop_typing("iMessage;-;user@example.com")

        assert info_calls == ["/api/v1/server/info"]
        assert len(deletes) == 1
        assert "/api/v1/chat/iMessage%3B-%3Buser%40example.com/typing" in deletes[0][0]
        assert deletes[0][1] == 5

    @pytest.mark.asyncio
    async def test_stop_typing_bounds_cold_helper_refresh(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter.client = object()
        adapter._private_api_enabled = False
        adapter._helper_connected = False
        refresh_cancelled = asyncio.Event()

        async def hanging_refresh():
            try:
                await asyncio.Event().wait()
            finally:
                refresh_cancelled.set()

        monkeypatch.setattr(adapter, "_refresh_helper_state", hanging_refresh)
        monkeypatch.setattr(
            "gateway.platforms.bluebubbles._STOP_TYPING_HELPER_REFRESH_SECONDS",
            0.01,
        )

        await asyncio.wait_for(adapter.stop_typing("unavailable-chat"), timeout=0.1)
        await asyncio.wait_for(adapter.stop_typing("unavailable-chat"), timeout=0.1)

        assert refresh_cancelled.is_set()

    @pytest.mark.asyncio
    async def test_cancelled_typing_start_is_reconciled_after_late_delivery(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch)
        adapter._private_api_enabled = True
        adapter._helper_connected = True
        post_started = asyncio.Event()
        release_post = asyncio.Event()
        events = []

        class FakeClient:
            async def post(self, url, timeout):
                post_started.set()
                try:
                    await release_post.wait()
                except asyncio.CancelledError:
                    # Model an HTTP stack where cancellation arrives after the
                    # request has already crossed the process boundary.
                    await release_post.wait()
                events.append("start")

            async def delete(self, url, timeout):
                events.append("stop")

        adapter.client = FakeClient()

        start_task = asyncio.create_task(
            adapter.send_typing("iMessage;-;user@example.com")
        )
        await post_started.wait()
        start_task.cancel()
        stop_task = asyncio.create_task(
            adapter.stop_typing("iMessage;-;user@example.com")
        )
        await asyncio.sleep(0)
        release_post.set()
        await asyncio.gather(start_task, stop_task)

        assert events == ["start", "stop"]

    @pytest.mark.asyncio
    async def test_repeated_stop_typing_emits_one_transition(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._private_api_enabled = True
        adapter._helper_connected = True
        events = []

        class FakeClient:
            async def post(self, url, timeout):
                events.append("start")

            async def delete(self, url, timeout):
                events.append("stop")

        adapter.client = FakeClient()

        await adapter.send_typing("iMessage;-;user@example.com")
        await adapter.stop_typing("iMessage;-;user@example.com")
        await adapter.stop_typing("iMessage;-;user@example.com")

        assert events == ["start", "stop"]

    @pytest.mark.asyncio
    async def test_background_cleanup_stops_active_typing_once(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._private_api_enabled = True
        adapter._helper_connected = True
        events = []

        class FakeClient:
            async def post(self, url, timeout):
                events.append("start")

            async def delete(self, url, timeout):
                events.append("stop")

        adapter.client = FakeClient()

        await adapter.send_typing("iMessage;-;user@example.com")
        await adapter.cancel_background_tasks()
        await adapter.cancel_background_tasks()
        await adapter.send_typing("iMessage;-;user@example.com")

        assert events == ["start", "stop"]
        assert adapter._typing_transition_locks == {}
        assert adapter._typing_pending_stops == set()
        assert adapter._typing_stop_tasks == {}

    @pytest.mark.asyncio
    async def test_pending_stop_retries_after_helper_cold_boot(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._private_api_enabled = True
        adapter._helper_connected = True
        events = []
        info_calls = 0

        class FakeClient:
            async def post(self, url, timeout):
                events.append("start")

            async def delete(self, url, timeout):
                events.append("stop")

        async def fake_api_get(path):
            nonlocal info_calls
            info_calls += 1
            helper_connected = info_calls > 1
            return {
                "data": {
                    "private_api": True,
                    "helper_connected": helper_connected,
                }
            }

        adapter.client = FakeClient()
        await adapter.send_typing("iMessage;-;user@example.com")
        adapter._helper_connected = False
        adapter._helper_last_refresh_at = 0.0
        monkeypatch.setattr(adapter, "_api_get", fake_api_get)
        monkeypatch.setattr(
            "gateway.platforms.bluebubbles._HELPER_NEGATIVE_REFRESH_TTL_SECONDS",
            0.01,
        )

        await adapter.stop_typing("iMessage;-;user@example.com")
        assert events == ["start"]
        assert len(adapter._typing_stop_tasks) == 1

        for _ in range(20):
            if events == ["start", "stop"]:
                break
            await asyncio.sleep(0.01)

        assert events == ["start", "stop"]
        assert info_calls >= 2
        assert adapter._typing_pending_stops == set()

    @pytest.mark.asyncio
    async def test_mark_read_refreshes_stale_helper_state(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._private_api_enabled = True
        adapter._helper_connected = False
        posts = []

        class FakeClient:
            async def post(self, url, timeout):
                posts.append((url, timeout))

        async def fake_api_get(path):
            assert path == "/api/v1/server/info"
            return {"data": {"private_api": True, "helper_connected": True}}

        adapter.client = FakeClient()
        monkeypatch.setattr(adapter, "_api_get", fake_api_get)

        result = await adapter.mark_read("iMessage;-;user@example.com")

        assert result is True
        assert len(posts) == 1
        assert "/api/v1/chat/iMessage%3B-%3Buser%40example.com/read" in posts[0][0]

    @pytest.mark.asyncio
    async def test_threaded_reply_refreshes_stale_helper_state(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._private_api_enabled = True
        adapter._helper_connected = False
        adapter.client = object()
        payloads = []

        async def fake_api_get(path):
            assert path == "/api/v1/server/info"
            return {"data": {"private_api": True, "helper_connected": True}}

        async def fake_api_post(path, payload):
            assert path == "/api/v1/message/text"
            payloads.append(payload)
            return {"data": {"guid": "threaded-reply-guid"}}

        monkeypatch.setattr(adapter, "_api_get", fake_api_get)
        monkeypatch.setattr(adapter, "_api_post", fake_api_post)

        result = await adapter.send(
            "iMessage;-;user@example.com",
            "native reply",
            reply_to="origin-message-guid",
        )

        assert result.success is True
        assert payloads[0]["method"] == "private-api"
        assert payloads[0]["selectedMessageGuid"] == "origin-message-guid"
        assert payloads[0]["partIndex"] == 0


class _FakeBlueBubblesRequest:
    def __init__(self, payload, password="secret"):
        self.query = {"password": password}
        self.headers = {}
        self._body = json.dumps(payload).encode("utf-8")

    async def read(self):
        return self._body


class TestBlueBubblesMentionGating:
    @pytest.mark.parametrize(
        ("raw_reference", "expected"),
        [
            ("message-guid", ("message-guid", 0)),
            ("p:2/message-guid", ("message-guid", 2)),
            ("bp:message-guid", ("message-guid", 0)),
            (" message-guid", (None, 0)),
            ("message-guid ", (None, 0)),
            ("p:2/ message-guid", (None, 0)),
            (7, (None, 0)),
        ],
    )
    def test_reply_target_preserves_valid_forms_and_rejects_malformed_ids(
        self, monkeypatch, raw_reference, expected
    ):
        adapter = _make_adapter(monkeypatch)
        assert adapter._reply_target(raw_reference) == expected

    @pytest.mark.asyncio
    async def test_exact_message_lookup_is_bounded_to_current_chat(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        calls = []
        records = [
            {
                "guid": f"earlier-{index}",
                "chats": [{"guid": "iMessage;+;exact-family-guid"}],
            }
            for index in range(99)
        ]
        records.append(
            {
                "guid": "target/message-guid",
                "chats": [{"guid": "iMessage;+;exact-family-guid"}],
            }
        )

        async def fail_api_get(path):
            raise AssertionError("referenced-message lookup must not use a global endpoint")

        async def fake_api_post(path, payload):
            calls.append((path, payload))
            return {"data": records[: payload["limit"]]}

        monkeypatch.setattr(adapter, "_api_get", fail_api_get)
        monkeypatch.setattr(adapter, "_api_post", fake_api_post)

        record = await adapter._lookup_referenced_message(
            "iMessage;+;exact-family-guid",
            "target/message-guid",
            include_attachments=True,
        )

        assert record is not None
        assert calls == [
            (
                "/api/v1/message/query",
                {
                    "limit": 100,
                    "offset": 0,
                    "chatGuid": "iMessage;+;exact-family-guid",
                    "with": ["attachments", "chats"],
                },
            )
        ]

    @pytest.mark.asyncio
    async def test_unauthorized_group_sender_is_dropped_before_background_processing(
        self, monkeypatch
    ):
        adapter = _make_adapter(
            monkeypatch,
            require_mention=True,
            mention_patterns=["ducky"],
            send_read_receipts=False,
        )
        handled = []
        reply_lookups = []

        async def capture_message(event):
            handled.append(event)

        async def capture_reply_lookup(chat_id, message_id):
            reply_lookups.append((chat_id, message_id))
            return None, False, [], []

        adapter.set_authorization_check(lambda user_id, chat_type, chat_id: False)
        monkeypatch.setattr(adapter, "handle_message", capture_message)
        monkeypatch.setattr(adapter, "_lookup_reply_context", capture_reply_lookup)

        response = await adapter._handle_webhook(
            _FakeBlueBubblesRequest(
                {
                    "type": "new-message",
                    "data": {
                        "guid": "unauthorized-message-guid",
                        "text": "ducky, inspect this",
                        "threadOriginatorGuid": "quoted-message-guid",
                        "handle": {"address": "+155****0102"},
                        "isFromMe": False,
                        "isGroup": True,
                        "chatGuid": "iMessage;+;exact-family-guid",
                        "chatIdentifier": "family-group",
                    },
                }
            )
        )

        assert response.status == 200
        assert handled == []
        assert reply_lookups == []

    @pytest.mark.asyncio
    async def test_cancelled_reply_lookup_releases_identity_for_retry(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        lookup_started = asyncio.Event()
        handled = []

        async def blocking_reply_lookup(chat_id, message_id):
            lookup_started.set()
            await asyncio.Event().wait()

        async def completed_reply_lookup(chat_id, message_id):
            return "quoted text", False, [], []

        async def capture_message(event):
            handled.append(event)

        monkeypatch.setattr(adapter, "_lookup_reply_context", blocking_reply_lookup)
        monkeypatch.setattr(adapter, "handle_message", capture_message)
        payload = {
            "type": "new-message",
            "data": {
                "guid": "cancelled-reply-lookup-guid",
                "text": "please retry me",
                "threadOriginatorGuid": "quoted-message-guid",
                "handle": {"address": "user@example.com"},
                "isFromMe": False,
                "chatGuid": "iMessage;-;user@example.com",
                "chatIdentifier": "user@example.com",
            },
        }

        first = asyncio.create_task(
            adapter._handle_webhook(_FakeBlueBubblesRequest(payload))
        )
        await lookup_started.wait()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first

        assert adapter._pending_message_identities == set()

        monkeypatch.setattr(adapter, "_lookup_reply_context", completed_reply_lookup)
        retry = await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))
        await asyncio.sleep(0)

        assert retry.status == 200
        assert [event.text for event in handled] == ["please retry me"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "reply_field", ["threadOriginatorGuid", "associatedMessageGuid"]
    )
    async def test_reply_context_lookup_is_bounded_exact_chat_and_stages_attachments(
        self, monkeypatch, reply_field
    ):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        handled = []
        queries = []

        async def fake_handle_message(event):
            handled.append(event)

        async def fake_api_post(path, payload):
            queries.append((path, payload))
            return {
                "data": [
                    {
                        "guid": "replied-message-guid",
                        "text": "the exact earlier message",
                        "isFromMe": True,
                        "chats": [{"guid": "iMessage;+;exact-family-guid"}],
                        "attachments": [
                            {
                                "guid": "replied-photo-guid",
                                "mimeType": "image/png",
                                "transferName": "photo.png",
                            },
                            {
                                "guid": "replied-document-guid",
                                "mimeType": "application/pdf",
                                "transferName": "details.pdf",
                            },
                        ],
                    }
                ]
            }

        async def fake_download_attachment(att_guid, att_meta):
            return {
                "replied-photo-guid": "/tmp/exact-replied-photo.png",
                "replied-document-guid": "/tmp/exact-details.pdf",
            }[att_guid]

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        monkeypatch.setattr(adapter, "_api_post", fake_api_post)
        monkeypatch.setattr(adapter, "_download_attachment", fake_download_attachment)

        response = await adapter._handle_webhook(
            _FakeBlueBubblesRequest(
                {
                    "type": "new-message",
                    "data": {
                        "guid": "new-message-guid",
                        "text": "what is in this?",
                        reply_field: "replied-message-guid",
                        "handle": {"address": "+15555550100"},
                        "isFromMe": False,
                        "isGroup": True,
                        "chatGuid": "iMessage;+;exact-family-guid",
                        "chatIdentifier": "family-group",
                    },
                }
            )
        )
        await asyncio.sleep(0)

        assert response.status == 200
        assert len(handled) == 1
        event = handled[0]
        assert event.reply_to_message_id == "replied-message-guid"
        assert event.reply_to_text == "the exact earlier message"
        assert event.reply_to_is_own_message is True
        assert event.media_urls == [
            "/tmp/exact-replied-photo.png",
            "/tmp/exact-details.pdf",
        ]
        assert event.media_types == ["image/png", "application/pdf"]
        assert event.metadata["bluebubbles_reply_attachment_count"] == 2
        assert queries == [
            (
                "/api/v1/message/query",
                {
                    "limit": 100,
                    "offset": 0,
                    "chatGuid": "iMessage;+;exact-family-guid",
                    "with": ["attachments", "chats"],
                },
            )
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", ["missing", "ambiguous", "cross-chat", "over-bound"])
    async def test_referenced_message_lookup_rejects_unresolved_results(
        self, monkeypatch, mode
    ):
        adapter = _make_adapter(monkeypatch)
        target = {
            "guid": "target-guid",
            "text": "private text",
            "chats": [{"guid": "iMessage;+;current-chat"}],
        }
        records = {
            "missing": [],
            "ambiguous": [target, dict(target)],
            "cross-chat": [
                {
                    **target,
                    "chats": [{"guid": "iMessage;+;other-chat"}],
                }
            ],
            "over-bound": [target] + [
                {
                    "guid": f"filler-{index}",
                    "chats": [{"guid": "iMessage;+;current-chat"}],
                }
                for index in range(100)
            ],
        }[mode]

        async def fake_api_post(path, payload):
            return {"data": records}

        monkeypatch.setattr(adapter, "_api_post", fake_api_post)

        assert (
            await adapter._lookup_referenced_message(
                "iMessage;+;current-chat", "target-guid"
            )
            is None
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("chat_guid", "message_guid"),
        [("", "target-guid"), (" current-chat", "target-guid"), ("current-chat", 7)],
    )
    async def test_referenced_message_lookup_rejects_malformed_input_without_io(
        self, monkeypatch, chat_guid, message_guid
    ):
        adapter = _make_adapter(monkeypatch)

        async def fail_api_post(path, payload):
            raise AssertionError("malformed references must fail before I/O")

        monkeypatch.setattr(adapter, "_api_post", fail_api_post)

        assert await adapter._lookup_referenced_message(chat_guid, message_guid) is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", ["foreign", "missing", "error"])
    async def test_reply_context_missing_or_cross_chat_fails_closed(
        self, monkeypatch, mode
    ):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        handled = []
        downloads = []

        async def fake_handle_message(event):
            handled.append(event)

        async def fake_api_post(path, payload):
            if mode == "error":
                raise RuntimeError("query failed")
            if mode == "missing":
                return {"data": []}
            return {
                "data": [
                    {
                        "guid": "replied-message-guid",
                        "text": "foreign private context",
                        "chats": [{"guid": "iMessage;+;other-group"}],
                        "attachments": [{"guid": "foreign-attachment"}],
                    }
                ]
            }

        async def fake_download_attachment(att_guid, att_meta):
            downloads.append(att_guid)
            return "/tmp/should-not-stage"

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        monkeypatch.setattr(adapter, "_api_post", fake_api_post)
        monkeypatch.setattr(adapter, "_download_attachment", fake_download_attachment)

        await adapter._handle_webhook(
            _FakeBlueBubblesRequest(
                {
                    "type": "new-message",
                    "data": {
                        "guid": "new-message-guid",
                        "text": "ordinary reply",
                        "associatedMessageGuid": "replied-message-guid",
                        "handle": {"address": "user@example.com"},
                        "isFromMe": False,
                        "chatGuid": "iMessage;-;user@example.com",
                        "chatIdentifier": "user@example.com",
                    },
                }
            )
        )
        await asyncio.sleep(0)

        assert len(handled) == 1
        assert handled[0].reply_to_message_id == "replied-message-guid"
        assert handled[0].reply_to_text is None
        assert handled[0].media_urls == []
        assert downloads == []

    @pytest.mark.asyncio
    async def test_malformed_reply_reference_exposes_no_attachment_context(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        handled = []
        downloads = []

        async def fake_api_post(path, payload):
            return {
                "data": [
                    {
                        "guid": "replied-message-guid",
                        "text": "private quoted caption",
                        "isFromMe": True,
                        "chats": [{"guid": "iMessage;-;user@example.com"}],
                        "attachments": [
                            {
                                "guid": "private-attachment-guid",
                                "mimeType": "image/png",
                                "transferName": "private-photo.png",
                            }
                        ],
                    }
                ]
            }

        async def fake_download_attachment(att_guid, att_meta):
            downloads.append((att_guid, att_meta))
            return "/tmp/private-photo.png"

        async def capture_message(event):
            handled.append(event)

        monkeypatch.setattr(adapter, "handle_message", capture_message)
        monkeypatch.setattr(adapter, "_api_post", fake_api_post)
        monkeypatch.setattr(adapter, "_download_attachment", fake_download_attachment)

        await adapter._handle_webhook(
            _FakeBlueBubblesRequest(
                {
                    "type": "new-message",
                    "data": {
                        "guid": "new-message-guid",
                        "text": "ordinary reply",
                        "associatedMessageGuid": " replied-message-guid",
                        "handle": {"address": "user@example.com"},
                        "isFromMe": False,
                        "chatGuid": "iMessage;-;user@example.com",
                        "chatIdentifier": "user@example.com",
                    },
                }
            )
        )
        await asyncio.sleep(0)

        assert len(handled) == 1
        assert handled[0].reply_to_text is None
        assert handled[0].reply_to_is_own_message is False
        assert handled[0].media_urls == []
        assert handled[0].media_types == []
        assert handled[0].metadata == {}
        assert downloads == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("record_count", "expected_quote"),
        [(100, "boundary quote"), (101, None)],
    )
    async def test_reply_hydration_enforces_lookup_boundary_without_partial_leakage(
        self, monkeypatch, record_count, expected_quote
    ):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        handled = []
        downloads = []
        records = [
            {
                "guid": f"filler-{index}",
                "chats": [{"guid": "iMessage;+;family-group"}],
            }
            for index in range(record_count - 1)
        ]
        records.append(
            {
                "guid": "boundary-target",
                "text": "boundary quote",
                "isFromMe": True,
                "chats": [{"guid": "iMessage;+;family-group"}],
                "attachments": [
                    {
                        "guid": "boundary-attachment",
                        "mimeType": "image/png",
                        "transferName": "private.png",
                    }
                ],
            }
        )

        async def fake_api_post(path, payload):
            assert payload["limit"] == 100
            return {"data": records}

        async def fake_download_attachment(att_guid, att_meta):
            downloads.append((att_guid, att_meta))
            return "/tmp/boundary-attachment.png"

        monkeypatch.setattr(adapter, "_api_post", fake_api_post)
        monkeypatch.setattr(adapter, "_download_attachment", fake_download_attachment)
        monkeypatch.setattr(adapter, "handle_message", handled.append)

        await adapter._handle_webhook(
            _FakeBlueBubblesRequest(
                {
                    "type": "new-message",
                    "data": {
                        "guid": f"boundary-inbound-{record_count}",
                        "text": "keep the inbound text",
                        "threadOriginatorGuid": "boundary-target",
                        "handle": {"address": "+155****0100"},
                        "isFromMe": False,
                        "isGroup": True,
                        "chatGuid": "iMessage;+;family-group",
                        "chatIdentifier": "family-group",
                    },
                }
            )
        )
        await asyncio.sleep(0)

        assert len(handled) == 1
        event = handled[0]
        assert event.text == "keep the inbound text"
        assert event.reply_to_message_id == "boundary-target"
        assert event.reply_to_text == expected_quote
        if expected_quote is not None:
            assert event.reply_to_is_own_message is True
            assert event.media_urls == ["/tmp/boundary-attachment.png"]
            assert event.media_types == ["image/png"]
            assert event.metadata == {"bluebubbles_reply_attachment_count": 1}
            assert len(downloads) == 1
        else:
            assert event.reply_to_is_own_message is False
            assert event.media_urls == []
            assert event.media_types == []
            assert event.metadata == {}
            assert downloads == []

    @pytest.mark.asyncio
    async def test_colliding_cross_chat_reply_identifier_fails_closed(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        handled = []
        downloads = []

        async def fake_api_post(path, payload):
            return {
                "data": [
                    {
                        "guid": "colliding-guid",
                        "text": "same-chat text must not partially win",
                        "isFromMe": True,
                        "chats": [{"guid": "iMessage;+;family-group"}],
                        "attachments": [{"guid": "same-chat-private-file"}],
                    },
                    {
                        "guid": "colliding-guid",
                        "text": "foreign private text",
                        "isFromMe": True,
                        "chats": [{"guid": "iMessage;+;other-group"}],
                        "attachments": [{"guid": "foreign-private-file"}],
                    },
                ]
            }

        async def fake_download_attachment(att_guid, att_meta):
            downloads.append((att_guid, att_meta))
            return "/tmp/must-not-leak"

        monkeypatch.setattr(adapter, "_api_post", fake_api_post)
        monkeypatch.setattr(adapter, "_download_attachment", fake_download_attachment)
        monkeypatch.setattr(adapter, "handle_message", handled.append)

        await adapter._handle_webhook(
            _FakeBlueBubblesRequest(
                {
                    "type": "new-message",
                    "data": {
                        "guid": "collision-inbound",
                        "text": "ordinary reply survives",
                        "associatedMessageGuid": "colliding-guid",
                        "handle": {"address": "+155****0100"},
                        "isFromMe": False,
                        "isGroup": True,
                        "chatGuid": "iMessage;+;family-group",
                        "chatIdentifier": "family-group",
                    },
                }
            )
        )
        await asyncio.sleep(0)

        assert len(handled) == 1
        event = handled[0]
        assert event.text == "ordinary reply survives"
        assert event.reply_to_message_id == "colliding-guid"
        assert event.reply_to_text is None
        assert event.reply_to_is_own_message is False
        assert event.media_urls == []
        assert event.media_types == []
        assert event.metadata == {}
        assert downloads == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("reference", "expected_reply_id", "expected_queries"),
        [(None, None, 0), ("", None, 0), (" malformed", None, 0), (7, None, 0), ("missing-guid", "missing-guid", 1)],
    )
    async def test_unquoted_missing_and_malformed_references_leave_message_unchanged(
        self, monkeypatch, reference, expected_reply_id, expected_queries
    ):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        handled = []
        queries = []

        async def fake_api_post(path, payload):
            queries.append((path, payload))
            return {"data": []}

        monkeypatch.setattr(adapter, "_api_post", fake_api_post)
        monkeypatch.setattr(adapter, "handle_message", handled.append)
        data = {
            "guid": f"ordinary-{reference!r}",
            "text": "ordinary message text",
            "handle": {"address": "user@example.com"},
            "isFromMe": False,
            "chatGuid": "iMessage;-;user@example.com",
            "chatIdentifier": "user@example.com",
        }
        if reference is not None:
            data["threadOriginatorGuid"] = reference

        await adapter._handle_webhook(
            _FakeBlueBubblesRequest({"type": "new-message", "data": data})
        )
        await asyncio.sleep(0)

        assert len(handled) == 1
        event = handled[0]
        assert event.text == "ordinary message text"
        assert event.reply_to_message_id == expected_reply_id
        assert event.reply_to_text is None
        assert event.reply_to_is_own_message is False
        assert event.media_urls == []
        assert event.media_types == []
        assert event.metadata == {}
        assert len(queries) == expected_queries

    @pytest.mark.asyncio
    async def test_participant_names_preserve_canonical_identity_across_dm_and_group(self, monkeypatch):
        known = "+15555550100"
        unknown = "+15555550101"
        adapter = _make_adapter(
            monkeypatch,
            participant_names={known: "Mark", unknown: "", 7: "ignored"},
            send_read_receipts=False,
        )
        handled = []

        async def fake_handle_message(event):
            handled.append(event)

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)

        async def dispatch(guid, sender, *, group=False):
            data = {
                "guid": guid,
                "text": "hello",
                "handle": {"address": sender},
                "isFromMe": False,
                "chatGuid": (
                    "iMessage;+;family-group"
                    if group
                    else f"iMessage;-;{sender}"
                ),
                "chatIdentifier": "family-group" if group else sender,
                "isGroup": group,
            }
            response = await adapter._handle_webhook(
                _FakeBlueBubblesRequest({"type": "new-message", "data": data})
            )
            await asyncio.sleep(0)
            assert response.status == 200

        await dispatch("participant-known-dm", known)
        await dispatch("participant-known-group", known, group=True)
        await dispatch("participant-unknown", unknown)

        malformed = _make_adapter(
            monkeypatch,
            participant_names=[{known: "not-a-map"}],
            send_read_receipts=False,
        )
        monkeypatch.setattr(malformed, "handle_message", fake_handle_message)
        response = await malformed._handle_webhook(
            _FakeBlueBubblesRequest({
                "type": "new-message",
                "data": {
                    "guid": "participant-malformed",
                    "text": "hello",
                    "handle": {"address": known},
                    "isFromMe": False,
                    "chatGuid": f"iMessage;-;{known}",
                    "chatIdentifier": known,
                },
            })
        )
        await asyncio.sleep(0)
        assert response.status == 200

        assert [
            (event.source.user_id, event.source.user_name, event.source.chat_type)
            for event in handled
        ] == [
            (known, "Mark", "dm"),
            (known, "Mark", "group"),
            (unknown, unknown, "dm"),
            (known, known, "dm"),
        ]

    @pytest.mark.asyncio
    async def test_group_message_without_mention_is_acknowledged_and_skipped(self, monkeypatch):
        adapter = _make_adapter(
            monkeypatch,
            require_mention=True,
            send_read_receipts=False,
        )
        handled = []
        reactions = []

        async def fake_handle_message(event):
            handled.append(event)

        async def fake_send_reaction(*args, **kwargs):
            reactions.append((args, kwargs))

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        monkeypatch.setattr(adapter, "send_reaction", fake_send_reaction)
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
        assert reactions == []

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

    @pytest.mark.asyncio
    async def test_duplicate_new_and_updated_events_are_handled_once(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        handled = []

        async def fake_handle_message(event):
            handled.append(event)

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        message = {
            "guid": "same-message-guid",
            "text": "hello once",
            "handle": {"address": "user@example.com"},
            "isFromMe": False,
            "chatIdentifier": "user@example.com",
        }

        first, second = await asyncio.gather(
            adapter._handle_webhook(_FakeBlueBubblesRequest({
                "type": "new-message",
                "data": {**message, "chatGuid": "iMessage;-;user@example.com"},
            })),
            adapter._handle_webhook(_FakeBlueBubblesRequest({
                "type": "updated-message",
                "data": message,
            })),
        )
        await asyncio.sleep(0)

        assert first.status == 200
        assert second.status == 200
        assert [event.text for event in handled] == ["hello once"]

    @pytest.mark.asyncio
    async def test_incomplete_first_group_second_dispatches_one_group_turn(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        handled = []

        async def fake_handle_message(event):
            handled.append(event)

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        base = {
            "guid": "ambiguous-then-group-guid",
            "text": "family request",
            "handle": {"address": "+15555550100"},
            "isFromMe": False,
            "chatIdentifier": "+15555550100",
        }

        first = await adapter._handle_webhook(
            _FakeBlueBubblesRequest({"type": "new-message", "data": base})
        )
        await asyncio.sleep(0)
        second = await adapter._handle_webhook(
            _FakeBlueBubblesRequest(
                {
                    "type": "updated-message",
                    "data": {
                        **base,
                        "chatGuid": "iMessage;+;family-group",
                        "chatIdentifier": "family-group",
                        "isGroup": True,
                    },
                }
            )
        )
        await asyncio.sleep(0)

        assert first.status == 200
        assert second.status == 200
        assert [event.source.chat_id for event in handled] == [
            "iMessage;+;family-group"
        ]
        assert [event.source.chat_type for event in handled] == ["group"]

    @pytest.mark.asyncio
    async def test_authoritative_revision_wins_while_ambiguous_lookup_is_in_flight(
        self, monkeypatch
    ):
        import gateway.platforms.bluebubbles as bluebubbles_module

        monkeypatch.setattr(
            bluebubbles_module,
            "_PROVISIONAL_MESSAGE_WAIT_SECONDS",
            0.01,
        )
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        lookup_started = asyncio.Event()
        release_lookup = asyncio.Event()
        handled = []
        lookup_count = 0

        async def delayed_lookup(message_guid):
            nonlocal lookup_count
            lookup_count += 1
            if lookup_count == 1:
                lookup_started.set()
                await release_lookup.wait()
                return None
            return "iMessage;+;family-group"

        async def fake_handle_message(event):
            handled.append(event)

        monkeypatch.setattr(
            adapter,
            "_resolve_exact_message_chat_guid",
            delayed_lookup,
        )
        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        first = asyncio.create_task(
            adapter._handle_webhook(
                _FakeBlueBubblesRequest(
                    {
                        "type": "new-message",
                        "data": {
                            "guid": "in-flight-lookup-guid",
                            "text": "ambiguous text",
                            "handle": {"address": "+15555550100"},
                            "isFromMe": False,
                            "chatIdentifier": "+15555550100",
                        },
                    }
                )
            )
        )
        await lookup_started.wait()
        await adapter._handle_webhook(
            _FakeBlueBubblesRequest(
                {
                    "type": "updated-message",
                    "data": {
                        "guid": "in-flight-lookup-guid",
                        "text": "authoritative text",
                        "handle": {"address": "+15555550100"},
                        "isFromMe": False,
                        "chatGuid": "iMessage;+;family-group",
                        "chatIdentifier": "family-group",
                        "isGroup": True,
                    },
                }
            )
        )
        await asyncio.sleep(0)
        release_lookup.set()
        await first
        await asyncio.sleep(0.03)

        assert [
            (event.source.chat_id, event.text) for event in handled
        ] == [("iMessage;+;family-group", "authoritative text")]
        assert lookup_count == 1

    @pytest.mark.asyncio
    async def test_authoritative_revision_merges_provisional_message_text(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        handled = []

        async def fake_handle_message(event):
            handled.append(event)

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        first = {
            "guid": "provisional-text-guid",
            "text": "text only present in first revision",
            "handle": {"address": "+15555550100"},
            "isFromMe": False,
            "chatIdentifier": "+15555550100",
        }

        await adapter._handle_webhook(
            _FakeBlueBubblesRequest({"type": "new-message", "data": first})
        )
        await adapter._handle_webhook(
            _FakeBlueBubblesRequest(
                {
                    "type": "updated-message",
                    "data": {
                        **first,
                        "text": "",
                        "chatGuid": "iMessage;+;family-group",
                        "chatIdentifier": "family-group",
                        "isGroup": True,
                    },
                }
            )
        )
        await asyncio.sleep(0)

        assert [
            (event.source.chat_id, event.text) for event in handled
        ] == [
            (
                "iMessage;+;family-group",
                "text only present in first revision",
            )
        ]

    @pytest.mark.asyncio
    async def test_sparse_authoritative_revision_does_not_consume_provisional(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        handled = []
        lookup_count = 0

        async def resolve_on_retry(message_guid):
            nonlocal lookup_count
            lookup_count += 1
            return None

        async def exact_membership(chat_guid, message_guid, **kwargs):
            return {
                "guid": message_guid,
                "chats": [{"guid": chat_guid}],
            }

        async def fake_handle_message(event):
            handled.append(event)

        monkeypatch.setattr(
            adapter,
            "_resolve_exact_message_chat_guid",
            resolve_on_retry,
        )
        monkeypatch.setattr(adapter, "_query_exact_message", exact_membership)
        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        await adapter._handle_webhook(
            _FakeBlueBubblesRequest(
                {
                    "type": "new-message",
                    "data": {
                        "guid": "sparse-authoritative-guid",
                        "text": "ambiguous text",
                        "handle": {"address": "+15555550100"},
                        "isFromMe": False,
                        "chatIdentifier": "+15555550100",
                    },
                }
            )
        )
        await adapter._handle_webhook(
            _FakeBlueBubblesRequest(
                {
                    "type": "updated-message",
                    "data": {
                        "guid": "sparse-authoritative-guid",
                        "text": "authoritative text",
                        "isFromMe": False,
                        "chatGuid": "iMessage;+;family-group",
                        "chatIdentifier": "family-group",
                        "isGroup": True,
                    },
                }
            )
        )
        await asyncio.sleep(0)

        assert lookup_count == 1
        assert handled == []

    @pytest.mark.asyncio
    async def test_forged_chat_guid_is_rejected_against_rest_membership(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        membership_lookups = []
        authorization_calls = []
        handled = []

        async def exact_membership(chat_guid, message_guid, **kwargs):
            membership_lookups.append((chat_guid, message_guid))
            return None

        async def fake_handle_message(event):
            handled.append(event)

        monkeypatch.delattr(adapter, "_verify_inbound_message_membership")
        monkeypatch.setattr(adapter, "_query_exact_message", exact_membership)
        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        adapter.set_authorization_check(
            lambda *args: authorization_calls.append(args) or True
        )

        response = await adapter._handle_webhook(
            _FakeBlueBubblesRequest(
                {
                    "type": "new-message",
                    "data": {
                        "guid": "forged-chat-guid-message",
                        "text": "do not route to forged chat",
                        "handle": {"address": "user@example.com"},
                        "isFromMe": False,
                        "chatGuid": "iMessage;+;forged-chat",
                        "chatIdentifier": "forged-chat",
                        "isGroup": True,
                    },
                }
            )
        )

        assert response.status == 200
        assert membership_lookups == [
            ("iMessage;+;forged-chat", "forged-chat-guid-message")
        ]
        assert authorization_calls == []
        assert handled == []

    @pytest.mark.asyncio
    async def test_conflicting_scalar_chat_guids_fail_before_membership_lookup(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        membership_lookups = []
        handled = []

        async def membership(chat_guid, message_guid, **kwargs):
            membership_lookups.append((chat_guid, message_guid))
            return chat_guid

        monkeypatch.delattr(adapter, "_verify_inbound_message_membership")
        monkeypatch.setattr(adapter, "_query_exact_message", membership)
        monkeypatch.setattr(adapter, "handle_message", handled.append)

        response = await adapter._handle_webhook(
            _FakeBlueBubblesRequest(
                {
                    "type": "new-message",
                    "chatGuid": "iMessage;+;top-level-chat",
                    "data": {
                        "guid": "conflicting-chat-fields-guid",
                        "text": "do not choose either chat",
                        "handle": {"address": "user@example.com"},
                        "isFromMe": False,
                        "chatGuid": "iMessage;+;record-chat",
                        "isGroup": True,
                    },
                }
            )
        )

        assert response.status == 200
        assert membership_lookups == []
        assert handled == []

    @pytest.mark.asyncio
    async def test_conflicting_nested_chat_guids_fail_before_membership_lookup(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        membership_lookups = []
        handled = []

        async def membership(chat_guid, message_guid, **kwargs):
            membership_lookups.append((chat_guid, message_guid))
            return chat_guid

        monkeypatch.delattr(adapter, "_verify_inbound_message_membership")
        monkeypatch.setattr(adapter, "_query_exact_message", membership)
        monkeypatch.setattr(adapter, "handle_message", handled.append)

        response = await adapter._handle_webhook(
            _FakeBlueBubblesRequest(
                {
                    "type": "new-message",
                    "data": {
                        "guid": "conflicting-nested-chat-guid",
                        "text": "do not choose nested chat",
                        "handle": {"address": "user@example.com"},
                        "isFromMe": False,
                        "chats": [
                            {
                                "guid": "iMessage;+;nested-a",
                                "chatGuid": "iMessage;+;nested-b",
                            }
                        ],
                        "isGroup": True,
                    },
                }
            )
        )

        assert response.status == 200
        assert membership_lookups == []
        assert handled == []

    @pytest.mark.asyncio
    async def test_transient_candidate_membership_lookup_has_no_side_effects(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch, send_read_receipts=True)
        side_effects = []

        async def unavailable_membership(chat_guid, message_guid, **kwargs):
            raise TimeoutError("BlueBubbles lookup unavailable")

        async def capture(name, *args, **kwargs):
            side_effects.append((name, args, kwargs))

        monkeypatch.delattr(adapter, "_verify_inbound_message_membership")
        monkeypatch.setattr(adapter, "_query_exact_message", unavailable_membership)
        monkeypatch.setattr(
            adapter,
            "_download_attachment",
            lambda *args, **kwargs: capture("download", *args, **kwargs),
        )
        monkeypatch.setattr(
            adapter,
            "handle_message",
            lambda *args, **kwargs: capture("handle", *args, **kwargs),
        )
        monkeypatch.setattr(
            adapter,
            "mark_read",
            lambda *args, **kwargs: capture("read", *args, **kwargs),
        )
        adapter.set_authorization_check(
            lambda *args: side_effects.append(("authorize", args, {})) or True
        )

        response = await adapter._handle_webhook(
            _FakeBlueBubblesRequest(
                {
                    "type": "new-message",
                    "data": {
                        "guid": "transient-membership-guid",
                        "text": "hold until verified",
                        "handle": {"address": "user@example.com"},
                        "isFromMe": False,
                        "chatGuid": "iMessage;-;user@example.com",
                        "chatIdentifier": "user@example.com",
                        "attachments": [
                            {"guid": "private-photo", "mimeType": "image/png"}
                        ],
                    },
                }
            )
        )

        assert response.status == 200
        assert side_effects == []

    @pytest.mark.asyncio
    async def test_provisional_chat_a_cannot_leak_into_authoritative_chat_b(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        handled = []
        downloads = []

        async def exact_membership(chat_guid, message_guid, **kwargs):
            if chat_guid == "iMessage;+;chat-a":
                return None
            return {
                "guid": message_guid,
                "chats": [{"guid": "iMessage;+;chat-b"}],
            }

        async def fake_download(att_guid, att_meta):
            downloads.append(att_guid)
            return f"/tmp/{att_guid}"

        async def fake_handle_message(event):
            handled.append(event)

        monkeypatch.delattr(adapter, "_verify_inbound_message_membership")
        monkeypatch.setattr(adapter, "_query_exact_message", exact_membership)
        monkeypatch.setattr(adapter, "_download_attachment", fake_download)
        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        message_guid = "cross-chat-provisional-guid"
        sender = {"address": "same-sender@example.com"}

        await adapter._handle_webhook(
            _FakeBlueBubblesRequest(
                {
                    "type": "new-message",
                    "data": {
                        "guid": message_guid,
                        "text": "chat A secret",
                        "handle": sender,
                        "isFromMe": False,
                        "chatGuid": "iMessage;+;chat-a",
                        "chatIdentifier": "chat-a",
                        "isGroup": True,
                        "attachments": [
                            {"guid": "chat-a-private", "mimeType": "image/png"}
                        ],
                    },
                }
            )
        )
        await adapter._handle_webhook(
            _FakeBlueBubblesRequest(
                {
                    "type": "updated-message",
                    "data": {
                        "guid": message_guid,
                        "text": "chat B public",
                        "handle": sender,
                        "isFromMe": False,
                        "chatGuid": "iMessage;+;chat-b",
                        "chatIdentifier": "chat-b",
                        "isGroup": True,
                    },
                }
            )
        )
        await asyncio.sleep(0)

        assert downloads == []
        assert [(event.source.chat_id, event.text) for event in handled] == [
            ("iMessage;+;chat-b", "chat B public")
        ]

    @pytest.mark.asyncio
    async def test_authoritative_revision_merges_provisional_attachments_after_resolution(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        handled = []
        downloads = []

        async def fake_handle_message(event):
            handled.append(event)

        async def fake_download_attachment(att_guid, att_meta):
            downloads.append(att_guid)
            return f"/tmp/{att_guid}"

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        monkeypatch.setattr(adapter, "_download_attachment", fake_download_attachment)
        first = {
            "guid": "provisional-attachments-guid",
            "text": "inspect both",
            "handle": {"address": "+15555550100"},
            "isFromMe": False,
            "chatIdentifier": "+15555550100",
            "attachments": [
                {"guid": "first-photo", "mimeType": "image/png"},
            ],
        }

        await adapter._handle_webhook(
            _FakeBlueBubblesRequest({"type": "new-message", "data": first})
        )
        assert downloads == []
        await adapter._handle_webhook(
            _FakeBlueBubblesRequest(
                {
                    "type": "updated-message",
                    "data": {
                        **first,
                        "chatGuid": "iMessage;+;family-group",
                        "chatIdentifier": "family-group",
                        "isGroup": True,
                        "attachments": [
                            {"guid": "second-photo", "mimeType": "image/png"},
                        ],
                    },
                }
            )
        )
        await asyncio.sleep(0)

        assert downloads == ["first-photo", "second-photo"]
        assert [event.media_urls for event in handled] == [
            ["/tmp/first-photo", "/tmp/second-photo"]
        ]

    @pytest.mark.asyncio
    async def test_provisional_retry_resolves_authoritative_true_dm(
        self, monkeypatch
    ):
        import gateway.platforms.bluebubbles as bluebubbles_module

        monkeypatch.setattr(
            bluebubbles_module,
            "_PROVISIONAL_MESSAGE_WAIT_SECONDS",
            0.01,
        )
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        handled = []
        lookups = []

        async def resolve_on_retry(message_guid):
            lookups.append(message_guid)
            if len(lookups) == 1:
                return None
            return "iMessage;-;user@example.com"

        async def fake_handle_message(event):
            handled.append(event)

        monkeypatch.setattr(
            adapter,
            "_resolve_exact_message_chat_guid",
            resolve_on_retry,
        )
        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)

        response = await adapter._handle_webhook(
            _FakeBlueBubblesRequest(
                {
                    "type": "new-message",
                    "data": {
                        "guid": "true-dm-provisional-guid",
                        "text": "private request",
                        "handle": {"address": "user@example.com"},
                        "isFromMe": False,
                        "chatIdentifier": "user@example.com",
                    },
                }
            )
        )
        await asyncio.sleep(0.03)

        assert response.status == 200
        assert lookups == [
            "true-dm-provisional-guid",
            "true-dm-provisional-guid",
        ]
        assert [
            (event.source.chat_id, event.source.chat_type, event.text)
            for event in handled
        ] == [
            (
                "iMessage;-;user@example.com",
                "dm",
                "private request",
            )
        ]

    @pytest.mark.asyncio
    async def test_newer_provisional_revision_wins_while_retry_lookup_is_in_flight(
        self, monkeypatch
    ):
        import gateway.platforms.bluebubbles as bluebubbles_module

        monkeypatch.setattr(
            bluebubbles_module,
            "_PROVISIONAL_MESSAGE_WAIT_SECONDS",
            0.01,
        )
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        retry_started = asyncio.Event()
        release_retry = asyncio.Event()
        handled = []
        lookup_count = 0

        async def resolve_during_retry(message_guid):
            nonlocal lookup_count
            lookup_count += 1
            if lookup_count == 1:
                return None
            if lookup_count == 2:
                retry_started.set()
                await release_retry.wait()
                return "iMessage;-;user@example.com"
            return None

        async def fake_handle_message(event):
            handled.append(event)

        monkeypatch.setattr(
            adapter,
            "_resolve_exact_message_chat_guid",
            resolve_during_retry,
        )
        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        base = {
            "guid": "edited-during-retry-guid",
            "handle": {"address": "user@example.com"},
            "isFromMe": False,
            "chatIdentifier": "user@example.com",
        }

        await adapter._handle_webhook(
            _FakeBlueBubblesRequest(
                {"type": "new-message", "data": {**base, "text": "old text"}}
            )
        )
        await retry_started.wait()
        await adapter._handle_webhook(
            _FakeBlueBubblesRequest(
                {
                    "type": "updated-message",
                    "data": {**base, "text": "new text"},
                }
            )
        )
        release_retry.set()
        await asyncio.sleep(0.03)

        assert lookup_count == 3
        assert [event.text for event in handled] == ["new text"]

    @pytest.mark.asyncio
    async def test_unresolved_message_has_no_delivery_side_effects(self, monkeypatch):
        import gateway.platforms.bluebubbles as bluebubbles_module

        monkeypatch.setattr(
            bluebubbles_module,
            "_PROVISIONAL_MESSAGE_WAIT_SECONDS",
            0.01,
        )
        adapter = _make_adapter(
            monkeypatch,
            ack_reaction="like",
            send_read_receipts=True,
        )
        side_effects = []
        lookups = []

        async def unresolved_lookup(message_guid):
            lookups.append(message_guid)
            return None

        async def capture(name, *args, **kwargs):
            side_effects.append((name, args, kwargs))

        monkeypatch.setattr(
            adapter,
            "_resolve_exact_message_chat_guid",
            unresolved_lookup,
        )
        monkeypatch.setattr(
            adapter,
            "_download_attachment",
            lambda *args, **kwargs: capture("download", *args, **kwargs),
        )
        monkeypatch.setattr(
            adapter,
            "handle_message",
            lambda *args, **kwargs: capture("handle", *args, **kwargs),
        )
        monkeypatch.setattr(
            adapter,
            "mark_read",
            lambda *args, **kwargs: capture("read", *args, **kwargs),
        )
        monkeypatch.setattr(
            adapter,
            "send_typing",
            lambda *args, **kwargs: capture("typing", *args, **kwargs),
        )
        monkeypatch.setattr(
            adapter,
            "send_reaction",
            lambda *args, **kwargs: capture("reaction", *args, **kwargs),
        )
        adapter.set_authorization_check(
            lambda *args: side_effects.append(("authorize", args, {})) or True
        )

        response = await adapter._handle_webhook(
            _FakeBlueBubblesRequest(
                {
                    "type": "new-message",
                    "data": {
                        "guid": "never-resolved-guid",
                        "text": "do not leak",
                        "handle": {"address": "+15555550100"},
                        "isFromMe": False,
                        "chatIdentifier": "+15555550100",
                        "attachments": [
                            {"guid": "private-photo", "mimeType": "image/png"},
                        ],
                    },
                }
            )
        )
        await asyncio.sleep(0.03)

        assert response.status == 200
        assert lookups == ["never-resolved-guid", "never-resolved-guid"]
        assert side_effects == []
        assert adapter._provisional_messages == {}
        assert adapter._provisional_message_tasks == {}
        assert adapter._message_revision_serials == {}

    @pytest.mark.asyncio
    async def test_multi_chat_webhook_membership_is_not_treated_as_authoritative(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        handled = []
        authorization_calls = []

        async def unresolved_lookup(message_guid):
            return None

        async def fake_handle_message(event):
            handled.append(event)

        monkeypatch.setattr(
            adapter,
            "_resolve_exact_message_chat_guid",
            unresolved_lookup,
        )
        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        adapter.set_authorization_check(
            lambda *args: authorization_calls.append(args) or True
        )

        response = await adapter._handle_webhook(
            _FakeBlueBubblesRequest(
                {
                    "type": "new-message",
                    "data": {
                        "guid": "ambiguous-membership-guid",
                        "text": "must not choose first chat",
                        "handle": {"address": "+15555550100"},
                        "isFromMe": False,
                        "chats": [
                            {"guid": "iMessage;-;+15555550100"},
                            {"guid": "iMessage;+;family-group"},
                        ],
                    },
                }
            )
        )

        assert response.status == 200
        assert authorization_calls == []
        assert handled == []

    @pytest.mark.asyncio
    async def test_cancelled_membership_lookup_releases_provisional_for_retry(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        lookup_started = asyncio.Event()
        handled = []

        async def blocking_lookup(message_guid):
            lookup_started.set()
            await asyncio.Event().wait()

        async def resolved_lookup(message_guid):
            return "iMessage;-;user@example.com"

        async def fake_handle_message(event):
            handled.append(event)

        monkeypatch.setattr(
            adapter,
            "_resolve_exact_message_chat_guid",
            blocking_lookup,
        )
        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        payload = {
            "type": "new-message",
            "data": {
                "guid": "cancelled-membership-guid",
                "text": "retry after cancellation",
                "handle": {"address": "user@example.com"},
                "isFromMe": False,
                "chatIdentifier": "user@example.com",
            },
        }

        first = asyncio.create_task(
            adapter._handle_webhook(_FakeBlueBubblesRequest(payload))
        )
        await lookup_started.wait()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first

        assert adapter._provisional_messages == {}
        assert adapter._provisional_message_tasks == {}

        monkeypatch.setattr(
            adapter,
            "_resolve_exact_message_chat_guid",
            resolved_lookup,
        )
        retry = await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))
        await asyncio.sleep(0)

        assert retry.status == 200
        assert [event.text for event in handled] == ["retry after cancellation"]

    @pytest.mark.asyncio
    async def test_shutdown_clears_provisional_and_same_webhook_remains_retryable(
        self, monkeypatch
    ):
        import gateway.platforms.bluebubbles as bluebubbles_module

        monkeypatch.setattr(
            bluebubbles_module,
            "_PROVISIONAL_MESSAGE_WAIT_SECONDS",
            60,
        )
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        handled = []

        async def unresolved_lookup(message_guid):
            return None

        async def resolved_lookup(message_guid):
            return "iMessage;-;user@example.com"

        async def fake_handle_message(event):
            handled.append(event)

        monkeypatch.setattr(
            adapter,
            "_resolve_exact_message_chat_guid",
            unresolved_lookup,
        )
        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        payload = {
            "type": "new-message",
            "data": {
                "guid": "shutdown-provisional-guid",
                "text": "retry after shutdown",
                "handle": {"address": "user@example.com"},
                "isFromMe": False,
                "chatIdentifier": "user@example.com",
            },
        }

        await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))
        assert len(adapter._provisional_messages) == 1
        assert len(adapter._provisional_message_tasks) == 1

        await adapter.cancel_background_tasks()

        assert adapter._provisional_messages == {}
        assert adapter._provisional_message_tasks == {}
        monkeypatch.setattr(
            adapter,
            "_resolve_exact_message_chat_guid",
            resolved_lookup,
        )
        retry = await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))
        await asyncio.sleep(0)

        assert retry.status == 200
        assert [event.text for event in handled] == ["retry after shutdown"]

    @pytest.mark.asyncio
    async def test_lookup_resolved_group_authorizes_with_authoritative_chat(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        authorization_calls = []
        handled = []

        async def resolve_group(message_guid):
            assert message_guid == "lookup-group-guid"
            return "iMessage;+;authoritative-group"

        async def fake_handle_message(event):
            handled.append(event)

        def authorize(user_id, chat_type, chat_id):
            authorization_calls.append((user_id, chat_type, chat_id))
            return True

        monkeypatch.setattr(
            adapter,
            "_resolve_exact_message_chat_guid",
            resolve_group,
        )
        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        adapter.set_authorization_check(authorize)

        await adapter._handle_webhook(
            _FakeBlueBubblesRequest(
                {
                    "type": "new-message",
                    "data": {
                        "guid": "lookup-group-guid",
                        "text": "group request",
                        "handle": {"address": "+15555550100"},
                        "isFromMe": False,
                        "chatIdentifier": "+15555550100",
                    },
                }
            )
        )
        await asyncio.sleep(0)

        assert authorization_calls == [
            (
                "+15555550100",
                "group",
                "iMessage;+;authoritative-group",
            )
        ]
        assert [
            (event.source.chat_id, event.source.chat_type) for event in handled
        ] == [("iMessage;+;authoritative-group", "group")]

    @pytest.mark.asyncio
    async def test_same_guid_in_two_authoritative_chats_remains_isolated(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        handled = []

        async def fake_handle_message(event):
            handled.append(event)

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)

        for chat_guid in ("iMessage;+;group-a", "iMessage;+;group-b"):
            await adapter._handle_webhook(
                _FakeBlueBubblesRequest(
                    {
                        "type": "new-message",
                        "data": {
                            "guid": "provider-collision-guid",
                            "text": "same provider identity",
                            "handle": {"address": "+15555550100"},
                            "isFromMe": False,
                            "chatGuid": chat_guid,
                            "chatIdentifier": chat_guid.rsplit(";", 1)[-1],
                            "isGroup": True,
                        },
                    }
                )
            )
            await asyncio.sleep(0)

        assert [event.source.chat_id for event in handled] == [
            "iMessage;+;group-a",
            "iMessage;+;group-b",
        ]
        assert {
            key[0] for key in adapter._message_revision_serials
        } == {
            "iMessage;+;group-a",
            "iMessage;+;group-b",
        }

    @pytest.mark.asyncio
    async def test_meaningful_text_revision_with_same_guid_is_processed(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        handled = []

        async def fake_handle_message(event):
            handled.append(event)

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        base = {
            "guid": "edited-message-guid",
            "handle": {"address": "user@example.com"},
            "isFromMe": False,
            "chatGuid": "iMessage;-;user@example.com",
            "chatIdentifier": "user@example.com",
        }

        await adapter._handle_webhook(_FakeBlueBubblesRequest({
            "type": "new-message",
            "data": {**base, "text": "original text"},
        }))
        await asyncio.sleep(0)
        await adapter._handle_webhook(_FakeBlueBubblesRequest({
            "type": "updated-message",
            "data": {**base, "text": "edited text"},
        }))
        await asyncio.sleep(0)

        assert [event.text for event in handled] == ["original text", "edited text"]

    @pytest.mark.asyncio
    async def test_update_before_new_keeps_the_updated_revision(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        adapter._message_revision_wait_seconds = 0.02
        handled = []

        async def fake_handle_message(event):
            handled.append(event)

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        base = {
            "guid": "update-before-new-guid",
            "handle": {"address": "user@example.com"},
            "isFromMe": False,
            "chatGuid": "iMessage;-;user@example.com",
            "chatIdentifier": "user@example.com",
        }

        await adapter._handle_webhook(_FakeBlueBubblesRequest({
            "type": "updated-message",
            "data": {**base, "text": "edited text", "dateEdited": 200},
        }))
        await adapter._handle_webhook(_FakeBlueBubblesRequest({
            "type": "new-message",
            "data": {**base, "text": "original text", "dateCreated": 100},
        }))
        await asyncio.sleep(0.05)

        assert [event.text for event in handled] == ["edited text"]

    @pytest.mark.asyncio
    async def test_older_dated_update_cannot_replace_a_newer_update(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        adapter._message_revision_wait_seconds = 0.02
        handled = []

        async def fake_handle_message(event):
            handled.append(event)

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        base = {
            "guid": "out-of-order-updates-guid",
            "handle": {"address": "user@example.com"},
            "isFromMe": False,
            "chatGuid": "iMessage;-;user@example.com",
            "chatIdentifier": "user@example.com",
        }

        await adapter._handle_webhook(_FakeBlueBubblesRequest({
            "type": "updated-message",
            "data": {**base, "text": "newest text", "dateEdited": 300},
        }))
        await adapter._handle_webhook(_FakeBlueBubblesRequest({
            "type": "updated-message",
            "data": {**base, "text": "stale text", "dateEdited": 200},
        }))
        await asyncio.sleep(0.05)

        assert [event.text for event in handled] == ["newest text"]

    @pytest.mark.asyncio
    async def test_attachment_revision_with_same_guid_is_processed(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        handled = []

        async def fake_handle_message(event):
            handled.append(event)

        async def fake_download_attachment(att_guid, att_meta):
            return f"/tmp/{att_guid}"

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        monkeypatch.setattr(adapter, "_download_attachment", fake_download_attachment)
        base = {
            "guid": "attachment-revision-guid",
            "text": "",
            "handle": {"address": "user@example.com"},
            "isFromMe": False,
            "chatGuid": "iMessage;-;user@example.com",
            "chatIdentifier": "user@example.com",
        }

        await adapter._handle_webhook(_FakeBlueBubblesRequest({
            "type": "new-message",
            "data": {
                **base,
                "attachments": [
                    {
                        "guid": "same-attachment",
                        "mimeType": "application/octet-stream",
                        "uti": "public.data",
                    }
                ],
            },
        }))
        await asyncio.sleep(0)
        await adapter._handle_webhook(_FakeBlueBubblesRequest({
            "type": "updated-message",
            "data": {
                **base,
                "attachments": [
                    {
                        "guid": "same-attachment",
                        "mimeType": "application/octet-stream",
                        "uti": "public.caf",
                    }
                ],
            },
        }))
        await asyncio.sleep(0)

        assert [event.media_urls for event in handled] == [
            ["/tmp/same-attachment"],
            ["/tmp/same-attachment"],
        ]

    @pytest.mark.asyncio
    async def test_duplicate_attachment_revision_ignores_mutable_delivery_metadata(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        handled = []

        async def fake_handle_message(event):
            handled.append(event)

        async def fake_download_attachment(att_guid, att_meta):
            return f"/tmp/{att_guid}"

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        monkeypatch.setattr(adapter, "_download_attachment", fake_download_attachment)
        base = {
            "guid": "stable-attachment-identity-guid",
            "text": "same caption",
            "handle": {"address": "user@example.com"},
            "isFromMe": False,
            "chatGuid": "iMessage;-;user@example.com",
            "chatIdentifier": "user@example.com",
        }

        for delivered_at in (100, 200):
            await adapter._handle_webhook(_FakeBlueBubblesRequest({
                "type": "updated-message",
                "deliveryId": f"delivery-{delivered_at}",
                "data": {
                    **base,
                    "attachments": [
                        {
                            "guid": "stable-photo-guid",
                            "mimeType": "image/png",
                            "transferName": "photo.png",
                            "totalBytes": 1234,
                            "updatedAt": delivered_at,
                        }
                    ],
                },
            }))
            await asyncio.sleep(0)

        assert len(handled) == 1
        assert handled[0].media_urls == ["/tmp/stable-photo-guid"]

    @pytest.mark.asyncio
    async def test_text_and_attachment_revisions_with_same_guid_are_coalesced(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        adapter._message_revision_wait_seconds = 0.02
        handled = []

        async def fake_handle_message(event):
            handled.append(event)

        async def fake_download_attachment(att_guid, att_meta):
            return f"/tmp/{att_guid}"

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        monkeypatch.setattr(adapter, "_download_attachment", fake_download_attachment)
        base = {
            "guid": "split-composition-guid",
            "text": "What size is this part?",
            "handle": {"address": "user@example.com"},
            "isFromMe": False,
            "chatGuid": "iMessage;-;user@example.com",
            "chatIdentifier": "user@example.com",
        }

        await adapter._handle_webhook(_FakeBlueBubblesRequest({
            "type": "new-message",
            "data": base,
        }))
        await adapter._handle_webhook(_FakeBlueBubblesRequest({
            "type": "updated-message",
            "data": {
                **base,
                "text": "",
                "attachments": [
                    {"guid": "part-photo", "mimeType": "image/png"}
                ],
            },
        }))
        await asyncio.sleep(0.05)

        assert len(handled) == 1
        assert handled[0].text == "What size is this part?"
        assert handled[0].media_urls == ["/tmp/part-photo"]
        assert handled[0].message_id == "split-composition-guid"

    @pytest.mark.asyncio
    async def test_text_edit_preserves_pending_attachment_revision(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        adapter._message_revision_wait_seconds = 0.02
        handled = []

        async def fake_handle_message(event):
            handled.append(event)

        async def fake_download_attachment(att_guid, att_meta):
            return f"/tmp/{att_guid}"

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        monkeypatch.setattr(adapter, "_download_attachment", fake_download_attachment)
        base = {
            "guid": "attachment-then-edit-guid",
            "handle": {"address": "user@example.com"},
            "isFromMe": False,
            "chatGuid": "iMessage;-;user@example.com",
            "chatIdentifier": "user@example.com",
        }

        await adapter._handle_webhook(_FakeBlueBubblesRequest({
            "type": "updated-message",
            "data": {
                **base,
                "text": "Original caption",
                "attachments": [
                    {"guid": "pending-photo", "mimeType": "image/png"}
                ],
            },
        }))
        await adapter._handle_webhook(_FakeBlueBubblesRequest({
            "type": "updated-message",
            "data": {**base, "text": "Edited caption"},
        }))
        await asyncio.sleep(0.05)

        assert len(handled) == 1
        assert handled[0].text == "Edited caption"
        assert handled[0].media_urls == ["/tmp/pending-photo"]
        assert handled[0].media_types == ["image/png"]
        assert handled[0].message_type is MessageType.PHOTO

    @pytest.mark.asyncio
    async def test_same_guid_isolated_by_exact_chat_and_sender(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        adapter._message_revision_wait_seconds = 0.01
        handled = []

        async def fake_handle_message(event):
            handled.append(event)

        async def fake_download_attachment(att_guid, att_meta):
            return f"/tmp/{att_guid}"

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        monkeypatch.setattr(adapter, "_download_attachment", fake_download_attachment)
        guid = "provider-collision-guid"
        payloads = [
            {
                "guid": guid,
                "text": "dm media",
                "handle": {"address": "same@example.com"},
                "isFromMe": False,
                "chatGuid": "iMessage;-;same@example.com",
                "chatIdentifier": "same@example.com",
                "attachments": [{"guid": "dm-photo", "mimeType": "image/png"}],
            },
            {
                "guid": guid,
                "text": "group from same sender",
                "handle": {"address": "same@example.com"},
                "isFromMe": False,
                "isGroup": True,
                "chatGuid": "iMessage;+;shared-group",
                "chatIdentifier": "shared-group",
            },
            {
                "guid": guid,
                "text": "group from other sender",
                "handle": {"address": "other@example.com"},
                "isFromMe": False,
                "isGroup": True,
                "chatGuid": "iMessage;+;shared-group",
                "chatIdentifier": "shared-group",
            },
        ]

        await asyncio.gather(*(
            adapter._handle_webhook(_FakeBlueBubblesRequest({
                "type": "updated-message",
                "data": payload,
            }))
            for payload in payloads
        ))
        await asyncio.sleep(0.04)

        assert sorted(event.text for event in handled) == sorted(
            payload["text"] for payload in payloads
        )
        assert all(event.message_id == guid for event in handled)
        assert {event.text: event.media_urls for event in handled} == {
            "dm media": ["/tmp/dm-photo"],
            "group from same sender": [],
            "group from other sender": [],
        }

    @pytest.mark.asyncio
    async def test_same_identifier_different_chat_guids_do_not_coalesce(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        adapter._message_revision_wait_seconds = 0.01
        handled = []

        async def fake_handle_message(event):
            handled.append(event)

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        base = {
            "guid": "shared-provider-guid",
            "handle": {"address": "same@example.com"},
            "isFromMe": False,
            "isGroup": True,
            "chatIdentifier": "shared-display-name",
        }

        await asyncio.gather(
            adapter._handle_webhook(_FakeBlueBubblesRequest({
                "type": "updated-message",
                "data": {**base, "chatGuid": "iMessage;+;group-one", "text": "one"},
            })),
            adapter._handle_webhook(_FakeBlueBubblesRequest({
                "type": "updated-message",
                "data": {**base, "chatGuid": "iMessage;+;group-two", "text": "two"},
            })),
        )
        await asyncio.sleep(0.04)

        assert sorted(event.text for event in handled) == ["one", "two"]

    @pytest.mark.asyncio
    async def test_transient_text_edit_retry_keeps_merged_media(
        self, monkeypatch
    ):
        adapter = _make_adapter(
            monkeypatch,
            send_read_receipts=False,
            message_retry_base_delay_seconds=0,
        )
        adapter._message_revision_wait_seconds = 0.01
        handled = []

        async def flaky_handle_message(event):
            handled.append(event)
            if len(handled) == 1:
                raise ConnectionError("retry merged revision")

        async def fake_download_attachment(att_guid, att_meta):
            return f"/tmp/{att_guid}"

        monkeypatch.setattr(adapter, "handle_message", flaky_handle_message)
        monkeypatch.setattr(adapter, "_download_attachment", fake_download_attachment)
        base = {
            "guid": "failed-attachment-edit-guid",
            "handle": {"address": "user@example.com"},
            "isFromMe": False,
            "chatGuid": "iMessage;-;user@example.com",
            "chatIdentifier": "user@example.com",
        }
        richer_payload = {
            "type": "updated-message",
            "data": {
                **base,
                "text": "Original caption",
                "attachments": [
                    {"guid": "retry-photo", "mimeType": "image/png"}
                ],
            },
        }

        await adapter._handle_webhook(_FakeBlueBubblesRequest(richer_payload))
        await adapter._handle_webhook(_FakeBlueBubblesRequest({
            "type": "updated-message",
            "data": {**base, "text": "Edited caption"},
        }))
        await asyncio.sleep(0.03)

        assert len(handled) == 2
        assert handled[0].text == "Edited caption"
        assert handled[0].media_urls == ["/tmp/retry-photo"]
        assert handled[1].media_urls == ["/tmp/retry-photo"]

    @pytest.mark.asyncio
    async def test_transient_sparse_dispatch_retry_restores_media(self, monkeypatch):
        adapter = _make_adapter(
            monkeypatch,
            send_read_receipts=False,
            message_retry_base_delay_seconds=0,
        )
        adapter._message_revision_wait_seconds = 0.01
        handled = []

        async def flaky_handle_message(event):
            handled.append(event)
            if len(handled) == 1:
                raise ConnectionError("retry sparse revision")

        async def fake_download_attachment(att_guid, att_meta):
            return f"/tmp/{att_guid}"

        monkeypatch.setattr(adapter, "handle_message", flaky_handle_message)
        monkeypatch.setattr(adapter, "_download_attachment", fake_download_attachment)
        base = {
            "guid": "failed-rich-sparse-retry-guid",
            "handle": {"address": "user@example.com"},
            "isFromMe": False,
            "chatGuid": "iMessage;-;user@example.com",
            "chatIdentifier": "user@example.com",
        }
        sparse_payload = {
            "type": "updated-message",
            "data": {**base, "text": "Newest caption"},
        }

        await adapter._handle_webhook(_FakeBlueBubblesRequest({
            "type": "updated-message",
            "data": {
                **base,
                "text": "Older caption",
                "attachments": [
                    {"guid": "retry-from-context-photo", "mimeType": "image/png"}
                ],
            },
        }))
        await adapter._handle_webhook(_FakeBlueBubblesRequest(sparse_payload))
        await asyncio.sleep(0.03)

        assert [event.text for event in handled] == ["Newest caption", "Newest caption"]
        assert [event.media_urls for event in handled] == [
            ["/tmp/retry-from-context-photo"],
            ["/tmp/retry-from-context-photo"],
        ]
        assert adapter._message_revision_media == {}

    @pytest.mark.asyncio
    async def test_in_progress_attachment_download_extends_coalesce_window(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        adapter._message_revision_wait_seconds = 0.01
        download_started = asyncio.Event()
        release_download = asyncio.Event()
        handled = []

        async def fake_handle_message(event):
            handled.append(event)

        async def fake_download_attachment(att_guid, att_meta):
            download_started.set()
            await release_download.wait()
            return f"/tmp/{att_guid}"

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        monkeypatch.setattr(adapter, "_download_attachment", fake_download_attachment)
        base = {
            "guid": "slow-attachment-guid",
            "text": "Please inspect this",
            "handle": {"address": "user@example.com"},
            "isFromMe": False,
            "chatGuid": "iMessage;-;user@example.com",
            "chatIdentifier": "user@example.com",
        }

        await adapter._handle_webhook(_FakeBlueBubblesRequest({
            "type": "new-message",
            "data": base,
        }))
        updated_task = asyncio.create_task(
            adapter._handle_webhook(_FakeBlueBubblesRequest({
                "type": "updated-message",
                "data": {
                    **base,
                    "attachments": [
                        {"guid": "slow-photo", "mimeType": "image/png"}
                    ],
                },
            }))
        )
        await download_started.wait()
        await asyncio.sleep(0.03)
        assert handled == []

        release_download.set()
        await updated_task
        await asyncio.sleep(0.03)

        assert len(handled) == 1
        assert handled[0].media_urls == ["/tmp/slow-photo"]

    @pytest.mark.asyncio
    async def test_cancelled_attachment_download_releases_coalesce_hold(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        adapter._message_revision_wait_seconds = 0.01
        download_started = asyncio.Event()
        handled = []

        async def fake_handle_message(event):
            handled.append(event)

        async def fake_download_attachment(att_guid, att_meta):
            download_started.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        monkeypatch.setattr(adapter, "_download_attachment", fake_download_attachment)
        base = {
            "guid": "cancelled-download-guid",
            "text": "Keep this caption",
            "handle": {"address": "user@example.com"},
            "isFromMe": False,
            "chatGuid": "iMessage;-;user@example.com",
            "chatIdentifier": "user@example.com",
        }

        await adapter._handle_webhook(_FakeBlueBubblesRequest({
            "type": "new-message",
            "data": base,
        }))
        updated_task = asyncio.create_task(
            adapter._handle_webhook(_FakeBlueBubblesRequest({
                "type": "updated-message",
                "data": {
                    **base,
                    "attachments": [
                        {"guid": "cancelled-photo", "mimeType": "image/png"}
                    ],
                },
            }))
        )
        await download_started.wait()
        updated_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await updated_task
        await asyncio.sleep(0.03)

        assert [event.text for event in handled] == ["Keep this caption"]
        assert adapter._active_attachment_revisions == {}

    @pytest.mark.asyncio
    async def test_cancellation_after_download_releases_coalesce_hold(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        adapter._message_revision_wait_seconds = 0.01
        queue_started = asyncio.Event()
        handled = []

        async def fake_handle_message(event):
            handled.append(event)

        async def fake_download_attachment(att_guid, att_meta):
            return f"/tmp/{att_guid}"

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        monkeypatch.setattr(adapter, "_download_attachment", fake_download_attachment)
        base = {
            "guid": "cancelled-after-download-guid",
            "text": "Keep this caption",
            "handle": {"address": "user@example.com"},
            "isFromMe": False,
            "chatGuid": "iMessage;-;user@example.com",
            "chatIdentifier": "user@example.com",
        }

        await adapter._handle_webhook(_FakeBlueBubblesRequest({
            "type": "new-message",
            "data": base,
        }))

        async def blocking_queue(*args, **kwargs):
            queue_started.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(adapter, "_queue_message_revision", blocking_queue)
        updated_task = asyncio.create_task(
            adapter._handle_webhook(_FakeBlueBubblesRequest({
                "type": "updated-message",
                "data": {
                    **base,
                    "attachments": [
                        {"guid": "downloaded-photo", "mimeType": "image/png"}
                    ],
                },
            }))
        )
        await queue_started.wait()
        updated_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await updated_task
        await asyncio.sleep(0.03)

        assert [event.text for event in handled] == ["Keep this caption"]
        assert adapter._active_attachment_revisions == {}
        assert adapter._pending_message_identities == set()

    @pytest.mark.asyncio
    async def test_slow_older_attachment_revision_cannot_replace_newer(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        adapter._message_revision_wait_seconds = 0.01
        slow_started = asyncio.Event()
        release_slow = asyncio.Event()
        handled = []

        async def fake_handle_message(event):
            handled.append(event)

        async def fake_download_attachment(att_guid, att_meta):
            if att_guid == "older-photo":
                slow_started.set()
                await release_slow.wait()
            return f"/tmp/{att_guid}"

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        monkeypatch.setattr(adapter, "_download_attachment", fake_download_attachment)
        base = {
            "guid": "out-of-order-guid",
            "text": "Inspect the latest photo",
            "handle": {"address": "user@example.com"},
            "isFromMe": False,
            "chatGuid": "iMessage;-;user@example.com",
            "chatIdentifier": "user@example.com",
        }

        await adapter._handle_webhook(_FakeBlueBubblesRequest({
            "type": "new-message",
            "data": base,
        }))
        older_task = asyncio.create_task(
            adapter._handle_webhook(_FakeBlueBubblesRequest({
                "type": "updated-message",
                "data": {
                    **base,
                    "dateEdited": 100,
                    "attachments": [
                        {"guid": "older-photo", "mimeType": "image/png"}
                    ],
                },
            }))
        )
        await slow_started.wait()
        await adapter._handle_webhook(_FakeBlueBubblesRequest({
            "type": "updated-message",
            "data": {
                **base,
                "dateEdited": 200,
                "attachments": [
                    {"guid": "newer-photo", "mimeType": "image/png"}
                ],
            },
        }))
        release_slow.set()
        await older_task
        await adapter._handle_webhook(_FakeBlueBubblesRequest({
            "type": "updated-message",
            "data": {
                **base,
                "dateEdited": 100,
                "attachments": [
                    {"guid": "older-photo", "mimeType": "image/png"}
                ],
            },
        }))
        await asyncio.sleep(0.03)

        assert len(handled) == 1
        assert handled[0].media_urls == ["/tmp/newer-photo"]

    @pytest.mark.asyncio
    async def test_slow_older_rich_revision_merges_into_newer_sparse_text(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        adapter._message_revision_wait_seconds = 0.01
        slow_started = asyncio.Event()
        release_slow = asyncio.Event()
        handled = []

        async def fake_handle_message(event):
            handled.append(event)

        async def fake_download_attachment(att_guid, att_meta):
            slow_started.set()
            await release_slow.wait()
            return f"/tmp/{att_guid}"

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        monkeypatch.setattr(adapter, "_download_attachment", fake_download_attachment)
        base = {
            "guid": "older-rich-newer-sparse-guid",
            "handle": {"address": "user@example.com"},
            "isFromMe": False,
            "chatGuid": "iMessage;-;user@example.com",
            "chatIdentifier": "user@example.com",
        }

        older_task = asyncio.create_task(
            adapter._handle_webhook(_FakeBlueBubblesRequest({
                "type": "updated-message",
                "data": {
                    **base,
                    "text": "Older caption",
                    "attachments": [
                        {"guid": "slow-rich-photo", "mimeType": "image/png"}
                    ],
                },
            }))
        )
        await slow_started.wait()
        await adapter._handle_webhook(_FakeBlueBubblesRequest({
            "type": "updated-message",
            "data": {**base, "text": "Newest caption"},
        }))
        release_slow.set()
        await older_task
        await asyncio.sleep(0.03)

        assert len(handled) == 1
        assert handled[0].text == "Newest caption"
        assert handled[0].media_urls == ["/tmp/slow-rich-photo"]
        assert handled[0].message_id == base["guid"]

    @pytest.mark.asyncio
    async def test_retry_of_superseded_revision_cannot_replace_richer_revision(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        adapter._message_revision_wait_seconds = 0.02
        handled = []

        async def fake_handle_message(event):
            handled.append(event)

        async def fake_download_attachment(att_guid, att_meta):
            return f"/tmp/{att_guid}"

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        monkeypatch.setattr(adapter, "_download_attachment", fake_download_attachment)
        base = {
            "guid": "superseded-retry-guid",
            "text": "Inspect this image",
            "handle": {"address": "user@example.com"},
            "isFromMe": False,
            "chatGuid": "iMessage;-;user@example.com",
            "chatIdentifier": "user@example.com",
        }
        text_payload = {"type": "new-message", "data": base}

        await adapter._handle_webhook(_FakeBlueBubblesRequest(text_payload))
        await adapter._handle_webhook(_FakeBlueBubblesRequest({
            "type": "updated-message",
            "data": {
                **base,
                "attachments": [
                    {"guid": "richer-photo", "mimeType": "image/png"}
                ],
            },
        }))
        await adapter._handle_webhook(_FakeBlueBubblesRequest(text_payload))
        await asyncio.sleep(0.05)

        assert len(handled) == 1
        assert handled[0].media_urls == ["/tmp/richer-photo"]

    @pytest.mark.asyncio
    async def test_group_attachment_revision_inherits_accepted_caption(self, monkeypatch):
        adapter = _make_adapter(
            monkeypatch,
            require_mention=True,
            send_read_receipts=False,
        )
        adapter._message_revision_wait_seconds = 0.02
        handled = []

        async def fake_handle_message(event):
            handled.append(event)

        async def fake_download_attachment(att_guid, att_meta):
            return f"/tmp/{att_guid}"

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        monkeypatch.setattr(adapter, "_download_attachment", fake_download_attachment)
        base = {
            "guid": "group-split-guid",
            "handle": {"address": "user@example.com"},
            "isFromMe": False,
            "isGroup": True,
            "chatGuid": "iMessage;+;family-group",
            "chatIdentifier": "family-group",
        }

        await adapter._handle_webhook(_FakeBlueBubblesRequest({
            "type": "new-message",
            "data": {**base, "text": "Hermes, what is this?"},
        }))
        await adapter._handle_webhook(_FakeBlueBubblesRequest({
            "type": "updated-message",
            "data": {
                **base,
                "text": "",
                "attachments": [
                    {"guid": "group-photo", "mimeType": "image/png"}
                ],
            },
        }))
        await asyncio.sleep(0.05)

        assert len(handled) == 1
        assert handled[0].text == "what is this?"
        assert handled[0].media_urls == ["/tmp/group-photo"]

    @pytest.mark.asyncio
    async def test_late_group_attachment_revision_keeps_accepted_caption(self, monkeypatch):
        adapter = _make_adapter(
            monkeypatch,
            require_mention=True,
            send_read_receipts=False,
        )
        adapter._message_revision_wait_seconds = 0.01
        handled = []

        async def fake_handle_message(event):
            handled.append(event)

        async def fake_download_attachment(att_guid, att_meta):
            return f"/tmp/{att_guid}"

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        monkeypatch.setattr(adapter, "_download_attachment", fake_download_attachment)
        base = {
            "guid": "late-group-split-guid",
            "handle": {"address": "user@example.com"},
            "isFromMe": False,
            "isGroup": True,
            "chatGuid": "iMessage;+;family-group",
            "chatIdentifier": "family-group",
        }

        await adapter._handle_webhook(_FakeBlueBubblesRequest({
            "type": "new-message",
            "data": {**base, "text": "Hermes, identify this"},
        }))
        await asyncio.sleep(0.03)
        await adapter._handle_webhook(_FakeBlueBubblesRequest({
            "type": "updated-message",
            "data": {
                **base,
                "text": "",
                "attachments": [
                    {"guid": "late-group-photo", "mimeType": "image/png"}
                ],
            },
        }))
        await asyncio.sleep(0.03)

        assert [event.text for event in handled] == [
            "identify this",
            "identify this",
        ]
        assert handled[-1].media_urls == ["/tmp/late-group-photo"]

    @pytest.mark.asyncio
    async def test_group_caption_is_not_inherited_across_sender_or_chat(self, monkeypatch):
        adapter = _make_adapter(
            monkeypatch,
            require_mention=True,
            send_read_receipts=False,
        )
        adapter._message_revision_wait_seconds = 0.02
        handled = []

        async def fake_handle_message(event):
            handled.append(event)

        async def fake_download_attachment(att_guid, att_meta):
            return f"/tmp/{att_guid}"

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        monkeypatch.setattr(adapter, "_download_attachment", fake_download_attachment)
        guid = "defensive-collision-guid"

        await adapter._handle_webhook(_FakeBlueBubblesRequest({
            "type": "new-message",
            "data": {
                "guid": guid,
                "text": "Hermes, private caption",
                "handle": {"address": "first@example.com"},
                "isFromMe": False,
                "isGroup": True,
                "chatGuid": "iMessage;+;first-group",
                "chatIdentifier": "first-group",
            },
        }))
        await adapter._handle_webhook(_FakeBlueBubblesRequest({
            "type": "updated-message",
            "data": {
                "guid": guid,
                "text": "",
                "handle": {"address": "second@example.com"},
                "isFromMe": False,
                "isGroup": True,
                "chatGuid": "iMessage;+;second-group",
                "chatIdentifier": "second-group",
                "attachments": [
                    {"guid": "unrelated-photo", "mimeType": "image/png"}
                ],
            },
        }))
        await asyncio.sleep(0.05)

        assert [event.text for event in handled] == ["private caption"]
        assert handled[0].media_urls == []

    @pytest.mark.asyncio
    async def test_late_revision_cancels_in_flight_dispatch_before_commit(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        adapter._message_revision_wait_seconds = 0.01
        started = asyncio.Event()
        release = asyncio.Event()
        cancelled = False
        handled = []

        async def fake_handle_message(event):
            nonlocal cancelled
            handled.append(event.text)
            if len(handled) == 1:
                started.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    cancelled = True
                    raise

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        base = {
            "guid": "late-revision-guid",
            "handle": {"address": "user@example.com"},
            "isFromMe": False,
            "chatGuid": "iMessage;-;user@example.com",
            "chatIdentifier": "user@example.com",
        }

        await adapter._handle_webhook(_FakeBlueBubblesRequest({
            "type": "new-message",
            "data": {**base, "text": "first revision"},
        }))
        await started.wait()
        await adapter._handle_webhook(_FakeBlueBubblesRequest({
            "type": "updated-message",
            "data": {**base, "text": "late revision"},
        }))
        await asyncio.sleep(0.03)
        release.set()
        await asyncio.sleep(0)

        assert cancelled is True
        assert handled == ["first revision", "late revision"]

    @pytest.mark.asyncio
    async def test_shutdown_clears_pending_revision_state(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        adapter._message_revision_wait_seconds = 60
        handled = []

        async def fake_handle_message(event):
            handled.append(event)

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        await adapter._handle_webhook(_FakeBlueBubblesRequest({
            "type": "new-message",
            "data": {
                "guid": "shutdown-pending-guid",
                "text": "held during shutdown",
                "handle": {"address": "user@example.com"},
                "isFromMe": False,
                "chatGuid": "iMessage;-;user@example.com",
                "chatIdentifier": "user@example.com",
            },
        }))

        await adapter.cancel_background_tasks()

        assert handled == []
        assert adapter._pending_message_revisions == {}
        assert adapter._pending_message_revision_tasks == {}
        assert adapter._pending_message_identities == set()

    @pytest.mark.asyncio
    async def test_attachment_readiness_transition_is_processed(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        adapter._message_revision_wait_seconds = 0.02
        handled = []
        downloads = 0

        async def fake_handle_message(event):
            handled.append(event)

        async def fake_download_attachment(att_guid, att_meta):
            nonlocal downloads
            downloads += 1
            return None if downloads == 1 else f"/tmp/{att_guid}"

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        monkeypatch.setattr(adapter, "_download_attachment", fake_download_attachment)
        payload = {
            "type": "updated-message",
            "data": {
                "guid": "attachment-ready-guid",
                "text": "caption",
                "handle": {"address": "user@example.com"},
                "isFromMe": False,
                "chatGuid": "iMessage;-;user@example.com",
                "chatIdentifier": "user@example.com",
                "attachments": [{"guid": "same-attachment", "mimeType": "image/png"}],
            },
        }

        await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))
        await asyncio.sleep(0)
        await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))
        await asyncio.sleep(0.05)

        assert [event.media_urls for event in handled] == [
            ["/tmp/same-attachment"],
        ]

    @pytest.mark.asyncio
    async def test_failed_dispatch_releases_guid_for_retry(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        attempts = 0
        handled = []

        async def fake_handle_message(event):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("transient dispatch failure")
            handled.append(event.text)

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        payload = {
            "type": "new-message",
            "data": {
                "guid": "retryable-message-guid",
                "text": "retry me",
                "handle": {"address": "user@example.com"},
                "isFromMe": False,
                "chatGuid": "iMessage;-;user@example.com",
                "chatIdentifier": "user@example.com",
            },
        }

        first = await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))
        await asyncio.sleep(0)
        second = await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))
        await asyncio.sleep(0)

        assert first.status == 200
        assert second.status == 200
        assert attempts == 2
        assert handled == ["retry me"]

    @pytest.mark.asyncio
    async def test_cancelled_dispatch_releases_identity_for_retry(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        attempts = 0
        handled = []

        async def fake_handle_message(event):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise asyncio.CancelledError()
            handled.append(event.text)

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        payload = {
            "type": "new-message",
            "data": {
                "guid": "cancelled-message-guid",
                "text": "retry cancellation",
                "handle": {"address": "user@example.com"},
                "isFromMe": False,
                "chatGuid": "iMessage;-;user@example.com",
                "chatIdentifier": "user@example.com",
            },
        }

        await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))
        await asyncio.sleep(0)
        await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))
        await asyncio.sleep(0)

        assert attempts == 2
        assert handled == ["retry cancellation"]

    @pytest.mark.asyncio
    async def test_pending_identity_is_not_evicted_by_completed_lru(self, monkeypatch):
        import gateway.platforms.bluebubbles as bluebubbles_module

        monkeypatch.setattr(bluebubbles_module, "_MESSAGE_DEDUP_SIZE", 1)
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        started = asyncio.Event()
        release = asyncio.Event()
        attempts = {"pending-guid": 0, "completed-guid": 0}

        async def fake_handle_message(event):
            attempts[event.message_id] += 1
            if event.message_id == "pending-guid":
                started.set()
                await release.wait()

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)

        def payload(guid, text):
            return {
                "type": "new-message",
                "data": {
                    "guid": guid,
                    "text": text,
                    "handle": {"address": "user@example.com"},
                    "isFromMe": False,
                    "chatGuid": "iMessage;-;user@example.com",
                    "chatIdentifier": "user@example.com",
                },
            }

        await adapter._handle_webhook(
            _FakeBlueBubblesRequest(payload("pending-guid", "still running"))
        )
        await started.wait()
        await adapter._handle_webhook(
            _FakeBlueBubblesRequest(payload("completed-guid", "done"))
        )
        await asyncio.sleep(0)
        await adapter._handle_webhook(
            _FakeBlueBubblesRequest(payload("pending-guid", "still running"))
        )
        await asyncio.sleep(0)
        pending_attempts = attempts["pending-guid"]
        release.set()
        await asyncio.sleep(0)

        assert pending_attempts == 1


class TestBlueBubblesWebhookParsing:

    @pytest.mark.asyncio
    async def test_unresolved_inbound_event_never_routes_to_sender_dm(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        handled = []

        async def unresolved(message_guid):
            return None

        monkeypatch.setattr(adapter, "_resolve_exact_message_chat_guid", unresolved)
        monkeypatch.setattr(adapter, "handle_message", handled.append)

        response = await adapter._handle_webhook(
            _FakeBlueBubblesRequest(
                {
                    "type": "new-message",
                    "data": {
                        "guid": "unresolved-sender-guid",
                        "text": "must remain unresolved",
                        "handle": {"address": "user@example.com"},
                        "isFromMe": False,
                    },
                }
            )
        )
        await asyncio.sleep(0)

        assert response.status == 200
        assert handled == []


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
    @pytest.mark.parametrize(
        ("data", "expected"),
        [
            (
                {
                    "guid": "message/guid",
                    "chats": [{"guid": "iMessage;-;user@example.com"}],
                },
                "iMessage;-;user@example.com",
            ),
            ({"guid": "other-guid", "chats": [{"guid": "chat-a"}]}, None),
            ({"guid": "message/guid", "chats": []}, None),
            (
                {
                    "guid": "message/guid",
                    "chats": [{"guid": "chat-a"}, {"guid": "chat-b"}],
                },
                None,
            ),
            ({"guid": "message/guid", "chats": [None, {}]}, None),
        ],
    )
    async def test_exact_message_chat_resolution_requires_one_membership(
        self, monkeypatch, data, expected
    ):
        adapter = _make_adapter(monkeypatch)
        adapter.client = object()
        requests = []

        async def fake_api_get(path):
            requests.append(path)
            return {"data": data}

        monkeypatch.setattr(adapter, "_api_get", fake_api_get)

        assert await adapter._resolve_exact_message_chat_guid("message/guid") == expected
        assert requests == ["/api/v1/message/message%2Fguid?with=chats"]



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


class TestBlueBubblesAttachmentReadiness:
    @staticmethod
    def _payload(*, attachments=None, text="inspect this"):
        data = {
            "guid": "attachment-readiness-guid",
            "text": text,
            "handle": {"address": "user@example.com"},
            "isFromMe": False,
            "chatGuid": "iMessage;-;user@example.com",
            "chatIdentifier": "user@example.com",
        }
        if attachments is not None:
            data["attachments"] = attachments
        return {"type": "updated-message", "data": data}

    @pytest.mark.asyncio
    async def test_message_without_attachments_dispatches_immediately(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        handled = []
        monkeypatch.setattr(adapter, "handle_message", handled.append)

        await adapter._handle_webhook(
            _FakeBlueBubblesRequest(self._payload(attachments=None))
        )
        await asyncio.sleep(0)

        assert [event.media_urls for event in handled] == [[]]

    @pytest.mark.asyncio
    async def test_immediately_ready_attachments_dispatch_together(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        handled = []
        monkeypatch.setattr(adapter, "handle_message", handled.append)
        monkeypatch.setattr(
            adapter,
            "_download_attachment",
            lambda guid, _meta: asyncio.sleep(0, result=f"/tmp/{guid}"),
        )

        await adapter._handle_webhook(
            _FakeBlueBubblesRequest(
                self._payload(
                    attachments=[
                        {"guid": "ready-a", "mimeType": "image/png"},
                        {"guid": "ready-b", "mimeType": "image/jpeg"},
                    ]
                )
            )
        )
        await asyncio.sleep(0)

        assert [event.media_urls for event in handled] == [
            ["/tmp/ready-a", "/tmp/ready-b"]
        ]

    @pytest.mark.asyncio
    async def test_delayed_attachment_readiness_defers_and_retries_once(
        self, monkeypatch
    ):
        adapter = _make_adapter(
            monkeypatch,
            send_read_receipts=False,
            attachment_retry_delay_seconds=0.01,
        )
        handled = []
        downloads = 0

        async def download(guid, _meta):
            nonlocal downloads
            downloads += 1
            return None if downloads == 1 else f"/tmp/{guid}"

        async def refreshed(_chat_guid, _message_guid, **_kwargs):
            return self._payload(
                attachments=[
                    {
                        "guid": "delayed-photo",
                        "mimeType": "image/png",
                        "transferState": 5,
                    }
                ]
            )["data"]

        monkeypatch.setattr(adapter, "handle_message", handled.append)
        monkeypatch.setattr(adapter, "_download_attachment", download)
        monkeypatch.setattr(adapter, "_query_exact_message", refreshed)

        await adapter._handle_webhook(
            _FakeBlueBubblesRequest(
                self._payload(
                    attachments=[
                        {"guid": "delayed-photo", "mimeType": "image/png"}
                    ]
                )
            )
        )
        assert handled == []
        await asyncio.sleep(0.05)

        assert [event.media_urls for event in handled] == [["/tmp/delayed-photo"]]
        assert downloads == 2

    @pytest.mark.asyncio
    async def test_mixed_attachment_readiness_never_emits_partial_media(
        self, monkeypatch
    ):
        adapter = _make_adapter(
            monkeypatch,
            send_read_receipts=False,
            attachment_retry_delay_seconds=0.01,
        )
        handled = []
        attempts = {}

        async def download(guid, _meta):
            attempts[guid] = attempts.get(guid, 0) + 1
            if guid == "pending-b" and attempts[guid] == 1:
                return None
            return f"/tmp/{guid}"

        async def refreshed(_chat_guid, _message_guid, **_kwargs):
            return self._payload(
                attachments=[
                    {"guid": "ready-a", "mimeType": "image/png", "transferState": 5},
                    {"guid": "pending-b", "mimeType": "image/jpeg", "transferState": 5},
                ]
            )["data"]

        monkeypatch.setattr(adapter, "handle_message", handled.append)
        monkeypatch.setattr(adapter, "_download_attachment", download)
        monkeypatch.setattr(adapter, "_query_exact_message", refreshed)

        await adapter._handle_webhook(
            _FakeBlueBubblesRequest(
                self._payload(
                    attachments=[
                        {"guid": "ready-a", "mimeType": "image/png"},
                        {"guid": "pending-b", "mimeType": "image/jpeg"},
                    ]
                )
            )
        )
        assert handled == []
        await asyncio.sleep(0.05)

        assert [event.media_urls for event in handled] == [
            ["/tmp/ready-a", "/tmp/pending-b"]
        ]

    @pytest.mark.asyncio
    async def test_newer_revision_removing_attachment_cancels_pending_materialization(
        self, monkeypatch
    ):
        adapter = _make_adapter(
            monkeypatch,
            send_read_receipts=False,
            attachment_retry_delay_seconds=0.01,
        )
        handled = []
        monkeypatch.setattr(adapter, "handle_message", handled.append)
        monkeypatch.setattr(
            adapter,
            "_download_attachment",
            lambda *_args: asyncio.sleep(0, result=None),
        )

        await adapter._handle_webhook(
            _FakeBlueBubblesRequest(
                self._payload(
                    attachments=[
                        {"guid": "removed-photo", "mimeType": "image/png"}
                    ],
                    text="old revision",
                )
            )
        )
        await adapter._handle_webhook(
            _FakeBlueBubblesRequest(
                self._payload(attachments=[], text="new revision without photo")
            )
        )
        await asyncio.sleep(0.05)

        assert [(event.text, event.media_urls) for event in handled] == [
            ("new revision without photo", [])
        ]

    @pytest.mark.asyncio
    async def test_terminal_attachment_failure_drops_revision_without_dispatch(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        handled = []
        downloads = []

        async def download(guid, _meta):
            downloads.append(guid)
            return f"/tmp/{guid}"

        monkeypatch.setattr(adapter, "handle_message", handled.append)
        monkeypatch.setattr(adapter, "_download_attachment", download)

        await adapter._handle_webhook(
            _FakeBlueBubblesRequest(
                self._payload(
                    attachments=[
                        {
                            "guid": "failed-photo",
                            "mimeType": "image/png",
                            "error": 1,
                        }
                    ]
                )
            )
        )
        await asyncio.sleep(0)

        assert handled == []
        assert downloads == []


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
        monkeypatch.delenv("BLUEBUBBLES_WEBHOOK_HOST", raising=False)
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

