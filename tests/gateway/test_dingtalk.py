"""Tests for DingTalk platform adapter."""
import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from gateway.config import Platform, PlatformConfig


# ---------------------------------------------------------------------------
# Requirements check
# ---------------------------------------------------------------------------


class TestDingTalkRequirements:

    def test_returns_false_when_sdk_missing(self, monkeypatch):
        with patch.dict("sys.modules", {"dingtalk_stream": None}):
            monkeypatch.setattr(
                "gateway.platforms.dingtalk.DINGTALK_STREAM_AVAILABLE", False
            )
            from gateway.platforms.dingtalk import check_dingtalk_requirements
            assert check_dingtalk_requirements() is False

    def test_returns_false_when_env_vars_missing(self, monkeypatch):
        monkeypatch.setattr(
            "gateway.platforms.dingtalk.DINGTALK_STREAM_AVAILABLE", True
        )
        monkeypatch.setattr("gateway.platforms.dingtalk.HTTPX_AVAILABLE", True)
        monkeypatch.delenv("DINGTALK_CLIENT_ID", raising=False)
        monkeypatch.delenv("DINGTALK_CLIENT_SECRET", raising=False)
        from gateway.platforms.dingtalk import check_dingtalk_requirements
        assert check_dingtalk_requirements() is False

    def test_returns_true_when_all_available(self, monkeypatch):
        monkeypatch.setattr(
            "gateway.platforms.dingtalk.DINGTALK_STREAM_AVAILABLE", True
        )
        monkeypatch.setattr("gateway.platforms.dingtalk.HTTPX_AVAILABLE", True)
        monkeypatch.setenv("DINGTALK_CLIENT_ID", "test-id")
        monkeypatch.setenv("DINGTALK_CLIENT_SECRET", "test-secret")
        from gateway.platforms.dingtalk import check_dingtalk_requirements
        assert check_dingtalk_requirements() is True


# ---------------------------------------------------------------------------
# Adapter construction
# ---------------------------------------------------------------------------


class TestDingTalkAdapterInit:

    def test_reads_config_from_extra(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        config = PlatformConfig(
            enabled=True,
            extra={"client_id": "cfg-id", "client_secret": "cfg-secret"},
        )
        adapter = DingTalkAdapter(config)
        assert adapter._client_id == "cfg-id"
        assert adapter._client_secret == "cfg-secret"
        assert adapter.name == "Dingtalk"  # base class uses .title()

    def test_falls_back_to_env_vars(self, monkeypatch):
        monkeypatch.setenv("DINGTALK_CLIENT_ID", "env-id")
        monkeypatch.setenv("DINGTALK_CLIENT_SECRET", "env-secret")
        from gateway.platforms.dingtalk import DingTalkAdapter
        config = PlatformConfig(enabled=True)
        adapter = DingTalkAdapter(config)
        assert adapter._client_id == "env-id"
        assert adapter._client_secret == "env-secret"


# ---------------------------------------------------------------------------
# Message text extraction
# ---------------------------------------------------------------------------


class TestExtractText:

    def test_extracts_dict_text(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        msg = MagicMock()
        msg.text = {"content": "  hello world  "}
        msg.rich_text = None
        assert DingTalkAdapter._extract_text(msg) == "hello world"

    def test_extracts_string_text(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        msg = MagicMock()
        msg.text = "plain text"
        msg.rich_text = None
        assert DingTalkAdapter._extract_text(msg) == "plain text"

    def test_falls_back_to_rich_text(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        msg = MagicMock()
        msg.text = ""
        msg.rich_text = [{"text": "part1"}, {"text": "part2"}, {"image": "url"}]
        assert DingTalkAdapter._extract_text(msg) == "part1 part2"

    def test_returns_empty_for_no_content(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        msg = MagicMock()
        msg.text = ""
        msg.rich_text = None
        assert DingTalkAdapter._extract_text(msg) == ""


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:

    def test_first_message_not_duplicate(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        adapter = DingTalkAdapter(PlatformConfig(enabled=True))
        assert adapter._dedup.is_duplicate("msg-1") is False

    def test_second_same_message_is_duplicate(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        adapter = DingTalkAdapter(PlatformConfig(enabled=True))
        adapter._dedup.is_duplicate("msg-1")
        assert adapter._dedup.is_duplicate("msg-1") is True

    def test_different_messages_not_duplicate(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        adapter = DingTalkAdapter(PlatformConfig(enabled=True))
        adapter._dedup.is_duplicate("msg-1")
        assert adapter._dedup.is_duplicate("msg-2") is False

    def test_cache_cleanup_on_overflow(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        adapter = DingTalkAdapter(PlatformConfig(enabled=True))
        max_size = adapter._dedup._max_size
        # Fill beyond max
        for i in range(max_size + 10):
            adapter._dedup.is_duplicate(f"msg-{i}")
        # Cache should have been pruned
        assert len(adapter._dedup._seen) <= max_size + 10


# ---------------------------------------------------------------------------
# Send
# ---------------------------------------------------------------------------


class TestSend:

    @pytest.mark.asyncio
    async def test_send_posts_to_webhook(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        adapter = DingTalkAdapter(PlatformConfig(enabled=True))

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "OK"

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        adapter._http_client = mock_client

        result = await adapter.send(
            "chat-123", "Hello!",
            metadata={"session_webhook": "https://dingtalk.example/webhook"}
        )
        assert result.success is True
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "https://dingtalk.example/webhook"
        payload = call_args[1]["json"]
        assert payload["msgtype"] == "markdown"
        assert payload["markdown"]["title"] == "Hermes"
        assert payload["markdown"]["text"] == "Hello!"

    @pytest.mark.asyncio
    async def test_send_fails_without_webhook(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        adapter = DingTalkAdapter(PlatformConfig(enabled=True))
        adapter._http_client = AsyncMock()

        result = await adapter.send("chat-123", "Hello!")
        assert result.success is False
        assert "session_webhook" in result.error

    @pytest.mark.asyncio
    async def test_send_uses_cached_webhook(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        adapter = DingTalkAdapter(PlatformConfig(enabled=True))

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        adapter._http_client = mock_client
        adapter._session_webhooks["chat-123"] = ("https://cached.example/webhook", 9999999999999)

        result = await adapter.send("chat-123", "Hello!")
        assert result.success is True
        assert mock_client.post.call_args[0][0] == "https://cached.example/webhook"

    @pytest.mark.asyncio
    async def test_send_handles_http_error(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        adapter = DingTalkAdapter(PlatformConfig(enabled=True))

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        adapter._http_client = mock_client

        result = await adapter.send(
            "chat-123", "Hello!",
            metadata={"session_webhook": "https://example/webhook"}
        )
        assert result.success is False
        assert "400" in result.error


# ---------------------------------------------------------------------------
# Connect / disconnect
# ---------------------------------------------------------------------------


class TestConnect:

    @pytest.mark.asyncio
    async def test_disconnect_closes_session_websocket(self):
        from gateway.platforms.dingtalk import DingTalkAdapter

        adapter = DingTalkAdapter(PlatformConfig(enabled=True))
        websocket = AsyncMock()
        blocker = asyncio.Event()

        async def _run_forever():
            try:
                await blocker.wait()
            except asyncio.CancelledError:
                return

        adapter._stream_client = SimpleNamespace(websocket=websocket)
        adapter._stream_task = asyncio.create_task(_run_forever())
        adapter._running = True

        await adapter.disconnect()

        websocket.close.assert_awaited_once()
        assert adapter._stream_task is None

    @pytest.mark.asyncio
    async def test_connect_fails_without_sdk(self, monkeypatch):
        monkeypatch.setattr(
            "gateway.platforms.dingtalk.DINGTALK_STREAM_AVAILABLE", False
        )
        from gateway.platforms.dingtalk import DingTalkAdapter
        adapter = DingTalkAdapter(PlatformConfig(enabled=True))
        result = await adapter.connect()
        assert result is False

    @pytest.mark.asyncio
    async def test_connect_fails_without_credentials(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        adapter = DingTalkAdapter(PlatformConfig(enabled=True))
        adapter._client_id = ""
        adapter._client_secret = ""
        result = await adapter.connect()
        assert result is False

    @pytest.mark.asyncio
    async def test_disconnect_cleans_up(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        adapter = DingTalkAdapter(PlatformConfig(enabled=True))
        adapter._session_webhooks["a"] = "http://x"
        adapter._dedup._seen["b"] = 1.0
        adapter._http_client = AsyncMock()
        adapter._stream_task = None

        await adapter.disconnect()
        assert len(adapter._session_webhooks) == 0
        assert len(adapter._dedup._seen) == 0
        assert adapter._http_client is None


# ---------------------------------------------------------------------------
# Platform enum
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# SDK compatibility regression tests (dingtalk-stream >= 0.20 / 0.24)
# ---------------------------------------------------------------------------


class TestWebhookDomainAllowlist:
    """Guard the webhook origin allowlist against regression.

    The SDK started returning reply webhooks on ``oapi.dingtalk.com`` in
    addition to ``api.dingtalk.com``. Both must be accepted, and hostile
    lookalikes must still be rejected (SSRF defence-in-depth).
    """

    def test_api_domain_accepted(self):
        from gateway.platforms.dingtalk import _DINGTALK_WEBHOOK_RE
        assert _DINGTALK_WEBHOOK_RE.match(
            "https://api.dingtalk.com/robot/send?access_token=x"
        )

    def test_oapi_domain_accepted(self):
        from gateway.platforms.dingtalk import _DINGTALK_WEBHOOK_RE
        assert _DINGTALK_WEBHOOK_RE.match(
            "https://oapi.dingtalk.com/robot/send?access_token=x"
        )

    def test_http_rejected(self):
        from gateway.platforms.dingtalk import _DINGTALK_WEBHOOK_RE
        assert not _DINGTALK_WEBHOOK_RE.match("http://api.dingtalk.com/robot/send")

    def test_suffix_attack_rejected(self):
        from gateway.platforms.dingtalk import _DINGTALK_WEBHOOK_RE
        assert not _DINGTALK_WEBHOOK_RE.match(
            "https://api.dingtalk.com.evil.example/"
        )

    def test_unsanctioned_subdomain_rejected(self):
        from gateway.platforms.dingtalk import _DINGTALK_WEBHOOK_RE
        # Only api.* and oapi.* are allowed — e.g. eapi.dingtalk.com must not slip through
        assert not _DINGTALK_WEBHOOK_RE.match("https://eapi.dingtalk.com/robot/send")


class TestHandlerProcessIsAsync:
    """dingtalk-stream >= 0.20 requires ``process`` to be a coroutine."""

    def test_process_is_coroutine_function(self):
        from gateway.platforms.dingtalk import _IncomingHandler
        assert asyncio.iscoroutinefunction(_IncomingHandler.process)


class TestExtractText:
    """_extract_text must handle both legacy and current SDK payload shapes.

    Before SDK 0.20 ``message.text`` was a ``dict`` with a ``content`` key.
    From 0.20 onward it is a ``TextContent`` dataclass whose ``__str__``
    returns ``"TextContent(content=...)"`` — falling back to ``str(text)``
    leaks that repr into the agent's input.
    """

    def test_text_as_dict_legacy(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        msg = MagicMock()
        msg.text = {"content": "hello world"}
        msg.rich_text_content = None
        msg.rich_text = None
        assert DingTalkAdapter._extract_text(msg) == "hello world"

    def test_text_as_textcontent_object(self):
        """SDK >= 0.20 shape: object with ``.content`` attribute."""
        from gateway.platforms.dingtalk import DingTalkAdapter

        class FakeTextContent:
            content = "hello from new sdk"

            def __str__(self):  # mimic real SDK repr
                return f"TextContent(content={self.content})"

        msg = MagicMock()
        msg.text = FakeTextContent()
        msg.rich_text_content = None
        msg.rich_text = None
        result = DingTalkAdapter._extract_text(msg)
        assert result == "hello from new sdk"
        assert "TextContent(" not in result

    def test_text_content_attr_with_empty_string(self):
        from gateway.platforms.dingtalk import DingTalkAdapter

        class FakeTextContent:
            content = ""

        msg = MagicMock()
        msg.text = FakeTextContent()
        msg.rich_text_content = None
        msg.rich_text = None
        assert DingTalkAdapter._extract_text(msg) == ""

    def test_rich_text_content_new_shape(self):
        """SDK >= 0.20 exposes rich text as ``message.rich_text_content.rich_text_list``."""
        from gateway.platforms.dingtalk import DingTalkAdapter

        class FakeRichText:
            rich_text_list = [{"text": "hello "}, {"text": "world"}]

        msg = MagicMock()
        msg.text = None
        msg.rich_text_content = FakeRichText()
        msg.rich_text = None
        result = DingTalkAdapter._extract_text(msg)
        assert "hello" in result and "world" in result

    def test_rich_text_legacy_shape(self):
        """Legacy ``message.rich_text`` list remains supported."""
        from gateway.platforms.dingtalk import DingTalkAdapter
        msg = MagicMock()
        msg.text = None
        msg.rich_text_content = None
        msg.rich_text = [{"text": "legacy "}, {"text": "rich"}]
        result = DingTalkAdapter._extract_text(msg)
        assert "legacy" in result and "rich" in result

    def test_empty_message(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        msg = MagicMock()
        msg.text = None
        msg.rich_text_content = None
        msg.rich_text = None
        assert DingTalkAdapter._extract_text(msg) == ""


# ---------------------------------------------------------------------------
# Group gating — require_mention + allowed_users (parity with other platforms)
# ---------------------------------------------------------------------------


def _make_gating_adapter(monkeypatch, *, extra=None, env=None):
    """Build a DingTalkAdapter with only the gating fields populated.

    Clears every DINGTALK_* gating env var before applying the caller's
    overrides so individual tests stay isolated.
    """
    for key in (
        "DINGTALK_REQUIRE_MENTION",
        "DINGTALK_MENTION_PATTERNS",
        "DINGTALK_FREE_RESPONSE_CHATS",
        "DINGTALK_ALLOWED_USERS",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)
    from gateway.platforms.dingtalk import DingTalkAdapter
    return DingTalkAdapter(PlatformConfig(enabled=True, extra=extra or {}))


class TestAllowedUsersGate:

    def test_empty_allowlist_allows_everyone(self, monkeypatch):
        adapter = _make_gating_adapter(monkeypatch)
        assert adapter._is_user_allowed("anyone", "any-staff") is True

    def test_wildcard_allowlist_allows_everyone(self, monkeypatch):
        adapter = _make_gating_adapter(monkeypatch, extra={"allowed_users": ["*"]})
        assert adapter._is_user_allowed("anyone", "any-staff") is True

    def test_matches_sender_id_case_insensitive(self, monkeypatch):
        adapter = _make_gating_adapter(
            monkeypatch, extra={"allowed_users": ["SenderABC"]}
        )
        assert adapter._is_user_allowed("senderabc", "") is True

    def test_matches_staff_id(self, monkeypatch):
        adapter = _make_gating_adapter(
            monkeypatch, extra={"allowed_users": ["staff_1234"]}
        )
        assert adapter._is_user_allowed("", "staff_1234") is True

    def test_rejects_unknown_user(self, monkeypatch):
        adapter = _make_gating_adapter(
            monkeypatch, extra={"allowed_users": ["staff_1234"]}
        )
        assert adapter._is_user_allowed("other-sender", "other-staff") is False

    def test_env_var_csv_populates_allowlist(self, monkeypatch):
        adapter = _make_gating_adapter(
            monkeypatch, env={"DINGTALK_ALLOWED_USERS": "alice,bob,carol"}
        )
        assert adapter._is_user_allowed("alice", "") is True
        assert adapter._is_user_allowed("dave", "") is False


class TestMentionPatterns:

    def test_empty_patterns_list(self, monkeypatch):
        adapter = _make_gating_adapter(monkeypatch)
        assert adapter._mention_patterns == []
        assert adapter._message_matches_mention_patterns("anything") is False

    def test_pattern_matches_text(self, monkeypatch):
        adapter = _make_gating_adapter(
            monkeypatch, extra={"mention_patterns": ["^hermes"]}
        )
        assert adapter._message_matches_mention_patterns("hermes please help") is True
        assert adapter._message_matches_mention_patterns("please hermes help") is False

    def test_pattern_is_case_insensitive(self, monkeypatch):
        adapter = _make_gating_adapter(
            monkeypatch, extra={"mention_patterns": ["^hermes"]}
        )
        assert adapter._message_matches_mention_patterns("HERMES help") is True

    def test_invalid_regex_is_skipped_not_raised(self, monkeypatch):
        adapter = _make_gating_adapter(
            monkeypatch,
            extra={"mention_patterns": ["[unclosed", "^valid"]},
        )
        # Invalid pattern dropped, valid one kept
        assert len(adapter._mention_patterns) == 1
        assert adapter._message_matches_mention_patterns("valid trigger") is True

    def test_env_var_json_populates_patterns(self, monkeypatch):
        adapter = _make_gating_adapter(
            monkeypatch,
            env={"DINGTALK_MENTION_PATTERNS": '["^bot", "^assistant"]'},
        )
        assert len(adapter._mention_patterns) == 2
        assert adapter._message_matches_mention_patterns("bot ping") is True

    def test_env_var_newline_fallback_when_not_json(self, monkeypatch):
        adapter = _make_gating_adapter(
            monkeypatch,
            env={"DINGTALK_MENTION_PATTERNS": "^bot\n^assistant"},
        )
        assert len(adapter._mention_patterns) == 2


class TestShouldProcessMessage:

    def test_dm_always_accepted(self, monkeypatch):
        adapter = _make_gating_adapter(
            monkeypatch, extra={"require_mention": True}
        )
        msg = MagicMock(is_in_at_list=False)
        assert adapter._should_process_message(msg, "hi", is_group=False, chat_id="dm1") is True

    def test_group_rejected_when_require_mention_and_no_trigger(self, monkeypatch):
        adapter = _make_gating_adapter(
            monkeypatch, extra={"require_mention": True}
        )
        msg = MagicMock(is_in_at_list=False)
        assert adapter._should_process_message(msg, "hi", is_group=True, chat_id="grp1") is False

    def test_group_accepted_when_require_mention_disabled(self, monkeypatch):
        adapter = _make_gating_adapter(
            monkeypatch, extra={"require_mention": False}
        )
        msg = MagicMock(is_in_at_list=False)
        assert adapter._should_process_message(msg, "hi", is_group=True, chat_id="grp1") is True

    def test_group_accepted_when_bot_is_mentioned(self, monkeypatch):
        adapter = _make_gating_adapter(
            monkeypatch, extra={"require_mention": True}
        )
        msg = MagicMock(is_in_at_list=True)
        assert adapter._should_process_message(msg, "hi", is_group=True, chat_id="grp1") is True

    def test_group_accepted_when_text_matches_wake_word(self, monkeypatch):
        adapter = _make_gating_adapter(
            monkeypatch,
            extra={"require_mention": True, "mention_patterns": ["^hermes"]},
        )
        msg = MagicMock(is_in_at_list=False)
        assert adapter._should_process_message(msg, "hermes help", is_group=True, chat_id="grp1") is True

    def test_group_accepted_when_chat_in_free_response_list(self, monkeypatch):
        adapter = _make_gating_adapter(
            monkeypatch,
            extra={"require_mention": True, "free_response_chats": ["grp1"]},
        )
        msg = MagicMock(is_in_at_list=False)
        assert adapter._should_process_message(msg, "hi", is_group=True, chat_id="grp1") is True
        # Different group still blocked
        assert adapter._should_process_message(msg, "hi", is_group=True, chat_id="grp2") is False


# ---------------------------------------------------------------------------
# _IncomingHandler.process — session_webhook extraction & fire-and-forget
# ---------------------------------------------------------------------------


class TestIncomingHandlerProcess:
    """Verify that _IncomingHandler.process correctly converts callback data
    and dispatches message processing as a background task (fire-and-forget)
    so the SDK ACK is returned immediately."""

    @pytest.mark.asyncio
    async def test_process_extracts_session_webhook(self):
        """session_webhook must be populated from callback data."""
        from gateway.platforms.dingtalk import _IncomingHandler, DingTalkAdapter

        adapter = DingTalkAdapter(PlatformConfig(enabled=True))
        adapter._on_message = AsyncMock()
        handler = _IncomingHandler(adapter, asyncio.get_running_loop())

        callback = MagicMock()
        callback.data = {
            "msgtype": "text",
            "text": {"content": "hello"},
            "senderId": "user1",
            "conversationId": "conv1",
            "sessionWebhook": "https://oapi.dingtalk.com/robot/sendBySession?session=abc",
            "msgId": "msg-001",
        }

        result = await handler.process(callback)
        # Should return ACK immediately (STATUS_OK = 200)
        assert result[0] == 200

        # Let the background task run
        await asyncio.sleep(0.05)

        # _on_message should have been called with a ChatbotMessage
        adapter._on_message.assert_called_once()
        chatbot_msg = adapter._on_message.call_args[0][0]
        assert chatbot_msg.session_webhook == "https://oapi.dingtalk.com/robot/sendBySession?session=abc"

    @pytest.mark.asyncio
    async def test_process_fallback_session_webhook_when_from_dict_misses_it(self):
        """If ChatbotMessage.from_dict does not map sessionWebhook (e.g. SDK
        version mismatch), the handler should fall back to extracting it
        directly from the raw data dict."""
        from gateway.platforms.dingtalk import _IncomingHandler, DingTalkAdapter

        adapter = DingTalkAdapter(PlatformConfig(enabled=True))
        adapter._on_message = AsyncMock()
        handler = _IncomingHandler(adapter, asyncio.get_running_loop())

        callback = MagicMock()
        # Use a key that from_dict might not recognise in some SDK versions
        callback.data = {
            "msgtype": "text",
            "text": {"content": "hi"},
            "senderId": "user2",
            "conversationId": "conv2",
            "session_webhook": "https://oapi.dingtalk.com/robot/sendBySession?session=def",
            "msgId": "msg-002",
        }

        await handler.process(callback)
        await asyncio.sleep(0.05)

        adapter._on_message.assert_called_once()
        chatbot_msg = adapter._on_message.call_args[0][0]
        assert chatbot_msg.session_webhook == "https://oapi.dingtalk.com/robot/sendBySession?session=def"

    @pytest.mark.asyncio
    async def test_process_returns_ack_immediately(self):
        """process() must not block on _on_message — it should return
        the ACK tuple before the message is fully processed."""
        from gateway.platforms.dingtalk import _IncomingHandler, DingTalkAdapter

        processing_started = asyncio.Event()
        processing_gate = asyncio.Event()

        async def slow_on_message(msg):
            processing_started.set()
            await processing_gate.wait()  # Block until we release

        adapter = DingTalkAdapter(PlatformConfig(enabled=True))
        adapter._on_message = slow_on_message
        handler = _IncomingHandler(adapter, asyncio.get_running_loop())

        callback = MagicMock()
        callback.data = {
            "msgtype": "text",
            "text": {"content": "test"},
            "senderId": "u",
            "conversationId": "c",
            "sessionWebhook": "https://oapi.dingtalk.com/x",
            "msgId": "m",
        }

        # process() should return immediately even though _on_message blocks
        result = await handler.process(callback)
        assert result[0] == 200

        # Clean up: release the gate so the background task finishes
        processing_gate.set()
        await asyncio.sleep(0.05)


# ---------------------------------------------------------------------------
# Text extraction — mention preservation + platform sanity
# ---------------------------------------------------------------------------

class TestExtractTextMentions:

    def test_preserves_at_mentions_in_text(self):
        """@mentions are routing signals (via isInAtList), not text to strip.

        Stripping all @handles collateral-damages emails, SSH URLs, and
        literal references the user wrote.
        """
        from gateway.platforms.dingtalk import DingTalkAdapter
        cases = [
            ("@bot hello", "@bot hello"),
            ("contact alice@example.com", "contact alice@example.com"),
            ("git@github.com:foo/bar.git", "git@github.com:foo/bar.git"),
            ("what does @openai think", "what does @openai think"),
            ("@机器人 转发给 @老王", "@机器人 转发给 @老王"),
        ]
        for text, expected in cases:
            msg = MagicMock()
            msg.text = text
            msg.rich_text = None
            msg.rich_text_content = None
            assert DingTalkAdapter._extract_text(msg) == expected, (
                f"mangled: {text!r} -> {DingTalkAdapter._extract_text(msg)!r}"
            )

    def test_dingtalk_in_platform_enum(self):
        assert Platform.DINGTALK.value == "dingtalk"


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Concurrency — chat-scoped message context
# ---------------------------------------------------------------------------


class TestMessageContextIsolation:

    def test_contexts_keyed_by_chat_id(self):
        """Two concurrent chats must not clobber each other's context."""
        from gateway.platforms.dingtalk import DingTalkAdapter
        adapter = DingTalkAdapter(PlatformConfig(enabled=True))

        msg_a = MagicMock(conversation_id="chat-A", sender_staff_id="user-A")
        msg_b = MagicMock(conversation_id="chat-B", sender_staff_id="user-B")
        adapter._message_contexts["chat-A"] = msg_a
        adapter._message_contexts["chat-B"] = msg_b

        assert adapter._message_contexts["chat-A"] is msg_a
        assert adapter._message_contexts["chat-B"] is msg_b






# ---------------------------------------------------------------------------
# Card lifecycle: finalize via metadata["streaming"]
# ---------------------------------------------------------------------------


class TestCardLifecycle:

    @pytest.fixture
    def adapter_with_card(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        a = DingTalkAdapter(PlatformConfig(
            enabled=True,
            extra={"card_template_id": "tmpl-1"},
        ))
        a._card_sdk = MagicMock()
        a._card_sdk.create_card_with_options_async = AsyncMock()
        a._card_sdk.deliver_card_with_options_async = AsyncMock()
        a._card_sdk.streaming_update_with_options_async = AsyncMock()
        a._http_client = AsyncMock()
        a._get_access_token = AsyncMock(return_value="token")
        # Minimal message context
        msg = MagicMock(
            conversation_id="chat-1",
            conversation_type="1",
            sender_staff_id="staff-1",
            message_id="user-msg-1",
        )
        a._message_contexts["chat-1"] = msg
        a._session_webhooks["chat-1"] = (
            "https://api.dingtalk.com/x", 9999999999999,
        )
        return a

    @pytest.mark.asyncio
    async def test_final_reply_finalizes_card(self, adapter_with_card):
        """send(reply_to=...) creates a closed card (final response path)."""
        a = adapter_with_card
        result = await a.send("chat-1", "Hello", reply_to="user-msg-1")
        assert result.success
        call = a._card_sdk.streaming_update_with_options_async.call_args
        assert call[0][0].is_finalize is True
        # Not tracked as streaming — it's already closed.
        assert "chat-1" not in a._streaming_cards

    @pytest.mark.asyncio
    async def test_intermediate_send_stays_streaming(self, adapter_with_card):
        """send() without reply_to creates an OPEN card (tool progress /
        commentary / streaming first chunk).  No flicker closed→streaming
        when edit_message follows."""
        a = adapter_with_card
        result = await a.send("chat-1", "💻 terminal: ls")
        assert result.success
        call = a._card_sdk.streaming_update_with_options_async.call_args
        assert call[0][0].is_finalize is False
        # Tracked for sibling cleanup.
        assert result.message_id in a._streaming_cards.get("chat-1", {})

    @pytest.mark.asyncio
    async def test_done_fires_only_when_reply_to_is_set(self, adapter_with_card):
        """reply_to distinguishes final response (base.py) from tool-progress
        sends (run.py).  Done must only fire for the former."""
        a = adapter_with_card
        fired: list[str] = []
        a._fire_done_reaction = lambda cid: fired.append(cid)

        # Tool-progress / commentary path: no reply_to — no Done.
        await a.send("chat-1", "tool line")
        assert fired == []

        # Final response path: reply_to set — Done fires.
        await a.send("chat-1", "final", reply_to="user-msg-1")
        assert fired == ["chat-1"]

    @pytest.mark.asyncio
    async def test_edit_message_finalize_fires_done(self, adapter_with_card):
        """Stream consumer's final edit_message(finalize=True) fires Done."""
        a = adapter_with_card
        fired: list[str] = []
        a._fire_done_reaction = lambda cid: fired.append(cid)

        await a.send("chat-1", "initial")
        # Reopen via edit_message(finalize=False) then close.
        await a.edit_message(
            chat_id="chat-1", message_id="track-X",
            content="streaming...", finalize=False,
        )
        await a.edit_message(
            chat_id="chat-1", message_id="track-X",
            content="final", finalize=True,
        )
        assert "chat-1" in fired

    @pytest.mark.asyncio
    async def test_edit_message_finalize_false_tracks_sibling(self, adapter_with_card):
        """After edit_message(finalize=False), card is tracked as open."""
        a = adapter_with_card
        await a.edit_message(
            chat_id="chat-1", message_id="track-1",
            content="partial", finalize=False,
        )
        assert "chat-1" in a._streaming_cards
        assert a._streaming_cards["chat-1"].get("track-1") == "partial"

    @pytest.mark.asyncio
    async def test_next_send_auto_closes_sibling_streaming_cards(
        self, adapter_with_card,
    ):
        """Tool-progress card left open (send without reply_to + edits) must
        be auto-closed when the final-reply send arrives."""
        a = adapter_with_card
        # First tool: intermediate send — card stays open.
        r1 = await a.send("chat-1", "💻 tool1")
        # Second tool: edit_message(finalize=False) — keeps streaming.
        await a.edit_message(
            chat_id="chat-1", message_id=r1.message_id,
            content="💻 tool1\n💻 tool2", finalize=False,
        )
        assert r1.message_id in a._streaming_cards.get("chat-1", {})
        a._card_sdk.streaming_update_with_options_async.reset_mock()

        # Final response send auto-closes the sibling.
        await a.send("chat-1", "final answer", reply_to="user-msg")

        calls = a._card_sdk.streaming_update_with_options_async.call_args_list
        assert len(calls) >= 2
        # First call was the sibling close with last-seen tool-progress content.
        first_req = calls[0][0][0]
        assert first_req.out_track_id == r1.message_id
        assert first_req.is_finalize is True
        assert "tool1" in first_req.content
        # Streaming tracking is cleared after close.
        assert "chat-1" not in a._streaming_cards

    @pytest.mark.asyncio
    async def test_edit_message_requires_message_id(self, adapter_with_card):
        a = adapter_with_card
        result = await a.edit_message(
            chat_id="chat-1", message_id="", content="x", finalize=True,
        )
        assert result.success is False
        a._card_sdk.streaming_update_with_options_async.assert_not_called()

    def test_fire_done_reaction_is_idempotent(self, adapter_with_card):
        a = adapter_with_card
        captured = []
        def _capture(coro):
            captured.append(coro)
        a._spawn_bg = _capture

        a._fire_done_reaction("chat-1")
        a._fire_done_reaction("chat-1")
        assert len(captured) == 1
        captured[0].close()



# ---------------------------------------------------------------------------
# AI Card Tests
# ---------------------------------------------------------------------------

class TestDingTalkAdapterAICards:
    @pytest.fixture
    def config(self):
        return PlatformConfig(
            enabled=True,
            extra={
                "client_id": "test_id",
                "client_secret": "test_secret",
                "card_template_id": "test_card_template",
            },
        )

    @pytest.fixture
    def mock_stream_client(self):
        client = MagicMock()
        client.get_access_token = MagicMock(return_value="test_token")
        return client

    @pytest.fixture
    def mock_http_client(self):
        return AsyncMock()

    @pytest.fixture
    def mock_message(self):
        msg = MagicMock()
        msg.message_id = "test_msg_id"
        msg.conversation_id = "test_conv_id"
        msg.conversation_type = "1"
        msg.sender_id = "sender1"
        msg.sender_nick = "Test User"
        msg.sender_staff_id = "staff1"
        msg.text = MagicMock(content="Hello")
        msg.session_webhook = "https://api.dingtalk.com/robot/sendBySession?session=test"
        msg.session_webhook_expired_time = 999999999999
        msg.create_at = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
        msg.at_users = []
        return msg

    @pytest.mark.asyncio
    async def test_send_uses_ai_card_if_configured(self, config, mock_stream_client, mock_http_client, mock_message):
        from gateway.platforms.dingtalk import DingTalkAdapter

        adapter = DingTalkAdapter(config)
        adapter._stream_client = mock_stream_client
        adapter._http_client = mock_http_client
        adapter._message_contexts["test_conv_id"] = mock_message
        adapter._session_webhooks = {"test_conv_id": ("https://api.dingtalk.com/robot/sendBySession?session=test", 9999999999999)}
        adapter._card_template_id = "test_card_template"

        # Mock the card SDK with proper async methods
        mock_card_sdk = MagicMock()
        mock_card_sdk.create_card_with_options_async = AsyncMock()
        mock_card_sdk.deliver_card_with_options_async = AsyncMock()
        mock_card_sdk.streaming_update_with_options_async = AsyncMock()
        adapter._card_sdk = mock_card_sdk

        # Mock access token
        adapter._get_access_token = AsyncMock(return_value="test_token")

        result = await adapter.send("test_conv_id", "Hello World")

        mock_card_sdk.create_card_with_options_async.assert_called_once()
        mock_card_sdk.deliver_card_with_options_async.assert_called_once()
        mock_card_sdk.streaming_update_with_options_async.assert_called_once()
        assert result.success is True
