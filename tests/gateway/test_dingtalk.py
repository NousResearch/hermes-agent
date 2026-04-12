"""Tests for DingTalk platform adapter."""
import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from gateway.config import Platform, PlatformConfig


# ---------------------------------------------------------------------------
# Session webhook allowlist (SSRF-safe DingTalk reply URLs)
# ---------------------------------------------------------------------------


class TestSessionWebhookAllowed:

    def test_allows_api_and_oapi_https(self):
        from gateway.platforms.dingtalk import _session_webhook_is_allowed
        assert _session_webhook_is_allowed(
            "https://api.dingtalk.com/v1.0/gateway/robot/stream/callback/xxx"
        )
        assert _session_webhook_is_allowed(
            "https://oapi.dingtalk.com/robot/sendBySession?session=abc"
        )

    def test_rejects_non_dingtalk_and_non_https(self):
        from gateway.platforms.dingtalk import _session_webhook_is_allowed
        assert _session_webhook_is_allowed("https://evil.com/callback") is False
        assert _session_webhook_is_allowed("http://api.dingtalk.com/x") is False
        assert _session_webhook_is_allowed("https://not-api.dingtalk.com/x") is False
        assert _session_webhook_is_allowed("") is False


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
# Stream envelope → ChatbotMessage
# ---------------------------------------------------------------------------


class TestNormalizeStreamMessage:
    """Test that _IncomingHandler.process converts CallbackMessage → ChatbotMessage."""

    def test_unwraps_callback_message_data(self):
        from types import SimpleNamespace

        from gateway.platforms.dingtalk import DingTalkAdapter

        inner = {
            "msgId": "mid-1",
            "conversationId": "conv-1",
            "conversationType": "1",
            "senderId": "uid-1",
            "senderNick": "Tester",
            "createAt": "1700000000000",
            "sessionWebhook": "https://api.dingtalk.com/v1.0/gateway/robot/stream/callback/x",
            "msgtype": "text",
            "text": {"content": "hello from stream"},
        }
        # Simulate what _IncomingHandler.process does: read .data and call from_dict
        wrapped = SimpleNamespace(data=inner)
        data = getattr(wrapped, "data", None)
        assert isinstance(data, dict) and data
        from dingtalk_stream import ChatbotMessage
        out = ChatbotMessage.from_dict(data)
        assert out.message_id == "mid-1"
        assert out.conversation_id == "conv-1"
        assert DingTalkAdapter._extract_text(out) == "hello from stream"

    def test_leaves_plain_chatbot_message_unchanged(self):
        from unittest.mock import MagicMock

        from gateway.platforms.dingtalk import DingTalkAdapter

        msg = MagicMock()
        msg.text = {"content": "plain"}
        msg.rich_text_content = None
        msg.message_type = None
        msg.image_content = None
        # If message has no .data dict, it should be passed through as-is
        assert DingTalkAdapter._extract_text(msg) == "plain"


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
        msg.rich_text_content = None
        msg.message_type = None
        msg.image_content = None
        assert DingTalkAdapter._extract_text(msg) == ""

    def test_extracts_textcontent_object(self):
        from gateway.platforms.dingtalk import DingTalkAdapter

        class TC:
            content = "  from sdk  "

        msg = MagicMock()
        msg.text = TC()
        msg.rich_text_content = None
        msg.message_type = "text"
        msg.image_content = None
        assert DingTalkAdapter._extract_text(msg) == "from sdk"

    def test_extracts_rich_text_content_list(self):
        from gateway.platforms.dingtalk import DingTalkAdapter

        class RTC:
            rich_text_list = [{"text": "hello"}, {"pic": 1}, {"text": "world"}]

        msg = MagicMock()
        msg.text = None
        msg.rich_text_content = RTC()
        msg.message_type = "richText"
        msg.image_content = None
        assert DingTalkAdapter._extract_text(msg) == "hello world"

    def test_picture_message_placeholder(self):
        from gateway.platforms.dingtalk import DingTalkAdapter

        class Img:
            download_code = "dl123"

        msg = MagicMock()
        msg.text = None
        msg.rich_text_content = None
        msg.message_type = "picture"
        msg.image_content = Img()
        assert DingTalkAdapter._extract_text(msg) == "[图片]"


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:

    def test_first_message_not_duplicate(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        adapter = DingTalkAdapter(PlatformConfig(enabled=True))
        assert adapter._is_duplicate("msg-1") is False

    def test_second_same_message_is_duplicate(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        adapter = DingTalkAdapter(PlatformConfig(enabled=True))
        adapter._is_duplicate("msg-1")
        assert adapter._is_duplicate("msg-1") is True

    def test_different_messages_not_duplicate(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        adapter = DingTalkAdapter(PlatformConfig(enabled=True))
        adapter._is_duplicate("msg-1")
        assert adapter._is_duplicate("msg-2") is False

    def test_cache_cleanup_on_overflow(self):
        from gateway.platforms.dingtalk import DingTalkAdapter, DEDUP_MAX_SIZE
        adapter = DingTalkAdapter(PlatformConfig(enabled=True))
        # Fill beyond max
        for i in range(DEDUP_MAX_SIZE + 10):
            adapter._is_duplicate(f"msg-{i}")
        # Cache should have been pruned
        assert len(adapter._seen_messages) <= DEDUP_MAX_SIZE + 10


# ---------------------------------------------------------------------------
# Markdown chunk splitting
# ---------------------------------------------------------------------------


class TestSplitMarkdownChunks:

    def test_short_text_not_split(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        chunks = DingTalkAdapter._split_markdown_chunks("hello", 100)
        assert chunks == ["hello"]

    def test_long_text_split_on_lines(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        text = "\n".join(f"line {i}" for i in range(100))
        chunks = DingTalkAdapter._split_markdown_chunks(text, 200)
        assert len(chunks) > 1
        rejoined = "\n".join(chunks)
        assert rejoined == text

    def test_code_fence_preserved_across_chunks(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        text = "before\n```\n" + "x\n" * 100 + "```\nafter"
        chunks = DingTalkAdapter._split_markdown_chunks(text, 200)
        assert len(chunks) > 1
        # Each chunk that starts mid-code should begin with ```
        for chunk in chunks[1:]:
            if "x" in chunk and not chunk.startswith("after"):
                assert chunk.startswith("```")

    def test_empty_text(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        assert DingTalkAdapter._split_markdown_chunks("", 100) == [""]


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
        adapter._session_webhooks["chat-123"] = "https://cached.example/webhook"

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
        adapter._seen_messages["b"] = 1.0
        adapter._http_client = AsyncMock()
        adapter._stream_task = None

        await adapter.disconnect()
        assert len(adapter._session_webhooks) == 0
        assert len(adapter._seen_messages) == 0
        assert adapter._http_client is None


# ---------------------------------------------------------------------------
# Platform enum
# ---------------------------------------------------------------------------


class TestPlatformEnum:

    def test_dingtalk_in_platform_enum(self):
        assert Platform.DINGTALK.value == "dingtalk"


# ---------------------------------------------------------------------------
# Access token management
# ---------------------------------------------------------------------------


class TestAccessToken:

    @pytest.mark.asyncio
    async def test_caches_token(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        adapter = DingTalkAdapter(PlatformConfig(enabled=True, extra={"client_id": "id", "client_secret": "sec"}))

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"accessToken": "tok-123", "expireIn": 7200}
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        adapter._http_client = mock_client

        token = await adapter._get_access_token()
        assert token == "tok-123"
        # Second call should use cache, not make another request
        token2 = await adapter._get_access_token()
        assert token2 == "tok-123"
        assert mock_client.post.call_count == 1

    @pytest.mark.asyncio
    async def test_returns_none_on_http_error(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        adapter = DingTalkAdapter(PlatformConfig(enabled=True, extra={"client_id": "id", "client_secret": "sec"}))

        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        adapter._http_client = mock_client

        token = await adapter._get_access_token()
        assert token is None

    @pytest.mark.asyncio
    async def test_returns_none_without_http_client(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        adapter = DingTalkAdapter(PlatformConfig(enabled=True))
        adapter._http_client = None
        token = await adapter._get_access_token()
        assert token is None


# ---------------------------------------------------------------------------
# Emotion reaction API
# ---------------------------------------------------------------------------


class TestEmotionApi:

    @pytest.mark.asyncio
    async def test_attach_succeeds(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        adapter = DingTalkAdapter(PlatformConfig(enabled=True, extra={"client_id": "bot1", "client_secret": "sec"}))
        adapter._access_token = "tok"
        adapter._access_token_expires = 9999999999.0

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        adapter._http_client = mock_client

        ok = await adapter._emotion_api("reply", "msg-1", "conv-1", [0])
        assert ok is True
        call_args = mock_client.post.call_args
        assert "emotion/reply" in call_args[0][0]
        payload = call_args[1]["json"]
        assert payload["robotCode"] == "bot1"
        assert payload["openMsgId"] == "msg-1"
        assert payload["openConversationId"] == "conv-1"

    @pytest.mark.asyncio
    async def test_recall_succeeds(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        adapter = DingTalkAdapter(PlatformConfig(enabled=True, extra={"client_id": "bot1", "client_secret": "sec"}))
        adapter._access_token = "tok"
        adapter._access_token_expires = 9999999999.0

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        adapter._http_client = mock_client

        ok = await adapter._emotion_api("recall", "msg-1", "conv-1", [0])
        assert ok is True
        assert "emotion/recall" in mock_client.post.call_args[0][0]

    @pytest.mark.asyncio
    async def test_returns_false_without_token(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        adapter = DingTalkAdapter(PlatformConfig(enabled=True))
        adapter._http_client = AsyncMock()
        ok = await adapter._emotion_api("reply", "msg-1", "conv-1", [0])
        assert ok is False


# ---------------------------------------------------------------------------
# Reactions enabled config
# ---------------------------------------------------------------------------


class TestReactionsEnabled:

    def test_enabled_by_default(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        adapter = DingTalkAdapter(PlatformConfig(enabled=True))
        assert adapter._reactions_enabled() is True

    def test_disabled_via_env(self, monkeypatch):
        monkeypatch.setenv("DINGTALK_REACTIONS", "false")
        from gateway.platforms.dingtalk import DingTalkAdapter
        adapter = DingTalkAdapter(PlatformConfig(enabled=True))
        assert adapter._reactions_enabled() is False

    def test_disabled_via_config(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        adapter = DingTalkAdapter(PlatformConfig(enabled=True, extra={"reactions": "false"}))
        assert adapter._reactions_enabled() is False

    def test_config_overrides_env(self, monkeypatch):
        monkeypatch.setenv("DINGTALK_REACTIONS", "false")
        from gateway.platforms.dingtalk import DingTalkAdapter
        adapter = DingTalkAdapter(PlatformConfig(enabled=True, extra={"reactions": "true"}))
        assert adapter._reactions_enabled() is True


# ---------------------------------------------------------------------------
# Processing lifecycle hooks
# ---------------------------------------------------------------------------


class TestProcessingHooks:

    @pytest.mark.asyncio
    async def test_on_processing_start_calls_emotion_api(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        adapter = DingTalkAdapter(PlatformConfig(enabled=True, extra={"client_id": "bot1", "client_secret": "sec"}))
        adapter._emotion_api = AsyncMock(return_value=True)

        raw = MagicMock()
        raw.message_id = "msg-1"
        raw.conversation_id = "conv-1"
        event = MagicMock()
        event.raw_message = raw

        await adapter.on_processing_start(event)
        adapter._emotion_api.assert_called_once_with("reply", "msg-1", "conv-1", [0, 0.4, 1.2])

    @pytest.mark.asyncio
    async def test_on_processing_complete_calls_recall(self):
        from gateway.platforms.dingtalk import DingTalkAdapter, ProcessingOutcome
        adapter = DingTalkAdapter(PlatformConfig(enabled=True, extra={"client_id": "bot1", "client_secret": "sec"}))
        adapter._emotion_api = AsyncMock(return_value=True)

        raw = MagicMock()
        raw.message_id = "msg-1"
        raw.conversation_id = "conv-1"
        event = MagicMock()
        event.raw_message = raw

        await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)
        adapter._emotion_api.assert_called_once_with("recall", "msg-1", "conv-1", [0, 1.5, 5.0])

    @pytest.mark.asyncio
    async def test_hooks_noop_when_disabled(self, monkeypatch):
        monkeypatch.setenv("DINGTALK_REACTIONS", "false")
        from gateway.platforms.dingtalk import DingTalkAdapter, ProcessingOutcome
        adapter = DingTalkAdapter(PlatformConfig(enabled=True))
        adapter._emotion_api = AsyncMock()

        event = MagicMock()
        await adapter.on_processing_start(event)
        await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)
        adapter._emotion_api.assert_not_called()

    @pytest.mark.asyncio
    async def test_hooks_noop_when_no_msg_id(self):
        from gateway.platforms.dingtalk import DingTalkAdapter, ProcessingOutcome
        adapter = DingTalkAdapter(PlatformConfig(enabled=True))
        adapter._emotion_api = AsyncMock()

        raw = MagicMock(spec=[])  # no attributes
        event = MagicMock()
        event.raw_message = raw

        await adapter.on_processing_start(event)
        adapter._emotion_api.assert_not_called()
