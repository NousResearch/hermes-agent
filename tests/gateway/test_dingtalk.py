"""Tests for DingTalk platform adapter."""
import asyncio
import contextlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(**extra):
    from gateway.platforms.dingtalk import DingTalkAdapter
    return DingTalkAdapter(PlatformConfig(enabled=True, extra=extra))


def _make_chatbot_message(**overrides):
    """Build a minimal ChatbotMessage-like mock."""
    defaults = dict(
        message_id="msg-001",
        text="",
        rich_text=None,
        rich_text_content=None,
        content=None,
        extensions=None,
        message_type="text",
        conversation_id="conv-group-1",
        conversation_type="2",
        conversation_title="Test Group",
        sender_id="user-sender",
        sender_nick="Alice",
        sender_staff_id="staff-alice",
        chatbot_user_id="bot-id",
        chatbotUserId=None,
        senderId=None,
        is_echo=False,
        isEcho=False,
        message_source=None,
        messageSource=None,
        session_webhook="https://api.dingtalk.com/v1.0/robot/webhook/123",
        create_at=str(int(time.time() * 1000)),
        robot_code="rbot",
        robotCode=None,
        picture_download_code=None,
        pictureDownloadCode=None,
        download_code=None,
        downloadCode=None,
        image_content=None,
        file_name=None,
        fileName=None,
        _raw_data=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _seed_session_context(
    adapter,
    chat_id: str = "c1",
    *,
    chat_type: str = "dm",
    conversation_id: str | None = "conv-1",
    sender_staff_id: str | None = "s1",
    session_webhook: str = "https://api.dingtalk.com/wh",
    **extra,
):
    """Store a minimal session context for outbound send tests."""
    context = {"chat_type": chat_type, "session_webhook": session_webhook}
    if conversation_id is not None:
        context["conversation_id"] = conversation_id
    if sender_staff_id is not None:
        context["sender_staff_id"] = sender_staff_id
    context.update(extra)
    adapter._session_context[chat_id] = context
    adapter._session_webhooks[chat_id] = session_webhook
    return context


# ===========================================================================
# 1. Requirements check
# ===========================================================================

class TestDingTalkRequirements:

    def test_returns_false_when_sdk_missing(self, monkeypatch):
        monkeypatch.setattr("gateway.platforms.dingtalk.DINGTALK_STREAM_AVAILABLE", False)
        from gateway.platforms.dingtalk import check_dingtalk_requirements
        assert check_dingtalk_requirements() is False

    def test_returns_false_when_httpx_missing(self, monkeypatch):
        monkeypatch.setattr("gateway.platforms.dingtalk.DINGTALK_STREAM_AVAILABLE", True)
        monkeypatch.setattr("gateway.platforms.dingtalk.HTTPX_AVAILABLE", False)
        from gateway.platforms.dingtalk import check_dingtalk_requirements
        assert check_dingtalk_requirements() is False

    def test_returns_false_when_env_vars_missing(self, monkeypatch):
        monkeypatch.setattr("gateway.platforms.dingtalk.DINGTALK_STREAM_AVAILABLE", True)
        monkeypatch.setattr("gateway.platforms.dingtalk.HTTPX_AVAILABLE", True)
        monkeypatch.delenv("DINGTALK_CLIENT_ID", raising=False)
        monkeypatch.delenv("DINGTALK_CLIENT_SECRET", raising=False)
        from gateway.platforms.dingtalk import check_dingtalk_requirements
        assert check_dingtalk_requirements() is False

    def test_returns_true_when_all_available(self, monkeypatch):
        monkeypatch.setattr("gateway.platforms.dingtalk.DINGTALK_STREAM_AVAILABLE", True)
        monkeypatch.setattr("gateway.platforms.dingtalk.HTTPX_AVAILABLE", True)
        monkeypatch.setenv("DINGTALK_CLIENT_ID", "id")
        monkeypatch.setenv("DINGTALK_CLIENT_SECRET", "sec")
        from gateway.platforms.dingtalk import check_dingtalk_requirements
        assert check_dingtalk_requirements() is True


# ===========================================================================
# 2. Adapter construction
# ===========================================================================

class TestDingTalkAdapterInit:

    def test_reads_config_from_extra(self):
        adapter = _make_adapter(client_id="cfg-id", client_secret="cfg-sec")
        assert adapter._client_id == "cfg-id"
        assert adapter._client_secret == "cfg-sec"
        assert adapter.name == "Dingtalk"

    def test_falls_back_to_env_vars(self, monkeypatch):
        monkeypatch.setenv("DINGTALK_CLIENT_ID", "env-id")
        monkeypatch.setenv("DINGTALK_CLIENT_SECRET", "env-sec")
        adapter = _make_adapter()
        assert adapter._client_id == "env-id"
        assert adapter._client_secret == "env-sec"

    def test_robot_code_defaults_to_client_id(self):
        adapter = _make_adapter(client_id="cid", client_secret="s")
        assert adapter._robot_code == "cid"
        assert adapter._configured_robot_code == ""

    def test_robot_code_from_extra(self):
        adapter = _make_adapter(client_id="cid", client_secret="s", robot_code="custom")
        assert adapter._robot_code == "custom"
        assert adapter._configured_robot_code == "custom"

    def test_session_context_initialized(self):
        adapter = _make_adapter()
        assert adapter._session_context == {}
        assert adapter._access_token == ""
        assert adapter._access_token_expires_at == 0.0

    def test_supports_message_editing_is_false(self):
        adapter = _make_adapter()
        assert adapter.SUPPORTS_MESSAGE_EDITING is False


# ===========================================================================
# 3. _extract_text
# ===========================================================================

class TestExtractText:

    def _call(self, msg):
        from gateway.platforms.dingtalk import DingTalkAdapter
        return DingTalkAdapter._extract_text(msg)

    def test_dict_text(self):
        msg = _make_chatbot_message(text={"content": "  hello  "})
        assert self._call(msg) == "hello"

    def test_string_text(self):
        msg = _make_chatbot_message(text="plain")
        assert self._call(msg) == "plain"

    def test_text_object_with_content_attr(self):
        msg = _make_chatbot_message(text=SimpleNamespace(content="  obj  "))
        assert self._call(msg) == "obj"

    def test_rich_text_fallback(self):
        msg = _make_chatbot_message(
            text="",
            rich_text=[{"text": "a"}, {"text": "b"}, {"image": "url"}],
        )
        assert self._call(msg) == "a b"

    def test_audio_recognition_fallback(self):
        msg = _make_chatbot_message(text="", content={"recognition": "speech"})
        assert self._call(msg) == "speech"

    def test_empty(self):
        msg = _make_chatbot_message(text="")
        assert self._call(msg) == ""


# ===========================================================================
# 4. _should_ignore_message
# ===========================================================================

class TestShouldIgnoreMessage:

    def _call(self, msg):
        from gateway.platforms.dingtalk import DingTalkAdapter
        return DingTalkAdapter._should_ignore_message(msg)

    def test_ignores_self(self):
        msg = _make_chatbot_message(sender_id="bot-1", chatbot_user_id="bot-1")
        assert self._call(msg) is True

    def test_ignores_echo(self):
        msg = _make_chatbot_message(is_echo=True)
        assert self._call(msg) is True

    def test_ignores_isEcho_variant(self):
        msg = _make_chatbot_message(isEcho=True)
        assert self._call(msg) is True

    def test_ignores_sync_source(self):
        msg = _make_chatbot_message(message_source="sync")
        assert self._call(msg) is True

    def test_ignores_bot_source(self):
        msg = _make_chatbot_message(message_source="bot")
        assert self._call(msg) is True

    def test_accepts_normal(self):
        msg = _make_chatbot_message()
        assert self._call(msg) is False


# ===========================================================================
# 5. Deduplication
# ===========================================================================

class TestDeduplication:

    def test_first_not_dup(self):
        adapter = _make_adapter()
        assert adapter._dedup.is_duplicate("m1") is False

    def test_second_is_dup(self):
        adapter = _make_adapter()
        adapter._dedup.is_duplicate("m1")
        assert adapter._dedup.is_duplicate("m1") is True

    def test_different_not_dup(self):
        adapter = _make_adapter()
        adapter._dedup.is_duplicate("m1")
        assert adapter._dedup.is_duplicate("m2") is False


# ===========================================================================
# 6. Helpers
# ===========================================================================

class TestHelpers:

    # -- _redact / _mask_secret --

    def test_mask_secret_short(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        assert DingTalkAdapter._mask_secret("short") == "***"

    def test_mask_secret_long(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        assert DingTalkAdapter._mask_secret("abcdefghijklmnop") == "abcd***mnop"

    def test_redact_masks_query_params(self):
        adapter = _make_adapter()
        url = "https://api.dingtalk.com/wh?session=abcdefghijklmnopqrstuvwxyz"
        redacted = adapter._redact(url)
        assert "abcdefghijklmnopqrstuvwxyz" not in redacted
        assert "***" in redacted

    # -- _coerce_mapping --

    def test_coerce_dict(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        assert DingTalkAdapter._coerce_mapping({"a": 1}) == {"a": 1}

    def test_coerce_json_str(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        assert DingTalkAdapter._coerce_mapping('{"b": 2}') == {"b": 2}

    def test_coerce_bad_json(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        assert DingTalkAdapter._coerce_mapping("nope") == {}

    def test_coerce_none(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        assert DingTalkAdapter._coerce_mapping(None) == {}

    def test_coerce_namespace(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        assert DingTalkAdapter._coerce_mapping(SimpleNamespace(x=1)) == {"x": 1}

    # -- _response_media_type --

    def test_media_type_from_header(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        resp = MagicMock(headers={"content-type": "image/png; charset=utf-8"})
        assert DingTalkAdapter._response_media_type(resp, fallback="image/jpeg") == "image/png"

    def test_media_type_fallback(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        resp = MagicMock(headers={})
        assert DingTalkAdapter._response_media_type(resp, fallback="audio/ogg") == "audio/ogg"

    def test_media_type_octet_uses_fallback(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        resp = MagicMock(headers={"content-type": "application/octet-stream"})
        assert DingTalkAdapter._response_media_type(resp, fallback="image/jpeg") == "image/jpeg"

    # -- _get_attr_variants --

    def test_attr_variants_first(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        obj = SimpleNamespace(sender_id="abc", senderId=None)
        assert DingTalkAdapter._get_attr_variants(obj, "sender_id", "senderId") == "abc"

    def test_attr_variants_second(self):
        from gateway.platforms.dingtalk import DingTalkAdapter
        obj = SimpleNamespace(sender_id=None, senderId="xyz")
        assert DingTalkAdapter._get_attr_variants(obj, "sender_id", "senderId") == "xyz"


# ===========================================================================
# 7. Send (text / markdown)
# ===========================================================================

class TestSend:

    @pytest.mark.asyncio
    async def test_posts_to_webhook(self):
        adapter = _make_adapter()
        resp = MagicMock(status_code=200, text="OK")
        client = AsyncMock()
        client.post = AsyncMock(return_value=resp)
        adapter._http_client = client

        result = await adapter.send(
            "c1", "Hello!",
            metadata={"session_webhook": "https://dingtalk.example/wh"},
        )
        assert result.success is True
        assert result.message_id is None
        call = client.post.call_args
        assert call[0][0] == "https://dingtalk.example/wh"
        body = call[1]["json"]
        assert body["msgtype"] == "markdown"
        assert body["markdown"]["title"] == "Hermes"
        assert body["markdown"]["text"] == "Hello!"

    @pytest.mark.asyncio
    async def test_fails_without_webhook(self):
        adapter = _make_adapter()
        adapter._http_client = AsyncMock()
        result = await adapter.send("c1", "Hi")
        assert result.success is False
        assert "session_webhook" in result.error

    @pytest.mark.asyncio
    async def test_uses_cached_webhook(self):
        adapter = _make_adapter()
        resp = MagicMock(status_code=200)
        client = AsyncMock()
        client.post = AsyncMock(return_value=resp)
        adapter._http_client = client
        adapter._session_webhooks["c1"] = "https://cached/wh"

        result = await adapter.send("c1", "Hi")
        assert result.success is True
        assert client.post.call_args[0][0] == "https://cached/wh"

    @pytest.mark.asyncio
    async def test_uses_session_context_webhook(self):
        adapter = _make_adapter()
        resp = MagicMock(status_code=200)
        client = AsyncMock()
        client.post = AsyncMock(return_value=resp)
        adapter._http_client = client
        adapter._session_context["c1"] = {"session_webhook": "https://context/wh"}

        result = await adapter.send("c1", "Hi")
        assert result.success is True
        assert client.post.call_args[0][0] == "https://context/wh"

    @pytest.mark.asyncio
    async def test_http_error(self):
        adapter = _make_adapter()
        resp = MagicMock(status_code=400, text="Bad")
        client = AsyncMock()
        client.post = AsyncMock(return_value=resp)
        adapter._http_client = client

        result = await adapter.send("c1", "Hi", metadata={"session_webhook": "https://x/wh"})
        assert result.success is False
        assert "400" in result.error


# ===========================================================================
# 8. Connect / disconnect
# ===========================================================================

class TestConnect:

    @pytest.mark.asyncio
    async def test_fails_without_sdk(self, monkeypatch):
        monkeypatch.setattr("gateway.platforms.dingtalk.DINGTALK_STREAM_AVAILABLE", False)
        adapter = _make_adapter()
        assert await adapter.connect() is False

    @pytest.mark.asyncio
    async def test_fails_without_httpx(self, monkeypatch):
        monkeypatch.setattr("gateway.platforms.dingtalk.DINGTALK_STREAM_AVAILABLE", True)
        monkeypatch.setattr("gateway.platforms.dingtalk.HTTPX_AVAILABLE", False)
        adapter = _make_adapter(client_id="cid", client_secret="sec")
        assert await adapter.connect() is False

    @pytest.mark.asyncio
    async def test_fails_without_credentials(self):
        adapter = _make_adapter()
        adapter._client_id = ""
        adapter._client_secret = ""
        assert await adapter.connect() is False

    @pytest.mark.asyncio
    async def test_lock_denied(self, monkeypatch):
        monkeypatch.setattr("gateway.platforms.dingtalk.DINGTALK_STREAM_AVAILABLE", True)
        monkeypatch.setattr("gateway.platforms.dingtalk.HTTPX_AVAILABLE", True)
        adapter = _make_adapter(client_id="cid", client_secret="sec")
        adapter._acquire_platform_lock = MagicMock(return_value=False)
        assert await adapter.connect() is False
        adapter._acquire_platform_lock.assert_called_once_with(
            "dingtalk-client-id", "cid", "DingTalk client_id",
        )

    @pytest.mark.asyncio
    async def test_releases_lock_on_failure(self, monkeypatch):
        monkeypatch.setattr("gateway.platforms.dingtalk.DINGTALK_STREAM_AVAILABLE", True)
        monkeypatch.setattr("gateway.platforms.dingtalk.HTTPX_AVAILABLE", True)
        adapter = _make_adapter(client_id="cid", client_secret="sec")
        adapter._acquire_platform_lock = MagicMock(return_value=True)
        adapter._release_platform_lock = MagicMock()
        with patch("gateway.platforms.dingtalk.dingtalk_stream") as mock_sdk:
            mock_sdk.Credential.side_effect = RuntimeError("boom")
            assert await adapter.connect() is False
        adapter._release_platform_lock.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_cleans_up(self):
        adapter = _make_adapter()
        adapter._session_webhooks["a"] = "http://x"
        adapter._session_context["a"] = {"chat_type": "dm"}
        adapter._dedup._seen["b"] = 1.0
        adapter._http_client = AsyncMock()
        adapter._stream_task = None
        adapter._release_platform_lock = MagicMock()

        await adapter.disconnect()
        assert len(adapter._session_webhooks) == 0
        assert len(adapter._session_context) == 0
        assert len(adapter._dedup._seen) == 0
        assert adapter._http_client is None
        adapter._release_platform_lock.assert_called_once()


class TestRunStream:

    @pytest.mark.asyncio
    async def test_awaits_async_stream_start(self):
        adapter = _make_adapter()
        started = asyncio.Event()

        async def async_start():
            started.set()
            raise asyncio.CancelledError()

        adapter._stream_client = SimpleNamespace(start=async_start)
        adapter._running = True
        task = asyncio.create_task(adapter._run_stream())
        try:
            await asyncio.wait_for(started.wait(), timeout=0.2)
            await asyncio.wait_for(task, timeout=0.2)
        finally:
            adapter._running = False
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task


# ===========================================================================
# 9. get_chat_info
# ===========================================================================

class TestGetChatInfo:

    @pytest.mark.asyncio
    async def test_uses_session_context(self):
        adapter = _make_adapter()
        adapter._session_context["c1"] = {"chat_type": "group", "chat_name": "Dev"}
        info = await adapter.get_chat_info("c1")
        assert info["name"] == "Dev"
        assert info["type"] == "group"
        assert info["chat_id"] == "c1"

    @pytest.mark.asyncio
    async def test_fallback_without_context(self):
        adapter = _make_adapter()
        info = await adapter.get_chat_info("some-id")
        assert info["name"] == "some-id"
        assert info["type"] == "dm"


# ===========================================================================
# 10. _resolve_robot_target
# ===========================================================================

class TestResolveRobotTarget:

    def test_group(self):
        adapter = _make_adapter()
        ctx = {"chat_type": "group", "conversation_id": "conv-1"}
        ep, target = adapter._resolve_robot_target("c1", ctx, kind_label="image")
        assert "groupMessages" in ep
        assert target == {"openConversationId": "conv-1"}

    def test_dm_staff_id(self):
        adapter = _make_adapter()
        ctx = {"chat_type": "dm", "sender_staff_id": "s1"}
        ep, target = adapter._resolve_robot_target("c1", ctx, kind_label="file")
        assert "oToMessages" in ep
        assert target == {"userIds": ["s1"]}

    def test_dm_conversation_uses_private_chat_endpoint(self):
        adapter = _make_adapter()
        ctx = {
            "chat_type": "dm",
            "conversation_id": "conv-1",
            "sender_staff_id": "s1",
            "session_webhook": "https://oapi.dingtalk.com/robot/sendBySession?session=sess-123",
        }
        ep, target = adapter._resolve_robot_target("c1", ctx, kind_label="file")
        assert "oToMessages" in ep
        assert target == {"userIds": ["s1"]}

    def test_dm_private_chat_endpoint_without_user_id(self):
        adapter = _make_adapter()
        ctx = {
            "chat_type": "dm",
            "conversation_id": "conv-1",
            "session_webhook": "https://oapi.dingtalk.com/robot/sendBySession?session=sess-123",
        }
        ep, target = adapter._resolve_robot_target("c1", ctx, kind_label="file")
        assert "privateChatMessages" in ep
        assert target == {"openConversationId": "conv-1", "token": "sess-123"}

    def test_dm_user_id_fallback(self):
        adapter = _make_adapter()
        ctx = {"chat_type": "dm", "user_id": "u1"}
        ep, target = adapter._resolve_robot_target("c1", ctx, kind_label="file")
        assert target == {"userIds": ["u1"]}

    def test_group_falls_back_to_chat_id(self):
        adapter = _make_adapter()
        ep, target = adapter._resolve_robot_target(
            "conv-from-chat-id", {"chat_type": "group"}, kind_label="img",
        )
        assert "groupMessages" in ep
        assert target == {"openConversationId": "conv-from-chat-id"}

    def test_dm_missing_user(self):
        adapter = _make_adapter()
        with pytest.raises(RuntimeError, match="Missing user ID"):
            adapter._resolve_robot_target("c1", {"chat_type": "dm"}, kind_label="f")


# ===========================================================================
# 11. Access token
# ===========================================================================

class TestAccessToken:

    @pytest.mark.asyncio
    async def test_cached(self):
        adapter = _make_adapter(client_id="cid", client_secret="sec")
        adapter._access_token = "cached"
        adapter._access_token_expires_at = time.time() + 3600
        adapter._http_client = AsyncMock()
        assert await adapter._get_access_token() == "cached"
        adapter._http_client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_refresh_expired(self):
        adapter = _make_adapter(client_id="cid", client_secret="sec")
        adapter._access_token_expires_at = time.time() - 10
        resp = MagicMock(status_code=200)
        resp.json.return_value = {"accessToken": "new", "expireIn": 7200}
        client = AsyncMock()
        client.post = AsyncMock(return_value=resp)
        adapter._http_client = client
        assert await adapter._get_access_token() == "new"
        assert adapter._access_token == "new"

    @pytest.mark.asyncio
    async def test_http_error(self):
        adapter = _make_adapter(client_id="cid", client_secret="sec")
        adapter._access_token_expires_at = 0
        resp = MagicMock(status_code=500)
        client = AsyncMock()
        client.post = AsyncMock(return_value=resp)
        adapter._http_client = client
        with pytest.raises(RuntimeError, match="accessToken request failed"):
            await adapter._get_access_token()


# ===========================================================================
# 12. send_image_file
# ===========================================================================

class TestSendImageFile:

    @pytest.mark.asyncio
    async def test_success(self, tmp_path, caplog):
        adapter = _make_adapter(client_id="cid", client_secret="sec")
        _seed_session_context(adapter, chat_type="group")
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        adapter._upload_media = AsyncMock(return_value="mid")
        adapter._send_image_message = AsyncMock(return_value=MagicMock(success=True, message_id="s1"))
        adapter._send_media_caption = AsyncMock()

        with caplog.at_level(logging.INFO, logger="gateway.platforms.dingtalk"):
            result = await adapter.send_image_file("c1", str(img), caption="Look!")
        assert result.success is True
        adapter._upload_media.assert_called_once_with(str(img), media_type="image")
        assert "Image sent to DingTalk" in caplog.text
        assert "test.jpg" in caplog.text
        assert "s1" in caplog.text


# ===========================================================================
# 13. send_image (remote URL)
# ===========================================================================

class TestSendImage:

    @pytest.mark.asyncio
    async def test_downloads_and_sends(self):
        adapter = _make_adapter(client_id="cid", client_secret="sec")
        adapter._session_context["c1"] = {
            "chat_type": "dm", "sender_staff_id": "s1",
            "session_webhook": "https://api.dingtalk.com/wh",
        }
        resp = MagicMock(status_code=200, content=b"\xff\xd8\xff" + b"\x00" * 50)
        resp.headers = {"content-type": "image/jpeg"}
        adapter._download_remote_response = AsyncMock(return_value=resp)
        adapter.send_image_file = AsyncMock(return_value=MagicMock(success=True))

        result = await adapter.send_image("c1", "https://example.com/img.jpg")
        assert result.success is True
        adapter._download_remote_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_download_error(self):
        adapter = _make_adapter()
        adapter._session_context["c1"] = {"session_webhook": "https://api.dingtalk.com/wh"}
        adapter._download_remote_response = AsyncMock(side_effect=RuntimeError("net err"))
        adapter.send = AsyncMock(return_value=MagicMock(success=True))

        result = await adapter.send_image("c1", "https://example.com/bad.jpg")
        assert result.success is False


# ===========================================================================
# 14. send_document
# ===========================================================================

class TestSendDocument:

    @pytest.mark.asyncio
    async def test_success(self, tmp_path, caplog):
        adapter = _make_adapter(client_id="cid", client_secret="sec")
        _seed_session_context(adapter)
        doc = tmp_path / "report.pdf"
        doc.write_bytes(b"%PDF" + b"\x00" * 100)

        adapter._upload_media = AsyncMock(return_value="mid")
        adapter._send_file_message = AsyncMock(return_value=MagicMock(success=True, message_id="pqk-123"))
        adapter._send_media_caption = AsyncMock()

        with caplog.at_level(logging.INFO, logger="gateway.platforms.dingtalk"):
            result = await adapter.send_document("c1", str(doc), caption="Report")
        assert result.success is True
        adapter._upload_media.assert_called_once_with(str(doc), media_type="file")
        adapter._send_file_message.assert_called_once()
        assert "File sent to DingTalk" in caplog.text
        assert "report.pdf" in caplog.text
        assert "pqk-123" in caplog.text


# ===========================================================================
# 15. send_voice
# ===========================================================================

class TestSendVoice:

    @pytest.mark.asyncio
    async def test_native_with_duration_ogg(self, tmp_path, caplog):
        adapter = _make_adapter(client_id="cid", client_secret="sec")
        _seed_session_context(adapter)
        audio = tmp_path / "v.ogg"
        audio.write_bytes(b"OggS" + b"\x00" * 100)

        adapter._upload_media = AsyncMock(return_value="mid")
        adapter._send_audio_message = AsyncMock(return_value=MagicMock(success=True, message_id="audio-123"))
        adapter._send_media_caption = AsyncMock()

        with caplog.at_level(logging.INFO, logger="gateway.platforms.dingtalk"):
            result = await adapter.send_voice("c1", str(audio), duration_ms=5000)
        assert result.success is True
        adapter._upload_media.assert_called_once_with(str(audio), media_type="voice")
        adapter._send_audio_message.assert_called_once_with("c1", "mid", 5000, metadata=None)
        assert "Audio sent to DingTalk" in caplog.text
        assert "v.ogg" in caplog.text
        assert "audio-123" in caplog.text

    @pytest.mark.asyncio
    async def test_fallback_without_duration(self, tmp_path):
        adapter = _make_adapter(client_id="cid", client_secret="sec")
        _seed_session_context(adapter)
        audio = tmp_path / "v.ogg"
        audio.write_bytes(b"OggS" + b"\x00" * 100)

        adapter._upload_media = AsyncMock(return_value="mid")
        adapter._send_file_message = AsyncMock(return_value=MagicMock(success=True))
        adapter._send_media_caption = AsyncMock()

        result = await adapter.send_voice("c1", str(audio))
        assert result.success is True
        adapter._upload_media.assert_called_once_with(str(audio), media_type="file")
        adapter._send_file_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_fallback_non_ogg_amr(self, tmp_path):
        adapter = _make_adapter(client_id="cid", client_secret="sec")
        _seed_session_context(adapter)
        audio = tmp_path / "v.mp3"
        audio.write_bytes(b"\xff\xfb" + b"\x00" * 100)

        adapter._upload_media = AsyncMock(return_value="mid")
        adapter._send_file_message = AsyncMock(return_value=MagicMock(success=True))
        adapter._send_media_caption = AsyncMock()

        # mp3 with duration -> still fallback
        result = await adapter.send_voice("c1", str(audio), duration_ms=3000)
        assert result.success is True
        adapter._upload_media.assert_called_once_with(str(audio), media_type="file")

    

# ===========================================================================
# 16. local media path validation
# ===========================================================================

class TestLocalMediaPathValidation:

    @pytest.mark.asyncio
    @pytest.mark.parametrize("method_name, path_text", [
        ("send_image_file", "/no/such/file.jpg"),
        ("send_document", "/no/file.pdf"),
        ("send_voice", "/no/audio.ogg"),
        ("send_video", "/no/video.mp4"),
    ])
    async def test_missing_file_rejected(self, method_name, path_text):
        adapter = _make_adapter()
        _seed_session_context(adapter, chat_type="dm", conversation_id=None, sender_staff_id=None)
        adapter.send = AsyncMock(return_value=MagicMock(success=True))

        result = await getattr(adapter, method_name)("c1", path_text)

        assert result.success is False
        assert result.error == f"File not found: {path_text}"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("method_name", [
        "send_image_file",
        "send_document",
        "send_voice",
        "send_video",
    ])
    async def test_directory_path_rejected(self, method_name, tmp_path):
        adapter = _make_adapter()
        _seed_session_context(adapter, chat_type="dm", conversation_id=None, sender_staff_id=None)
        adapter.send = AsyncMock(return_value=MagicMock(success=True))

        result = await getattr(adapter, method_name)("c1", str(tmp_path))

        assert result.success is False
        assert result.error == f"Not a file: {tmp_path}"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("method_name", [
        "send_image_file",
        "send_document",
        "send_voice",
        "send_video",
    ])
    @pytest.mark.parametrize("path_text", [
        "/path/to/file",
        "/Users/.../dingtalk-gateway-user-message.txt",
    ])
    async def test_placeholder_path_rejected(self, method_name, path_text):
        adapter = _make_adapter()
        _seed_session_context(adapter, chat_type="dm", conversation_id=None, sender_staff_id=None)
        adapter.send = AsyncMock(return_value=MagicMock(success=True))

        result = await getattr(adapter, method_name)("c1", path_text)

        assert result.success is False
        assert result.error == f"Placeholder file path: {path_text}"


# ===========================================================================
# 17. send_video
# ===========================================================================

class TestSendVideo:

    @pytest.mark.asyncio
    async def test_native_with_all_params(self, tmp_path, caplog):
        adapter = _make_adapter(client_id="cid", client_secret="sec")
        _seed_session_context(adapter, chat_type="group")
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"\x00" * 100)
        thumb = tmp_path / "thumb.jpg"
        thumb.write_bytes(b"\xff\xd8" + b"\x00" * 50)

        adapter._upload_media = AsyncMock(side_effect=["vid-m", "cover-m"])
        adapter._send_video_message = AsyncMock(return_value=MagicMock(success=True, message_id="video-123"))
        adapter._send_media_caption = AsyncMock()

        with caplog.at_level(logging.INFO, logger="gateway.platforms.dingtalk"):
            result = await adapter.send_video(
                "c1", str(video), duration_seconds=30, thumbnail_path=str(thumb),
            )
        assert result.success is True
        assert adapter._upload_media.call_count == 2
        adapter._send_video_message.assert_called_once()
        assert "Video sent to DingTalk" in caplog.text
        assert "clip.mp4" in caplog.text
        assert "video-123" in caplog.text

    @pytest.mark.asyncio
    async def test_fallback_without_thumbnail(self, tmp_path):
        adapter = _make_adapter(client_id="cid", client_secret="sec")
        _seed_session_context(adapter)
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"\x00" * 100)

        adapter._upload_media = AsyncMock(return_value="mid")
        adapter._send_file_message = AsyncMock(return_value=MagicMock(success=True))
        adapter._send_media_caption = AsyncMock()

        result = await adapter.send_video("c1", str(video), duration_seconds=30)
        assert result.success is True
        adapter._upload_media.assert_called_once_with(str(video), media_type="file")

    @pytest.mark.asyncio
    async def test_fallback_non_mp4(self, tmp_path):
        adapter = _make_adapter(client_id="cid", client_secret="sec")
        _seed_session_context(adapter)
        video = tmp_path / "clip.avi"
        video.write_bytes(b"\x00" * 100)
        thumb = tmp_path / "thumb.jpg"
        thumb.write_bytes(b"\xff\xd8" + b"\x00" * 50)

        adapter._upload_media = AsyncMock(return_value="mid")
        adapter._send_file_message = AsyncMock(return_value=MagicMock(success=True))
        adapter._send_media_caption = AsyncMock()

        result = await adapter.send_video(
            "c1", str(video), duration_seconds=30, thumbnail_path=str(thumb),
        )
        assert result.success is True
        adapter._upload_media.assert_called_once_with(str(video), media_type="file")

# ===========================================================================
# 18. send_animation
# ===========================================================================

class TestSendAnimation:

    @pytest.mark.asyncio
    async def test_delegates_to_send_image(self):
        adapter = _make_adapter()
        adapter.send_image = AsyncMock(return_value=MagicMock(success=True))
        result = await adapter.send_animation("c1", "https://example.com/a.gif")
        assert result.success is True
        adapter.send_image.assert_called_once_with(
            "c1", "https://example.com/a.gif",
            caption=None, reply_to=None, metadata=None,
        )


# ===========================================================================
# 19. _send_robot_media_message
# ===========================================================================

class TestSendRobotMediaMessage:

    @pytest.mark.asyncio
    async def test_group_endpoint(self):
        adapter = _make_adapter(client_id="cid", client_secret="sec", robot_code="rbot")
        adapter._session_context["c1"] = {"chat_type": "group", "conversation_id": "conv-1"}
        adapter._access_token = "tok"
        adapter._access_token_expires_at = time.time() + 3600

        resp = MagicMock(status_code=200)
        resp.json.return_value = {"processQueryKey": "pqk"}
        client = AsyncMock()
        client.post = AsyncMock(return_value=resp)
        adapter._http_client = client

        result = await adapter._send_robot_media_message(
            "c1", msg_key="sampleImageMsg", msg_param={"photoURL": "m1"}, kind_label="image",
        )
        assert result.success is True
        call = client.post.call_args
        assert "groupMessages" in call[0][0]
        body = call[1]["json"]
        assert body["robotCode"] == "rbot"
        assert body["openConversationId"] == "conv-1"

    @pytest.mark.asyncio
    async def test_dm_endpoint(self):
        adapter = _make_adapter(client_id="cid", client_secret="sec", robot_code="rbot")
        adapter._session_context["u1"] = {
            "chat_type": "dm",
            "conversation_id": "conv-1",
            "sender_staff_id": "s1",
            "session_webhook": "https://oapi.dingtalk.com/robot/sendBySession?session=sess-123",
        }
        adapter._access_token = "tok"
        adapter._access_token_expires_at = time.time() + 3600

        resp = MagicMock(status_code=200)
        resp.json.return_value = {"processQueryKey": "pqk"}
        client = AsyncMock()
        client.post = AsyncMock(return_value=resp)
        adapter._http_client = client

        result = await adapter._send_robot_media_message(
            "u1", msg_key="sampleFile",
            msg_param={"mediaId": "m1", "fileName": "f.pdf", "fileType": "pdf"},
            kind_label="file",
        )
        assert result.success is True
        assert "oToMessages" in client.post.call_args[0][0]
        body = client.post.call_args[1]["json"]
        assert body["userIds"] == ["s1"]

    @pytest.mark.asyncio
    async def test_dm_private_endpoint_without_user_id(self):
        adapter = _make_adapter(client_id="cid", client_secret="sec", robot_code="rbot")
        adapter._session_context["u1"] = {
            "chat_type": "dm",
            "conversation_id": "conv-1",
            "session_webhook": "https://oapi.dingtalk.com/robot/sendBySession?session=sess-123",
        }
        adapter._access_token = "tok"
        adapter._access_token_expires_at = time.time() + 3600

        resp = MagicMock(status_code=200)
        resp.json.return_value = {"processQueryKey": "pqk"}
        client = AsyncMock()
        client.post = AsyncMock(return_value=resp)
        adapter._http_client = client

        result = await adapter._send_robot_media_message(
            "u1", msg_key="sampleFile",
            msg_param={"mediaId": "m1", "fileName": "f.pdf", "fileType": "pdf"},
            kind_label="file",
        )
        assert result.success is True
        assert "privateChatMessages" in client.post.call_args[0][0]
        body = client.post.call_args[1]["json"]
        assert body["openConversationId"] == "conv-1"
        assert body["token"] == "sess-123"

    @pytest.mark.asyncio
    async def test_prefers_session_robot_code(self):
        adapter = _make_adapter(client_id="cid", client_secret="sec")
        adapter._session_context["u1"] = {
            "chat_type": "dm",
            "conversation_id": "conv-1",
            "sender_staff_id": "s1",
            "robot_code": "session-rbot",
            "session_webhook": "https://oapi.dingtalk.com/robot/sendBySession?session=sess-123",
        }
        adapter._access_token = "tok"
        adapter._access_token_expires_at = time.time() + 3600

        resp = MagicMock(status_code=200)
        resp.json.return_value = {"processQueryKey": "pqk"}
        client = AsyncMock()
        client.post = AsyncMock(return_value=resp)
        adapter._http_client = client

        result = await adapter._send_robot_media_message(
            "u1",
            msg_key="sampleFile",
            msg_param={"mediaId": "m1", "fileName": "f.pdf", "fileType": "pdf"},
            kind_label="file",
        )
        assert result.success is True
        assert client.post.call_args[1]["json"]["robotCode"] == "session-rbot"

    @pytest.mark.asyncio
    async def test_robot_not_found_mentions_client_id_fallback(self):
        adapter = _make_adapter(client_id="cid", client_secret="sec")
        adapter._session_context["u1"] = {
            "chat_type": "dm",
            "conversation_id": "conv-1",
            "sender_staff_id": "s1",
            "session_webhook": "https://oapi.dingtalk.com/robot/sendBySession?session=sess-123",
        }
        adapter._access_token = "tok"
        adapter._access_token_expires_at = time.time() + 3600

        resp = MagicMock(status_code=400, text='{"code":"resource.not.found","message":"robot 不存在"}')
        client = AsyncMock()
        client.post = AsyncMock(return_value=resp)
        adapter._http_client = client

        result = await adapter._send_robot_media_message(
            "u1",
            msg_key="sampleFile",
            msg_param={"mediaId": "m1", "fileName": "f.pdf", "fileType": "pdf"},
            kind_label="file",
        )
        assert result.success is False
        assert "robotCodeSource=client_id_fallback" in (result.error or "")
        assert "/v1.0/robot/oToMessages/batchSend" in (result.error or "")

    @pytest.mark.asyncio
    async def test_business_error_in_200_response_is_failure(self):
        adapter = _make_adapter(client_id="cid", client_secret="sec", robot_code="rbot")
        adapter._session_context["u1"] = {
            "chat_type": "dm",
            "conversation_id": "conv-1",
            "sender_staff_id": "s1",
            "session_webhook": "https://oapi.dingtalk.com/robot/sendBySession?session=sess-123",
        }
        adapter._access_token = "tok"
        adapter._access_token_expires_at = time.time() + 3600

        resp = MagicMock(status_code=200, text='{"code":"resource.not.found","message":"robot 不存在"}')
        resp.json.return_value = {"code": "resource.not.found", "message": "robot 不存在"}
        client = AsyncMock()
        client.post = AsyncMock(return_value=resp)
        adapter._http_client = client

        result = await adapter._send_robot_media_message(
            "u1",
            msg_key="sampleImageMsg",
            msg_param={"photoURL": "m1"},
            kind_label="image",
        )
        assert result.success is False
        assert "resource.not.found" in (result.error or "")
        assert "/v1.0/robot/oToMessages/batchSend" in (result.error or "")

    @pytest.mark.asyncio
    async def test_missing_context(self):
        adapter = _make_adapter()
        adapter._http_client = AsyncMock()
        result = await adapter._send_robot_media_message(
            "unknown", msg_key="sampleImageMsg", msg_param={"photoURL": "x"}, kind_label="image",
        )
        assert result.success is False
        assert "Missing user ID" in result.error


# ===========================================================================
# 19. _query_message_file_download_url
# ===========================================================================

class TestQueryMessageFileDownloadUrl:

    @pytest.mark.asyncio
    async def test_success(self):
        adapter = _make_adapter(client_id="cid", client_secret="sec")
        adapter._access_token = "tok"
        adapter._access_token_expires_at = time.time() + 3600
        resp = MagicMock(status_code=200)
        resp.json.return_value = {"downloadUrl": "https://files.dingtalk.com/abc"}
        client = AsyncMock()
        client.post = AsyncMock(return_value=resp)
        adapter._http_client = client

        url = await adapter._query_message_file_download_url(download_code="dc", robot_code="rbot")
        assert url == "https://files.dingtalk.com/abc"

    @pytest.mark.asyncio
    async def test_http_error(self):
        adapter = _make_adapter(client_id="cid", client_secret="sec")
        adapter._access_token = "tok"
        adapter._access_token_expires_at = time.time() + 3600
        resp = MagicMock(status_code=500)
        client = AsyncMock()
        client.post = AsyncMock(return_value=resp)
        adapter._http_client = client

        with pytest.raises(RuntimeError, match="download URL request failed"):
            await adapter._query_message_file_download_url(download_code="dc", robot_code="rbot")


# ===========================================================================
# 20. _download_remote_response
# ===========================================================================

class TestDownloadRemoteResponse:

    @pytest.mark.asyncio
    async def test_success(self):
        adapter = _make_adapter()
        resp = MagicMock(status_code=200, content=b"data")
        client = AsyncMock()
        client.get = AsyncMock(return_value=resp)
        adapter._http_client = client
        r = await adapter._download_remote_response("https://example.com/f")
        assert r.content == b"data"

    @pytest.mark.asyncio
    async def test_http_error(self):
        adapter = _make_adapter()
        resp = MagicMock(status_code=404)
        client = AsyncMock()
        client.get = AsyncMock(return_value=resp)
        adapter._http_client = client
        with pytest.raises(RuntimeError, match="media download failed"):
            await adapter._download_remote_response("https://example.com/x")

    @pytest.mark.asyncio
    async def test_no_client(self):
        adapter = _make_adapter()
        adapter._http_client = None
        with pytest.raises(RuntimeError, match="HTTP client not initialized"):
            await adapter._download_remote_response("https://example.com/x")


# ===========================================================================
# 21. _on_message (inbound message processing)
# ===========================================================================

class TestOnMessage:

    @pytest.mark.asyncio
    async def test_text_group_message(self):
        adapter = _make_adapter()
        adapter.handle_message = AsyncMock()
        adapter._http_client = AsyncMock()

        msg = _make_chatbot_message(text="hello", message_type="text")
        await adapter._on_message(msg)

        adapter.handle_message.assert_called_once()
        event = adapter.handle_message.call_args[0][0]
        assert event.text == "hello"
        assert event.source.chat_type == "group"
        assert event.source.chat_id == "conv-group-1"

    @pytest.mark.asyncio
    async def test_dm_message_uses_sender_id(self):
        adapter = _make_adapter()
        adapter.handle_message = AsyncMock()
        adapter._http_client = AsyncMock()

        msg = _make_chatbot_message(
            text="hi", conversation_type="1", conversation_id="",
        )
        await adapter._on_message(msg)

        event = adapter.handle_message.call_args[0][0]
        assert event.source.chat_type == "dm"
        assert event.source.chat_id == "user-sender"

    @pytest.mark.asyncio
    async def test_ignores_self_message(self):
        adapter = _make_adapter()
        adapter.handle_message = AsyncMock()
        msg = _make_chatbot_message(sender_id="bot-id", chatbot_user_id="bot-id")
        await adapter._on_message(msg)
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_ignores_duplicate(self):
        adapter = _make_adapter()
        adapter.handle_message = AsyncMock()
        adapter._http_client = AsyncMock()
        msg = _make_chatbot_message(text="hi", message_id="dup-1")
        await adapter._on_message(msg)
        await adapter._on_message(msg)
        assert adapter.handle_message.call_count == 1

    @pytest.mark.asyncio
    async def test_captures_session_context(self):
        adapter = _make_adapter()
        adapter.handle_message = AsyncMock()
        adapter._http_client = AsyncMock()

        msg = _make_chatbot_message(text="hi")
        await adapter._on_message(msg)

        ctx = adapter._session_context.get("conv-group-1")
        assert ctx is not None
        assert ctx["chat_type"] == "group"
        assert ctx["sender_staff_id"] == "staff-alice"
        assert ctx["robot_code"] == "rbot"

    @pytest.mark.asyncio
    async def test_captures_robot_code_from_raw_payload(self):
        adapter = _make_adapter()
        adapter.handle_message = AsyncMock()
        adapter._http_client = AsyncMock()

        msg = _make_chatbot_message(
            text="hi",
            robot_code="",
            _raw_data={"robotCode": "raw-rbot", "senderStaffId": "staff-alice"},
        )
        await adapter._on_message(msg)

        ctx = adapter._session_context.get("conv-group-1")
        assert ctx is not None
        assert ctx["robot_code"] == "raw-rbot"

    @pytest.mark.asyncio
    async def test_caches_session_webhook(self):
        adapter = _make_adapter()
        adapter.handle_message = AsyncMock()
        adapter._http_client = AsyncMock()

        msg = _make_chatbot_message(text="hi")
        await adapter._on_message(msg)

        assert adapter._session_webhooks.get("conv-group-1") == "https://api.dingtalk.com/v1.0/robot/webhook/123"

    @pytest.mark.asyncio
    async def test_rejects_non_dingtalk_webhook(self):
        adapter = _make_adapter()
        adapter.handle_message = AsyncMock()
        adapter._http_client = AsyncMock()

        msg = _make_chatbot_message(
            text="hi",
            session_webhook="https://evil.com/steal",
        )
        await adapter._on_message(msg)
        assert "conv-group-1" not in adapter._session_webhooks

    @pytest.mark.asyncio
    async def test_empty_message_skipped(self):
        adapter = _make_adapter()
        adapter.handle_message = AsyncMock()
        msg = _make_chatbot_message(text="", message_type="text")
        await adapter._on_message(msg)
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_picture_message_type(self):
        adapter = _make_adapter()
        adapter.handle_message = AsyncMock()
        adapter._http_client = AsyncMock()
        adapter._download_inbound_picture = AsyncMock(return_value=("/tmp/img.jpg", "image/jpeg"))

        msg = _make_chatbot_message(text="", message_type="picture")
        await adapter._on_message(msg)

        from gateway.platforms.base import MessageType
        event = adapter.handle_message.call_args[0][0]
        assert event.message_type == MessageType.PHOTO
        assert event.media_urls == ["/tmp/img.jpg"]
        assert event.media_types == ["image/jpeg"]

    @pytest.mark.asyncio
    async def test_audio_message_type(self):
        adapter = _make_adapter()
        adapter.handle_message = AsyncMock()
        adapter._http_client = AsyncMock()
        adapter._download_inbound_audio = AsyncMock(return_value=("/tmp/a.ogg", "audio/ogg"))

        msg = _make_chatbot_message(text="", message_type="audio")
        await adapter._on_message(msg)

        from gateway.platforms.base import MessageType
        event = adapter.handle_message.call_args[0][0]
        assert event.message_type == MessageType.AUDIO

    @pytest.mark.asyncio
    async def test_video_message_type(self):
        adapter = _make_adapter()
        adapter.handle_message = AsyncMock()
        adapter._http_client = AsyncMock()
        adapter._download_inbound_video = AsyncMock(return_value=("/tmp/v.mp4", "video/mp4"))

        msg = _make_chatbot_message(text="", message_type="video")
        await adapter._on_message(msg)

        from gateway.platforms.base import MessageType
        event = adapter.handle_message.call_args[0][0]
        assert event.message_type == MessageType.VIDEO

    @pytest.mark.asyncio
    async def test_file_message_type(self):
        adapter = _make_adapter()
        adapter.handle_message = AsyncMock()
        adapter._http_client = AsyncMock()
        adapter._download_inbound_file = AsyncMock(return_value=("/tmp/f.pdf", "application/pdf"))

        msg = _make_chatbot_message(text="", message_type="file")
        await adapter._on_message(msg)

        from gateway.platforms.base import MessageType
        event = adapter.handle_message.call_args[0][0]
        assert event.message_type == MessageType.DOCUMENT

    @pytest.mark.asyncio
    async def test_media_download_failure_still_delivers_text(self):
        adapter = _make_adapter()
        adapter.handle_message = AsyncMock()
        adapter._http_client = AsyncMock()
        adapter._download_inbound_picture = AsyncMock(side_effect=RuntimeError("net"))

        msg = _make_chatbot_message(text="caption text", message_type="picture")
        await adapter._on_message(msg)

        event = adapter.handle_message.call_args[0][0]
        assert event.text == "caption text"
        assert event.media_urls == []


# ===========================================================================
# 22. _IncomingHandler
# ===========================================================================

class TestIncomingHandler:

    @pytest.mark.asyncio
    async def test_raw_process_converts_callback_to_chatbot_message(self, monkeypatch):
        from gateway.platforms.dingtalk import _IncomingHandler

        adapter = _make_adapter()
        adapter._on_message = AsyncMock()
        loop = asyncio.get_running_loop()
        handler = _IncomingHandler(adapter, loop)

        class _DoneFuture:
            def result(self, timeout=None):
                return None

        monkeypatch.setattr(
            "gateway.platforms.dingtalk.asyncio.run_coroutine_threadsafe",
            lambda coro, _loop: _DoneFuture(),
        )

        callback = SimpleNamespace(
            headers=SimpleNamespace(message_id="cb-1"),
            data={
                "msgId": "msg-123",
                "senderId": "user-123",
                "senderNick": "Alice",
                "senderStaffId": "staff-123",
                "robotCode": "rbot-123",
                "conversationId": "conv-123",
                "conversationType": "1",
                "conversationTitle": "Alice",
                "sessionWebhook": "https://api.dingtalk.com/v1.0/robot/webhook/123",
                "createAt": str(int(time.time() * 1000)),
                "msgtype": "text",
                "text": {"content": "ping"},
            },
        )

        ack = await handler.raw_process(callback)

        adapter._on_message.assert_awaited_once()
        inbound = adapter._on_message.await_args.args[0]
        assert inbound.sender_id == "user-123"
        assert inbound.sender_staff_id == "staff-123"
        assert inbound.robot_code == "rbot-123"
        assert inbound._raw_data["robotCode"] == "rbot-123"
        assert inbound.message_type == "text"
        assert inbound.text.content == "ping"
        assert ack.code == 200
        assert ack.headers.message_id == "cb-1"


# ===========================================================================
# 23. _send_error_notice
# ===========================================================================

class TestSendErrorNotice:

    @pytest.mark.asyncio
    async def test_sends_via_webhook(self):
        adapter = _make_adapter()
        adapter.send = AsyncMock(return_value=MagicMock(success=True))
        ctx = {"session_webhook": "https://api.dingtalk.com/wh"}
        await adapter._send_error_notice("c1", ctx, "something broke")
        adapter.send.assert_called_once()
        call_text = adapter.send.call_args[0][1]
        assert "something broke" in call_text

    @pytest.mark.asyncio
    async def test_no_webhook_skips(self):
        adapter = _make_adapter()
        adapter.send = AsyncMock()
        await adapter._send_error_notice("c1", {}, "err")
        adapter.send.assert_not_called()


# ===========================================================================
# 24. Platform enum
# ===========================================================================

class TestPlatformEnum:

    def test_dingtalk_in_platform_enum(self):
        assert Platform.DINGTALK.value == "dingtalk"
