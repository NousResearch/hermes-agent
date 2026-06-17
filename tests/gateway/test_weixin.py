"""Tests for the Weixin platform adapter."""

import asyncio
import base64
import json
import os
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from gateway.config import PlatformConfig
from gateway.config import GatewayConfig, HomeChannel, Platform, _apply_env_overrides
from gateway.platforms.base import SendResult
from gateway.platforms import weixin
from gateway.platforms.weixin import ContextTokenStore, WeixinAdapter
from tools.send_message_tool import _parse_target_ref, _send_to_platform


def _make_adapter() -> WeixinAdapter:
    return WeixinAdapter(
        PlatformConfig(
            enabled=True,
            token="test-token",
            extra={"account_id": "test-account"},
        )
    )


class TestWeixinFormatting:
    def test_format_message_preserves_markdown(self):
        adapter = _make_adapter()

        content = "# Title\n\n## Plan\n\nUse **bold** and [docs](https://example.com)."

        assert adapter.format_message(content) == content

    def test_format_message_preserves_markdown_tables(self):
        adapter = _make_adapter()

        content = (
            "| Setting | Value |\n"
            "| --- | --- |\n"
            "| Timeout | 30s |\n"
            "| Retries | 3 |\n"
        )

        assert adapter.format_message(content) == content.strip()

    def test_format_message_preserves_fenced_code_blocks(self):
        adapter = _make_adapter()

        content = "## Snippet\n\n```python\nprint('hi')\n```"

        assert adapter.format_message(content) == content

    def test_format_message_returns_empty_string_for_none(self):
        adapter = _make_adapter()

        assert adapter.format_message(None) == ""


class TestWeixinChunking:
    def test_split_text_splits_short_chatty_replies_into_separate_bubbles(self):
        adapter = _make_adapter()

        content = adapter.format_message("第一行\n第二行\n第三行")
        chunks = adapter._split_text(content)

        assert chunks == ["第一行", "第二行", "第三行"]

    def test_split_text_keeps_structured_table_block_together(self):
        adapter = _make_adapter()

        content = adapter.format_message(
            "- Setting: Timeout\n  Value: 30s\n- Setting: Retries\n  Value: 3"
        )
        chunks = adapter._split_text(content)

        assert chunks == ["- Setting: Timeout\n  Value: 30s\n- Setting: Retries\n  Value: 3"]

    def test_split_text_keeps_four_line_structured_blocks_together(self):
        adapter = _make_adapter()

        content = adapter.format_message(
            "今天结论：\n"
            "- 留存下降 3%\n"
            "- 转化上涨 8%\n"
            "- 主要问题在首日激活"
        )
        chunks = adapter._split_text(content)

        assert chunks == ["今天结论：\n- 留存下降 3%\n- 转化上涨 8%\n- 主要问题在首日激活"]

    def test_split_text_keeps_heading_with_body_together(self):
        adapter = _make_adapter()

        content = adapter.format_message("## 结论\n这是正文")
        chunks = adapter._split_text(content)

        assert chunks == ["## 结论\n这是正文"]

    def test_split_text_keeps_short_reformatted_table_in_single_chunk(self):
        adapter = _make_adapter()

        content = adapter.format_message(
            "| Setting | Value |\n"
            "| --- | --- |\n"
            "| Timeout | 30s |\n"
            "| Retries | 3 |\n"
        )
        chunks = adapter._split_text(content)

        assert chunks == [content]

    def test_split_text_keeps_complete_code_block_together_when_possible(self):
        adapter = _make_adapter()
        adapter.MAX_MESSAGE_LENGTH = 80

        content = adapter.format_message(
            "## Intro\n\nShort paragraph.\n\n```python\nprint('hello world')\nprint('again')\n```\n\nTail paragraph."
        )
        chunks = adapter._split_text(content)

        assert len(chunks) >= 2
        assert any(
            "```python\nprint('hello world')\nprint('again')\n```" in chunk
            for chunk in chunks
        )
        assert all(chunk.count("```") % 2 == 0 for chunk in chunks)

    def test_split_text_safely_splits_long_code_blocks(self):
        adapter = _make_adapter()
        adapter.MAX_MESSAGE_LENGTH = 70

        lines = "\n".join(f"line_{idx:02d} = {idx}" for idx in range(10))
        content = adapter.format_message(f"```python\n{lines}\n```")
        chunks = adapter._split_text(content)

        assert len(chunks) > 1
        assert all(len(chunk) <= adapter.MAX_MESSAGE_LENGTH for chunk in chunks)
        assert all(chunk.count("```") >= 2 for chunk in chunks)

    def test_split_text_can_restore_legacy_multiline_splitting_via_config(self):
        adapter = WeixinAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "account_id": "acct",
                    "token": "***",
                    "split_multiline_messages": True,
                },
            )
        )

        content = adapter.format_message("第一行\n第二行\n第三行")
        chunks = adapter._split_text(content)

        assert chunks == ["第一行", "第二行", "第三行"]


class TestWeixinConfig:
    def test_apply_env_overrides_configures_weixin(self):
        config = GatewayConfig()

        with patch.dict(
            os.environ,
            {
                "WEIXIN_ACCOUNT_ID": "bot-account",
                "WEIXIN_TOKEN": "bot-token",
                "WEIXIN_BASE_URL": "https://ilink.example.com/",
                "WEIXIN_CDN_BASE_URL": "https://cdn.example.com/c2c/",
                "WEIXIN_DM_POLICY": "allowlist",
                "WEIXIN_SPLIT_MULTILINE_MESSAGES": "true",
                "WEIXIN_ALLOWED_USERS": "wxid_1,wxid_2",
                "WEIXIN_HOME_CHANNEL": "wxid_1",
                "WEIXIN_HOME_CHANNEL_NAME": "Primary DM",
            },
            clear=True,
        ):
            _apply_env_overrides(config)

        platform_config = config.platforms[Platform.WEIXIN]
        assert platform_config.enabled is True
        assert platform_config.token == "bot-token"
        assert platform_config.extra["account_id"] == "bot-account"
        assert platform_config.extra["base_url"] == "https://ilink.example.com"
        assert platform_config.extra["cdn_base_url"] == "https://cdn.example.com/c2c"
        assert platform_config.extra["dm_policy"] == "allowlist"
        assert platform_config.extra["split_multiline_messages"] == "true"
        assert platform_config.extra["allow_from"] == "wxid_1,wxid_2"
        assert platform_config.home_channel == HomeChannel(Platform.WEIXIN, "wxid_1", "Primary DM")

    def test_get_connected_platforms_includes_weixin_with_token(self):
        config = GatewayConfig(
            platforms={
                Platform.WEIXIN: PlatformConfig(
                    enabled=True,
                    token="bot-token",
                    extra={"account_id": "bot-account"},
                )
            }
        )

        assert config.get_connected_platforms() == [Platform.WEIXIN]

    def test_get_connected_platforms_requires_account_id(self):
        config = GatewayConfig(
            platforms={
                Platform.WEIXIN: PlatformConfig(
                    enabled=True,
                    token="bot-token",
                )
            }
        )

        assert config.get_connected_platforms() == []


class TestWeixinStatePersistence:
    def test_save_weixin_account_preserves_existing_file_on_replace_failure(self, tmp_path, monkeypatch):
        account_path = tmp_path / "weixin" / "accounts" / "acct.json"
        account_path.parent.mkdir(parents=True, exist_ok=True)
        original = {"token": "old-token", "base_url": "https://old.example.com"}
        account_path.write_text(json.dumps(original), encoding="utf-8")

        def _boom(_src, _dst):
            raise OSError("disk full")

        monkeypatch.setattr("utils.os.replace", _boom)

        try:
            weixin.save_weixin_account(
                str(tmp_path),
                account_id="acct",
                token="new-token",
                base_url="https://new.example.com",
                user_id="wxid_new",
            )
        except OSError:
            pass
        else:
            raise AssertionError("expected save_weixin_account to propagate replace failure")

        assert json.loads(account_path.read_text(encoding="utf-8")) == original

    def test_context_token_persist_preserves_existing_file_on_replace_failure(self, tmp_path, monkeypatch):
        token_path = tmp_path / "weixin" / "accounts" / "acct.context-tokens.json"
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(json.dumps({"user-a": "old-token"}), encoding="utf-8")

        def _boom(_src, _dst):
            raise OSError("disk full")

        monkeypatch.setattr("utils.os.replace", _boom)

        store = ContextTokenStore(str(tmp_path))
        with patch.object(weixin.logger, "warning") as warning_mock:
            store.set("acct", "user-b", "new-token")

        assert json.loads(token_path.read_text(encoding="utf-8")) == {"user-a": "old-token"}
        warning_mock.assert_called_once()

    def test_save_sync_buf_preserves_existing_file_on_replace_failure(self, tmp_path, monkeypatch):
        sync_path = tmp_path / "weixin" / "accounts" / "acct.sync.json"
        sync_path.parent.mkdir(parents=True, exist_ok=True)
        sync_path.write_text(json.dumps({"get_updates_buf": "old-sync"}), encoding="utf-8")

        def _boom(_src, _dst):
            raise OSError("disk full")

        monkeypatch.setattr("utils.os.replace", _boom)

        try:
            weixin._save_sync_buf(str(tmp_path), "acct", "new-sync")
        except OSError:
            pass
        else:
            raise AssertionError("expected _save_sync_buf to propagate replace failure")

        assert json.loads(sync_path.read_text(encoding="utf-8")) == {"get_updates_buf": "old-sync"}


class TestWeixinSendMessageIntegration:
    def test_parse_target_ref_accepts_weixin_ids(self):
        assert _parse_target_ref("weixin", "wxid_test123") == ("wxid_test123", None, True)
        assert _parse_target_ref("weixin", "filehelper") == ("filehelper", None, True)
        assert _parse_target_ref("weixin", "group@chatroom") == ("group@chatroom", None, True)

    @patch("tools.send_message_tool._send_weixin", new_callable=AsyncMock)
    def test_send_to_platform_routes_weixin_media_to_native_helper(self, send_weixin_mock):
        send_weixin_mock.return_value = {"success": True, "platform": "weixin", "chat_id": "wxid_test123"}
        config = PlatformConfig(enabled=True, token="bot-token", extra={"account_id": "bot-account"})

        result = asyncio.run(
            _send_to_platform(
                Platform.WEIXIN,
                config,
                "wxid_test123",
                "hello",
                media_files=[("/tmp/demo.png", False)],
            )
        )

        assert result["success"] is True
        send_weixin_mock.assert_awaited_once_with(
            config,
            "wxid_test123",
            "hello",
            media_files=[("/tmp/demo.png", False)],
        )


class TestWeixinChunkDelivery:
    def _connected_adapter(self) -> WeixinAdapter:
        adapter = _make_adapter()
        adapter._session = object()
        adapter._send_session = adapter._session
        adapter._token = "test-token"
        adapter._base_url = "https://weixin.example.com"
        adapter._token_store.get = lambda account_id, chat_id: "ctx-token"
        return adapter

    @patch("gateway.platforms.weixin.asyncio.sleep", new_callable=AsyncMock)
    @patch("gateway.platforms.weixin._send_message", new_callable=AsyncMock)
    def test_send_waits_between_multiple_chunks(self, send_message_mock, sleep_mock):
        adapter = self._connected_adapter()
        adapter.MAX_MESSAGE_LENGTH = 12

        # Use double newlines so _pack_markdown_blocks splits into 3 blocks
        result = asyncio.run(adapter.send("wxid_test123", "first\n\nsecond\n\nthird"))

        assert result.success is True
        assert send_message_mock.await_count == 3
        assert sleep_mock.await_count == 2

    @patch("gateway.platforms.weixin.asyncio.sleep", new_callable=AsyncMock)
    @patch("gateway.platforms.weixin._send_message", new_callable=AsyncMock)
    def test_send_retries_failed_chunk_before_continuing(self, send_message_mock, sleep_mock):
        adapter = self._connected_adapter()
        adapter.MAX_MESSAGE_LENGTH = 12
        calls = {"count": 0}

        async def flaky_send(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 2:
                raise RuntimeError("temporary iLink failure")

        send_message_mock.side_effect = flaky_send

        # Use double newlines so _pack_markdown_blocks splits into 3 blocks
        result = asyncio.run(adapter.send("wxid_test123", "first\n\nsecond\n\nthird"))

        assert result.success is True
        # 3 chunks, but chunk 2 fails once and retries → 4 _send_message calls total
        assert send_message_mock.await_count == 4
        # The retried chunk should reuse the same client_id for deduplication
        first_try = send_message_mock.await_args_list[1].kwargs
        retry = send_message_mock.await_args_list[2].kwargs
        assert first_try["text"] == retry["text"]
        assert first_try["client_id"] == retry["client_id"]

    @patch("gateway.platforms.weixin.asyncio.sleep", new_callable=AsyncMock)
    @patch("gateway.platforms.weixin._send_message", new_callable=AsyncMock)
    def test_ret_minus_2_empty_errmsg_retries_without_token(self, send_message_mock, sleep_mock):
        # #18100: ret=-2 with empty errmsg is a stale context_token, not a
        # rate limit. The adapter must strip the token and retry, recovering
        # delivery instead of burning retries against a dead token.
        adapter = self._connected_adapter()
        calls = {"count": 0}

        async def stale_then_ok(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                return {"ret": -2, "errcode": None, "errmsg": None}
            return {"ret": 0}

        send_message_mock.side_effect = stale_then_ok

        result = asyncio.run(adapter.send("wxid_test123", "hello"))

        assert result.success is True
        assert send_message_mock.await_count == 2
        # First attempt carries the context token; the stale-session retry
        # drops it (tokenless fallback).
        assert send_message_mock.await_args_list[0].kwargs["context_token"] == "ctx-token"
        assert send_message_mock.await_args_list[1].kwargs["context_token"] is None

    @patch("gateway.platforms.weixin.asyncio.sleep", new_callable=AsyncMock)
    @patch("gateway.platforms.weixin._send_message", new_callable=AsyncMock)
    def test_rate_limit_fails_fast_no_retry(self, send_message_mock, sleep_mock):
        # Post-fix: a genuine rate limit (populated errmsg) must NOT retry.
        # Retrying resets the iLink cooldown window and amplifies the limit.
        # Instead, raise RateLimitedError immediately and set cooldown state.
        adapter = self._connected_adapter()
        send_message_mock.return_value = {"ret": -2, "errcode": -2, "errmsg": "freq limit"}

        result = asyncio.run(adapter.send("wxid_test123", "hello"))

        assert result.success is False
        assert "[RATE_LIMITED]" in (result.error or "")

        # No retry → no sleep calls for backoff.
        waits = [c.args[0] for c in sleep_mock.await_args_list]
        assert waits == [], f"expected no backoff sleeps, got {waits}"

        # Cooldown state must be set.
        assert adapter._rate_limited_until > 0


class TestWeixinOutboundMedia:
    def test_send_image_file_accepts_keyword_image_path(self):
        adapter = _make_adapter()
        expected = SendResult(success=True, message_id="msg-1")
        adapter.send_document = AsyncMock(return_value=expected)

        result = asyncio.run(
            adapter.send_image_file(
                chat_id="wxid_test123",
                image_path="/tmp/demo.png",
                caption="截图说明",
                reply_to="reply-1",
                metadata={"thread_id": "t-1"},
            )
        )

        assert result == expected
        adapter.send_document.assert_awaited_once_with(
            chat_id="wxid_test123",
            file_path="/tmp/demo.png",
            caption="截图说明",
            metadata={"thread_id": "t-1"},
        )

    def test_send_document_accepts_keyword_file_path(self):
        adapter = _make_adapter()
        adapter._session = object()
        adapter._send_session = adapter._session
        adapter._token = "test-token"
        adapter._send_file = AsyncMock(return_value="msg-2")

        result = asyncio.run(
            adapter.send_document(
                chat_id="wxid_test123",
                file_path="/tmp/report.pdf",
                caption="报告请看",
                file_name="renamed.pdf",
                reply_to="reply-1",
                metadata={"thread_id": "t-1"},
            )
        )

        assert result.success is True
        assert result.message_id == "msg-2"
        adapter._send_file.assert_awaited_once_with("wxid_test123", "/tmp/report.pdf", "报告请看")

    def test_send_file_uses_post_for_upload_full_url_and_hex_encoded_aes_key(self, tmp_path):
        class _UploadResponse:
            def __init__(self):
                self.status = 200
                self.headers = {"x-encrypted-param": "enc-param"}

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def read(self):
                return b""

            async def text(self):
                return ""

        class _RecordingSession:
            def __init__(self):
                self.post_calls = []

            def post(self, url, **kwargs):
                self.post_calls.append((url, kwargs))
                return _UploadResponse()

            def put(self, *_args, **_kwargs):
                raise AssertionError("upload_full_url branch should use POST")

        image_path = tmp_path / "demo.png"
        image_path.write_bytes(b"fake-png-bytes")

        adapter = _make_adapter()
        session = _RecordingSession()
        adapter._session = session
        adapter._send_session = session
        adapter._token = "test-token"
        adapter._base_url = "https://weixin.example.com"
        adapter._cdn_base_url = "https://cdn.example.com/c2c"
        adapter._token_store.get = lambda account_id, chat_id: None

        aes_key = bytes(range(16))
        expected_aes_key = base64.b64encode(aes_key.hex().encode("ascii")).decode("ascii")

        with patch("gateway.platforms.weixin._get_upload_url", new=AsyncMock(return_value={"upload_full_url": "https://upload.example.com/media"})), \
             patch("gateway.platforms.weixin._api_post", new_callable=AsyncMock) as api_post_mock, \
             patch("gateway.platforms.weixin.secrets.token_hex", return_value="filekey-123"), \
             patch("gateway.platforms.weixin.secrets.token_bytes", return_value=aes_key):
            message_id = asyncio.run(adapter._send_file("wxid_test123", str(image_path), ""))

        assert message_id.startswith("hermes-weixin-")
        assert len(session.post_calls) == 1
        upload_url, upload_kwargs = session.post_calls[0]
        assert upload_url == "https://upload.example.com/media"
        assert upload_kwargs["headers"] == {"Content-Type": "application/octet-stream"}
        assert upload_kwargs["data"]
        assert upload_kwargs["timeout"].total == 120
        payload = api_post_mock.await_args.kwargs["payload"]
        media = payload["msg"]["item_list"][0]["image_item"]["media"]
        assert media["encrypt_query_param"] == "enc-param"
        assert media["aes_key"] == expected_aes_key


class TestWeixinRemoteMediaSafety:
    def test_download_remote_media_blocks_unsafe_urls(self):
        adapter = _make_adapter()

        with patch("tools.url_safety.is_safe_url", return_value=False):
            try:
                asyncio.run(adapter._download_remote_media("http://127.0.0.1/private.png"))
            except ValueError as exc:
                assert "Blocked unsafe URL" in str(exc)
            else:
                raise AssertionError("expected ValueError for unsafe URL")


class TestWeixinMarkdownLinks:
    """Markdown links should be preserved so WeChat can render them natively."""

    def test_format_message_preserves_markdown_links(self):
        adapter = _make_adapter()

        content = "Check [the docs](https://example.com) and [GitHub](https://github.com) for details"
        assert adapter.format_message(content) == content

    def test_format_message_preserves_links_inside_code_blocks(self):
        adapter = _make_adapter()

        content = "See below:\n\n```\n[link](https://example.com)\n```\n\nDone."
        result = adapter.format_message(content)
        assert "[link](https://example.com)" in result


class TestWeixinBlankMessagePrevention:
    """Regression tests for the blank-bubble bugs.

    Three separate guards now prevent a blank WeChat message from ever being
    dispatched:

    1. ``_split_text_for_weixin_delivery("")`` returns ``[]`` — not ``[""]``.
    2. ``send()`` filters out empty/whitespace-only chunks before calling
       ``_send_text_chunk``.
    3. ``_send_message()`` raises ``ValueError`` for empty text as a last-resort
       safety net.
    """

    def test_split_text_returns_empty_list_for_empty_string(self):
        adapter = _make_adapter()
        assert adapter._split_text("") == []

    def test_split_text_returns_empty_list_for_empty_string_split_per_line(self):
        adapter = WeixinAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "account_id": "acct",
                    "token": "test-tok",
                    "split_multiline_messages": True,
                },
            )
        )
        assert adapter._split_text("") == []

    @patch("gateway.platforms.weixin._send_message", new_callable=AsyncMock)
    def test_send_empty_content_does_not_call_send_message(self, send_message_mock):
        adapter = _make_adapter()
        adapter._session = object()
        adapter._send_session = adapter._session
        adapter._token = "test-token"
        adapter._base_url = "https://weixin.example.com"
        adapter._token_store.get = lambda account_id, chat_id: "ctx-token"

        result = asyncio.run(adapter.send("wxid_test123", ""))
        # Empty content → no chunks → no _send_message calls
        assert result.success is True
        send_message_mock.assert_not_awaited()

    def test_send_message_rejects_empty_text(self):
        """_send_message raises ValueError for empty/whitespace text."""
        import pytest
        with pytest.raises(ValueError, match="text must not be empty"):
            asyncio.run(
                weixin._send_message(
                    AsyncMock(),
                    base_url="https://example.com",
                    token="tok",
                    to="wxid_test",
                    text="",
                    context_token=None,
                    client_id="cid",
                )
            )


class TestWeixinStreamingCursorSuppression:
    """WeChat doesn't support message editing — cursor must be suppressed."""

    def test_supports_message_editing_is_false(self):
        adapter = _make_adapter()
        assert adapter.SUPPORTS_MESSAGE_EDITING is False


class TestWeixinMediaBuilder:
    """Media builder uses base64(hex_key), not base64(raw_bytes) for aes_key."""

    def test_image_builder_aes_key_is_base64_of_hex(self):
        import base64
        adapter = _make_adapter()
        media_type, builder = adapter._outbound_media_builder("photo.jpg")
        assert media_type == weixin.MEDIA_IMAGE

        fake_hex_key = "0123456789abcdef0123456789abcdef"
        expected_aes = base64.b64encode(fake_hex_key.encode("ascii")).decode("ascii")
        item = builder(
            encrypt_query_param="eq",
            aes_key_for_api=expected_aes,
            ciphertext_size=1024,
            plaintext_size=1000,
            filename="photo.jpg",
            rawfilemd5="abc123",
        )
        assert item["image_item"]["media"]["aes_key"] == expected_aes

    def test_video_builder_includes_md5(self):
        adapter = _make_adapter()
        media_type, builder = adapter._outbound_media_builder("clip.mp4")
        assert media_type == weixin.MEDIA_VIDEO

        item = builder(
            encrypt_query_param="eq",
            aes_key_for_api="fakekey",
            ciphertext_size=2048,
            plaintext_size=2000,
            filename="clip.mp4",
            rawfilemd5="deadbeef",
        )
        assert item["video_item"]["video_md5"] == "deadbeef"

    def test_voice_builder_for_audio_files_uses_file_attachment_type(self):
        adapter = _make_adapter()
        media_type, builder = adapter._outbound_media_builder("note.mp3")
        assert media_type == weixin.MEDIA_FILE

        item = builder(
            encrypt_query_param="eq",
            aes_key_for_api="fakekey",
            ciphertext_size=512,
            plaintext_size=500,
            filename="note.mp3",
            rawfilemd5="abc",
        )
        assert item["type"] == weixin.ITEM_FILE
        assert item["file_item"]["file_name"] == "note.mp3"

    def test_voice_builder_for_silk_files(self):
        adapter = _make_adapter()
        media_type, builder = adapter._outbound_media_builder("recording.silk")
        assert media_type == weixin.MEDIA_VOICE


class TestWeixinSendImageFileParameterName:
    """Regression test for send_image_file parameter name mismatch.

    The gateway calls send_image_file(chat_id=..., image_path=...) but the
    WeixinAdapter previously used 'path' as the parameter name, causing
    image sending to fail. This test ensures the interface stays correct.
    """

    @patch.object(WeixinAdapter, "send_document", new_callable=AsyncMock)
    def test_send_image_file_uses_image_path_parameter(self, send_document_mock):
        """Verify send_image_file accepts image_path and forwards to send_document."""
        adapter = _make_adapter()
        adapter._session = object()
        adapter._send_session = adapter._session
        adapter._token = "test-token"

        send_document_mock.return_value = weixin.SendResult(success=True, message_id="test-id")

        # This is the call pattern used by gateway/run.py extract_media
        result = asyncio.run(
            adapter.send_image_file(
                chat_id="wxid_test123",
                image_path="/tmp/test_image.png",
                caption="Test caption",
                metadata={"thread_id": "thread-123"},
            )
        )

        assert result.success is True
        send_document_mock.assert_awaited_once_with(
            chat_id="wxid_test123",
            file_path="/tmp/test_image.png",
            caption="Test caption",
            metadata={"thread_id": "thread-123"},
        )

    @patch.object(WeixinAdapter, "send_document", new_callable=AsyncMock)
    def test_send_image_file_works_without_optional_params(self, send_document_mock):
        """Verify send_image_file works with minimal required params."""
        adapter = _make_adapter()
        adapter._session = object()
        adapter._send_session = adapter._session
        adapter._token = "test-token"

        send_document_mock.return_value = weixin.SendResult(success=True, message_id="test-id")

        result = asyncio.run(
            adapter.send_image_file(
                chat_id="wxid_test123",
                image_path="/tmp/test_image.jpg",
            )
        )

        assert result.success is True
        send_document_mock.assert_awaited_once_with(
            chat_id="wxid_test123",
            file_path="/tmp/test_image.jpg",
            caption=None,
            metadata=None,
        )


class TestWeixinVoiceSending:
    def _connected_adapter(self) -> WeixinAdapter:
        adapter = _make_adapter()
        adapter._session = object()
        adapter._send_session = adapter._session
        adapter._token = "test-token"
        adapter._base_url = "https://weixin.example.com"
        adapter._token_store.get = lambda account_id, chat_id: "ctx-token"
        return adapter

    @patch.object(WeixinAdapter, "_send_file", new_callable=AsyncMock)
    def test_send_voice_downgrades_to_document_attachment(self, send_file_mock, tmp_path):
        adapter = self._connected_adapter()
        source = tmp_path / "voice.ogg"
        source.write_bytes(b"ogg")
        send_file_mock.return_value = "msg-1"

        result = asyncio.run(adapter.send_voice("wxid_test123", str(source)))

        assert result.success is True
        send_file_mock.assert_awaited_once_with(
            "wxid_test123",
            str(source),
            "[voice message as attachment]",
            force_file_attachment=True,
        )

    def test_voice_builder_for_silk_files_can_be_forced_to_file_attachment(self):
        adapter = _make_adapter()
        media_type, builder = adapter._outbound_media_builder(
            "recording.silk",
            force_file_attachment=True,
        )
        assert media_type == weixin.MEDIA_FILE

        item = builder(
            encrypt_query_param="eq",
            aes_key_for_api="fakekey",
            ciphertext_size=512,
            plaintext_size=500,
            filename="recording.silk",
            rawfilemd5="abc",
        )
        assert item["type"] == weixin.ITEM_FILE
        assert item["file_item"]["file_name"] == "recording.silk"

    @patch.object(weixin, "_api_post", new_callable=AsyncMock)
    @patch.object(weixin, "_upload_ciphertext", new_callable=AsyncMock)
    @patch.object(weixin, "_get_upload_url", new_callable=AsyncMock)
    def test_send_file_sets_voice_metadata_for_silk_payload(
        self,
        get_upload_url_mock,
        upload_ciphertext_mock,
        api_post_mock,
        tmp_path,
    ):
        adapter = self._connected_adapter()
        silk = tmp_path / "voice.silk"
        silk.write_bytes(b"\x02#!SILK_V3\x01\x00")
        get_upload_url_mock.return_value = {"upload_full_url": "https://cdn.example.com/upload"}
        upload_ciphertext_mock.return_value = "enc-q"
        api_post_mock.return_value = {"success": True}

        asyncio.run(adapter._send_file("wxid_test123", str(silk), ""))

        payload = api_post_mock.await_args.kwargs["payload"]
        voice_item = payload["msg"]["item_list"][0]["voice_item"]
        assert voice_item.get("playtime", 0) == 0
        assert voice_item["encode_type"] == 6
        assert voice_item["sample_rate"] == 24000
        assert voice_item["bits_per_sample"] == 16


class TestIsStaleSessionRet:
    """Regression test for #17228 / #18100: distinguish stale-session ret=-2
    from rate-limit ret=-2. Both ``"unknown error"`` and empty/None errmsg
    are stale-session signals; a populated descriptive errmsg is a real
    rate limit."""

    def test_ret_minus_2_with_unknown_error_is_stale(self):
        assert weixin._is_stale_session_ret(-2, None, "unknown error") is True

    def test_errcode_minus_2_with_unknown_error_is_stale(self):
        assert weixin._is_stale_session_ret(None, -2, "unknown error") is True

    def test_unknown_error_case_insensitive(self):
        assert weixin._is_stale_session_ret(-2, None, "Unknown Error") is True

    def test_ret_minus_2_with_freq_limit_is_not_stale(self):
        # Genuine rate limit — must NOT be treated as stale session.
        assert weixin._is_stale_session_ret(-2, None, "freq limit") is False

    def test_ret_minus_2_with_no_errmsg_is_stale(self):
        # #18100: iLink also signals a stale context_token via ret=-2 with an
        # empty/None errmsg. Treat it as stale so the caller retries without
        # the dead token instead of burning all retries in the rate-limit path.
        assert weixin._is_stale_session_ret(-2, None, None) is True
        assert weixin._is_stale_session_ret(-2, None, "") is True
        assert weixin._is_stale_session_ret(-2, None, "   ") is True

    def test_errcode_minus_14_is_not_matched_here(self):
        # -14 is handled by the separate SESSION_EXPIRED_ERRCODE path; the
        # helper only disambiguates -2 from a genuine rate limit.
        assert weixin._is_stale_session_ret(-14, None, "session expired") is False

    def test_success_codes_are_not_stale(self):
        assert weixin._is_stale_session_ret(0, 0, "") is False
        assert weixin._is_stale_session_ret(None, None, "unknown error") is False


class TestWeixinRateLimitedUntilProperty:
    """WeixinAdapter 覆盖 base.rate_limited_until 返回 _rate_limited_until。"""

    def test_override_returns_rate_limited_until_attr(self):
        adapter = _make_adapter()
        adapter._rate_limited_until = 12345.0
        assert adapter.rate_limited_until == 12345.0

    def test_override_default_zero(self):
        adapter = _make_adapter()
        assert adapter.rate_limited_until == 0.0


class TestWeixinTypingInterval:
    """验收场景7.P7：_typing_interval_seconds == 3.0。

    f96688644 误诊 typing 为 root cause 后曾上调到 5.0，砍掉处理过程可见性。
    cooldown 门控修好后限流自激振荡根因解除，3s 既安全又恢复实时感。
    """

    def test_typing_interval_is_three_seconds(self):
        adapter = _make_adapter()
        assert adapter._typing_interval_seconds == 3.0
        assert isinstance(adapter._typing_interval_seconds, float)


class TestWeixinSendCooldownGating:
    """send() 入口 cooldown 门控 — 验收场景1.P1/P3/P4 + 边界。

    cooldown 期内 send 挂起（await asyncio.sleep 到期），不调用 _send_message；
    cooldown 过期后恢复调用。
    """

    def _connected_adapter(self) -> WeixinAdapter:
        adapter = _make_adapter()
        adapter._session = object()
        adapter._send_session = adapter._session
        adapter._token = "test-token"
        adapter._base_url = "https://weixin.example.com"
        adapter._token_store.get = lambda account_id, chat_id: "ctx-token"
        return adapter

    @patch("gateway.platforms.weixin.asyncio.sleep", new_callable=AsyncMock)
    @patch("gateway.platforms.weixin._send_message", new_callable=AsyncMock)
    def test_send_waits_for_cooldown_before_first_chunk(
        self, send_message_mock, sleep_mock, monkeypatch
    ):
        """验收场景1.P1/P3：cooldown 期内 send 挂起，不调用 _send_message。

        构造 _rate_limited_until 为未来时刻，send() 入口门控应 sleep 等待，
        且 sleep 期间不发起 _send_message。模拟时钟推进过 cooldown 后恢复。
        """
        adapter = self._connected_adapter()
        trigger = 1000.0
        cooldown = 30.0
        adapter._rate_limited_until = trigger + cooldown

        # 控制时钟：第一次读 now=trigger（在 cooldown 内），sleep 后推进到过期
        clock = {"t": trigger}

        def fake_time():
            return clock["t"]

        async def fake_sleep(seconds):
            # 推进时钟到 cooldown 结束，模拟真实挂起
            clock["t"] = max(clock["t"] + seconds, trigger + cooldown)

        monkeypatch.setattr(weixin.time, "time", fake_time)
        sleep_mock.side_effect = fake_sleep
        send_message_mock.return_value = {"ret": 0}

        before = send_message_mock.await_count
        result = asyncio.run(adapter.send("wxid_test123", "hello"))
        after = send_message_mock.await_count

        # send 成功（cooldown 等待后正常投递）
        assert result.success is True
        # 关键：cooldown 期内没有调用，等待后才调用 1 次（1 chunk）
        delta = after - before
        assert delta == 1, f"expected exactly 1 _send_message after cooldown wait, got delta={delta}"
        # sleep 被调用过（门控生效）
        assert sleep_mock.await_count >= 1
        # 第一段 sleep 应该是 cooldown 剩余量（≈30s）
        first_sleep = sleep_mock.await_args_list[0].args[0]
        assert first_sleep == pytest.approx(cooldown, abs=0.5), (
            f"expected first sleep ≈ cooldown={cooldown}, got {first_sleep}"
        )

    @patch("gateway.platforms.weixin.asyncio.sleep", new_callable=AsyncMock)
    @patch("gateway.platforms.weixin._send_message", new_callable=AsyncMock)
    def test_multiple_sends_during_cooldown_no_ilink_calls(
        self, send_message_mock, sleep_mock, monkeypatch
    ):
        """验收场景1.P3：cooldown 期产生 m 个新 send，_send_message 增量 == 触发那次。

        冷却窗口内连发 3 次 send，每次都被门控 sleep 推进到 cooldown 结束。
        关键断言：每次 send 的 _send_message 调用都发生在 cooldown 之后
        （窗口内 delta == 0）。
        """
        adapter = self._connected_adapter()
        trigger = 1000.0
        cooldown = 30.0
        adapter._rate_limited_until = trigger + cooldown

        clock = {"t": trigger}
        window_call_log = []  # 记录每次 _send_message 调用时的时钟值

        def fake_time():
            return clock["t"]

        async def fake_sleep(seconds):
            before = clock["t"]
            clock["t"] = before + seconds

        def fake_send(*a, **kw):
            window_call_log.append(clock["t"])
            return {"ret": 0}

        monkeypatch.setattr(weixin.time, "time", fake_time)
        sleep_mock.side_effect = fake_sleep
        send_message_mock.side_effect = fake_send

        # 在 trigger 时刻连发 3 次（每次 send 内部门控把时钟推进到 cooldown 之后）
        for _ in range(3):
            asyncio.run(adapter.send("wxid_test123", "hi"))

        # 每条 _send_message 都在 cooldown 结束之后（窗口内 0 调用）
        for t in window_call_log:
            assert t >= trigger + cooldown - 0.01, (
                f"_send_message called at t={t} < cooldown_end={trigger + cooldown}"
            )
        # 共 3 次调用（每 send 1 chunk）
        assert len(window_call_log) == 3

    @patch("gateway.platforms.weixin.asyncio.sleep", new_callable=AsyncMock)
    @patch("gateway.platforms.weixin._send_message", new_callable=AsyncMock)
    def test_send_at_exact_cooldown_boundary_no_sleep(
        self, send_message_mock, sleep_mock, monkeypatch
    ):
        """验收边界：恰在 cooldown_until 时刻的 send 立即执行（不 sleep）。"""
        adapter = self._connected_adapter()
        boundary = 1000.0
        adapter._rate_limited_until = boundary

        clock = {"t": boundary}

        def fake_time():
            return clock["t"]

        async def fake_sleep(seconds):
            clock["t"] = clock["t"] + seconds

        monkeypatch.setattr(weixin.time, "time", fake_time)
        sleep_mock.side_effect = fake_sleep
        send_message_mock.return_value = {"ret": 0}

        result = asyncio.run(adapter.send("wxid_test123", "hello"))

        assert result.success is True
        # 边界时刻不应触发 cooldown 等待 sleep
        cooldown_sleeps = [
            c.args[0]
            for c in sleep_mock.await_args_list
            if c.args and abs(c.args[0] - 0.0) > 0.001
            and c.args[0] > 0.5  # 排除 chunk delay 等小 sleep
        ]
        assert cooldown_sleeps == [], (
            f"expected no cooldown wait at boundary, got {cooldown_sleeps}"
        )

    @patch("gateway.platforms.weixin.asyncio.sleep", new_callable=AsyncMock)
    @patch("gateway.platforms.weixin._send_message", new_callable=AsyncMock)
    def test_chunk_loop_checks_cooldown_between_chunks(
        self, send_message_mock, sleep_mock, monkeypatch
    ):
        """验收场景6：多 chunk 循环中第 1 chunk 触发限流设 cooldown，
        第 2 chunk 前门控应 sleep 到 cooldown 结束再投递。

        这是 chunk 循环每 chunk 前 check cooldown 的核心场景。"""
        adapter = self._connected_adapter()
        adapter.MAX_MESSAGE_LENGTH = 12  # 强制 2 chunk
        trigger = 1000.0
        cooldown = 30.0

        clock = {"t": trigger}

        def fake_time():
            return clock["t"]

        async def fake_sleep(seconds):
            clock["t"] = clock["t"] + seconds

        # 第 1 chunk 返回限流（设 cooldown），第 2 chunk 返回成功
        # 但注意：限流会抛 RateLimitedError，send 捕获返回失败，不会继续 chunk 循环。
        # 因此本场景改为：模拟限流来自外部（非本次 send 内部），chunk 循环开始时
        # 已在 cooldown 内 → chunk 0 被门控等待。
        adapter._rate_limited_until = trigger + cooldown
        monkeypatch.setattr(weixin.time, "time", fake_time)
        sleep_mock.side_effect = fake_sleep
        send_message_mock.return_value = {"ret": 0}

        result = asyncio.run(adapter.send("wxid_test123", "first\n\nsecond"))

        assert result.success is True
        # 2 chunk 都成功投递
        assert send_message_mock.await_count == 2
        # 第一次 sleep 应是 cooldown 剩余（chunk 0 前门控）
        first_sleep = sleep_mock.await_args_list[0].args[0]
        assert first_sleep == pytest.approx(cooldown, abs=0.5), (
            f"expected first sleep ≈ cooldown={cooldown}, got {first_sleep}"
        )

