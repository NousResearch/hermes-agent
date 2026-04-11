"""Tests for the Weixin platform adapter."""

import asyncio
import json
import os
from unittest.mock import AsyncMock, patch

from gateway.config import PlatformConfig
from gateway.config import GatewayConfig, HomeChannel, Platform, _apply_env_overrides
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
    def test_format_message_preserves_markdown_and_rewrites_headers(self):
        adapter = _make_adapter()

        content = "# Title\n\n## Plan\n\nUse **bold** and [docs](https://example.com)."

        assert (
            adapter.format_message(content)
            == "【Title】\n\n**Plan**\n\nUse **bold** and [docs](https://example.com)."
        )

    def test_format_message_rewrites_markdown_tables(self):
        adapter = _make_adapter()

        content = (
            "| Setting | Value |\n"
            "| --- | --- |\n"
            "| Timeout | 30s |\n"
            "| Retries | 3 |\n"
        )

        assert adapter.format_message(content) == (
            "- Setting: Timeout\n"
            "  Value: 30s\n"
            "- Setting: Retries\n"
            "  Value: 3"
        )

    def test_format_message_preserves_fenced_code_blocks(self):
        adapter = _make_adapter()

        content = "## Snippet\n\n```python\nprint('hi')\n```"

        assert adapter.format_message(content) == "**Snippet**\n\n```python\nprint('hi')\n```"

    def test_format_message_returns_empty_string_for_none(self):
        adapter = _make_adapter()

        assert adapter.format_message(None) == ""


class TestWeixinChunking:
    def test_split_text_keeps_short_multiline_message_in_single_chunk(self):
        adapter = _make_adapter()

        content = adapter.format_message("第一行\n第二行\n第三行")
        chunks = adapter._split_text(content)

        assert chunks == ["第一行\n第二行\n第三行"]

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
