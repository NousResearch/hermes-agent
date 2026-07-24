"""Tests for the Weixin platform adapter."""

import asyncio
import json
import os
from unittest.mock import AsyncMock, patch

import pytest

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

        assert chunks == ["**结论**\n这是正文"]

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


class TestWeixinSessionExpired:
    """Stale session detection: _is_stale_session_ret + ALERT_SCRIPT wiring."""

    def test_is_stale_session_ret_with_unknown_error(self):
        """ret=-2 + errmsg='unknown error' -> stale session."""
        from gateway.platforms.weixin import _is_stale_session_ret
        assert _is_stale_session_ret(-2, None, "unknown error") is True

    def test_is_stale_session_ret_with_empty_errmsg(self):
        """ret=-2 + empty errmsg -> NOT stale (only 'unknown error' triggers)."""
        from gateway.platforms.weixin import _is_stale_session_ret
        assert _is_stale_session_ret(-2, None, "") is False
        assert _is_stale_session_ret(-2, None, None) is False

    def test_is_stale_session_ret_with_rate_limit_errmsg(self):
        """ret=-2 + errmsg='rate limited' -> NOT stale (genuine rate limit)."""
        from gateway.platforms.weixin import _is_stale_session_ret
        assert _is_stale_session_ret(-2, None, "rate limited") is False

    def test_is_stale_session_ret_with_errcode_minus_14(self):
        """errcode=-14 -> NOT caught by _is_stale_session_ret (only checks -2)."""
        from gateway.platforms.weixin import _is_stale_session_ret
        assert _is_stale_session_ret(None, -14, None) is False

    def test_alert_script_constant(self):
        """ALERT_SCRIPT is defined (may be empty if env var not set)."""
        from gateway.platforms.weixin import ALERT_SCRIPT
        assert isinstance(ALERT_SCRIPT, str)

    def test_fire_alert_no_script(self):
        """_fire_alert is a no-op when ALERT_SCRIPT is empty."""
        from gateway.platforms.weixin import _fire_alert
        # Should not raise even with no script configured
        _fire_alert("test_event", "test_detail")


class TestWeixinPollStaleDetection:
    """_poll_loop stale detection: uses upstream _is_stale_session_ret."""

    def _connected_adapter(self) -> WeixinAdapter:
        adapter = _make_adapter()
        adapter._session = object()
        adapter._poll_session = object()
        adapter._token = "test-token"
        adapter._base_url = "https://weixin.example.com"
        adapter._token_store.get = lambda account_id, chat_id: "ctx-token"
        adapter._running = True
        adapter._hermes_home = "/tmp/test-hermes"
        return adapter

    @patch("gateway.platforms.weixin.asyncio.sleep", new_callable=AsyncMock)
    @patch("gateway.platforms.weixin._get_updates", new_callable=AsyncMock)
    def test_poll_stale_on_ret_minus_2(self, get_updates_mock, sleep_mock):
        """ret=-2 with 'unknown error' -> stale detection -> 600s pause."""

        call_count = {"n": 0}
        async def side_effect(*a, **kw):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return {"ret": -2, "errmsg": "unknown error"}
            raise asyncio.CancelledError()

        get_updates_mock.side_effect = side_effect
        adapter = self._connected_adapter()
        try:
            asyncio.run(adapter._poll_loop())
        except (asyncio.CancelledError, RuntimeError):
            pass
        sleep_calls = [a.args[0] for a in sleep_mock.await_args_list]
        assert 600 in sleep_calls, f"Expected 600s sleep, got {sleep_mock.await_args_list}"

    @patch("gateway.platforms.weixin.asyncio.sleep", new_callable=AsyncMock)
    @patch("gateway.platforms.weixin._get_updates", new_callable=AsyncMock)
    def test_poll_stale_on_ret_minus_3(self, get_updates_mock, sleep_mock):
        """ret=-3 -> NOT stale (only -2/'unknown error' triggers), normal backoff."""
        call_count = {"n": 0}
        async def side_effect(*a, **kw):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return {"ret": -3, "errmsg": "unknown error"}
            raise asyncio.CancelledError()

        get_updates_mock.side_effect = side_effect
        adapter = self._connected_adapter()
        try:
            asyncio.run(adapter._poll_loop())
        except (asyncio.CancelledError, RuntimeError):
            pass
        sleep_calls = [a.args[0] for a in sleep_mock.await_args_list]
        # ret=-3 is treated as regular failure, normal backoff (2s), not 600s
        assert 600 not in sleep_calls, f"ret=-3 should NOT trigger stale pause, got {sleep_mock.await_args_list}"

    @patch("gateway.platforms.weixin.asyncio.sleep", new_callable=AsyncMock)
    @patch("gateway.platforms.weixin._get_updates", new_callable=AsyncMock)
    def test_poll_stale_on_errcode_minus_14(self, get_updates_mock, sleep_mock):
        """errcode=-14 (SESSION_EXPIRED_ERRCODE) -> stale detection -> 600s pause."""
        from gateway.platforms.weixin import SESSION_EXPIRED_ERRCODE

        call_count = {"n": 0}
        async def side_effect(*a, **kw):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return {"ret": 0, "errcode": SESSION_EXPIRED_ERRCODE}
            raise asyncio.CancelledError()

        get_updates_mock.side_effect = side_effect
        adapter = self._connected_adapter()
        try:
            asyncio.run(adapter._poll_loop())
        except (asyncio.CancelledError, RuntimeError):
            pass
        sleep_calls = [a.args[0] for a in sleep_mock.await_args_list]
        assert 600 in sleep_calls, f"Expected 600s sleep, got {sleep_mock.await_args_list}"

    @patch("gateway.platforms.weixin._save_sync_buf")
    @patch("gateway.platforms.weixin.asyncio.sleep", new_callable=AsyncMock)
    @patch("gateway.platforms.weixin._get_updates", new_callable=AsyncMock)
    def test_poll_stale_clears_sync_buffer(self, get_updates_mock, sleep_mock, save_mock):
        """Stale detection clears sync buffer before retrying."""
        

        call_count = {"n": 0}
        async def side_effect(*a, **kw):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return {"ret": -2, "errmsg": "unknown error"}
            raise asyncio.CancelledError()

        get_updates_mock.side_effect = side_effect
        adapter = self._connected_adapter()
        try:
            asyncio.run(adapter._poll_loop())
        except (asyncio.CancelledError, RuntimeError):
            pass
        save_mock.assert_called()
        # Verify empty buffer was saved
        args = save_mock.call_args
        assert args[0][2] == "" or args[1].get("sync_buf") == "" or args[0][-1] == "", f"Expected empty sync_buf, got {args}"
