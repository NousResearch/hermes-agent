"""Tests for Telegram reply_to_mode functionality.

Covers the threading behavior control for multi-chunk replies:
- "off": Never thread replies to original message
- "first": Only first chunk threads (default)
- "all": All chunks thread to original message
"""
import os
import sys
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from gateway.config import PlatformConfig, GatewayConfig, Platform, _apply_env_overrides


def _ensure_telegram_mock():
    """Mock the telegram package if it's not installed."""
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return
    mod = MagicMock()
    mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    mod.constants.ChatType.GROUP = "group"
    mod.constants.ChatType.SUPERGROUP = "supergroup"
    mod.constants.ChatType.CHANNEL = "channel"
    mod.constants.ChatType.PRIVATE = "private"
    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, mod)


_ensure_telegram_mock()

from gateway.platforms.telegram import TelegramAdapter  # noqa: E402


@pytest.fixture()
def adapter_factory():
    """Factory to create TelegramAdapter with custom reply_to_mode."""
    def create(reply_to_mode: str = "first"):
        config = PlatformConfig(enabled=True, token="test-token", reply_to_mode=reply_to_mode)
        return TelegramAdapter(config)
    return create


class TestReplyToModeConfig:
    """Tests for reply_to_mode configuration loading."""

    def test_default_mode_is_first(self, adapter_factory):
        adapter = adapter_factory()
        assert adapter._reply_to_mode == "first"

    def test_off_mode(self, adapter_factory):
        adapter = adapter_factory(reply_to_mode="off")
        assert adapter._reply_to_mode == "off"

    def test_first_mode(self, adapter_factory):
        adapter = adapter_factory(reply_to_mode="first")
        assert adapter._reply_to_mode == "first"

    def test_all_mode(self, adapter_factory):
        adapter = adapter_factory(reply_to_mode="all")
        assert adapter._reply_to_mode == "all"

    def test_invalid_mode_stored_as_is(self, adapter_factory):
        """Invalid modes are stored but _should_thread_reply handles them."""
        adapter = adapter_factory(reply_to_mode="invalid")
        assert adapter._reply_to_mode == "invalid"

    def test_none_mode_defaults_to_first(self):
        config = PlatformConfig(enabled=True, token="test-token")
        adapter = TelegramAdapter(config)
        assert adapter._reply_to_mode == "first"

    def test_empty_string_mode_defaults_to_first(self):
        config = PlatformConfig(enabled=True, token="test-token", reply_to_mode="")
        adapter = TelegramAdapter(config)
        assert adapter._reply_to_mode == "first"


class TestShouldThreadReply:
    """Tests for _should_thread_reply method."""

    def test_no_reply_to_returns_false(self, adapter_factory):
        adapter = adapter_factory(reply_to_mode="first")
        assert adapter._should_thread_reply(None, 0) is False
        assert adapter._should_thread_reply("", 0) is False

    def test_off_mode_never_threads(self, adapter_factory):
        adapter = adapter_factory(reply_to_mode="off")
        assert adapter._should_thread_reply("msg-123", 0) is False
        assert adapter._should_thread_reply("msg-123", 1) is False
        assert adapter._should_thread_reply("msg-123", 5) is False

    def test_first_mode_only_first_chunk(self, adapter_factory):
        adapter = adapter_factory(reply_to_mode="first")
        assert adapter._should_thread_reply("msg-123", 0) is True
        assert adapter._should_thread_reply("msg-123", 1) is False
        assert adapter._should_thread_reply("msg-123", 2) is False
        assert adapter._should_thread_reply("msg-123", 10) is False

    def test_all_mode_all_chunks(self, adapter_factory):
        adapter = adapter_factory(reply_to_mode="all")
        assert adapter._should_thread_reply("msg-123", 0) is True
        assert adapter._should_thread_reply("msg-123", 1) is True
        assert adapter._should_thread_reply("msg-123", 2) is True
        assert adapter._should_thread_reply("msg-123", 10) is True

    def test_invalid_mode_falls_back_to_first(self, adapter_factory):
        """Invalid mode behaves like 'first' - only first chunk threads."""
        adapter = adapter_factory(reply_to_mode="invalid")
        assert adapter._should_thread_reply("msg-123", 0) is True
        assert adapter._should_thread_reply("msg-123", 1) is False


class TestSendWithReplyToMode:
    """Tests for send() method respecting reply_to_mode."""

    @pytest.mark.asyncio
    async def test_off_mode_no_reply_threading(self, adapter_factory):
        adapter = adapter_factory(reply_to_mode="off")
        adapter._bot = MagicMock()
        adapter._bot.send_message = AsyncMock(return_value=MagicMock(message_id=1))
        adapter.truncate_message = lambda content, max_len, **kw: ["chunk1", "chunk2", "chunk3"]

        await adapter.send("12345", "test content", reply_to="999")

        for call in adapter._bot.send_message.call_args_list:
            assert call.kwargs.get("reply_to_message_id") is None

    @pytest.mark.asyncio
    async def test_first_mode_only_first_chunk_threads(self, adapter_factory):
        adapter = adapter_factory(reply_to_mode="first")
        adapter._bot = MagicMock()
        adapter._bot.send_message = AsyncMock(return_value=MagicMock(message_id=1))
        adapter.truncate_message = lambda content, max_len, **kw: ["chunk1", "chunk2", "chunk3"]

        await adapter.send("12345", "test content", reply_to="999")

        calls = adapter._bot.send_message.call_args_list
        assert len(calls) == 3
        assert calls[0].kwargs.get("reply_to_message_id") == 999
        assert calls[1].kwargs.get("reply_to_message_id") is None
        assert calls[2].kwargs.get("reply_to_message_id") is None

    @pytest.mark.asyncio
    async def test_all_mode_all_chunks_thread(self, adapter_factory):
        adapter = adapter_factory(reply_to_mode="all")
        adapter._bot = MagicMock()
        adapter._bot.send_message = AsyncMock(return_value=MagicMock(message_id=1))
        adapter.truncate_message = lambda content, max_len, **kw: ["chunk1", "chunk2", "chunk3"]

        await adapter.send("12345", "test content", reply_to="999")

        calls = adapter._bot.send_message.call_args_list
        assert len(calls) == 3
        for call in calls:
            assert call.kwargs.get("reply_to_message_id") == 999

    @pytest.mark.asyncio
    async def test_no_reply_to_param_no_threading(self, adapter_factory):
        adapter = adapter_factory(reply_to_mode="all")
        adapter._bot = MagicMock()
        adapter._bot.send_message = AsyncMock(return_value=MagicMock(message_id=1))
        adapter.truncate_message = lambda content, max_len, **kw: ["chunk1", "chunk2"]

        await adapter.send("12345", "test content", reply_to=None)

        calls = adapter._bot.send_message.call_args_list
        for call in calls:
            assert call.kwargs.get("reply_to_message_id") is None

    @pytest.mark.asyncio
    async def test_single_chunk_respects_mode(self, adapter_factory):
        adapter = adapter_factory(reply_to_mode="first")
        adapter._bot = MagicMock()
        adapter._bot.send_message = AsyncMock(return_value=MagicMock(message_id=1))
        adapter.truncate_message = lambda content, max_len, **kw: ["single chunk"]

        await adapter.send("12345", "test", reply_to="999")

        calls = adapter._bot.send_message.call_args_list
        assert len(calls) == 1
        assert calls[0].kwargs.get("reply_to_message_id") == 999


class TestConfigSerialization:
    """Tests for reply_to_mode serialization."""

    def test_to_dict_includes_reply_to_mode(self):
        config = PlatformConfig(enabled=True, token="test", reply_to_mode="all")
        result = config.to_dict()
        assert result["reply_to_mode"] == "all"

    def test_from_dict_loads_reply_to_mode(self):
        data = {"enabled": True, "token": "test", "reply_to_mode": "off"}
        config = PlatformConfig.from_dict(data)
        assert config.reply_to_mode == "off"

    def test_from_dict_defaults_to_first(self):
        data = {"enabled": True, "token": "test"}
        config = PlatformConfig.from_dict(data)
        assert config.reply_to_mode == "first"


class TestEnvVarOverride:
    """Tests for TELEGRAM_REPLY_TO_MODE environment variable override."""

    def _make_config(self):
        config = GatewayConfig()
        config.platforms[Platform.TELEGRAM] = PlatformConfig(enabled=True, token="test")
        return config

    def test_env_var_sets_off_mode(self):
        config = self._make_config()
        with patch.dict(os.environ, {"TELEGRAM_REPLY_TO_MODE": "off"}, clear=False):
            _apply_env_overrides(config)
        assert config.platforms[Platform.TELEGRAM].reply_to_mode == "off"

    def test_env_var_sets_all_mode(self):
        config = self._make_config()
        with patch.dict(os.environ, {"TELEGRAM_REPLY_TO_MODE": "all"}, clear=False):
            _apply_env_overrides(config)
        assert config.platforms[Platform.TELEGRAM].reply_to_mode == "all"

    def test_env_var_case_insensitive(self):
        config = self._make_config()
        with patch.dict(os.environ, {"TELEGRAM_REPLY_TO_MODE": "ALL"}, clear=False):
            _apply_env_overrides(config)
        assert config.platforms[Platform.TELEGRAM].reply_to_mode == "all"

    def test_env_var_invalid_value_ignored(self):
        config = self._make_config()
        with patch.dict(os.environ, {"TELEGRAM_REPLY_TO_MODE": "banana"}, clear=False):
            _apply_env_overrides(config)
        assert config.platforms[Platform.TELEGRAM].reply_to_mode == "first"

    def test_env_var_empty_value_ignored(self):
        config = self._make_config()
        with patch.dict(os.environ, {"TELEGRAM_REPLY_TO_MODE": ""}, clear=False):
            _apply_env_overrides(config)
        assert config.platforms[Platform.TELEGRAM].reply_to_mode == "first"
