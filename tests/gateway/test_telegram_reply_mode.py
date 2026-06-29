"""Tests for Telegram reply_to_mode functionality.

Covers the threading behavior control for multi-chunk replies:
- "off": Never thread replies to original message
- "first": Single-chunk replies thread; multi-chunk sends suppress native reply metadata
- "all": All chunks thread to original message
"""
import os
import sys
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from gateway.config import PlatformConfig, GatewayConfig, Platform, _apply_env_overrides, load_gateway_config


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

from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


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
    async def test_first_mode_multipart_reply_suppresses_native_reply(self, adapter_factory):
        adapter = adapter_factory(reply_to_mode="first")
        adapter._bot = MagicMock()
        adapter._bot.send_message = AsyncMock(return_value=MagicMock(message_id=1))
        adapter.truncate_message = lambda content, max_len, **kw: ["chunk1", "chunk2", "chunk3"]

        await adapter.send("12345", "test content", reply_to="999")

        calls = adapter._bot.send_message.call_args_list
        assert len(calls) == 3
        assert calls[0].kwargs.get("reply_to_message_id") is None
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


class TestTelegramYamlConfigLoading:
    """Tests for reply_to_mode loaded from config.yaml telegram section."""

    def _write_config(self, tmp_path, content: str):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(content, encoding="utf-8")
        return hermes_home

    def test_top_level_reply_to_mode_off(self, tmp_path, monkeypatch):
        """YAML 1.1 parses bare 'off' as boolean False — must map back to 'off'."""
        hermes_home = self._write_config(tmp_path, "telegram:\n  reply_to_mode: off\n")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.delenv("TELEGRAM_REPLY_TO_MODE", raising=False)

        load_gateway_config()

        assert os.environ.get("TELEGRAM_REPLY_TO_MODE") == "off"

    def test_top_level_reply_to_mode_all(self, tmp_path, monkeypatch):
        hermes_home = self._write_config(tmp_path, "telegram:\n  reply_to_mode: all\n")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.delenv("TELEGRAM_REPLY_TO_MODE", raising=False)

        load_gateway_config()

        assert os.environ.get("TELEGRAM_REPLY_TO_MODE") == "all"

    def test_extra_reply_to_mode_off(self, tmp_path, monkeypatch):
        """telegram.extra.reply_to_mode is also honoured."""
        hermes_home = self._write_config(
            tmp_path, "telegram:\n  extra:\n    reply_to_mode: \"off\"\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.delenv("TELEGRAM_REPLY_TO_MODE", raising=False)

        load_gateway_config()

        assert os.environ.get("TELEGRAM_REPLY_TO_MODE") == "off"

    def test_env_var_takes_precedence_over_yaml(self, tmp_path, monkeypatch):
        """Existing TELEGRAM_REPLY_TO_MODE env var is not overwritten by YAML."""
        hermes_home = self._write_config(tmp_path, "telegram:\n  reply_to_mode: all\n")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("TELEGRAM_REPLY_TO_MODE", "first")

        load_gateway_config()

        assert os.environ.get("TELEGRAM_REPLY_TO_MODE") == "first"

    def test_top_level_takes_precedence_over_extra(self, tmp_path, monkeypatch):
        """telegram.reply_to_mode wins over telegram.extra.reply_to_mode."""
        hermes_home = self._write_config(
            tmp_path,
            "telegram:\n  reply_to_mode: all\n  extra:\n    reply_to_mode: \"off\"\n",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.delenv("TELEGRAM_REPLY_TO_MODE", raising=False)

        load_gateway_config()

        assert os.environ.get("TELEGRAM_REPLY_TO_MODE") == "all"


class TestDMTopicFallbackReplyToMode:
    """Tests for reply_to_mode enforcement on DM topic fallback paths.

    Regression tests for https://github.com/NousResearch/hermes-agent/issues/23994:
    reply_to_mode 'off' was ignored when sending via Hermes-created DM topic
    lanes (telegram_dm_topic_reply_fallback metadata), causing quote bubbles
    despite the user setting reply_to_mode: 'off'.
    """

    DM_TOPIC_METADATA = {
        "thread_id": "42",
        "telegram_dm_topic_reply_fallback": True,
        "telegram_reply_to_message_id": "12345",
    }

    # -- _reply_to_message_id_for_send classmethod --

    def test_reply_to_id_suppressed_when_off(self):
        """reply_to_mode='off' suppresses reply anchor for DM topic fallback."""
        result = TelegramAdapter._reply_to_message_id_for_send(
            None, self.DM_TOPIC_METADATA, reply_to_mode="off",
        )
        assert result is None

    def test_reply_to_id_returned_when_first(self):
        """reply_to_mode='first' still returns reply anchor for DM topic fallback."""
        result = TelegramAdapter._reply_to_message_id_for_send(
            None, self.DM_TOPIC_METADATA, reply_to_mode="first",
        )
        assert result == 12345

    def test_reply_to_id_returned_when_all(self):
        """reply_to_mode='all' still returns reply anchor for DM topic fallback."""
        result = TelegramAdapter._reply_to_message_id_for_send(
            None, self.DM_TOPIC_METADATA, reply_to_mode="all",
        )
        assert result == 12345

    def test_reply_to_id_returned_when_no_mode(self):
        """Without reply_to_mode, behavior is unchanged (backward compat)."""
        result = TelegramAdapter._reply_to_message_id_for_send(
            None, self.DM_TOPIC_METADATA,
        )
        assert result == 12345

    def test_explicit_reply_to_overrides_mode(self):
        """Explicit reply_to param always wins, regardless of mode."""
        result = TelegramAdapter._reply_to_message_id_for_send(
            "999", self.DM_TOPIC_METADATA, reply_to_mode="off",
        )
        assert result == 999

    # -- _thread_kwargs_for_send classmethod --

    def test_thread_kwargs_suppressed_reply_anchor_when_off(self):
        """reply_to_mode='off' returns thread_id without reply anchor."""
        result = TelegramAdapter._thread_kwargs_for_send(
            "100", "42", self.DM_TOPIC_METADATA,
            reply_to_message_id=None, reply_to_mode="off",
        )
        assert result == {"message_thread_id": 42}

    def test_thread_kwargs_returns_full_when_first(self):
        """reply_to_mode='first' returns thread_id (reply anchor in send kwargs)."""
        result = TelegramAdapter._thread_kwargs_for_send(
            "100", "42", self.DM_TOPIC_METADATA,
            reply_to_message_id=12345, reply_to_mode="first",
        )
        assert result == {"message_thread_id": 42}

    def test_thread_kwargs_no_mode_backward_compat(self):
        """Without reply_to_mode, behavior is unchanged."""
        result = TelegramAdapter._thread_kwargs_for_send(
            "100", "42", self.DM_TOPIC_METADATA,
            reply_to_message_id=12345,
        )
        assert result == {"message_thread_id": 42}

    # -- send() integration test --

    @pytest.mark.asyncio
    async def test_send_dm_topic_off_no_quote(self, adapter_factory):
        """send() with DM topic fallback and reply_to_mode='off' skips reply."""
        adapter = adapter_factory(reply_to_mode="off")
        adapter._bot = MagicMock()
        adapter._bot.send_message = AsyncMock(return_value=MagicMock(message_id=1))
        adapter.truncate_message = lambda content, max_len, **kw: ["chunk1"]

        await adapter.send("12345", "test content", metadata=self.DM_TOPIC_METADATA)

        call = adapter._bot.send_message.call_args_list[0]
        assert call.kwargs.get("reply_to_message_id") is None

    @pytest.mark.asyncio
    async def test_send_dm_topic_first_still_quotes(self, adapter_factory):
        """send() with DM topic fallback and reply_to_mode='first' still quotes."""
        adapter = adapter_factory(reply_to_mode="first")
        adapter._bot = MagicMock()
        adapter._bot.send_message = AsyncMock(return_value=MagicMock(message_id=1))
        adapter.truncate_message = lambda content, max_len, **kw: ["chunk1"]

        await adapter.send("12345", "test content", metadata=self.DM_TOPIC_METADATA)

        call = adapter._bot.send_message.call_args_list[0]
        assert call.kwargs.get("reply_to_message_id") == 12345
