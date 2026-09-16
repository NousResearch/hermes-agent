"""Tests for Discord reply_to_mode functionality.

Covers the threading behavior control for multi-chunk replies:
- "off": Never reply-reference to original message
- "first": Only first chunk uses reply reference (default)
- "all": All chunks reply-reference the original message

Also covers reply_to_text extraction from incoming messages.
"""
import os
import sys
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from gateway.config import PlatformConfig, GatewayConfig, Platform, _apply_env_overrides, load_gateway_config


def _ensure_discord_mock():
    """Install a mock discord module when discord.py isn't available."""
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return

    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.ui = SimpleNamespace(View=object, button=lambda *a, **k: (lambda fn: fn), Button=object)
    discord_mod.ButtonStyle = SimpleNamespace(success=1, primary=2, secondary=2, danger=3, green=1, grey=2, blurple=2, red=3)
    discord_mod.Color = SimpleNamespace(orange=lambda: 1, green=lambda: 2, blue=lambda: 3, red=lambda: 4, purple=lambda: 5)
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )

    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod

    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


@pytest.fixture()
def adapter_factory():
    """Factory to create DiscordAdapter with custom reply_to_mode."""
    def create(reply_to_mode: str = "first"):
        config = PlatformConfig(enabled=True, token="test-token", reply_to_mode=reply_to_mode)
        return DiscordAdapter(config)
    return create


class TestReplyToModeConfig:
    """Tests for reply_to_mode configuration loading."""

    def test_default_mode_is_first(self, adapter_factory):
        adapter = adapter_factory()
        assert adapter._reply_to_mode == "first"

    def test_off_mode(self, adapter_factory):
        adapter = adapter_factory(reply_to_mode="off")
        assert adapter._reply_to_mode == "off"


def _make_discord_adapter(reply_to_mode: str = "first"):
    """Create a DiscordAdapter with mocked client and channel for send() tests."""
    config = PlatformConfig(enabled=True, token="test-token", reply_to_mode=reply_to_mode)
    adapter = DiscordAdapter(config)

    # Mock the Discord client and channel. Reply references are built from
    # ids via discord.MessageReference — no fetch round trip — so the
    # harness only needs a send() capture; the auto-attribute covers the
    # fetch_message.assert_not_called() assertions below.
    mock_channel = AsyncMock()
    ref_reference = MagicMock(name="MessageReference")

    sent_msg = MagicMock()
    sent_msg.id = 42
    mock_channel.send = AsyncMock(return_value=sent_msg)

    mock_client = MagicMock()
    mock_client.get_channel = MagicMock(return_value=mock_channel)

    adapter._client = mock_client
    adapter._test_expected_reference = ref_reference
    return adapter, mock_channel, ref_reference


class TestSendWithReplyToMode:
    """Tests for send() method respecting reply_to_mode."""

    @pytest.mark.asyncio
    async def test_off_mode_no_reply_reference(self):
        adapter, channel, ref_msg = _make_discord_adapter("off")
        adapter.truncate_message = lambda content, max_len, **kw: ["chunk1", "chunk2", "chunk3"]

        await adapter.send("12345", "test content", reply_to="999")

        # Should never try to fetch the reference message
        channel.fetch_message.assert_not_called()
        # All chunks sent without reference
        for call in channel.send.call_args_list:
            assert call.kwargs.get("reference") is None


    @pytest.mark.asyncio
    async def test_single_chunk_off_mode(self):
        adapter, channel, ref_msg = _make_discord_adapter("off")
        adapter.truncate_message = lambda content, max_len, **kw: ["single chunk"]

        await adapter.send("12345", "test", reply_to="999")

        channel.fetch_message.assert_not_called()
        calls = channel.send.call_args_list
        assert len(calls) == 1
        assert calls[0].kwargs.get("reference") is None


    @pytest.mark.asyncio
    async def test_first_mode_constructs_reference_without_fetch(self):
        """Pin: replies build the MessageReference from ids — no
        fetch_message round trip. Fails pre-fix, which fetched the target
        just to call to_reference() on it."""
        adapter, channel, _ = _make_discord_adapter("first")
        adapter.truncate_message = lambda content, max_len, **kw: ["chunk1", "chunk2"]

        await adapter.send("12345", "test content", reply_to="999")

        channel.fetch_message.assert_not_called()
        calls = channel.send.call_args_list
        assert len(calls) == 2
        assert calls[0].kwargs.get("reference") is not None  # first chunk
        assert calls[1].kwargs.get("reference") is None      # later chunks


class TestConfigSerialization:
    """Tests for reply_to_mode serialization (shared with Telegram)."""


    def test_from_dict_loads_reply_to_mode(self):
        data = {"enabled": True, "token": "***", "reply_to_mode": "off"}
        config = PlatformConfig.from_dict(data)
        assert config.reply_to_mode == "off"


class TestEnvVarOverride:
    """Tests for DISCORD_REPLY_TO_MODE environment variable override."""

    def _make_config(self):
        config = GatewayConfig()
        config.platforms[Platform.DISCORD] = PlatformConfig(enabled=True, token="test")
        return config

    def test_env_var_sets_off_mode(self):
        config = self._make_config()
        with patch.dict(os.environ, {"DISCORD_REPLY_TO_MODE": "off"}, clear=False):
            _apply_env_overrides(config)
        assert config.platforms[Platform.DISCORD].reply_to_mode == "off"


    def test_env_var_creates_platform_config_if_missing(self):
        """DISCORD_REPLY_TO_MODE creates PlatformConfig even without DISCORD_BOT_TOKEN."""
        config = GatewayConfig()
        assert Platform.DISCORD not in config.platforms
        with patch.dict(os.environ, {"DISCORD_REPLY_TO_MODE": "off"}, clear=False):
            _apply_env_overrides(config)
        assert Platform.DISCORD in config.platforms
        assert config.platforms[Platform.DISCORD].reply_to_mode == "off"


# ------------------------------------------------------------------
# Tests for reply_to_text extraction in _handle_message
# ------------------------------------------------------------------

# Build FakeDMChannel as a subclass of the real discord.DMChannel when the
# library is installed — this guarantees isinstance() checks pass in
# production code regardless of test ordering or monkeypatch state.
try:
    import discord as _discord_lib
    _DMChannelBase = _discord_lib.DMChannel
except (ImportError, AttributeError):
    _DMChannelBase = object


class FakeDMChannel(_DMChannelBase):
    """Minimal DM channel stub (skips mention / channel-allow checks)."""
    def __init__(self, channel_id: int = 100, name: str = "dm"):
        # Do NOT call super().__init__() — real DMChannel requires State
        self.id = channel_id
        self.name = name


def _make_message(*, content: str = "hi", reference=None):
    """Build a mock Discord message for _handle_message tests."""
    author = SimpleNamespace(id=42, display_name="TestUser", name="TestUser")
    return SimpleNamespace(
        id=999,
        content=content,
        mentions=[],
        attachments=[],
        reference=reference,
        created_at=datetime.now(timezone.utc),
        channel=FakeDMChannel(),
        author=author,
    )


@pytest.fixture
def reply_text_adapter(monkeypatch):
    """DiscordAdapter wired for _handle_message → handle_message capture."""
    config = PlatformConfig(enabled=True, token="fake-token")
    adapter = DiscordAdapter(config)
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=999))
    adapter._text_batch_delay_seconds = 0
    adapter.handle_message = AsyncMock()
    return adapter


class TestReplyToText:
    """Tests for reply_to_text populated by _handle_message."""

    @pytest.mark.asyncio
    async def test_no_reference_both_none(self, reply_text_adapter):
        message = _make_message(reference=None)

        await reply_text_adapter._handle_message(message)

        event = reply_text_adapter.handle_message.await_args.args[0]
        assert event.reply_to_message_id is None
        assert event.reply_to_text is None


    @pytest.mark.asyncio
    async def test_reference_with_empty_resolved_content(self, reply_text_adapter):
        """Empty string content should become None, not leak as empty string."""
        resolved_msg = SimpleNamespace(content="")
        ref = SimpleNamespace(message_id=555, resolved=resolved_msg)
        message = _make_message(reference=ref)

        await reply_text_adapter._handle_message(message)

        event = reply_text_adapter.handle_message.await_args.args[0]
        assert event.reply_to_message_id == "555"
        assert event.reply_to_text is None


class TestYamlConfigLoading:
    """Tests for reply_to_mode loaded from config.yaml discord section."""

    def _write_config(self, tmp_path, content: str):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(content, encoding="utf-8")
        return hermes_home


    def test_extra_reply_to_mode_off(self, tmp_path, monkeypatch):
        """discord.extra.reply_to_mode is also honoured."""
        hermes_home = self._write_config(
            tmp_path, "discord:\n  extra:\n    reply_to_mode: \"off\"\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.delenv("DISCORD_REPLY_TO_MODE", raising=False)

        load_gateway_config()

        assert os.environ.get("DISCORD_REPLY_TO_MODE") == "off"


    def test_top_level_takes_precedence_over_extra(self, tmp_path, monkeypatch):
        """discord.reply_to_mode wins over discord.extra.reply_to_mode."""
        hermes_home = self._write_config(
            tmp_path,
            "discord:\n  reply_to_mode: all\n  extra:\n    reply_to_mode: \"off\"\n",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.delenv("DISCORD_REPLY_TO_MODE", raising=False)

        load_gateway_config()

        assert os.environ.get("DISCORD_REPLY_TO_MODE") == "all"


class TestVoiceReplyReference:
    """send_voice builds its reply reference from ids too — same no-fetch
    pin as the text path (construction happens before the file read)."""

    @pytest.mark.asyncio
    async def test_voice_reply_constructs_reference_without_fetch(self, tmp_path, monkeypatch):
        adapter, channel, _ = _make_discord_adapter("first")
        audio = tmp_path / "clip.ogg"
        audio.write_bytes(b"OggS" + b"\x00" * 64)
        monkeypatch.setattr(adapter, "_is_forum_parent", lambda _c: True)
        forum_posts = []
        async def fake_forum_post(_channel, **kwargs):
            forum_posts.append(kwargs)
            return MagicMock(success=True, message_id="77")
        monkeypatch.setattr(adapter, "_forum_post_file", fake_forum_post)

        await adapter.send_voice("12345", str(audio), reply_to="999")

        channel.fetch_message.assert_not_called()
