"""Tests for voice mode platform isolation (bug #12542).

Voice mode state stored as {chat_id: mode} without a platform namespace
caused collisions: Telegram chat '123' and Slack chat '123' shared the
same key. The fix prefixes keys with platform value: 'telegram:123' vs
'slack:123'.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


from gateway.config import Platform
from gateway.run import GatewayRunner


class TestVoiceKeyHelper:
    """Test the _voice_key helper method."""


    def test_voice_key_different_platforms_same_chat_id(self):
        """Same chat_id on different platforms yields different keys."""
        runner = _make_runner()
        key_telegram = runner._voice_key(Platform.TELEGRAM, "123")
        key_slack = runner._voice_key(Platform.SLACK, "123")
        key_discord = runner._voice_key(Platform.DISCORD, "123")
        assert key_telegram != key_slack
        assert key_slack != key_discord
        assert key_telegram == "telegram:123"
        assert key_slack == "slack:123"
        assert key_discord == "discord:123"


class TestVoiceModePlatformIsolation:
    """Test that voice mode state is isolated by platform."""

    def test_telegram_and_slack_voice_mode_independent(self):
        """Setting voice mode for Telegram chat '123' does not affect Slack chat '123'."""
        runner = _make_runner()

        # Enable voice mode for Telegram chat '123'
        runner._voice_mode[runner._voice_key(Platform.TELEGRAM, "123")] = "all"
        # Enable voice mode for Slack chat '123' to a different mode
        runner._voice_mode[runner._voice_key(Platform.SLACK, "123")] = "voice_only"

        # Verify they are independent
        assert runner._voice_mode.get(runner._voice_key(Platform.TELEGRAM, "123")) == "all"
        assert runner._voice_mode.get(runner._voice_key(Platform.SLACK, "123")) == "voice_only"

        # Disabling Telegram should not affect Slack
        runner._voice_mode[runner._voice_key(Platform.TELEGRAM, "123")] = "off"
        assert runner._voice_mode.get(runner._voice_key(Platform.TELEGRAM, "123")) == "off"
        assert runner._voice_mode.get(runner._voice_key(Platform.SLACK, "123")) == "voice_only"


class TestLegacyKeyMigration:
    """Test migration of legacy unprefixed keys in _load_voice_modes."""

    def test_load_voice_modes_skips_legacy_keys(self):
        """_load_voice_modes skips keys without ':' prefix and logs a warning."""
        runner = _make_runner()

        # Simulate legacy persisted data with unprefixed keys
        legacy_data = {
            "123": "all",
            "456": "voice_only",
            # Also includes a properly prefixed key (from after the fix)
            "telegram:789": "off",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            voice_path = Path(tmpdir) / "gateway_voice_mode.json"
            voice_path.write_text(json.dumps(legacy_data))

            with patch.object(runner, "_VOICE_MODE_PATH", voice_path):
                with patch("gateway.run_voice.logger") as mock_logger:
                    result = runner._load_voice_modes()

            # Legacy keys without ':' should be skipped
            assert "123" not in result
            assert "456" not in result
            # Prefixed key should be preserved
            assert result.get("telegram:789") == "off"
            # Warning should be logged for each legacy key
            assert mock_logger.warning.called
            warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
            assert any("Skipping legacy unprefixed voice mode key" in str(c) for c in warning_calls)


class TestSyncVoiceModeStateToAdapter:
    """Test _sync_voice_mode_state_to_adapter filters by platform."""

    def test_sync_only_includes_platform_chats(self):
        """Only chats matching the adapter's platform are synced."""
        runner = _make_runner()

        # Set up voice mode state with multiple platforms
        runner._voice_mode = {
            "telegram:123": "off",      # Should sync
            "telegram:456": "all",       # Should NOT sync (mode is not "off")
            "slack:123": "off",          # Should NOT sync (different platform)
            "discord:789": "off",        # Should NOT sync (different platform)
        }

        # Create a mock Telegram adapter
        mock_adapter = MagicMock()
        mock_adapter.platform = Platform.TELEGRAM
        mock_adapter._auto_tts_disabled_chats = set()

        runner._sync_voice_mode_state_to_adapter(mock_adapter)

        # Only telegram:123 should be in disabled_chats (mode="off" for telegram)
        assert mock_adapter._auto_tts_disabled_chats == {"123"}


class TestVoiceModeProfileIsolation:
    """Two multiplexed bots in one Discord channel keep independent /voice
    state and voice transcripts dispatch through the bot that heard them
    (#75198 voice half)."""

    @staticmethod
    def _discord_adapter(owner=None):
        from unittest.mock import AsyncMock

        a = MagicMock()
        a.platform = Platform.DISCORD
        a._owner_profile = owner
        a._voice_text_channels = {111: 123}
        a._voice_sources = {}
        a._voice_input_callback = None
        a._on_voice_disconnect = None
        a._voice_mode_getter = None
        a._auto_tts_enabled_chats = set()
        a._auto_tts_disabled_chats = set()
        a._client = MagicMock()
        a._client.get_channel = MagicMock(return_value=None)
        a.handle_message = AsyncMock()
        return a

    @pytest.mark.asyncio
    async def test_voice_state_and_transcripts_stay_with_the_owning_bot(self, tmp_path):
        from types import SimpleNamespace

        from gateway.platforms.base import MessageEvent, MessageType, SessionSource

        runner = _make_runner()
        runner._VOICE_MODE_PATH = tmp_path / "voice.json"
        runner._is_user_authorized = lambda source: True
        default_ad = self._discord_adapter()
        bot2_ad = self._discord_adapter(owner="bot2")
        runner.adapters = {Platform.DISCORD: default_ad}
        runner._profile_adapters = {"bot2": {Platform.DISCORD: bot2_ad}}
        # Inbound event from bot2's transport in channel 123 (same id the
        # default bot also sees).
        src = SessionSource(platform=Platform.DISCORD, chat_id="123", user_id="u1",
                            chat_type="channel", profile="bot2")
        src._transport_adapter_ref = lambda: bot2_ad

        await runner._handle_voice_command(
            MessageEvent(text="/voice tts", message_type=MessageType.TEXT, source=src)
        )
        assert runner._voice_mode == {"bot2:discord:123": "all"}
        assert "123" in bot2_ad._auto_tts_enabled_chats
        assert "123" not in default_ad._auto_tts_enabled_chats

        # A transcript captured by bot2's adapter runs through bot2, not default.
        runner._bind_voice_input_callback(bot2_ad)
        await bot2_ad._voice_input_callback(guild_id=111, user_id=42, transcript="hi")
        bot2_ad.handle_message.assert_awaited_once()
        default_ad.handle_message.assert_not_awaited()
        assert bot2_ad.handle_message.call_args[0][0].source.profile == "bot2"

        # Timeout cleanup from bot2's channel disables bot2's auto-TTS only.
        join = MessageEvent(text="/voice channel", message_type=MessageType.TEXT, source=src)
        join.raw_message = SimpleNamespace(guild_id=111, guild=None)
        bot2_ad.join_voice_channel = AsyncMock(return_value=True)
        ch = MagicMock(); ch.name = "General"
        bot2_ad.get_user_voice_channel = AsyncMock(return_value=ch)
        await runner._handle_voice_channel_join(join)
        bot2_ad._on_voice_disconnect("123")
        assert runner._voice_mode["bot2:discord:123"] == "off"
        assert "123" in bot2_ad._auto_tts_disabled_chats
        assert "123" not in default_ad._auto_tts_disabled_chats

    def test_sync_restores_only_the_owning_profiles_chats(self):
        runner = _make_runner()
        runner._voice_mode = {"discord:1": "all", "bot2:discord:2": "all"}
        default_ad = MagicMock(); default_ad.platform = Platform.DISCORD
        default_ad._owner_profile = None; default_ad._auto_tts_enabled_chats = set()
        bot2_ad = MagicMock(); bot2_ad.platform = Platform.DISCORD
        bot2_ad._owner_profile = "bot2"; bot2_ad._auto_tts_enabled_chats = set()
        runner._sync_voice_mode_state_to_adapter(default_ad)
        runner._sync_voice_mode_state_to_adapter(bot2_ad)
        assert default_ad._auto_tts_enabled_chats == {"1"}
        assert bot2_ad._auto_tts_enabled_chats == {"2"}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_runner() -> GatewayRunner:
    """Create a minimal GatewayRunner for testing."""
    with patch("gateway.run.GatewayRunner._load_voice_modes", return_value={}):
        runner = GatewayRunner.__new__(GatewayRunner)
        runner._voice_mode = {}
        runner.adapters = {}
    return runner
