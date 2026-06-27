"""Tests for voice mode platform isolation (bug #12542).

Voice mode state stored as {chat_id: mode} without a platform namespace
caused collisions: Telegram chat '123' and Slack chat '123' shared the
same key. The fix prefixes keys with platform value: 'telegram:123' vs
'slack:123'.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner


class TestVoiceKeyHelper:
    """Test the _voice_key helper method."""

    def test_voice_key_format(self):
        """_voice_key returns 'platform:chat_id' format."""
        runner = _make_runner()
        assert runner._voice_key(Platform.TELEGRAM, "123") == "telegram:123"
        assert runner._voice_key(Platform.SLACK, "456") == "slack:456"
        assert runner._voice_key(Platform.DISCORD, "789") == "discord:789"

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
                with patch("gateway.run.logger") as mock_logger:
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

    def test_load_voice_modes_preserves_prefixed_keys(self):
        """_load_voice_modes correctly loads platform-prefixed keys."""
        runner = _make_runner()

        persisted_data = {
            "telegram:123": "all",
            "slack:456": "voice_only",
            "discord:789": "off",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            voice_path = Path(tmpdir) / "gateway_voice_mode.json"
            voice_path.write_text(json.dumps(persisted_data))

            with patch.object(runner, "_VOICE_MODE_PATH", voice_path):
                result = runner._load_voice_modes()

        assert result.get("telegram:123") == "all"
        assert result.get("slack:456") == "voice_only"
        assert result.get("discord:789") == "off"

    def test_load_voice_modes_invalid_modes_filtered(self):
        """_load_voice_modes filters out invalid mode values."""
        runner = _make_runner()

        data = {
            "telegram:123": "all",
            "telegram:456": "invalid_mode",
            "telegram:789": "voice_only",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            voice_path = Path(tmpdir) / "gateway_voice_mode.json"
            voice_path.write_text(json.dumps(data))

            with patch.object(runner, "_VOICE_MODE_PATH", voice_path):
                result = runner._load_voice_modes()

        assert result.get("telegram:123") == "all"
        assert "telegram:456" not in result
        assert result.get("telegram:789") == "voice_only"


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

    def test_sync_clears_existing_state(self):
        """_sync_voice_mode_state_to_adapter clears existing disabled_chats first."""
        runner = _make_runner()

        runner._voice_mode = {
            "telegram:123": "off",
        }

        mock_adapter = MagicMock()
        mock_adapter.platform = Platform.TELEGRAM
        mock_adapter._auto_tts_disabled_chats = {"old_chat_id", "another_old"}

        runner._sync_voice_mode_state_to_adapter(mock_adapter)

        # Old entries should be cleared
        assert mock_adapter._auto_tts_disabled_chats == {"123"}

    def test_sync_returns_early_without_platform(self):
        """_sync_voice_mode_state_to_adapter returns early if adapter has no platform."""
        runner = _make_runner()
        runner._voice_mode = {"telegram:123": "off"}

        mock_adapter = MagicMock()
        mock_adapter.platform = None
        mock_adapter._auto_tts_disabled_chats = {"old"}

        runner._sync_voice_mode_state_to_adapter(mock_adapter)

        # disabled_chats should not be modified
        assert mock_adapter._auto_tts_disabled_chats == {"old"}

    def test_sync_returns_early_without_auto_tts_disabled_chats(self):
        """_sync_voice_mode_state_to_adapter returns early if adapter lacks _auto_tts_disabled_chats."""
        runner = _make_runner()
        runner._voice_mode = {"telegram:123": "off"}

        mock_adapter = MagicMock(spec=[])  # No _auto_tts_disabled_chats attribute

        # Should not raise
        runner._sync_voice_mode_state_to_adapter(mock_adapter)


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
