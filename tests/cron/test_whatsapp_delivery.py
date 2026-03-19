"""
Tests for WhatsApp cron delivery with human-readable labels.

Issue #1945: Cron jobs to WhatsApp fail with Baileys jidDecode error when
deliver uses human-readable contact label.

The fix: _resolve_delivery_target should use resolve_channel_name to convert
human-readable labels (e.g., "whatsapp:Alice (dm)") to numeric JIDs
(e.g., "12345678901234@lid").
"""

import pytest
from unittest.mock import patch, MagicMock


class TestResolveDeliveryTarget:
    """Tests for _resolve_delivery_target function."""

    def test_resolve_local_delivery(self):
        """Test that 'local' delivery returns None (no remote delivery)."""
        from cron.scheduler import _resolve_delivery_target
        
        job = {"deliver": "local"}
        result = _resolve_delivery_target(job)
        assert result is None

    def test_resolve_origin_delivery(self):
        """Test that 'origin' delivery returns the origin info."""
        from cron.scheduler import _resolve_delivery_target
        
        job = {
            "deliver": "origin",
            "origin": {
                "platform": "telegram",
                "chat_id": "123456789",
                "thread_id": None,
            }
        }
        result = _resolve_delivery_target(job)
        assert result == {
            "platform": "telegram",
            "chat_id": "123456789",
            "thread_id": None,
        }

    @patch("gateway.channel_directory.resolve_channel_name")
    def test_resolve_whatsapp_human_readable_label(self, mock_resolve):
        """Test that human-readable labels are resolved to JIDs."""
        from cron.scheduler import _resolve_delivery_target
        
        # Mock the channel directory resolution
        mock_resolve.return_value = "12345678901234@lid"
        
        job = {
            "id": "test-job-123",
            "deliver": "whatsapp:Alice (dm)",  # Human-readable label
        }
        
        result = _resolve_delivery_target(job)
        
        # Should have called resolve_channel_name
        mock_resolve.assert_called_once_with("whatsapp", "Alice (dm)")
        
        # Should return resolved JID
        assert result == {
            "platform": "whatsapp",
            "chat_id": "12345678901234@lid",
            "thread_id": None,
        }

    @patch("gateway.channel_directory.resolve_channel_name")
    def test_resolve_numeric_id_unchanged(self, mock_resolve):
        """Test that numeric IDs are used directly (not resolved)."""
        from cron.scheduler import _resolve_delivery_target
        
        # resolve_channel_name returns None for non-existent names
        mock_resolve.return_value = None
        
        job = {
            "id": "test-job-123",
            "deliver": "whatsapp:12345678901234@lid",  # Already a JID
        }
        
        result = _resolve_delivery_target(job)
        
        # Should still return the original JID
        assert result == {
            "platform": "whatsapp",
            "chat_id": "12345678901234@lid",
            "thread_id": None,
        }

    @patch("gateway.channel_directory.resolve_channel_name")
    def test_resolve_telegram_human_readable(self, mock_resolve):
        """Test resolution works for other platforms too."""
        from cron.scheduler import _resolve_delivery_target
        
        mock_resolve.return_value = "987654321"
        
        job = {
            "id": "test-job-456",
            "deliver": "telegram:John Doe",
        }
        
        result = _resolve_delivery_target(job)
        
        mock_resolve.assert_called_once_with("telegram", "John Doe")
        assert result == {
            "platform": "telegram",
            "chat_id": "987654321",
            "thread_id": None,
        }

    @patch("gateway.channel_directory.resolve_channel_name")
    def test_resolve_discord_channel_name(self, mock_resolve):
        """Test resolution for Discord channel names."""
        from cron.scheduler import _resolve_delivery_target
        
        mock_resolve.return_value = "111222333444555666"
        
        job = {
            "id": "test-job-789",
            "deliver": "discord:#general",
        }
        
        result = _resolve_delivery_target(job)
        
        mock_resolve.assert_called_once_with("discord", "#general")
        assert result == {
            "platform": "discord",
            "chat_id": "111222333444555666",
            "thread_id": None,
        }


class TestWhatsAppDeliveryIntegration:
    """Integration tests simulating actual WhatsApp delivery scenarios."""

    @patch("gateway.channel_directory.resolve_channel_name")
    def test_real_world_whatsapp_label_format(self, mock_resolve):
        """Test with actual label format from send_message(action='list')."""
        from cron.scheduler import _resolve_delivery_target
        
        # This is the actual format returned by send_message(action="list")
        mock_resolve.return_value = "5511987654321@s.whatsapp.net"
        
        # User pastes this from the list output
        job = {
            "id": "daily-report-123",
            "deliver": "whatsapp:Maria Silva (dm)",
            "prompt": "Send daily report",
        }
        
        result = _resolve_delivery_target(job)
        
        assert result is not None
        assert result["platform"] == "whatsapp"
        assert result["chat_id"] == "5511987654321@s.whatsapp.net"
        assert "@" in result["chat_id"]  # Valid JID format

    @patch("gateway.channel_directory.resolve_channel_name")
    def test_whatsapp_group_label(self, mock_resolve):
        """Test resolution of WhatsApp group names."""
        from cron.scheduler import _resolve_delivery_target
        
        mock_resolve.return_value = "123456789@g.us"  # Group JID format
        
        job = {
            "id": "group-announcement",
            "deliver": "whatsapp:Team Updates (group)",
        }
        
        result = _resolve_delivery_target(job)
        
        assert result["chat_id"] == "123456789@g.us"
        assert "@g.us" in result["chat_id"]  # Group JID indicator

    def test_fallback_when_no_match(self):
        """Test that unknown labels are passed through unchanged."""
        from cron.scheduler import _resolve_delivery_target
        
        # When resolve_channel_name returns None, original value should be used
        with patch("gateway.channel_directory.resolve_channel_name", return_value=None):
            job = {
                "id": "test-job",
                "deliver": "whatsapp:NonExistentContact",
            }
            
            result = _resolve_delivery_target(job)
            
            # Should still return the original value (might fail at send time)
            assert result["chat_id"] == "NonExistentContact"


class TestDeliveryTargetEdgeCases:
    """Edge case tests for delivery target resolution."""

    def test_missing_origin_returns_none(self):
        """Test origin delivery without origin info."""
        from cron.scheduler import _resolve_delivery_target
        
        job = {"deliver": "origin"}  # No origin field
        result = _resolve_delivery_target(job)
        assert result is None

    @patch("gateway.channel_directory.resolve_channel_name")
    def test_colons_in_label(self, mock_resolve):
        """Test handling of labels with colons in them."""
        from cron.scheduler import _resolve_delivery_target
        
        mock_resolve.return_value = "123456789"
        
        # Label like "whatsapp:John: Sales Team (dm)" has colon in name
        job = {
            "deliver": "whatsapp:John: Sales Team (dm)",
        }
        
        result = _resolve_delivery_target(job)
        
        # Should split only on first colon
        mock_resolve.assert_called_once_with("whatsapp", "John: Sales Team (dm)")

    @patch("gateway.channel_directory.resolve_channel_name")
    def test_special_characters_in_label(self, mock_resolve):
        """Test handling of special characters in labels."""
        from cron.scheduler import _resolve_delivery_target
        
        mock_resolve.return_value = "123456789@lid"
        
        job = {
            "deliver": "whatsapp:José García 🇪🇸 (dm)",
        }
        
        result = _resolve_delivery_target(job)
        
        mock_resolve.assert_called_once_with("whatsapp", "José García 🇪🇸 (dm)")
        assert result["chat_id"] == "123456789@lid"
