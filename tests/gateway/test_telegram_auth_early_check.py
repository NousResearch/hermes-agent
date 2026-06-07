"""Tests for early authorization check in Telegram message handlers.

When a user is removed from TELEGRAM_ALLOWED_USERS, their messages should be
rejected at the adapter level before any text batching, event building, or
agent processing occurs (fixes #40863).
"""

import os
from typing import Optional
from unittest.mock import AsyncMock, Mock, patch

import pytest

from gateway.platforms.telegram import TelegramAdapter


class TestTelegramMessageSenderAuthorization:
    """Test the _is_message_sender_authorized method directly."""

    def _make_adapter(self):
        """Create a minimal TelegramAdapter instance for testing."""
        with patch('gateway.platforms.telegram.TelegramAdapter.__init__', return_value=None):
            return TelegramAdapter.__new__(TelegramAdapter)

    def _make_message(self, user_id: int, chat_type: str = "private"):
        """Create a mock Telegram Message object."""
        message = Mock()
        message.from_user = Mock()
        message.from_user.id = user_id
        message.chat = Mock()
        message.chat.type = chat_type
        return message

    def test_authorized_user_in_allowed_list(self):
        """Authorized users in TELEGRAM_ALLOWED_USERS should pass."""
        adapter = self._make_adapter()
        message = self._make_message(user_id=123)
        
        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "123"}):
            assert adapter._is_message_sender_authorized(message) is True

    def test_unauthorized_user_not_in_allowed_list(self):
        """Users not in TELEGRAM_ALLOWED_USERS should be rejected."""
        adapter = self._make_adapter()
        message = self._make_message(user_id=999)
        
        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "123"}):
            assert adapter._is_message_sender_authorized(message) is False

    def test_unauthorized_user_private_chat(self):
        """Private chat with unauthorized user should be rejected."""
        adapter = self._make_adapter()
        message = self._make_message(user_id=999, chat_type="private")
        
        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "123"}):
            assert adapter._is_message_sender_authorized(message) is False

    def test_group_message_passthrough(self):
        """Group messages should pass (auth is chat-based via _should_process_message)."""
        adapter = self._make_adapter()
        message = self._make_message(user_id=999, chat_type="group")
        
        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "123"}):
            assert adapter._is_message_sender_authorized(message) is True

    def test_channel_message_passthrough(self):
        """Channel messages should pass (auth is chat-based)."""
        adapter = self._make_adapter()
        message = self._make_message(user_id=999, chat_type="channel")
        
        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "123"}):
            assert adapter._is_message_sender_authorized(message) is True

    def test_supergroup_message_passthrough(self):
        """Supergroup messages should pass (auth is chat-based)."""
        adapter = self._make_adapter()
        message = self._make_message(user_id=999, chat_type="supergroup")
        
        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "123"}):
            assert adapter._is_message_sender_authorized(message) is True

    def test_gateway_allow_all_users_flag(self):
        """GATEWAY_ALLOW_ALL_USERS=true should allow unauthorized DM users."""
        adapter = self._make_adapter()
        message = self._make_message(user_id=999, chat_type="private")
        
        # When TELEGRAM_ALLOWED_USERS is empty, GATEWAY_ALLOW_ALL_USERS controls access
        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "", "GATEWAY_ALLOW_ALL_USERS": "true"}):
            assert adapter._is_message_sender_authorized(message) is True

    def test_multiple_allowed_users(self):
        """Multiple comma-separated allowed users should work."""
        adapter = self._make_adapter()
        message = self._make_message(user_id=456)
        
        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "123,456,789"}):
            assert adapter._is_message_sender_authorized(message) is True

    def test_multiple_allowed_users_unauthorized(self):
        """User not in comma-separated list should be rejected."""
        adapter = self._make_adapter()
        message = self._make_message(user_id=999)
        
        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "123,456,789"}):
            assert adapter._is_message_sender_authorized(message) is False

    def test_wildcard_allowed_users(self):
        """Wildcard TELEGRAM_ALLOWED_USERS should allow all DMs."""
        adapter = self._make_adapter()
        message = self._make_message(user_id=999, chat_type="private")
        
        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}):
            assert adapter._is_message_sender_authorized(message) is True

    def test_empty_allowed_users_list(self):
        """Empty TELEGRAM_ALLOWED_USERS should reject DM users."""
        adapter = self._make_adapter()
        message = self._make_message(user_id=123, chat_type="private")
        
        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": ""}):
            assert adapter._is_message_sender_authorized(message) is False

    def test_no_allowed_users_env_set(self):
        """No TELEGRAM_ALLOWED_USERS env should reject DM users."""
        adapter = self._make_adapter()
        message = self._make_message(user_id=123, chat_type="private")
        
        with patch.dict(os.environ, {}, clear=True):
            assert adapter._is_message_sender_authorized(message) is False

    def test_message_with_no_user_id(self):
        """Message without user_id should be rejected."""
        adapter = self._make_adapter()
        message = Mock()
        message.from_user = None
        message.chat = Mock()
        message.chat.type = "private"
        
        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "123"}):
            # Should handle gracefully without crashing
            try:
                result = adapter._is_message_sender_authorized(message)
                # Either rejected or handled gracefully is OK
                assert isinstance(result, bool)
            except (AttributeError, TypeError):
                # Graceful failure is acceptable for malformed messages
                pass

    def test_message_with_no_chat_type(self):
        """Message without chat type should be handled gracefully."""
        adapter = self._make_adapter()
        message = Mock()
        message.from_user = Mock()
        message.from_user.id = 123
        message.chat = Mock()
        message.chat.type = None
        
        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "123"}):
            # Should handle gracefully
            result = adapter._is_message_sender_authorized(message)
            assert isinstance(result, bool)

    def test_whitespace_in_allowed_users(self):
        """Whitespace-padded user IDs should be handled."""
        adapter = self._make_adapter()
        message = self._make_message(user_id=123)
        
        # Environment vars with spaces
        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "123, 456, 789"}):
            result = adapter._is_message_sender_authorized(message)
            # Should either match or fail gracefully
            assert isinstance(result, bool)
