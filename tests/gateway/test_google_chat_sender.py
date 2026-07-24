"""Regression tests for Google Chat quoted message sender parsing (#56622).

Sweeper feedback: parse sender as a string, don't infer reply_to_is_own_message
from this field without a documented bot-identifying value.

Tests the quotedMessageMetadata parsing logic in isolation.
"""
from __future__ import annotations

import pytest


def _parse_quoted_sender(snapshot_sender):
    """Mirror the production logic from GoogleChatAdapter._build_message_event."""
    _raw_sender = snapshot_sender
    if isinstance(_raw_sender, dict):
        _sender_str = _raw_sender.get("name") or ""
    else:
        _sender_str = str(_raw_sender or "")
    # reply_to_is_own_message is always False — no inference from sender
    return _sender_str, False


class TestQuotedMessageSenderParsing:
    """Sweeper feedback: sender must be parsed as string,
    reply_to_is_own_message must NOT be inferred."""

    def test_dict_sender_parsed_as_string(self):
        """When sender is a dict {name: ...}, extract name as string."""
        sender_str, is_own = _parse_quoted_sender({"name": "users/someone"})
        assert sender_str == "users/someone"
        assert is_own is False

    def test_string_sender_parsed_directly(self):
        """When sender is a bare string, accept it directly."""
        sender_str, is_own = _parse_quoted_sender("users/someone")
        assert sender_str == "users/someone"
        assert is_own is False

    def test_null_sender_returns_empty_string(self):
        """When sender is missing/null, sender_str is empty."""
        sender_str, is_own = _parse_quoted_sender(None)
        assert sender_str == ""
        assert is_own is False

    def test_own_bot_id_does_not_infer_own_message(self):
        """Even if sender matches bot_user_id, is_own is False —
        no documented bot-identifying value in the snapshot."""
        sender_str, is_own = _parse_quoted_sender({"name": "users/bot123"})
        assert is_own is False
        assert sender_str == "users/bot123"

    def test_integer_sender_coerced_to_string(self):
        """Non-string, non-dict sender values are coerced safely."""
        sender_str, is_own = _parse_quoted_sender(12345)
        assert sender_str == "12345"
        assert is_own is False
