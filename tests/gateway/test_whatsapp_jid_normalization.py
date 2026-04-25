"""Tests for WhatsApp chat JID normalization.

The bridge passes chatId straight to Baileys' sock.sendMessage(), which
internally calls jidDecode(). Bare numeric IDs return undefined and crash
the bridge. The gateway must append @g.us / @s.whatsapp.net before sending.
"""

import pytest

from gateway.platforms.whatsapp import WhatsAppAdapter


@pytest.mark.parametrize(
    "raw, expected",
    [
        # Newer-format group ID (starts with 120363) → @g.us
        ("120363328279139738", "120363328279139738@g.us"),
        # Legacy group ID (<phone>-<timestamp>) → @g.us
        ("60123456789-1234567890", "60123456789-1234567890@g.us"),
        # DM phone number → @s.whatsapp.net
        ("60123456789", "60123456789@s.whatsapp.net"),
        # Already-suffixed JIDs pass through unchanged
        ("120363328279139738@g.us", "120363328279139738@g.us"),
        ("60123456789@s.whatsapp.net", "60123456789@s.whatsapp.net"),
        ("1234567890@lid", "1234567890@lid"),
        # Empty string returns empty (callers handle their own validation)
        ("", ""),
    ],
)
def test_normalize_chat_jid(raw, expected):
    assert WhatsAppAdapter._normalize_chat_jid(raw) == expected
