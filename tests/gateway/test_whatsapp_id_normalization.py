"""Tests for WhatsApp ID normalization — device suffix stripping.

WhatsApp multi-device IDs look like "number:device@domain".  The normalizer
should strip the ":device" part so that bot-self detection, mention matching,
and quoted-participant checks all work regardless of which device sent the
message.

The same logic exists in both the Python adapter and the Node.js bridge;
this file covers the Python side.
"""

import pytest

from gateway.platforms.whatsapp import WhatsAppAdapter


@pytest.mark.parametrize(
    "raw, expected",
    [
        # Standard device-suffix stripping
        ("4912345678:3@s.whatsapp.net", "4912345678@s.whatsapp.net"),
        ("4912345678:0@s.whatsapp.net", "4912345678@s.whatsapp.net"),
        ("4912345678:42@s.whatsapp.net", "4912345678@s.whatsapp.net"),
        # LID format
        ("123456789:5@lid", "123456789@lid"),
        # Already clean — no colon, should pass through
        ("4912345678@s.whatsapp.net", "4912345678@s.whatsapp.net"),
        # Group IDs — no colon, should pass through
        ("120363001234567890@g.us", "120363001234567890@g.us"),
        # Edge: colon but no @ — leave as-is
        ("something:else", "something:else"),
        # Edge: @ but no colon — leave as-is
        ("number@domain", "number@domain"),
        # Edge: colon after @ — leave as-is
        ("number@domain:extra", "number@domain:extra"),
        # Edge: empty / None
        ("", ""),
        (None, ""),
        # Edge: whitespace
        ("  4912345678:3@s.whatsapp.net  ", "4912345678@s.whatsapp.net"),
    ],
)
def test_normalize_whatsapp_id(raw, expected):
    assert WhatsAppAdapter._normalize_whatsapp_id(raw) == expected
