"""Unit tests for gateway.whatsapp_identity.to_whatsapp_jid.

``to_whatsapp_jid`` is the outbound inverse of
``normalize_whatsapp_identifier``: it builds the bridge-safe JID a send
must use. Baileys' ``jidDecode`` crashes on a bare phone number (#8637),
so every outbound target must be rewritten to ``<digits>@s.whatsapp.net``
before it reaches the bridge.
"""

import pytest

from gateway.whatsapp_identity import to_whatsapp_jid


class TestToWhatsappJid:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            # bare phone numbers → user JID
            ("+50766715226", "50766715226@s.whatsapp.net"),
            ("50766715226", "50766715226@s.whatsapp.net"),
            # human-formatted phone numbers get stripped to digits
            ("+1 (555) 123-4567", "15551234567@s.whatsapp.net"),
            ("+1.555.123.4567", "15551234567@s.whatsapp.net"),
        ],
    )
    def test_bare_phone_becomes_user_jid(self, raw, expected):
        assert to_whatsapp_jid(raw) == expected

    @pytest.mark.parametrize(
        "jid",
        [
            "50766715226@s.whatsapp.net",  # already a user JID
            "123456789-987654321@g.us",    # group JID
            "130631430344750@lid",         # linked identity
            "status@broadcast",            # broadcast pseudo-chat
            "123@newsletter",              # channel/newsletter
        ],
    )
    def test_fully_qualified_jid_passes_through(self, jid):
        assert to_whatsapp_jid(jid) == jid

    def test_device_suffixed_colon_form_collapses_to_at(self):
        # ``user:device@domain`` (legacy) → ``user@domain``
        assert to_whatsapp_jid("60123456789:47@s.whatsapp.net") == (
            "60123456789@s.whatsapp.net"
        )

    @pytest.mark.parametrize("empty", ["", "   ", None])
    def test_empty_input_returns_empty(self, empty):
        assert to_whatsapp_jid(empty) == ""

    def test_unrecognized_target_passes_through_unchanged(self):
        # Not a phone, no ``@`` — leave it for the bridge to reject with a
        # meaningful error rather than mangling it into a bogus JID.
        assert to_whatsapp_jid("not-a-number") == "not-a-number"
