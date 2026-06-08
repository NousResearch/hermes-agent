"""Tests for WhatsApp outgoing chat ID normalization."""

import pytest

from gateway.platforms.whatsapp import WhatsAppAdapter


class TestNormalizeOutgoingChatId:
    """Tests for _normalize_outgoing_chat_id()."""

    def test_bare_phone_gets_suffix(self):
        assert (
            WhatsAppAdapter._normalize_outgoing_chat_id("15005004144")
            == "15005004144@s.whatsapp.net"
        )

    def test_phone_with_plus_gets_suffix(self):
        assert (
            WhatsAppAdapter._normalize_outgoing_chat_id("+15005004144")
            == "+15005004144@s.whatsapp.net"
        )

    def test_full_jid_passthrough(self):
        cid = "15005004144@s.whatsapp.net"
        assert WhatsAppAdapter._normalize_outgoing_chat_id(cid) == cid

    def test_group_jid_passthrough(self):
        cid = "120363044444444444@g.us"
        assert WhatsAppAdapter._normalize_outgoing_chat_id(cid) == cid

    def test_empty_string_passthrough(self):
        assert WhatsAppAdapter._normalize_outgoing_chat_id("") == ""

    def test_whitespace_stripped(self):
        assert (
            WhatsAppAdapter._normalize_outgoing_chat_id("  15005004144  ")
            == "15005004144@s.whatsapp.net"
        )

    def test_group_id_with_dash_not_normalized(self):
        """Group IDs like '15005004144-1234567890' should NOT get @s.whatsapp.net."""
        cid = "15005004144-1234567890"
        assert WhatsAppAdapter._normalize_outgoing_chat_id(cid) == cid

    def test_status_broadcast_passthrough(self):
        cid = "status@broadcast"
        assert WhatsAppAdapter._normalize_outgoing_chat_id(cid) == cid
