"""Tests for gateway.whatsapp_identity — normalize_whatsapp_identifier."""

from __future__ import annotations

import pytest

from gateway.whatsapp_identity import (
    _SAFE_IDENTIFIER_RE,
    normalize_whatsapp_identifier,
)


class TestNormalizeWhatsappIdentifier:
    def test_jid_phone_format(self):
        result = normalize_whatsapp_identifier("15551234567@s.whatsapp.net")
        assert result == "15551234567"

    def test_jid_with_device(self):
        result = normalize_whatsapp_identifier("15551234567:47@s.whatsapp.net")
        assert result == "15551234567"

    def test_lid_format(self):
        result = normalize_whatsapp_identifier("999999999999999@lid")
        assert result == "999999999999999"

    def test_bare_number(self):
        result = normalize_whatsapp_identifier("15551234567")
        assert result == "15551234567"

    def test_plus_prefixed(self):
        result = normalize_whatsapp_identifier("+15551234567")
        assert result == "15551234567"

    def test_plus_with_jid(self):
        result = normalize_whatsapp_identifier("+15551234567@s.whatsapp.net")
        assert result == "15551234567"

    def test_none_input(self):
        result = normalize_whatsapp_identifier(None)  # type: ignore[arg-type]
        assert result == ""

    def test_empty_string(self):
        assert normalize_whatsapp_identifier("") == ""

    def test_whitespace_only(self):
        result = normalize_whatsapp_identifier("   ")
        assert result == ""

    def test_strips_whitespace(self):
        result = normalize_whatsapp_identifier("  15551234567@s.whatsapp.net  ")
        assert result == "15551234567"

    def test_international_number(self):
        result = normalize_whatsapp_identifier("+60123456789@s.whatsapp.net")
        assert result == "60123456789"

    def test_short_number(self):
        result = normalize_whatsapp_identifier("123456@lid")
        assert result == "123456"

    def test_only_plus(self):
        """Bare '+' normalizes to empty."""
        result = normalize_whatsapp_identifier("+")
        assert result == ""


class TestSafeIdentifierRegex:
    def test_allows_numeric(self):
        assert _SAFE_IDENTIFIER_RE.match("1234567890") is not None

    def test_allows_alphanumeric(self):
        assert _SAFE_IDENTIFIER_RE.match("abc123DEF") is not None

    def test_allows_at_dot_plus_dash(self):
        assert _SAFE_IDENTIFIER_RE.match("test@example.com") is not None
        assert _SAFE_IDENTIFIER_RE.match("test.example") is not None
        assert _SAFE_IDENTIFIER_RE.match("+12345") is not None
        assert _SAFE_IDENTIFIER_RE.match("test-example") is not None

    def test_rejects_slash(self):
        assert _SAFE_IDENTIFIER_RE.match("../../etc") is None

    def test_rejects_backslash(self):
        assert _SAFE_IDENTIFIER_RE.match("test\\path") is None

    def test_rejects_spaces(self):
        assert _SAFE_IDENTIFIER_RE.match("test value") is None

    def test_rejects_special_chars(self):
        assert _SAFE_IDENTIFIER_RE.match("test<script>") is None
        assert _SAFE_IDENTIFIER_RE.match("test;rm") is None
