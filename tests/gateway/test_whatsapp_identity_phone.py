"""Tests for the phone normalizer behind ``/access`` (gateway/whatsapp_identity.py).

The normalizer is dependency-free (no libphonenumber) and derives the default country
code from the bot's own number, so the same local input resolves correctly on bots in
different countries.  These cases pin that contract, including the deliberate
limitations: ambiguous bare prefixes resolve to the HOME country, and foreign numbers
typed without ``+`` are the caller's problem.
"""
from __future__ import annotations

import pytest

from gateway.whatsapp_identity import default_country_code, normalize_phone_e164


class TestDefaultCountryCode:
    @pytest.mark.parametrize("own,cc", [
        ("6287784454555", "62"),      # Indonesia
        ("15551234567", "1"),         # NANP
        ("447911123456", "44"),       # UK
        ("819012345678", "81"),       # Japan
        ("", ""),
        ("12345", ""),                # too short to be a real number
    ])
    def test_longest_prefix_match(self, own, cc):
        assert default_country_code(own) == cc


class TestNormalizePhoneE164:
    def test_indonesian_bot_local_formats(self):
        cc = "62"
        assert normalize_phone_e164("0812-3456-789", cc) == "628123456789"
        assert normalize_phone_e164("08123456789", cc) == "628123456789"
        assert normalize_phone_e164("8123456789", cc) == "628123456789"  # trunk 0 omitted
        assert normalize_phone_e164("628123456789", cc) == "628123456789"  # already cc
        assert normalize_phone_e164("+62 812 3456 789", cc) == "628123456789"
        assert normalize_phone_e164("0062 812 3456 789", cc) == "628123456789"

    def test_us_bot_local_formats(self):
        cc = "1"
        assert normalize_phone_e164("5551234567", cc) == "15551234567"
        assert normalize_phone_e164("(555) 123-4567", cc) == "15551234567"
        assert normalize_phone_e164("15551234567", cc) == "15551234567"  # already cc
        assert normalize_phone_e164("+1 555 123 4567", cc) == "15551234567"

    def test_uk_bot_local_formats(self):
        cc = "44"
        assert normalize_phone_e164("07911 123456", cc) == "447911123456"
        assert normalize_phone_e164("020 7946 0958", cc) == "442079460958"
        assert normalize_phone_e164("447911123456", cc) == "447911123456"

    def test_explicit_plus_is_international_from_anywhere(self):
        assert normalize_phone_e164("+15551234567", "62") == "15551234567"
        assert normalize_phone_e164("+62 812 3456 789", "1") == "628123456789"

    def test_dial_out_prefixes(self):
        assert normalize_phone_e164("011 44 7911 123456", "1") == "447911123456"   # NANP
        assert normalize_phone_e164("0011 61 412 345 678", "62") == "61412345678"  # AU
        assert normalize_phone_e164("0062 812 3456 789", "44") == "628123456789"   # generic 00

    def test_ambiguous_bare_prefix_prefers_home_country(self):
        # '55' is Brazil's code, but on a US bot '5551234567' is a local NANP number.
        # The local reading wins; foreign numbers must be typed with '+'.
        assert normalize_phone_e164("5551234567", "1") == "15551234567"
        assert normalize_phone_e164("8123456789", "62") == "628123456789"  # '81' = Japan

    def test_no_home_context_rejects_local_formats(self):
        assert normalize_phone_e164("0812345678", "") == ""
        assert normalize_phone_e164("5551234567", "") == ""

    def test_invalid_input_returns_empty(self):
        assert normalize_phone_e164("", "62") == ""
        assert normalize_phone_e164("abc", "62") == ""
        assert normalize_phone_e164("12345", "62") == ""  # too short even after prepend
        assert normalize_phone_e164("628123456789012345", "62") == ""  # > E.164 max
        assert normalize_phone_e164("+62", "62") == ""  # cc alone is not a number
