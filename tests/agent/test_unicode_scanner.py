"""Tests for agent.unicode_scanner — ZWJ-aware invisible-unicode detection."""

import pytest

from agent.unicode_scanner import find_unsafe_invisibles, INVISIBLE_CHARS_BLOCKLIST, ZWJ


class TestFindUnsafeInvisibles:
    """Core scanner: find_unsafe_invisibles()."""

    def test_clean_content_returns_empty(self):
        assert find_unsafe_invisibles("Hello, world! 🧙‍♂️ 👨‍👩‍👧") == []

    def test_zero_width_space_detected(self):
        content = "normal\u200btext"
        result = find_unsafe_invisibles(content)
        assert len(result) == 1
        assert result[0] == ('\u200b', 0x200B)

    def test_bom_detected(self):
        content = "\ufeffleading BOM"
        result = find_unsafe_invisibles(content)
        assert len(result) == 1
        assert result[0] == ('\ufeff', 0xFEFF)

    def test_directional_override_detected(self):
        content = "\u202esuspicious"
        result = find_unsafe_invisibles(content)
        assert len(result) == 1
        assert result[0] == ('\u202e', 0x202E)

    # ── ZWJ in emoji clusters (should be ALLOWED) ────────────────────────

    def test_zwj_in_gendered_emoji_allowed(self):
        """🧙‍♂️ = U+1F9D9 + ZWJ + U+2642 — ZWJ should be allowed."""
        assert find_unsafe_invisibles("🧙‍♂️") == []

    def test_zwj_in_family_emoji_allowed(self):
        """👨‍👩‍👧 = U+1F468 + ZWJ + U+1F469 + ZWJ + U+1F467 — ZWJs allowed."""
        assert find_unsafe_invisibles("👨‍👩‍👧") == []

    def test_zwj_in_flag_emoji_allowed(self):
        """🏳️‍🌈 = 🏳 + VS16 + ZWJ + 🌈 — ZWJ after VS16 should be allowed."""
        assert find_unsafe_invisibles("🏳️‍🌈") == []

    def test_zwj_in_health_emoji_allowed(self):
        """👩‍⚕️ = U+1F469 + ZWJ + U+2695 — ZWJ should be allowed."""
        assert find_unsafe_invisibles("👩‍⚕️") == []

    def test_zwj_in_runner_emoji_allowed(self):
        """🏃‍♀️ — ZWJ should be allowed."""
        assert find_unsafe_invisibles("🏃‍♀️") == []

    # ── ZWJ in suspicious positions (should be BLOCKED) ──────────────────

    def test_standalone_zwj_blocked(self):
        """ZWJ with non-pictographic neighbours is suspicious."""
        content = "ignore\u200dprevious"
        result = find_unsafe_invisibles(content)
        assert len(result) == 1
        assert result[0][1] == 0x200D

    def test_zwj_at_start_of_content_blocked(self):
        """ZWJ at the very start with no left neighbour."""
        content = "\u200dhello"
        result = find_unsafe_invisibles(content)
        assert len(result) == 1
        assert result[0][1] == 0x200D

    def test_zwj_at_end_of_content_blocked(self):
        """ZWJ at the end with no right neighbour."""
        content = "hello\u200d"
        result = find_unsafe_invisibles(content)
        assert len(result) == 1
        assert result[0][1] == 0x200D

    def test_zwj_between_letters_blocked(self):
        """ZWJ between two ASCII letters is suspicious."""
        content = "a\u200db"
        result = find_unsafe_invisibles(content)
        assert len(result) == 1
        assert result[0][1] == 0x200D

    # ── Multiple issues ──────────────────────────────────────────────────

    def test_mixed_invisible_chars(self):
        """ZWJ in emoji allowed, other invisibles detected."""
        content = "🧙‍♂️ has a zero-width space\u200bhere"
        result = find_unsafe_invisibles(content)
        assert len(result) == 1
        assert result[0] == ('\u200b', 0x200B)

    def test_multiple_dangerous_invisibles(self):
        content = "\u200b\u202etext"
        result = find_unsafe_invisibles(content)
        assert len(result) == 2

    # ── Custom blocklist ─────────────────────────────────────────────────

    def test_custom_blocklist(self):
        content = "normal\u200btext"
        # Only scan for directional overrides, not ZWSP
        result = find_unsafe_invisibles(content, blocklist={'\u202e'})
        assert result == []

    # ── Blocklist contents ───────────────────────────────────────────────

    def test_blocklist_contains_all_original_chars(self):
        """The shared blocklist must contain every char from the original sets."""
        original = {'\u200b', '\u200c', '\u200d', '\u2060', '\ufeff',
                    '\u202a', '\u202b', '\u202c', '\u202d', '\u202e'}
        assert INVISIBLE_CHARS_BLOCKLIST == original
