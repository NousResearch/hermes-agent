"""Tests for hermes_cli/tips.py — random tip display at session start."""

import pytest
from hermes_cli.tips import TIPS, TIPS_EN, TIPS_ZH, get_random_tip


class TestTipsCorpus:
    """Validate the tip corpus itself."""

    def test_english_has_at_least_80_tips(self):
        assert len(TIPS_EN) >= 80, f"Expected 80+ English tips, got {len(TIPS_EN)}"

    def test_chinese_has_at_least_50_tips(self):
        assert len(TIPS_ZH) >= 50, f"Expected 50+ Chinese tips, got {len(TIPS_ZH)}"

    def test_backward_compat_alias(self):
        assert TIPS is TIPS_EN

    def test_no_duplicates_en(self):
        assert len(TIPS_EN) == len(set(TIPS_EN)), "Duplicate English tips found"

    def test_no_duplicates_zh(self):
        assert len(TIPS_ZH) == len(set(TIPS_ZH)), "Duplicate Chinese tips found"

    def test_all_tips_are_strings(self):
        for i, tip in enumerate(TIPS_EN):
            assert isinstance(tip, str), f"EN Tip {i} is not a string: {type(tip)}"
        for i, tip in enumerate(TIPS_ZH):
            assert isinstance(tip, str), f"ZH Tip {i} is not a string: {type(tip)}"

    def test_no_empty_tips(self):
        for i, tip in enumerate(TIPS_EN):
            assert tip.strip(), f"EN Tip {i} is empty or whitespace-only"
        for i, tip in enumerate(TIPS_ZH):
            assert tip.strip(), f"ZH Tip {i} is empty or whitespace-only"

    def test_max_length_reasonable(self):
        """Tips should fit on a single terminal line (~120 chars max)."""
        for i, tip in enumerate(TIPS_EN):
            assert len(tip) <= 150, (
                f"EN Tip {i} too long ({len(tip)} chars): {tip[:60]}..."
            )
        for i, tip in enumerate(TIPS_ZH):
            assert len(tip) <= 150, (
                f"ZH Tip {i} too long ({len(tip)} chars): {tip[:60]}..."
            )

    def test_no_leading_trailing_whitespace(self):
        for i, tip in enumerate(TIPS_EN):
            assert tip == tip.strip(), f"EN Tip {i} has leading/trailing whitespace"
        for i, tip in enumerate(TIPS_ZH):
            assert tip == tip.strip(), f"ZH Tip {i} has leading/trailing whitespace"


class TestGetRandomTip:
    """Validate the get_random_tip() function."""

    def test_returns_string(self):
        tip = get_random_tip()
        assert isinstance(tip, str)
        assert len(tip) > 0

    def test_returns_tip_from_corpus(self):
        tip = get_random_tip()
        assert tip in TIPS_EN or tip in TIPS_ZH

    def test_randomness(self):
        """Multiple calls should eventually return different tips."""
        seen = set()
        for _ in range(50):
            seen.add(get_random_tip())
        # With 200+ tips and 50 draws, we should see at least 10 unique
        assert len(seen) >= 10, f"Only got {len(seen)} unique tips in 50 draws"


class TestTipIntegrationInCLI:
    """Test that the tip display code in cli.py works correctly."""

    def test_tip_import_works(self):
        """The import used in cli.py must succeed."""
        from hermes_cli.tips import get_random_tip
        assert callable(get_random_tip)

    def test_tip_display_format(self):
        """Verify the Rich markup format doesn't break."""
        tip = get_random_tip()
        color = "#B8860B"
        markup = f"[dim {color}]✦ Tip: {tip}[/]"
        # Should not contain nested/broken Rich tags
        assert markup.count("[/]") == 1
        assert "[dim #B8860B]" in markup
