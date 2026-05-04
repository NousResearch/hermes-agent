"""Tests for Unicode dash normalization in /insights command flag parsing.

Telegram on iOS auto-converts -- to em/en dashes. The /insights handler uses
the shared insights parser, which normalizes these before parsing flags.
"""
import pytest

from agent.insights import parse_insights_args


class TestInsightsUnicodeDashFlags:
    """--days and --source must survive iOS Unicode dash conversion."""

    @pytest.mark.parametrize("input_str,expected", [
        # Standard double hyphen (baseline)
        ("--days 7", {"days": 7}),
        ("--source telegram", {"source": "telegram"}),
        # Em dash (U+2014)
        ("\u2014days 7", {"days": 7}),
        ("\u2014source telegram", {"source": "telegram"}),
        # En dash (U+2013)
        ("\u2013days 7", {"days": 7}),
        ("\u2013source telegram", {"source": "telegram"}),
        # Figure dash (U+2012)
        ("\u2012days 7", {"days": 7}),
        # Horizontal bar (U+2015)
        ("\u2015days 7", {"days": 7}),
        # Combined flags with em dashes
        ("\u2014days 30 \u2014source cli", {"days": 30, "source": "cli"}),
        # Qualitative mode also accepts Unicode dashes
        ("\u2014qualitative \u2014days 14", {"qualitative": True, "days": 14}),
        ("\u2014qualitative \u2014no-write", {"qualitative": True, "write_report": False}),
    ])
    def test_unicode_dash_normalized(self, input_str, expected):
        result = parse_insights_args(input_str)
        for key, value in expected.items():
            assert result[key] == value

    def test_regular_hyphens_unaffected(self):
        """Normal --days/--source must pass through unchanged."""
        result = parse_insights_args("--days 7 --source discord")
        assert result["days"] == 7
        assert result["source"] == "discord"

    def test_bare_number_still_works(self):
        """Shorthand /insights 7 (no flag) must not be mangled."""
        assert parse_insights_args("7")["days"] == 7

    def test_no_flags_unchanged(self):
        """Input with no flags passes through as-is."""
        assert parse_insights_args("")["days"] == 30
        assert parse_insights_args("30")["days"] == 30
