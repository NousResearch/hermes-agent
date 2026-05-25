"""Tests for the date-injection fairness shim."""
from datetime import date
from benchmarks.dated_ingestion import parse_session_date, resolve_relative_dates


class TestDateInjection:
    """Verify the date-injection shim correctly enriches benchmark data."""

    def test_parse_session_date_valid(self):
        """Valid LoCoMo-format date strings should parse correctly."""
        dt = parse_session_date("1:56 pm on 8 May, 2023")
        assert dt is not None
        assert dt == date(2023, 5, 8)

    def test_parse_session_date_alternate_format(self):
        """Another valid format should parse."""
        dt = parse_session_date("10:00 am on 15 June, 2023")
        assert dt == date(2023, 6, 15)

    def test_parse_session_date_invalid_returns_none(self):
        """Invalid date strings should return None."""
        assert parse_session_date("not-a-date") is None
        assert parse_session_date("") is None
        assert parse_session_date(None) is None

    def test_resolve_yesterday(self):
        """'yesterday' relative to a known date should resolve correctly."""
        results = resolve_relative_dates("I went there yesterday", date(2023, 5, 8))
        assert len(results) > 0
        assert any("7 May 2023" in r for r in results)

    def test_resolve_last_year(self):
        """'last year' should map to previous calendar year."""
        results = resolve_relative_dates("That was last year", date(2023, 5, 8))
        assert len(results) > 0
        assert any("2022" in r for r in results)

    def test_no_markers_returns_empty(self):
        """Text with no relative-time markers should return empty list."""
        results = resolve_relative_dates("Hello world", date(2023, 5, 8))
        assert results == []

    def test_multiple_markers_in_one_text(self):
        """Text with multiple markers should resolve all of them."""
        results = resolve_relative_dates(
            "I went yesterday and last week I also went",
            date(2023, 5, 8)
        )
        assert len(results) >= 2
