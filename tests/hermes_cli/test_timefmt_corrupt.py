"""Corrupt-timestamp hardening for hermes_cli.timefmt.relative_time (#102399).

Kept in a separate module from test_timefmt.py (open PR #98670 owns
that path) to avoid colliding with it.
"""

import time

from hermes_cli.timefmt import relative_time


class TestCorruptTimestampsRenderUnknown:
    def test_text_timestamp(self):
        assert relative_time("not-a-timestamp") == "?"

    def test_iso_string(self):
        assert relative_time("2026-01-01") == "?"

    def test_short_string(self):
        assert relative_time("abc") == "?"

    def test_list_and_dict(self):
        assert relative_time([1]) == "?"
        assert relative_time({"t": 1}) == "?"

    def test_non_finite_floats(self):
        assert relative_time(float("nan")) == "?"
        assert relative_time(float("inf")) == "?"
        # finite but out of fromtimestamp range -> "?" not OverflowError
        assert relative_time(-1e300) == "?"

    def test_numeric_string_coerces(self):
        assert relative_time(str(time.time() - 30)) == "just now"

    def test_valid_epochs_unchanged(self):
        assert relative_time(None) == "?"
        assert relative_time(0) == "?"
        assert relative_time(time.time() - 30) == "just now"
        assert relative_time(time.time() - 7200) == "2h ago"
