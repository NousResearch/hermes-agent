"""Tests for parse_reset_after_seconds — honoring the reset window an
aggregator gateway states in its 429 body when it sends no Retry-After header.

The sample strings are verbatim shapes taken from a self-hosted multi-provider
router's error output.
"""

import pytest

from agent.retry_utils import parse_reset_after_seconds


class TestParsesRealGatewayShapes:
    def test_seconds_only(self):
        text = (
            'HTTP 429: [antigravity/claude-sonnet-4-6] [429]: {\n  "error": {\n'
            '    "code": 429,\n    "message": "Resource has been exhausted '
            '(e.g. check quota).",\n    "status": "RESOURCE_EXHAUSTED"\n  }\n}\n '
            '(reset after 4s)'
        )
        assert parse_reset_after_seconds(text) == 4.0

    def test_minutes_and_seconds(self):
        text = (
            'HTTP 429: [claude/claude-opus-5] [429]: {"type":"error","error":'
            '{"type":"rate_limit_error"}} (reset after 2m 13s)'
        )
        assert parse_reset_after_seconds(text) == 133.0

    def test_minutes_only(self):
        text = '[kiro/claude-sonnet-4.5] [400]: ... (reset after 2m)'
        assert parse_reset_after_seconds(text) == 120.0

    def test_hours_component_is_parsed_but_capped_away(self):
        # The h-branch parses, then the 600s cap rejects it: an hour-long wall
        # is not something to sleep through — fall back to normal failover.
        assert parse_reset_after_seconds("(reset after 1h 5m)") is None
        # Just under the cap still comes back.
        assert parse_reset_after_seconds("(reset after 9m 30s)") == 570.0

    def test_unavailable_shape_without_json_body(self):
        text = 'HTTP 429: [antigravity/gemini-pro-agent] Unavailable (reset after 2m 43s)'
        assert parse_reset_after_seconds(text) == 163.0

    @pytest.mark.parametrize("raw,expected", [
        ("(reset after 30s)", 30.0),
        ("(reset after 16s)", 16.0),
        ("(reset after 9s)", 9.0),
        ("(reset after 4m 15s)", 255.0),
        ("(RESET AFTER 12S)", 12.0),
    ])
    def test_observed_windows(self, raw, expected):
        assert parse_reset_after_seconds(raw) == expected


class TestRejectsUnusableValues:
    def test_no_hint_returns_none(self):
        assert parse_reset_after_seconds("HTTP 429: rate limited") is None

    def test_empty_and_none(self):
        assert parse_reset_after_seconds("") is None
        assert parse_reset_after_seconds(None) is None

    def test_zero_window_is_none(self):
        assert parse_reset_after_seconds("(reset after 0s)") is None

    def test_window_beyond_cap_is_none(self):
        # 20 minutes is not an interactive wait — fall back to normal failover.
        assert parse_reset_after_seconds("(reset after 20m)") is None

    def test_does_not_confuse_milliseconds_for_minutes(self):
        # "500ms" must not be read as 500 minutes by the m-branch.
        assert parse_reset_after_seconds("(reset after 500ms)") is None
