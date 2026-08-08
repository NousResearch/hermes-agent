"""Tests for _extract_retry_delay_seconds in credential_pool.

This function is the fallback retry-delay parser used by
_normalize_error_context when the upstream error_context does not
already carry a ``reset_at`` timestamp.  The regex must be at least as
broad as the one in ``extract_api_error_context`` (agent_runtime_helpers)
so that the fallback path never silently degrades to the default 1-hour
cooldown when the message contains a parseable duration.
"""

from __future__ import annotations

import pytest

from agent.credential_pool import _extract_retry_delay_seconds


# ── quotaResetDelay format ──────────────────────────────────────────

class TestQuotaResetDelay:
    def test_ms_suffix(self):
        assert _extract_retry_delay_seconds("quotaResetDelay: 5000ms") == 5.0

    def test_s_suffix(self):
        assert _extract_retry_delay_seconds("quotaResetDelay: 30s") == 30.0

    def test_decimal_ms(self):
        assert _extract_retry_delay_seconds("quotaResetDelay: 1500ms") == 1.5


# ── "retry after N seconds" format ────────────────────────────────

class TestRetryAfterSeconds:
    def test_retry_after_seconds(self):
        assert _extract_retry_delay_seconds("retry after 60 seconds") == 60.0

    def test_retry_after_s(self):
        assert _extract_retry_delay_seconds("retry after 90s") == 90.0

    def test_retry_s_no_after(self):
        assert _extract_retry_delay_seconds("retry 30s") == 30.0

    def test_retry_uppercase(self):
        assert _extract_retry_delay_seconds("Retry After 120 Seconds") == 120.0


# ── "Resets in Xhr Ymin" — OpenCode Go format (original coverage) ─

class TestResetsInOriginal:
    def test_hr_min(self):
        assert _extract_retry_delay_seconds("Resets in 4hr 5min") == 14700.0

    def test_hr_min_spaced(self):
        assert _extract_retry_delay_seconds("resets in 4 hr 5 min") == 14700.0

    def test_hr_only(self):
        assert _extract_retry_delay_seconds("Resets in 4hr") == 14400.0

    def test_min_only(self):
        assert _extract_retry_delay_seconds("Resets in 5min") == 300.0


# ── "Resets in" with broader unit aliases (previously unparseable) ─

class TestResetsInBroadened:
    """These formats were NOT parsed by the old regex but ARE parsed by
    extract_api_error_context.  The fallback must match the upstream
    coverage to avoid silent 1-hour default cooldown."""

    def test_single_letter_h_m(self):
        assert _extract_retry_delay_seconds("Resets in 4h 5m") == 14700.0

    def test_sec_only(self):
        assert _extract_retry_delay_seconds("Resets in 90 sec") == 90.0

    def test_s_only(self):
        assert _extract_retry_delay_seconds("Resets in 90s") == 90.0

    def test_full_words_hr_min(self):
        assert _extract_retry_delay_seconds("Resets in 2 hours 30 minutes") == 9000.0

    def test_hour_full_only(self):
        assert _extract_retry_delay_seconds("Resets in 1 hour") == 3600.0

    def test_minutes_full_only(self):
        assert _extract_retry_delay_seconds("Resets in 45 minutes") == 2700.0

    def test_seconds_full_only(self):
        assert _extract_retry_delay_seconds("Resets in 30 seconds") == 30.0

    def test_h_m_single_letter(self):
        assert _extract_retry_delay_seconds("Resets in 5h 30m") == 19800.0

    def test_hrs_mins(self):
        assert _extract_retry_delay_seconds("Resets in 2hrs 15mins") == 8100.0

    def test_all_three_components(self):
        assert _extract_retry_delay_seconds("Resets in 1h 30m 15s") == 5415.0


# ── Embedded in longer messages ────────────────────────────────────

class TestEmbeddedInMessage:
    def test_rate_limit_prefix(self):
        msg = "rate limit exceeded. Resets in 2hr 30min"
        assert _extract_retry_delay_seconds(msg) == 9000.0

    def test_429_prefix(self):
        msg = "429: Resets in 1hr 0min"
        assert _extract_retry_delay_seconds(msg) == 3600.0

    def test_quota_exceeded_prefix(self):
        msg = "Quota exceeded for generate_content_free_tier_requests. Resets in 24 hours"
        assert _extract_retry_delay_seconds(msg) == 86400.0


# ── Negative cases ──────────────────────────────────────────────────

class TestNoMatch:
    def test_empty_string(self):
        assert _extract_retry_delay_seconds("") is None

    def test_none_relevant(self):
        assert _extract_retry_delay_seconds("no relevant message") is None

    def test_random_numbers(self):
        assert _extract_retry_delay_seconds("error 12345 occurred") is None

    def test_resets_without_duration(self):
        assert _extract_retry_delay_seconds("resets in") is None
