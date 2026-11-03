"""Tests for unified_reset header validation in extract_api_error_context.

Covers the NaN / Inf / out-of-range-epoch guard added to
agent/agent_runtime_helpers.py:

- 'nan'  -> context['reset_at'] must NOT be set
- 'inf'  -> context['reset_at'] must NOT be set
- '-inf' -> context['reset_at'] must NOT be set
- negative epoch (-1.0) -> context['reset_at'] must NOT be set
- epoch too far in the past (pre-2020) -> NOT set
- valid epoch -> context['reset_at'] IS set and takes priority over retry-after
- invalid unified_reset -> fallback to retry-after populates reset_at
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from agent.agent_runtime_helpers import extract_api_error_context


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_error(unified_reset=None, retry_after=None):
    """Build a minimal fake API error with optional header values."""
    headers = {}
    if unified_reset is not None:
        headers["anthropic-ratelimit-unified-reset"] = unified_reset
    if retry_after is not None:
        headers["retry-after"] = retry_after
    mock_response = SimpleNamespace(headers=headers)
    err = Exception("rate limited")
    err.response = mock_response  # type: ignore[attr-defined]
    err.body = None  # type: ignore[attr-defined]
    return err


# ---------------------------------------------------------------------------
# NaN / Inf / negative — must be rejected
# ---------------------------------------------------------------------------

class TestUnifiedResetRejected:
    def test_nan_does_not_set_reset_at(self):
        ctx = extract_api_error_context(_make_error(unified_reset="nan"))
        assert "reset_at" not in ctx, "'nan' should not populate reset_at"

    def test_positive_inf_does_not_set_reset_at(self):
        ctx = extract_api_error_context(_make_error(unified_reset="inf"))
        assert "reset_at" not in ctx, "'inf' should not populate reset_at"

    def test_negative_inf_does_not_set_reset_at(self):
        ctx = extract_api_error_context(_make_error(unified_reset="-inf"))
        assert "reset_at" not in ctx, "'-inf' should not populate reset_at"

    def test_negative_epoch_does_not_set_reset_at(self):
        ctx = extract_api_error_context(_make_error(unified_reset="-1.0"))
        assert "reset_at" not in ctx, "negative epoch should not populate reset_at"

    def test_pre_2020_epoch_does_not_set_reset_at(self):
        # 2019-01-01 epoch = 1546300800 — below the 1_600_000_000 threshold
        ctx = extract_api_error_context(_make_error(unified_reset="1546300800"))
        assert "reset_at" not in ctx, "pre-2020 epoch should not populate reset_at"

    def test_far_future_epoch_does_not_set_reset_at(self):
        # Year 2200 would be epoch ~7258118400, above the 4_000_000_000 bound
        ctx = extract_api_error_context(_make_error(unified_reset="7258118400"))
        assert "reset_at" not in ctx, "far-future epoch should not populate reset_at"


# ---------------------------------------------------------------------------
# Valid epoch — must be accepted and take priority over retry-after
# ---------------------------------------------------------------------------

class TestUnifiedResetAccepted:
    def test_valid_epoch_sets_reset_at(self):
        valid = "1950000000"  # well within [2020-2096]
        ctx = extract_api_error_context(_make_error(unified_reset=valid))
        assert ctx.get("reset_at") == pytest.approx(1_950_000_000.0)

    def test_valid_epoch_beats_retry_after(self):
        """unified_reset (epoch) must take priority over retry-after (relative)."""
        valid = "1950000000"
        ctx = extract_api_error_context(
            _make_error(unified_reset=valid, retry_after="3600")
        )
        # reset_at should be the absolute epoch, not time.time() + 3600
        assert ctx.get("reset_at") == pytest.approx(1_950_000_000.0)


# ---------------------------------------------------------------------------
# Fallback: invalid unified_reset -> retry-after still works
# ---------------------------------------------------------------------------

class TestUnifiedResetFallback:
    def test_nan_unified_reset_falls_back_to_retry_after(self, monkeypatch):
        """When unified_reset is invalid, retry-after must still set reset_at."""
        import agent.agent_runtime_helpers as _mod
        monkeypatch.setattr(_mod.time, "time", lambda: 1_000_000.0)

        ctx = extract_api_error_context(
            _make_error(unified_reset="nan", retry_after="7200")
        )
        assert ctx.get("reset_at") == pytest.approx(1_000_000.0 + 7200.0)

    def test_negative_epoch_falls_back_to_retry_after(self, monkeypatch):
        import agent.agent_runtime_helpers as _mod
        monkeypatch.setattr(_mod.time, "time", lambda: 2_000_000.0)

        ctx = extract_api_error_context(
            _make_error(unified_reset="-100", retry_after="1800")
        )
        assert ctx.get("reset_at") == pytest.approx(2_000_000.0 + 1800.0)
