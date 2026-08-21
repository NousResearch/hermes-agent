"""Unit tests for the `hermes sessions cost` computation helpers."""

import pytest

from hermes_cli.session_cost import (
    CACHE_HIT_WARN_THRESHOLD_PCT,
    DEFAULT_CACHE_HIT_RATIO,
    cache_hit_ratio_from_config,
    format_hit_pct,
    format_tokens,
    format_usd,
    session_cost_breakdown,
)


def _row(**overrides):
    row = {
        "input_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "output_tokens": 0,
        "estimated_cost_usd": None,
    }
    row.update(overrides)
    return row


class TestCacheHitPct:
    def test_hit_pct_uses_input_side_denominator(self):
        # hit% = cache_read / (input + cache_read + cache_write)
        b = session_cost_breakdown(_row(input_tokens=1000, cache_read_tokens=500))
        assert b["cache_hit_pct"] == pytest.approx(100 / 3)

    def test_hit_pct_includes_cache_write_in_denominator(self):
        b = session_cost_breakdown(
            _row(
                input_tokens=1000,
                cache_read_tokens=500,
                cache_write_tokens=100,
            )
        )
        # 500 / (1000 + 500 + 100)
        assert b["cache_hit_pct"] == pytest.approx(31.25)

    def test_hit_pct_is_none_without_input_side_tokens(self):
        b = session_cost_breakdown(_row(output_tokens=100))
        assert b["cache_hit_pct"] is None
        assert b["below_threshold"] is False

    def test_zero_cache_read_is_zero_hit(self):
        b = session_cost_breakdown(_row(input_tokens=1000))
        assert b["cache_hit_pct"] == pytest.approx(0.0)


class TestCounterfactual:
    def test_all_cache_reads_cost_10x_at_default_ratio(self):
        # Every input-side token is a cache read billed at 10% of cold:
        # the 0% counterfactual is 10x the recorded cost.
        b = session_cost_breakdown(
            _row(cache_read_tokens=1000, estimated_cost_usd=1.0)
        )
        assert b["counterfactual_cost"] == pytest.approx(10.0)
        assert b["savings"] == pytest.approx(9.0)

    def test_mixed_session_attribution(self):
        b = session_cost_breakdown(
            _row(
                input_tokens=3000,
                cache_read_tokens=1000,
                cache_write_tokens=1000,
                output_tokens=5000,
                estimated_cost_usd=10.0,
            )
        )
        # f_in = 5000/10000 = 0.5, f_cr = 1000/5000 = 0.2
        # savings = 10 * 0.5 * 0.2 * (1/0.1 - 1) = 9.0
        assert b["savings"] == pytest.approx(9.0)
        assert b["counterfactual_cost"] == pytest.approx(19.0)

    def test_no_cache_reads_no_savings(self):
        b = session_cost_breakdown(
            _row(input_tokens=1000, estimated_cost_usd=5.0)
        )
        assert b["savings"] == pytest.approx(0.0)
        assert b["counterfactual_cost"] == pytest.approx(5.0)

    def test_unknown_cost_yields_unknown_counterfactual(self):
        b = session_cost_breakdown(_row(input_tokens=1000, cache_read_tokens=500))
        assert b["counterfactual_cost"] is None
        assert b["savings"] is None

    def test_zero_tokens_with_cost_is_pass_through(self):
        b = session_cost_breakdown(_row(estimated_cost_usd=2.0))
        assert b["counterfactual_cost"] == pytest.approx(2.0)
        assert b["savings"] == pytest.approx(0.0)

    def test_custom_ratio_changes_counterfactual(self):
        b = session_cost_breakdown(
            _row(cache_read_tokens=1000, estimated_cost_usd=1.0),
            cache_hit_ratio=0.25,
        )
        # 1/0.25 - 1 = 3 -> 4x total
        assert b["counterfactual_cost"] == pytest.approx(4.0)

    def test_invalid_ratio_falls_back_to_default(self):
        b = session_cost_breakdown(
            _row(cache_read_tokens=1000, estimated_cost_usd=1.0),
            cache_hit_ratio=1.5,
        )
        assert b["counterfactual_cost"] == pytest.approx(10.0)


class TestWarnMarker:
    def test_below_threshold_flagged(self):
        b = session_cost_breakdown(
            _row(
                input_tokens=1000,
                cache_read_tokens=690,
                cache_write_tokens=310,
            )
        )
        assert b["cache_hit_pct"] < CACHE_HIT_WARN_THRESHOLD_PCT
        assert b["below_threshold"] is True

    def test_at_threshold_not_flagged(self):
        b = session_cost_breakdown(
            _row(
                input_tokens=30,
                cache_read_tokens=70,
            )
        )
        assert b["cache_hit_pct"] == pytest.approx(CACHE_HIT_WARN_THRESHOLD_PCT)
        assert b["below_threshold"] is False

    def test_above_threshold_not_flagged(self):
        b = session_cost_breakdown(
            _row(input_tokens=100, cache_read_tokens=900)
        )
        assert b["below_threshold"] is False

    def test_no_usage_not_flagged(self):
        b = session_cost_breakdown(_row())
        assert b["below_threshold"] is False


class TestTokenFields:
    def test_missing_columns_default_to_zero(self):
        b = session_cost_breakdown({"id": "x"})
        assert b["input_tokens"] == 0
        assert b["cache_read_tokens"] == 0
        assert b["cache_write_tokens"] == 0
        assert b["output_tokens"] == 0
        assert b["input_side_tokens"] == 0
        assert b["total_tokens"] == 0

    def test_non_numeric_cost_becomes_none(self):
        b = session_cost_breakdown(
            _row(input_tokens=10, estimated_cost_usd="n/a")
        )
        assert b["estimated_cost"] is None
        assert b["counterfactual_cost"] is None


class TestCacheHitRatioFromConfig:
    def test_missing_uses_default(self):
        assert cache_hit_ratio_from_config({}) == DEFAULT_CACHE_HIT_RATIO
        assert cache_hit_ratio_from_config(None) == DEFAULT_CACHE_HIT_RATIO

    def test_explicit_value_wins(self):
        assert (
            cache_hit_ratio_from_config(
                {"cost": {"cache_hit_ratio": 0.25}}
            )
            == pytest.approx(0.25)
        )

    def test_invalid_values_fall_back(self):
        for bad in (0, -1, 1.5, 2, "x", None):
            assert (
                cache_hit_ratio_from_config({"cost": {"cache_hit_ratio": bad}})
                == DEFAULT_CACHE_HIT_RATIO
            )

    def test_non_dict_cost_section_falls_back(self):
        assert (
            cache_hit_ratio_from_config({"cost": "oops"})
            == DEFAULT_CACHE_HIT_RATIO
        )


class TestFormatting:
    def test_format_usd(self):
        assert format_usd(None) == "—"
        assert format_usd(1.5) == "$1.50"
        assert format_usd(1234.567) == "$1,234.57"
        # sub-cent amounts keep 4 decimals so tiny sessions are not $0.00
        assert format_usd(0.004) == "$0.0040"

    def test_format_hit_pct(self):
        assert format_hit_pct(None) == "—"
        assert format_hit_pct(31.25) == "31.2%"
        assert format_hit_pct(100.0) == "100.0%"

    def test_format_tokens(self):
        assert format_tokens(None) == "—"
        assert format_tokens(0) == "0"
        assert format_tokens(12345) == "12,345"
