"""Tests for token usage footer formatting in gateway responses."""

import pytest


def _format_usage_footer(input_tokens, output_tokens, model, cost_amount=None, cost_status="estimated"):
    """Extract the formatting logic from gateway/run.py for unit testing."""

    def _fmt(n):
        return f"{n/1000:.1f}k" if n >= 1000 else str(n)

    _cost_str = ""
    if cost_amount is not None:
        _prefix = "~" if cost_status == "estimated" else ""
        _cost_str = f" · {_prefix}${float(cost_amount):.4f}"

    _model_short = model.split("/")[-1] if "/" else model

    return (
        f"📊 {_fmt(input_tokens)} in · {_fmt(output_tokens)} out"
        f"{_cost_str} · `{_model_short}`"
    )


class TestUsageFooterFormatting:
    def test_small_token_counts(self):
        result = _format_usage_footer(500, 200, "anthropic/claude-sonnet-4")
        assert "500 in" in result
        assert "200 out" in result
        assert "claude-sonnet-4" in result

    def test_large_token_counts(self):
        result = _format_usage_footer(12000, 5600, "openai/gpt-4o")
        assert "12.0k in" in result
        assert "5.6k out" in result
        assert "gpt-4o" in result

    def test_model_without_provider(self):
        result = _format_usage_footer(100, 50, "claude-sonnet-4")
        assert "claude-sonnet-4" in result

    def test_estimated_cost(self):
        result = _format_usage_footer(1000, 500, "claude", cost_amount=0.0234, cost_status="estimated")
        assert "~$0.0234" in result

    def test_exact_cost(self):
        result = _format_usage_footer(1000, 500, "claude", cost_amount=0.0234, cost_status="cached")
        assert "$0.0234" in result
        assert "~$0.0234" not in result

    def test_no_cost(self):
        result = _format_usage_footer(100, 50, "claude")
        assert "$" not in result

    def test_zero_tokens(self):
        result = _format_usage_footer(0, 0, "claude")
        assert "0 in" in result
        assert "0 out" in result


class TestConfigDefault:
    def test_token_usage_footer_default_false(self):
        from hermes_cli.config import DEFAULT_CONFIG
        assert DEFAULT_CONFIG["display"]["token_usage_footer"] is False
