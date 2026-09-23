"""Tests for token/context count display formatting — decimal units per industry convention.

Hermes displays token counts with decimal (1000-based) units, lowercase ``k`` for
thousands and uppercase ``M``/``B`` for millions/billions. This matches the
convention used by Claude Code (``43k/200k``), GitHub Copilot CLI (``26.6k/200k``),
and every major LLM billing table (prices are quoted per 1M = 1,000,000 tokens).
"""

import hermes_cli.banner as banner
from agent.usage_pricing import format_token_count_compact as pricing_format_token_count_compact
from cli import format_token_count_compact as cli_format_token_count_compact


class TestFormatContextLength:
    """_format_context_length must use decimal units (k=1000, M=1,000,000)."""

    def test_k_values_round(self):
        assert banner._format_context_length(65536) == "65.5k"    # 65536 / 1000 = 65.5
        assert banner._format_context_length(131072) == "131.1k"  # 131072 / 1000 = 131.07
        assert banner._format_context_length(262144) == "262.1k"  # 262144 / 1000 = 262.14
        assert banner._format_context_length(200000) == "200k"    # Claude 200K window, decimal
        assert banner._format_context_length(128000) == "128k"    # GPT-class 128K window

    def test_m_values(self):
        # 1,000,000 tokens → "1M" — the 1M-token window marketing name
        assert banner._format_context_length(1000000) == "1M"
        assert banner._format_context_length(2000000) == "2M"

    def test_m_values_fractional(self):
        assert banner._format_context_length(1500000) == "1.5M"

    def test_edge_cases(self):
        assert banner._format_context_length(1000) == "1k"        # exactly 1k
        assert banner._format_context_length(999) == "999"        # below k threshold
        assert banner._format_context_length(0) == "0"            # zero


class TestPricingFormatTokenCountCompact:
    """format_token_count_compact (pricing module) uses decimal units."""

    def test_k(self):
        assert pricing_format_token_count_compact(1000) == "1k"
        assert pricing_format_token_count_compact(12450) == "12.4k"
        assert pricing_format_token_count_compact(65536) == "65.5k"

    def test_m(self):
        assert pricing_format_token_count_compact(1000000) == "1M"
        assert pricing_format_token_count_compact(2097152) == "2.1M"

    def test_b(self):
        assert pricing_format_token_count_compact(1000000000) == "1B"

    def test_edge_cases(self):
        assert pricing_format_token_count_compact(999) == "999"
        assert pricing_format_token_count_compact(-1000) == "-1k"


class TestCliFormatTokenCountCompact:
    """CLI format_token_count_compact (cli.py, used by the status bar)
    uses decimal units."""

    def test_k(self):
        assert cli_format_token_count_compact(1000) == "1k"
        assert cli_format_token_count_compact(200000) == "200k"

    def test_m(self):
        assert cli_format_token_count_compact(1000000) == "1M"

    def test_b(self):
        assert cli_format_token_count_compact(1000000000) == "1B"

    def test_edge_cases(self):
        assert cli_format_token_count_compact(999) == "999"
        assert cli_format_token_count_compact(0) == "0"