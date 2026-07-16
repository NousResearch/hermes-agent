"""Tests for context length display formatting — binary units per Hermes convention."""

import hermes_cli.banner as banner
from agent.usage_pricing import format_token_count_compact as pricing_format_token_count_compact
from cli import format_token_count_compact as cli_format_token_count_compact


class TestFormatContextLength:
    """_format_context_length must use binary units (K=1024, M=1024²) per Hermes docs:
    'k = 1000, uppercase K = 1024'."""

    def test_k_values_round(self):
        assert banner._format_context_length(65536) == "64K"       # 64 × 1024
        assert banner._format_context_length(131072) == "128K"     # 128 × 1024
        assert banner._format_context_length(262144) == "256K"     # 256 × 1024
        assert banner._format_context_length(524288) == "512K"     # 512 × 1024
        assert banner._format_context_length(1048576) == "1M"      # 1 × 1024²
        assert banner._format_context_length(2097152) == "2M"      # 2 × 1024²

    def test_k_values_fractional(self):
        # Values that don't round cleanly show one decimal place
        assert banner._format_context_length(64000) == "62.5K"     # 64000 / 1024 = 62.5
        assert banner._format_context_length(70000) == "68.4K"     # 70000 / 1024 ≈ 68.36

    def test_m_values_fractional(self):
        assert banner._format_context_length(1572864) == "1.5M"    # 1.5 × 1024²

    def test_edge_cases(self):
        assert banner._format_context_length(1024) == "1K"         # exactly 1K
        assert banner._format_context_length(1023) == "1023"       # below K threshold
        assert banner._format_context_length(999) == "999"         # small number
        assert banner._format_context_length(0) == "0"             # zero


class TestPricingFormatTokenCountCompact:
    """format_token_count_compact (pricing module) must use binary units."""

    def test_binary_k(self):
        assert pricing_format_token_count_compact(1024) == "1K"
        assert pricing_format_token_count_compact(65536) == "64K"
        assert pricing_format_token_count_compact(131072) == "128K"

    def test_binary_m(self):
        assert pricing_format_token_count_compact(1048576) == "1M"
        assert pricing_format_token_count_compact(2097152) == "2M"

    def test_binary_b(self):
        assert pricing_format_token_count_compact(1073741824) == "1B"

    def test_edge_cases(self):
        assert pricing_format_token_count_compact(999) == "999"
        assert pricing_format_token_count_compact(-1024) == "-1K"


class TestCliFormatTokenCountCompact:
    """CLI format_token_count_compact (cli.py, used by the status bar)
    must also use binary units (K=1024, M=1024², B=1024³)."""

    def test_binary_k(self):
        assert cli_format_token_count_compact(1024) == "1K"
        assert cli_format_token_count_compact(65536) == "64K"
        assert cli_format_token_count_compact(131072) == "128K"

    def test_binary_m(self):
        assert cli_format_token_count_compact(1048576) == "1M"
        assert cli_format_token_count_compact(2097152) == "2M"

    def test_binary_b(self):
        assert cli_format_token_count_compact(1073741824) == "1B"

    def test_fractional_k(self):
        # CLI formatter uses 2 decimal places for <10, 1 decimal for 10-100, 0 for >=100
        assert cli_format_token_count_compact(64000) == "62.5K"
        assert cli_format_token_count_compact(70000) == "68.4K"
        assert cli_format_token_count_compact(5000) == "4.88K"

    def test_fractional_m(self):
        assert cli_format_token_count_compact(1572864) == "1.5M"
        assert cli_format_token_count_compact(1048577) == "1M"

    def test_edge_cases(self):
        assert cli_format_token_count_compact(999) == "999"
        assert cli_format_token_count_compact(1023) == "1023"
        assert cli_format_token_count_compact(0) == "0"
        assert cli_format_token_count_compact(-1024) == "-1K"
