"""Token/context display contract: decimal units with a lowercase ``k``.

Hermes divides token counts by 1000 (decimal) and labels the result ``k`` / ``M`` / ``B`` —
matching Claude Code (``43k/200k``), GitHub Copilot CLI, and per-1M token price tables. The
label must not be the binary-looking uppercase ``K``, which reads as 1024 while the divisor
is 1000.
"""

import hermes_cli.banner as banner
from agent.usage_pricing import format_token_count_compact as pricing_compact
from cli import format_token_count_compact as cli_compact


def test_context_length_formatter_is_decimal_with_lowercase_k():
    """``_format_context_length`` renders the decimal value of the count: a 65,536-token
    window reads 65.5k (not the binary 64k), and the round window sizes keep round labels."""
    assert banner._format_context_length(1000) == "1k"
    assert banner._format_context_length(65536) == "65.5k"      # 65536 / 1000
    assert banner._format_context_length(200000) == "200k"      # 200k-token window
    assert banner._format_context_length(1000000) == "1M"       # the 1M-token window
    assert banner._format_context_length(999) == "999"          # below the k threshold


def test_token_formatters_agree_on_decimal_lowercase_suffix():
    """The three token-count formatters (banner/status bar, CLI, pricing) are one
    convention: the same count never renders with a binary-looking uppercase ``K``."""
    for value in (1_000, 65_536, 200_000):
        rendered = {
            banner._format_context_length(value),
            cli_compact(value),
            pricing_compact(value),
        }
        assert all(text.endswith("k") for text in rendered), rendered
