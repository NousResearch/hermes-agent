"""Unit tests for the Telegram context-window HUD formatter.

The formatter is a pure function — no agent state, no platform calls.  These
tests pin down the user-facing output shape so the gateway integration can
trust it.
"""

from __future__ import annotations

import pytest

from gateway.context_hud import compact_tokens, format_hud


# ── compact_tokens ────────────────────────────────────────────────────────


class TestCompactTokens:
    def test_below_1k_is_plain_int(self):
        assert compact_tokens(0) == "0"
        assert compact_tokens(1) == "1"
        assert compact_tokens(999) == "999"

    def test_thousands_use_k_suffix(self):
        assert compact_tokens(1000) == "1k"
        assert compact_tokens(23_400) == "23k"
        assert compact_tokens(250_000) == "250k"
        assert compact_tokens(999_499) == "999k"

    def test_millions_use_M_with_one_decimal(self):
        assert compact_tokens(1_000_000) == "1.0M"
        assert compact_tokens(1_234_567) == "1.2M"
        assert compact_tokens(12_500_000) == "12.5M"

    def test_negative_clamped_to_zero(self):
        assert compact_tokens(-50) == "0"


# ── format_hud — shape and contents ───────────────────────────────────────


class TestFormatHudShape:
    def test_two_lines(self):
        hud = format_hud(used_tokens=23_000, context_length=250_000)
        assert hud is not None
        assert hud.count("\n") == 1, "HUD must be exactly two lines"

    def test_first_line_has_used_limit_percent(self):
        hud = format_hud(used_tokens=23_000, context_length=250_000)
        first, _ = hud.split("\n", 1)
        assert "23k" in first
        assert "250k" in first
        assert "9%" in first  # 23000/250000 = 9.2% -> rounds to 9
        assert "/" in first

    def test_second_line_is_bar_with_brackets(self):
        hud = format_hud(used_tokens=23_000, context_length=250_000, bar_width=20)
        _, second = hud.split("\n", 1)
        assert second.startswith("[")
        assert second.endswith("]")
        inside = second[1:-1]
        assert len(inside) == 20, f"bar must be exactly bar_width chars, got {len(inside)}"
        assert set(inside) <= {"█", "░"}, "only filled and empty cells allowed"

    def test_bar_width_param_changes_length(self):
        hud = format_hud(used_tokens=23_000, context_length=250_000, bar_width=10)
        _, second = hud.split("\n", 1)
        assert len(second) == 12  # 10 cells + 2 brackets


# ── format_hud — percent math and clamping ────────────────────────────────


class TestFormatHudMath:
    def test_zero_used_renders_zero_percent(self):
        # 0% would be hidden by default hide_below_percent=5 — disable hiding
        hud = format_hud(used_tokens=0, context_length=200_000, hide_below_percent=0)
        first, second = hud.split("\n", 1)
        assert "0%" in first
        assert second == "[" + "░" * 20 + "]"

    def test_full_usage_clamps_to_100(self):
        hud = format_hud(
            used_tokens=999_999, context_length=200_000, hide_below_percent=0
        )
        first, second = hud.split("\n", 1)
        assert "100%" in first
        assert second == "[" + "█" * 20 + "]"

    def test_negative_used_treated_as_zero(self):
        hud = format_hud(used_tokens=-5, context_length=200_000, hide_below_percent=0)
        assert "0%" in hud

    def test_zero_context_length_returns_none(self):
        assert format_hud(used_tokens=10, context_length=0) is None
        assert format_hud(used_tokens=10, context_length=-1) is None


# ── hide_below_percent ────────────────────────────────────────────────────


class TestHideBelowThreshold:
    def test_under_threshold_returns_none(self):
        # 1k / 250k = 0.4% — under default 5% threshold
        assert format_hud(used_tokens=1_000, context_length=250_000) is None

    def test_at_threshold_renders(self):
        # 5%  exactly should render
        hud = format_hud(used_tokens=10_000, context_length=200_000, hide_below_percent=5)
        assert hud is not None
        assert "5%" in hud

    def test_disable_hiding(self):
        hud = format_hud(used_tokens=1, context_length=250_000, hide_below_percent=0)
        assert hud is not None


# ── threshold labels ──────────────────────────────────────────────────────


class TestThresholdLabels:
    def test_normal_no_label_when_flag_off(self):
        hud = format_hud(
            used_tokens=10_000, context_length=200_000, show_warning_label=False
        )
        assert " warn" not in hud.lower()
        assert " danger" not in hud.lower()
        assert " critical" not in hud.lower()

    def test_warn_label_at_60_when_flag_on(self):
        hud = format_hud(
            used_tokens=120_000, context_length=200_000, show_warning_label=True
        )
        assert "warn" in hud.lower()
        assert "danger" not in hud.lower()

    def test_danger_label_at_75(self):
        hud = format_hud(
            used_tokens=150_000, context_length=200_000, show_warning_label=True
        )
        assert "danger" in hud.lower()
        assert "critical" not in hud.lower()

    def test_critical_label_at_90(self):
        hud = format_hud(
            used_tokens=190_000, context_length=200_000, show_warning_label=True
        )
        assert "critical" in hud.lower()


# ── hard constraints ─────────────────────────────────────────────────────


class TestHardConstraints:
    def test_no_emoji_anywhere(self):
        # Try a sampling of usage levels; none should include emoji
        for used, limit in [(0, 100), (5, 100), (50, 100), (90, 100), (200, 100)]:
            hud = format_hud(
                used_tokens=used,
                context_length=limit,
                hide_below_percent=0,
                show_warning_label=True,
            )
            if hud is None:
                continue
            # Scan for the most common emoji ranges and the obvious culprits
            for ch in hud:
                cp = ord(ch)
                assert not (0x1F300 <= cp <= 0x1FAFF), f"emoji {ch!r} in HUD"
                assert not (0x2600 <= cp <= 0x27BF), f"symbol/emoji {ch!r} in HUD"
                assert ch not in "✨📊⚡⏱️🤖✅❌⚠️🔴🟢🟡", f"emoji {ch!r} in HUD"

    def test_no_em_dash(self):
        for used in [0, 5_000, 100_000, 200_000]:
            hud = format_hud(
                used_tokens=used, context_length=200_000, hide_below_percent=0
            )
            if hud is None:
                continue
            assert "—" not in hud
            assert "–" not in hud

    def test_bar_width_stable_across_percentages(self):
        widths = []
        for pct_target in (0.0, 0.1, 0.5, 0.9, 1.0):
            hud = format_hud(
                used_tokens=int(200_000 * pct_target),
                context_length=200_000,
                hide_below_percent=0,
                bar_width=20,
            )
            _, second = hud.split("\n", 1)
            widths.append(len(second))
        assert len(set(widths)) == 1, f"bar widths drift: {widths}"
