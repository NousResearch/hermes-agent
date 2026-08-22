"""Tests for agent.budget_hint — context-window budget hint injection.

Adapted from openai/codex's TokenBudgetRemainingContext: when the estimated
request approaches the model's context window, the model is told how much
room remains so it can keep responses focused and avoid forced truncation or
lossy compression.
"""

from __future__ import annotations

from agent.budget_hint import build_budget_hint


class TestBuildBudgetHint:
    def test_below_threshold_returns_none(self):
        # 10K / 100K = 10% < 70% threshold → no hint, prompt prefix stays stable.
        assert build_budget_hint(10_000, 100_000, 0.70) is None

    def test_at_threshold_returns_hint(self):
        hint = build_budget_hint(70_000, 100_000, 0.70)
        assert hint is not None
        assert "70%" in hint
        assert "30,000" in hint  # remaining tokens formatted with commas

    def test_above_threshold_returns_hint(self):
        hint = build_budget_hint(85_000, 100_000, 0.70)
        assert hint is not None
        assert "85%" in hint
        assert "15,000" in hint

    def test_full_window_reports_zero_remaining(self):
        hint = build_budget_hint(100_000, 100_000, 0.50)
        assert hint is not None
        assert "100%" in hint
        assert "0" in hint

    def test_disabled_threshold_never_injects(self):
        # threshold <= 0 disables the hint entirely, even at 100% usage.
        assert build_budget_hint(100_000, 100_000, 0.0) is None
        assert build_budget_hint(100_000, 100_000, -1.0) is None

    def test_degenerate_inputs_return_none(self):
        assert build_budget_hint(-1, 100_000, 0.70) is None  # negative usage
        assert build_budget_hint(50_000, 0, 0.70) is None  # unknown window
        assert build_budget_hint(50_000, -100, 0.70) is None  # negative window

    def test_hint_mentions_context_budget_explicitly(self):
        hint = build_budget_hint(80_000, 100_000, 0.50)
        assert hint is not None
        assert "Context budget" in hint
        assert "tokens" in hint


class TestComposeUserApiContentBudgetHint:
    """compose_user_api_content forwards budget_hint as a third injection."""

    def _compose(self):
        from agent.turn_context import compose_user_api_content

        return compose_user_api_content

    def test_budget_hint_appended_after_other_injections(self):
        compose = self._compose()
        out = compose("hello", "MEM", "PLUGIN", "HINT")
        assert out is not None
        assert out.startswith("hello")
        assert "HINT" in out
        # Hint comes last; memory block comes before plugin context.
        assert out.index("HINT") > out.index("PLUGIN")
        assert out.index("PLUGIN") > out.index("MEM")

    def test_budget_hint_only_injection(self):
        compose = self._compose()
        out = compose("hello", "", "", "HINT")
        assert out is not None
        assert out == "hello\n\nHINT"

    def test_empty_budget_hint_keeps_old_behavior(self):
        # Backward compatibility: the 4th arg defaults to "" and changes nothing.
        compose = self._compose()
        assert compose("hello", "", "") is None  # no injections → None (unchanged)
        out = compose("hello", "MEM", "PLUGIN")
        assert out is not None
        assert "HINT" not in out
        assert "MEM" in out and "PLUGIN" in out

    def test_non_string_content_returns_none_even_with_hint(self):
        compose = self._compose()
        # Multimodal content defeats injection; the hint must not force one.
        assert compose(["part1", "part2"], "", "", "HINT") is None
