"""Tests for the stable-tier system-prompt budget allocator.

``_apply_stable_budget`` enforces ``agent.max_system_prompt_tokens`` by
dropping whole lower-priority sections from the stable tier until the
joined estimate fits the budget.  It is deterministic (computed once per
session at build time) so it never invalidates the upstream prefix cache.

Drop priority convention: higher number = dropped first; ``0`` = never
dropped (identity-critical).  Budget ``0`` = unlimited (no truncation).
"""

from __future__ import annotations

import logging

import pytest

from agent.model_metadata import estimate_tokens_rough
from agent.system_prompt import _apply_stable_budget


def _join(*texts: str) -> str:
    return "\n\n".join(t.strip() for t in texts if t and t.strip())


class TestBudgetDisabled:
    def test_budget_zero_keeps_everything_in_order(self):
        sections = [(0, "alpha"), (5, "beta"), (4, "gamma")]
        assert _apply_stable_budget(sections, 0) == _join("alpha", "beta", "gamma")

    def test_negative_budget_treated_as_unlimited(self):
        sections = [(0, "alpha"), (5, "beta")]
        assert _apply_stable_budget(sections, -1) == _join("alpha", "beta")

    def test_empty_and_whitespace_sections_excluded(self):
        sections = [(0, "alpha"), (5, "   "), (4, ""), (3, "gamma")]
        assert _apply_stable_budget(sections, 0) == _join("alpha", "gamma")


class TestUnderBudget:
    def test_keeps_everything_when_already_under_budget(self):
        sections = [(0, "a" * 40), (5, "b" * 40)]
        # ~20 tokens total; budget far above
        result = _apply_stable_budget(sections, 10_000)
        assert ("a" * 40) in result
        assert ("b" * 40) in result


class TestTruncation:
    def test_drops_highest_priority_section_first(self):
        protected = "P" * 40   # ~10 tok
        skills = "S" * 400      # ~100 tok, drop_priority 5
        env = "E" * 200         # ~50 tok, drop_priority 4
        sections = [(0, protected), (5, skills), (4, env)]

        # Full ~161 tok; dropping only skills drops to ~61 tok.
        result = _apply_stable_budget(sections, 100)

        assert skills not in result          # highest priority dropped
        assert protected in result
        assert env in result                 # lower priority survived
        assert estimate_tokens_rough(result) <= 100

    def test_drops_in_priority_order_until_under_budget(self):
        protected = "P" * 40    # ~10 tok
        skills = "S" * 400      # drop_priority 5
        env = "E" * 200         # drop_priority 4
        sections = [(0, protected), (5, skills), (4, env)]

        # Budget 20 forces dropping both skills (5) then env (4).
        result = _apply_stable_budget(sections, 20)

        assert skills not in result
        assert env not in result
        assert protected in result
        assert estimate_tokens_rough(result) <= 20

    def test_survivors_keep_original_order(self):
        sections = [(0, "first"), (5, "B" * 400), (0, "third")]
        result = _apply_stable_budget(sections, 5)
        assert result == _join("first", "third")


class TestProtectedSections:
    def test_protected_sections_never_dropped_even_over_budget(self, caplog):
        protected = "P" * 400  # ~100 tok, drop_priority 0
        sections = [(0, protected)]

        with caplog.at_level(logging.WARNING, logger="agent.system_prompt"):
            result = _apply_stable_budget(sections, 10)

        assert result == protected
        assert any(r.levelno >= logging.WARNING for r in caplog.records)

    def test_no_warning_when_within_budget(self, caplog):
        sections = [(0, "P" * 40)]
        with caplog.at_level(logging.WARNING, logger="agent.system_prompt"):
            _apply_stable_budget(sections, 10_000)
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


class TestDeterminism:
    def test_identical_inputs_produce_identical_output(self):
        sections = [(0, "P" * 40), (5, "S" * 400), (4, "E" * 200), (2, "X" * 100)]
        a = _apply_stable_budget(sections, 70)
        b = _apply_stable_budget(sections, 70)
        assert a == b
