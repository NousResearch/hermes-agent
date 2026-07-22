"""Tests for Executive v2 classifier: goal_class, risk_profile, complexity."""

from __future__ import annotations

from agent.executive.classifier import (
    classify_goal_class,
    classify_objective,
    compute_risk_profile,
    estimate_complexity,
)
from agent.executive.types import Complexity, GoalClass, RiskProfile


def test_classify_research_keyword_matches():
    toks = ["investiga", "machine", "learning"]
    assert classify_goal_class(toks) == GoalClass.RESEARCH


def test_classify_build_keyword_matches():
    toks = ["implementa", "una", "api", "rest"]
    assert classify_goal_class(toks) == GoalClass.BUILD


def test_classify_strategic_tie_breaker_wins_over_build():
    """If STRATEGIC and BUILD both have signals, STRATEGIC wins."""
    toks = ["consigue", "bancarizar", "productos", "onion"]
    # "consigue" is in STRATEGIC; "bancarizar" is NOT in any class
    # (well, "banking" is HIGH_RISK but not in goal_class keywords).
    # So STRATEGIC has 1, others have 0.
    assert classify_goal_class(toks) == GoalClass.STRATEGIC


def test_classify_strategic_with_high_risk_tokens_returns_high():
    classified = classify_objective(["consigue", "payment", "production"])
    assert classified.goal_class == GoalClass.STRATEGIC
    assert classified.risk_profile == RiskProfile.HIGH


def test_classify_low_risk_default_for_neutral_tokens():
    classified = classify_objective(["investiga", "X"])
    # No HIGH or MEDIUM tokens.
    assert classified.risk_profile == RiskProfile.LOW


def test_estimate_complexity_xs_through_xl_with_token_counts():
    assert estimate_complexity([]) == Complexity.XS
    assert estimate_complexity(["a", "b", "c"]) == Complexity.XS
    assert estimate_complexity(["a"] * 4) == Complexity.S
    assert estimate_complexity(["a"] * 10) == Complexity.S
    assert estimate_complexity(["a"] * 11) == Complexity.M
    assert estimate_complexity(["a"] * 30) == Complexity.M
    assert estimate_complexity(["a"] * 31) == Complexity.L
    assert estimate_complexity(["a"] * 100) == Complexity.L
    assert estimate_complexity(["a"] * 101) == Complexity.XL
    assert estimate_complexity(["a"] * 1000) == Complexity.XL


def test_compute_risk_profile_strategic_no_risk_is_medium():
    """STRATEGIC alone (no risk tokens) is MEDIUM by default."""
    assert compute_risk_profile(GoalClass.STRATEGIC, ["consigue", "x"]) == RiskProfile.MEDIUM


def test_compute_risk_profile_high_risk_tokens_anywhere():
    assert compute_risk_profile(GoalClass.BUILD, ["delete", "production"]) == RiskProfile.HIGH
    assert compute_risk_profile(GoalClass.OTHER, ["pii", "personal"]) == RiskProfile.HIGH


def test_compute_risk_profile_medium_risk_tokens():
    assert compute_risk_profile(GoalClass.BUILD, ["test", "staging"]) == RiskProfile.MEDIUM
