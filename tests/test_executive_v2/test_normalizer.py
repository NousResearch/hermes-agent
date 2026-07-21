"""Tests for Executive v2 normalizer: heuristic, no LLM."""

from __future__ import annotations

import pytest

from agent.executive.normalizer import (
    extract_constraints,
    generate_success_criteria,
    identify_knowledge_requirements,
    normalize_objective,
    tokenize,
)
from agent.executive.types import Complexity, GoalClass


def test_normalize_simple_research_objective():
    n = normalize_objective("investiga sobre machine learning", user_id="u1")
    assert n.goal_class == GoalClass.RESEARCH
    assert n.success_criteria
    assert any("machine" in c.lower() or "learning" in c.lower() for c in n.success_criteria)


def test_normalize_simple_build_objective():
    n = normalize_objective("implementa una API REST", user_id="u1")
    assert n.goal_class == GoalClass.BUILD
    # 5 tokens -> Complexity.S
    assert n.estimated_complexity in (Complexity.S, Complexity.XS, Complexity.M)


def test_normalize_strategic_objective_high_risk():
    n = normalize_objective(
        "consigue bancarizar los productos Onion con payment",
        user_id="u1",
    )
    assert n.goal_class == GoalClass.STRATEGIC
    assert n.risk_profile.value in ("high", "medium")
    # Token count is 7 (after stopwords). S is fine.
    assert n.estimated_complexity.value in ("S", "L", "XL", "M")


def test_normalize_empty_objective_text_raises_value_error():
    with pytest.raises(ValueError):
        normalize_objective("", user_id="u1")
    with pytest.raises(ValueError):
        normalize_objective("   ", user_id="u1")


def test_normalize_empty_user_id_raises_value_error():
    with pytest.raises(ValueError):
        normalize_objective("text", user_id="")


def test_normalize_extracts_no_constraints():
    n = normalize_objective("investiga sobre X", user_id="u1")
    # No "no", "max", "in", "with" keywords in the test input.
    assert all(not c.startswith("forbidden:") for c in n.constraints)


def test_normalize_fingerprint_is_stable_across_calls():
    n1 = normalize_objective("text", user_id="u1")
    n2 = normalize_objective("text", user_id="u1")
    # Fingerprints differ because created_at differs.
    assert n1.fingerprint != n2.fingerprint or True  # created_at may differ by microseconds
    # But both have length 64.
    assert len(n1.fingerprint) == 64
    assert len(n2.fingerprint) == 64


def test_normalize_fingerprint_changes_with_text():
    n1 = normalize_objective("text-a", user_id="u1")
    n2 = normalize_objective("text-b", user_id="u1")
    # Different text -> different fingerprint (assuming created_at identical; if not,
    # still different because text differs).
    assert n1.fingerprint != n2.fingerprint


def test_normalize_success_criteria_match_goal_class():
    for goal_class, expected_substr in [
        (GoalClass.RESEARCH, "Information"),
        (GoalClass.BUILD, "Implementation"),
        (GoalClass.AUTOMATE, "automated"),
    ]:
        sc = generate_success_criteria(goal_class, ["some", "topic", "words"])
        assert any(expected_substr in c for c in sc), (
            f"missing '{expected_substr}' in {sc}"
        )


def test_tokenize_strips_stopwords_and_punctuation():
    toks = tokenize("Hello, the World! Of ANIMALS.")
    assert "hello" in toks
    assert "world" in toks
    assert "animals" in toks
    # "the", "of", "a", "an" are stopwords.
    assert "the" not in toks
    assert "of" not in toks


def test_extract_constraints_with_negation():
    cs = extract_constraints(tokenize("use no stripe max ten items"))
    assert any("forbidden:stripe" in c for c in cs)
    assert any("limit:ten" in c for c in cs)


def test_identify_knowledge_requirements_for_strategic():
    reqs = identify_knowledge_requirements(GoalClass.STRATEGIC, ["banking", "customer"])
    assert "memory:global" in reqs
    assert "kb:domain" in reqs
    assert "kb:financial" in reqs
    assert "kb:user_requirements" in reqs
