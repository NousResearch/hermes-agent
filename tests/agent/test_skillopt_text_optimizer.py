"""Behavior contracts for lightweight SkillOpt-style text gates.

These tests pin the safe first slice: bounded text edits on one skill document
and a pure validation gate. They do not call an LLM or mutate real skills.
"""

from __future__ import annotations

import math

import pytest

from agent.skillopt_text_optimizer import (
    AtomicEdit,
    GateDecision,
    apply_bounded_edits,
    evaluate_skill_gate,
)


def test_gate_accepts_strict_improvement_as_new_best():
    decision = evaluate_skill_gate(
        candidate_skill="new",
        candidate_score=0.72,
        current_skill="old",
        current_score=0.70,
        best_skill="old",
        best_score=0.71,
        step=3,
    )

    assert decision == GateDecision(
        action="accept_new_best",
        current_skill="new",
        current_score=0.72,
        best_skill="new",
        best_score=0.72,
        best_step=3,
    )


def test_gate_rejects_ties_to_prevent_unvalidated_drift():
    decision = evaluate_skill_gate(
        candidate_skill="changed",
        candidate_score=0.70,
        current_skill="stable",
        current_score=0.70,
        best_skill="best",
        best_score=0.75,
        step=4,
    )

    assert decision.action == "reject"
    assert decision.current_skill == "stable"
    assert decision.best_skill == "best"


def test_gate_accept_preserves_prior_best_step_when_not_new_best():
    decision = evaluate_skill_gate(
        candidate_skill="better-current",
        candidate_score=0.73,
        current_skill="old-current",
        current_score=0.70,
        best_skill="historic-best",
        best_score=0.80,
        best_step=9,
        step=12,
    )

    assert decision.action == "accept"
    assert decision.current_skill == "better-current"
    assert decision.best_skill == "historic-best"
    assert decision.best_score == 0.80
    assert decision.best_step == 9


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("candidate_score", math.nan),
        ("current_score", math.inf),
        ("best_score", -math.inf),
        ("epsilon", math.nan),
    ],
)
def test_gate_rejects_non_finite_scores(field, value):
    kwargs = dict(
        candidate_skill="bad",
        candidate_score=0.3,
        current_skill="old",
        current_score=0.1,
        best_skill="best",
        best_score=0.2,
        step=1,
    )
    kwargs[field] = value

    with pytest.raises(ValueError, match="finite"):
        evaluate_skill_gate(**kwargs)


def test_apply_bounded_edits_respects_learning_rate_budget():
    skill = "# Skill\n\nUse search.\n"
    candidate, applied = apply_bounded_edits(
        skill,
        [
            AtomicEdit(op="append", content="Always verify with tests."),
            AtomicEdit(op="append", content="Prefer small patches."),
        ],
        edit_budget=1,
    )

    assert applied == 1
    assert "Always verify with tests." in candidate
    assert "Prefer small patches." not in candidate


def test_apply_edits_supports_replace_insert_after_and_delete():
    skill = "# Skill\n\nUse grep.\nRun tests.\n"
    candidate, applied = apply_bounded_edits(
        skill,
        [
            AtomicEdit(op="replace", target="Use grep.", content="Use `search_files`."),
            AtomicEdit(op="insert_after", target="Use `search_files`.", content="Use `read_file` for exact lines."),
            AtomicEdit(op="delete", target="Run tests."),
        ],
        edit_budget=3,
    )

    assert applied == 3
    assert "Use `search_files`." in candidate
    assert "Use `read_file` for exact lines." in candidate
    assert "Run tests." not in candidate


def test_apply_edits_rejects_missing_replace_target_without_partial_mutation():
    skill = "# Skill\n\nStable text.\n"

    with pytest.raises(ValueError, match="target not found"):
        apply_bounded_edits(
            skill,
            [AtomicEdit(op="replace", target="Missing", content="New")],
            edit_budget=1,
        )


def test_apply_edits_protects_slow_update_block_from_step_edits():
    skill = "# Skill\n\n<!-- SLOW_UPDATE_START -->\ndo not touch\n<!-- SLOW_UPDATE_END -->\n"

    with pytest.raises(ValueError, match="protected slow-update"):
        apply_bounded_edits(
            skill,
            [AtomicEdit(op="replace", target="do not touch", content="changed")],
            edit_budget=1,
        )


def test_apply_edits_fail_closed_on_unterminated_slow_update_append():
    skill = "# Skill\n\n<!-- SLOW_UPDATE_START -->\nunterminated\n"

    with pytest.raises(ValueError, match="unterminated slow-update"):
        apply_bounded_edits(
            skill,
            [AtomicEdit(op="append", content="Should not append inside protected tail.")],
            edit_budget=1,
        )


def test_apply_edits_rejects_overlarge_candidate():
    with pytest.raises(ValueError, match="candidate skill exceeds"):
        apply_bounded_edits(
            "# Skill\n",
            [AtomicEdit(op="append", content="word " * 20)],
            edit_budget=1,
            max_words=5,
        )
