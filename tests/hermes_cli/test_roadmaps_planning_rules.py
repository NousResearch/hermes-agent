"""Tests for the versioned Roadmaps planning rules (T5c, Vision session).

The rules module is the single source of truth for the system prompt the
Vision session agent receives: objective, constraints, strict JSON output
schema, quality rules, and behavior rules. Rules are versioned: changing
them produces a new version, never a silent edit.
"""

from __future__ import annotations

import json

import pytest

from hermes_cli.roadmaps_planning_rules import (
    PLANNING_RULES_VERSION,
    PlanningRulesVersionError,
    get_planning_rules,
)


def test_planning_rules_version_constant_is_current():
    assert isinstance(PLANNING_RULES_VERSION, str)
    assert PLANNING_RULES_VERSION == "1.0"


def test_get_planning_rules_default_returns_current_version():
    rules = get_planning_rules()
    assert rules["version"] == PLANNING_RULES_VERSION
    assert rules["version"] == "1.0"


def test_get_planning_rules_explicit_current_version():
    rules = get_planning_rules("1.0")
    assert rules["version"] == "1.0"


def test_get_planning_rules_blank_version_falls_back_to_current():
    assert get_planning_rules("")["version"] == PLANNING_RULES_VERSION
    assert get_planning_rules("   ")["version"] == PLANNING_RULES_VERSION


@pytest.mark.parametrize("bad", ["0.9", "2.0", "v1", "1.0.1", "garbage", 1])
def test_get_planning_rules_unknown_version_raises_clean_error(bad):
    with pytest.raises(PlanningRulesVersionError) as exc:
        get_planning_rules(bad)
    message = str(exc.value)
    assert "version" in message
    assert repr(bad) in message


def test_get_planning_rules_none_means_current_version():
    # None (the API default) selects the current version — it is not an error.
    assert get_planning_rules(None)["version"] == PLANNING_RULES_VERSION


def test_planning_rules_prompt_is_a_non_empty_string():
    rules = get_planning_rules()
    prompt = rules["prompt"]
    assert isinstance(prompt, str)
    assert len(prompt) > 500  # a real system prompt, not a stub


def test_planning_rules_prompt_declares_objective():
    prompt = get_planning_rules()["prompt"]
    assert "objectif" in prompt.lower() or "objective" in prompt.lower()
    # The objective is to help produce an impeccable, solid plan — the best path.
    assert any(word in prompt.lower() for word in ("impeccable", "meilleure voie", "best path", "solide"))


def test_planning_rules_prompt_declares_constraints():
    prompt = get_planning_rules()["prompt"]
    lowered = prompt.lower()
    assert "contrainte" in lowered or "constraint" in lowered
    # Constraints cover scope: roadmap, project, context.
    assert any(word in lowered for word in ("scope", "roadmap", "projet", "project"))
    assert any(word in lowered for word in ("contexte", "context"))


def test_planning_rules_prompt_declares_strict_json_output_structure():
    prompt = get_planning_rules()["prompt"]
    lowered = prompt.lower()
    assert "json" in lowered
    assert "```json" in prompt or "``` json" in prompt
    # The strict schema keys are all present.
    for key in ("nodes", "relations", "todos", "node_id", "kind", "title",
                "relation_id", "from_node_id", "to_node_id", "todo_id"):
        assert key in prompt
    assert "objective" in prompt
    assert "phase" in prompt
    assert "milestone" in prompt
    assert "step" in prompt
    assert "decision" in prompt


def test_planning_rules_prompt_declares_quality_rules():
    prompt = get_planning_rules()["prompt"]
    lowered = prompt.lower()
    assert "qualité" in lowered or "quality" in lowered
    # Quality rules: coherent steps, explicit dependencies, no duplicates,
    # measurable milestones, controlled vocabulary, plan transitions.
    assert any(word in lowered for word in ("dépendance", "dependenc", "depends_on"))
    assert any(word in lowered for word in ("doublon", "duplicat"))
    assert any(word in lowered for word in ("mesurable", "measurable", "jalon"))
    assert any(word in lowered for word in ("vocabulaire", "vocabulary", "milestone/epic/task"))
    assert any(word in lowered for word in ("transition", "proposé", "validé", "activé"))


def test_planning_rules_prompt_declares_behavior_rules():
    prompt = get_planning_rules()["prompt"]
    lowered = prompt.lower()
    assert "comportement" in lowered or "behavior" in lowered or "behaviour" in lowered
    # Behavior rules: never invent facts, ask for clarification when ambiguous,
    # propose alternatives, refactor/redo the plan when needed.
    assert "jamais" in lowered and any(word in lowered for word in ("inventer", "invente", "invent"))
    assert any(word in lowered for word in ("clarification", "ambigu", "préciser", "demander"))
    assert any(word in lowered for word in ("alternative", "option"))
    assert any(word in lowered for word in ("refondre", "rework", "reformuler", "récrire", "réécrire"))


def test_planning_rules_are_json_serializable():
    # The RPC returns the rules payload; it must survive json.dumps untouched
    # (no secrets, no non-serializable objects).
    rules = get_planning_rules()
    round_tripped = json.loads(json.dumps(rules))
    assert round_tripped == rules
    assert round_tripped["version"] == PLANNING_RULES_VERSION


def test_planning_rules_are_immutable_per_version():
    # Same version → identical prompt bytes (a version is a frozen snapshot).
    assert get_planning_rules("1.0")["prompt"] == get_planning_rules()["prompt"]
