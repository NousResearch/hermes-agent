"""Canary tests for Executive Runtime Objective Completeness Analyzer.

The analyzer is intentionally limited to readonly/pure objective analysis:
no Strategy Builder, no Execution Contract, no Kanban, no Goal Runner,
no workers, no NotebookLM, no GBrain/Obsidian writes, no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "agent" / "executive" / "schemas" / "analysis_schema.json"
MODULE_PATH = REPO_ROOT / "agent" / "executive" / "objective_completeness_analyzer.py"


def test_analyzer_accepts_explicit_canary_objective_and_emits_minimum_fields():
    from agent.executive.objective_completeness_analyzer import analyze_objective

    result = analyze_objective(
        """
        Implementar el primer canary funcional del Executive Runtime Decision Engine:
        Objective Completeness Analyzer. Únicamente analizar objetivos humanos y producir
        analysis.json. No Strategy Builder. No Execution Contract. No Kanban. No Goal.
        Entregables: objective_completeness_analyzer.py, analysis_schema.json,
        canary_validation.md, rollback.md, manifest.json, verified_hashes.txt.
        Validaciones: compile PASS, tests específicos PASS, JSON schema válido,
        hashes PASS, rollback PASS, cero side effects fuera del scope.
        """,
        user_id="canary-user",
    )
    data = result.to_dict()

    assert set(data) == {
        "objective_fingerprint",
        "normalized_objective",
        "objective_type",
        "confidence",
        "ambiguity_score",
        "known_information",
        "missing_information",
        "contradictions",
        "recommended_questions",
        "ready_for_strategy",
    }
    assert len(data["objective_fingerprint"]) == 64
    assert data["objective_type"] == "BUILD"
    assert 0.0 <= data["confidence"] <= 1.0
    assert 0.0 <= data["ambiguity_score"] <= 1.0
    assert data["ready_for_strategy"] is True
    assert data["contradictions"] == []
    assert "analysis.json" in data["known_information"]["deliverables"]
    assert "no_kanban" in data["known_information"]["forbidden_actions"]


def test_analyzer_marks_vague_objective_as_not_ready_and_recommends_questions():
    from agent.executive.objective_completeness_analyzer import analyze_objective

    data = analyze_objective("make Hermes smarter about objectives", user_id="canary-user").to_dict()

    assert data["ready_for_strategy"] is False
    assert data["ambiguity_score"] >= 0.5
    missing_kinds = {item["kind"] for item in data["missing_information"]}
    assert {"missing_target", "missing_success_criteria", "missing_scope_boundary"} <= missing_kinds
    assert len(data["recommended_questions"]) >= 3


def test_analyzer_detects_readonly_mutation_contradiction_without_planning():
    from agent.executive.objective_completeness_analyzer import analyze_objective

    data = analyze_objective(
        "Readonly only: implement the strategy builder, create Kanban tasks, run workers, and push changes",
        user_id="canary-user",
    ).to_dict()

    assert data["ready_for_strategy"] is False
    assert data["contradictions"], "expected readonly/mutation contradiction"
    assert any("readonly" in c.lower() for c in data["contradictions"])
    assert all("strategy" not in str(item).lower() or "builder" in str(item).lower() for item in data["known_information"].values())


def test_write_analysis_json_writes_only_requested_file(tmp_path):
    from agent.executive.objective_completeness_analyzer import analyze_objective, write_analysis_json

    output_path = tmp_path / "analysis.json"
    before = set(tmp_path.iterdir())
    result = analyze_objective(
        "Analyze a human objective for completeness and write analysis.json only",
        user_id="canary-user",
    )

    written = write_analysis_json(result, output_path)

    after = set(tmp_path.iterdir())
    assert written == output_path
    assert after - before == {output_path}
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded == result.to_dict()


def test_analysis_schema_validates_analyzer_output():
    from agent.executive.objective_completeness_analyzer import analyze_objective

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    data = analyze_objective(
        "Verify Objective Completeness Analyzer can emit analysis.json with tests and rollback docs",
        user_id="canary-user",
    ).to_dict()

    try:
        import jsonschema
    except ImportError:  # pragma: no cover - validation still checks parser and required fields
        pytest.skip("jsonschema is not installed")

    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(data, schema)


def test_analyzer_has_no_prohibited_runtime_side_effect_references():
    content = MODULE_PATH.read_text(encoding="utf-8")
    prohibited = [
        "NotebookLM",
        "gbrain",
        "obsidian",
        "kanban_apply",
        "GoalManager",
        "worker_dispatch",
        "subprocess",
        "requests",
        "httpx",
        "urllib",
        "openai",
        "anthropic",
        "policy_persist",
        "plan_apply",
        "bridge_apply",
    ]
    found = [term for term in prohibited if term in content]
    assert not found, f"prohibited runtime reference(s): {found}"
