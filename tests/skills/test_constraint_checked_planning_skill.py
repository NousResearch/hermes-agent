import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SKILL = REPO / "optional-skills/autonomous-ai-agents/constraint-checked-planning"
SPEC = importlib.util.spec_from_file_location("evaluate_plan", SKILL / "scripts/evaluate_plan.py")
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)


def _json(relative):
    import json

    return json.loads((SKILL / relative).read_text(encoding="utf-8"))


def test_feasible_example_passes_all_declared_constraints():
    result = EVALUATOR.evaluate(_json("examples/problem.json"), _json("examples/plan-feasible.json"))
    assert result["feasible"] is True
    assert result["errors"] == []
    assert result["summary"]["total_monthly_cost"] == 10000


def test_infeasible_example_reports_redundancy_violation():
    result = EVALUATOR.evaluate(_json("examples/problem.json"), _json("examples/plan-infeasible.json"))
    assert result["feasible"] is False
    assert any("distinct zones" in error for error in result["errors"])


def test_capacity_and_budget_violations_are_independent():
    problem = {
        "budget_limit": 5,
        "zone_capacity_kw": {"a": 4},
        "workloads": [{"id": "w", "required_kw": 2, "min_zones": 1}],
    }
    plan = {"placements": [{"workload_id": "w", "zone": "a", "kw": 5, "monthly_cost": 8}]}
    result = EVALUATOR.evaluate(problem, plan)
    assert result["feasible"] is False
    assert any("capacity exceeded" in error for error in result["errors"])
    assert any("budget exceeded" in error for error in result["errors"])


def test_rejects_boolean_as_numeric_capacity():
    problem = {
        "budget_limit": 100,
        "zone_capacity_kw": {"a": True},
        "workloads": [{"id": "w", "required_kw": 1, "min_zones": 1}],
    }
    result = EVALUATOR.evaluate(problem, {"placements": []})
    assert result["feasible"] is False
    assert any("zone_capacity_kw[a] must be a number" == error for error in result["errors"])


def test_rejects_unhashable_assignment_identifiers_without_crashing():
    problem = {
        "budget_limit": 100,
        "zone_capacity_kw": {"a": 5},
        "workloads": [{"id": "w", "required_kw": 1, "min_zones": 1}],
    }
    plan = {"placements": [{"workload_id": ["w"], "zone": {"a": 1}, "kw": 1, "monthly_cost": 1}]}
    result = EVALUATOR.evaluate(problem, plan)
    assert result["feasible"] is False
    assert any("workload_id must be a non-empty string" in error for error in result["errors"])
