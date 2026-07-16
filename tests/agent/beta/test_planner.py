from agent.beta.planner import ExecutionPlan, PlanStep, build_plan, replan
from agent.beta.risk import RiskLevel
from agent.beta.router import RoutingDecision
from agent.beta.specialists import Specialist, SpecialistRegistry


def registry():
    return SpecialistRegistry((
        Specialist(id="dba", name="DBA", description="db", capabilities=("database",), keywords=("postgres",)),
        Specialist(id="dba-alt", name="DBA Alt", description="db2", capabilities=("database",), keywords=("sql",)),
        Specialist(id="qa-auditor", name="QA", description="qa", capabilities=("quality-assurance",), keywords=("review",)),
    ))


def test_plan_has_dependency_order():
    decision = RoutingDecision(
        intent="diagnosis", delegation_needed=True, specialists=("dba",),
        initial_risk="low", parallelizable=False, rationale="database request",
        confidence=1.0,
    )
    plan = build_plan("PostgreSQL is slow", decision, registry())
    assert [step.id for step in plan.ordered_steps()] == ["investigate-1", "consolidate"]
    assert plan.steps[-1].dependencies == ("investigate-1",)


def test_cycle_is_rejected():
    plan = ExecutionPlan(id="p", request="x", steps=(
        PlanStep(id="a", objective="a", capability="x", specialist_id="dba", dependencies=("b",)),
        PlanStep(id="b", objective="b", capability="x", specialist_id="dba", dependencies=("a",)),
    ))
    try:
        plan.ordered_steps()
    except ValueError as exc:
        assert "cycle" in str(exc)
    else:
        raise AssertionError("cycle should fail")


def test_replan_uses_alternate_specialist():
    plan = ExecutionPlan(id="p", request="x", steps=(
        PlanStep(id="a", objective="a", capability="database", specialist_id="dba", risk=RiskLevel.LOW),
    ))
    updated = replan(plan, "a", registry())
    assert updated.steps[0].specialist_id == "dba-alt"
    assert updated.revision == 2
