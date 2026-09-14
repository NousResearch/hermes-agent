"""The supervisor: declaring objectives, routing them, and ordering their work.

Everything here is pure — no runtime, no board. The routing rule is the governance control
this phase exists for, so it is tested as a rule rather than through a submission: a
delegation that must be refused has to be refused for every plan shape, not only the one
the example bundle happens to use.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from nova.errors import SpecError
from nova.spec import AgentSpec, load_bundle
from nova.spec.objective import ObjectiveSpec, load_objectives
from nova.supervisor import plan_order, route_objective
from nova.supervisor.route import (
    DISABLED_ASSIGNEE,
    NOT_DELEGABLE,
    UNKNOWN_ASSIGNEE,
    UNKNOWN_OWNER,
    RoutingError,
)
from nova.supervisor.submit import build_work_items, work_key

from .conftest import EXAMPLE_BUNDLE


def agent(agent_id: str, *, may_assign_to=(), enabled: bool = True) -> AgentSpec:
    return AgentSpec(
        id=agent_id,
        name=agent_id.title(),
        role="worker",
        enabled=enabled,
        delegation=__import__(
            "nova.spec.agent", fromlist=["DelegationSpec"]
        ).DelegationSpec(may_assign_to=tuple(may_assign_to)),
    )


def objective(**overrides) -> ObjectiveSpec:
    data = {
        "id": "obj",
        "title": "An objective",
        "owner": "lead",
        "steps": [
            {"id": "first", "title": "First", "assignee": "lead"},
            {"id": "second", "title": "Second", "assignee": "helper", "depends_on": ["first"]},
        ],
    }
    data.update(overrides)
    return ObjectiveSpec.parse(data)


# -- declaration ------------------------------------------------------------


def test_an_objective_with_no_steps_is_refused():
    """An objective that declares no work is a configuration mistake, not an empty plan."""
    with pytest.raises(SpecError, match="declares no steps"):
        ObjectiveSpec.parse({"id": "o", "title": "T", "owner": "a", "steps": []})


def test_a_dependency_on_an_unknown_step_is_refused():
    with pytest.raises(SpecError, match="depends on unknown step"):
        objective(steps=[{"id": "a", "title": "A", "assignee": "lead", "depends_on": ["ghost"]}])


def test_a_dependency_cycle_is_named_at_load():
    """The runtime would accept it: each task would simply wait forever, which looks like a
    stuck worker rather than a malformed plan."""
    with pytest.raises(SpecError, match="cycle"):
        objective(
            steps=[
                {"id": "a", "title": "A", "assignee": "lead", "depends_on": ["c"]},
                {"id": "b", "title": "B", "assignee": "lead", "depends_on": ["a"]},
                {"id": "c", "title": "C", "assignee": "lead", "depends_on": ["b"]},
            ]
        )


def test_a_step_depending_on_itself_is_refused():
    with pytest.raises(SpecError, match="itself"):
        objective(steps=[{"id": "a", "title": "A", "assignee": "lead", "depends_on": ["a"]}])


def test_duplicate_step_ids_are_refused():
    with pytest.raises(SpecError, match="duplicate step id"):
        objective(
            steps=[
                {"id": "a", "title": "A", "assignee": "lead"},
                {"id": "a", "title": "Also A", "assignee": "lead"},
            ]
        )


def test_an_unknown_key_is_refused():
    """A typo must not silently produce a step with different behaviour."""
    with pytest.raises(SpecError, match="priorty"):
        objective(
            steps=[{"id": "a", "title": "A", "assignee": "lead", "priorty": 3}]
        )


def test_a_missing_assignee_is_refused():
    """There is no default assignee here, deliberately: the runtime's decomposer has one,
    and routing work to a fallback is precisely the ungoverned behaviour this replaces."""
    with pytest.raises(SpecError, match="assignee"):
        objective(steps=[{"id": "a", "title": "A"}])


def test_an_absent_objectives_directory_is_a_valid_state(tmp_path):
    assert load_objectives(tmp_path) == ()


def test_an_objective_naming_an_unknown_agent_fails_bundle_load(tmp_path):
    import shutil

    root = tmp_path / "b"
    shutil.copytree(EXAMPLE_BUNDLE, root)
    path = root / "objectives" / "quarterly-refund-audit.yaml"
    path.write_text(
        path.read_text(encoding="utf-8").replace("owner: operations", "owner: nobody"),
        encoding="utf-8",
    )
    with pytest.raises(SpecError, match="unknown agent"):
        load_bundle(root)


# -- routing: the control this phase exists for -----------------------------


def test_a_delegation_the_owner_declared_is_permitted():
    agents = [agent("lead", may_assign_to=["helper"]), agent("helper")]
    decision = route_objective(objective(), agents)
    assert decision.allowed
    assert decision.refusals == ()


def test_a_delegation_the_owner_never_declared_is_refused():
    """The whole point. Before this, `may_assign_to` was compiled into the profile under a
    key the runtime ignores and enforced nowhere."""
    agents = [agent("lead"), agent("helper")]
    decision = route_objective(objective(), agents)
    assert not decision.allowed
    assert [step.reason for step in decision.refusals] == [NOT_DELEGABLE]
    # The message has to say what to change, or an operator is left guessing.
    assert "may_assign_to" in decision.refusals[0].detail


def test_an_owner_may_always_assign_to_itself():
    """Self-assignment is not delegation, so it needs no grant."""
    agents = [agent("lead")]
    decision = route_objective(
        objective(steps=[{"id": "only", "title": "T", "assignee": "lead"}]), agents
    )
    assert decision.allowed


def test_an_unknown_assignee_is_refused_and_named():
    agents = [agent("lead", may_assign_to=["helper"])]
    decision = route_objective(objective(), agents)
    assert [step.reason for step in decision.refusals] == [UNKNOWN_ASSIGNEE]


def test_a_disabled_assignee_is_refused():
    """A disabled agent has no profile, so the step would sit on the board forever."""
    agents = [agent("lead", may_assign_to=["helper"]), agent("helper", enabled=False)]
    decision = route_objective(objective(), agents)
    assert [step.reason for step in decision.refusals] == [DISABLED_ASSIGNEE]


def test_an_unknown_owner_refuses_every_step_not_just_the_first():
    """One message per step, because an operator fixing this needs the whole picture."""
    agents = [agent("helper")]
    decision = route_objective(objective(), agents)
    assert len(decision.refusals) == 2
    assert {step.reason for step in decision.refusals} == {UNKNOWN_OWNER}


def test_routing_never_raises_on_a_bad_plan():
    """Refusal is a normal outcome with detail an operator needs, not an exception that
    reduces three bad delegations to a sentence about the first."""
    decision = route_objective(objective(), [])
    assert decision.allowed is False
    assert decision.explain()


def test_a_runtime_decomposed_step_is_flagged_as_ungoverned():
    """NOVA cannot constrain which profiles the runtime's own decomposer picks, so it says
    so rather than implying the children are governed."""
    agents = [agent("lead", may_assign_to=["helper"]), agent("helper")]
    spec = objective(
        steps=[{"id": "open", "title": "Investigate", "assignee": "lead", "decompose": True}]
    )
    decision = route_objective(spec, agents)
    assert decision.allowed
    assert decision.ungoverned_steps == ("open",)
    assert any("not governed" in warning for warning in decision.warnings)


def test_a_disabled_owner_refuses_its_own_steps_and_says_why():
    """A disabled agent has no profile, so work assigned to it can never run — including
    work it assigned to itself. The warning explains the objective-level consequence; the
    refusal is what stops the unrunnable step reaching the board."""
    agents = [agent("lead", may_assign_to=["helper"], enabled=False), agent("helper")]
    decision = route_objective(objective(), agents)
    assert not decision.allowed
    assert [step.reason for step in decision.refusals] == [DISABLED_ASSIGNEE]
    assert any("disabled" in warning for warning in decision.warnings)


def test_a_disabled_owner_that_assigns_nothing_to_itself_still_warns():
    """Nothing is refused — every step can run — but the objective's own completion has
    no one to judge it, which a reader needs told."""
    agents = [agent("lead", may_assign_to=["helper"], enabled=False), agent("helper")]
    spec = objective(steps=[{"id": "only", "title": "T", "assignee": "helper"}])
    decision = route_objective(spec, agents)
    assert decision.allowed
    assert any("disabled" in warning for warning in decision.warnings)


# -- ordering ---------------------------------------------------------------


def test_dependencies_always_precede_their_dependants():
    """Submission needs this: the runtime resolves parents by id, so a parent must exist."""
    spec = objective(
        steps=[
            {"id": "last", "title": "C", "assignee": "lead", "depends_on": ["mid"]},
            {"id": "mid", "title": "B", "assignee": "lead", "depends_on": ["first"]},
            {"id": "first", "title": "A", "assignee": "lead"},
        ]
    )
    assert [step.id for step in plan_order(spec)] == ["first", "mid", "last"]


def test_independent_steps_keep_declaration_order():
    """So the board reads the way the file reads."""
    spec = objective(
        steps=[
            {"id": "b", "title": "B", "assignee": "lead"},
            {"id": "a", "title": "A", "assignee": "lead"},
        ]
    )
    assert [step.id for step in plan_order(spec)] == ["b", "a"]


def test_ordering_an_unreachable_plan_says_so_rather_than_truncating():
    """Loading refuses cycles, but a plan built in code can still reach here, and a silent
    partial order would submit half an objective."""
    built = ObjectiveSpec(
        id="o",
        title="T",
        owner="lead",
        steps=(
            __import__("nova.spec.objective", fromlist=["PlanStep"]).PlanStep(
                id="a", title="A", assignee="lead", depends_on=("b",)
            ),
            __import__("nova.spec.objective", fromlist=["PlanStep"]).PlanStep(
                id="b", title="B", assignee="lead", depends_on=("a",)
            ),
        ),
    )
    with pytest.raises(RoutingError, match="can never start"):
        plan_order(built)


# -- work items -------------------------------------------------------------


def test_work_keys_are_stable_and_namespaced_by_objective():
    """Two objectives with a 'review' step must not share an idempotency key."""
    assert work_key("close", "review") == "close:review"
    assert work_key("audit", "review") != work_key("close", "review")


def test_dependencies_are_carried_as_keys_not_runtime_ids():
    items = build_work_items(objective())
    second = next(item for item in items if item.key.endswith(":second"))
    assert second.depends_on == ("obj:first",)


def test_every_step_body_restates_the_objective_and_its_acceptance():
    """A worker starts fresh and sees only this. An agent that completes its task correctly
    and misses the point of it is the expensive kind of wrong."""
    spec = objective(acceptance="the ledger balances", description="Close the month")
    body = build_work_items(spec)[0].body
    assert "An objective" in body
    assert "the ledger balances" in body
    assert "Close the month" in body


def test_the_tenant_is_stamped_on_every_item():
    items = build_work_items(objective(), tenant_id="acme")
    assert all(item.tenant_id == "acme" for item in items)


# -- the example bundle -----------------------------------------------------


def test_the_example_objective_routes_cleanly():
    bundle = load_bundle(EXAMPLE_BUNDLE)
    assert bundle.objectives
    decision = route_objective(bundle.objectives[0], bundle.agents)
    assert decision.allowed, decision.explain()


def test_revoking_the_example_delegation_refuses_the_example_objective():
    """Proves the example is actually governed by the declaration, not passing by accident."""
    bundle = load_bundle(EXAMPLE_BUNDLE)
    stripped = tuple(
        replace(spec, delegation=replace(spec.delegation, may_assign_to=()))
        if spec.id == "operations"
        else spec
        for spec in bundle.agents
    )
    decision = route_objective(bundle.objectives[0], stripped)
    assert not decision.allowed
    assert all(step.reason == NOT_DELEGABLE for step in decision.refusals)
