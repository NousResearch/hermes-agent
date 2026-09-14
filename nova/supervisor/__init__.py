"""NOVA Supervisor — business objectives, routed under the tenant's own delegation policy.

The runtime already decomposes, routes, fans out and collects: ``kanban_decompose`` asks an
LLM to break a triage card into a task graph and pick an assignee for each child, and the
dispatcher runs them in parallel. None of that is reimplemented here, and reimplementing it
would have been the mistake this phase exists to avoid.

What the runtime does **not** do is govern the routing. Its decomposer picks any profile on
the host, validated only against "is that a real profile"
(``hermes_cli/kanban_decompose.py::_normalize_assignee_choice``). NOVA agents declare
``delegation.may_assign_to``; until this phase that declaration was carried into the
profile under a key the runtime ignores and enforced **nowhere** — a governance control
that looked present in a review and did nothing.

So the supervisor is the missing half, not a second scheduler. The *declaration* of an
objective lives in :mod:`nova.spec.objective`, with the other specs — it is a declaration
like an agent or an identity, and keeping it there is also what stops the dependency arrow
from looping back through the runtime contract. What lives here is the behaviour:

``route``      the decision, pure: may this step go to this agent, under this policy?
``submit``     plan -> work items on the runtime's own board, idempotent and audited.
``report``     the objective's state, collected back from the runtime.

**Routing is checked before anything is written.** That is what makes it a real control
rather than a report: NOVA is the writer, so a plan that violates the tenant's delegation
policy produces no tasks at all, rather than tasks that a reviewer discovers afterwards.

The honest limit of that, stated here because it belongs next to the claim: NOVA governs
the work **NOVA submits**. A human running ``hermes kanban create --assignee X``, or the
runtime's own decomposer choosing an assignee, is outside this boundary — see
``docs/platform/PHASE_5.md`` for what the control plane shows in those cases.
"""

from nova.spec.objective import ObjectiveSpec, PlanStep, load_objectives
from nova.supervisor.report import ObjectiveReport, StepReport, collect
from nova.supervisor.route import (
    RoutingDecision,
    RoutingError,
    plan_order,
    route_objective,
)
from nova.supervisor.submit import SubmitReport, submit_objective

__all__ = [
    "ObjectiveReport",
    "ObjectiveSpec",
    "PlanStep",
    "RoutingDecision",
    "RoutingError",
    "StepReport",
    "SubmitReport",
    "collect",
    "load_objectives",
    "plan_order",
    "route_objective",
    "submit_objective",
]
