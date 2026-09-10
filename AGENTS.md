# Project Agent Rules

Before project mutation, enter the registered project and selected feature/task through `project_enter`.

For a new patch/minor/major version: product `spec.md` -> Ponytail architecture `plan.md` -> Ponytail complete `tasks.md` -> exact-task implementation -> release and product readback.

Execution discipline (Feature 006):

1. Binding map: after the three canonical artifacts are ACCEPTED, execute only their tasks in dependency order; work outside the map is a process violation except read-only diagnosis that informs an amendment.
2. Scope: only the user adds functionality or expands scope. Reviewers, agents, and subagents may not add requirements, components, tests, or verification beyond accepted artifacts. A design deficiency routes as: evidence + impact + minimal amendment -> user decision -> amendment -> resume. In a conflict, the accepted artifact wins unless the finding is a Defect or a Material risk with a concrete mechanism.
3. Reviewer mandate: every review names the artifacts, acceptance criteria, and classification - Defect (blocks), Material risk with mechanism (blocks), Hardening (does not block), Ceremony (does not block). Open-ended "find everything" mandates are forbidden.
4. Tests: every test cites one described behavior or one cited material risk. No exhaustive platform-internal enumeration; one consolidated scan proves removed machinery is gone; regression tests only for defects that occurred or user-accepted risks.
5. Proof: default = focused tests of changed behavior + existing suite + the task's named acceptance check. Heavier proofs require a task basis, a named material risk, or an explicit user decision. Evidence is per task acceptance, not per checkpoint ceremony.
6. Ponytail: a pass is valid only with a removal list and a mapping of every retained component/task/test to an accepted requirement, architecture component, or necessary delivery/safety obligation. Ponytail runs at architecture and task-map design only, never during ordinary execution.
