# Readiness Gate

Before generating any tracking, judge whether the plan can actually be built. If the user only asked to check readiness, this gate is the deliverable — report the verdict and stop.

Inventory what planning actually exists: scan `{planning_artifacts}` and `{project_knowledge}` for intent and planning artifacts — briefs, PRFAQs, PRDs, specs, UX outputs, architecture, epics and stories. Identify documents by reading what they are, not by filename patterns; projects arrive with different artifact mixes and naming.

Assess the plan as a whole against one question: **could a developer implement these epics without inventing decisions nothing records?**

- Requirements and decisions in the intent artifacts trace forward into stories; stories trace back to recorded intent — flag orphans in both directions
- Epics deliver user value and carry no forward dependencies; stories are independently completable
- Architecture and UX decisions the stories rely on are recorded somewhere, not assumed
- Conflicts between artifacts (a spec and an epic disagreeing) are surfaced, not silently resolved

A missing document type is only a finding if stories depend on decisions nothing records — a project with no UX artifact and no UI stories is fine.

Deliver a verdict:

- **PASS** — state it in one line; for the full sprint-planning intent, continue with `generate-tracking.md`
- **CONCERNS** — list them briefly with where each gap lives; ask `{user_name}` whether to proceed anyway or fix first
- **FAIL** — the plan is not implementable as recorded. Present findings ordered by severity, name the skill that fixes each (the relevant plan skill, or `bmad-correct-course` for cross-cutting changes), offer to save the findings to `{planning_artifacts}/implementation-readiness.md`, and stop
