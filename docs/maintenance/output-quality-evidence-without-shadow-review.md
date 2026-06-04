# Output-quality safeguards without a shadow reviewer layer

Task: `t_2b6f9220`

This supersedes the closed separate output-quality audit/reviewer design (prior PR #23). The replacement is a narrow completion-evidence policy plus objective watchdog evidence-presence and activation checks. It is meant to prevent false confidence in `done`/`merged` work while preserving the responsibility boundary: the responsible profile owns review, testing, deployment validation, and judgment; watchdogs verify only durable evidence and objective activation facts.

## Anti-duplication rule

Watchdogs must not become subjective reviewers. A global or project watchdog may check whether required completion evidence exists, whether cited handles are internally consistent, and whether simple activation probes pass. It must not rerun the responsible profile's full test plan, judge code quality, re-review architecture, decide whether a test plan is sufficient for the feature, or shadow-approve deployment work.

If the evidence is missing or mechanically inconsistent, the watchdog reports `waiting` with the exact missing evidence. If the responsible explicitly records a human decision that is still needed, the task remains `blocked`. If evidence exists and objective probes pass, the watchdog does not assert the work is good; it asserts only that activation evidence is present.

## Required responsible completion evidence

Before marking deployment/runtime/user-visible work `done`, the responsible handoff must attach evidence at the smallest practical scope:

1. Scope and user-visible effect
   - What changed, who/what should see it, and the intended runtime surface: CLI command, gateway route, dashboard page, Matrix room behavior, API endpoint, cron job, systemd unit, or documentation-only artifact.
2. Source-control evidence
   - Branch or PR URL/number when applicable.
   - Merge state when claiming merged work.
   - Commit SHA(s) or tag(s) that contain the accepted change.
   - Local checkout/worktree path and `git rev-parse HEAD` when local activation matters.
3. Verification evidence
   - Tests/checks run, with command, result, and relevant scope.
   - For docs-only changes, `git diff --check` or equivalent formatting/link sanity is sufficient unless the docs build is in scope.
   - For code/config/runtime changes, include the targeted test or probe that demonstrates the changed path, not only a broad unrelated suite.
4. Activation evidence for runtime/deployment work
   - Service/process restart or reload evidence after the code/config change, when required for activation.
   - Runtime version/commit/config observed by the service if available.
   - Endpoint/API/CLI/Matrix/dashboard probe output that exercises the changed surface.
   - Kanban/Matrix/GitHub state evidence when the change is a workflow/status/routing change.
5. Residual risk and judgment
   - Any known untested areas, fallback assumptions, manual review needs, and why the responsible believes the work is complete enough to mark `done`.
   - Human approval handle when policy requires it.

Recommended Kanban metadata keys for completed deployment/runtime/user-visible work:

```json
{
  "evidence_version": 1,
  "completion_class": "docs|code|runtime|deployment|workflow",
  "source_control": {
    "branch": "...",
    "pr": "...",
    "merged": true,
    "commits": ["..."]
  },
  "verification": [
    {"command": "...", "result": "passed", "scope": "..."}
  ],
  "activation": [
    {"surface": "...", "probe": "...", "result": "passed", "observed_commit": "..."}
  ],
  "responsible_judgment": {
    "residual_risk": "...",
    "human_approval": "... or null"
  }
}
```

## Objective watchdog checks

These checks are objective enough for project-owned watchdogs because they inspect presence, consistency, or a simple external observable. They do not require judging whether the responsible chose the best design or enough tests.

1. Evidence schema presence
   - A `done` task in a configured completion class includes `evidence_version`, source-control evidence when applicable, verification evidence, and activation evidence when the task claims runtime/deployment activation.
2. PR/merge consistency
   - Referenced PR exists and is merged when the handoff claims merged work.
   - Referenced commit is reachable from the declared branch/mainline or local checkout.
3. Local activation consistency
   - Local checkout exists and contains the referenced commit when the handoff claims local deployment from that checkout.
   - The claimed runtime command, process, service, or config path exists where the responsible says it does.
4. Restart/reload after change
   - A service restart/reload timestamp or journal/process marker is newer than the activated commit/config timestamp when that evidence is available and the service requires restart.
5. Surface probe
   - A declared endpoint/API/CLI/Matrix/dashboard probe returns the expected mechanical status, version, state label, or sentinel string.
   - For Kanban/Matrix/GitHub workflow changes, the watchdog can query the relevant state and compare it to declared canonical values.
6. Task-state activation evidence
   - A task marked `done` after deployment work includes the evidence handles above and does not rely only on prose such as "merged" or "should be live".

## Responsible-owned judgment that watchdogs must not duplicate

These remain owned by the responsible profile and, where required, human review:

- Whether the implementation design is appropriate.
- Whether the complete test plan is sufficient for the feature's risk.
- Whether a code review should approve the implementation.
- Whether UX/product behavior is desirable beyond a mechanical probe.
- Whether security, privacy, migration, or rollout trade-offs are acceptable.
- Whether a residual risk is acceptable for release.
- Whether a flaky, partial, or environment-specific failure should block release.

A watchdog may report that the responsible did not attach a required judgment/risk note; it must not substitute its own judgment for that note.

## Missing-evidence detection protocol

Project-owned watchdogs should detect missing evidence by reading the responsible handoff and performing only bounded checks:

1. Classify the task from explicit metadata (`completion_class`) or conservative keywords in the task title/body/handoff.
2. Load the required evidence fields for that class.
3. Validate handles mechanically: URLs/PRs/commits/checkout paths/probe commands/state fields.
4. Run only declared lightweight activation probes, or a project-configured fixed probe list. Do not invent a full regression suite.
5. Emit one of the canonical live states:
   - `working` when active remediation/checking is underway or immediately queued.
   - `waiting` when evidence is missing, stale, inconsistent, or an automated/system dependency is pending.
   - `blocked` only when the handoff explicitly requires human action/decision.
   - `dormant` when no active actionable assignment exists.
6. Report the smallest actionable gap, for example: `waiting — done task t_x claims dashboard deployment but has no post-restart dashboard probe`.

## Fit with the lightweight global watchdog

The lightweight global watchdog from `t_b3b34717` remains downstream-only and limited to global invariants: PA health, Kanban dashboard health, project-watchdog presence/aliveness/version, exactly-one dispatch owner per board, and cross-project summary drift.

It must not expand into project output review. Its only relationship to completion evidence is meta-level: confirm that each active project has a project-owned watchdog capable of enforcing that project's evidence-presence policy, and report missing/stale project-watchdog capability as a global invariant problem. It should not inspect every project task's output quality or rerun project activation probes itself.

## Minimal replacement for PR #23

Replace the closed separate-layer design with these smaller artifacts:

1. Policy doc: this file defines the anti-duplication boundary, required responsible evidence, objective watchdog checks, missing-evidence protocol, and status semantics.
2. Responsible completion template: project coordinators may add the JSON metadata shape above to responsible SOUL/profile policy, Kanban task templates, or project docs.
3. Project-watchdog revision: each project-owned watchdog may validate evidence presence and declared activation probes for its own project only.
4. Global-watchdog guardrail: global audit remains limited to global invariants and project-watchdog presence/aliveness/version.
5. Tests: keep a focused policy test that fails if this document drops the anti-duplication rule, required evidence sections, objective-only watchdog boundary, global-watchdog non-expansion rule, or canonical live status terms.

## Status-line semantics

All user-facing profile/project notifications must use the canonical pair:

```text
Self status: WORKING|WAITING|BLOCKED|DORMANT — <specific active action/wait/human blocker/dormant condition for this profile itself>
Lineage status: WORKING|WAITING|BLOCKED|DORMANT — <aggregate status for structural descendants; DORMANT when none exist or none have active work>
```

Do not introduce role-specific status labels or legacy "not blocked" phrasing. `blocked` is only for concrete human intervention. Non-human waits such as CI, PR guard, worker availability, service restart, deployment propagation, or missing mechanical evidence are `waiting`.
