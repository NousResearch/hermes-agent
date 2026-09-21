---
name: feature-pipeline
version: 2
description: >
  Enforced feature-track pipeline for tier=full tasks. 12 stages: Intake →
  Research → PRD → Spec → Council → Sign-off → Tech Review → Decompose →
  Execute → PR+QA → Audit → Final Sign-off → Document. Each stage has an
  artifact-or-block gate. Express path drops PRD, Council, Tech Review.
category: governance
always_skills: []
---

# Feature Pipeline

Gated progression for tier=full features. Every stage produces an artifact.
No artifact = no promotion. No exceptions.

## Pipeline Stages (12 stages, design doc 3)

1. **Research** — Remii produces research-brief.md. Gate: `validate_research_artifact()`.
2. **PRD** — Kensei produces prd.md (Problem, Users, Scope, Out of Scope, Metrics).
   Gate: `validate_prd_artifact()`.
3. **Spec** — Octacon produces spec.md (Architecture, Interfaces, Test Strategy).
   Gate: `validate_spec_artifact()`.
4. **Council** — LLM council deliberates on PRD+Spec. Three-phase: parallel independent
   review, anonymised cross-ranking, chairman synthesis. Gate: `validate_council_artifact()`
   (runs deliberation on first tick, reads cached verdict on subsequent ticks).
5. **Sign-off** — Human approval gate. Uses event-table check (`human_approved` events),
   not thread-blocking. Owner: Sahil via CLI/Discord.
6. **Tech Review** — Octacon validates implementability. Gate: `validate_tech_review_artifact()`
   (checks tech-review.md with Architecture + Risk Assessment sections).
7. **Decompose** — Orchestrator breaks into child tasks (WS-1 contract).
   Gate: `validate_decompose_artifact()` (checks decompose-output.md with
   Child Tasks + Acceptance Criteria + Test Plan).
8. **Execute** — Workers claim and implement. Pass-through stage: no gate function,
   dispatcher auto-advances to next stage.
9. **PR+QA** — Pull request open + tests green. Pass-through: no gate function,
   dispatcher auto-advances to audit.
10. **Audit** — Quan fleet (code/arch/perf/security/UX) + Kensei-review multi-layer audit.
    Gate: `validate_audit_artifact()`. Three verdicts:
    - PASS — gate passes, advance to final_sign_off.
    - CONDITIONAL — gate passes, auto-creates `[audit-followup]` child task, advance.
    - BLOCKED — gate fails, bounces to spec (capped at max_revise_loops).
11. **Final Sign-off** — Human approval gate. Same event-table mechanism as stage 5.
    Owner: Sahil via CLI/Discord.
12. **Document** — Light writes wiki/docs. Gate: `validate_document_artifact()`
    (checks docs-output.md with Overview + Usage sections).

## Express Path

Drops PRD, Council, and Tech Review. Stages (9):
`research → spec → sign_off → decompose → execute → pr+qa → audit → final_sign_off → document`

Per-task mode stored in `tasks.pipeline_mode` column (`full` or `express`).
Set via `hermes feature create --express`. Skipped stages are logged as
`bypass_record` events for Denji governance review.

Config: `pipeline.express_enabled: true` (default: true)

## Artifact Storage

All artifacts at `~/.hermes/feature-artifacts/<task_id>/`. Git-tracked for audit trail.

Per-stage filenames:
- research: `research-brief.md`
- prd: `prd.md`
- spec: `spec.md`
- council: `council-verdict.md`
- tech_review: `tech-review.md`
- decompose: `decompose-output.md`
- audit: `audit-report.md`
- document: `docs-output.md`

## Gate Functions

In `hermes_cli/feature_pipeline.py`:

- `validate_intake_brief(body)` — checks for ## Problem + ## Success Criteria
- `validate_research_artifact(artifact_dir)` — checks research-brief.md + ## Findings
- `validate_prd_artifact(artifact_dir)` — checks prd.md with 5 sections
- `validate_spec_artifact(artifact_dir)` — checks spec.md with 3 sections
- `validate_council_artifact(artifact_dir)` — runs deliberation / reads cached verdict
- `validate_tech_review_artifact(artifact_dir)` — checks tech-review.md
- `validate_decompose_artifact(artifact_dir)` — checks decompose-output.md (WS-1 contract)
- `validate_audit_artifact(artifact_dir)` — parses PASS/CONDITIONAL/BLOCKED from audit-report.md
- `validate_document_artifact(artifact_dir)` — checks docs-output.md

Helper functions:
- `check_human_approved(conn, task_id, stage)` — event-table check for human gates
- `time_in_stage_hours(conn, task_id, stage)` — idle time for stale-nudge
- `get_audit_verdict(artifact_dir)` — read verdict from audit-report.md for dispatcher

Gate returns: `Optional[str]` — None = pass, string = blocked with reason.

Human gates (`sign_off`, `final_sign_off`) are NOT in `GATE_FUNCTIONS`. They use
event-table checks. Pass-through stages (`execute`, `pr+qa`) have no gate function.
`HUMAN_GATE_STAGES = {"sign_off", "final_sign_off"}`
`PASS_THROUGH_STAGES = {"execute", "pr+qa"}`

## Council (Phase B)

`hermes_cli/council.py` — 672 LOC three-phase deliberation:
1. Independent review (parallel, all panel models via ThreadPoolExecutor)
2. Cross-ranking (anonymised Reviewer A/B/C labels)
3. Chairman synthesis (APPROVED/REVISE with deduplicated issues)

Config: `council:` section in config.yaml (panel, chairman, token_cap, timeouts).
Per-member fallback chains: transient errors (429, 402, connection) retry;
permanent errors fail immediately. Chairman failure -> auto-REVISE.

Loop cap: council REVISE bounces back to spec with `max_revise_loops` cap (default 4).

## CLI Commands

```bash
hermes feature create "Feature name" --priority high
hermes feature create "Express feature" --express
hermes feature status [task_id]
hermes feature advance <task_id>
hermes feature sign-off <task_id> --note "looks good"
hermes feature reject <task_id> "needs more error handling coverage"
```

Reject bounce targets: `sign_off` -> `spec`, `final_sign_off` -> `tech_review`.
Both commands validate the task is at a human-gate stage before writing.

## Stage Ownership

| Stage         | Owner           |
|---------------|-----------------|
| Research      | remii           |
| PRD           | kensei          |
| Spec          | octacon         |
| Council       | council service |
| Sign-off      | human           |
| Tech Review   | octacon         |
| Decompose     | orchestrator    |
| Execute       | leads           |
| PR+QA         | quan            |
| Audit         | quan + kensei-review |
| Final Sign-off | human          |
| Document      | light           |

## Config

```yaml
pipeline:
  max_revise_loops: 4              # Max council revisions before escalation
  sign_off_timeout_hours: 48       # Hours before stale-nudge on human gates
  token_cap: null                  # Per-council token cap (None = no cap)
  express_enabled: true            # Allow express path
  artifact_dir: feature-artifacts  # Relative to HERMES_HOME
  spawn_frequency_threshold: 8     # Denji promotion proposal trigger
  stage_owners:
    research: remii
    prd: kensei
    spec: octacon
    council: ""
    sign_off: ""
    tech_review: octacon
    decompose: kensei
    execute: octacon
    pr+qa: quan
    audit: quan
    final_sign_off: ""
    document: light
```

## VALID_STATUSES

All 22 kanban statuses (Phase A->D):
`triage, todo, scheduled, ready, running, blocked, review, done, archived, backlog,
research, prd, spec, council, sign_off, tech_review, decompose, execute, pr+qa,
audit, final_sign_off, document`

`PIPELINE_STAGES` in feature_pipeline.py defines the canonical order for all 12 stages.
`EXPRESS_PIPELINE_STAGES` defines the 9-stage express subset.

## Notification Events

The gateway kanban notifier (`gateway/run.py` `_kanban_notifier_watcher`) delivers
pipeline events alongside terminal kanban events via `NOTIFY_KINDS`:

- `pipeline_advanced` — task moved to next stage. Payload: `from_stage`, `to_stage`.
- `pipeline_complete` — task reached end of pipeline. Payload: `stage`, `mode`.
- `gate_failed` — gate check failed. Payload: `stage`, `reason`. Council REVISE adds `bounced_to` and `revise_count`.

Events are written in `kanban_db.py` dispatch section. The notifier claims them via
`NOTIFY_KINDS = TERMINAL_KINDS + PIPELINE_KINDS`. Pipeline events do NOT trigger
unsubscription (tasks continue to next stage).

## Denji Wiring (Phase D)

- `_record_pipeline_spawn(conn, task_id, stage, assignee)` — fires each time
  dispatcher spawns a lead for a pipeline stage. Accumulated in `task_events`.
- `get_spawn_frequency(conn, days=7)` — aggregates by (assignee, stage) for
  Denji's review-cycle scripts. Configurable threshold (default 8).
- `_record_denji_review_signal(conn, task_id, signal_type, **details)` — emits
  typed signals (audit_conditional, audit_passed) for Denji consumer.
- `_create_audit_followup_task(conn, parent_id, verdict)` — creates `[audit-followup]`
  child task linked via `audit_followup` relation for CONDITIONAL verdicts.
- `_record_bypass_record(conn, task_id, skipped_stages, launched_by, mode)` —
  logs express-path launches for Denji governance sampling.

## Dashboard (Phase A.8)

Backend endpoint: `GET /api/pipeline/stages` — returns `{stages: {...}, pending_gates: [...]}`
with gate status for each pipeline task. Frontend: Kanban page shows pipeline stages
in ORDER + COLUMN_TONE, with a pending-gate alert bar (auto-refresh 15s).

See `kensei-dashboard` skill for implementation details.

## Updating Pipeline Stages

When modifying the pipeline (adding/removing/reordering stages):

1. Update `PIPELINE_STAGES` in `hermes_cli/feature_pipeline.py`
2. Sync `VALID_STATUSES` in `hermes_cli/kanban_db.py`
3. Add gate functions for new stages following existing pattern
4. Update `GATE_FUNCTIONS` mapping
5. Update dispatcher logic in `kanban_db.py` (dispatch branches, helpers, events)
6. Add schema migrations (use `_add_column_if_missing()` for new columns)
7. Update tests (assertions, stage sequences, new gate tests)
8. Update `EXPRESS_PIPELINE_STAGES` if new stages should be skipped in express mode

## Pitfalls

- **kensei-intake profile does NOT exist.** PRD stage is owned by the `kensei` (default)
  profile. Do not create or reference a kensei-intake profile.
- **Task.pipeline_stage is a dataclass field.** When adding the `pipeline_stage` column
  to the schema, you MUST also add it to the `Task` dataclass (field + `from_row()`
  constructor) or `task.pipeline_stage` will raise AttributeError at runtime.
- **Task.tier is a dataclass field — SAME CLASS OF BUG.** The `tier` column exists in
  the DB schema (line 1073) and migration (line 1746), and `feature.py` line 140
  references `task.tier`. If `tier` is missing from the `Task` dataclass (field +
  `from_row()`), `hermes feature status <id>` crashes with `AttributeError: 'Task'
  object has no attribute 'tier'`. This was live on 07/06/26 and silently broke the
  pipeline status CLI for all tasks. Any new column added to the tasks table schema
  MUST be added to the dataclass in the same commit.
- **Reviewer concurrency lane** (`_max_review_spawn`): Review dispatch uses a dedicated
  cap `max(1, max_spawn // 2)` instead of sharing the ready-task pool. This prevents
  review starvation under load — without it, all `max_spawn` slots could be filled by
  workers and no reviewer would spawn. The counter `_review_spawned` is incremented
  on each review spawn in the dispatch loop. See Phase 2 P2-1. The total fleet
  concurrency ceiling is `max_spawn + max_review_spawn`.
- **Human gates use event-table state, not approval.py.** `sign_off`/`final_sign_off`
  check `task_events` for `human_approved` rows. They do NOT block threads.
- **Council deliberation is SLOW** — only fires once per council stage entry.
  Subsequent ticks read the cached `council-verdict.md`.
- **Council REVISE bounce is special-cased** in dispatcher: resets to `spec` with
  loop counter in events table, does NOT spawn a lead.
- **Imports in dispatcher** are at function scope, not module top — `PIPELINE_STAGES`,
  `GATE_FUNCTIONS`, `HUMAN_GATE_STAGES`, `get_next_stage`, `check_human_approved`,
  `time_in_stage_hours`.
- **Pass-through stages auto-advance.** `execute` and `pr+qa` have no gate function;
  dispatcher moves them forward each tick.
- **Audit gate has three verdicts.** PASS and CONDITIONAL both pass the gate
  (CONDITIONAL auto-creates follow-up). Only BLOCKED bounces to spec.
- **Pipeline mode is per-task.** Stored in `tasks.pipeline_mode` column.
  `get_next_stage()` accepts a `pipeline_mode` parameter. Default is `full`.
- **Express path drops PRD, Council, and Tech Review only.** Audit is RETAINED in
  express mode (design doc 3: express still gates on multi-layer audit). Skipped stages
  are logged as `bypass_record` events for Denji to sample. The pitfall in earlier
  revisions of this file said "drops Audit" — that is WRONG. Audit is not optional in
  any path.
- **events vs task_events table**: all kanban events use `task_events`, NOT `events`.
- **Council REVISE loop counter** uses `council_revise` event rows counted at dispatch
  time, not stored in the tasks table.
- **Gateway notifier uses NOTIFY_KINDS** — pipeline events are dispatched alongside
  terminal events but do NOT trigger unsubscription.
- **`feature create` board routing may differ from `feature status` default board.**
  `hermes feature create --board default` may silently write to a different board
  (e.g., `apps`) depending on tenant resolution. Always verify the task's actual
  board with `hermes feature status <id>` after creation. On 07/06/26, task
  `t_222b1043` was created with `--board default` but landed on the `apps` board
  at `kanban/boards/apps/kanban.db`. The `feature status` CLI uses the current board
  resolution, so it may not find the task without explicit board routing.
- **Pipeline gate loop killed by ready-task pool saturation (07/06/26).**
  `dispatch_once()` at `kanban_db.py:7540` had `if max_spawn reached: break` at the
  top of the pipeline loop. This meant gate checks, council launches, and human-gate
  nudges ALL stopped when the ready-task pool was full. Gate checks are near-zero-cost
  (file stat + regex). The fix: removed the break — gate checks fire unconditionally.
  Only spawn-on-failure still gates on pool capacity.
- **Nonspawnable assignee deadlock (07/06/26).** When a pipeline task's assignee is an
  orchestrator profile (`kensei`) with no worker instance, `profile_exists()` returns
  False and the task is `skipped_nonspawnable` — silently, every tick, forever. The fix:
  `_get_stage_owner(stage)` reads `pipeline.stage_owners` from config.yaml, falls back
  to the correct lead (e.g. `remii` for `research`, `octacon` for `spec`), and reassigns
  the task with an `assigned` event for audit trail. This only triggers when the original
  assignee is confirmed nonspawnable.
- **WS-1 contract gate blocks pipeline workers (07/06/26).** When a `tier=full` task
  reaches a pipeline stage and the dispatcher spawns a worker, `validate_task_contract()`
  runs before the worker can claim the task. If the task body lacks ## Acceptance
  Criteria + ## Test Plan, the worker gets `claim_rejected` + `blocked`. Even though
  the pipeline worker is supposed to produce the artifact (e.g. research-brief.md),
  it can never start because the contract gate fires first. The body MUST carry AC +
  Test Plan sections before any pipeline worker can claim it. See
  `references/canary-debug-log-070626.md` for the full event trace.
- **Pipeline spawn uses wrong claim function (08/06/26).** The pipeline gate-fail
  path in `dispatch_once()` calls `claim_task()` to atomically claim the task before
  spawning. But `claim_task()` transitions `ready -> running` (`WHERE status = 'ready'`).
  Pipeline tasks have status `research`/`prd`/`spec`/`council` — NOT `ready`. The
  CAS always returns `rowcount=0`, `claim_task()` returns `None`, and the spawn is
  skipped silently. Result: gate_failed event fires every tick, stage-owner reassignment
  works, but NO WORKER EVER SPAWNS. The fix is `claim_pipeline_task()` — same pattern
  as `claim_review_task()` (which already handles `review -> running`). After the worker
  writes the artifact, it MUST call `complete_pipeline_task()` — NOT `kanban_complete` —
  to return the task to its original pipeline stage so the gate re-checks on the next tick.
  See `references/pipeline-claim-fix-080626.md`.
- **Pipeline workers call complete_pipeline_task, not kanban_complete.** After writing
  the stage artifact (e.g. research-brief.md, prd.md, spec.md), call
  `complete_pipeline_task(task_id, result="artifact written")`. This returns the task
  to its original pipeline status (e.g. `research`), sets `pipeline_stage` back, and
  clears the claim lock. The gate re-checks on the next dispatcher tick and advances
  if the artifact passes. Using `kanban_complete` would mark the task `done` and break
  the pipeline. Trigger: whenever you were spawned by the dispatcher at a pipeline
  stage (research/prd/spec/council).
- **Worker skill resolution failure (08/06/26).** Even when the feature-pipeline
  SKILL.md exists at the correct path, Hermes may fail to resolve it at startup:
  `Error: Unknown skill(s): feature-pipeline`. The worker exits immediately. The
  dispatcher detects the crash and retries, but the retry hits the same failure.
  Symptom: `gave_up` event after two consecutive crashes. Check with:
  `hermes --skills feature-pipeline -p <profile> --help > /dev/null`
  The pipeline claim/complete cycle is NOT at fault. See
  `references/pipeline-canary-debug-080626.md`.
- **Dispatcher four-branch shape is canonical.** `dispatch_once()` must have exactly:
  (1) gate passed -> next_stage, (2) council bounce, (3) audit bounce, (4) human gate
  event check, (5) else gate_failed. If you see two audit blocks or council mixed into
  an audit branch, the chain is corrupted — patch it back. See
  `references/kanban-ops-pipeline-extension-patterns.md` 2.
- **Every dispatcher UPDATE needs row id.** Common bug: `(value, value)` with 2
  bindings and 3 placeholders. Symptom: `sqlite3.ProgrammingError: Incorrect number
  of bindings supplied`.
- **Audit regex needs bold-marker pre-cleaning.** The reviewer pattern `\bcode\s*:\s*PASS\b`
  breaks when surrounded by `**code:** PASS`. Always pre-clean with
  `re.sub(r'\*{1,2}', '', body)` before word-boundary matching.
- **Audit Kensei-Review content min 50 chars.** Short stubs fail the gate.
  Minimum 50 chars of substantive content required.
- **Audit verdict format is `**Verdict: PASS**`, not `**PASS**`.** The parser regex
  requires the literal word "verdict" before the colon. Wrong format treats as
  missing verdict, bounces to spec.

## References

- `references/phase-a-implementation-notes.md` — Phase A concrete pitfalls
- `references/phase-b-council-plan.md` — Phase B design decisions
- `references/phase-c-implementation-notes.md` — Phase C human gate patterns
- `references/council-config-reference.md` — Council config schema and defaults
- `references/kanban-ops-pipeline-extension-patterns.md` — Phase D audit-specific pitfalls
- `references/canary-debug-log-070626.md` — First live pipeline canary event trace (t_222b1043)
- `references/spawn-eligibility-policy.md` — Two-layer spawn guard, nonspawnable_profiles config
- `references/pipeline-claim-fix-080626.md` — claim_pipeline_task + complete_pipeline_task design
- `references/pipeline-canary-debug-080626.md` — Full autonomous canary trace (t_88d4db38, claim/complete cycle proven)