---
title: "source-of-truth: Kanban drain controller for blocked/todo queues"
status: proposed
date: 2026-06-13
type: design
owner: factory-orchestrator
source_task: t_dcfc0b32
---

# source-of-truth: Kanban drain controller for blocked/todo queues

## Problem / outcome

The Kanban board can accumulate hundreds of open cards outside the dispatchable queue when review handoffs, human holds, parent-gated work, timeout/gave-up rows, and duplicates all collapse into `blocked` or `todo` without a machine-readable reason class.

Outcome: add a safe drain controller design that classifies non-dispatchable rows, routes only machine-safe classes, and leaves human/credential/prod-risk holds untouched. The controller must make stalled work visible, auditable, idempotent, and review-aware without bypassing the existing Kanban lifecycle.

## In-scope behavior

- Classify `todo`, `blocked`, and `review` rows into drain classes with explicit evidence.
- Convert review-required handoffs into the review lane without rerunning the implementation worker just to mark review.
- Consume Review Packet PASS/FAIL outcomes against the source card.
- Reslice timeout/gave-up work only when the failure is not a human/credential/prod-risk hold.
- Mark true superseded duplicates as archived or linked to their canonical card with an audit comment.
- Support per-profile cap overrides for controlled drain lanes.
- Provide a dry-run/report mode before any mutation.

## Non-goals

- No deploy, merge, release, or production promotion.
- No unsafe auto-approval of code changes.
- No automatic handling of credentials, secrets, auth blockers, payment operations, public posting, prod-risk approvals, or other human/operator decisions.
- No parent dependency bypass.
- No replacement for the dispatcher; the controller is a classifier/action planner layered beside the existing Kanban kernel.

## Domain terms / classes

- Dispatchable: a task that the dispatcher may claim now. Today this means an unclaimed `ready` row, or an unclaimed `review` row, with a real/spawnable assignee and remaining global/per-profile capacity. A drain controller may report dispatchability; it must not redefine the claim invariant.
- Sticky blocked: a `blocked` task whose latest `blocked`/`unblocked` event is `blocked`. Current `_has_sticky_block()` treats this as an explicit worker/operator block that `recompute_ready()` must not auto-promote.
- Review-required: a code-change handoff that says review is needed, currently often encoded as `kanban_block(reason="review-required: ...")` plus a preceding structured comment. This is not a human hold if the required next step is a normal reviewer lane.
- Human hold: a block that needs a human decision, scope clarification, UX/product call, credential, secret, auth fix, production approval, public-posting approval, payment approval, or other non-agent authority.
- Credential hold: a human hold specifically caused by missing/invalid credentials, quota requiring operator action, OAuth/device-code requirements, or secret/config access.
- Prod-risk hold: a human hold whose next action could affect production infrastructure, public posting, data retention, PII, payments, secrets, or high-blast-radius state.
- Timeout/gave-up: a task blocked or stalled by `timed_out`, `gave_up`, repeated spawn failures, protocol violations, or max-runtime exhaustion. This class is drainable only after the controller proves it is not sticky-human/credential/prod-risk.
- Superseded duplicate: a card whose requested outcome is already represented by another canonical task, PR, child task, or completed artifact. Supersession must be explicit and auditable; title similarity is never enough.
- Parent-gated todo: a `todo` task with one or more parent links where at least one parent is not terminal. Parent-gated todo is not stuck and must not be promoted until all parents are `done` or otherwise terminal under the existing dependency rules.

## Lifecycle states

Existing states remain authoritative: `triage | todo | ready | running | review | blocked | done | archived`.

Drain-class overlay:

```text
todo(parent-gated)         -> no-op/report until parents terminal
blocked(human_hold)        -> no-op/report; needs operator unblock/decision
blocked(credential_hold)   -> no-op/report; needs credential/operator action
blocked(prod_risk_hold)    -> no-op/report; needs explicit approval
blocked(review_required)   -> review, with audit comment and reviewer routing
review + Review PASS       -> done, with review packet metadata/comment
review + Review FAIL       -> ready or blocked, with failure packet and owner route
blocked(timeout_gave_up)   -> resliced child tasks or ready retry only if safe
blocked(superseded)        -> archived, with canonical link and audit comment
unknown blocked/todo       -> no-op/report until classified
```

## Ownership boundaries

- CEO owns priority.
- Product Manager owns scope and acceptance of product/UX decisions.
- Factory Orchestrator owns lane movement, drain policy configuration, and implementation routing.
- Design-with-Docs owns this source-of-truth artifact, term definitions, invariants, and ADR triggers.
- Reviewer lane owns Review Packet PASS/FAIL evidence; it does not silently merge or deploy unless a separate authorized workflow says so.
- Implementation workers own code changes and fix follow-up tasks; they do not close their own review-required source card without review evidence.
- Human/operator owns human holds, credential holds, prod-risk holds, and unsafe ambiguity.

## Invariants

1. Never auto-run human, credential, secret, auth, payment, public-posting, data-retention, PII, prod-risk, or high-blast-radius holds.
2. Never bypass parent dependencies. Parent-gated todo remains gated until existing parent-terminal rules pass.
3. Never rerun a completed code-change card merely to mark review; Review Packet PASS can close the source card directly through an audited review path.
4. Every drain action must append an audit comment to the source card before or in the same transaction as the state change.
5. Every drain action must be idempotent. Re-running the controller with the same inputs must not duplicate comments, child tasks, review packets, or archive actions.
6. Unknown or conflicting classification is a no-op plus report, not a mutation.
7. Existing dispatcher claim rules remain authoritative. The controller prepares/moves rows; it does not directly execute worker code.
8. Sticky blocked remains safe by default. Only explicit machine-readable `block_class=review_required|timeout_gave_up|superseded_duplicate` may be considered for automated drain.
9. Review FAIL must preserve the source-card audit trail and route back to the correct implementation owner or create a follow-up task; it must not mark done.
10. Dry-run/report mode must be available for every mutation class.

## Interfaces / contracts

### Block metadata contract

Preferred durable shape: structured block metadata, either as `tasks.block_class` plus `tasks.block_metadata` or as the latest `blocked` event payload with equivalent fields. The implementation should choose the smallest migration that preserves queryability.

Required fields:

```json
{
  "block_class": "review_required | human_hold | credential_hold | prod_risk_hold | timeout_gave_up | superseded_duplicate | parent_gated | unknown",
  "source": "worker | operator | dispatcher | drain_controller | reviewer",
  "reason": "human-readable one-line reason",
  "evidence": ["comment:<id>", "run:<id>", "event:<id>", "task:<id>"],
  "owner": "profile-or-human-role",
  "created_at": 1780000000
}
```

Rules:

- `kanban_block(reason=...)` without `block_class` remains `unknown` or sticky-human by default.
- Prefix-only parsing such as `review-required:` may be supported as a migration bridge, not as the final source of truth.
- `human_hold`, `credential_hold`, and `prod_risk_hold` are terminal for automation until explicit unblock/reclassify.

### Drain action contract

Each mutation writes an idempotent action marker:

```json
{
  "drain_action_id": "sha256(board, task_id, class, target_state, evidence_ids)",
  "action": "route_to_review | consume_review_pass | consume_review_fail | reslice_timeout | archive_superseded",
  "source_task_id": "t_x",
  "target_task_ids": ["t_y"],
  "previous_status": "blocked",
  "new_status": "review",
  "evidence": ["comment:<id>", "run:<id>", "event:<id>"],
  "dry_run": false
}
```

Idempotency key placement can be a dedicated `task_drain_actions` table or a unique event/comment marker. A dedicated table is cleaner for queries; an event marker is a smaller first slice.

### Review Packet contract

Reviewer output must be structured and attached to the source card:

```json
{
  "review_packet_version": 1,
  "source_task_id": "t_x",
  "verdict": "PASS | FAIL",
  "reviewer": "profile-or-human",
  "evidence": {
    "diff_path": "optional path or PR URL",
    "tests_run": ["command strings"],
    "findings": []
  },
  "required_followups": []
}
```

PASS consumption:

- Verify packet references the source task and has review evidence.
- Append audit comment with `drain_action_id`.
- Mark the source card `done` without respawning the implementation worker.
- Preserve review run metadata.

FAIL consumption:

- Append audit comment with findings and owner.
- Route the source card back to implementation (`ready`) only if it is safe and scoped.
- Otherwise create/route child fix tasks and keep the source blocked/review-gated until fixes pass.
- Never merge, deploy, or mark done.

### Drain controller command/API shape

Proposed operator surface:

```bash
hermes kanban drain --dry-run --board <slug> [--class review_required] [--limit N]
hermes kanban drain --apply --board <slug> --class review_required --limit N
hermes kanban drain report --board <slug>
```

Report fields:

- counts by status and block_class
- dispatchable count
- parent-gated todo count
- sticky human/credential/prod-risk hold count
- review-required routable count
- timeout/gave-up candidates and unsafe exclusions
- superseded candidates and canonical links
- per-profile cap bottlenecks
- proposed actions with evidence and idempotency keys

## Data / state implications

- Add queryable block classification or structured event payloads. If using event payloads only, add helper queries so the dashboard/diagnostics do not need to parse comments.
- Add a durable idempotency marker for drain actions.
- Dashboard and CLI diagnostics should display block_class and drain eligibility separately from raw `status`.
- Review-required source cards should move through `review`, not remain indefinitely sticky-blocked when all required handoff metadata exists.
- Parent links remain unchanged; controller never writes around them.
- Superseded duplicate handling should preserve audit history by archiving, not deleting.

## Proposed mechanics

1. Classify first: the controller reads tasks, latest block/unblock events, comments, runs, parent links, assignee/profile existence, and caps. It emits a report before mutation.
2. Require explicit class for automation: only `review_required`, safe `timeout_gave_up`, and verified `superseded_duplicate` may mutate automatically. `unknown` is report-only.
3. Route review-required to review lane: if a blocked card has review handoff metadata and no human/prod/credential hold, append an audit comment and set status to `review` with reviewer routing. Do not spawn the implementation worker.
4. Consume Review Packet: PASS closes source as done with metadata; FAIL comments findings and routes fix work to the owning implementation lane or child tasks.
5. Reslice timeout/gave-up: for safe timeout classes, create smaller child tasks or lower-scope retries with parent/child links and archive/block the original as appropriate. Do not blindly reset failure counters.
6. Supersede duplicates: only archive when a canonical task/PR/artifact is cited and the duplicate has no unique acceptance criteria. Comment both duplicate and canonical when possible.
7. Cap overrides: allow controller-specific reviewer/drain profiles to have configured burst caps, e.g. `kanban.drain.max_in_progress_per_profile_overrides`, while retaining global caps for normal workers.
8. Dry-run by default: apply mode requires explicit `--apply` and emits the same report plus committed action IDs.

## Minimal implementation slices

### Slice 1: classifier + dry-run report

Acceptance criteria:

- `hermes kanban drain --dry-run` reports counts by status and inferred/provided class.
- Parent-gated todo rows are listed but never proposed for promotion.
- Sticky blocked rows without safe class are report-only.
- Tests cover review-required prefix bridge, unknown sticky block, parent-gated todo, and cap bottleneck reporting.

### Slice 2: structured block_class write/read path

Acceptance criteria:

- `kanban_block`/CLI block can persist `block_class` and metadata.
- Existing `reason`-only blocks remain backward-compatible and sticky-safe.
- Dashboard/CLI show block_class when present.
- Tests prove human/credential/prod-risk classes are never proposed for auto-drain.

### Slice 3: review-required auto routing

Acceptance criteria:

- A blocked `review_required` source card with handoff metadata moves to `review` exactly once.
- Controller appends an audit comment containing action ID and evidence.
- Re-running apply mode does not duplicate comments or state changes.
- Tests prove implementation worker is not respawned merely to mark review.

### Slice 4: Review Packet PASS/FAIL consumption

Acceptance criteria:

- PASS marks the source card done with review metadata and no implementation rerun.
- FAIL preserves findings and routes scoped follow-up work without marking done.
- Invalid/missing packet leaves the card unchanged and reports the reason.
- Tests cover idempotent PASS, idempotent FAIL, and malformed packet no-op.

### Slice 5: timeout/gave-up reslicing

Acceptance criteria:

- Safe timeout/gave-up rows can be resliced into child tasks with explicit scope and parent links.
- Human/credential/prod-risk causes are excluded.
- Failure counters are not blindly cleared on the source card.
- Tests cover timeout reslice, gave-up safe reslice, auth blocker exclusion, and duplicate-run idempotency.

### Slice 6: superseded/archive handling + cap overrides

Acceptance criteria:

- Superseded duplicate archive requires canonical evidence and writes audit comments.
- Per-profile cap override is limited to configured drain/reviewer profiles and visible in dry-run report.
- Tests prove normal dispatcher caps remain unchanged for ordinary ready rows.

## Risks / controls

- Risk: auto-drain could move true human holds. Control: safe classes require structured metadata; unknown stays report-only.
- Risk: review PASS could close work without evidence. Control: packet schema and source-task match required.
- Risk: duplicate archive could hide unique work. Control: require canonical link and no unique acceptance criteria; archive, never delete.
- Risk: cap override could overload a profile. Control: overrides are opt-in by profile/class and reported in dry-run.
- Risk: lifecycle mutation has high blast radius for Software Factory OS. Control: require ADR and Security/Risk review before enabling apply mode by default.

## Open questions

1. Should block_class live on `tasks` for fast filtering, on latest `task_events.payload` for append-only history, or both?
2. Which profile owns default review-required routing: original assignee with `sdlc-review`, a dedicated reviewer profile, or board config?
3. Should Review Packet PASS create a synthetic review run when the packet came from a human/dashboard action rather than a spawned review worker?
4. What exact timeout/gave-up evidence is sufficient to classify as safe for reslice rather than human hold?
5. Should superseded duplicates archive immediately or move to a `blocked` superseded class for a human grace period?

## ADR triggers

ADR required before implementation because this changes durable Kanban lifecycle semantics, introduces new block classification contracts, and affects automated movement of high-volume Software Factory OS work queues.

Security/Risk review required before enabling apply mode because misclassification can affect infrastructure/high-blast-radius task routing and because hold classes explicitly include credentials, secrets, PII, payments, public posting, and production-risk boundaries.

## Authoritative sources reviewed

- Task `t_dcfc0b32` body and observed evidence from 2026-06-13.
- `AGENTS.md` repo development guide and workspace charter.
- `website/docs/user-guide/features/kanban.md` for canonical board/task/link/comment/workspace/dispatcher concepts and status set.
- `website/docs/user-guide/features/kanban-worker-lanes.md` for lifecycle terminators, review-required convention, audit trail, and failure modes.
- `docs/kanban/multi-gateway.md` for single-dispatcher posture and board DB ownership.
- `hermes_cli/kanban_db.py` around `_has_sticky_block()`, `recompute_ready()`, `claim_review_task()`, and `dispatch_once()` review dispatch.
- `gateway/kanban_watchers.py` for gateway dispatcher config, stale timeout, default assignee, and per-profile cap handling.
- `tests/hermes_cli/test_kanban_blocked_sticky.py` for sticky block regression contracts.
- `tests/hermes_cli/test_kanban_db.py` review dispatch tests for review status, `claim_review_task()`, dry-run, nonspawnable review rows, and `sdlc-review` skill loading.

## Handoff to Factory Orchestrator

- Artifact path: `docs/plans/2026-06-13-kanban-drain-controller-source-of-truth.md`.
- Domain definitions: see Domain terms/classes.
- Scope/non-goals: see In-scope behavior and Non-goals.
- Interfaces/contracts: block metadata, drain action, Review Packet, and CLI/API report contracts above.
- Data/state changes: queryable block_class metadata, idempotent drain action marker, review-lane source-card closure, dashboard/CLI diagnostics display.
- Acceptance criteria/tests: implementation slices 1-6.
- Risk triggers: ADR plus Security/Risk review before apply-mode rollout.
- Open questions: resolve before code routing.
