---
name: kanban-ultragoal-ingress
description: Use when Chris says `ULTRAGOAL로 진행해` or equivalent; fail-closed into hermes-execution-routing's Kanban Ultragoal lane without falling back to Autopilot or generic Hermes direct.
version: 1.1.0
author: Hayase Yuuka
license: MIT
metadata:
  hermes:
    tags: [kanban, ultragoal, ingress, routing, execution-boundary]
    related_skills: [hermes-execution-routing, kanban-native-work-execution]
    requires_toolsets: [terminal, file]
---
# Kanban Ultragoal Ingress

Use this skill only as the thin natural-language ingress for explicit Ultragoal operator commands such as `ULTRAGOAL로 진행해`, `ultragoal로 구현해`, `BO-123 ultragoal로 진행`, or `이 parent ultragoal로 계속해`.

## Contract

This skill is **not** a second SSOT and not the executor itself.

1. Load and apply `hermes-execution-routing` first.
2. Record/respect the top-level lane as `Kanban Ultragoal`.
3. Preserve current runtime wire compatibility when needed: `routing_verdict=direct-kanban` may still be required by `kanban-ultragoal` even though the human lane is `Kanban Ultragoal`.
4. Keep Kanban as execution authority, Done Criteria, lifecycle, and audit SSOT.
5. Treat `.hermes/goal-runs/<id>/ledger.jsonl` as execution journal/proof only.
6. Confirm target card/parent, execution approval, allowed mutations, and forbidden side effects before calling `kanban-ultragoal`.
7. If any prerequisite is missing, fail closed with a blocker. Do **not** fall back to Autopilot, generic Hermes direct, free-floating Codex, or ordinary conversation.

## Execution shape

When prerequisites are present:

```text
natural phrase
→ kanban-ultragoal-ingress
→ hermes-execution-routing verdict: Kanban Ultragoal
→ current wire verdict if needed: direct-kanban
→ kanban-ultragoal pilot-check/run/status/resume/review-ready
→ verifier/cleanup proof
→ Kanban evidence
```

## Canonical operator sequence

Use the live `hermes kanban-ultragoal --help` surface before rediscovering command syntax. As of the verified CLI help, the canonical lifecycle commands are:

```bash
hermes kanban-ultragoal --json pilot-check <task-or-parent>
hermes kanban-ultragoal --json authority-snapshot <task-or-parent>
hermes kanban-ultragoal --json run <task-or-parent>
hermes kanban-ultragoal --json status <run_id>
hermes kanban-ultragoal --json resume <run_id>
hermes kanban-ultragoal --json tick <run_id>
hermes kanban-ultragoal --json build-closeout-evidence <run_id>
hermes kanban-ultragoal --json closeout-review-ready <run_id>
```

For evidence-lifecycle adapters, use the existing explicit record commands rather than hand-writing ad hoc closeout JSON when the runtime supports them:

```bash
hermes kanban-ultragoal --json record-worker-done <run_id> <json-or-file>
hermes kanban-ultragoal --json record-verifier-result <run_id> <json-or-file>
hermes kanban-ultragoal --json record-reviewer-result <run_id> <json-or-file>
hermes kanban-ultragoal --json record-pr-created <run_id> <json-or-file>
hermes kanban-ultragoal --json record-ci-result <run_id> <json-or-file>
hermes kanban-ultragoal --json record-cleanup-proof <run_id> <json-or-file>
hermes kanban-ultragoal --json mark-review-ready <run_id>
```

Rules:

- `pilot-check` and authority reread come before mutation/run start.
- `run`/`start` begins only inside the approved Kanban Ultragoal lane and side-effect boundary.
- `status`/`resume`/`tick` are run-management surfaces, not proof of completion.
- `build-closeout-evidence` and `closeout-review-ready` are the preferred bridge to governed Kanban closeout when available.
- If live CLI help changes, trust the live help and update this section; do not invent commands from memory.

## Stable aggregate pointer

Borrow the useful Gajae-Code Ultragoal pattern without importing `.gjc/ultragoal` as authority: the active Hermes `/goal` objective should be a **stable pointer**, not a mutable enumeration of every child/subgoal.

The pointer should identify:

- live Kanban authority item: internal task id and public id when available;
- run root: `$HERMES_HOME/goal-runs/<repo-slug>/<run_id>` or equivalent Hermes evidence store;
- Done Criteria snapshot/hash;
- current run ledger head or latest terminal event;
- forbidden side effects and approval boundary.

Rules:

- Do not treat child/subgoal text embedded in `/goal` as SSOT.
- Child/subgoal additions, splits, reorderings, supersessions, blocker metadata, and review evidence belong in the goal-run ledger/artifacts and governed Kanban closeout evidence.
- If the active `/goal` pointer, run journal, and live Kanban card/parent diverge, Kanban wins and the run must stop, reconcile, or report a blocker.
- `goal.complete` or a terminal goal-run status is not Kanban completion; Kanban lifecycle closeout remains separate authority.

## Required preflight

Before mutation or run creation, verify:

- live Kanban authority for the target card/parent;
- `execution_approved=true` or equivalent current-turn Chris authorization;
- Done Criteria are present and durable;
- no forbidden side effect is needed: merge, deploy/live apply, gateway restart/reload, prod/customer/env/secret/provider mutation, external send;
- the requested lane is Ultragoal, not Autopilot.

## Structured steering contract

Ultragoal run steering must be explicit, structured, and audited. Broad natural-language prompt text must not silently mutate run state.

Allowed run-local steering kinds:

```yaml
allowed_steering_kinds:
  - add_subgoal
  - split_subgoal
  - reorder_pending
  - revise_pending_wording
  - annotate_ledger
  - mark_blocked_superseded
```

Steering rules:

- Accepted and rejected steering attempts append audit entries to the run ledger/evidence store.
- Do not hard-delete required goals/subgoals; supersede or defer with metadata instead.
- Do not auto-complete work through steering.
- Do not weaken Done Criteria, verifier expectations, cleanup gates, forbidden side effects, or the original authority scope.
- Do not edit the stable aggregate pointer as a way to change the approved objective.
- Kanban parent/card authority changes require the normal Kanban/admission/approval path, not a run-local steering directive.
- If steering changes the intended child/subgoal topology, re-read live Kanban and reconcile parent/child closeout expectations before continuing execution.

## Freshness gate before checkpoint / closeout

Before `build-closeout-evidence`, `closeout-review-ready`, parent `review_ready`, `mark-review-ready`, or any final completion report, re-read and compare:

- live Kanban card/parent and child hierarchy;
- execution approval and forbidden side-effect boundary;
- Done Criteria snapshot/hash;
- run ledger head / latest terminal event;
- active Hermes `/goal` stable pointer when available;
- PR URL/number/head SHA/check rollup when code PR evidence is required;
- verifier/reviewer evidence freshness;
- cleanup/residue evidence and retained-artifact TTLs.

Freshness rules:

- If Kanban and the run journal diverge, Kanban wins and the run must stop, reconcile, or report a blocker.
- Stale or missing freshness evidence blocks `review_ready` claims.
- Parent `review_ready` requires each in-scope child to have its own Kanban closeout (`review_phase=review_ready` or explicit no-code/out-of-scope exception). Parent-embedded `childEvidence` alone is insufficient.
- A terminal run state, worker report, or `/goal` completion is not enough without fresh authority and closeout evidence.

If Chris explicitly asks for live application and includes a bounded gateway restart in the latest message, e.g. `gateway restart 1회 포함해서 Ultragoal allowlist live 적용해`, treat that as current-turn authority for exactly that live-load window. Do not carry forward a stale no-restart boundary from earlier context or compaction. Execute the narrow live apply path, consume at most the approved restart count, and verify runtime truth before claiming the route is usable. See `references/live-allowlist-apply-with-restart.md`.

Reference: `references/live-natural-ingress-readonly-smoke.md`: prove authorized Discord-thread phrases route to `/kanban-ultragoal-ingress`, unauthorized senders fail closed, non-Discord/non-thread surfaces do not match, the skill is available, focused routing tests pass, and Kanban DB quick_check remains healthy. Treat that smoke as circuit verification only, not real task execution.

Reference: `references/parent-task-ultragoal-e2e.md` captures the parent-task Ultragoal dogfood contract: preserve parent topology instead of shrinking to a child slice, avoid Autopilot/Kanban-dispatcher fallback, manage implementation/verifier/reviewer as internal parent-run phases, and reconcile Ultragoal ledger evidence into governed Kanban closeout evidence.

Reference: `references/ultragoal-closeout-bridge.md` captures the durable fix pattern from parent Ultragoal dogfood: store goal-run artifacts under Hermes state instead of the product repo, use `build-closeout-evidence` / `closeout-review-ready` when available, support no-new-diff existing-PR reconciliation, include cleanup-gate tests, and parse `gh pr checks` as tab-separated output because check names contain spaces.

Reference: `references/review-ready-blocked-status.md` captures the Kanban v1 status/phase split: `review_phase=review_ready` intentionally keeps raw `status=blocked` as a review-handoff holding state until `closed` maps to `done`; verify phase/evidence/events before explaining completion.

## First live smoke target selection

When Chris is dogfooding a newly live Ultragoal allowlist / natural ingress path and asks whether to execute “a card,” do **not** blindly pick the only ready card or a tempting parent objective. First separate *ingress smoke* from *real implementation execution*:

1. Verify runtime readiness before target choice: gateway `active/running`, PID changed after any approved restart, loaded config/allowlist includes Chris's immutable Discord user id, unauthorized user ids fail the resolver, Kanban DB quick_check passes, and no active/stale Kanban writer or post-restart DB error is present.
2. Prefer a disposable/read-only smoke card when the goal is only to prove the live route and lifecycle wiring.
3. If Chris suggests an existing parent card, read the parent, children, and source RALPLAN/plan before judging. Parent cards are usually too broad for first smoke if they include migration, release, production DB verification, provider/customer side effects, or multi-child closeout.
4. If Chris asks for a **parent-grade** test, do not pick a card merely because it has children. First answer what the parent is actually trying to accomplish and whether that work is currently needed. For fork/internal cleanup parents, verify upstream-vs-fork delta, current worktree/process residue, and the presence of a real plan/spec before recommending execution; if the card is admission-only with no structured Done Criteria, recommend activation/spec normalization or a different parent instead of forcing a run.
5. Align the target with the thing being tested:
   - natural Ultragoal ingress smoke → small/disposable or local-only card;
   - parent/child closeout matrix → parent with children and low external side effects;
   - “parent approval should progress children without asking after each child” → an Autopilot parent-scope progression card, not a generic cleanup parent.
6. Prefer a narrow child slice with local-only verification over the parent when available. A good smoke child is a regression-test or read-only evidence slice whose boundaries can explicitly forbid migration, PR/push/merge/release, gateway restart, production DB mutation, env/secret/provider/customer side effects, and external sends.
7. Report the recommendation as a target choice with boundaries, not as continued debate: e.g. “Parent DC-024 is too heavy; DC-025 is the safer Ultragoal smoke because it can stay local regression-test only.” If later evidence shows the selected parent is the wrong class of work, explicitly retract and redirect rather than defending the earlier pick.

This keeps the first live smoke useful without accidentally converting an ingress check into a production-affecting implementation lane or fake-progress cleanup program.

## Blocker format

```text
Ultragoal ingress blocked
Reason: <missing target | missing Kanban authority | execution not approved | forbidden side effect | runtime unavailable>
Lane: Kanban Ultragoal
Wire compatibility: direct-kanban if current CLI requires it
No fallback: Autopilot/Hermes direct/Codex not used
Needed next: <exact approval/fact/action>
```

## Quality-gate evidence matrix

Before claiming Ultragoal `review_ready` or equivalent final closeout, materialize evidence in a concrete shape compatible with governed Kanban closeout. Worker self-report alone never passes.

Recommended matrix:

```yaml
quality_gate:
  architect_review:
    architecture_status: CLEAR|WATCH|BLOCK
    product_status: CLEAR|WATCH|BLOCK
    code_status: CLEAR|WATCH|BLOCK
    recommendation: APPROVE|COMMENT|REQUEST_CHANGES
    evidence: <paths/commands/review id>
    blockers: []
  verifier_qa:
    status: passed|failed|blocked
    contract_coverage:
      - done_criterion: <id/text>
        status: covered|not_applicable|failed
        evidence_refs: []
    surface_evidence:
      - surface: gui|web|cli|api|package|algorithm|math|none
        invocation: <real command/action/test>
        verdict: passed|failed|blocked
        artifact_refs: []
    adversarial_cases:
      - scenario: <boundary/failure-mode>
        expected_behavior: <contract>
        verdict: passed|failed
        artifact_refs: []
  iteration:
    full_rerun: true|false
    rerun_commands: []
    blockers: []
```

Rules:

- Plan/code mismatch is a blocker, not something to paper over with implementation intent.
- Missing required evidence rows are not passes.
- GUI/web surfaces require browser automation, screenshot, or image-verdict evidence when applicable.
- CLI/API/package surfaces require real invocation evidence.
- Algorithm/math surfaces require boundary, property, adversarial, or failure-mode evidence.
- If any lane is `WATCH`, `BLOCK`, `REQUEST_CHANGES`, `failed`, missing, or stale, do not claim `review_ready`; record blocker/remediation work instead.
- `full_rerun=true` is required after remediation before final closeout.
- Runtime hard gate now expects the reviewer artifact (`reviews/final.json` / `record-reviewer-result`) to carry this matrix as `quality_gate` or `qualityGate`; missing/partial matrix blocks `mark-review-ready`, `build-closeout-evidence`, and `closeout-review-ready`.

## Dogfood findings / pitfalls

- Recovered or restored parent cards can have hierarchy links and `routing_verdict=direct-kanban` but still fail `kanban-ultragoal pilot-check` because `executionApproved` and structured Done Criteria are absent. Before starting a real parent run, run `hermes kanban-ultragoal --json pilot-check <task>` and normalize the authority record with an audit backup/comment if normal CLI admission cannot express the missing fields.
- Current parent goal projection may only carry child task ids in `goals.json`, not child public ids/titles. Treat `childTaskIds` + Kanban `show` as the authority for child labels and record this as a runtime dogfood finding rather than assuming the projection is human-readable.
- When Chris asks for a parent e2e / original-OMX-like Ultragoal run, do not shrink the work to one safe child slice. Preserve the parent envelope and child/subgoal topology; reduce risk by keeping forbidden side effects closed, not by removing the parent behavior being tested.
- In parent-task Ultragoal mode, `worker` language means the internal implementation phase unless Chris explicitly asks for dispatcher workers. Do not use Autopilot or Kanban dispatcher child-worker dispatch as a fallback; implementation/verifier/reviewer should be recorded as phases inside the parent run.
- Current `kanban-ultragoal mark-review-ready` requires PR and CI evidence. For a real Ultragoal execution, PR creation/push is part of the normal review-ready path unless Chris explicitly says no PR/push in the current turn. Merge/release/deploy/runtime/prod/provider/customer side effects remain separate approval gates.
- `kanban-ultragoal` run evidence is not automatically valid Kanban closeout evidence. Before Kanban `review_ready`, materialize governed closeout schema (`kanban_done_criteria_ledger.v1`, `kanban_worker_evidence.v1`, `kanban_verifier_result.v1`) and apply lifecycle in order: children/parent `worker_done` first, then children `review_ready`, then parent `review_ready`. On controller versions with the bridge, prefer `hermes kanban-ultragoal --json build-closeout-evidence <run_id>` and `hermes kanban-ultragoal --json closeout-review-ready <run_id>` over hand-written `/tmp` evidence JSON.
- Goal-run artifacts must not dirty the target product repo. Prefer the Hermes state root (`$HERMES_HOME/goal-runs/<repo-slug>/<run_id>`) and treat repo-local `.hermes/goal-runs/<run_id>` as legacy/fallback evidence only. If repo-local artifacts appear, preserve/archive them outside the product repo, remove the untracked product-repo residue, and update cleanup/residue evidence with retained path, reason, and TTL before claiming clean closeout.
- When a parent/child run is a reconciliation of an already landed PR, model it as a no-new-diff existing-PR artifact instead of inventing new code or pretending an open PR exists. The review package should explicitly carry `kind=no_new_diff_existing_pr_artifact`, empty `changed_files`, PR/head-SHA/check evidence, and a `no_pr_exception`/`no_pr_reason`.
- When changing Ultragoal storage or cleanup semantics, include `tests/hermes_cli/test_ultragoal_cleanup_gate.py` in the local verification set. Focused parent-mode tests can pass while cleanup-gate tests still assume repo-local `.hermes/goal-runs` paths.
- DailyChingu parent dogfood example: DC-024 first stopped incorrectly at local `review_passed`; after Chris corrected the boundary, PR #177 was opened, CI passed, Ultragoal state reached `review_ready`, and Kanban closeout recorded `review_phase=review_ready` with `status=blocked` as the v1 review-handoff holding status. Follow-up truth check showed the child cards DC-025..DC-028 still had `status=triage`, `review_phase=NULL`, and no closeout evidence. Treat this as a controller/SSOT bug: parent `review_ready` must either drive child lifecycle closeouts or fail-closed with a child matrix blocker; embedded parent `lastTerminalReport.childEvidence` is not equivalent to child Kanban `review_ready`.
- Runtime hard gate now materializes `child_closeout_matrix` during closeout evidence build and requires each parent-scope child to have explicit `childEvidence` and `childCleanup` rows before parent `review_ready` closeout proceeds. This is still not a substitute for Kanban child transitions; it is the preflight matrix that prevents a parent-only proof bundle from silently skipping child accounting.
- Do not interpret raw `status=blocked` on a `review_ready` card as a failure without checking `review_phase`, evidence, and events. In Kanban v1, non-closed closeout phases intentionally keep raw `status=blocked`; only `review_phase=closed` maps to raw `status=done`. When reporting to Chris, explain this as `review handoff holding`, not as ordinary blocker state.

## Verification checklist

- [ ] `hermes-execution-routing` loaded before executor selection.
- [ ] Lane recorded as `Kanban Ultragoal`.
- [ ] Current wire value compatibility handled explicitly.
- [ ] Kanban authority re-read.
- [ ] Target card/parent resolved.
- [ ] `pilot-check` run and blockers handled before mutation/run start.
- [ ] Execution approval and side-effect boundaries checked.
- [ ] For parent mode, child ids/public ids/titles are re-read from Kanban, not trusted solely from `goals.json`.
- [ ] If Chris requested parent e2e/original-OMX behavior, parent topology is preserved; do not reduce to a child-only slice.
- [ ] Implementation/verifier/reviewer are internal parent-run phases unless dispatcher workers are explicitly requested; Autopilot and Kanban dispatcher fallback remain forbidden.
- [ ] Ultragoal evidence is translated into governed Kanban closeout schema before lifecycle closeout; use `build-closeout-evidence` when available instead of manual `/tmp` JSON.
- [ ] Parent/child lifecycle order is enforced via `closeout-review-ready` or equivalent governed bridge: child/parent `worker_done` first, child `review_ready` next, parent `review_ready` last.
- [ ] Repo-local goal-run artifacts do not leave the product repo dirty; default/target evidence root is `$HERMES_HOME/goal-runs/<repo-slug>/<run_id>` or an equivalent Hermes evidence store, with cleanup/residue evidence including path, reason, and TTL/revisit.
- [ ] Existing merged PR reconciliation is represented as a no-new-diff review package with PR/head-SHA/check evidence and `no_pr_exception`/`no_pr_reason`; do not fabricate a new diff to satisfy the PR gate.
- [ ] Verification for storage/cleanup/bridge changes includes `test_ultragoal_cleanup_gate.py` as well as parent-mode/operator/closeout tests.
- [ ] For parent `review_ready`, each in-scope child has its own Kanban closeout (`review_phase=review_ready` or a documented no-code/out-of-scope exception) before claiming parent completion; parent-embedded `childEvidence` alone is insufficient.
- [ ] `kanban-ultragoal` used only after preflight.
- [ ] Missing prerequisites fail closed, not into Autopilot.
- [ ] If Chris explicitly forbids PR/push, stop at `review_passed`/blocked rather than forcing `review_ready`; otherwise treat PR creation + CI evidence as required Ultragoal completion work.
