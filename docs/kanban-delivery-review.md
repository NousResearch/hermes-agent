# Exact-head delivery continuation

## One contract, not two competing routes

`review_requirement` is the persisted, immutable pair ownership contract from the
peer's source. `delivery_review` contains only generation evidence. No task IDs,
PRs, branches, or reviewer ownership are inferred from titles or prose.

An implementation worker calls its existing `kanban_complete` on its OWN card:

```json
{
  "review_requirement": {
    "required": true,
    "owner": "orchestrator",
    "review_task_id": "<existing sole review child>"
  },
  "delivery_review": {
    "head": "<full 40-character lowercase git SHA>",
    "evidence": {
      "artifact": "<stable PR URL or repository/branch identity>",
      "checks": [{"command": "<focused command>", "result": "passed"}],
      "ci_head": "<same SHA>", "ci": "success", "draft": false,
      "proof_head": "<same SHA>", "proof": "passed"
    }
  }
}
```

The owned run closes into a parked handoff, not final delivery completion. On the
next EXISTING dispatcher tick the controller validates the run, pair, exact SHA,
artifact identity and focused evidence. In one SQLite IMMEDIATE transaction it
records implementation PHASE completion and enqueues the existing reviewer. This
releases the dependency without asking the worker to mutate another card or
waiting for the human to complete/reopen the parent. Normal capacity/auth/quota,
claim and process-ownership gates still govern launch. No second reviewer exists.

The reviewer calls `kanban_complete` on its OWN review card with:

```json
{"delivery_review": {"head": "<exact candidate SHA>", "verdict": "BLOCK", "findings": "<actionable findings>"}}
```

or `verdict: "PASS"`. The reviewer must independently verify the candidate and
submitted evidence; metadata is not a substitute for GitHub/CI/browser evidence.
A BLOCK requires nonempty findings and returns the same implementation for ONE
claim. A corrected, NEW SHA returns to the same reviewer, including a reviewer
whose last run is done. Repeating the same pair/SHA cannot replenish permission.
A reviewer crash/timeout/exhaustion or actual blocked card never grants rework.
Scientific/authorization blockers remain parked for operator disposition.

PASS alone does not mean Ready. `delivery_accepted` requires a real claimed
independent reviewer run after this generation was enqueued, exact head match,
CI success, non-Draft, and exact-head proof. Missing, stale, red, or blocked gates
produce a visible hold. Local-commit-only work can be reviewed but does not claim
hosted CI/nonDraft/browser acceptance: do not invent those fields to get Ready.
No PR is marked nonDraft, merged, deployed, or otherwise changed by this module.

## Canonical events for the notifier owner

All controller events are durable and emitted at most once per transition. The
`transition` key is `<implementation task>:<review task>:<SHA>` (hold keys append
a reason). Existing origin subscriptions/cursors are left intact.

- `delivery_phase_completed`: implementation_task, review_task, head, transition,
  acceptance="pending", evidence. This is Ready=NO.
- `delivery_changes_requested`: same pair/head fields, status="ready", review_run,
  findings. Ready here means work queue, NOT product acceptance.
- `delivery_accepted`: same fields, review_run, acceptance="ready", evidence.
- `delivery_review_hold`: same fields, reason, deduped reason-specific transition.
- Internal reviewer events: `delivery_review_enqueued` and
  `delivery_review_candidate`. The latter carries exact evidence and instructions
  above the stale opening body in `build_worker_context`.
- An ordinary `completed` event from the generation reviewer carries
  completion_kind="delivery_review_result", ready=false. It MUST NOT be rendered
  as review_approved or final Ready; the parent controller evaluates acceptance.

## Explicit delivery roles and artifact preservation

Optional `metadata.delivery_dispatch` carries `assignee`, `role`, `phase`, and
`validation_target: {kind, commit}`. Supported role/phase pairs are
`open_pr_remediation` / `pr_head` and `post_merge_validation` / `merged_checkout`.
The commit must be an exact lowercase 40-character SHA. An incompatible explicit
contract produces a visible delivery hold before claiming a worker. Untyped
legacy work retains existing behavior; prose is not interpreted as permission.
Acceptance evidence with `validation_target` must name the candidate `pr_head`;
merged-checkout or deployed claims cannot satisfy that gate.

Ancestor reopen preserves a completed PASS only when its latest completed run
names both an exact SHA and explicit artifact/repository identity. This retains
historical proof, not approval for a new head. Other descendants retain normal
invalidation semantics; a new generation still requires exact-head review.

## Durable notifications and verification

The existing notifier uses a fenced SQLite delivery ledger and stable content
markers. Ambiguous transport acceptance is reconciled against provider history,
or parked for an operator if it cannot be proven. Retrying a tick does not create
another review or resend an acknowledged phase notice. Stop notices do not wake
a new worker merely because a dependency, crash, or operator decision is pending.

Focused tests exercise actual isolated SQLite lifecycle, dispatch, completion,
notifier and CLI paths. Positive CI/proof values in fixtures are synthetic inputs,
not claims of hosted CI or deployment. Runtime activation is a separate operator
step; source tests never establish what code an already-running gateway loaded.
