# kanban-parked-triage diagnose runbook

`kanban-parked-triage diagnose` is a read-only helper for GitHub-backed Hermes Kanban tasks. It classifies a parked issue/task from GitHub issue/PR evidence plus Kanban task graph/runs/events/comments.

Default command:

```bash
kanban-parked-triage diagnose --issue <number> [--task-id <id>] [--repo GTZhou/TianGongKaiWu] [--board tiangongkaiwu]
```

Useful options:

```bash
kanban-parked-triage diagnose --issue 156 --task-id t_8216c3dd --format json
kanban-parked-triage schema
kanban-parked-triage runbook --format json
```

The helper has no side effects. It does not modify GitHub, Kanban, Telegram, labels, profile runtime, gateway, watcher, or services.

## JSON result schema

```json
{
  "schema": "kanban-parked-triage-result:v1",
  "generated_at": "UTC ISO-8601 timestamp",
  "issue": {
    "number": 156,
    "url": "https://github.com/GTZhou/TianGongKaiWu/issues/156",
    "state": "OPEN",
    "labels": ["已接单"],
    "title": "issue title"
  },
  "task_id": "t_...",
  "board": "tiangongkaiwu",
  "current_status": "blocked|running|done|OPEN|...",
  "is_current_block": true,
  "parked_type": "one of the types below",
  "confidence": "high|medium|low",
  "evidence": [
    {"source": "kanban.task|kanban.child|kanban.run|kanban.comment|kanban.event|github.issue|github.pr|github.comment", "summary": "redacted excerpt"}
  ],
  "recommendation": {
    "summary": "classification summary",
    "next_action": "recommended next step",
    "automation_level": "可自动推进|需窄续跑|需人类决策|需另立基础设施 issue|无需推进",
    "owner": "recommended owner",
    "allowed_actions": [],
    "forbidden_actions": []
  },
  "warnings": [],
  "side_effects": []
}
```

Public output redacts obvious credentials and raw Telegram numeric locators such as
`telegram:<numeric-chat-id>[:thread-id]`.

## Classification runbook

### review-required-with-child

Signal: parent task is parked with `review-required`, and an existing child task title/body/assignee indicates review/audit/审计/审核/复审.

Recommended action: recover/complete the parent handoff so the existing review child can be promoted.

Do not: rerun the executor, let executor self-review, or create a duplicate review child.

### review-required-no-child

Signal: `review-required` evidence exists, PR/commit/tests or artifact evidence exists, but no review child is visible.

Recommended action: create/dispatch a review child with issue URL, PR/artifact URL, submitted head SHA, changed files, tests, non-goals, and acceptance criteria.

Do not: make the executor audit its own work or reimplement the whole task.

### returned-awaiting-rework

Signal: GitHub/Kanban evidence includes `returned`, `已退回`, `审核退回`, or `blocking_findings`.

Recommended action: form or promote a rework + re-review chain; rework only handles the blocking findings delta.

Do not: park indefinitely at reviewer blocked, ignore the return packet, or re-review unrelated scope.

### needs-evidence

Signal: required evidence is missing or placeholder text appears (`待补`, `TODO`, `TBD`, `PR URL: 待补`, `missing evidence`).

Recommended action: create a narrow evidence-fix task or unblock a narrow continuation to supply PR URL, head SHA, tests, trace receipt, label readback, or Kanban metadata.

Do not: redo the implementation or claim completion without readable evidence.

### needs-human-decision

Signal: evidence asks for `needs-human-decision`, product/risk/permission decision, `拍板`, or equivalent.

Recommended action: block on one concrete decision question with path/risk trade-offs.

Do not: guess the decision or bypass authorization/risk boundaries.

### budget-exhausted-with-artifact

Signal: latest run/comment indicates iteration/runtime budget exhaustion while PR/commit/head SHA/tests/artifact evidence exists.

Recommended action: narrow continuation only for durable handoff bookkeeping: GitHub comment, lifecycle label, trace receipt, and `kanban_complete` metadata.

Do not: rerun the implementation from scratch or expand self-review.

### infra-missing

Signal: helper/profile/gateway/watcher/Telegram/GitHub permission/spawn capability is missing, for example `trace-missing`, `github-permission-missing`, `Platform 'telegram' is not configured`, or `spawn_failed`.

Recommended action: preserve business evidence, block precisely, and open a separate infrastructure issue/task owned by the right operator.

Do not: repair Hermes core/profile/gateway/watcher or touch credentials inside the business card.

### stale-or-contradictory-trace-intent

Signal: trace-intent evidence asks for a lifecycle event that contradicts source task/PR/review evidence, such as a returned trace after the source was approved/merged/closed.

Recommended action: do not send the trace; write an audit note and ask the orchestrator/intent owner to correct it.

Do not: emit false actor-owned lifecycle traces.

### historical-only-not-current-block

Signal: blocked/parked appears only in historical comments/events while the current task/issue is done, archived, or closed.

Recommended action: treat it as timing/review evidence only; do not reopen or rerun the workflow.

Do not: interpret historical blocked events as current blockers or create duplicate follow-up tasks.
